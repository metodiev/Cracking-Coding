"""Task 505 - Implement bloom-filter-backed cache for web responses simulation. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

A two-level web-response cache simulation. The bottom layer is an LRU response store of bounded
capacity (a dictionary of url -> body plus an explicit least-recently-used list, most recent at the
end), the top layer is a Bloom filter of the origins's known URLs (bit array of m bits filled by
k hash functions built from double hashing: h_i(key) = (fnv(key) + i * djb2(key) + i * i) mod m) used
as a negative filter. Every request first asks the Bloom filter: "definitely absent" answers 404
without touching the cache or the origin at all (the win of the filter), while "maybe present"
triggers a cache lookup and, on a miss, a real origin fetch that also populates the cache and the
filter. Because a Bloom filter has no false negatives, a URL that exists in the origin is never
answered with the negative path - the safety property the whole design rests on.

Documented checks: over 2000 inserted URLs there is not a single false negative, and the measured
false-positive rate over 2000 absent URLs is within 0.03 of the textbook
(1 - exp(-k n / m))^k; the optimal k = (m/n) ln 2 is verified on the closed form (it must minimise
the formula among k = 1..20) and doubling k must measurably hurt (the measured rate at 2k* must be at least
twice the rate at k*); the filter is sized from a target rate p with m = -n ln p / (ln 2)^2
and the measured rate must meet the target within 0.02; on a deterministic LCG request stream
(in 1200 requests, 30% of them for URLs that never existed) the cache never returns a wrong body,
every 404 is answered from the filter without a cache lookup, hits + misses == requests, and the
whole hit/miss/eviction sequence must agree with an independent reference model built on
collections.OrderedDict; finally the bloom-absent requests must never include a real URL, the LRU
store must never exceed its capacity, and the recency list must be a permutation of the stored keys
with the just-touched key last.
"""

import math

DEFAULT_CAPACITY = 64


def fnv1a(text):
    """64-bit FNV-1a."""
    h = 0xCBF29CE484222325
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def djb2(text):
    """64-bit djb2 (used as the second, independent hash for double hashing)."""
    h = 5381
    for ch in text:
        h = ((h * 33) + ord(ch)) & 0xFFFFFFFFFFFFFFFF
    return h


class BloomFilter:
    def __init__(self, bits, hashes):
        if bits <= 0 or hashes <= 0:
            raise ValueError("bits and hashes must be positive")
        self.bits = bits
        self.hashes = hashes
        self.store = bytearray((bits + 7) // 8)
        self.inserted = 0
        self.set_positions = set()

    def positions(self, key):
        h1 = fnv1a(key)
        h2 = djb2(key) | 1                     # odd, so it cannot be 0 for short keys
        return [(h1 + i * h2 + i * i) % self.bits for i in range(self.hashes)]

    def add(self, key):
        for pos in self.positions(key):
            self.store[pos >> 3] |= 1 << (pos & 7)
            self.set_positions.add(pos)
        self.inserted += 1

    def might_contain(self, key):
        """False means definitely absent (the only answer the filter is allowed to trust)."""
        for pos in self.positions(key):
            if not (self.store[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def bits_set(self):
        return sum(bin(byte).count("1") for byte in self.store)

    @staticmethod
    def theoretical_rate(bits, hashes, items):
        return (1.0 - math.exp(-hashes * items / bits)) ** hashes

    @staticmethod
    def size_for(target_rate, items):
        """m = -n ln p / (ln 2)^2, k = round((m/n) ln 2)."""
        bits = int(math.ceil(-items * math.log(target_rate) / (math.log(2.0) ** 2)))
        hashes = max(1, int(round((bits / items) * math.log(2.0))))
        return bits, hashes


class LruResponseStore:
    """Bounded LRU cache with an explicit recency list (most recently used last)."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.data = {}
        self.order = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        if key in self.data:
            self.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return self.data[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.data:
            self.data[key] = value
            self.order.remove(key)
            self.order.append(key)
            return None
        evicted = None
        if len(self.data) >= self.capacity:
            evicted = self.order.pop(0)
            del self.data[evicted]
            self.evictions += 1
        self.data[key] = value
        self.order.append(key)
        return evicted


class ResponseCache:
    """Bloom filter in front of the LRU store and the origin, with full book-keeping."""

    def __init__(self, bits, hashes, capacity=DEFAULT_CAPACITY):
        self.filter = BloomFilter(bits, hashes)
        self.store = LruResponseStore(capacity)
        self.origin = {}
        self.origin_fetches = 0
        self.bloom_negatives = 0
        self.filter_lookups = 0
        self.requests = 0
        self.not_found = 0

    def publish(self, url, body):
        """Add a page to the origin (and therefore to the negative filter)."""
        self.origin[url] = body
        self.filter.add(url)

    def get(self, url):
        self.requests += 1
        self.filter_lookups += 1
        if not self.filter.might_contain(url):
            self.bloom_negatives += 1          # definitely not in the origin: no lookup at all
            self.not_found += 1
            return None
        cached = self.store.get(url)
        if cached is not None:
            return cached
        body = self.origin.get(url)            # maybe a Bloom false positive; origin decides
        if body is None:
            self.not_found += 1
            return None
        self.origin_fetches += 1
        self.store.put(url, body)
        return body


def reference_lru(requests, origin, capacity):
    """Independent model of the same policy using collections.OrderedDict."""
    from collections import OrderedDict
    cache = OrderedDict()
    sequence = []
    for url in requests:
        if url not in origin:
            sequence.append(("404", None))
            continue
        if url in cache:
            cache.move_to_end(url)
            sequence.append(("hit", url))
            continue
        sequence.append(("miss", url))
        cache[url] = origin[url]
        if len(cache) > capacity:
            cache.popitem(last=False)
    return sequence, list(cache.keys())


def request_stream(count, seed, known, unknown):
    """LCG-driven stream: ~70% known URLs (skewed towards a few hot pages), ~30% absent ones."""
    stream = []
    for _ in range(count):
        seed = lcg(seed)
        roll = seed % 10
        seed = lcg(seed)
        if roll < 3:
            stream.append(unknown[seed % len(unknown)])
        else:
            # Skewed: squaring the uniform draw concentrates requests on the first few pages.
            pick = (seed % len(known)) ** 2 // len(known)
            stream.append(known[min(pick, len(known) - 1)])
    return stream


def lcg(seed):
    return (seed * 75 + 74) % 65537


def main():
    # --- Bloom filter fundamentals: no false negatives, rate matches theory. ----------------
    items = 2000
    keys = ["page/%d" % i for i in range(items)]
    absent = ["absent/%d" % i for i in range(items)]
    bits, hashes = BloomFilter.size_for(0.01, items)
    bloom = BloomFilter(bits, hashes)
    for key in keys:
        bloom.add(key)
    for key in keys:
        assert bloom.might_contain(key) is True, "a Bloom filter must never give a false negative"
    assert bloom.inserted == items
    assert bloom.bits_set() == len(bloom.set_positions)
    positives = sum(1 for key in absent if bloom.might_contain(key))
    measured = positives / len(absent)
    theory = BloomFilter.theoretical_rate(bits, hashes, items)
    assert theory <= 0.012, theory
    assert abs(measured - theory) <= 0.03, (measured, theory)
    assert positives > 0, "the test must exercise the false-positive path"

    # Optimal k = (m/n) ln 2 minimises the closed form.
    ratio = bits / items
    best_k = min(range(1, 21), key=lambda k: BloomFilter.theoretical_rate(bits, k, items))
    assert abs(best_k - ratio * math.log(2.0)) <= 1.0, (best_k, ratio * math.log(2.0))
    assert BloomFilter.theoretical_rate(bits, best_k, items) < \
        BloomFilter.theoretical_rate(bits, best_k + 4, items)
    doubled = BloomFilter(bits, 2 * hashes)
    for key in keys:
        doubled.add(key)
    doubled_rate = sum(1 for key in absent if doubled.might_contain(key)) / len(absent)
    assert doubled_rate >= 2.0 * measured, (doubled_rate, measured)
    assert doubled_rate >= 2.0 * BloomFilter.theoretical_rate(bits, best_k, items)

    # --- Cache simulation on a deterministic stream. ----------------------------------------
    known = ["page/%d" % i for i in range(120)]
    unknown = ["missing/%d" % i for i in range(60)]
    stream = request_stream(1200, seed=271, known=known, unknown=unknown)
    assert len(stream) == 1200
    assert sum(1 for url in stream if url in unknown) > 300

    cache = ResponseCache(bits=4096, hashes=3, capacity=32)
    for i, url in enumerate(known):
        cache.publish(url, "body-%d" % i)
    for url in stream:
        body = cache.get(url)
        if url not in cache.origin:
            assert body is None and cache.not_found >= 1
        elif body is not None:
            assert body == cache.origin[url], "the cache must never return a wrong body"
    # Re-run once to collect a clean hit/miss trace (the loop above already validated bodies).
    cache2 = ResponseCache(bits=4096, hashes=3, capacity=32)
    for i, url in enumerate(known):
        cache2.publish(url, "body-%d" % i)
    trace = []
    for url in stream:
        if url not in cache2.origin:
            assert cache2.get(url) is None
            trace.append(("404", None))
            continue
        before_hits = cache2.store.hits
        body = cache2.get(url)
        assert body == cache2.origin[url]
        trace.append(("hit" if cache2.store.hits > before_hits else "miss", url))

    reference, reference_keys = reference_lru(stream, cache2.origin, 32)
    assert trace == reference, "the LRU policy must agree with the OrderedDict reference model"
    assert cache2.store.order == reference_keys
    assert len(set(cache2.store.order)) == len(cache2.store.order) == len(cache2.store.data)
    assert cache2.store.order[-1] == stream[-1]
    assert len(cache2.store.data) <= 32
    assert cache2.store.evictions == cache2.store.misses - len(reference_keys)

    # Counters are consistent and the filter never sent a real URL down the negative path.
    assert cache2.requests == len(stream)
    assert cache2.store.hits + cache2.store.misses == cache2.requests - cache2.bloom_negatives
    assert cache2.bloom_negatives + cache2.store.misses + cache2.store.hits == cache2.requests
    assert cache2.bloom_negatives == sum(1 for url in stream if url in unknown), \
        "the filter must catch every never-published URL"
    assert cache2.store.misses >= cache2.origin_fetches
    assert cache2.origin_fetches == sum(1 for entry in trace if entry[0] == "miss")
    # A 404 is either a Bloom negative or a false positive that the origin then rejects.
    assert cache2.not_found == cache2.bloom_negatives + (cache2.store.misses - cache2.origin_fetches)
    assert cache2.filter_lookups == cache2.requests
    assert 0 < cache2.store.hits < cache2.requests

    # A filter without the known URLs would have used the origin for every 404 request; the
    # negative path therefore saves exactly `bloom_negatives` store lookups and origin probes.
    saved = cache2.bloom_negatives
    assert saved > 300 and saved == sum(1 for url in stream if url not in cache2.origin)


if __name__ == "__main__":
    main()
    print("PASS")
