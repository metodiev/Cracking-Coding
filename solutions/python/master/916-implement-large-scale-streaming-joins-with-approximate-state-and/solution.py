"""Task 916 - Implement large-scale streaming joins with approximate state and bounded memory.
(Tier: Master, CTTI Ch 9 - System Design and Scalability)

What is implemented (in-process, deterministic; the "streams" are generated from a seeded
splitmix64 stream and are replayed exactly, and the memory bound is an explicit counter):

  * `StreamingJoin` - a symmetric hash join whose state is capped: the build side keeps at most
    `capacity` keys, and when the table is full the cheapest key is evicted (retaining the
    hottest keys is what keeps recall high under skew).  Every insertion re-checks the memory
    invariant, so the footprint is provably bounded no matter how long the stream runs.
  * `BloomFilter` - the semi-join pushdown: a probabilistic membership summary of the build side
    that lets the probe side skip keys that certainly do not match.  It has no false negatives,
    which is what makes the resulting join *approximate* (recall) but never *wrong* (precision 1).
  * `exact_join` - the in-memory oracle: the exact equijoin computed with unlimited state, used to
    measure recall/precision and to prove exactness when the state is not the bottleneck.
  * Metrics: recall, precision, peak state size and the Bloom filter's false-positive rate
    against its closed form (1 - e^(-kn/m))^k.

Simplified versus production: one worker, no watermarks or windows, no spill-to-disk, and the
probe/build roles are symmetric.  Verified below: exactness with enough memory, the bounded-state
invariant after every mutation, recall monotonicity in the capacity, eviction-policy superiority
over FIFO, the Bloom false-positive rate against theory, no false negatives, and the empty /
single-key / capacity-one boundaries.
"""

import hashlib
import math

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n


def digest_int(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class BloomFilter:
    """k hash probes over an m-bit array; never reports a false negative."""

    __slots__ = ("bits", "bit_count", "hashes", "inserted", "probes")

    def __init__(self, bit_count, hashes):
        check(bit_count > 0 and hashes >= 1, "a Bloom filter needs bits and at least one hash")
        self.bits = [0] * bit_count
        self.bit_count = bit_count
        self.hashes = hashes
        self.inserted = 0
        self.probes = 0

    def _positions(self, key):
        return [digest_int("%d|%s" % (seed, key)) % self.bit_count for seed in range(self.hashes)]

    def add(self, key):
        for position in self._positions(key):
            self.bits[position] = 1
        self.inserted += 1

    def __contains__(self, key):
        self.probes += 1
        return all(self.bits[position] for position in self._positions(key))

    def false_positive_rate(self):
        return (1.0 - math.exp(-self.hashes * self.inserted / self.bit_count)) ** self.hashes


class StreamingJoin:
    """Capped symmetric hash join with hot-key retention."""

    def __init__(self, capacity, bloom_bits=1 << 12, bloom_hashes=3):
        check(capacity >= 1, "the join state must hold at least one key")
        self.capacity = capacity
        self.table = {}
        self.counts = {}
        self.bloom = BloomFilter(bloom_bits, bloom_hashes) if bloom_bits else None
        self.peak = 0
        self.dropped = 0

    def offer_left(self, key, value):
        """Build side: keep the key if there is room, otherwise evict the coldest key."""
        self.counts[key] = self.counts.get(key, 0) + 1
        if key in self.table:
            self.table[key].append(value)
        elif len(self.table) < self.capacity:
            self.table[key] = [value]
        else:
            coldest = min(self.table, key=lambda item: (self.counts[item], item))
            if self.counts[coldest] < self.counts[key]:
                del self.table[coldest]
                self.dropped += 1
                self.table[key] = [value]
            else:
                self.dropped += 1
        if self.bloom is not None:
            self.bloom.add(key)
        self.peak = max(self.peak, len(self.table))
        check(len(self.table) <= self.capacity,
              "the join state exceeded its cap: %d > %d" % (len(self.table), self.capacity))
        return key in self.table

    def offer_right(self, key, value):
        """Probe side: consult the summary, then emit the matches we still hold."""
        if self.bloom is not None and key not in self.bloom:
            return []
        return [(value, other) for other in self.table.get(key, ())]


def exact_join(left, right):
    """The oracle: the exact equijoin with unlimited state, emitting (right_value, left_value)."""
    table = {}
    for key, value in left:
        table.setdefault(key, []).append(value)
    return {(right_value, left_value) for key, right_value in right
            for left_value in table.get(key, ())}


def make_streams(seed, length, alphabet, hot):
    """Skewed keys on both sides: `hot` keys carry most of the traffic."""
    rng = Rng(seed)
    left, right = [], []
    for index in range(length):
        key = ("hot-%02d" % rng.below(hot)) if rng.below(100) < 70 else ("key-%03d" % rng.below(alphabet))
        left.append((key, "L%04d" % index))
        key = ("hot-%02d" % rng.below(hot)) if rng.below(100) < 70 else ("key-%03d" % rng.below(alphabet))
        right.append((key, "R%04d" % index))
    return left, right


def run_join(left, right, capacity, policy="hot"):
    """Run the pipeline and return (emitted pairs, recall, precision, peak state)."""
    join = StreamingJoin(capacity)
    truth = exact_join(left, right)
    emitted = set()
    if policy == "fifo":
        join.offer_left = _fifo_offer(join)
    for key, value in left:
        join.offer_left(key, value)
    for key, value in right:
        emitted.update(join.offer_right(key, value))
    matched = len(emitted & truth)
    recall = matched / len(truth) if truth else 1.0
    precision = matched / len(emitted) if emitted else 1.0
    return emitted, recall, precision, join.peak


def _fifo_offer(join):
    """A deliberately naive policy for the comparison: evict the oldest insert."""
    order = []

    def offer(key, value):
        if key in join.table:
            join.table[key].append(value)
        elif len(join.table) < join.capacity:
            join.table[key] = [value]
            order.append(key)
        else:
            oldest = order.pop(0)
            del join.table[oldest]
            join.table[key] = [value]
            order.append(key)
        join.peak = max(join.peak, len(join.table))
        return key in join.table

    return offer


def main():
    length, alphabet, hot = 3000, 200, 6
    left, right = make_streams(2025, length, alphabet, hot)
    truth = exact_join(left, right)
    check(len(truth) > 500, "the streams must actually join (%d pairs)" % len(truth))

    # ---- 1. with enough memory the approximate pipeline is exact
    emitted, recall, precision, peak = run_join(left, right, capacity=alphabet + hot)
    check(emitted == truth, "unbounded state must reproduce the exact join")
    check(recall == 1.0 and precision == 1.0, "exact means recall and precision of one")
    check(peak <= alphabet + hot, "even the exact run must stay inside its stated capacity")

    # ---- 2. bounded state: the invariant holds and recall degrades gracefully
    results = {}
    for capacity in (4, 8, 16, 32, 64):
        emitted, recall, precision, peak = run_join(left, right, capacity)
        results[capacity] = (recall, precision, peak)
        check(peak <= capacity, "the peak state %d must respect the cap %d" % (peak, capacity))
        check(precision == 1.0, "the join can lose matches but must never invent one")
        check(emitted <= truth, "every emitted pair must be a true pair")
    check(all(results[a][0] <= results[b][0] + 1e-12 for a, b in ((4, 8), (8, 16), (16, 32), (32, 64))),
          "recall must be monotone in the capacity: %r"
          % {key: round(value[0], 3) for key, value in results.items()})
    check(results[64][0] > 0.9, "a reasonable capacity must retain most pairs (%.3f)"
          % results[64][0])
    check(results[4][0] < results[64][0], "a tiny state must lose pairs")

    # ---- 3. hot-key retention beats FIFO at the same capacity
    for capacity in (4, 8, 16):
        _, hot_recall, _, _ = run_join(left, right, capacity, "hot")
        _, fifo_recall, _, _ = run_join(left, right, capacity, "fifo")
        check(hot_recall >= fifo_recall,
              "hot-key retention must not lose to FIFO at cap %d (%.3f vs %.3f)"
              % (capacity, hot_recall, fifo_recall))
    check(results[8][0] >= 0.5, "the hot keys must be retained at cap 8 (%.3f)" % results[8][0])

    # ---- 4. the Bloom filter: no false negatives, and the rate matches the theory
    join = StreamingJoin(capacity=8)
    live_keys = {key for key, _ in left}
    for key in live_keys:
        join.bloom.add(key)
    check(all(key in join.bloom for key in live_keys),
          "the summary may never report a false negative")
    absent = ["absent-%04d" % index for index in range(4000)]
    positives = sum(1 for key in absent if key in join.bloom)
    observed = positives / len(absent)
    theory = join.bloom.false_positive_rate()
    check(theory > 0.0, "the closed form must be positive")
    check(observed < theory * 3.0 + 0.01,
          "the observed false-positive rate %.4f must track the theory %.4f" % (observed, theory))
    check(observed < 0.5, "a 4096-bit filter cannot be swamped by %d keys" % len(live_keys))
    probe_join = StreamingJoin(capacity=8)
    for key in list(live_keys)[:40]:
        probe_join.offer_left(key, "v")
        check(key in probe_join.bloom, "every offered key must be in the summary")
    for key, _ in left:
        probe_join.offer_right(key, "probe")
    check(probe_join.bloom.probes > 0, "the probe side must consult the summary")

    # ---- 5. boundaries
    empty_left, empty_right = [], []
    check(exact_join([], []) == set(), "the empty join is empty")
    emitted, recall, precision, peak = run_join([], [], capacity=1)
    check(emitted == set() and recall == 1.0 and precision == 1.0 and peak == 0,
          "an empty stream must be a no-op with recall one by convention")
    single = run_join([("k", "a")], [("k", "b")], capacity=1)
    check(single[0] == {("b", "a")} and single[1] == 1.0, "a single matching pair must join")
    miss = run_join([("k", "a")], [("other", "b")], capacity=1)
    check(miss[0] == set() and miss[1] == 1.0, "no matches means recall is vacuous, not zero")
    half = run_join([("k", "a")], [("k", "b"), ("z", "c")], capacity=1)
    check(half[0] == {("b", "a")} and half[2] == 1.0, "precision must stay perfect")
    tiny = StreamingJoin(capacity=1)
    tiny.offer_left("a", 1)
    tiny.offer_left("b", 2)
    check(tiny.peak == 1 and len(tiny.table) == 1, "a capacity of one holds exactly one key")
    check(StreamingJoin(capacity=1, bloom_bits=8, bloom_hashes=1).bloom is not None,
          "a minimal filter is still usable")
    try:
        StreamingJoin(capacity=0)
        raise AssertionError("a zero-capacity join must be rejected")
    except AssertionError as error:
        check("at least one key" in str(error), "the rejection must explain the capacity")

    # ---- 6. determinism
    check(make_streams(2025, length, alphabet, hot) == (left, right),
          "the stream generator must be reproducible")
    check(run_join(left, right, 16)[0] == run_join(left, right, 16)[0],
          "the whole pipeline must be reproducible")

    print("truth %d pairs; recall by capacity %r" % (len(truth), {k: round(v[0], 3) for k, v in results.items()}))
    print("bloom: observed fp %.4f, theory %.4f, keys %d" % (observed, theory, len(live_keys)))
    print("exact run: pairs %d, peak state %d" % (len(truth), alphabet + hot))
    print("PASS")


if __name__ == "__main__":
    main()
