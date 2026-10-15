"""Task 194 - Implement caching of expensive function results with TTL. (Tier: Beginner, CTCI Ch 7)

TTLCache stores (value, expires_at_ms) entries and uses an injected clock: every get,
put and memoize call receives the current time in milliseconds, which keeps tests
deterministic. An entry inserted at t with ttl is valid while now < t + ttl, so an
access at exactly t + ttl sees it as expired (strict boundary). Expired entries are
removed lazily on access and counted as evictions. Stats: hits, misses, evictions.
Values are expected to be non-None (None is the miss result).
"""


class TTLCache:
    def __init__(self):
        self._entries = {}          # key -> (value, expires_at_ms)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def put(self, key, value, now_ms, ttl_ms):
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be positive")
        self._entries[key] = (value, now_ms + ttl_ms)

    def get(self, key, now_ms):
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at = entry
        if now_ms >= expires_at:                     # exactly at TTL -> expired
            del self._entries[key]
            self.evictions += 1
            self.misses += 1
            return None
        self.hits += 1
        return value

    def memoize(self, fn, key, now_ms, ttl_ms):
        cached = self.get(key, now_ms)
        if cached is not None:
            return cached
        value = fn(key)
        self.put(key, value, now_ms, ttl_ms)
        return value

    def size(self):
        return len(self._entries)


def main():
    cache = TTLCache()
    cache.put("a", 10, 1000, 100)                   # valid while now < 1100
    assert cache.get("a", 1099) == 10
    assert cache.get("a", 1100) is None             # exactly at TTL -> expired
    assert (cache.hits, cache.misses, cache.evictions) == (1, 1, 1)
    assert cache.size() == 0                        # expired entry dropped lazily

    cache.put("b", 20, 0, 50)
    assert cache.get("b", 25) == 20
    cache.put("b", 21, 30, 50)                      # refresh -> expires at 80
    assert cache.get("b", 79) == 21
    assert cache.get("b", 80) is None

    missing = TTLCache()
    assert missing.get("nope", 0) is None
    assert (missing.hits, missing.misses) == (0, 1)
    try:
        missing.put("x", 1, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected error for non-positive ttl")

    calls = []

    def slow(n):
        calls.append(n)
        return n * n

    memo = TTLCache()
    assert memo.memoize(slow, 7, 0, 100) == 49      # computes (miss)
    assert memo.memoize(slow, 7, 99, 100) == 49     # cache hit
    assert calls == [7]
    assert memo.memoize(slow, 7, 100, 100) == 49    # expired -> recompute
    assert calls == [7, 7]
    assert (memo.hits, memo.misses, memo.evictions) == (1, 2, 1)


if __name__ == "__main__":
    main()
    print("PASS")
