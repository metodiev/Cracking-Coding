"""Task 159 - Implement function memoizer decorator (or equivalent). (Tier: Beginner, CTCI Ch 8)

Unlike the simple wrapper from task 128, Memoizer is a small class that
  * wraps a single-argument pure function (called as memo(n)),
  * records hits / misses and exposes the current cache size,
  * supports an optional capacity with FIFO eviction - when the cache is full
    the OLDEST inserted key leaves first, and a hit never changes insertion
    order (this is FIFO, deliberately not LRU; see task 160 for an LRU cache).

Tests drive it with Collatz step counts (a pure but slowish function).
"""

from collections import deque


class Memoizer:
    def __init__(self, fn, capacity=None):
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be >= 1")
        self._fn = fn
        self._capacity = capacity
        self._cache = {}
        self._order = deque()
        self.hits = 0
        self.misses = 0

    @property
    def size(self):
        return len(self._cache)

    def __call__(self, n):
        if n in self._cache:
            self.hits += 1
            return self._cache[n]
        self.misses += 1
        value = self._fn(n)
        if self._capacity is not None and len(self._cache) >= self._capacity:
            evicted = self._order.popleft()      # FIFO: oldest insertion goes first
            del self._cache[evicted]
        self._cache[n] = value
        self._order.append(n)
        return value

    def cached(self, n):
        """True iff n is currently cached (used to assert eviction)."""
        return n in self._cache


def collatz_steps(n):
    """Steps for n to reach 1; pure but slow enough at n=27: collatz_steps(27) == 111."""
    if n < 1:
        raise ValueError("n must be >= 1")
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def main():
    # 1) no capacity: everything stays cached
    memo = Memoizer(collatz_steps)
    assert memo(27) == 111
    assert memo(6) == 8 and memo(7) == 16
    assert (memo.hits, memo.misses, memo.size) == (0, 3, 3)
    assert memo(27) == 111 and memo(6) == 8
    assert (memo.hits, memo.misses) == (2, 3)

    # 2) capacity 3 with FIFO eviction
    small = Memoizer(collatz_steps, capacity=3)
    for n in (27, 6, 7):
        small(n)
    assert small.size == 3 and (small.hits, small.misses) == (0, 3)
    assert small(27) == 111                      # hit; FIFO order unchanged by hits
    assert small.cached(27)
    assert small(9) == 19                        # miss -> the OLDEST insertion (27) is evicted
    assert small.size == 3 and not small.cached(27) and small.cached(9)
    assert small(6) == 8                         # still cached
    assert (small.hits, small.misses) == (2, 4)

    # 3) capacity 1 behaves like a one-slot cache
    tiny = Memoizer(lambda n: n * n, capacity=1)
    assert tiny(3) == 9 and tiny(4) == 16
    assert tiny.size == 1 and not tiny.cached(3) and tiny.cached(4)
    assert tiny(4) == 16 and (tiny.hits, tiny.misses) == (1, 2)

    # capacity must be positive
    try:
        Memoizer(collatz_steps, capacity=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # stats are per-instance, not shared
    other = Memoizer(lambda n: n + 1)
    assert other(1) == 2 and (other.hits, other.misses) == (0, 1)
    assert (memo.hits, memo.misses) == (2, 3)


if __name__ == "__main__":
    main()
    print("PASS")
