"""Task 158 - Implement simple memoized prime checker cache. (Tier: Beginner, CTCI Ch 6)

PrimeChecker runs trial division up to sqrt(n) and stores every verdict in a
cache. hits/misses counters prove that repeated queries for the same number are
served from the cache instead of being recomputed.
"""


class PrimeChecker:
    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def is_prime(self, n):
        """Memoized primality test: n < 2 is never prime."""
        if n in self._cache:
            self.hits += 1
            return self._cache[n]
        self.misses += 1
        verdict = self._check(n)
        self._cache[n] = verdict
        return verdict

    @property
    def size(self):
        return len(self._cache)

    @staticmethod
    def _check(n):
        """Trial division up to sqrt(n): O(sqrt(n)) time, O(1) space."""
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True


def main():
    checker = PrimeChecker()
    assert checker.is_prime(97) is True
    assert (checker.hits, checker.misses, checker.size) == (0, 1, 1)

    # repeated queries for the same number hit the cache
    assert checker.is_prime(97) is True
    assert checker.is_prime(97) is True
    assert (checker.hits, checker.misses, checker.size) == (2, 1, 1)

    for n, expected in ((0, False), (1, False), (2, True), (100, False), (7919, True), (7917, False)):
        assert checker.is_prime(n) is expected, n
    assert (checker.hits, checker.misses, checker.size) == (2, 7, 7)

    # repeat the whole set: every answer now comes from the cache
    for n, expected in ((0, False), (1, False), (2, True), (97, True), (100, False), (7919, True), (7917, False)):
        assert checker.is_prime(n) is expected, n
    assert (checker.hits, checker.misses, checker.size) == (9, 7, 7)


if __name__ == "__main__":
    main()
    print("PASS")
