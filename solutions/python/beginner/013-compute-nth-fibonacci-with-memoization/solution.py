"""Task 013 - Compute nth Fibonacci with memoization.  (Tier: Beginner, CTCI Ch 8)

O(n) time / O(n) space using an explicit cache.
"""


def fib_memo(n, cache=None):
    if cache is None:
        cache = {0: 0, 1: 1}
    if n in cache:
        return cache[n]
    cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]


def main():
    assert fib_memo(50) == 12586269025
    assert fib_memo(0) == 0 and fib_memo(1) == 1
    assert fib_memo(10) == 55
    print("PASS")


if __name__ == "__main__":
    main()
