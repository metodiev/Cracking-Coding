
"""Task 128 - Implement basic memoization wrapper. (Tier: Beginner, CTCI Ch 8)

A generic decorator that caches results by call arguments (kwargs normalized).
"""

from functools import wraps


def memoize(fn):
    cache = {}

    @wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    return wrapper


def main():
    calls = {"n": 0}

    @memoize
    def fib(n):
        calls["n"] += 1
        return n if n <= 1 else fib(n - 1) + fib(n - 2)

    assert fib(30) == 832040
    assert calls["n"] == 31, calls["n"]          # n+1 distinct subproblems, not 2^n
    assert fib.cache[((10,), ())] == 55          # cache is inspectable (keyed by call args)

    @memoize
    def grid_paths(rows, cols, blocked=()):
        # blocked stays a tuple of pairs: memoize keys must be hashable
        if rows < 0 or cols < 0 or (rows, cols) in blocked:
            return 0
        if rows == 0 or cols == 0:
            return 1
        return grid_paths(rows - 1, cols, blocked) + grid_paths(rows, cols - 1, blocked)

    assert grid_paths(3, 3) == 20
    assert grid_paths(3, 3) == 20                # cached, same result
    assert grid_paths(2, 2, ((1, 1),)) == 2


if __name__ == "__main__":
    main()
    print("PASS")
