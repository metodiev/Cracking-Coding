"""Task 012 - Compute nth Fibonacci using recursion.  (Tier: Beginner, CTCI Ch 8)

O(2^n) - intentionally naive; see task 013 for memoization.
"""


def fib(n):
    return n if n <= 1 else fib(n - 1) + fib(n - 2)


def main():
    assert [fib(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
    assert fib(20) == 6765
    print("PASS")


if __name__ == "__main__":
    main()
