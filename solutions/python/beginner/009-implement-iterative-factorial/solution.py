"""Task 009 - Implement iterative factorial.  (Tier: Beginner, CTCI Ch 8)"""


def factorial_iterative(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def main():
    assert factorial_iterative(0) == 1
    assert factorial_iterative(1) == 1
    assert factorial_iterative(5) == 120
    assert factorial_iterative(10) == 3628800
    print("PASS")


if __name__ == "__main__":
    main()
