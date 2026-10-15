"""Task 010 - Implement recursive factorial.  (Tier: Beginner, CTCI Ch 8)"""


def factorial_recursive(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    return 1 if n <= 1 else n * factorial_recursive(n - 1)


def main():
    assert factorial_recursive(0) == 1
    assert factorial_recursive(1) == 1
    assert factorial_recursive(5) == 120
    assert factorial_recursive(10) == 3628800
    print("PASS")


if __name__ == "__main__":
    main()
