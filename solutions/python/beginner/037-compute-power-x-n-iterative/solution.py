"""Task 037 - Compute power(x, n) iterative. (Tier: Beginner, CTCI Ch 6)"""

def power(x, n):
    """Iterative; supports negative exponents via reciprocal."""
    exp = abs(n)
    result = 1.0
    base = float(x)
    for _ in range(exp):
        result *= base
    return result if n >= 0 else 1.0 / result


def main():
    assert power(2, 10) == 1024.0
    assert power(2, -2) == 0.25
    assert power(5, 0) == 1.0
    assert power(3, 3) == 27.0


if __name__ == "__main__":
    main()
    print("PASS")
