"""Task 038 - Implement binary exponentiation (fast power). (Tier: Beginner, CTCI Ch 6)"""

def power_fast(base, exp):
    """Exponentiation by squaring: O(log exp) multiplications (exp >= 0)."""
    result = 1
    while exp:
        if exp & 1:
            result *= base
        base *= base
        exp >>= 1
    return result


def main():
    assert power_fast(3, 13) == 1594323
    assert power_fast(2, 10) == 1024
    assert power_fast(7, 0) == 1


if __name__ == "__main__":
    main()
    print("PASS")
