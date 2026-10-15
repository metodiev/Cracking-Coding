"""Task 030 - Reverse digits of an integer. (Tier: Beginner, CTCI Ch 6)"""

def reverse_digits(n):
    """Sign preserved; clamps like Java Integer on overflow for parity across languages."""
    sign = -1 if n < 0 else 1
    value = abs(n)
    result = 0
    while value:
        result = result * 10 + value % 10
        value //= 10
    result *= sign
    return 0 if result < -(2 ** 31) or result > 2 ** 31 - 1 else result


def main():
    assert reverse_digits(123) == 321
    assert reverse_digits(-123) == -321
    assert reverse_digits(0) == 0
    assert reverse_digits(1534236469) == 0     # overflows 32-bit → 0


if __name__ == "__main__":
    main()
    print("PASS")
