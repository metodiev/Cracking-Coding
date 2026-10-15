"""Task 028 - Count number of set bits in an integer. (Tier: Beginner, CTCI Ch 5)"""

def count_set_bits(x):
    """Kernighan: x & (x - 1) clears the lowest set bit."""
    count = 0
    x &= 0xFFFFFFFF
    while x:
        x &= x - 1
        count += 1
    return count


def main():
    assert count_set_bits(0) == 0
    assert count_set_bits(1) == 1
    assert count_set_bits(255) == 8
    assert count_set_bits(0b101101) == 4


if __name__ == "__main__":
    main()
    print("PASS")
