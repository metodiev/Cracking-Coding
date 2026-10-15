"""Task 029 - Check if integer is power of two. (Tier: Beginner, CTCI Ch 5)"""

def is_power_of_two(x):
    """Exactly one set bit: x > 0 and x & (x - 1) == 0."""
    return x > 0 and (x & (x - 1)) == 0


def main():
    assert is_power_of_two(1)
    assert is_power_of_two(1024)
    assert not is_power_of_two(0)
    assert not is_power_of_two(12)
    assert not is_power_of_two(-4)


if __name__ == "__main__":
    main()
    print("PASS")
