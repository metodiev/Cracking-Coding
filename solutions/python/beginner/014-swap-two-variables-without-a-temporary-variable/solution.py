"""Task 014 - Swap two variables without a temporary variable.  (Tier: Beginner, CTCI Ch 5)

Two classic tricks: arithmetic and XOR (note: XOR requires integers).
"""


def swap_arithmetic(a, b):
    a = a + b
    b = a - b
    a = a - b
    return a, b


def swap_xor(a, b):
    a ^= b
    b ^= a
    a ^= b
    return a, b


def main():
    assert swap_arithmetic(5, 3) == (3, 5)
    assert swap_xor(5, 3) == (3, 5)
    assert swap_arithmetic(-1, 7) == (7, -1)
    assert swap_xor(0, 42) == (42, 0)
    print("PASS")


if __name__ == "__main__":
    main()
