"""Task 027 - Find missing number in 1..n given n-1 elements. (Tier: Beginner, CTCI Ch 1)"""

def missing_number(nums, n):
    """Sum trick: expected n(n+1)/2 minus actual sum."""
    return n * (n + 1) // 2 - sum(nums)


def main():
    assert missing_number([1, 2, 4, 5], 5) == 3
    assert missing_number([2], 2) == 1
    assert missing_number([1], 2) == 2


if __name__ == "__main__":
    main()
    print("PASS")
