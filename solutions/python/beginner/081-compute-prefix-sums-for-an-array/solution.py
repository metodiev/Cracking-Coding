"""Task 081 - Compute prefix sums for an array. (Tier: Beginner, CTCI Ch 1)

prefix[i] = sum of the first i elements, so len(prefix) == n + 1 and prefix[0] == 0.
"""


def prefix_sums(nums):
    prefix = [0] * (len(nums) + 1)
    for i, value in enumerate(nums):
        prefix[i + 1] = prefix[i] + value
    return prefix


def main():
    assert prefix_sums([]) == [0]
    assert prefix_sums([5]) == [0, 5]
    assert prefix_sums([1, 2, 3]) == [0, 1, 3, 6]
    assert prefix_sums([-1, 1, -1]) == [0, -1, 0, -1]
    assert prefix_sums([4, 4, 4, 4]) == [0, 4, 8, 12, 16]
    assert len(prefix_sums([1, 2, 3, 4, 5])) == 6


if __name__ == "__main__":
    main()
    print("PASS")
