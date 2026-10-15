
"""Task 215 - Partition array into equal-sum subsets (subset sum). (Tier: Intermediate, CTCI Ch 8)

Reduces to "is there a subset summing to total/2?" — 1-D boolean DP, O(n * total) time.
"""


def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    half = total // 2
    dp = [False] * (half + 1)
    dp[0] = True
    for x in nums:
        for s in range(half, x - 1, -1):       # descending keeps items single-use
            if dp[s - x]:
                dp[s] = True
    return dp[half]


def main():
    assert can_partition([1, 5, 11, 5])                 # 11  vs 11
    assert not can_partition([1, 2, 3, 5])              # total 11 is odd
    assert can_partition([2, 2])
    assert can_partition([1, 1])                        # two halves of 1
    assert not can_partition([1])
    assert can_partition([])                            # two empty halves
    assert can_partition([3, 3, 3, 3])                  # 6 vs 6


if __name__ == "__main__":
    main()
    print("PASS")
