
"""Task 218 - Count subsets that sum to target (DP). (Tier: Intermediate, CTCI Ch 8)

Subsets are counted by INDEX (so zeros double the count: include/exclude the zero).
dp[s] = number of subsets of processed items summing to s. O(n * target).
"""


def count_subsets(nums, target):
    if target < 0:
        return 0
    dp = [0] * (target + 1)
    dp[0] = 1
    for x in nums:
        for s in range(target, x - 1, -1):
            dp[s] += dp[s - x]
    return dp[target]


def main():
    assert count_subsets([1, 1, 1, 1], 2) == 6            # C(4,2)
    assert count_subsets([1, 2, 3], 3) == 2               # [3], [1,2]
    assert count_subsets([], 0) == 1                      # the empty subset
    assert count_subsets([], 5) == 0
    assert count_subsets([0], 0) == 2                     # {} and {0} (by index)
    assert count_subsets([2, 3, 5], 5) == 2               # [5], [2,3]
    assert count_subsets([1, 2, 3], 100) == 0


if __name__ == "__main__":
    main()
    print("PASS")
