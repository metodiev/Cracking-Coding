"""Task 228 - Maximum sum increasing subsequence. (Tier: Intermediate, CTCI Ch 8)

Strictly increasing subsequence with the largest sum. dp[i] = nums[i] plus the
best dp[j] among j < i with nums[j] < nums[i]; starting fresh (0) is always
allowed, so a lone negative element still beats extending below it.
O(n^2) time, O(n) space. Empty input returns 0.
"""


def max_sum_increasing_subsequence(nums):
    if not nums:
        return 0
    dp = list(nums)
    best = dp[0]
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + nums[i] > dp[i]:
                dp[i] = dp[j] + nums[i]
        best = max(best, dp[i])
    return best


def main():
    assert max_sum_increasing_subsequence([1, 101, 2, 3, 100, 4]) == 106   # 1+2+3+100
    assert max_sum_increasing_subsequence([10, 1]) == 10
    assert max_sum_increasing_subsequence([5, 4, 3, 2, 1]) == 5            # all falling
    assert max_sum_increasing_subsequence([1, 2, 3, 4]) == 10
    assert max_sum_increasing_subsequence([7]) == 7
    assert max_sum_increasing_subsequence([]) == 0
    assert max_sum_increasing_subsequence([3, 3]) == 3                     # not strict
    assert max_sum_increasing_subsequence([-2, -1, -3]) == -1              # all negative


if __name__ == "__main__":
    main()
    print("PASS")
