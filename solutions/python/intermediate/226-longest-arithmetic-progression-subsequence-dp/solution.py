"""Task 226 - Longest arithmetic progression subsequence (DP). (Tier: Intermediate, CTCI Ch 8)

Not necessarily contiguous. dp[i][d] = length of the longest arithmetic
subsequence ending at index i with common difference d; every pair j < i with
d = nums[i] - nums[j] extends dp[j][d] by one. Difference 0 is allowed (equal
values form an AP). O(n^2) time, O(n^2) space. Empty input returns 0.
"""


def longest_arithmetic_subsequence(nums):
    if not nums:
        return 0
    dp = [dict() for _ in nums]        # dp[i][d] = longest AP ending at i, step d
    best = 1
    for i in range(1, len(nums)):
        for j in range(i):
            d = nums[i] - nums[j]
            length = dp[j].get(d, 1) + 1
            if length > dp[i].get(d, 0):
                dp[i][d] = length
            best = max(best, length)
    return best


def main():
    assert longest_arithmetic_subsequence([3, 6, 9, 12]) == 4
    assert longest_arithmetic_subsequence([9, 4, 7, 2, 10]) == 3   # 4, 7, 10
    assert longest_arithmetic_subsequence([1, 2]) == 2
    assert longest_arithmetic_subsequence([4]) == 1
    assert longest_arithmetic_subsequence([]) == 0
    assert longest_arithmetic_subsequence([1, 1, 1]) == 3          # difference 0
    assert longest_arithmetic_subsequence([2, 4, 6, 8, 10]) == 5
    assert longest_arithmetic_subsequence([1, 5, 9, 13, 2, 6, 10, 14]) == 4
    assert longest_arithmetic_subsequence([5, 5, 3, 1]) == 3       # 5, 3, 1 (step -2)


if __name__ == "__main__":
    main()
    print("PASS")
