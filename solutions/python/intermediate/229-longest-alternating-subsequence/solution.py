"""Task 229 - Longest alternating subsequence. (Tier: Intermediate, CTCI Ch 8)

Longest up-down (wiggle) subsequence. Greedy: any direction change between two
consecutive elements adds one to the best alternating length so far, because a
late turning point can always stand in for a whole monotone stretch. Equal
neighbours are skipped (the subsequence must be strictly alternating). up/down
hold the best lengths whose last step is an increase/decrease.
O(n) time, O(1) space. Empty input returns 0.
"""


def longest_alternating_subsequence(nums):
    if not nums:
        return 0
    up = down = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            up = down + 1
        elif nums[i] < nums[i - 1]:
            down = up + 1
    return max(up, down)


def main():
    assert longest_alternating_subsequence([1, 7, 4, 9, 2, 5]) == 6
    assert longest_alternating_subsequence([1, 17, 5, 10, 13, 15, 10, 5, 16, 8]) == 7
    assert longest_alternating_subsequence([8]) == 1
    assert longest_alternating_subsequence([]) == 0
    assert longest_alternating_subsequence([1, 2, 3, 4]) == 2      # monotone
    assert longest_alternating_subsequence([2, 1]) == 2
    assert longest_alternating_subsequence([3, 3]) == 1            # equal is not a step
    assert longest_alternating_subsequence([3, 3, 3, 2]) == 2


if __name__ == "__main__":
    main()
    print("PASS")
