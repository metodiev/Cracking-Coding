
"""Task 212 - Find subarray with maximum sum (Kadane's algorithm). (Tier: Intermediate, CTCI Ch 8)

cur = max(x, cur + x) at every position; also track the winning window indices.
O(n) time, O(1) space. An empty-array input raises (no subarray exists).
"""


def max_subarray(nums):
    """Returns (sum, start, end) with end exclusive."""
    if not nums:
        raise ValueError("empty array has no subarray")
    best = cur = nums[0]
    best_start = best_end = 0
    start = 0
    for i in range(1, len(nums)):
        if cur + nums[i] < nums[i]:
            start = i                      # start a fresh window at i
            cur = nums[i]
        else:
            cur += nums[i]
        if cur > best:
            best = cur
            best_start, best_end = start, i
    return best, best_start, best_end + 1


def kadane_sum(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def main():
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == (6, 3, 7)
    assert kadane_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray([-3, -1, -2]) == (-1, 1, 2)      # all negative: best single
    assert max_subarray([1]) == (1, 0, 1)
    assert kadane_sum([5, -1, 5]) == 9
    try:
        max_subarray([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
