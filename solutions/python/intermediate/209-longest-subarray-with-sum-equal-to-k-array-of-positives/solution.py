
"""Task 209 - Longest subarray with sum equal to K (array of positives). (Tier: Intermediate, CTCI Ch 1)

Sliding window (valid only because all values are positive, so the sum is monotone
in the window size). O(n) time, O(1) space.
"""


def longest_subarray_sum_k(nums, target):
    if any(x <= 0 for x in nums):
        raise ValueError("this sliding-window variant requires positive values")
    lo = 0
    total = 0
    best = 0
    for hi, x in enumerate(nums):
        total += x
        while total > target and lo <= hi:
            total -= nums[lo]
            lo += 1
        if total == target:
            best = max(best, hi - lo + 1)
    return best


def main():
    assert longest_subarray_sum_k([1, 2, 3, 1, 1, 1, 1], 6) == 4     # [3,1,1,1]
    assert longest_subarray_sum_k([1, 1, 1, 1], 2) == 2
    assert longest_subarray_sum_k([5], 5) == 1
    assert longest_subarray_sum_k([1, 2, 3], 100) == 0
    assert longest_subarray_sum_k([2, 2, 2], 6) == 3
    try:
        longest_subarray_sum_k([1, -1, 1], 0)
        raise AssertionError("expected ValueError for non-positive input")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
