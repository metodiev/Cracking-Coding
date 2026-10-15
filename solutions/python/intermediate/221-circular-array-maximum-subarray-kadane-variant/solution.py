"""Task 221 - Circular array maximum subarray (Kadane variant). (Tier: Intermediate, CTCI Ch 8)

On a circular array a subarray may wrap past the end. The answer is either a
plain linear maximum subarray, or a wrapped one: total - minimum subarray, where
the removed middle is the cheapest part. When every element is negative the wrap
case would be the empty set, so the best single element wins instead.
O(n) time, O(1) space. Empty input raises.
"""


def max_circular_subarray(nums):
    if not nums:
        raise ValueError("empty array has no subarray")
    best = cur = nums[0]           # Kadane: maximum subarray
    worst = low = nums[0]          # Kadane: minimum subarray
    total = sum(nums)
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
        low = min(x, low + x)
        worst = min(worst, low)
    if best < 0:
        return best                # all negative: wrapping would pick nothing
    return max(best, total - worst)


def main():
    assert max_circular_subarray([5, -3, 5]) == 10        # wrap: [5] + [5]
    assert max_circular_subarray([-3, -2, -3]) == -2      # all negative
    assert max_circular_subarray([1, -2, 3, -2]) == 3     # linear beats wrap
    assert max_circular_subarray([1, 2, 3]) == 6          # all positive
    assert max_circular_subarray([8, -8, 9]) == 17        # wrap: [9] + [8]
    assert max_circular_subarray([11, -1, 11]) == 22      # wrap skips the -1
    assert max_circular_subarray([5]) == 5
    assert max_circular_subarray([-5]) == -5
    assert max_circular_subarray([0]) == 0
    try:
        max_circular_subarray([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
