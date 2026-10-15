"""Task 092 - Max over sliding window using deque. (Tier: Beginner, CTCI Ch 3)

O(n): a monotonic deque holds indices whose values are decreasing; pop from the
back while the new value is >= the back, pop the front when it leaves the window.
"""

from collections import deque


def max_sliding_window_naive(nums, k):
    n = len(nums)
    if k <= 0 or k > n:
        return []
    result = []
    for start in range(n - k + 1):
        best = nums[start]
        for i in range(start + 1, start + k):
            if nums[i] > best:
                best = nums[i]
        result.append(best)
    return result


def max_sliding_window_deque(nums, k):
    n = len(nums)
    if k <= 0 or k > n:
        return []
    dq = deque()  # indices of candidates, front is the max of the current window
    result = []
    for i in range(n):
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result


def main():
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    assert max_sliding_window_deque(nums, 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window_deque(nums, 1) == nums
    assert max_sliding_window_deque(nums, 8) == [7]
    assert max_sliding_window_deque([9, 9, 9], 2) == [9, 9]
    assert max_sliding_window_deque([-5, -2, -9], 2) == [-2, -2]
    assert max_sliding_window_deque([5], 1) == [5]
    assert max_sliding_window_deque([1, 2, 3], 0) == []
    assert max_sliding_window_deque([1, 2, 3], 4) == []
    assert max_sliding_window_deque([], 1) == []

    arr = [9, -4, 7, 7, 2, -1, 8, 0, 5, 3]
    for k in range(0, len(arr) + 2):
        assert max_sliding_window_deque(arr, k) == max_sliding_window_naive(arr, k)

    samples = [
        ([4, 2, 12, 3, 8, 7, 5, 9, 1, 6], 4),
        ([-1, -1, -2, -3], 2),
        ([1, 1, 1, 1], 3),
        ([10, -10, 5, -5, 0], 5),
        ([7, 2], 2),
    ]
    for sample, k in samples:
        assert max_sliding_window_deque(sample, k) == max_sliding_window_naive(sample, k)


if __name__ == "__main__":
    main()
    print("PASS")
