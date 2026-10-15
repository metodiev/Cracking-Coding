
"""Task 208 - Implement sliding window minimum using deque. (Tier: Intermediate, CTCI Ch 3)

Monotonic deque of indices with increasing values: the front is always the current
window minimum. Each index enters/leaves once -> O(n) total.
"""

from collections import deque


def sliding_window_minimum(nums, k):
    if k <= 0:
        raise ValueError("k must be positive")
    result = []
    dq = deque()                       # indices, values increasing
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] >= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result


def main():
    assert sliding_window_minimum([1, 3, -1, -3, 5, 3, 6, 7], 3) == [-1, -3, -3, -3, 3, 3]
    assert sliding_window_minimum([1], 1) == [1]
    assert sliding_window_minimum([1, 2], 5) == []
    assert sliding_window_minimum([5, 5, 5], 2) == [5, 5]
    assert sliding_window_minimum([9, 8, 7], 1) == [9, 8, 7]
    assert sliding_window_minimum([], 3) == []


if __name__ == "__main__":
    main()
    print("PASS")
