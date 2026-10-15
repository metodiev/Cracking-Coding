"""Task 091 - Max over sliding window naive. (Tier: Beginner, CTCI Ch 1)

O(n*k): for every start index, scan the k elements of the window.
"""


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


def main():
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    assert max_sliding_window_naive(nums, 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window_naive(nums, 1) == nums
    assert max_sliding_window_naive(nums, 8) == [7]
    assert max_sliding_window_naive([9, 9, 9], 2) == [9, 9]
    assert max_sliding_window_naive([-5, -2, -9], 2) == [-2, -2]
    assert max_sliding_window_naive([5], 1) == [5]
    assert max_sliding_window_naive([1, 2, 3], 0) == []
    assert max_sliding_window_naive([1, 2, 3], 4) == []
    assert max_sliding_window_naive([], 1) == []


if __name__ == "__main__":
    main()
    print("PASS")
