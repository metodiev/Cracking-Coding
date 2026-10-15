"""Task 090 - Implement sliding window sum of size k. (Tier: Beginner, CTCI Ch 1)

O(n) running-sum window: subtract the element leaving the window and add the one
entering it. Returns the sums of every contiguous window of size k in
left-to-right order; k == 0 returns [] (no windows) and k > n returns [] (too
few elements); k < 0 raises ValueError.
"""


def window_sums(nums, k):
    if k < 0:
        raise ValueError("k must be >= 0")
    n = len(nums)
    if k == 0 or k > n:
        return []
    window = sum(nums[:k])
    result = [window]
    for i in range(k, n):
        window += nums[i] - nums[i - k]
        result.append(window)
    return result


def main():
    assert window_sums([1, 2, 3, 4, 5], 3) == [6, 9, 12]
    assert window_sums([1, 2, 3, 4, 5], 1) == [1, 2, 3, 4, 5]
    assert window_sums([1, 2, 3, 4, 5], 5) == [15]
    assert window_sums([1, 2, 3, 4, 5], 6) == []      # k > n -> no windows
    assert window_sums([1, 2, 3, 4, 5], 0) == []      # k == 0 -> no windows
    assert window_sums([], 1) == []
    assert window_sums([], 0) == []
    assert window_sums([4], 1) == [4]
    assert window_sums([-1, -2, -3, -4], 2) == [-3, -5, -7]
    assert window_sums([2, 2, 2, 2], 2) == [4, 4, 4]

    try:
        window_sums([1, 2], -1)
        raise AssertionError("expected ValueError for k = -1")
    except ValueError:
        pass

    # O(n) on a large input: window count and end windows match the closed form
    n, k = 20000, 100
    nums = list(range(n))
    sums = window_sums(nums, k)
    assert len(sums) == n - k + 1
    assert sums[0] == k * (k - 1) // 2
    assert sums[-1] == (n - k) * k + k * (k - 1) // 2


if __name__ == "__main__":
    main()
    print("PASS")
