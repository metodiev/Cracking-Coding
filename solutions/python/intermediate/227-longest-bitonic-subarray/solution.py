"""Task 227 - Longest bitonic subarray. (Tier: Intermediate, CTCI Ch 8)

Contiguous subarray that strictly increases and then strictly decreases; a
purely increasing or purely decreasing run counts in full (degenerate bitonic),
and equal neighbours break strictness. For each index i, inc[i] is the length of
the strictly increasing run ending at i and dec[i] the strictly decreasing run
starting at i; the candidate through i is inc[i] + dec[i] - 1.
O(n) time, O(n) space. Empty input returns 0.
"""


def longest_bitonic_subarray(nums):
    n = len(nums)
    if n == 0:
        return 0
    inc = [1] * n
    for i in range(1, n):
        if nums[i] > nums[i - 1]:
            inc[i] = inc[i - 1] + 1
    dec = [1] * n
    for i in range(n - 2, -1, -1):
        if nums[i] > nums[i + 1]:
            dec[i] = dec[i + 1] + 1
    return max(inc[i] + dec[i] - 1 for i in range(n))


def main():
    assert longest_bitonic_subarray([2, 1, 4, 5, 6, 3, 2]) == 6   # [1,4,5,6,3,2]
    assert longest_bitonic_subarray([1, 5, 1]) == 3                # peak
    assert longest_bitonic_subarray([1, 2, 3, 4]) == 4             # monotone up
    assert longest_bitonic_subarray([4, 3, 2, 1]) == 4             # monotone down
    assert longest_bitonic_subarray([7]) == 1
    assert longest_bitonic_subarray([]) == 0
    assert longest_bitonic_subarray([1, 1]) == 1                   # not strict
    assert longest_bitonic_subarray([12, 4, 78, 90, 45, 23, 100]) == 5   # [4,78,90,45,23]


if __name__ == "__main__":
    main()
    print("PASS")
