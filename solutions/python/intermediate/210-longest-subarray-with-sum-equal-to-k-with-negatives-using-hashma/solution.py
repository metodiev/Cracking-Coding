
"""Task 210 - Longest subarray with sum equal to K (with negatives) using hashmap. (Tier: Intermediate, CTCI Ch 16)

Prefix sums + earliest index of each prefix: for each j, a subarray (i, j] sums to k
iff prefix[j] - prefix[i] == k, so look up prefix[j] - k. O(n) time, O(n) space.
"""


def longest_subarray_sum_k(nums, target):
    first_index = {0: -1}              # prefix sum 0 occurs before the array
    prefix = 0
    best = 0
    for j, x in enumerate(nums):
        prefix += x
        need = prefix - target
        if need in first_index:
            best = max(best, j - first_index[need])
        if prefix not in first_index:
            first_index[prefix] = j    # keep the earliest occurrence
    return best


def main():
    assert longest_subarray_sum_k([1, 2, 3, 1, 1, 1, 1], 6) == 4
    assert longest_subarray_sum_k([-1, 1, 1], 1) == 3      # the whole array sums to 1
    assert longest_subarray_sum_k([1, -1, 5, -2, 3], 3) == 4      # [1,-1,5,-2]
    assert longest_subarray_sum_k([-2, -1, 2, 1], 1) == 2         # [2,1]
    assert longest_subarray_sum_k([1, 2, 3], 100) == 0
    assert longest_subarray_sum_k([], 0) == 0
    assert longest_subarray_sum_k([0, 0, 0], 0) == 3


if __name__ == "__main__":
    main()
    print("PASS")
