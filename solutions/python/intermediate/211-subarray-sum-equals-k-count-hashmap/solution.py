
"""Task 211 - Subarray sum equals K count (hashmap). (Tier: Intermediate, CTCI Ch 16)

prefix[j] - prefix[i] == k  =>  count pairs of equal prefix values shifted by k.
One pass with a frequency map of prefix sums. O(n).
"""

from collections import defaultdict


def count_subarrays_sum_k(nums, target):
    counts = defaultdict(int)
    counts[0] = 1
    prefix = 0
    total = 0
    for x in nums:
        prefix += x
        total += counts[prefix - target]
        counts[prefix] += 1
    return total


def main():
    assert count_subarrays_sum_k([1, 1, 1], 2) == 2
    assert count_subarrays_sum_k([1, 2, 3], 3) == 2          # [3], [1,2]
    assert count_subarrays_sum_k([-1, -1, 1], 0) == 1
    assert count_subarrays_sum_k([1, 2, 3], 100) == 0
    assert count_subarrays_sum_k([0, 0], 0) == 3             # [0],[0],[0,0]
    assert count_subarrays_sum_k([], 0) == 0


if __name__ == "__main__":
    main()
    print("PASS")
