
"""Task 204 - Count pairs with given difference K in array (hash). (Tier: Intermediate, CTCI Ch 16)

Count index pairs (i < j) with |a[i] - a[j]| == k using value frequencies.
For k == 0 the answer is sum of C(freq, 2) over values.
"""

from collections import Counter


def count_pairs_with_difference(nums, k):
    if k < 0:
        raise ValueError("k must be >= 0")
    freq = Counter(nums)
    if k == 0:
        return sum(c * (c - 1) // 2 for c in freq.values())
    total = 0
    for value, count in freq.items():
        total += count * freq.get(value + k, 0)
    return total


def main():
    assert count_pairs_with_difference([1, 5, 3, 4, 2], 2) == 3      # (1,3), (3,5), (2,4)
    assert count_pairs_with_difference([1, 1, 1, 1], 0) == 6        # C(4,2)
    assert count_pairs_with_difference([1, 2, 3, 4], 10) == 0
    assert count_pairs_with_difference([], 3) == 0
    assert count_pairs_with_difference([5, 5, 5, 1], 4) == 3         # each 5 pairs with the 1
    assert count_pairs_with_difference([1, 1, 1, 1], 1) == 0


if __name__ == "__main__":
    main()
    print("PASS")
