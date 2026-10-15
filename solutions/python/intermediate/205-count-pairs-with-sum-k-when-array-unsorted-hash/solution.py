
"""Task 205 - Count pairs with sum K when array unsorted (hash). (Tier: Intermediate, CTCI Ch 16)

Single pass: for each x count already-seen (target - x) values. O(n).
"""

from collections import defaultdict


def count_pairs_with_sum(nums, target):
    seen = defaultdict(int)
    total = 0
    for x in nums:
        total += seen[target - x]
        seen[x] += 1
    return total


def main():
    assert count_pairs_with_sum([1, 5, 3, 3, 3], 6) == 4          # (1,5) + 3 pairs of 3s
    assert count_pairs_with_sum([1, 2, 3, 4], 5) == 2             # (1,4), (2,3)
    assert count_pairs_with_sum([1, 1, 1, 1], 2) == 6             # C(4,2)
    assert count_pairs_with_sum([1, 2, 3], 100) == 0
    assert count_pairs_with_sum([], 5) == 0
    assert count_pairs_with_sum([0, 0], 0) == 1


if __name__ == "__main__":
    main()
    print("PASS")
