
"""Task 201 - Two-sum unique pairs (return all unique pairs). (Tier: Intermediate, CTCI Ch 16)

Return every distinct pair of VALUES (x, y) with x + y == target, x <= y.
O(n) with a set of seen values; output sorted for determinism.
"""


def two_sum_unique_pairs(nums, target):
    seen = set()
    pairs = set()
    for x in nums:
        y = target - x
        if y in seen:
            pairs.add((min(x, y), max(x, y)))
        seen.add(x)
    return sorted(pairs)


def main():
    assert two_sum_unique_pairs([1, 1, 2, 3, 2], 4) == [(1, 3), (2, 2)]
    assert two_sum_unique_pairs([1, 5, 3, 3, 5], 6) == [(1, 5), (3, 3)]
    assert two_sum_unique_pairs([1, 2, 3], 100) == []
    assert two_sum_unique_pairs([], 5) == []
    assert two_sum_unique_pairs([0, 0, 0], 0) == [(0, 0)]
    assert two_sum_unique_pairs([-1, 1, 0], 0) == [(-1, 1)]


if __name__ == "__main__":
    main()
    print("PASS")
