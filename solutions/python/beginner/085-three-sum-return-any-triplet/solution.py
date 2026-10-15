"""Task 085 - Three-sum (return any triplet). (Tier: Beginner, CTCI Ch 1)

Sort a copy, then fix the smallest value and use two pointers for the rest:
O(n^2) time. Returns any triplet of values (v1 <= v2 <= v3) drawn from three
distinct positions that sums to 0, or [] when none exists.
"""


def three_sum(nums):
    if len(nums) < 3:
        return []
    values = sorted(nums)
    n = len(values)
    for i in range(n - 2):
        if i > 0 and values[i] == values[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            total = values[i] + values[lo] + values[hi]
            if total == 0:
                return [values[i], values[lo], values[hi]]
            if total < 0:
                lo += 1
            else:
                hi -= 1
    return []


def contains_triplet(nums, triplet):
    if len(triplet) != 3:
        return False
    remaining = list(nums)
    for v in triplet:
        if v not in remaining:
            return False
        remaining.remove(v)
    return True


def main():
    nums = [-1, 0, 1, 2, -1, -4]
    t = three_sum(nums)
    assert t == [-1, -1, 2] or t == [-1, 0, 1], t
    assert sum(t) == 0 and t == sorted(t) and contains_triplet(nums, t)

    assert three_sum([0, 0, 0]) == [0, 0, 0]
    assert three_sum([-3, 1, 2]) == [-3, 1, 2]
    assert three_sum([1, -1, 0]) == [-1, 0, 1]
    assert three_sum([-2, 0, 0, 2, 2]) == [-2, 0, 2]
    assert three_sum([1, 2, 3]) == []
    assert three_sum([1, 1, 1]) == []
    assert three_sum([0, 0]) == []
    assert three_sum([]) == []

    # arbitrary order must not matter; any valid triplet is accepted
    nums2 = [8, -4, -2, 3, 1, -5, 7]
    t2 = three_sum(nums2)
    assert len(t2) == 3 and sum(t2) == 0, t2
    assert t2 == sorted(t2) and contains_triplet(nums2, t2), t2


if __name__ == "__main__":
    main()
    print("PASS")
