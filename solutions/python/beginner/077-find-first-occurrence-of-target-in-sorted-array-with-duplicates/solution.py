
"""Task 077 - Find first occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10)

Lower-bound binary search: shrink hi until lo == hi, then verify.
"""


def first_occurrence(a, target):
    lo, hi = 0, len(a)          # [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(a) and a[lo] == target else -1


def main():
    a = [1, 2, 2, 2, 3, 4, 4]
    assert first_occurrence(a, 2) == 1
    assert first_occurrence(a, 4) == 5
    assert first_occurrence(a, 1) == 0
    assert first_occurrence(a, 9) == -1
    assert first_occurrence([], 1) == -1


if __name__ == "__main__":
    main()
    print("PASS")
