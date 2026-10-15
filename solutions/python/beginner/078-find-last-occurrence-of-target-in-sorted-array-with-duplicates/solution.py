
"""Task 078 - Find last occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10)"""


def last_occurrence(a, target):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    return idx if idx >= 0 and a[idx] == target else -1


def main():
    a = [1, 2, 2, 2, 3, 4, 4]
    assert last_occurrence(a, 2) == 3
    assert last_occurrence(a, 4) == 6
    assert last_occurrence(a, 1) == 0
    assert last_occurrence(a, 9) == -1


if __name__ == "__main__":
    main()
    print("PASS")
