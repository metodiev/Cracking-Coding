
"""Task 079 - Find insert position for target in sorted array. (Tier: Beginner, CTCI Ch 10)

Classic lower_bound: index where target should be inserted to keep the array sorted.
"""


def search_insert(a, target):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    assert search_insert([1, 3, 5, 6], 5) == 2
    assert search_insert([1, 3, 5, 6], 2) == 1
    assert search_insert([1, 3, 5, 6], 7) == 4
    assert search_insert([1, 3, 5, 6], 0) == 0
    assert search_insert([], 42) == 0


if __name__ == "__main__":
    main()
    print("PASS")
