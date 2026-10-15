
"""Task 075 - Binary search iterative on sorted array. (Tier: Beginner, CTCI Ch 10)

O(log n); lo + (hi - lo) // 2 avoids overflow in fixed-width languages.
"""


def binary_search(a, target):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if a[mid] == target:
            return mid
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def main():
    a = [1, 3, 5, 7, 9, 11]
    assert binary_search(a, 7) == 3
    assert binary_search(a, 1) == 0
    assert binary_search(a, 11) == 5
    assert binary_search(a, 4) == -1
    assert binary_search([], 5) == -1


if __name__ == "__main__":
    main()
    print("PASS")
