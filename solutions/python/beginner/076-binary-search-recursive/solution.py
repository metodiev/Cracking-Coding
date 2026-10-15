
"""Task 076 - Binary search recursive. (Tier: Beginner, CTCI Ch 10)"""


def binary_search_recursive(a, target, lo=None, hi=None):
    if lo is None:
        lo, hi = 0, len(a) - 1
    if lo > hi:
        return -1
    mid = lo + (hi - lo) // 2
    if a[mid] == target:
        return mid
    if a[mid] < target:
        return binary_search_recursive(a, target, mid + 1, hi)
    return binary_search_recursive(a, target, lo, mid - 1)


def main():
    a = [2, 4, 6, 8, 10]
    assert binary_search_recursive(a, 8) == 3
    assert binary_search_recursive(a, 2) == 0
    assert binary_search_recursive(a, 10) == 4
    assert binary_search_recursive(a, 5) == -1
    assert binary_search_recursive([], 1) == -1


if __name__ == "__main__":
    main()
    print("PASS")
