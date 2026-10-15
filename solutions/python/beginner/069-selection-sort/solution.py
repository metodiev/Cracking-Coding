
"""Task 069 - Selection sort. (Tier: Beginner, CTCI Ch 10)

Repeat: find the minimum of the unsorted suffix, swap it into place. O(n^2) comparisons,
O(n) swaps; not stable by default.
"""


def selection_sort(a):
    n = len(a)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
    return a


def main():
    assert selection_sort([5, 3, 8, 1, 9, 2]) == [1, 2, 3, 5, 8, 9]
    assert selection_sort([]) == []
    assert selection_sort([2, 1]) == [1, 2]
    assert selection_sort([1, 1, 1]) == [1, 1, 1]


if __name__ == "__main__":
    main()
    print("PASS")
