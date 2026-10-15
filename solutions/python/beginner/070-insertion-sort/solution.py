
"""Task 070 - Insertion sort. (Tier: Beginner, CTCI Ch 10)

Grow a sorted prefix by inserting each next element into place. O(n) on nearly sorted
data, O(n^2) worst case, stable and great for n < ~16.
"""


def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


def main():
    assert insertion_sort([5, 3, 8, 1, 9, 2]) == [1, 2, 3, 5, 8, 9]
    assert insertion_sort([1, 2, 3]) == [1, 2, 3]          # already sorted
    assert insertion_sort([]) == []
    assert insertion_sort([2, 1]) == [1, 2]


if __name__ == "__main__":
    main()
    print("PASS")
