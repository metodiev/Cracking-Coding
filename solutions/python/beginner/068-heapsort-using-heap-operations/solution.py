
"""Task 068 - Heapsort using heap operations. (Tier: Beginner, CTCI Ch 10)

In-place: build a max-heap in the array, then repeatedly swap the root to the end.
O(n log n) time, O(1) extra space, not stable.
"""


def sift_down(a, i, size):
    while True:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < size and a[left] > a[largest]:
            largest = left
        if right < size and a[right] > a[largest]:
            largest = right
        if largest == i:
            return
        a[i], a[largest] = a[largest], a[i]
        i = largest


def heap_sort(a):
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):     # build max-heap: O(n)
        sift_down(a, i, n)
    for end in range(n - 1, 0, -1):         # repeatedly extract max
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)
    return a


def main():
    assert heap_sort([5, 3, 8, 1, 9, 2]) == [1, 2, 3, 5, 8, 9]
    assert heap_sort([]) == []
    assert heap_sort([7]) == [7]
    assert heap_sort([3, 3, 1]) == [1, 3, 3]


if __name__ == "__main__":
    main()
    print("PASS")
