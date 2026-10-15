
"""Task 071 - Bubble sort (optimized). (Tier: Beginner, CTCI Ch 10)

Optimizations: (1) stop early when a pass makes no swaps; (2) shrink the
unsorted suffix by the number of swaps made (position of the last swap).
"""


def bubble_sort(a):
    n = len(a)
    end = n
    while end > 1:
        last_swap = 0
        for i in range(1, end):
            if a[i - 1] > a[i]:
                a[i - 1], a[i] = a[i], a[i - 1]
                last_swap = i
        if last_swap == 0:      # no swaps -> already sorted
            break
        end = last_swap
    return a


def main():
    assert bubble_sort([5, 3, 8, 1, 9, 2]) == [1, 2, 3, 5, 8, 9]
    assert bubble_sort([1, 2, 3]) == [1, 2, 3]      # early exit path
    assert bubble_sort([]) == []
    assert bubble_sort([2, 1]) == [1, 2]


if __name__ == "__main__":
    main()
    print("PASS")
