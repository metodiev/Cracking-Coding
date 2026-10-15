"""Task 087 - Implement merge and count inversions using merge sort. (Tier: Beginner, CTCI Ch 10)

Augmented merge sort: when an element from the right half wins the merge, every
remaining element of the left half is larger than it, so the running count grows
by the number of those remaining elements. O(n log n) time, O(n) space.
"""


def count_inversions(nums):
    return _merge_sort(nums)[1]


def _merge_sort(items):
    """Return (sorted_copy, inversion_count)."""
    if len(items) <= 1:
        return list(items), 0
    mid = len(items) // 2
    left, left_count = _merge_sort(items[:mid])
    right, right_count = _merge_sort(items[mid:])
    merged = []
    count = left_count + right_count
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            count += len(left) - i
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged, count


def count_inversions_naive(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                count += 1
    return count


def pseudo_random_list(size, seed=7):
    values = []
    state = seed
    for _ in range(size):
        state = (state * 48271) % 2147483647
        values.append(state % 1000)
    return values


def main():
    assert count_inversions([]) == 0
    assert count_inversions([1]) == 0
    assert count_inversions([1, 2, 3, 4]) == 0
    assert count_inversions([3, 2, 1]) == 3
    assert count_inversions([5, 4, 3, 2, 1]) == 10
    assert count_inversions([2, 1, 2, 1]) == 3
    assert count_inversions([2, 4, 1, 3, 5]) == 3
    assert count_inversions([1, 1, 1]) == 0
    assert count_inversions(list(range(300, 0, -1))) == 44850

    # agreement with the naive O(n^2) count on pseudo-random arrays
    for size in (2, 7, 50, 200):
        nums = pseudo_random_list(size)
        assert count_inversions(nums) == count_inversions_naive(nums), size


if __name__ == "__main__":
    main()
    print("PASS")
