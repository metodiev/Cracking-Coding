"""Task 263 - Merge K sorted arrays (divide & conquer). (Tier: Intermediate, CTCI Ch 10)

Pairwise divide-and-conquer merge: recursively split the k arrays into two halves,
merge each half, then 2-way merge the results with a two-pointer scan (NO heap).
log k levels each doing O(n) work -> O(n log k) time, O(n) output space.

merge_k_heap is the matching heap-based method (task 262's idea) kept here as an
independent cross-check: both must agree on every input, and a disagreement would
expose a bug in one of them. merge_oracle flattens + sorts.

Conventions: k = 0 -> []; empty arrays inside the list are fine; the input arrays are
never modified (k = 1 returns a fresh copy).
"""

import heapq


def merge_two(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_k_divide(arrays):
    def divide(lo, hi):
        if hi - lo == 1:
            return list(arrays[lo])
        mid = (lo + hi) // 2
        return merge_two(divide(lo, mid), divide(mid, hi))

    return [] if not arrays else divide(0, len(arrays))


def merge_k_heap(arrays):
    """Independent heap-based merge used only for cross-checks (same result expected)."""
    heap = []
    for idx, arr in enumerate(arrays):
        if arr:
            heap.append((arr[0], idx, 0))
    heapq.heapify(heap)
    out = []
    while heap:
        val, idx, pos = heapq.heappop(heap)
        out.append(val)
        if pos + 1 < len(arrays[idx]):
            heapq.heappush(heap, (arrays[idx][pos + 1], idx, pos + 1))
    return out


def merge_oracle(arrays):
    """Flat sort reference used only for cross-checks."""
    return sorted(v for arr in arrays for v in arr)


def main():
    classic = [[1, 4, 5], [1, 3, 4], [2, 6]]
    assert merge_k_divide(classic) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert merge_k_divide([]) == []                                    # k = 0
    assert merge_k_divide([[3, 4]]) == [3, 4]                          # k = 1
    assert merge_k_divide([[]]) == []
    assert merge_k_divide([[1], [2, 3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]     # unequal lengths
    assert merge_k_divide([[2, 2], [2, 2], [2]]) == [2, 2, 2, 2, 2]           # duplicates
    assert merge_k_divide([[], [1], [], [2, 3]]) == [1, 2, 3]                 # empty arrays inside
    assert merge_k_divide([[-3, -1], [-2, 0]]) == [-3, -2, -1, 0]

    original = [3, 4]
    copy = merge_k_divide([original])
    assert copy == original and copy is not original       # k = 1 returns a fresh copy

    fixed = (classic, [], [[3, 4]], [[]], [[1], [2, 3, 4, 5], [6]], [[2, 2], [2, 2], [2]], [[-3, -1], [-2, 0]])
    for arrays in fixed:                                   # divide & conquer == heap == flat sort
        assert merge_k_divide(arrays) == merge_k_heap(arrays) == merge_oracle(arrays)

    import random

    rng = random.Random(263)
    for _ in range(300):
        k = rng.randint(0, 6)
        arrays = [sorted(rng.randint(-6, 6) for _ in range(rng.randint(0, 6))) for _ in range(k)]
        assert merge_k_divide(arrays) == merge_k_heap(arrays) == merge_oracle(arrays)


if __name__ == "__main__":
    main()
    print("PASS")
