"""Task 256 - Top K frequent elements in array (heap). (Tier: Intermediate, CTCI Ch 10)

Count occurrences with a hash map, then stream the distinct values through a min-heap of
size k holding the best candidates. Documented tie-break: an entry is "worse" when its count
is smaller, or - on equal counts - when its value is larger. Evicting the worst keeps the k
most frequent values, and the result is ordered frequency-descending with value-ascending
tie-breaks. Time O(n + d log k) for d distinct values, space O(d + k).
"""

import heapq


def top_k_frequent(nums, k):
    if k <= 0:
        return []
    freq = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    heap = []                                  # (count, -value): smallest tuple = worst
    for value, count in freq.items():
        entry = (count, -value)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [(-neg) for _, neg in sorted(heap, key=lambda e: (-e[0], -e[1]))]


def main():
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]      # 3x1, 2x2, 1x3
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 3) == [1, 2, 3]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 6) == [1, 2, 3]   # k >= distinct: everything
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 99) == [1, 2, 3]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 0) == []
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 1) == [1]

    assert top_k_frequent([4, 4, 5, 5, 6], 2) == [4, 5]         # count tie: value ascending
    assert top_k_frequent([3, 1, 2], 2) == [1, 2]               # all count 1: value ascending
    assert top_k_frequent([2, 2, 3, 3, 1], 1) == [2]            # tie at the eviction edge
    assert top_k_frequent([-1, -1, 2, 2], 2) == [-1, 2]
    assert top_k_frequent([5], 1) == [5]
    assert top_k_frequent([], 3) == []
    assert top_k_frequent([7, 7, 7], 3) == [7]


if __name__ == "__main__":
    main()
    print("PASS")
