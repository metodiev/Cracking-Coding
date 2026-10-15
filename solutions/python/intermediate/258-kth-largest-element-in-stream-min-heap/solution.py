"""Task 258 - Kth largest element in a stream (min-heap). (Tier: Intermediate, CTCI Ch 10)

Keep a min-heap of at most k values: it holds exactly the k largest values seen so far, so
its root is the k-th largest. add() returns None until at least k values have been seen
(documenting the behaviour when k is larger than the initial count). Each add is O(log k).
"""

import heapq


class KthLargest:
    def __init__(self, k, nums):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.heap = list(nums)
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        if len(self.heap) < self.k:
            return None
        return self.heap[0]


def main():
    kl = KthLargest(3, [4, 5, 8, 2])
    assert kl.add(3) == 4
    assert kl.add(5) == 5
    assert kl.add(10) == 5
    assert kl.add(9) == 8
    assert kl.add(4) == 8

    # k larger than the initial count: None until k values have been seen
    grow = KthLargest(5, [3, 1])
    assert grow.add(2) is None
    assert grow.add(7) is None
    assert grow.add(9) == 1                    # 5 values {1,2,3,7,9} -> 5th largest is 1
    assert grow.add(4) == 2                    # {2,3,4,7,9} -> 2

    top = KthLargest(1, [4, 5, 8, 2])          # k = 1 is the running maximum
    assert top.add(3) == 8
    assert top.add(10) == 10
    assert top.add(4) == 10

    dupes = KthLargest(2, [5, 5, 5])
    assert dupes.add(5) == 5
    assert dupes.add(4) == 5
    assert dupes.add(6) == 5
    assert dupes.add(7) == 6

    empty = KthLargest(2, [])
    assert empty.add(5) is None
    assert empty.add(3) == 3
    assert empty.add(10) == 5
    assert empty.add(8) == 8

    start = KthLargest(2, [1, 2, 3])           # initial pile is trimmed to the top k
    assert start.heap == [2, 3] or start.heap == [3, 2]
    assert start.add(0) == 2


if __name__ == "__main__":
    main()
    print("PASS")
