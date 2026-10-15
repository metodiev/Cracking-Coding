"""Task 065 - Implement priority queue using built-in heap. (Tier: Beginner, CTCI Ch 3)

Python ships heapq: a min-heap over lists. Max-priority via negated keys; ties via tuples.
"""

import heapq


class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0        # keeps FIFO order among equal priorities (stable)

    def push(self, item, priority):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        if not self._heap:
            raise IndexError("queue empty")
        return heapq.heappop(self._heap)[2]

    def peek(self):
        if not self._heap:
            raise IndexError("queue empty")
        return self._heap[0][2]

    def __len__(self):
        return len(self._heap)


def main():
    pq = PriorityQueue()
    pq.push("low", 5)
    pq.push("high", 1)
    pq.push("mid", 3)
    assert pq.peek() == "high"
    assert [pq.pop(), pq.pop(), pq.pop()] == ["high", "mid", "low"]
    assert len(pq) == 0


if __name__ == "__main__":
    main()
    print("PASS")
