"""Task 050 - Implement queue with array (enqueue/dequeue). (Tier: Beginner, CTCI Ch 3)"""

class QueueArray:
    """Circular buffer: O(1) enqueue and dequeue, no shifting."""

    def __init__(self, capacity):
        self._data = [0] * capacity
        self._head = 0
        self._size = 0

    def enqueue(self, value):
        if self._size == len(self._data):
            raise OverflowError("queue full")
        self._data[(self._head + self._size) % len(self._data)] = value
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("queue empty")
        value = self._data[self._head]
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        return value

    def front(self):
        if self.is_empty():
            raise IndexError("queue empty")
        return self._data[self._head]

    def is_empty(self):
        return self._size == 0

    def size(self):
        return self._size


def main():
    q = QueueArray(3)
    q.enqueue(1); q.enqueue(2); q.enqueue(3)
    assert q.front() == 1
    assert q.dequeue() == 1
    q.enqueue(4)                     # wraps around
    assert q.dequeue() == 2
    assert q.dequeue() == 3
    assert q.dequeue() == 4
    assert q.is_empty()


if __name__ == "__main__":
    main()
    print("PASS")
