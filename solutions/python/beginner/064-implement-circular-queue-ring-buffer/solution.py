"""Task 064 - Implement circular queue (ring buffer). (Tier: Beginner, CTCI Ch 3)"""


class CircularQueue:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data = [None] * capacity
        self._head = 0
        self._size = 0

    def enqueue(self, value):
        if self.is_full():
            raise OverflowError("queue full")
        self._data[(self._head + self._size) % len(self._data)] = value
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("queue empty")
        value = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        return value

    def front(self):
        if self.is_empty():
            raise IndexError("queue empty")
        return self._data[self._head]

    def rear(self):
        if self.is_empty():
            raise IndexError("queue empty")
        return self._data[(self._head + self._size - 1) % len(self._data)]

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == len(self._data)

    def size(self):
        return self._size


def main():
    q = CircularQueue(3)
    assert q.is_empty() and not q.is_full()
    q.enqueue(1); q.enqueue(2); q.enqueue(3)
    assert q.is_full() and q.size() == 3
    assert q.front() == 1 and q.rear() == 3
    assert q.dequeue() == 1
    q.enqueue(4)                       # wraps to slot 0
    assert q.front() == 2 and q.rear() == 4
    assert [q.dequeue(), q.dequeue(), q.dequeue()] == [2, 3, 4]
    assert q.is_empty()


if __name__ == "__main__":
    main()
    print("PASS")
