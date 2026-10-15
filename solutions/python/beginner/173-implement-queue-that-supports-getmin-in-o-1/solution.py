
"""Task 173 - Implement queue that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3)

Two min-stacks: `inbox` for enqueue, `outbox` for dequeue. Each stack pair keeps a
running minimum, so min = min(inbox.min, outbox.min). All operations amortized O(1).
"""


class _MinStack:
    def __init__(self):
        self._data = []
        self._mins = []

    def push(self, value):
        self._data.append(value)
        self._mins.append(value if not self._mins else min(value, self._mins[-1]))

    def pop(self):
        self._mins.pop()
        return self._data.pop()

    def peek(self):
        return self._data[-1]

    def min(self):
        return self._mins[-1]

    def __len__(self):
        return len(self._data)


class MinQueue:
    def __init__(self):
        self._in = _MinStack()
        self._out = _MinStack()

    def enqueue(self, value):
        self._in.push(value)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("queue empty")
        if len(self._out) == 0:
            while len(self._in) > 0:
                self._out.push(self._in.pop())
        return self._out.pop()

    def min(self):
        if self.is_empty():
            raise IndexError("queue empty")
        if len(self._in) == 0:
            return self._out.min()
        if len(self._out) == 0:
            return self._in.min()
        return min(self._in.min(), self._out.min())

    def is_empty(self):
        return len(self._in) == 0 and len(self._out) == 0


def main():
    q = MinQueue()
    for v in (3, 1, 4, 1, 5):
        q.enqueue(v)
    assert q.min() == 1
    assert q.dequeue() == 3 and q.min() == 1
    assert q.dequeue() == 1 and q.min() == 1      # duplicate minimum still present
    assert q.dequeue() == 4 and q.min() == 1
    assert q.dequeue() == 1 and q.min() == 5
    q.enqueue(0)
    assert q.min() == 0
    assert q.dequeue() == 5 and q.dequeue() == 0 and q.is_empty()


if __name__ == "__main__":
    main()
    print("PASS")
