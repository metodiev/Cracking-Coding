"""Task 066 - Implement min-heap push/pop. (Tier: Beginner, CTCI Ch 4)"""

class MinHeap:
    """Array-backed binary min-heap: parent <= children at every node."""

    def __init__(self):
        self._data = []

    def push(self, value):
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self):
        if not self._data:
            raise IndexError("heap empty")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return top

    def peek(self):
        if not self._data:
            raise IndexError("heap empty")
        return self._data[0]

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._data[parent] <= self._data[i]:
                break
            self._data[i], self._data[parent] = self._data[parent], self._data[i]
            i = parent

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            smallest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == i:
                break
            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest

    def __len__(self):
        return len(self._data)


def main():
    h = MinHeap()
    for v in (5, 3, 8, 1, 9, 2):
        h.push(v)
    assert h.peek() == 1
    drained = [h.pop() for _ in range(6)]
    assert drained == [1, 2, 3, 5, 8, 9], drained
    assert len(h) == 0

    h2 = MinHeap()
    h2.push(42)
    h2.push(7)
    assert h2.pop() == 7
    assert h2.pop() == 42


if __name__ == "__main__":
    main()
    print("PASS")
