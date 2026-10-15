"""Task 067 - Implement max-heap push/pop. (Tier: Beginner, CTCI Ch 4)"""


class MaxHeap:
    """Parent >= children. Same sifts as a min-heap with flipped comparisons."""

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
            if self._data[parent] >= self._data[i]:
                break
            self._data[i], self._data[parent] = self._data[parent], self._data[i]
            i = parent

    def _sift_down(self, i):
        n = len(self._data)
        while True:
            largest = i
            left, right = 2 * i + 1, 2 * i + 2
            if left < n and self._data[left] > self._data[largest]:
                largest = left
            if right < n and self._data[right] > self._data[largest]:
                largest = right
            if largest == i:
                break
            self._data[i], self._data[largest] = self._data[largest], self._data[i]
            i = largest

    def __len__(self):
        return len(self._data)


def main():
    h = MaxHeap()
    for v in (5, 3, 8, 1, 9, 2):
        h.push(v)
    assert h.peek() == 9
    assert [h.pop() for _ in range(6)] == [9, 8, 5, 3, 2, 1]
    assert len(h) == 0


if __name__ == "__main__":
    main()
    print("PASS")
