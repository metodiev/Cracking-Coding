"""Task 051 - Implement dynamic array (basic ArrayList). (Tier: Beginner, CTCI Ch 1)"""
class DynamicArray:
    """Doubling amortized O(1) append, like ArrayList."""

    def __init__(self):
        self._data = [0] * 2
        self._size = 0

    def push_back(self, value):
        if self._size == len(self._data):
            self._grow()
        self._data[self._size] = value
        self._size += 1

    def _grow(self):
        bigger = [0] * (len(self._data) * 2)
        bigger[:self._size] = self._data[:self._size]
        self._data = bigger

    def get(self, index):
        self._check(index)
        return self._data[index]

    def set(self, index, value):
        self._check(index)
        self._data[index] = value

    def pop_back(self):
        if self._size == 0:
            raise IndexError("empty")
        self._size -= 1
        return self._data[self._size]

    def _check(self, index):
        if not 0 <= index < self._size:
            raise IndexError("index out of range")

    def size(self):
        return self._size

    def capacity(self):
        return len(self._data)


def main():
    a = DynamicArray()
    for i in range(5):
        a.push_back(i * 10)
    assert a.size() == 5
    assert a.capacity() == 8          # doubled from 2 -> 4 -> 8
    assert a.get(3) == 30
    a.set(0, 99)
    assert a.get(0) == 99
    assert a.pop_back() == 40
    assert a.size() == 4


if __name__ == "__main__":
    main()
    print("PASS")
