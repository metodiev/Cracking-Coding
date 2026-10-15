
"""Task 179 - Implement simple heap-based median-of-stream class. (Tier: Beginner, CTCI Ch 4)

Same two-heap idea as task 164, but the heaps are hand-rolled here (no heapq), so
this file is fully self-contained: explicit sift-up/sift-down on arrays.
"""


class _Heap:
    """max_first=True builds a max-heap; otherwise a min-heap.

    Implementation note: values are stored as-is and compared through _better().
    """

    def __init__(self, max_first):
        self._data = []
        self._max_first = max_first

    def _better(self, a, b):
        return a > b if self._max_first else a < b

    def push(self, value):
        self._data.append(value)
        i = len(self._data) - 1
        while i > 0:
            parent = (i - 1) // 2
            if not self._better(self._data[i], self._data[parent]):
                break
            self._data[i], self._data[parent] = self._data[parent], self._data[i]
            i = parent

    def pop(self):
        if not self._data:
            raise IndexError("heap empty")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            i = 0
            while True:
                best = i
                left, right = 2 * i + 1, 2 * i + 2
                if left < len(self._data) and self._better(self._data[left], self._data[best]):
                    best = left
                if right < len(self._data) and self._better(self._data[right], self._data[best]):
                    best = right
                if best == i:
                    break
                self._data[i], self._data[best] = self._data[best], self._data[i]
                i = best
        return top

    def peek(self):
        if not self._data:
            raise IndexError("heap empty")
        return self._data[0]

    def __len__(self):
        return len(self._data)


class MedianOfStream:
    def __init__(self):
        self.low = _Heap(max_first=True)     # lower half (max on top)
        self.high = _Heap(max_first=False)   # upper half (min on top)
        self.count = 0

    def add(self, value):
        self.count += 1
        if len(self.low) == 0 or value <= self.low.peek():
            self.low.push(value)
        else:
            self.high.push(value)
        if len(self.low) > len(self.high) + 1:
            self.high.push(self.low.pop())
        elif len(self.high) > len(self.low):
            self.low.push(self.high.pop())

    def median(self):
        if self.count == 0:
            raise ValueError("no values yet")
        if len(self.low) == len(self.high):
            return (self.low.peek() + self.high.peek()) / 2.0
        return float(self.low.peek())


def main():
    m = MedianOfStream()
    for value, expected in [(1, 1.0), (2, 1.5), (3, 2.0), (4, 2.5), (5, 3.0)]:
        m.add(value)
        assert m.median() == expected, (value, m.median())

    m2 = MedianOfStream()
    m2.add(10)
    assert m2.median() == 10.0
    m2.add(-5)
    assert m2.median() == 2.5
    m2.add(-5)
    assert m2.median() == -5.0

    # randomized cross-check against sorting the values seen so far
    import random
    rng = random.Random(5)
    m3 = MedianOfStream()
    seen = []
    for _ in range(200):
        x = rng.randint(-50, 50)
        seen.append(x)
        m3.add(x)
        seen.sort()
        n = len(seen)
        expected = seen[n // 2] if n % 2 else (seen[n // 2 - 1] + seen[n // 2]) / 2.0
        assert m3.median() == expected, (x, m3.median(), expected)


if __name__ == "__main__":
    main()
    print("PASS")
