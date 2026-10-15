
"""Task 172 - Implement stack that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3)

A parallel "running minimum" stack: mins.push(min(value, mins.peek())).
push/pop/min are all O(1); O(n) extra space.
"""


class MinStack:
    def __init__(self):
        self._main = []
        self._mins = []

    def push(self, value):
        self._main.append(value)
        self._mins.append(value if not self._mins else min(value, self._mins[-1]))

    def pop(self):
        if not self._main:
            raise IndexError("stack empty")
        self._mins.pop()
        return self._main.pop()

    def peek(self):
        if not self._main:
            raise IndexError("stack empty")
        return self._main[-1]

    def min(self):
        if not self._mins:
            raise IndexError("stack empty")
        return self._mins[-1]

    def is_empty(self):
        return not self._main

    def size(self):
        return len(self._main)


def main():
    s = MinStack()
    for v in (5, 3, 7, 3, 9):
        s.push(v)
    assert s.min() == 3 and s.peek() == 9 and s.size() == 5
    assert s.pop() == 9 and s.min() == 3
    s.push(1)
    assert s.min() == 1
    assert s.pop() == 1 and s.min() == 3
    assert s.pop() == 3 and s.min() == 3      # duplicate minimum still tracked
    assert s.pop() == 7 and s.min() == 3
    assert s.pop() == 3 and s.min() == 5
    assert s.pop() == 5 and s.is_empty()


if __name__ == "__main__":
    main()
    print("PASS")
