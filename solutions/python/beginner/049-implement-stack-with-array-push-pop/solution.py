"""Task 049 - Implement stack with array (push/pop). (Tier: Beginner, CTCI Ch 3)"""

class StackArray:
    def __init__(self, capacity):
        self._data = [0] * capacity
        self._top = -1

    def push(self, value):
        if self._top + 1 == len(self._data):
            raise OverflowError("stack full")
        self._top += 1
        self._data[self._top] = value

    def pop(self):
        if self.is_empty():
            raise IndexError("stack empty")
        value = self._data[self._top]
        self._top -= 1
        return value

    def peek(self):
        if self.is_empty():
            raise IndexError("stack empty")
        return self._data[self._top]

    def is_empty(self):
        return self._top < 0

    def size(self):
        return self._top + 1


def main():
    s = StackArray(3)
    assert s.is_empty()
    s.push(1); s.push(2); s.push(3)
    assert s.peek() == 3 and s.size() == 3
    assert s.pop() == 3
    assert s.pop() == 2
    assert s.pop() == 1
    assert s.is_empty()


if __name__ == "__main__":
    main()
    print("PASS")
