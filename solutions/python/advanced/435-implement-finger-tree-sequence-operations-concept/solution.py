"""Task 435 - Implement finger tree (sequence operations) concept. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Concept version of a finger tree: a sequence annotated by a monoid so that every subtree caches
the annotation of the elements below it, which is what lets a real finger tree answer prefix and
range measures in O(log n). The annotation monoid used here is M = (count, sum) with identity
(0, 0) and componentwise addition; measure() returns the cached annotation of the whole sequence
and measure_range(lo, hi) folds it over a range. The spine operations push_front / push_back /
pop_front / pop_back / front / back / concat / split_at are simulated over a plain list while
keeping the annotation exact - documented honest O(n) internals, no worst-case finger-tree bound
claimed - and every scripted operation is asserted against a plain-list oracle (values, size,
cached measure and the monoid law combine(measure(a), measure(b)) == measure(concat(a, b))).
"""

IDENTITY = (0, 0)


def combine(left, right):
    return (left[0] + right[0], left[1] + right[1])


class FingerTree:
    def __init__(self, items=()):
        self.items = list(items)
        self._measure = (0, 0)
        for value in self.items:
            self._measure = combine(self._measure, (1, value))

    def size(self):
        return len(self.items)

    def measure(self):
        return self._measure

    def measure_range(self, lo, hi):
        if not 0 <= lo <= hi <= len(self.items):
            raise IndexError("range out of bounds")
        total = IDENTITY
        for value in self.items[lo:hi]:
            total = combine(total, (1, value))
        return total

    def front(self):
        return self.items[0] if self.items else None

    def back(self):
        return self.items[-1] if self.items else None

    def push_front(self, value):
        self.items.insert(0, value)
        self._measure = combine((1, value), self._measure)
        return self

    def push_back(self, value):
        self.items.append(value)
        self._measure = combine(self._measure, (1, value))
        return self

    def pop_front(self):
        if not self.items:
            return None
        value = self.items.pop(0)
        self._measure = combine((-1, -value), self._measure)
        return value

    def pop_back(self):
        if not self.items:
            return None
        value = self.items.pop()
        self._measure = combine((-1, -value), self._measure)
        return value

    def concat(self, other):
        joined = FingerTree(self.items)
        joined.items.extend(other.items)
        joined._measure = combine(self._measure, other._measure)
        return joined

    def split_at(self, index):
        if not 0 <= index <= len(self.items):
            raise IndexError("split index out of bounds")
        left = FingerTree(self.items[:index])
        right = FingerTree(self.items[index:])
        return left, right

    def to_list(self):
        return list(self.items)


def main():
    empty = FingerTree()
    assert empty.size() == 0
    assert empty.measure() == (0, 0)
    assert empty.to_list() == []
    assert empty.front() is None
    assert empty.back() is None
    assert empty.pop_front() is None
    assert empty.pop_back() is None

    tree = FingerTree()
    tree.push_back(1).push_back(2).push_back(3).push_front(0)
    assert tree.to_list() == [0, 1, 2, 3]
    assert tree.size() == 4
    assert tree.measure() == (4, 6)
    assert tree.front() == 0 and tree.back() == 3

    left, right = tree.split_at(0)
    assert left.to_list() == [] and left.measure() == (0, 0)
    assert right.to_list() == [0, 1, 2, 3] and right.measure() == (4, 6)
    left, right = tree.split_at(4)
    assert left.to_list() == [0, 1, 2, 3] and right.to_list() == []
    assert right.measure() == (0, 0)
    left, right = tree.split_at(2)
    assert left.to_list() == [0, 1] and right.to_list() == [2, 3]
    assert combine(left.measure(), right.measure()) == tree.measure()
    assert tree.measure_range(1, 3) == (2, 3)
    assert tree.measure_range(2, 2) == (0, 0)

    assert FingerTree([9]).concat(FingerTree()).to_list() == [9]
    assert FingerTree().concat(FingerTree([7])).to_list() == [7]
    assert FingerTree([1, 2]).concat(FingerTree([3])).measure() == (3, 6)
    assert tree.concat(tree).to_list() == [0, 1, 2, 3, 0, 1, 2, 3]

    tree = FingerTree([5, 6, 7])
    assert tree.pop_front() == 5
    assert tree.pop_back() == 7
    assert tree.to_list() == [6]
    assert tree.measure() == (1, 6)
    assert tree.pop_front() == 6
    assert tree.pop_front() is None
    assert tree.to_list() == []
    assert tree.measure() == (0, 0)

    tree = FingerTree()
    oracle = []
    seed = 4350
    for step in range(400):
        seed = (seed * 75 + 74) % 65537
        op = seed % 5
        if op == 0:
            seed = (seed * 75 + 74) % 65537
            value = seed % 50 - 25
            tree.push_back(value)
            oracle.append(value)
        elif op == 1:
            seed = (seed * 75 + 74) % 65537
            value = seed % 50 - 25
            tree.push_front(value)
            oracle.insert(0, value)
        elif op == 2:
            assert tree.pop_front() == (oracle.pop(0) if oracle else None)
        elif op == 3:
            assert tree.pop_back() == (oracle.pop() if oracle else None)
        else:
            seed = (seed * 75 + 74) % 65537
            index = seed % (len(oracle) + 1)
            left, right = tree.split_at(index)
            assert left.to_list() == oracle[:index], step
            assert right.to_list() == oracle[index:], step
            assert combine(left.measure(), right.measure()) == tree.measure(), step
            tree = left.concat(right)
        assert tree.to_list() == oracle, step
        assert tree.size() == len(oracle), step
        assert tree.measure() == (len(oracle), sum(oracle)), step
        if oracle:
            assert tree.front() == oracle[0] and tree.back() == oracle[-1], step
        seed = (seed * 75 + 74) % 65537
        lo = seed % (len(oracle) + 1)
        seed = (seed * 75 + 74) % 65537
        hi = lo + seed % (len(oracle) - lo + 1)
        assert tree.measure_range(lo, hi) == (hi - lo, sum(oracle[lo:hi])), step
    assert len(oracle) > 0


if __name__ == "__main__":
    main()
    print("PASS")
