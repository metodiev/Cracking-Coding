"""Task 181 - Range update, point query with Fenwick tree. (Tier: Beginner, CTCI Ch 4)

Dual of task 180: the Fenwick tree stores a *difference* array, so
- range_add(lo, hi, value): two point updates (+value at lo, -value at hi + 1)
- point_query(index): prefix sum of the difference array up to index
Both are O(log n); a whole range update therefore touches only O(log n) nodes.
"""


class RangeUpdateFenwick:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (size + 1)

    def _add(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def range_add(self, lo, hi, value):
        """Add `value` to every position in [lo, hi] (0-based, inclusive)."""
        if lo < 0 or hi >= self.n or lo > hi:
            raise IndexError("range out of bounds")
        self._add(lo + 1, value)
        self._add(hi + 2, -value)      # hi + 2 > n is a silent no-op suffix update

    def point_query(self, index):
        """Current value at `index` (0-based)."""
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        i = index + 1
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def snapshot(fw, size):
    return [fw.point_query(i) for i in range(size)]


def main():
    fw = RangeUpdateFenwick(6)
    fw.range_add(1, 3, 5)
    assert snapshot(fw, 6) == [0, 5, 5, 5, 0, 0]
    fw.range_add(0, 5, -2)
    assert snapshot(fw, 6) == [-2, 3, 3, 3, -2, -2]
    fw.range_add(2, 2, 10)                            # single position
    assert snapshot(fw, 6) == [-2, 3, 13, 3, -2, -2]
    fw.range_add(5, 5, 7)                             # ends exactly at n - 1
    assert snapshot(fw, 6) == [-2, 3, 13, 3, -2, 5]
    fw.range_add(0, 0, -3)                            # starts at position 0
    assert fw.point_query(0) == -5

    for bad in ((-1, 2, 1), (0, 6, 1), (3, 2, 1)):
        try:
            fw.range_add(*bad)
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError for %r" % (bad,))
    for bad_index in (-1, 6):
        try:
            fw.point_query(bad_index)
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError for index %d" % bad_index)

    empty = RangeUpdateFenwick(0)
    try:
        empty.point_query(0)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError on empty structure")

    # cross-check every position against a plain array after each random range add
    import random
    rng = random.Random(11)
    size = 12
    values = [0] * size
    fw2 = RangeUpdateFenwick(size)
    for _ in range(80):
        lo = rng.randrange(size)
        hi = rng.randrange(lo, size)
        value = rng.randint(-9, 9)
        for k in range(lo, hi + 1):
            values[k] += value
        fw2.range_add(lo, hi, value)
        assert snapshot(fw2, size) == values, (lo, hi, value)


if __name__ == "__main__":
    main()
    print("PASS")
