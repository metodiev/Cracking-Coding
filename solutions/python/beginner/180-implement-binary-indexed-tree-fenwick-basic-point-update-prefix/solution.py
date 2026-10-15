
"""Task 180 - Implement binary indexed tree (Fenwick) basic. (Tier: Beginner, CTCI Ch 4)

1-based internal indexing over a 0-based array:
- update(i, delta): walk i += i & -i
- prefix_sum(i): walk i -= i & -i
Both O(log n). Handles point updates and prefix queries; range sums via subtraction.
"""


class Fenwick:
    def __init__(self, values):
        self.n = len(values)
        self.tree = [0] * (self.n + 1)
        for i, v in enumerate(values):
            self._update_index(i + 1, v)

    def _update_index(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def update(self, index, delta):
        """Add `delta` to values[index] (0-based)."""
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        self._update_index(index + 1, delta)

    def prefix_sum(self, index):
        """Sum of values[0..index] inclusive (0-based; pass -1 for empty sum)."""
        i = index + 1
        if i < 0 or i > self.n:
            raise IndexError("index out of range")
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total

    def range_sum(self, lo, hi):
        if lo > hi:
            return 0
        return self.prefix_sum(hi) - self.prefix_sum(lo - 1)


def main():
    values = [1, 3, 5, 7, 9, 11]
    fw = Fenwick(values)
    assert fw.prefix_sum(0) == 1
    assert fw.prefix_sum(5) == 36
    assert fw.range_sum(1, 3) == 15               # 3 + 5 + 7
    assert fw.range_sum(0, 0) == 1
    assert fw.range_sum(3, 2) == 0                # empty range

    fw.update(2, 5)                               # 5 -> 10
    values[2] += 5
    assert fw.prefix_sum(5) == 41
    fw.update(0, -1)                              # 1 -> 0
    values[0] -= 1
    assert fw.prefix_sum(0) == 0

    # cross-check every prefix and range against the plain array after each update
    import random
    rng = random.Random(9)
    fw2 = Fenwick(values)
    for _ in range(50):
        i = rng.randrange(len(values))
        delta = rng.randint(-5, 5)
        values[i] += delta
        fw2.update(i, delta)
        for end in range(len(values)):
            assert fw2.prefix_sum(end) == sum(values[:end + 1]), (end, fw2.prefix_sum(end))
        lo = rng.randrange(len(values))
        hi = rng.randrange(lo, len(values))
        assert fw2.range_sum(lo, hi) == sum(values[lo:hi + 1]), (lo, hi)

    empty = Fenwick([])
    assert empty.prefix_sum(-1) == 0


if __name__ == "__main__":
    main()
    print("PASS")
