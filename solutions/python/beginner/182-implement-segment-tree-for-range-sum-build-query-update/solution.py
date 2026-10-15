"""Task 182 - Implement segment tree for range sum (build/query/update). (Tier: Beginner, CTCI Ch 4)

Recursive segment tree stored in a flat array of size 4n; node 1 is the root and
covers [0, n - 1], children of node i are 2i and 2i + 1.
- build: O(n)
- update(index, value): assign a single position, O(log n)
- range_sum(lo, hi): inclusive range sum, O(log n)
"""


class SegmentTree:
    def __init__(self, values):
        self.n = len(values)
        self.tree = [0] * (4 * max(self.n, 1))
        if self.n:
            self._build(1, 0, self.n - 1, list(values))

    def _build(self, node, lo, hi, values):
        if lo == hi:
            self.tree[node] = values[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, index, value):
        """Assign `value` to values[index] (O(log n))."""
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node, lo, hi, index, value):
        if lo == hi:
            self.tree[node] = value
            return
        mid = (lo + hi) // 2
        if index <= mid:
            self._update(2 * node, lo, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, hi, index, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_sum(self, lo, hi):
        """Sum of values[lo..hi] inclusive; an empty range (lo > hi) yields 0."""
        if lo > hi:
            return 0
        if lo < 0 or hi >= self.n:
            raise IndexError("range out of bounds")
        return self._query(1, 0, self.n - 1, lo, hi)

    def _query(self, node, lo, hi, qlo, qhi):
        if qlo <= lo and hi <= qhi:
            return self.tree[node]
        mid = (lo + hi) // 2
        total = 0
        if qlo <= mid:
            total += self._query(2 * node, lo, mid, qlo, qhi)
        if qhi > mid:
            total += self._query(2 * node + 1, mid + 1, hi, qlo, qhi)
        return total


def main():
    values = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(values)
    assert st.range_sum(0, 5) == 36
    assert st.range_sum(0, 0) == 1
    assert st.range_sum(5, 5) == 11
    assert st.range_sum(2, 3) == 12
    assert st.range_sum(3, 2) == 0                     # empty range

    st.update(2, 10)                                   # 5 -> 10
    values[2] = 10
    assert st.range_sum(1, 3) == 20
    st.update(0, -4)
    values[0] = -4
    assert st.range_sum(0, 5) == 36
    assert st.range_sum(0, 0) == -4

    single = SegmentTree([7])
    assert single.range_sum(0, 0) == 7
    single.update(0, 2)
    assert single.range_sum(0, 0) == 2

    empty = SegmentTree([])
    assert empty.range_sum(0, -1) == 0

    for bad in ((-1, 2), (0, 6)):
        try:
            st.range_sum(*bad)
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError for %r" % (bad,))
    try:
        st.update(6, 0)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError on update")

    # cross-check every range against the plain array after each random update
    import random
    rng = random.Random(21)
    data = [rng.randint(-20, 20) for _ in range(17)]
    st2 = SegmentTree(data)
    for _ in range(40):
        i = rng.randrange(len(data))
        value = rng.randint(-20, 20)
        data[i] = value
        st2.update(i, value)
        for lo in range(len(data)):
            for hi in range(lo, len(data)):
                assert st2.range_sum(lo, hi) == sum(data[lo:hi + 1]), (lo, hi)


if __name__ == "__main__":
    main()
    print("PASS")
