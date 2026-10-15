"""Task 183 - Implement iterative segment tree. (Tier: Beginner, CTCI Ch 4)

Bottom-up segment tree in a flat array of size 2n: leaves live at [n, 2n) and
parents at [1, n) with tree[i] = tree[2i] + tree[2i + 1]. Everything is a loop,
so there is no recursion overhead; compared to the recursive tree of task 182 it
needs 2n slots instead of 4n and never recurses, while producing the same
answers for the same build/update/query operations.
"""


class IterativeSegmentTree:
    def __init__(self, values):
        self.n = len(values)
        self.tree = [0] * (2 * self.n)
        for i, value in enumerate(values):
            self.tree[self.n + i] = value
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index, value):
        """Assign `value` to values[index] (O(log n))."""
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        i = self.n + index
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def range_sum(self, lo, hi):
        """Sum of values[lo..hi] inclusive; an empty range (lo > hi) yields 0."""
        if lo > hi:
            return 0
        if lo < 0 or hi >= self.n:
            raise IndexError("range out of bounds")
        total = 0
        left = lo + self.n
        right = hi + 1 + self.n               # classic half-open [left, right)
        while left < right:
            if left & 1:
                total += self.tree[left]
                left += 1
            if right & 1:
                right -= 1
                total += self.tree[right]
            left //= 2
            right //= 2
        return total


def main():
    values = [1, 3, 5, 7, 9, 11]
    st = IterativeSegmentTree(values)
    assert st.range_sum(0, 5) == 36
    assert st.range_sum(0, 0) == 1
    assert st.range_sum(5, 5) == 11
    assert st.range_sum(2, 3) == 12
    assert st.range_sum(1, 4) == 24
    assert st.range_sum(3, 2) == 0                     # empty range

    st.update(2, 10)
    values[2] = 10
    assert st.range_sum(0, 5) == 41
    assert st.range_sum(2, 2) == 10
    st.update(0, -4)
    values[0] = -4
    assert st.range_sum(0, 5) == 36

    single = IterativeSegmentTree([7])
    assert single.range_sum(0, 0) == 7
    single.update(0, 2)
    assert single.range_sum(0, 0) == 2

    empty = IterativeSegmentTree([])
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

    # cross-check every range against a naive loop after each random update
    import random
    rng = random.Random(33)
    data = [rng.randint(-20, 20) for _ in range(24)]
    st2 = IterativeSegmentTree(data)
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
