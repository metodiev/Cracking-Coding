"""Task 361 - Implement interval stabbing queries with segment tree. (Tier: Intermediate, CTCI Ch 4)

Stabbing queries ask "how many intervals cover point p?".  All query points are known up
front, so compress them to sorted indices and keep a segment tree over those compressed
points holding cover counts.  Each interval [left, right] (inclusive integer endpoints)
adds +1 to exactly the compressed range of query points it contains; a point query then
walks the leaf-to-root path summing the "range add" values stored on the covering nodes,
so no push-down is needed for range-add/point-query.  A hand-built example is asserted
and the structure is cross-checked against a naive cover count on deterministic LCG
intervals.
"""

from bisect import bisect_left, bisect_right


class StabbingSegmentTree:
    """Range add / point query over the distinct query points supplied at build time."""

    def __init__(self, points):
        self.points = sorted(set(points))
        self.n = len(self.points)
        self.tree = [0] * (2 * self.n) if self.n else []

    def add_interval(self, left, right):
        """Adds 1 to every registered point p with left <= p <= right."""
        if self.n == 0:
            return
        lo = bisect_left(self.points, left)
        hi = bisect_right(self.points, right)          # exclusive compressed index
        l, r = lo + self.n, hi + self.n
        while l < r:
            if l & 1:
                self.tree[l] += 1
                l += 1
            if r & 1:
                r -= 1
                self.tree[r] += 1
            l >>= 1
            r >>= 1

    def cover_count(self, point):
        """Number of added intervals that cover the registered query point."""
        index = bisect_left(self.points, point)
        assert index < self.n and self.points[index] == point, "point was not registered"
        total = 0
        i = index + self.n
        while i >= 1:
            total += self.tree[i]
            i >>= 1
        return total


def naive_cover_count(intervals, point):
    return sum(1 for left, right in intervals if left <= point <= right)


def lcg_intervals(count, seed):
    intervals = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        left = seed % 30
        seed = (seed * 75 + 74) % 65537
        width = 1 + seed % 10
        intervals.append((left, left + width))
    return intervals, seed


def main():
    # Hand-built example from the task description: intervals [1,4], [3,6], [5,8].
    intervals = [[1, 4], [3, 6], [5, 8]]
    points = [0, 2, 4, 6, 7, 9]
    tree = StabbingSegmentTree(points)
    for left, right in intervals:
        tree.add_interval(left, right)
    expected = [0, 1, 2, 2, 1, 0]                      # for 0, 2, 4, 6, 7, 9
    got = [tree.cover_count(p) for p in points]
    assert got == expected, got
    for p, want in zip(points, expected):
        assert naive_cover_count(intervals, p) == want, p

    # Adding the same interval twice doubles the counts.
    doubled = StabbingSegmentTree(points)
    for left, right in intervals + intervals:
        doubled.add_interval(left, right)
    assert [doubled.cover_count(p) for p in points] == [2 * v for v in expected]

    # Duplicate query points collapse to one leaf; single point works; no points is a no-op.
    dup = StabbingSegmentTree([3, 3, 5, 3])
    dup.add_interval(1, 4)
    assert dup.points == [3, 5] and dup.cover_count(3) == 1 and dup.cover_count(5) == 0
    solo = StabbingSegmentTree([7])
    solo.add_interval(7, 7)
    solo.add_interval(8, 9)
    assert solo.cover_count(7) == 1
    StabbingSegmentTree([]).add_interval(-3, 12)       # must not fail

    # LCG cross-check against the naive cover count, including points outside all intervals.
    seed = 20261
    for trial in range(25):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 40
        intervals, seed = lcg_intervals(count, seed)
        probe = list(range(-2, 46))
        tree = StabbingSegmentTree(probe)
        for left, right in intervals:
            tree.add_interval(left, right)
        for p in probe:
            assert tree.cover_count(p) == naive_cover_count(intervals, p), (trial, p)


if __name__ == "__main__":
    main()
    print("PASS")
