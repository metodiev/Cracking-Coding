"""Task 363 - Implement kd-tree for 2D nearest neighbor (basic). (Tier: Intermediate, CTCI Ch 4)

Build a 2D kd-tree by choosing, at every recursion level, the axis alternating with depth
(0 = x, 1 = y) and splitting at the median of the current points after sorting by that
axis with the input index as the tie-break, so the tree is balanced and fully
deterministic.  Nearest-neighbour search keeps the best candidate as the pair
(squared distance, input index): exploring the far child whenever the splitting
distance squared is <= the current best squared distance makes equal-distance ties
resolve to the smallest input index, matching brute force exactly.  Many LCG point
sets, including queries that are themselves stored points, are cross-checked against
the brute-force nearest neighbour.
"""


class KdTree:
    """Static 2D kd-tree with nearest-neighbour queries; points are (x, y) tuples."""

    def __init__(self, points):
        self.points = [tuple(p) for p in points]
        self.root = self._build(list(range(len(self.points))), 0)

    def _build(self, ids, depth):
        if not ids:
            return None
        axis = depth % 2
        ids = sorted(ids, key=lambda i: (self.points[i][axis], i))
        mid = len(ids) // 2
        point_index = ids[mid]
        return (self.points[point_index], point_index, axis,
                self._build(ids[:mid], depth + 1),
                self._build(ids[mid + 1:], depth + 1))

    def nearest(self, query):
        """Returns (input index, point, squared distance) of the nearest stored point."""
        assert self.root is not None, "empty kd-tree"
        best = self._search(self.root, tuple(query), None)
        index = best[1]
        return index, self.points[index], best[0]

    def _search(self, node, query, best):
        if node is None:
            return best
        point, index, axis, left, right = node
        diff = query[0] - point[0], query[1] - point[1]
        candidate = (diff[0] * diff[0] + diff[1] * diff[1], index)
        if best is None or candidate < best:
            best = candidate
        offset = query[axis] - point[axis]
        near, far = (left, right) if offset <= 0 else (right, left)
        best = self._search(near, query, best)
        # `<=` also explores the far side when the bound ties, so an equal-distance
        # point with a smaller input index can still win.
        if offset * offset <= best[0]:
            best = self._search(far, query, best)
        return best


def brute_nearest(points, query):
    """Independent oracle: (squared distance, index) minimum, ties by input index."""
    best = None
    for index, (x, y) in enumerate(points):
        dx = query[0] - x
        dy = query[1] - y
        candidate = (dx * dx + dy * dy, index)
        if best is None or candidate < best:
            best = candidate
    return best[1], tuple(points[best[1]]), best[0]


def lcg_points(count, seed):
    points = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        x = seed % 101 - 50
        seed = (seed * 75 + 74) % 65537
        y = seed % 101 - 50
        points.append((x, y))
    return points, seed


def main():
    # Hand-checked set: (3,3) is closest to (3,4), (1,1) is in the set, etc.
    points = [(0, 0), (3, 4), (1, 1), (-5, -5)]
    tree = KdTree(points)
    assert tree.nearest((3, 3)) == (1, (3, 4), 1)
    assert tree.nearest((1, 1)) == (2, (1, 1), 0)       # query is itself a stored point
    assert tree.nearest((0, 0)) == (0, (0, 0), 0)
    assert tree.nearest((-4, -6)) == (3, (-5, -5), 2)
    assert tree.nearest((7, 7)) == (1, (3, 4), 25)

    # Deterministic tie-break: four points at distance squared 2 from (1,1); index 0 wins.
    tie = KdTree([(0, 0), (2, 0), (0, 2), (2, 2)])
    assert tie.nearest((1, 1)) == (0, (0, 0), 2)
    assert brute_nearest([(0, 0), (2, 0), (0, 2), (2, 2)], (1, 1)) == (0, (0, 0), 2)

    # Duplicate points: the earliest input index among equal points must win.
    dup = KdTree([(5, 5), (5, 5), (1, 1)])
    assert dup.nearest((4, 4)) == (0, (5, 5), 2)
    assert dup.nearest((1, 1)) == (2, (1, 1), 0)

    single = KdTree([(-2, 7)])
    assert single.nearest((0, 0)) == (0, (-2, 7), 53)   # 4 + 49

    # LCG cross-check: many sets, queries both random and drawn from the set itself.
    seed = 60601
    for trial in range(40):
        seed = (seed * 75 + 74) % 65537
        size = 1 + seed % 40
        points, seed = lcg_points(size, seed)
        tree = KdTree(points)
        queries = list(points)                          # query points that are in the set
        for _ in range(12):
            seed = (seed * 75 + 74) % 65537
            x = seed % 121 - 60
            seed = (seed * 75 + 74) % 65537
            y = seed % 121 - 60
            queries.append((x, y))
        for query in queries:
            got = tree.nearest(query)
            want = brute_nearest(points, query)
            assert got == want, (trial, query, got, want)

    # Duplicate-heavy sets (coordinates in a tiny window) agree with brute force too.
    seed = 777
    for trial in range(10):
        seed = (seed * 75 + 74) % 65537
        size = 2 + seed % 30
        points = []
        for _ in range(size):
            seed = (seed * 75 + 74) % 65537
            x = seed % 4
            seed = (seed * 75 + 74) % 65537
            y = seed % 4
            points.append((x, y))
        tree = KdTree(points)
        for query in [(0, 0), (1, 1), (3, 3), (2, 1), (1, 2)]:
            assert tree.nearest(query) == brute_nearest(points, query), (trial, query)


if __name__ == "__main__":
    main()
    print("PASS")
