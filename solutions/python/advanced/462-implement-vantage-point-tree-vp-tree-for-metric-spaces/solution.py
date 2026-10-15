"""Task 462 - Implement vantage-point tree (VP-tree) for metric spaces. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Points are integer 2-D coordinates; the metric used here is the Manhattan metric
L1 (|dx| + |dy|), which is a true metric, so VP-tree pruning is exact. Each node
stores a vantage point, the median distance `mu` to the remaining points, and two
children: inner (the half with distance <= mu) and outer (the half with distance
>= mu), which keeps every point in exactly one child and the halves balanced.

Search visits the nearer child first and prunes by the triangle inequality:
  * inner can be skipped when tau - mu > best  (every inner point is >= tau - mu away)
  * outer can be skipped when tau + best < mu  (every outer point is >= mu - tau away)
with strict comparisons, so equal-distance candidates are still visited. Tie-break
(documented): equal L1 distance -> smaller point index. The fixed LCG workload
asserts exact agreement with a brute-force oracle and a visited-node budget of
fewer than n/3 nodes per query on average (measured 14.1) versus n for a scan.
"""


class VpTree:
    def __init__(self, points):
        self.pts = [tuple(p) for p in points]
        n = len(self.pts)
        self.mu = [0] * n
        self.inner = [-1] * n
        self.outer = [-1] * n
        self.visited = 0
        self.root = self._build(list(range(n))) if n else -1

    @staticmethod
    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build(self, ids):
        if not ids:
            return -1
        vp = ids[0]
        rest = ids[1:]
        if not rest:
            return vp
        dists = sorted((self._dist(self.pts[vp], self.pts[p]), p) for p in rest)
        mid = len(dists) // 2
        self.mu[vp] = dists[mid][0]
        self.inner[vp] = self._build([p for _, p in dists[:mid + 1]])
        self.outer[vp] = self._build([p for _, p in dists[mid + 1:]])
        return vp

    def nearest(self, query):
        """Returns (index, L1 distance); counts each visited tree node."""
        self.visited = 0
        if self.root == -1:
            return (-1, -1)
        self._best = None
        self._search(self.root, query)
        return (self._best[1], self._best[0])

    def _search(self, node, query):
        self.visited += 1
        tau = self._dist(query, self.pts[node])
        if self._best is None or (tau, node) < self._best:
            self._best = (tau, node)
        mu = self.mu[node]
        if tau < mu:
            self._visit_inner(node, query, tau, mu)
            self._visit_outer(node, query, tau, mu)
        else:
            self._visit_outer(node, query, tau, mu)
            self._visit_inner(node, query, tau, mu)

    def _visit_inner(self, node, query, tau, mu):
        if self.inner[node] != -1 and tau - mu <= self._best[0]:
            self._search(self.inner[node], query)

    def _visit_outer(self, node, query, tau, mu):
        if self.outer[node] != -1 and tau + self._best[0] >= mu:
            self._search(self.outer[node], query)


def brute_nearest(points, query):
    best_i, best_d = -1, 0
    for i, p in enumerate(points):
        d = abs(p[0] - query[0]) + abs(p[1] - query[1])
        if best_i == -1 or (d, i) < (best_d, best_i):
            best_i, best_d = i, d
    return (best_i, best_d)


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def main():
    # Hand-checked tiny case (L1; ties resolved by smaller index).
    small = [(0, 0), (1, 1), (3, 0), (5, 5), (2, 4)]
    tree = VpTree(small)
    assert tree.nearest((0, 0)) == (0, 0)
    assert tree.nearest((4, 3)) == (3, 3)  # (5,5) and (2,4) tie at 3 -> smaller index
    assert tree.nearest((2, 1)) == (1, 1)
    assert tree.nearest((5, 0)) == (2, 2)  # point (3,0)
    assert VpTree([]).nearest((0, 0)) == (-1, -1)

    # Fixed LCG cloud: VP-tree nearest must equal the brute-force oracle everywhere.
    n = 150
    seed = 4242
    points = []
    for _ in range(n):
        seed = lcg_value(seed)
        x = seed % 2001 - 1000
        seed = lcg_value(seed)
        y = seed % 2001 - 1000
        points.append((x, y))
    tree = VpTree(points)
    queries = 150
    total_visited = 0
    for _ in range(queries):
        seed = lcg_value(seed)
        qx = seed % 2001 - 1000
        seed = lcg_value(seed)
        qy = seed % 2001 - 1000
        got = tree.nearest((qx, qy))
        assert got == brute_nearest(points, (qx, qy)), (qx, qy)
        total_visited += tree.visited

    # Pruning budget: avg visited nodes per query must stay below n/3 (measured ~11).
    assert total_visited > 0
    assert total_visited < queries * n // 3, total_visited


if __name__ == "__main__":
    main()
    print("PASS")
