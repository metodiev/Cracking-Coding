"""Task 461 - Implement kd-tree nearest neighbor with pruning heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A 2-D kd-tree (median split on alternating axes) keeps a bounding box per node.
The nearest-neighbour search is best-first: search-queue entries are ordered by the
squared distance from the query point to the child's bounding box, and a subtree
whose lower bound strictly exceeds the best found squared distance is pruned
(the break is strict `>`, so equal-distance candidates are still explored).
Tie-break rule (documented): equal squared distance -> smaller point index.

The fixed LCG workload asserts (a) exact agreement with a brute-force oracle and
(b) a visited-node budget: the best-first search visits fewer than n/4 nodes per
query on average over the fixed 200-query cloud, versus n for a linear scan.
"""
import heapq


class KdTree:
    def __init__(self, points):
        self.pts = [tuple(p) for p in points]
        n = len(self.pts)
        self.left = [-1] * n
        self.right = [-1] * n
        self.box = [None] * n
        self.visited = 0
        self.root = self._build(list(range(n)), 0) if n else -1

    def _build(self, ids, depth):
        if not ids:
            return -1
        axis = depth & 1
        ids.sort(key=lambda i: (self.pts[i][axis], self.pts[i][1 - axis], i))
        mid = len(ids) // 2
        node = ids[mid]
        self.left[node] = self._build(ids[:mid], depth + 1)
        self.right[node] = self._build(ids[mid + 1:], depth + 1)
        xs = [self.pts[i][0] for i in ids]
        ys = [self.pts[i][1] for i in ids]
        self.box[node] = (min(xs), min(ys), max(xs), max(ys))
        return node

    @staticmethod
    def _dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _bound(self, node, q):
        xmin, ymin, xmax, ymax = self.box[node]
        dx = 0 if xmin <= q[0] <= xmax else min(abs(q[0] - xmin), abs(q[0] - xmax))
        dy = 0 if ymin <= q[1] <= ymax else min(abs(q[1] - ymin), abs(q[1] - ymax))
        return dx * dx + dy * dy

    def nearest(self, query):
        """Best-first search; returns (index, squared distance). Counts popped nodes."""
        self.visited = 0
        if self.root == -1:
            return (-1, -1)
        best_i = -1
        best_d = 0
        queue = [(0, self.root)]
        while queue:
            bound, node = heapq.heappop(queue)
            if best_i != -1 and bound > best_d:
                break  # prune: every remaining subtree has bound > best found
            self.visited += 1
            d = self._dist2(self.pts[node], query)
            if best_i == -1 or (d, node) < (best_d, best_i):
                best_d, best_i = d, node
            for child in (self.left[node], self.right[node]):
                if child != -1:
                    heapq.heappush(queue, (self._bound(child, query), child))
        return (best_i, best_d)


def brute_nearest(points, query):
    best_i, best_d = -1, 0
    for i, p in enumerate(points):
        d = (p[0] - query[0]) ** 2 + (p[1] - query[1]) ** 2
        if best_i == -1 or (d, i) < (best_d, best_i):
            best_i, best_d = i, d
    return (best_i, best_d)


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def main():
    # Hand-checked tiny case (ties resolved by smaller index).
    small = [(0, 0), (2, 3), (5, 4), (4, 1), (9, 6)]
    tree = KdTree(small)
    assert tree.nearest((1, 1)) == (0, 2)
    assert tree.nearest((4, 2)) == (3, 1)
    assert tree.nearest((3, 3)) == (1, 1)
    assert tree.nearest((3, 2)) == (1, 2)  # dist^2 2 for idx 1 and idx 3 -> smaller index
    assert KdTree([]).nearest((0, 0)) == (-1, -1)

    # Fixed LCG cloud: kd-tree nearest must equal the brute-force oracle everywhere.
    n = 200
    seed = 1234
    points = []
    for _ in range(n):
        seed = lcg_value(seed)
        x = seed % 2001 - 1000
        seed = lcg_value(seed)
        y = seed % 2001 - 1000
        points.append((x, y))
    tree = KdTree(points)
    queries = 200
    total_visited = 0
    for _ in range(queries):
        seed = lcg_value(seed)
        qx = seed % 2001 - 1000
        seed = lcg_value(seed)
        qy = seed % 2001 - 1000
        got = tree.nearest((qx, qy))
        assert got == brute_nearest(points, (qx, qy)), (qx, qy)
        total_visited += tree.visited

    # Pruning budget: avg visited nodes per query must stay below n/4 (measured 8.0 here).
    assert total_visited > 0
    assert total_visited < queries * n // 4, total_visited


if __name__ == "__main__":
    main()
    print("PASS")
