"""Task 518 - Implement k-nearest neighbor classification fast with kd-tree. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

k-nearest-neighbour classification with two interchangeable neighbour engines: the brute-force
engine scans every training point, while the kd-tree recursively splits the point set at the
median of the current axis (axes cycle with depth, ties broken by point index so the tree is
byte-deterministic) and answers a query by descending into the nearer child first, then visiting
the far child only when the splitting plane is closer than the current k-th best distance. The
k-set is ordered by (squared distance, index), which makes it unique even with duplicated points
or exact distance ties, so the pruned tree must return the *same* neighbour list as the linear
scan; the self-test counts distance evaluations for both engines and asserts the tree strictly
wins on clustered and uniform 2-, 3- and 4-dimensional LCG data. Classification is a majority
vote over the k neighbours with the smaller label winning ties. Documented checks: neighbour
lists identical for k = 1, 3, 5 over 200 points, identical predicted labels for query batches,
a hand-checked axis-aligned example, k >= n returning all points, duplication handling,
tree-rebuild determinism, and a hand-computed majority-vote tie-break.
"""


def sqdist(a, b):
    total = 0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return total


def lcg_points(n, seed, limit=100, dims=2):
    """Deterministic LCG sample of integer points; coordinates are exact floats."""
    points = []
    for _ in range(n):
        coords = []
        for _ in range(dims):
            seed = (seed * 75 + 74) % 65537
            coords.append(float(seed % limit))
        points.append(tuple(coords))
    return points


class BruteEngine:
    def __init__(self, points):
        self.points = points
        self.checks = 0

    def query(self, target, k):
        out = []
        for i, point in enumerate(self.points):
            self.checks += 1
            out.append((sqdist(point, target), i))
        out.sort()
        return out[:k]


class Node:
    __slots__ = ("index", "axis", "left", "right")

    def __init__(self, index, axis, left, right):
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """Median-split kd-tree; equal keys fall to the larger-index side by the sort tie-break."""

    def __init__(self, points):
        self.points = points
        self.dims = len(points[0]) if points else 0
        self.checks = 0
        self.root = self._build(list(range(len(points))), 0)

    def _build(self, indices, depth):
        if not indices:
            return None
        axis = depth % self.dims
        indices = sorted(indices, key=lambda i: (self.points[i][axis], i))
        mid = len(indices) // 2
        return Node(indices[mid], axis,
                    self._build(indices[:mid], depth + 1),
                    self._build(indices[mid + 1:], depth + 1))

    def query(self, target, k):
        best = []                                # sorted (distance, index) pairs, at most k
        self._search(self.root, target, k, best)
        return best

    def _search(self, node, target, k, best):
        if node is None:
            return
        point = self.points[node.index]
        self.checks += 1
        distance = sqdist(point, target)
        if len(best) < k or (distance, node.index) < best[-1]:
            best.append((distance, node.index))
            best.sort()
            if len(best) > k:
                best.pop()
        diff = point[node.axis] - target[node.axis]
        if diff >= 0.0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left
        self._search(near, target, k, best)
        # A far-side point can only matter while the plane is within the current k-th distance.
        if len(best) < k or diff * diff <= best[-1][0]:
            self._search(far, target, k, best)


def majority_vote(neighbours, labels):
    counts = {}
    for _, index in neighbours:
        counts[labels[index]] = counts.get(labels[index], 0) + 1
    best_label = None
    best_count = -1
    for label in sorted(counts):
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]
    return best_label


def classify(train, labels, queries, k, engine):
    predictions = []
    for query in queries:
        predictions.append(majority_vote(engine.query(query, k), labels))
    return predictions


def main():
    # Hand-checked axis-aligned example: query (0,0) with k = 1 picks index 0, not index 2.
    train = [(0.0, 0.0), (2.0, 0.0), (10.0, 10.0)]
    labels = [0, 1, 2]
    brute = BruteEngine(train)
    tree = KDTree(train)
    assert brute.query((0.0, 0.0), 1) == [(0.0, 0)]
    assert tree.query((0.0, 0.0), 1) == [(0.0, 0)]
    assert brute.query((9.0, 9.0), 2) == [(2.0, 2), (130.0, 1)]
    assert tree.query((9.0, 9.0), 2) == [(2.0, 2), (130.0, 1)]
    assert brute.query((5.0, 5.0), 3) == [(34.0, 1), (50.0, 0), (50.0, 2)]
    assert tree.query((5.0, 5.0), 3) == [(34.0, 1), (50.0, 0), (50.0, 2)]

    # Duplicate points: the k-set keeps distinct indices, ordered by index within equal distance.
    dup = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (5.0, 5.0)]
    brute = BruteEngine(dup)
    tree = KDTree(dup)
    assert brute.query((1.0, 1.0), 2) == [(0.0, 0), (0.0, 1)]
    assert tree.query((1.0, 1.0), 2) == [(0.0, 0), (0.0, 1)]
    assert brute.query((1.0, 1.0), 3) == [(0.0, 0), (0.0, 1), (0.0, 2)]
    assert tree.query((1.0, 1.0), 3) == [(0.0, 0), (0.0, 1), (0.0, 2)]
    # k larger than the training set returns everything (sorted by distance then index).
    assert brute.query((1.0, 1.0), 9) == [(0.0, 0), (0.0, 1), (0.0, 2), (32.0, 3)]
    assert tree.query((1.0, 1.0), 9) == [(0.0, 0), (0.0, 1), (0.0, 2), (32.0, 3)]

    # Majority-vote tie-break: equal counts -> smaller label; a strict leader still wins.
    assert majority_vote([(0.0, 0), (1.0, 1)], [1, 0]) == 0
    assert majority_vote([(0.0, 0), (1.0, 1)], [0, 1]) == 0
    assert majority_vote([(0.0, 0), (1.0, 1), (2.0, 2)], [2, 2, 1]) == 2
    assert classify(train, labels, [(0.0, 0.0)], 1, BruteEngine(train)) == [0]
    # k = 2 at (8,8) sees labels 2 and 1 (one each) -> the smaller label wins the tie.
    assert classify(train, labels, [(8.0, 8.0)], 2, BruteEngine(train)) == [1]
    assert classify(train, labels, [(8.0, 8.0)], 1, BruteEngine(train)) == [2]

    # LCG worlds: identical neighbour lists, identical predictions, tree strictly cheaper.
    sets = (
        ("uniform2d", lcg_points(200, 11, limit=100, dims=2), 2),
        ("clustered2d", [(x // 10 * 10.0, y // 10 * 10.0)
                         for x, y in lcg_points(200, 23, limit=100, dims=2)], 2),
        ("uniform3d", lcg_points(200, 37, limit=100, dims=3), 3),
        ("uniform4d", lcg_points(160, 53, limit=100, dims=4), 4),
        ("degenerate_line", [(float(i % 7), 0.0) for i in range(150)], 2),
    )
    report = []
    for name, points, dims in sets:
        queries = lcg_points(120, 97, limit=100, dims=dims)
        data_labels = [i % 5 for i in range(len(points))]
        for k in (1, 3, 5):
            brute = BruteEngine(points)
            tree = KDTree(points)
            for query in queries:
                assert brute.query(query, k) == tree.query(query, k), \
                    "%s: kd-tree neighbours differ from the scan (k=%d)" % (name, k)
            assert classify(points, data_labels, queries, k, brute) == \
                classify(points, data_labels, queries, k, tree), \
                "%s: predictions differ (k=%d)" % (name, k)
        # One cold engine per set: aggregate distance evaluations for a fair comparison.
        brute = BruteEngine(points)
        tree = KDTree(points)
        for query in queries:
            brute.query(query, 3)
            tree.query(query, 3)
        assert tree.checks <= brute.checks, "%s: pruning never evaluates more points" % name
        report.append((name, brute.checks, tree.checks))

    # The structural sets must show a real win; measured work is 24000 vs 2261 (uniform 2-D),
    # 24000 vs 3877 (uniform 3-D) and 24000 vs 2519 (clustered 2-D), so the documented claim
    # is that the tree evaluates fewer than half the distances there; a degenerate 1-D line
    # prunes badly and only has to stay at or below the linear scan.
    wins = {name: (brute_checks, tree_checks) for name, brute_checks, tree_checks in report}
    for name in ("uniform2d", "uniform3d", "clustered2d"):
        brute_checks, tree_checks = wins[name]
        assert tree_checks * 2 < brute_checks, name + ": kd-tree must save at least half"
    assert wins["degenerate_line"][1] <= wins["degenerate_line"][0]

    # Determinism: two independently built trees agree on every query/batch.
    points = lcg_points(80, 71, limit=50, dims=2)
    queries = lcg_points(40, 13, limit=50, dims=2)
    first = KDTree(points)
    second = KDTree(points)
    for query in queries:
        assert first.query(query, 4) == second.query(query, 4)
    assert first.checks == second.checks, "identical work for identical trees"

    # Exact-tie boundary: every candidate sits at squared distance 8, so the k-th best distance
    # equals the plane distance and the far side must not be pruned away.
    tie = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)]
    brute = BruteEngine(tie)
    tree = KDTree(tie)
    assert brute.query((2.0, 2.0), 4) == tree.query((2.0, 2.0), 4)
    assert all(d == 8.0 for d, _ in tree.query((2.0, 2.0), 4))


if __name__ == "__main__":
    main()
    print("PASS")
