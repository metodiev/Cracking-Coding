"""Task 499 - Implement DBSCAN clustering (grid-based accelerate). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Density-based clustering (eps-neighbourhoods count points with squared distance <= eps^2,
minPts core threshold) with two interchangeable neighbourhood engines: the naive one scans all
n points, the accelerated one hashes points into a grid of cell size eps and probes only the
3x3 block of cells around a query - every point within eps must lie in that block, so the
results are identical while far fewer distances are evaluated (the self-test counts both and
asserts the grid strictly wins). Clusters are grown by the textbook queue expansion in index
order, which fixes the processing order of ambiguous border points, so the two engines must
produce element-wise identical labels (noise = -1); the comparison is also made
order-insensitively (sets of point-index sets per cluster plus the noise set). Test sets are
LCG-generated 2-D integer points, including duplicate-heavy sets, a mostly-noise sparse set,
a hand-checked chain, and points exactly eps apart (the <= boundary is inclusive).
"""

NOISE = -1


def sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def lcg_points(n, seed, limit=40):
    points = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        x = seed % limit
        seed = (seed * 75 + 74) % 65537
        y = seed % limit
        points.append((float(x), float(y)))
    return points


class NaiveEngine:
    def __init__(self, points, eps):
        self.points = points
        self.eps2 = eps * eps
        self.checks = 0

    def region(self, index):
        out = []
        px, py = self.points[index]
        for j, (qx, qy) in enumerate(self.points):
            self.checks += 1
            dx = px - qx
            dy = py - qy
            if dx * dx + dy * dy <= self.eps2:
                out.append(j)
        return out


class GridEngine:
    def __init__(self, points, eps):
        self.points = points
        self.eps2 = eps * eps
        self.eps = eps
        self.cells = {}
        for j, (x, y) in enumerate(points):
            key = (int(x // eps), int(y // eps))
            self.cells.setdefault(key, []).append(j)
        self.checks = 0

    def region(self, index):
        px, py = self.points[index]
        cx, cy = int(px // self.eps), int(py // self.eps)
        out = []
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in self.cells.get((gx, gy), ()):
                    self.checks += 1
                    dx = px - self.points[j][0]
                    dy = py - self.points[j][1]
                    if dx * dx + dy * dy <= self.eps2:
                        out.append(j)
        out.sort()
        return out


def dbscan(points, eps, min_pts, engine):
    n = len(points)
    labels = [NOISE] * n
    visited = [False] * n
    cluster = 0
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        neighbours = engine.region(start)
        if len(neighbours) < min_pts:
            continue                             # provisional noise; may become a border point
        labels[start] = cluster
        queue = [q for q in neighbours if q != start]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            if not visited[current]:
                visited[current] = True
                current_neighbours = engine.region(current)
                if len(current_neighbours) >= min_pts:
                    for q in current_neighbours:
                        if not visited[q]:
                            queue.append(q)
            if labels[current] == NOISE:
                labels[current] = cluster        # border point attaches to this cluster
        cluster += 1
    return labels


def canonical(labels):
    """Renumber clusters by first appearance so two runs compare label-for-label."""
    mapping = {}
    out = []
    for label in labels:
        if label == NOISE:
            out.append(NOISE)
        else:
            if label not in mapping:
                mapping[label] = len(mapping)
            out.append(mapping[label])
    return out


def partition(labels):
    """Order-insensitive partition: the set of per-cluster point-index sets plus the noise."""
    clusters = {}
    noise = set()
    for i, label in enumerate(labels):
        if label == NOISE:
            noise.add(i)
        else:
            clusters.setdefault(label, set()).add(i)
    return frozenset(frozenset(members) for members in clusters.values()), frozenset(noise)


def compare_set(points, eps, min_pts, label):
    naive = NaiveEngine(points, eps)
    grid = GridEngine(points, eps)
    naive_labels = dbscan(points, eps, min_pts, naive)
    grid_labels = dbscan(points, eps, min_pts, grid)
    assert canonical(naive_labels) == canonical(grid_labels), \
        "%s: engines disagree label-for-label" % label
    assert partition(naive_labels) == partition(grid_labels), "%s: partitions differ" % label
    assert grid.checks < naive.checks, "%s: grid must evaluate fewer distances" % label
    return naive_labels, naive.checks, grid.checks


def main():
    # Hand-checked chain: (0,0)-(1,0)-(2,0) with eps = 1.2 is one core chain (each interior
    # point sees 3 points, the ends see 2 and attach), (10,10) stays noise.
    chain = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (10.0, 10.0)]
    labels = dbscan(chain, 1.2, 2, NaiveEngine(chain, 1.2))
    assert labels == [0, 0, 0, NOISE], labels
    labels = dbscan(chain, 1.2, 3, NaiveEngine(chain, 1.2))
    assert labels == [0, 0, 0, NOISE], "all three points are core at minPts = 3"

    # Boundary: two points exactly eps apart are mutual neighbours (<= is inclusive).
    pair = [(0.0, 0.0), (2.0, 0.0), (20.0, 20.0)]
    labels = dbscan(pair, 2.0, 2, NaiveEngine(pair, 2.0))
    assert labels == [0, 0, NOISE]
    labels = dbscan(pair, 1.999, 2, NaiveEngine(pair, 1.999))
    assert labels == [NOISE, NOISE, NOISE], "just inside eps everything is noise"

    # LCG sets: naive and grid must agree exactly while the grid wins on distance checks.
    sets = (
        ("dense", lcg_points(60, 11), 3.0, 3),
        ("fine", lcg_points(80, 23), 2.0, 2),
        ("coarse", lcg_points(50, 37), 5.0, 4),
        ("sparse", lcg_points(40, 53, limit=100), 1.5, 5),
        ("duplicates", lcg_points(30, 67, limit=10), 2.0, 3),
    )
    report = []
    for name, points, eps, min_pts in sets:
        labels_naive, naive_checks, grid_checks = compare_set(points, eps, min_pts, name)
        report.append((name, len(points), eps, min_pts,
                       len({label for label in labels_naive if label != NOISE}),
                       sum(1 for label in labels_naive if label == NOISE),
                       naive_checks, grid_checks))

    # Documented structural checks on the dense set: several clusters, some noise allowed.
    dense = report[0]
    assert 1 <= dense[4] <= 12, "dense set: a handful of clusters"
    assert 0 <= dense[5] < dense[1], "dense set: not everything is noise"
    for name, count, _, _, _, _, naive_checks, grid_checks in report:
        assert grid_checks <= naive_checks // 2, name + ": the grid saves at least half"

    # All-noise control: points spaced far apart with a large minPts.
    lonely = lcg_points(12, 5, limit=1000)
    labels = dbscan(lonely, 1.0, 3, GridEngine(lonely, 1.0))
    assert set(labels) == {NOISE}, "scattered points are all noise"


if __name__ == "__main__":
    main()
    print("PASS")
