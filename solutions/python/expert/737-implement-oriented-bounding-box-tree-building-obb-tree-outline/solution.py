"""Task 737 - Implement oriented bounding box tree building (OBB-tree) outline.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

An OBB tree splits a point set into groups and wraps every group in a *rotated* box that hugs
its actual shape, instead of the axis-aligned box a kd-tree would use.  Building one node is:
the covariance matrix of the node's points describes the directions the points spread along,
its dominant eigenvector is the long axis of the cloud, so the node is split at the median
along that axis and the box is expressed in the eigenvector frame; the half extents are the
largest absolute projections of the points onto the three frame axes, which makes the box
tight by construction.

The eigenframe is found by numerically deterministic power iteration: start from a fixed unit
vector, repeatedly multiply by the covariance matrix and renormalise; the Rayleigh quotient is
non-decreasing and the residual ||A v - lambda v|| halves per iteration when the dominant
eigenvalue is separated.  The second axis is the dominant eigenvector of the deflated matrix
A - lambda1 e1 e1^T (power iteration again, projected orthogonally to e1), the third is the
cross product of the first two.  All of that runs in floating point.

Simplified, as an outline: covariance and median splitting instead of a surface-area-heuristic
top-down build, power iteration instead of a symmetric eigen solver, leaves from a fixed size
threshold, no refitting/merging of boxes while descending, and the frame is a float
approximation of an orthonormal basis - hence a documented tolerance in the containment checks.
Verified: every point really lies in the OBB of every node on its root path (so the subtree
cover is genuine), the leaf index sets partition the input exactly, leaves respect the size
threshold, the frame is orthonormal to 1e-9, every node is tight (some point attains each half
extent), power iteration converges (Rayleigh quotient non-decreasing, small residual), and on
rotated input the OBB volume is strictly below the axis-aligned volume while on axis-aligned
input the two agree - the independent tightness oracle.
"""
import math

LEAF_SIZE = 4
TOL = 1e-9


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def scale(a, s):
    return [v * s for v in a]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def mean_of(points):
    n = len(points)
    return [sum(p[i] for p in points) / n for i in range(3)]


def covariance(points, mean):
    n = len(points)
    matrix = [[0.0] * 3 for _ in range(3)]
    for p in points:
        d = sub(p, mean)
        for i in range(3):
            for j in range(3):
                matrix[i][j] += d[i] * d[j]
    return [[value / n for value in row] for row in matrix]


def mat_vec(matrix, vector):
    return [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]


def power_iteration(matrix, seed=None, iterations=64, orthogonal_to=None):
    """Dominant eigenvector (float) with the Rayleigh quotient trace of the iterations."""
    vector = list(seed) if seed else [1.0 / math.sqrt(3)] * 3
    if orthogonal_to is not None:
        projection = dot(vector, orthogonal_to)
        vector = [vector[i] - projection * orthogonal_to[i] for i in range(3)]
    length = norm(vector)
    vector = scale(vector, 1.0 / length) if length > TOL else [1.0, 0.0, 0.0]
    quotient = 0.0
    trace = []
    for _ in range(iterations):
        product = mat_vec(matrix, vector)
        if orthogonal_to is not None:            # keep the iterate orthogonal to e1
            projection = dot(product, orthogonal_to)
            product = [product[i] - projection * orthogonal_to[i] for i in range(3)]
        length = norm(product)
        if length <= TOL:
            break
        vector = scale(product, 1.0 / length)
        quotient = dot(vector, mat_vec(matrix, vector))
        trace.append(quotient)
    residual = norm([mat_vec(matrix, vector)[i] - quotient * vector[i] for i in range(3)])
    return vector, residual, trace


def orthonormal_frame(matrix):
    """(e1, e2, e3) from the covariance matrix; e2 lives in the deflated complement of e1."""
    e1, residual1, trace1 = power_iteration(matrix)
    lambda1 = trace1[-1] if trace1 else 0.0
    deflated = [[matrix[i][j] - lambda1 * e1[i] * e1[j] for j in range(3)] for i in range(3)]
    e2, residual2, trace2 = power_iteration(deflated, seed=(1.0, 2.0, 3.0), orthogonal_to=e1)
    e3 = cross(e1, e2)
    if norm(e3) <= TOL:                          # degenerate cloud: any completion will do
        e3 = cross(e1, [0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else cross(e1, [0.0, 1.0, 0.0])
    e3 = scale(e3, 1.0 / norm(e3))
    e2 = cross(e3, e1)
    e2 = scale(e2, 1.0 / norm(e2))
    return (e1, e2, e3), (residual1, residual2), (trace1, trace2)


class Obb:
    """Oriented box: centre plus an orthonormal frame and three half extents."""

    __slots__ = ("center", "axes", "half")

    def __init__(self, center, axes, half):
        self.center, self.axes, self.half = center, axes, half

    def local(self, point):
        return [dot(sub(point, self.center), axis) for axis in self.axes]

    def contains_point(self, point, tol=TOL):
        return all(abs(value) <= half + tol for value, half in zip(self.local(point), self.half))

    def volume(self):
        return (2 * self.half[0]) * (2 * self.half[1]) * (2 * self.half[2])


def build_obb_tree(points):
    """Flat node arrays: obb, indices, left, right.  Leaves hold at most LEAF_SIZE points."""
    obbs, index_sets, left, right = [], [], [], []
    frame_reports = []

    def recurse(indices):
        node = len(obbs)
        obbs.append(None)
        index_sets.append(tuple(indices))
        left.append(-1)
        right.append(-1)
        cloud = [points[i] for i in indices]
        center = mean_of(cloud)
        axes, residuals, quotients = orthonormal_frame(covariance(cloud, center))
        half = [max(abs(dot(sub(p, center), axis)) for p in cloud) for axis in axes]
        obbs[node] = Obb(center, axes, half)
        frame_reports.append((residuals, quotients))
        if len(indices) <= LEAF_SIZE:
            return node
        axis = max(range(3), key=lambda a: half[a])
        ordered = sorted(indices,
                         key=lambda i: (dot(sub(points[i], center), axes[axis]), i))
        mid = len(ordered) // 2
        left[node] = recurse(ordered[:mid])
        right[node] = recurse(ordered[mid:])
        return node

    root = recurse(list(range(len(points)))) if points else -1
    return {"obb": obbs, "indices": index_sets, "left": left, "right": right,
            "root": root, "frames": frame_reports}


def aabb_of(points):
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    return Obb([(lo[i] + hi[i]) / 2 for i in range(3)],
               [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
               [(hi[i] - lo[i]) / 2 for i in range(3)])


def ancestors(tree, node):
    """Node indices on the root -> node path (parents are always created before children)."""
    path = [node]
    while True:
        parent = None
        for candidate in range(len(tree["left"])):
            if tree["left"][candidate] == path[-1] or tree["right"][candidate] == path[-1]:
                parent = candidate
                break
        if parent is None:
            return path[::-1]
        path.append(parent)


def tree_invariants_ok(tree, points):
    """Every point sits in the OBB of every ancestor node, leaves partition the input, frames
    are orthonormal and every node is tight."""
    ok = tree["root"] >= 0
    root_set = sorted(tree["indices"][tree["root"]])
    ok = ok and root_set == list(range(len(points)))
    for node in range(len(tree["left"])):
        for i in tree["indices"][node]:
            for ancestor in ancestors(tree, node):
                ok = ok and tree["obb"][ancestor].contains_point(points[i])
        if tree["left"][node] < 0:
            ok = ok and len(tree["indices"][node]) <= LEAF_SIZE
        else:
            kids = (tree["left"][node], tree["right"][node])
            ok = ok and sorted(tree["indices"][kids[0]] + tree["indices"][kids[1]]) \
                == sorted(tree["indices"][node])
            ok = ok and set(tree["indices"][kids[0]]).isdisjoint(tree["indices"][kids[1]])
        obb = tree["obb"][node]
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                ok = ok and abs(dot(obb.axes[i], obb.axes[j]) - expected) < 1e-9
        cloud = [points[i] for i in tree["indices"][node]]
        for axis_index in range(3):
            best = max(abs(dot(sub(p, obb.center), obb.axes[axis_index])) for p in cloud)
            ok = ok and abs(best - obb.half[axis_index]) < 1e-9
    return ok


def rotated_slab(angle, seed):
    """A thin 8 x 1 x 0.2 slab of points rotated by `angle` about z: the shape where an oriented
    frame (8 x 1 x 0.2 box, volume 1.6) beats the axis-aligned box by a large factor."""
    points, state = [], seed
    for i in range(21):
        for y in (-0.5, 0.0, 0.5):
            for z in (-0.1, 0.1):
                state, r = splitmix64(state)
                x = -4.0 + 8.0 * i / 20
                xx = x * math.cos(angle) - y * math.sin(angle)
                yy = x * math.sin(angle) + y * math.cos(angle)
                points.append([xx, yy, z + (r % 7) * 1e-3])
    return points, state


def main():
    # 1. hand case: a 2 x 1 x 0.5 rectangular box rotated by a fixed 3D rotation keeps volume 1
    #    when treated as an oriented box, while its axis-aligned box is strictly bigger
    axis = [1.0 / math.sqrt(3)] * 3
    orthogonal = scale(cross(axis, [0.0, 1.0, 0.0]), 1.0 / norm(cross(axis, [0.0, 1.0, 0.0])))
    third = cross(axis, orthogonal)
    box = []
    for sx in (-1.0, 1.0):
        for sy in (-0.5, 0.5):
            for sz in (-0.25, 0.25):
                box.append([sx * axis[a] + sy * orthogonal[a] + sz * third[a] for a in range(3)])
    box_tree = build_obb_tree(box)
    box_obb = box_tree["obb"][box_tree["root"]]
    box_aabb = aabb_of(box)
    check(abs(box_obb.volume() - 1.0) < 1e-9,
          "the oriented box of a rotated 2 x 1 x 0.5 box must have volume exactly 1")
    check(box_obb.volume() < box_aabb.volume() - 1e-6,
          "the oriented box must beat the axis-aligned box on rotated input")
    check(tree_invariants_ok(box_tree, box), "box tree invariants (containment, tightness)")

    # 2. the same comparison on a rotated thin slab: the frame must recover the 8 x 1 x 0.2 shape,
    #    keeping the oriented volume near 1.6 while the axis-aligned box explodes
    cloud, seed = rotated_slab(math.pi / 6, 0x5EED)
    cloud_tree = build_obb_tree(cloud)
    check(tree_invariants_ok(cloud_tree, cloud), "slab tree invariants")
    cloud_obb = cloud_tree["obb"][cloud_tree["root"]]
    cloud_aabb = aabb_of(cloud)
    check(cloud_obb.volume() < cloud_aabb.volume() / 2,
          "a rotated slab must gain a large volume factor from the oriented frame")
    check(abs(cloud_obb.volume() - 1.6) < 0.25,
          "the oriented volume of an 8 x 1 x 0.2 slab must stay near 1.6")
    check(0.08 < cloud_obb.half[2] < 0.2, "the third extent must be the slab thickness")

    # 3. axis-aligned data: the oriented frame must find the coordinate axes and match the AABB
    grid = [[x * 1.0, y * 2.0, z * 0.5] for x in range(-4, 5) for y in range(-3, 4) for z in (-1, 1)]
    grid_tree = build_obb_tree(grid)
    grid_obb = grid_tree["obb"][grid_tree["root"]]
    check(tree_invariants_ok(grid_tree, grid), "grid tree invariants")
    check(abs(grid_obb.volume() - aabb_of(grid).volume()) < 1e-9,
          "on axis-aligned data the oriented box must equal the axis-aligned box")
    primary = max(range(3), key=lambda a: grid_obb.half[a])
    check(abs(abs(grid_obb.axes[primary][1]) - 1.0) < 1e-9,
          "the dominant axis of an elongated grid must be the y axis")

    # 4. power iteration numerics: Rayleigh quotient non-decreasing, residual small
    matrix = covariance(cloud, mean_of(cloud))
    _, residual, trace = power_iteration(matrix)
    check(all(trace[i] <= trace[i + 1] + 1e-9 for i in range(len(trace) - 1)),
          "power iteration must not decrease the Rayleigh quotient")
    check(residual < 1e-6 and trace[-1] > 0, "power iteration must converge to an eigenpair")
    check(len(trace) >= 10, "power iteration must really iterate")

    # 5. determinism: the whole tree is a pure function of the input
    again = build_obb_tree(cloud)
    check([(o.center, o.half) for o in again["obb"]] == [(o.center, o.half) for o in cloud_tree["obb"]],
          "rebuilding must reproduce every OBB exactly")
    check(again["indices"] == cloud_tree["indices"], "rebuilding must reproduce the partition")

    # 6. degenerate clouds: empty, single, duplicate, collinear, planar
    check(build_obb_tree([])["root"] == -1, "an empty point set has no tree")
    single = build_obb_tree([[1.0, 2.0, 3.0]])
    check(single["obb"][single["root"]].volume() == 0.0, "a single point has a zero-volume box")
    duplicates = build_obb_tree([[2.0, 2.0, 2.0]] * 9)
    check(duplicates["obb"][duplicates["root"]].contains_point([2.0, 2.0, 2.0]),
          "identical points must still be covered by their (degenerate) box")
    line = [[float(k), 2.0 * k, -0.5 * k] for k in range(-6, 7)]
    line_tree = build_obb_tree(line)
    check(tree_invariants_ok(line_tree, line), "collinear cloud invariants")
    check(line_tree["obb"][line_tree["root"]].half[2] < 1e-9,
          "a collinear cloud must have a zero thickness along the second axis")
    plane = [[float(a), float(b), 0.0] for a in range(-3, 4) for b in range(-3, 4)]
    plane_tree = build_obb_tree(plane)
    check(tree_invariants_ok(plane_tree, plane), "planar cloud invariants")
    check(plane_tree["obb"][plane_tree["root"]].volume() < 1e-9,
          "a planar cloud must have a zero-volume oriented box")

    # 7. the tree really splits: depth and leaf sizes behave, and leaves tile the input
    leaves = [n for n in range(len(cloud_tree["left"])) if cloud_tree["left"][n] < 0]
    check(len(leaves) >= len(cloud) // LEAF_SIZE, "leaves must respect the size threshold")
    check(sum(len(cloud_tree["indices"][n]) for n in leaves) == len(cloud),
          "leaves must partition the input exactly")
    depths = [len(ancestors(cloud_tree, n)) for n in leaves]
    check(max(depths) - min(depths) <= 2, "median splitting must keep the tree balanced")

    print(f"737 nodes={len(cloud_tree['left'])} leaves={len(leaves)} depth={max(depths)}")
    print(f"737 box volume obb={box_obb.volume():.4f} aabb={box_aabb.volume():.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
