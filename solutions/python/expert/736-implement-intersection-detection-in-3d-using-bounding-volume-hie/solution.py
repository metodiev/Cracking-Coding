"""Task 736 - Implement intersection detection in 3D using bounding volume hierarchies.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

A triangle soup is queried for intersecting pairs.  Testing every pair is O(n^2) exact
triangle-triangle tests; a bounding volume hierarchy replaces that scan by a tree of
axis-aligned boxes, so only pairs whose boxes overlap are tested.  The tree is built by median
split: at every node the axis with the widest centroid spread is chosen, the surviving
triangles are ordered by centroid (ties by centroid then index, so the tree is deterministic)
and the list is halved; leaves hold at most two triangles.  A box query walks the tree with an
explicit stack and descends only when the query box overlaps the node box - exact rational
comparisons, so conservative pruning never loses a real intersection.

The exact primitive is segment versus triangle: the signed distances of the two endpoints from
the triangle plane give the segment parameter t, and the hit point is classified against the
three edge planes.  A segment parallel to the plane goes to the coplanar branch, which projects
by dropping the axis of the largest normal component (that projection keeps a non-degenerate
triangle non-degenerate) and clips the projected segment's parameter interval against the three
half-planes of the projected triangle.  Triangle versus triangle is then "some edge of one
pierces the other", which also catches coplanar overlap and containment.  A degenerate
(zero-area) triangle has no plane, so it is tested through its three edge segments with an
exact 3D segment-segment test - a projected 2D test would happily call skew edges crossing.

Scaled down versus production BVHs: a static median-split tree instead of a SAH-built one, one
triangle-triangle test per candidate pair, and exact Fractions throughout (floats would let a
near-coplanar case flip a sign).  Inputs stay small on purpose, so the whole file runs in well
under a second.  Verified: the BVH pair set equals the brute-force all-pairs pair set on random
soups (the independent oracle), every node box contains its children's boxes and every leaf box
contains its triangles, pruning really is exercised, and rebuilds reproduce the same tree.
"""
from fractions import Fraction

LEAF_SIZE = 2
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def v3(x, y, z):
    return (Fraction(x), Fraction(y), Fraction(z))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def orient2(u, v, p):
    """Exact 2D orientation of the triple u,v,p (the plane analogue of cross3)."""
    return (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])


def point_on_segment_3d(p, a, b):
    """Closed 3D segment membership: collinear with [a,b] and inside its box."""
    return (cross3(sub(p, a), sub(b, a)) == (0, 0, 0)
            and all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in range(3)))


def segments_3d_intersect(p0, p1, q0, q1):
    """Closed segments in 3D share a point; solves the two-parameter system exactly and falls
    back to collinearity plus a 1D interval overlap when the directions are parallel."""
    d1, d2 = sub(p1, p0), sub(q1, q0)
    a, e = dot(d1, d1), dot(d2, d2)
    if a == 0 or e == 0:
        if a == 0 and e == 0:
            return p0 == q0
        return point_on_segment_3d(p0, q0, q1) if a == 0 else point_on_segment_3d(q0, p0, p1)
    r = sub(p0, q0)
    b, c, f = dot(d1, d2), dot(d1, r), dot(d2, r)
    denom = a * e - b * b
    if denom != 0:
        s, t = (b * f - c * e) / denom, (a * f - b * c) / denom
        if not (0 <= s <= 1 and 0 <= t <= 1):
            return False
        return all(p0[i] + s * d1[i] == q0[i] + t * d2[i] for i in range(3))
    if cross3(d1, r) != (0, 0, 0):
        return False                        # parallel but not collinear
    axis = max(range(3), key=lambda i: abs(d1[i]))
    return (max(min(p0[axis], p1[axis]), min(q0[axis], q1[axis]))
            <= min(max(p0[axis], p1[axis]), max(q0[axis], q1[axis])))


class Aabb:
    """Axis-aligned bounding box over exact rational coordinates."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    @staticmethod
    def of_points(points):
        return Aabb(tuple(min(p[i] for p in points) for i in range(3)),
                    tuple(max(p[i] for p in points) for i in range(3)))

    def overlaps(self, other):
        return all(self.lo[i] <= other.hi[i] and other.lo[i] <= self.hi[i] for i in range(3))

    def contains(self, other):
        return all(self.lo[i] <= other.lo[i] and other.hi[i] <= self.hi[i] for i in range(3))

    def contains_point(self, p):
        return all(self.lo[i] <= p[i] <= self.hi[i] for i in range(3))


class Counter:
    """Instrumentation: how much work each strategy actually did."""

    def __init__(self):
        self.segment_triangle_tests = 0
        self.nodes_visited = 0
        self.candidate_pairs = 0


def build_bvh(triangles):
    """Median-split BVH; returns (boxes, left, right, leaf_items, root) as flat arrays."""
    boxes, left, right, leaf_items = [], [], [], []

    def recurse(indices):
        node = len(boxes)
        boxes.append(Aabb.of_points([triangles[i][k] for i in indices for k in range(3)]))
        left.append(-1)
        right.append(-1)
        leaf_items.append(None)
        if len(indices) <= LEAF_SIZE:
            leaf_items[node] = tuple(indices)
            return node
        centroids = {i: tuple(sum(triangles[i][k][a] for k in range(3)) / 3 for a in range(3))
                     for i in indices}
        axis, spread = 0, None
        for a in range(3):
            values = [centroids[i][a] for i in indices]
            width = max(values) - min(values)
            if spread is None or width > spread:
                axis, spread = a, width
        ordered = sorted(indices, key=lambda i: (centroids[i][axis], centroids[i], i))
        mid = len(ordered) // 2
        left[node] = recurse(ordered[:mid])
        right[node] = recurse(ordered[mid:])
        return node

    root = recurse(list(range(len(triangles)))) if triangles else -1
    return boxes, left, right, leaf_items, root


def query_box(bvh, box, skip=-1, counter=None):
    """Every leaf triangle index whose box overlaps `box` (touching counts); a conservative
    superset of the truly overlapping triangles, returned sorted and deduplicated."""
    boxes, left, right, leaf_items, root = bvh
    hits = []
    if root < 0:
        return hits
    stack = [root]
    while stack:
        node = stack.pop()
        if counter is not None:
            counter.nodes_visited += 1
        if not boxes[node].overlaps(box):
            continue
        if left[node] < 0:
            hits.extend(i for i in leaf_items[node] if i != skip)
        else:
            stack.append(left[node])
            stack.append(right[node])
    return sorted(hits)


def coplanar_segment_hit(p0, p1, a, b, c, normal):
    """Segment and triangle lie in one plane: project by dropping the axis of the largest
    normal component and clip the projected segment's parameter interval against the three
    half-planes of the projected triangle.  Inputs are non-degenerate, and that projection
    keeps a non-degenerate triangle non-degenerate, so no 2D degenerate case can appear."""
    axis = max(range(3), key=lambda i: abs(normal[i]))
    keep = [i for i in range(3) if i != axis]
    q0, q1 = (p0[keep[0]], p0[keep[1]]), (p1[keep[0]], p1[keep[1]])
    tri = [(a[keep[0]], a[keep[1]]), (b[keep[0]], b[keep[1]]), (c[keep[0]], c[keep[1]])]
    sign = 1 if orient2(tri[0], tri[1], tri[2]) > 0 else -1
    t_low, t_high = Fraction(0), Fraction(1)
    for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        f0 = orient2(u, v, q0) * sign
        step = orient2(u, v, q1) * sign - f0
        if step == 0:
            if f0 < 0:
                return False                # the whole segment is outside this half-plane
        elif step > 0:
            t_low = max(t_low, -f0 / step)
        else:
            t_high = min(t_high, -f0 / step)
        if t_high < t_low:
            return False
    return True


def segment_hits_triangle(p0, p1, tri, counter=None):
    """Exact segment/triangle intersection (Moller-Trumbore, in rational arithmetic)."""
    if counter is not None:
        counter.segment_triangle_tests += 1
    a, b, c = tri
    normal = cross3(sub(b, a), sub(c, a))
    if normal == (0, 0, 0):
        return any(segments_3d_intersect(p0, p1, u, v)
                   for u, v in ((a, b), (b, c), (c, a)))
    direction = sub(p1, p0)
    denom = dot(normal, direction)
    if denom == 0:
        if dot(normal, sub(p0, a)) != 0:
            return False                    # parallel to the plane but off it
        return coplanar_segment_hit(p0, p1, a, b, c, normal)
    t = dot(normal, sub(a, p0)) / denom
    if t < 0 or t > 1:
        return False                        # the plane is crossed outside the segment
    q = tuple(p0[i] + t * direction[i] for i in range(3))
    return all(dot(normal, cross3(sub(r, p), sub(q, p))) >= 0
               for p, r in ((a, b), (b, c), (c, a)))


def triangles_intersect(t1, t2, counter=None):
    """Some edge of one triangle pierces the other - which also detects coplanar overlap and
    containment, because the segment test itself handles the coplanar and degenerate branches."""
    for tri, other in ((t1, t2), (t2, t1)):
        for u, v in ((0, 1), (1, 2), (2, 0)):
            if segment_hits_triangle(tri[u], tri[v], other, counter):
                return True
    return False


def soup_intersections(triangles, bvh=None, counter=None):
    """All intersecting pairs (i, j), i < j.  With `bvh` candidates come from the hierarchy;
    without it every pair is tested, which is the brute-force oracle."""
    pairs = []
    for i, tri in enumerate(triangles):
        if bvh is None:
            candidates = range(i + 1, len(triangles))
        else:
            candidates = [j for j in query_box(bvh, Aabb.of_points(tri), skip=i, counter=counter)
                          if j > i]
        for j in candidates:
            if counter is not None:
                counter.candidate_pairs += 1
            if triangles_intersect(tri, triangles[j], counter):
                pairs.append((i, j))
    return pairs


def bvh_invariants_ok(triangles, bvh):
    """Leaf boxes contain their triangles, parent boxes contain their children's boxes."""
    boxes, left, right, leaf_items, root = bvh
    ok = root >= 0 and all(boxes[root].contains_point(p) for tri in triangles for p in tri)
    for node, box in enumerate(boxes):
        if left[node] < 0:
            ok = ok and all(box.contains_point(p) for i in leaf_items[node] for p in triangles[i])
        else:
            ok = ok and box.contains(boxes[left[node]]) and box.contains(boxes[right[node]])
    return ok


def random_soup(count, span, size, seed):
    """Deterministic soup: triangles of extent < size scattered over a span-wide cube."""
    triangles = []
    for _ in range(count):
        seed, r = splitmix64(seed)
        base = [(r >> (8 * (a + 1))) % span for a in range(3)]
        corners = []
        for _ in range(3):
            corner = []
            for a in range(3):
                seed, r = splitmix64(seed)
                corner.append(base[a] + r % size)
            corners.append(tuple(corner))
        triangles.append(tuple(corners))
    return triangles, seed


def main():
    # 1. hand-checked segment primitives on a triangle in the plane z = 0
    plane = (v3(0, 0, 0), v3(4, 0, 0), v3(0, 4, 0))
    check(segment_hits_triangle(v3(1, 1, -1), v3(1, 1, 1), plane), "piercing segment must hit")
    check(not segment_hits_triangle(v3(5, 5, -1), v3(5, 5, 1), plane), "outside segment must miss")
    check(segment_hits_triangle(v3(1, 1, 0), v3(2, 1, 0), plane), "coplanar inside segment must hit")
    check(not segment_hits_triangle(v3(5, 5, 0), v3(6, 6, 0), plane), "coplanar outside must miss")
    check(not segment_hits_triangle(v3(9, 9, -1), v3(9, 9, 1), plane),
          "parallel to the plane but off it must miss")
    check(not segment_hits_triangle(v3(3, 3, 4), v3(3, 3, 1), plane),
          "a segment stopping short of the plane must miss (t outside [0,1])")
    check(segment_hits_triangle(v3(0, 0, -1), v3(0, 0, 1), plane),
          "a segment through a vertex must count as a hit (closed sets)")

    # 2. triangle-triangle cases: crossing, parallel, coplanar containment, disjoint
    upper = (v3(0, 0, 1), v3(4, 0, 1), v3(0, 4, 1))
    piercing = (v3(2, 2, -1), v3(2, 2, 1), v3(4, 4, 1))
    checker = (v3(1, 1, 0), v3(1, 1, 5), v3(4, 4, 5))
    big = (v3(-2, -2, 0), v3(6, -2, 0), v3(-2, 6, 0))
    far = (v3(10, 10, 0), v3(12, 10, 0), v3(10, 12, 0))
    check(not triangles_intersect(plane, upper, Counter()), "parallel triangles must not intersect")
    check(triangles_intersect(plane, checker, Counter()), "a plane-piercing triangle must hit")
    check(triangles_intersect(plane, big, Counter()), "coplanar containment must be detected")
    check(not triangles_intersect(plane, far, Counter()), "disjoint coplanar triangles must miss")
    check(triangles_intersect(plane, plane, Counter()), "a triangle must intersect itself")

    # 3. degenerate inputs: empty soup, single triangle, zero-area triangle
    empty = build_bvh([])
    check(empty[4] < 0 and soup_intersections([], empty) == [], "an empty soup has no pairs")
    check(soup_intersections([plane], build_bvh([plane])) == [],
          "one triangle cannot intersect itself in the pair listing")
    sliver = (v3(0, 0, 0), v3(4, 4, 0), v3(2, 2, 0))         # zero area, collinear corners
    check(triangles_intersect(sliver, piercing, Counter()),
          "a zero-area triangle must still be reported when an edge is crossed")
    check(not triangles_intersect(sliver, far, Counter()),
          "a zero-area triangle disjoint from the other must not hit")
    skew = (v3(2, -2, 3), v3(2, 6, 3), v3(6, 2, 3))          # misses the sliver's line
    check(not triangles_intersect(sliver, skew, Counter()),
          "skew edges next to a degenerate triangle must not be called an intersection")

    # 4. random soup: the BVH pair set must equal the brute-force all-pairs pair set
    soup, seed = random_soup(60, 40, 3, 0xBEEF)
    bvh = build_bvh(soup)
    check(bvh_invariants_ok(soup, bvh), "BVH boxes must contain their triangles and children")
    fast, slow = Counter(), Counter()
    with_bvh = soup_intersections(soup, bvh, fast)
    brute = soup_intersections(soup, None, slow)
    check(with_bvh == brute, f"BVH pairs {with_bvh} != brute-force pairs {brute}")
    dense, _ = random_soup(30, 6, 6, 0xD00D)
    dense_bvh, dense_brute = build_bvh(dense), soup_intersections(dense, None, Counter())
    check(len(dense_brute) > 0 and soup_intersections(dense, dense_bvh, Counter()) == dense_brute,
          "a crowded soup must really produce intersecting pairs, and the BVH must find them all")

    # 5. pruning really happens: fewer candidate pairs and fewer exact predicate calls
    check(fast.candidate_pairs < slow.candidate_pairs,
          "the BVH must visit fewer candidate pairs than the quadratic baseline")
    check(fast.segment_triangle_tests < slow.segment_triangle_tests,
          "the BVH must spend fewer exact segment/triangle tests")
    check(fast.nodes_visited > 0 and len(bvh[0]) > len(soup) // 2, "the traversal must walk a tree")

    # 6. box queries: conservative superset, touching boxes count, distant boxes return nothing
    touch = Aabb(v3(4, 0, 0), v3(4, 0, 0))
    overlaps = {j for j, tri in enumerate(soup) if Aabb.of_points(tri).overlaps(touch)}
    found = query_box(bvh, touch)
    check(overlaps <= set(found) and found == sorted(set(found)),
          "a point query must return every overlapping triangle, sorted and deduplicated")
    nowhere = Aabb(v3(100, 100, 100), v3(101, 101, 101))
    check(query_box(bvh, nowhere) == [] and query_box(empty, nowhere) == [],
          "a distant query box must return nothing, even on an empty tree")

    # 7. determinism: rebuilding reproduces the tree and the answers
    again = build_bvh(soup)
    check([(b.lo, b.hi) for b in again[0]] == [(b.lo, b.hi) for b in bvh[0]],
          "rebuilding the BVH must reproduce every node box exactly")
    check(soup_intersections(soup, again, Counter()) == with_bvh,
          "rebuilding must reproduce the intersection pairs")

    # 8. segment sweep: the pruned answer must equal the direct scan over all triangles
    sweep = Counter()
    for k in range(10):
        p0, p1 = v3(k * 2, 10, -5), v3(k * 2, 10, 15)
        by_bvh = sorted(j for j in query_box(bvh, Aabb.of_points([p0, p1]), counter=sweep)
                        if segment_hits_triangle(p0, p1, soup[j]))
        direct = sorted(j for j, tri in enumerate(soup) if segment_hits_triangle(p0, p1, tri))
        check(by_bvh == direct, f"segment sweep {k}: BVH {by_bvh} != direct scan {direct}")

    print(f"736 soup_triangles={len(soup)} intersecting_pairs={len(brute)} nodes={len(bvh[0])}")
    print(f"736 exact_tests bvh={fast.segment_triangle_tests} brute={slow.segment_triangle_tests} "
          f"candidates bvh={fast.candidate_pairs} brute={slow.candidate_pairs}")
    print("PASS")


if __name__ == "__main__":
    main()
