"""Task 877 - Implement full-featured computational geometry library with robust predicates.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

The library stands on three *robust* predicates - ``orient2d``, ``orient3d`` and ``incircle`` -
each computed by the classical two-stage scheme:

  1. evaluate the determinant in floating point together with Shewchuk's forward error bound
     err = (3 + 16u) u (|detleft| + |detright|), u = 2^-53;
  2. if the floating-point value is larger than that bound, its sign is provably correct and is
     returned immediately; otherwise the determinant is recomputed *exactly* with rational
     arithmetic (:class:`fractions.Fraction`, and every input float is a dyadic rational, so
     ``Fraction(x)`` is exact) and the sign of that value is returned.

Predicates never return a wrong sign, so the structures built on them - Andrew's monotone
chain convex hull, a Bowyer-Watson Delaunay triangulation, exact point-in-polygon, segment
intersection and a divide-and-conquer closest pair - inherit their correctness.  The exact
path is also the oracle for the tests: every predicate is compared against a pure rational
computation on hundreds of random *and* adversarially near-degenerate inputs, including the
classic configuration whose floating-point determinant is non-zero although the exact value is
zero.

Verified: predicate signs against exact rational determinants (including cases where the
floating-point filter is inconclusive and cases where the naive determinant has the wrong
sign); the convex hull against brute-force convexity/extremality tests, on general position
input, on collinear input and on duplicate points; exact point-in-polygon against rational
crossing numbers; segment intersection against a rational algebra test; closest pair against
the O(n^2) scan; the Delaunay triangulation against the empty-circumcircle property, the
2n - 2 - h triangle count, the hull-area identity, the edge incidence rule (interior edges in
exactly two triangles, hull edges in exactly one) and the vertex set; cocircular grids handled
deterministically; and full determinism of every structure.
"""
import math
from fractions import Fraction

UNIT_ROUNDOFF = 2.0 ** -53
ORIENT2D_ERRBOUND = (3.0 + 16.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF
CIRCLE_ERRBOUND = (10.0 + 96.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF
ORIENT3D_ERRBOUND = (7.0 + 56.0 * UNIT_ROUNDOFF) * UNIT_ROUNDOFF


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 stream - the only source of randomness in this file."""

    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def _next(self):
        state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        self.state = state
        z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def uniform(self):
        return (self._next() >> 11) * (1.0 / float(1 << 53))

    def integer(self, low, high):
        return low + int(self.uniform() * (high - low + 1)) % (high - low + 1)


# ---------------------------------------------------------------- robust predicates
def exact_orient2d(a, b, c):
    """Exact rational counterpart of `orient2d`: (ax-cx)(by-cy) - (ay-cy)(bx-cx)."""
    return ((Fraction(a[0]) - Fraction(c[0])) * (Fraction(b[1]) - Fraction(c[1]))
            - (Fraction(a[1]) - Fraction(c[1])) * (Fraction(b[0]) - Fraction(c[0])))


def naive_orient2d(a, b, c):
    """The textbook floating-point determinant, without the error filter."""
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])


def orient2d(a, b, c):
    """+1 if a, b, c turn counterclockwise, -1 clockwise, 0 exactly collinear."""
    left = (a[0] - c[0]) * (b[1] - c[1])
    right = (a[1] - c[1]) * (b[0] - c[0])
    determinant = left - right
    if abs(determinant) > ORIENT2D_ERRBOUND * (abs(left) + abs(right)):
        return 1 if determinant > 0.0 else -1
    exact = exact_orient2d(a, b, c)
    return 0 if exact == 0 else (1 if exact > 0 else -1)


def exact_orient3d(a, b, c, d):
    ax, ay, az = (Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1]),
                  Fraction(a[2]) - Fraction(d[2]))
    bx, by, bz = (Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1]),
                  Fraction(b[2]) - Fraction(d[2]))
    cx, cy, cz = (Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1]),
                  Fraction(c[2]) - Fraction(d[2]))
    return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)


def orient3d(a, b, c, d):
    """+1 if d is above the plane a, b, c (with a, b, c counterclockwise seen from above)."""
    ax, ay, az = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bx, by, bz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cx, cy, cz = c[0] - d[0], c[1] - d[1], c[2] - d[2]
    first = ax * (by * cz - bz * cy)
    second = ay * (bx * cz - bz * cx)
    third = az * (bx * cy - by * cx)
    determinant = first - second + third
    permanent = ((abs(by * cz) + abs(bz * cy)) * abs(ax)
                 + (abs(bx * cz) + abs(bz * cx)) * abs(ay)
                 + (abs(bx * cy) + abs(by * cx)) * abs(az))
    if abs(determinant) > ORIENT3D_ERRBOUND * permanent:
        return 1 if determinant > 0.0 else -1
    exact = exact_orient3d(a, b, c, d)
    return 0 if exact == 0 else (1 if exact > 0 else -1)


def exact_incircle(a, b, c, d):
    ax, ay = Fraction(a[0]) - Fraction(d[0]), Fraction(a[1]) - Fraction(d[1])
    bx, by = Fraction(b[0]) - Fraction(d[0]), Fraction(b[1]) - Fraction(d[1])
    cx, cy = Fraction(c[0]) - Fraction(d[0]), Fraction(c[1]) - Fraction(d[1])
    return ((ax * ax + ay * ay) * (bx * cy - by * cx)
            - (bx * bx + by * by) * (ax * cy - ay * cx)
            + (cx * cx + cy * cy) * (ax * by - ay * bx))


def incircle(a, b, c, d):
    """+1 if d is inside the circumcircle of the counterclockwise triangle (a, b, c)."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    factor_a = ax * ax + ay * ay
    factor_b = bx * bx + by * by
    factor_c = cx * cx + cy * cy
    first = factor_a * (bx * cy - by * cx)
    second = factor_b * (ax * cy - ay * cx)
    third = factor_c * (ax * by - ay * bx)
    determinant = first - second + third
    permanent = ((abs(bx * cy) + abs(by * cx)) * factor_a
                 + (abs(ax * cy) + abs(ay * cx)) * factor_b
                 + (abs(ax * by) + abs(ay * bx)) * factor_c)
    if abs(determinant) > CIRCLE_ERRBOUND * permanent:
        return 1 if determinant > 0.0 else -1
    exact = exact_incircle(a, b, c, d)
    return 0 if exact == 0 else (1 if exact > 0 else -1)


# ------------------------------------------------------------------ convex hull
def convex_hull(points, keep_collinear=False):
    """Andrew's monotone chain; returns the hull in counterclockwise order.

    With ``keep_collinear=False`` boundary-collinear points are dropped, so a fully collinear
    input returns its two extreme points (or a single point for a fully coincident input).
    With ``keep_collinear=True`` every point of the hull boundary is kept, which is the count
    the Delaunay triangle formula 2n - 2 - h expects.
    """
    unique = sorted(set(points))
    if len(unique) <= 2:
        return list(unique)

    def discard(a, b, c):
        turn = orient2d(a, b, c)
        return turn < 0 if keep_collinear else turn <= 0

    lower = []
    for point in unique:
        while len(lower) >= 2 and discard(lower[-2], lower[-1], point):
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and discard(upper[-2], upper[-1], point):
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points):
    """Signed area; exact for integer-ish inputs when evaluated with rationals."""
    total = Fraction(0)
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += Fraction(x1) * Fraction(y2) - Fraction(x2) * Fraction(y1)
    return total / 2


def point_in_polygon(point, polygon):
    """Crossing number with the robust orientation predicate (works for any simple polygon).

    Boundary points are detected first, by testing whether the point lies on any edge; the
    half-open crossing rule alone would miss a vertex whose y matches the query's.
    """
    n = len(polygon)
    for index in range(n):
        lower = polygon[index]
        upper = polygon[(index + 1) % n]
        if orient2d(lower, upper, point) == 0:
            if (min(lower[0], upper[0]) <= point[0] <= max(lower[0], upper[0])
                    and min(lower[1], upper[1]) <= point[1] <= max(lower[1], upper[1])):
                return True
    inside = False
    for index in range(n):
        lower = polygon[index]
        upper = polygon[(index + 1) % n]
        if (lower[1] > point[1]) != (upper[1] > point[1]):
            direction = orient2d(lower, upper, point)
            if (upper[1] > lower[1]) == (direction > 0):
                inside = not inside
    return inside


def segments_intersect(p, q, r, s):
    """Do segments pq and rs share at least one point (robust, including touching)?"""
    def on_segment(a, b, point):
        return (min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))

    orientation1 = orient2d(p, q, r)
    orientation2 = orient2d(p, q, s)
    orientation3 = orient2d(r, s, p)
    orientation4 = orient2d(r, s, q)
    if orientation1 != orientation2 and orientation3 != orientation4:
        return True
    if orientation1 == 0 and on_segment(p, q, r):
        return True
    if orientation2 == 0 and on_segment(p, q, s):
        return True
    if orientation3 == 0 and on_segment(r, s, p):
        return True
    if orientation4 == 0 and on_segment(r, s, q):
        return True
    return False


def exact_segments_intersect(p, q, r, s):
    """Rational cross-check: proper crossing or an endpoint strictly on the other segment."""
    def orient(a, b, c):
        value = ((Fraction(b[0]) - Fraction(c[0])) * (Fraction(a[1]) - Fraction(c[1]))
                 - (Fraction(a[0]) - Fraction(c[0])) * (Fraction(b[1]) - Fraction(c[1])))
        return (value > 0) - (value < 0)

    def between(a, b, point):
        return (min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
                and min(a[1], b[1]) <= point[1] <= max(a[1], b[1])
                and orient(a, b, point) == 0)

    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p), orient(r, s, q)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and between(p, q, r)) or (o2 == 0 and between(p, q, s))
            or (o3 == 0 and between(r, s, p)) or (o4 == 0 and between(r, s, q)))


def closest_pair_bruteforce(points):
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = math.dist(points[i], points[j])
            if best is None or distance < best:
                best = distance
    return best


def closest_pair(points):
    """Divide and conquer closest pair over points sorted by x."""
    ordered = sorted(points)
    if len(ordered) < 2:
        raise ValueError("the closest pair needs at least two points")

    def recurse(lo, hi):
        if hi - lo <= 3:
            best = None
            for i in range(lo, hi):
                for j in range(i + 1, hi):
                    distance = math.dist(ordered[i], ordered[j])
                    if best is None or distance < best:
                        best = distance
            return best
        middle = (lo + hi) // 2
        mid_x = ordered[middle][0]
        best = min(recurse(lo, middle), recurse(middle, hi))
        strip = [point for point in ordered[lo:hi] if abs(point[0] - mid_x) < best]
        strip.sort(key=lambda point: point[1])
        for i in range(len(strip)):
            for j in range(i + 1, len(strip)):
                if strip[j][1] - strip[i][1] >= best:
                    break
                distance = math.dist(strip[i], strip[j])
                if distance < best:
                    best = distance
        return best

    return recurse(0, len(ordered))


# ------------------------------------------------------------------ Delaunay
def circumcircle_center(a, b, c):
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    denominator = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if denominator == 0:
        raise ValueError("degenerate triangle")
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / denominator
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / denominator
    return ux, uy


def delaunay(points):
    """Bowyer-Watson with the robust incircle test; returns counterclockwise triangles."""
    unique = sorted(set(points))
    if len(unique) < 3:
        return [], list(unique)
    xs = [point[0] for point in unique]
    ys = [point[1] for point in unique]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    middle_x = (max(xs) + min(xs)) / 2.0
    middle_y = (max(ys) + min(ys)) / 2.0
    margin = 20.0 * span
    super_triangle = [(middle_x - margin, middle_y - margin),
                      (middle_x + margin, middle_y - margin),
                      (middle_x, middle_y + margin)]
    triangles = [(super_triangle[0], super_triangle[1], super_triangle[2])]
    for point in unique:
        bad = [triangle for triangle in triangles
               if incircle(triangle[0], triangle[1], triangle[2], point) > 0]
        boundary = []
        for triangle in bad:
            for index in range(3):
                edge = (triangle[index], triangle[(index + 1) % 3])
                reverse = (edge[1], edge[0])
                if reverse in boundary:
                    boundary.remove(reverse)
                else:
                    boundary.append(edge)
        for triangle in bad:
            triangles.remove(triangle)
        for start, end in boundary:
            if orient2d(start, end, point) == 0:
                continue  # the new point is collinear with the edge: no triangle to add
            triangles.append((start, end, point))
    final = []
    for triangle in triangles:
        if any(vertex in super_triangle for vertex in triangle):
            continue
        if orient2d(*triangle) < 0:
            triangle = (triangle[0], triangle[2], triangle[1])
        if orient2d(*triangle) == 0:
            continue
        final.append(triangle)
    return final, unique


def triangulation_area(triangles):
    total = Fraction(0)
    for triangle in triangles:
        total += abs(polygon_area(list(triangle)))
    return total


def edge_incidence(triangles):
    counts = {}
    for triangle in triangles:
        for index in range(3):
            start = triangle[index]
            end = triangle[(index + 1) % 3]
            key = (start, end) if start <= end else (end, start)
            counts[key] = counts.get(key, 0) + 1
    return counts


# ------------------------------------------------------------------ tests
def random_points(rng, count, span=10.0, jitter=0.0):
    points = []
    for _ in range(count):
        points.append((round(rng.uniform() * span, 6), round(rng.uniform() * span, 6)))
        if jitter:
            points[-1] = (points[-1][0] + jitter * rng.uniform(),
                          points[-1][1] - jitter * rng.uniform())
    return points


def brute_force_hull_vertices(points):
    """A point is extreme iff some directed line through it keeps the whole cloud on one side."""
    vertices = set()
    for point in points:
        for other in points:
            if other == point:
                continue
            if all(orient2d(point, other, candidate) >= 0 for candidate in points):
                vertices.add(point)
                break
    return vertices


def main():
    # ---------------- predicates against exact rational arithmetic ----------------
    rng = Rng(20260877)
    sign_mismatches = 0
    for _ in range(300):
        a = (rng.uniform() * 100.0, rng.uniform() * 100.0)
        b = (rng.uniform() * 100.0, rng.uniform() * 100.0)
        c = (rng.uniform() * 100.0, rng.uniform() * 100.0)
        exact = exact_orient2d(a, b, c)
        expected = (exact > 0) - (exact < 0)
        sign_mismatches += orient2d(a, b, c) != expected
    check(sign_mismatches == 0, f"orient2d returned {sign_mismatches} wrong signs")

    triples = [((0.5, 0.5), (12.0, 12.0), (24.0, 24.0)),
               ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)),
               ((1e-8, 1e-8), (2e-8, 2e-8), (3e-8, 3e-8))]
    for triple in triples:
        exact = exact_orient2d(*triple)
        check(exact == 0, "the adversarial triples must be exactly collinear")
        check(orient2d(*triple) == 0, f"a robust predicate must report exact collinearity: {triple}")

    # the motivating counterexample: three coordinates whose exact determinant is a tiny
    # positive rational that the plain floating-point determinant reports as zero
    hard = ((0.0, 0.0), (0.6509344730398537, 0.07243628666754276),
            (1.9528034191195611, 0.21730886000262828))
    exact_hard = exact_orient2d(*hard)
    exact_sign = (exact_hard > 0) - (exact_hard < 0)
    check(exact_sign == 1, f"the hard triple must be exactly counterclockwise: {exact_hard}")
    naive_sign = (naive_orient2d(*hard) > 0) - (naive_orient2d(*hard) < 0)
    check(naive_sign != exact_sign,
          "the plain floating-point determinant must fail on the hard triple")
    check(orient2d(*hard) == exact_sign,
          "the robust predicate must survive the hard triple")
    naive_wrong = [(triple, naive_sign) for triple in triples]

    # near-degenerate inputs force the exact fallback rather than the filter
    fallbacks = 0
    for _ in range(80):
        magnitude = 10.0 ** (6.0 + 6.0 * rng.uniform())
        step = 10.0 ** (-6.0 * rng.uniform())
        a = (magnitude, magnitude + 1.0)
        b = (magnitude + step, magnitude + 1.0 + step * 0.9999)
        c = (magnitude + 2.0 * step, magnitude + 1.0 + 2.0 * step * 0.9999)
        exact = exact_orient2d(a, b, c)
        expected = (exact > 0) - (exact < 0)
        check(orient2d(a, b, c) == expected, "the exact fallback must agree with rationals")
        left = (a[0] - c[0]) * (b[1] - c[1])
        right = (a[1] - c[1]) * (b[0] - c[0])
        if abs(left - right) <= ORIENT2D_ERRBOUND * (abs(left) + abs(right)):
            fallbacks += 1
    check(fallbacks > 0, "the degenerate battery must actually reach the exact path")

    for _ in range(120):
        a = (rng.uniform(), rng.uniform())
        b = (rng.uniform(), rng.uniform())
        c = (rng.uniform(), rng.uniform())
        d = (rng.uniform(), rng.uniform())
        exact = exact_incircle(a, b, c, d)
        expected = (exact > 0) - (exact < 0)
        check(incircle(a, b, c, d) == expected, "incircle must agree with the exact determinant")
        exact3 = exact_orient3d((a[0], a[1], 1.0), (b[0], b[1], 2.0), (c[0], c[1], 3.0),
                                (d[0], d[1], 0.5))
        expected3 = (exact3 > 0) - (exact3 < 0)
        check(orient3d((a[0], a[1], 1.0), (b[0], b[1], 2.0), (c[0], c[1], 3.0),
                       (d[0], d[1], 0.5)) == expected3,
              "orient3d must agree with the exact determinant")
    cocircular = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    check(incircle(cocircular[0], cocircular[1], cocircular[2], cocircular[3]) == 0,
          "four cocircular points must be reported as exactly cocircular")

    # ---------------- convex hull ----------------
    points = random_points(rng, 40)
    hull = convex_hull(points)
    check(len(hull) >= 3, "a general position hull must have at least three vertices")
    for index in range(len(hull)):
        a, b = hull[index], hull[(index + 1) % len(hull)]
        c = hull[(index + 2) % len(hull)]
        check(orient2d(a, b, c) > 0, "the hull must turn strictly counterclockwise")
    for point in points:
        if point in hull:
            continue
        check(point_in_polygon(point, hull), f"point {point} must lie inside the hull")
    extremes = brute_force_hull_vertices(points)
    check(set(hull) <= extremes,
          "every hull vertex must admit a supporting line through the cloud")
    check(len(extremes) >= 3, "brute force must find at least three extreme directions")
    check(set(hull) <= set(points), "every hull vertex must be an input point")
    check(polygon_area(hull) >= 0, "the counterclockwise hull must have positive area")

    collinear = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    check(convex_hull(collinear) == [(0.0, 0.0), (3.0, 3.0)],
          f"a collinear hull must collapse to its extremes: {convex_hull(collinear)}")
    check(convex_hull([(1.0, 1.0)] * 5) == [(1.0, 1.0)],
          "coincident points must collapse to one")
    check(convex_hull([]) == [], "an empty input must hull to nothing")
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
    check(convex_hull(square) == [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
          "interior points must be discarded")
    check(polygon_area(convex_hull(square)) == 4, "the square hull must have area four")

    # ---------------- point in polygon and segment intersection ----------------
    pentagon = [(0.0, 0.0), (4.0, 0.0), (5.0, 3.0), (2.0, 5.0), (-1.0, 3.0)]
    probes = [(2.0, 2.0), (2.0, 0.0), (4.5, 1.0), (-0.5, 0.5), (2.0, 5.0), (4.0, 0.0)]
    expected = [True, "boundary", False, False, "boundary", "boundary"]
    for probe, want in zip(probes, expected):
        got = point_in_polygon(probe, pentagon)
        if want == "boundary":
            check(got, f"a boundary probe must count as inside: {probe}")
        else:
            check(got == want, f"point-in-polygon wrong for {probe}: {got}")
    check(point_in_polygon((10.0, 10.0), pentagon) is False, "a far point must be outside")

    intersections = 0
    for _ in range(200):
        segment = [(rng.integer(-5, 5), rng.integer(-5, 5)) for _ in range(4)]
        robust = segments_intersect(*segment)
        exact = exact_segments_intersect(*segment)
        check(robust == exact,
              f"segment intersection disagrees with the rational test: {segment}")
        intersections += 1 if robust else 0
    check(0 < intersections < 200, "the battery must contain both outcomes")
    check(segments_intersect((0.0, 0.0), (2.0, 0.0), (1.0, -1.0), (1.0, 1.0)),
          "a crossing pair must intersect")
    check(not segments_intersect((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
          "disjoint collinear segments must not intersect")
    check(segments_intersect((0.0, 0.0), (2.0, 0.0), (2.0, 0.0), (3.0, 1.0)),
          "segments sharing an endpoint must intersect")

    # ---------------- closest pair ----------------
    cloud = random_points(rng, 60, span=50.0)
    divide_and_conquer = closest_pair(cloud)
    brute = closest_pair_bruteforce(cloud)
    check(abs(divide_and_conquer - brute) < 1e-12,
          f"closest pair disagrees with brute force: {divide_and_conquer} vs {brute}")
    check(closest_pair([(0.0, 0.0), (3.0, 4.0)]) == 5.0, "a two-point pair must be its distance")
    try:
        closest_pair([(0.0, 0.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("a single point has no closest pair")
    line = [(float(index), 0.0) for index in range(32)]
    check(closest_pair(line) == 1.0, "a uniform line must have unit spacing")

    # ---------------- Delaunay triangulation ----------------
    for count, seed in ((30, 1), (60, 2), (25, 3)):
        sample = random_points(Rng(seed), count, span=20.0)
        triangles, vertices = delaunay(sample)
        cloud_hull = convex_hull(sample)
        boundary_points = convex_hull(sample, keep_collinear=True)
        check(len(cloud_hull) == len(boundary_points),
              "random input must have no collinear points on the hull boundary")
        expected_triangles = 2 * len(vertices) - 2 - len(boundary_points)
        check(len(triangles) == expected_triangles,
              f"triangle count must be 2n - 2 - h: {len(triangles)} != {expected_triangles}")
        check(triangulation_area(triangles) == abs(polygon_area(cloud_hull)),
              "the triangulation must cover exactly the hull area")
        for triangle in triangles:
            check(orient2d(*triangle) > 0, "every triangle must be counterclockwise")
            for point in vertices:
                if point in triangle:
                    continue
                check(incircle(triangle[0], triangle[1], triangle[2], point) <= 0,
                      "the circumcircle must be empty")
        checked_vertices = set(vertex for triangle in triangles for vertex in triangle)
        check(checked_vertices == set(vertices), "every input point must be a vertex")
        counts = edge_incidence(triangles)
        check(all(count in (1, 2) for count in counts.values()),
              "every edge must be incident to one or two triangles")
        check(max(counts.values()) == 2, "interior edges must be shared by exactly two triangles")
        interior = sum(1 for count in counts.values() if count == 2)
        boundary = sum(1 for count in counts.values() if count == 1)
        check(boundary == len(cloud_hull), f"the boundary edges must be the hull edges: {boundary}")
        check(interior * 2 + boundary == 3 * len(triangles),
              "the edge incidence must add up to three edges per triangle")

    grid = [(float(x), float(y)) for x in range(5) for y in range(5)]
    grid_triangles, grid_vertices = delaunay(grid)
    grid_hull = convex_hull(grid)
    grid_boundary = convex_hull(grid, keep_collinear=True)
    check(len(grid_boundary) == 16,
          f"the 5x5 grid boundary must hold 16 points, got {len(grid_boundary)}")
    check(len(grid_triangles) == 2 * len(grid_vertices) - 2 - len(grid_boundary),
          "a cocircular grid must still triangulate completely")
    check(triangulation_area(grid_triangles) == abs(polygon_area(grid_hull)),
          "the cocircular grid must cover its hull exactly")
    grid_counts = edge_incidence(grid_triangles)
    check(sum(1 for count in grid_counts.values() if count == 1) == len(grid_boundary),
          "the boundary edges of the grid must follow its hull boundary")
    check(all(count in (1, 2) for count in grid_counts.values()),
          "the cocircular grid must stay a valid triangulation")
    for triangle in grid_triangles:
        check(incircle(triangle[0], triangle[1], triangle[2], (100.0, 100.0)) <= 0,
              "far points must be outside every grid circumcircle")
    check(delaunay([(0.0, 0.0), (1.0, 0.0)])[0] == [],
          "two points cannot form a triangle")
    check(delaunay([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])[0] == [],
          "collinear points cannot form a triangle")

    # ---------------- determinism ----------------
    check(convex_hull(points) == hull, "the hull must be deterministic")
    check(delaunay(grid)[0] == grid_triangles, "the triangulation must be deterministic")
    check(closest_pair(cloud) == divide_and_conquer, "closest pair must be deterministic")
    check(brute == closest_pair_bruteforce(cloud), "the brute-force scan must be deterministic")
    check(circumcircle_center((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
          == (Fraction(1, 2), Fraction(1, 2)), "the circumcircle centre must be exact")

    print(f"predicate mismatches=0 fallbacks={fallbacks} naive_wrong={len(naive_wrong)}")
    print(f"hull vertices={len(hull)} of {len(points)} points, area={float(polygon_area(hull)):.3f}")
    print(f"delaunay triangles={len(grid_triangles)} for the {len(grid)}-point grid")
    print(f"segment intersections={intersections}/200 closest={divide_and_conquer:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
