"""Task 742 - Implement polygon triangulation (ear clipping) and improvement.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Ear clipping triangulates a simple polygon greedily: a vertex is an *ear* when the triangle of
its two neighbours and itself is convex and contains no other polygon vertex (on its boundary
included), and repeatedly cutting ears off leaves the polygon with three vertices, i.e. n - 2
triangles in total.  Which ear to cut is a free choice, and that choice is what the improvement
part exploits: cutting the ear with the best minimum angle first, and afterwards flipping
internal diagonals whenever the flip raises the smallest angle involved, gives a substantially
better shaped mesh than blindly cutting the first ear found.

All the combinatorial decisions use exact integer orientations: a vertex is convex when the
orientation is strictly positive (the polygon is first normalised to counter-clockwise), and
"inside the ear" is a sign test of the three edge orientations.  Areas are exact rationals, so
the area check is an equality, not a tolerance.  Angles are only used as a quality score, which
is why they are floats and only ever compared with each other, never used in a predicate.

Verified, and this is the interesting part: the n - 2 triangles have positive area, their
vertices come from the polygon, their centroids lie inside the polygon and no triangle edge
properly crosses a polygon edge; the areas sum exactly to the shoelace area of the polygon,
which together with "every triangle is inside" proves the triangles are pairwise
interior-disjoint; the best-ear triangulation beats the first-ear one on the minimum angle; the
flip pass raises the minimum angle further, keeps the triangle count and the total area, and
ends in a state where no single flip improves it; a non-simple polygon is rejected up front.
"""
import math
from fractions import Fraction

EPS = 1e-12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def orient(a, b, c):
    """Exact orientation of the integer triple a,b,c: > 0 counter-clockwise."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def cross_signs(a, b, c, p):
    return orient(a, b, p), orient(b, c, p), orient(c, a, p)


def closed_triangle_contains(a, b, c, p):
    """p inside or on the closed triangle a,b,c (a,b,c counter-clockwise)."""
    d1, d2, d3 = cross_signs(a, b, c, p)
    return d1 >= 0 and d2 >= 0 and d3 >= 0


def proper_cross(a, b, p, q):
    """True when the interiors of the two segments meet."""
    return orient(a, b, p) * orient(a, b, q) < 0 and orient(p, q, a) * orient(p, q, b) < 0


def point_inside(polygon, p):
    """Even-odd ray casting with exact rational comparisons; boundary counts as inside."""
    px, py = Fraction(p[0]), Fraction(p[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if orient(a, b, p) == 0 and (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                                     and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])):
            return True
        if (a[1] > py) != (b[1] > py):
            x_cross = Fraction(a[0]) + (Fraction(b[0]) - Fraction(a[0])) * (py - a[1]) / (b[1] - a[1])
            if x_cross > px:
                inside = not inside
    return inside


def signed_area(polygon):
    """Twice the signed shoelace area as an exact rational."""
    total = Fraction(0)
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        total += Fraction(a[0]) * b[1] - Fraction(b[0]) * a[1]
    return total / 2


def triangle_area(a, b, c):
    return abs(Fraction(orient(a, b, c))) / 2


def min_angle(a, b, c):
    """Smallest interior angle of the triangle, in degrees (quality score only)."""
    angles = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        v1 = (q[0] - p[0], q[1] - p[1])
        v2 = (r[0] - p[0], r[1] - p[1])
        denominator = math.hypot(*v1) * math.hypot(*v2)
        cosine = 0.0 if denominator == 0 else (v1[0] * v2[0] + v1[1] * v2[1]) / denominator
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return min(angles)


def is_simple(polygon):
    """No two non-adjacent edges may cross properly, and adjacent edges may not overlap."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if a == b:
            return False
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            p, q = polygon[j], polygon[(j + 1) % n]
            if proper_cross(a, b, p, q):
                return False
    return True


def is_ear(polygon, cycle, position):
    """The vertex at `position` of the index cycle is an ear of `polygon` when its triangle is
    convex (counter-clockwise traversal) and free of every other polygon vertex."""
    n = len(cycle)
    previous, current, following = cycle[position - 1], cycle[position], cycle[(position + 1) % n]
    a, b, c = polygon[previous], polygon[current], polygon[following]
    if orient(a, b, c) <= 0:
        return False
    for k in range(n):
        index = cycle[k]
        if index in (previous, current, following):
            continue
        if closed_triangle_contains(a, b, c, polygon[index]):
            return False
    return True


def triangulate(polygon, best_ear=True):
    """Ear clipping; `best_ear` selects the ear with the largest minimum angle each round.
    Returns triangles as triples of vertex *indices* into `polygon`."""
    if len(polygon) < 3 or not is_simple(polygon):
        raise ValueError("ear clipping needs a simple polygon with at least three vertices")
    cycle = list(range(len(polygon)))
    if signed_area(polygon) < 0:
        cycle.reverse()                         # normalise the traversal to counter-clockwise
    triangles = []
    while len(cycle) > 3:
        choice = None
        for position in range(len(cycle)):
            if not is_ear(polygon, cycle, position):
                continue
            previous, current, following = (cycle[position - 1], cycle[position],
                                            cycle[(position + 1) % len(cycle)])
            score = min_angle(polygon[previous], polygon[current], polygon[following]) \
                if best_ear else 0.0
            if choice is None or score > choice[0]:
                choice = (score, position)
        if choice is None:
            raise ValueError("no ear found - the polygon is not simple")
        position = choice[1]
        triangles.append(tuple(sorted((cycle[position - 1], cycle[position],
                                       cycle[(position + 1) % len(cycle)]))))
        del cycle[position]
    triangles.append(tuple(sorted(cycle)))
    return triangles


def edge_map(triangles):
    """Undirected edge -> the triangles using it."""
    mapping = {}
    for index, tri in enumerate(triangles):
        for k in range(3):
            edge = tuple(sorted((tri[k], tri[(k + 1) % 3])))
            mapping.setdefault(edge, []).append(index)
    return mapping


def best_flip(triangles, points):
    """The single flip that raises the local minimum angle the most, or None."""
    best = None
    for edge, owners in edge_map(triangles).items():
        if len(owners) != 2:
            continue
        u, v = edge
        t1, t2 = triangles[owners[0]], triangles[owners[1]]
        p = next(x for x in t1 if x not in (u, v))
        q = next(x for x in t2 if x not in (u, v))
        if p == q or not proper_cross(points[p], points[q], points[u], points[v]):
            continue                            # not a convex quadrilateral
        current = min(min_angle(*(points[x] for x in t1)), min_angle(*(points[x] for x in t2)))
        left = tuple(sorted((p, u, q)))
        right = tuple(sorted((p, q, v)))
        candidate = min(min_angle(*(points[x] for x in left)),
                        min_angle(*(points[x] for x in right)))
        if candidate > current + 1e-9 and (best is None or candidate > best[0]):
            best = (candidate, owners[0], owners[1], left, right)
    return best


def improve(triangles, points):
    """Flip internal diagonals while the local minimum angle improves; returns the mesh and the
    number of flips, ending in a state where no single flip helps."""
    mesh = [tuple(sorted(t)) for t in triangles]
    flips = 0
    while True:
        best = best_flip(mesh, points)
        if best is None:
            return mesh, flips
        _, first, second, left, right = best
        mesh[first] = tuple(sorted(left))
        mesh[second] = tuple(sorted(right))
        flips += 1


def mesh_min_angle(triangles, points):
    return min(min_angle(*(points[x] for x in tri)) for tri in triangles)


def mesh_area(triangles, points):
    return sum(triangle_area(*(points[x] for x in tri)) for tri in triangles)


def mesh_valid(triangles, polygon):
    """Positivity, provenance, inside-ness and non-crossing of every triangle of the mesh."""
    for tri in triangles:
        a, b, c = (polygon[x] for x in tri)
        if triangle_area(a, b, c) <= 0:
            return False
        centroid = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
        if not point_inside(polygon, centroid):
            return False
        n = len(polygon)
        for k in range(3):
            p, q = polygon[tri[k]], polygon[tri[(k + 1) % 3]]
            for i in range(n):
                if proper_cross(p, q, polygon[i], polygon[(i + 1) % n]):
                    return False
    return True


def L_shape():
    return [(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)]


def star_shape():
    """A four-pointed star: strongly non-convex, so ear choice matters a lot."""
    return [(0, 0), (3, -4), (6, 0), (10, 3), (6, 6), (3, 10), (0, 6), (-4, 3)]


def zigzag():
    return [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (6, 4), (6, 6), (0, 6)]


def main():
    # 1. hand case: a convex hexagon triangulates into n - 2 triangles of the same total area
    hexagon = [(0, 0), (4, 0), (6, 2), (4, 4), (0, 4), (-2, 2)]
    mesh = triangulate(hexagon)
    check(len(mesh) == len(hexagon) - 2, "ear clipping must produce exactly n - 2 triangles")
    check(mesh_area(mesh, hexagon) == abs(signed_area(hexagon)),
          "the triangle areas must sum exactly to the polygon area")
    check(mesh_valid(mesh, hexagon), "every triangle must be positive, inside and non-crossing")
    check(all(x in range(len(hexagon)) for tri in mesh for x in tri),
          "every triangle vertex must be a polygon vertex")

    # 2. non-convex polygons: same invariants, plus the exact area identity
    for polygon, label in ((L_shape(), "L-shape"), (star_shape(), "star"), (zigzag(), "zigzag")):
        triangles = triangulate(polygon)
        check(len(triangles) == len(polygon) - 2, f"{label}: n - 2 triangles expected")
        check(mesh_area(triangles, polygon) == abs(signed_area(polygon)),
              f"{label}: areas must sum to the polygon area")
        check(mesh_valid(triangles, polygon), f"{label}: the mesh must be geometrically valid")
        check(len(set(triangles)) == len(triangles), f"{label}: no triangle may be repeated")

    # 3. the ear choice matters: the best-ear rule beats the first-ear rule on the minimum angle
    star = star_shape()
    first_mesh = triangulate(star, best_ear=False)
    best_mesh = triangulate(star, best_ear=True)
    check(len(first_mesh) == len(best_mesh) == len(star) - 2, "both rules must give n - 2 triangles")
    check(mesh_area(best_mesh, star) == mesh_area(first_mesh, star) == abs(signed_area(star)),
          "both rules must preserve the total area exactly")
    check(mesh_min_angle(best_mesh, star) >= mesh_min_angle(first_mesh, star) - 1e-9,
          "choosing the best ear first must not make the minimum angle worse")
    check(mesh_min_angle(best_mesh, star) > mesh_min_angle(first_mesh, star) + 1e-9,
          "on this star the best-ear rule must strictly improve the minimum angle")

    # 4. the flip pass improves further and ends locally optimal
    start_angle = mesh_min_angle(first_mesh, star)
    improved, flips = improve(first_mesh, star)
    check(flips > 0, "the flip pass must really flip something on this star")
    check(len(improved) == len(first_mesh), "flipping must not change the triangle count")
    check(mesh_area(improved, star) == abs(signed_area(star)), "flipping must preserve the area")
    check(mesh_valid(improved, star), "the improved mesh must still be geometrically valid")
    check(mesh_min_angle(improved, star) > start_angle,
          "flipping must raise the minimum angle of the mesh above the starting value")
    check(best_flip(improved, star) is None,
          "the improved mesh must be locally optimal: no single flip improves it")

    # 5. the improved mesh also beats the greedy best-ear mesh
    greedy_angle = mesh_min_angle(best_mesh, star)
    check(mesh_min_angle(improved, star) >= greedy_angle - 1e-9,
          "the improved mesh must be at least as good as the best-ear mesh")
    improved_l, flips_l = improve(best_mesh, star)
    check(mesh_min_angle(improved_l, star) >= greedy_angle - 1e-9
          and mesh_valid(improved_l, star), "improving a best-ear mesh must stay valid")

    # 6. rejection cases: self-intersecting and degenerate input
    bowtie = [(0, 0), (4, 4), (4, 0), (0, 4)]
    try:
        triangulate(bowtie)
        raise AssertionError("a self-intersecting polygon must be rejected")
    except ValueError:
        pass
    for bad in ([(0, 0), (1, 1)], [(0, 0), (0, 0), (1, 1)], []):
        try:
            triangulate(bad)
            raise AssertionError("degenerate input must be rejected")
        except ValueError:
            pass

    # 7. boundary sizes: a triangle stays a triangle, a quadrilateral splits in two
    triangle = [(0, 0), (5, 0), (0, 5)]
    check(len(triangulate(triangle)) == 1, "a triangle must be returned as a single triangle")
    quad = [(0, 0), (4, 0), (4, 4), (0, 4)]
    quad_mesh = triangulate(quad)
    check(len(quad_mesh) == 2 and mesh_area(quad_mesh, quad) == 16,
          "a square must split into two triangles of total area 16")
    check(mesh_valid(quad_mesh, quad), "the square mesh must be valid")

    # 8. clockwise input is normalised, not rejected
    clockwise = list(reversed(L_shape()))
    cw_mesh = triangulate(clockwise)
    check(len(cw_mesh) == len(clockwise) - 2 and mesh_valid(cw_mesh, clockwise),
          "a clockwise polygon must be normalised and triangulated just as well")

    # 9. determinism
    check(triangulate(star) == best_mesh and improve(first_mesh, star)[0] == improved,
          "triangulation and improvement must be deterministic")

    print(f"742 star triangles={len(best_mesh)} min_angle first={start_angle:.3f} "
          f"best_ear={greedy_angle:.3f} improved={mesh_min_angle(improved, star):.3f} flips={flips}")
    print(f"742 zigzag triangles={len(triangulate(zigzag()))} "
          f"area={float(abs(signed_area(zigzag()))):.1f}")
    print("PASS")


if __name__ == "__main__":
    main()
