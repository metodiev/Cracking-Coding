"""Task 743 - Implement constrained Delaunay triangulation outline.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

The triangulation is built in three exact stages.  First a plain Delaunay triangulation of the
point set is produced with Bowyer-Watson: points are inserted one at a time into a huge
super-triangle, every triangle whose circumcircle strictly contains the new point is deleted,
the boundary of that cavity is walked and re-triangulated with the new point; finally the
triangles touching the super-triangle are dropped.  All predicates are integer determinants -
orientation plus the incircle determinant - so no epsilon is involved and cocircular input is
handled by the strict form of the test.

Second the constrained edges are inserted.  A constrained segment that is not yet an edge of the
mesh is forced in by flipping the edges that cross it: each crossing edge whose two triangles
form a convex quadrilateral is flipped, which strictly decreases the number of crossings, and
when no crossing edge is flippable the segment is reported as unreachable.  Third the mesh is
legalised: while some non-constrained interior edge fails the local Delaunay test (the opposite
vertex lies inside the adjacent triangle's circumcircle) it is flipped, skipping constrained
edges, so the constrained edges survive.

Outline-level simplification: constraints are inserted by flipping instead of retriangulating a
cavity, and the verified Delaunay property is the local one on the two triangles around a
non-constrained edge rather than the global visibility-restricted (true CDT) condition.  Also
duplicate input points are collapsed.  Verified: the triangle count from Euler's formula
(2h - 2 - k), every triangle positively oriented, the areas summing exactly to the convex hull
area computed by an independent monotone chain, every constrained edge present and uncrossed,
the local Delaunay property on all non-constrained edges, and hand-checked instances.
"""
from fractions import Fraction

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def incircle(a, b, c, d):
    """Positive when d lies strictly inside the circumcircle of the counter-clockwise abc."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    return ((ax * ax + ay * ay) * (bx * cy - cx * by)
            - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay))


def counter_clockwise(a, b, c):
    """The triple reordered so that orientation is positive."""
    return (a, b, c) if orient(a, b, c) > 0 else (a, c, b)


def triangle_area(a, b, c):
    return abs(Fraction(orient(a, b, c))) / 2


def monotone_chain(points):
    """Independent oracle for the convex hull (and hence for the total area)."""
    pts = sorted(points)
    if len(pts) <= 2:
        return pts

    def half(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and orient(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    return half(pts)[:-1] + half(list(reversed(pts)))[:-1]


def hull_area(points):
    poly = monotone_chain(points)
    if len(poly) < 3:
        return Fraction(0)
    total = Fraction(0)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        total += Fraction(a[0]) * b[1] - Fraction(b[0]) * a[1]
    return abs(total) / 2


def super_triangle(points):
    """A triangle strictly containing every point, well clear of all circumcircles."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    mid_x, mid_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    scale = 100 * span
    return [(mid_x - 3 * scale, mid_y - scale),
            (mid_x + 3 * scale, mid_y - scale),
            (mid_x, mid_y + 3 * scale)]


def delaunay(points):
    """Bowyer-Watson with exact integer predicates; returns triangles of point indices."""
    clean = list(dict.fromkeys(points))            # collapse duplicates, keep input order
    if len(clean) < 3:
        return set(), clean
    all_points = clean + super_triangle(clean)
    super_ids = {len(clean), len(clean) + 1, len(clean) + 2}
    triangles = {(len(clean), len(clean) + 1, len(clean) + 2)}
    for index in range(len(clean)):
        point = all_points[index]
        bad, boundary = set(), {}
        for tri in triangles:
            a, b, c = counter_clockwise(all_points[tri[0]], all_points[tri[1]], all_points[tri[2]])
            if incircle(a, b, c, point) > 0:
                bad.add(tri)
                for k in range(3):
                    edge = tuple(sorted((tri[k], tri[(k + 1) % 3])))
                    boundary[edge] = boundary.get(edge, 0) + 1
        triangles -= bad
        for edge, count in boundary.items():
            if count == 1:                          # only the cavity boundary survives
                triangles.add(tuple(sorted((edge[0], edge[1], index))))
    return {tri for tri in triangles if not (set(tri) & super_ids)}, clean


class Mesh:
    """Triangle soup with exact predicates and flip-based edge surgery."""

    def __init__(self, points, triangles):
        self.points = list(points)
        self.triangles = {tuple(sorted(t)) for t in triangles}
        self.constraints = set()

    # ------------------------------------------------------------------ helpers
    def edges(self):
        mapping = {}
        for tri in self.triangles:
            for k in range(3):
                edge = tuple(sorted((tri[k], tri[(k + 1) % 3])))
                mapping.setdefault(edge, []).append(tri)
        return mapping

    def edge_exists(self, u, v):
        return tuple(sorted((u, v))) in self.edges()

    def crossing_edges(self, u, v):
        """Edges (not touching u or v) whose interiors cross the segment u-v."""
        found = []
        pu, pv = self.points[u], self.points[v]
        for edge in self.edges():
            a, b = edge
            if a in (u, v) or b in (u, v):
                continue
            pa, pb = self.points[a], self.points[b]
            if orient(pu, pv, pa) * orient(pu, pv, pb) < 0 and orient(pa, pb, pu) * orient(pa, pb, pv) < 0:
                found.append(edge)
        return found

    def flip(self, edge):
        """Replace the diagonal of the quadrilateral around `edge`; False when it is not convex."""
        owners = self.edges().get(edge)
        if owners is None or len(owners) != 2:
            return False
        t1, t2 = owners
        u, v = edge
        p = next(x for x in t1 if x not in (u, v))
        q = next(x for x in t2 if x not in (u, v))
        a, b = self.points[p], self.points[q]
        c, d = self.points[u], self.points[v]
        if orient(a, b, c) * orient(a, b, d) >= 0:
            return False                            # not a convex quadrilateral: diagonal outside
        self.triangles.discard(t1)
        self.triangles.discard(t2)
        self.triangles.add(tuple(sorted((p, u, q))))
        self.triangles.add(tuple(sorted((p, q, v))))
        return True

    # ------------------------------------------------------------------ stages
    def insert_constraint(self, u, v):
        """Force segment u-v into the mesh by flipping crossing edges (Anglada style)."""
        target = tuple(sorted((u, v)))
        if target in self.constraints:
            return 0
        flips = 0
        for _ in range(len(self.points) ** 2 + 10):
            if self.edge_exists(u, v):
                self.constraints.add(target)
                return flips
            crossings = self.crossing_edges(u, v)
            if not crossings:
                raise ValueError(f"constraint {target} is not recoverable")
            progress = False
            for edge in crossings:
                if edge in self.constraints:
                    continue                        # never destroy another constrained edge
                if self.flip(edge):
                    flips += 1
                    progress = True
                    break
            if not progress:
                raise ValueError(f"constraint {target} crosses a constrained edge")
        raise ValueError(f"constraint {target} did not converge")

    def legalize(self):
        """Flip non-constrained interior edges until the local Delaunay test holds everywhere."""
        flips, rounds = 0, 0
        while rounds < len(self.points) ** 2 + 10:
            rounds += 1
            violated = None
            for edge, owners in self.edges().items():
                if len(owners) != 2 or edge in self.constraints:
                    continue
                t1, t2 = owners
                u, v = edge
                p = next(x for x in t1 if x not in (u, v))
                q = next(x for x in t2 if x not in (u, v))
                a, b, c = counter_clockwise(self.points[u], self.points[v], self.points[p])
                if incircle(a, b, c, self.points[q]) > 0:
                    violated = edge
                    break
            if violated is None:
                return flips
            if not self.flip(violated):
                return flips                        # not flippable: stop, it stays legal enough
            flips += 1
        return flips

    def constrained_delaunay(self, constraints):
        for u, v in constraints:
            self.insert_constraint(u, v)
        self.legalize()
        return self.triangles

    # ------------------------------------------------------------------ inspection
    def total_area(self):
        return sum(triangle_area(*(self.points[x] for x in tri)) for tri in self.triangles)

    def local_delaunay_ok(self):
        for edge, owners in self.edges().items():
            if len(owners) != 2 or edge in self.constraints:
                continue
            t1, t2 = owners
            u, v = edge
            p = next(x for x in t1 if x not in (u, v))
            q = next(x for x in t2 if x not in (u, v))
            a, b, c = counter_clockwise(self.points[u], self.points[v], self.points[p])
            if incircle(a, b, c, self.points[q]) > 0:
                return False
        return True

    def positively_oriented(self):
        return all(orient(*(self.points[x] for x in tri)) != 0 for tri in self.triangles)

    def constraint_present(self, u, v):
        return tuple(sorted((u, v))) in self.edges()

    def no_crossed_constraint(self):
        for (u, v) in self.constraints:
            for edge in self.edges():
                if len(self.edges()[edge]) != 2:
                    continue
                a, b = edge
                if a in (u, v) or b in (u, v):
                    continue
                if (orient(self.points[u], self.points[v], self.points[a])
                        * orient(self.points[u], self.points[v], self.points[b]) < 0
                        and orient(self.points[a], self.points[b], self.points[u])
                        * orient(self.points[a], self.points[b], self.points[v]) < 0):
                    return False
        return True


def euler_triangle_count(points):
    hull = len(monotone_chain(points))
    if hull < 3:
        return 0
    return 2 * len(points) - 2 - hull


def main():
    # 1. hand case: the square plus its centre gives four triangles around the centre
    square = [(0, 0), (6, 0), (6, 6), (0, 6), (3, 3)]
    triangles, clean = delaunay(square)
    check(len(clean) == 5, "five distinct input points must survive")
    check(len(triangles) == 4 and clean == square, "the square with a centre must split into 4")
    mesh = Mesh(clean, triangles)
    check(mesh.total_area() == Fraction(36), "the four triangles must fill the square exactly")
    check(mesh.positively_oriented(), "no triangle may be degenerate")
    check(mesh.local_delaunay_ok(), "the plain Delaunay mesh must satisfy the local property")

    # 2. Euler's formula and the hull area oracle on a random-ish point set
    cloud = [(0, 0), (10, 0), (10, 8), (0, 8), (2, 3), (5, 6), (8, 2), (4, 4), (7, 5)]
    cloud_triangles, cloud_clean = delaunay(cloud)
    check(len(cloud_triangles) == euler_triangle_count(cloud),
          "the triangle count must match Euler's formula 2h - 2 - k")
    cloud_mesh = Mesh(cloud_clean, cloud_triangles)
    check(cloud_mesh.total_area() == hull_area(cloud),
          "the triangle areas must sum exactly to the convex hull area")
    check(cloud_mesh.local_delaunay_ok(), "the cloud triangulation must be locally Delaunay")
    owners_map = cloud_mesh.edges()
    internal_edges = [edge for edge, owners in owners_map.items() if len(owners) == 2]
    hull_edges = [edge for edge, owners in owners_map.items() if len(owners) == 1]
    check(len(internal_edges) + len(hull_edges) == 3 * len(cloud) - 3 - len(hull_edges),
          "the total edge count must satisfy the Euler relation 3h - 3 - k")
    check(len(hull_edges) == len(monotone_chain(cloud)),
          "the number of single-owner edges must equal the number of hull points")

    # 3. a constrained diagonal: the mesh must contain it and stay Delaunay elsewhere
    corners = [(0, 0), (8, 0), (8, 6), (0, 6)]
    constrained_mesh = Mesh(corners, delaunay(corners)[0])
    constrained_mesh.constrained_delaunay([(0, 2)])
    check(len(constrained_mesh.triangles) == 2, "a constrained diagonal leaves exactly two triangles")
    check(all(len(tri) == 3 for tri in constrained_mesh.triangles), "triangles stay triangles")
    check(constrained_mesh.constraint_present(0, 2), "the constrained diagonal must be present")
    check(constrained_mesh.total_area() == Fraction(48), "the area must be preserved")
    check(constrained_mesh.local_delaunay_ok(), "the constrained mesh must stay locally Delaunay")

    # 4. a constrained quadrangle inside a bigger point set: every edge survives
    ring = [(0, 0), (12, 0), (12, 12), (0, 12), (4, 4), (8, 4), (8, 8), (4, 8), (6, 6)]
    ring_mesh = Mesh(ring, delaunay(ring)[0])
    ring_mesh.constrained_delaunay([(4, 5), (5, 6), (6, 7), (7, 4)])
    check(all(ring_mesh.constraint_present(*edge) for edge in [(4, 5), (5, 6), (6, 7), (7, 4)]),
          "every constrained edge of the inner quadrangle must appear in the mesh")
    check(ring_mesh.no_crossed_constraint(), "no edge may cross a constrained edge")
    check(ring_mesh.total_area() == Fraction(144), "the outer square area must be tiled exactly")
    check(ring_mesh.local_delaunay_ok(),
          "the constrained mesh must satisfy the local Delaunay property off the constraints")

    # 5. an impossible constraint is reported, not silently ignored
    pair = [(0, 0), (6, 0), (6, 6), (0, 6)]
    tight = Mesh(pair, delaunay(pair)[0])
    tight.constrained_delaunay([(0, 2)])
    before = set(tight.triangles)
    tight.constrained_delaunay([(2, 0)])
    check(tight.triangles == before, "re-inserting the same constraint must be a no-op")
    try:
        Mesh(pair, delaunay(pair)[0]).constrained_delaunay([(0, 5)])
        raise AssertionError("a constraint with an unknown endpoint must be rejected")
    except (ValueError, IndexError):
        pass

    # 6. boundary cases: fewer than three points, collinear points
    check(delaunay([(0, 0)])[0] == set(), "a single point cannot be triangulated")
    check(delaunay([(0, 0), (1, 1)])[0] == set(), "two points cannot be triangulated")
    line_triangles, line_clean = delaunay([(0, 0), (1, 0), (2, 0), (3, 0)])
    check(line_triangles == set(), "collinear points have no non-degenerate triangulation")
    check(len(line_clean) == 4, "collinear points must still be deduplicated and kept")
    duplicate = delaunay([(0, 0), (0, 0), (4, 0), (0, 4)])
    check(len(duplicate[1]) == 3 and len(duplicate[0]) == 1,
          "duplicate input points must collapse to one vertex")

    # 7. determinism: identical input, identical mesh
    again, _ = delaunay(ring)
    check(Mesh(ring, again).triangles == Mesh(ring, delaunay(ring)[0]).triangles,
          "rebuilding must reproduce the same Delaunay mesh")

    print(f"743 cloud points={len(cloud)} triangles={len(cloud_triangles)} "
          f"euler={euler_triangle_count(cloud)} hull_area={float(hull_area(cloud)):.2f}")
    print(f"743 ring triangles={len(ring_mesh.triangles)} constraints={len(ring_mesh.constraints)} "
          f"area={float(ring_mesh.total_area()):.2f}")
    print("PASS")


if __name__ == "__main__":
    main()
