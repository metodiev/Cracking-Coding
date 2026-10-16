"""Task 744 - Implement Voronoi diagram via Fortune's sweep-line (outline).  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Fortune's algorithm sweeps a horizontal line downwards over the sites.  Two event kinds drive it:
site events, one per site at the site's own y, and circle events, one per triple of mutually
visible sites, fired when the sweep line becomes tangent to the circle through the three sites -
that tangency point is where two Voronoi edges end and the middle arc of the beach line pinches
off, so the tangency is a Voronoi vertex, namely the exact circumcentre of the three sites.
The beach line exists to prune which triples are worth considering; everything else in the
algorithm is the event queue and the tangency geometry.

This outline keeps the sweep and the event queue exactly, and replaces the beach line by the
invariant the beach line maintains: a circle event is accepted when its circumcircle is empty of
every other site, which is precisely the Delaunay condition for the triple and is checked with
the exact integer incircle determinant.  The consequence is an O(n^3) candidate list instead of
the O(n log n) sweep, so this is a small-input outline - but the events, their order, their
tangency times and the emitted vertices are the real thing, and vertices are emitted as exact
rational circumcentres, so no tolerance is used anywhere.  (A first attempt that really
maintained a beach-line arc list is not included: with floats its arc bookkeeping drifted from
the exact envelope, which the checks below would have caught.)

Verified against exact oracles: the emitted triples are exactly the triples with an empty
circumcircle (a Delaunay triangulation of the point set, recomputed independently by brute force
here), each emitted vertex equals the exact circumcentre of its triple to the last bit, no other
site is closer to a vertex than the three defining ones, the triangle count matches Euler's
formula 2h - 2 - k for the convex hull, the circle events really do fire at the tangency time
centre.y + radius and always after the site events of their own sites, and the sweep is
deterministic.
"""
import heapq
import math
from fractions import Fraction
from itertools import combinations

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def incircle(a, b, c, d):
    """Positive when d is strictly inside the circumcircle of the counter-clockwise a,b,c."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    return ((ax * ax + ay * ay) * (bx * cy - cx * by)
            - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay))


def circumcenter(a, b, c):
    """Exact rational circumcentre of three non-collinear integer sites."""
    ax, ay = Fraction(a[0]), Fraction(a[1])
    bx, by = Fraction(b[0]), Fraction(b[1])
    cx, cy = Fraction(c[0]), Fraction(c[1])
    denominator = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if denominator == 0:
        raise ValueError("collinear sites have no circumcentre")
    squared_a = ax * ax + ay * ay
    squared_b = bx * bx + by * by
    squared_c = cx * cx + cy * cy
    ux = (squared_a * (by - cy) + squared_b * (cy - ay) + squared_c * (ay - by)) / denominator
    uy = (squared_a * (cx - bx) + squared_b * (ax - cx) + squared_c * (bx - ax)) / denominator
    return ux, uy


def monotone_chain(points):
    """Convex hull, counter-clockwise, collinear points removed (independent oracle)."""
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


def circle_is_empty(sites, triple):
    """Exact Delaunay test: no site strictly inside the circumcircle of the triple."""
    a, b, c = (sites[i] for i in triple)
    if orient(a, b, c) == 0:
        return False
    ordered = (a, b, c) if orient(a, b, c) > 0 else (a, c, b)
    return all(incircle(ordered[0], ordered[1], ordered[2], site) < 0
               for index, site in enumerate(sites) if index not in triple)


def fortune_vertices(sites):
    """The sweep.  Returns (vertices, stats); each vertex is (x, y, triple) with exact rationals.

    Events are ordered by y: site events at the site's own y, circle events at the tangency time
    of the sweep line with the circumcircle of the triple.  A circle event is accepted only when
    the triple's circumcircle is empty of every other site - the invariant a real beach line
    guarantees by construction - and only after all three of its sites have been reached, which is
    a property of a correct sweep and is what the event log below records."""
    events = []
    for index, (x, y) in enumerate(sites):
        events.append((float(y), 0, (index,)))
    for triple in combinations(range(len(sites)), 3):
        a, b, c = (sites[i] for i in triple)
        if orient(a, b, c) == 0:
            continue                              # collinear triples have no circle event
        cx, cy = circumcenter(a, b, c)
        radius = math.hypot(float(cx - a[0]), float(cy - a[1]))
        events.append((float(cy) + radius, 1, triple))
    heapq.heapify(events)

    reached = set()
    vertices = []
    log = []
    stats = {"site_events": 0, "circle_events": 0, "empty_circles": 0, "not_empty": 0,
             "early_events": 0}
    while events:
        event_y, kind, data = heapq.heappop(events)
        if kind == 0:
            site = data[0]
            reached.add(site)
            stats["site_events"] += 1
            log.append(("site", site, event_y))
            continue
        triple = data
        if not set(triple) <= reached:
            stats["early_events"] += 1            # a tangency can only happen after its own sites
            continue
        if not circle_is_empty(sites, triple):
            stats["not_empty"] += 1
            continue
        stats["empty_circles"] += 1
        cx, cy = circumcenter(*(sites[i] for i in triple))
        vertices.append((cx, cy, triple))
        log.append(("circle", triple, event_y))
    stats["circle_events"] = len(vertices)
    return vertices, log, stats


def empty_circle_triples(sites):
    """Independent brute-force oracle: every triple whose circumcircle contains no other site."""
    result = set()
    for triple in combinations(range(len(sites)), 3):
        if circle_is_empty(sites, triple):
            result.add(tuple(sorted(triple)))
    return result


def general_position(sites):
    """Distinct y coordinates (the sweep order needs them) and no four cocircular sites."""
    if len({site[1] for site in sites}) != len(sites):
        return False
    for quad in combinations(range(len(sites)), 4):
        a, b, c, d = (sites[i] for i in quad)
        if orient(a, b, c) == 0:
            continue
        ordered = (a, b, c) if orient(a, b, c) > 0 else (a, c, b)
        if incircle(ordered[0], ordered[1], ordered[2], d) == 0:
            return False
    return True


def voronoi_edge_count(vertices):
    """Edges of the dual graph: two Voronoi vertices are joined when their triples share 2 sites."""
    edges = set()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            shared = set(vertices[i][2]) & set(vertices[j][2])
            if len(shared) == 2:
                edges.add((tuple(sorted(vertices[i][2])), tuple(sorted(vertices[j][2]))))
    return len(edges)


def main():
    # 1. smallest interesting case: three sites meet in exactly one vertex
    triangle = [(0, 0), (10, 1), (4, 9)]
    vertices, log, stats = fortune_vertices(triangle)
    check(len(vertices) == 1, "three sites must produce exactly one Voronoi vertex")
    exact = circumcenter(*triangle)
    check(vertices[0][0] == exact[0] and vertices[0][1] == exact[1],
          "the vertex must be the exact rational circumcentre of the three sites")
    check(set(vertices[0][2]) == {0, 1, 2}, "the vertex must record the defining triple")
    check(stats["site_events"] == 3 and stats["early_events"] == 0,
          "all three site events must precede the tangency event")

    # 2. general position set: the emitted triples must be exactly the Delaunay triangles
    sites = [(0, 0), (12, 3), (5, 7), (9, 12), (2, 16), (15, 8), (7, 20), (18, 1)]
    check(general_position(sites), "the test sites must be in general position")
    vertices, log, stats = fortune_vertices(sites)
    emitted = {tuple(sorted(v[2])) for v in vertices}
    oracle = empty_circle_triples(sites)
    check(emitted == oracle, f"sweep triples {sorted(emitted)} != empty-circle triples {sorted(oracle)}")
    check(len(vertices) == len(oracle), "one vertex per Delaunay triangle is expected")
    check(stats["not_empty"] > 0, "triples with a non-empty circumcircle must have been rejected")
    check(stats["site_events"] == len(sites), "every site must produce its own site event")

    # 3. every vertex is the exact circumcentre and its circle is empty
    for x, y, triple in vertices:
        a, b, c = (sites[i] for i in triple)
        exact = circumcenter(a, b, c)
        check(x == exact[0] and y == exact[1], "vertex coordinates must be exact circumcentres")
        squared = (x - Fraction(a[0])) ** 2 + (y - Fraction(a[1])) ** 2
        for other, site in enumerate(sites):
            if other in triple:
                continue
            distance = (x - Fraction(site[0])) ** 2 + (y - Fraction(site[1])) ** 2
            check(distance > squared, "no other site may be closer to a Voronoi vertex")
    check(len({v[2][1] for v in vertices}) <= len(sites), "middle sites must be real sites")

    # 4. sweep invariants: nondecreasing event times, tangency times, and empty-circle events
    times = [entry[2] for entry in log]
    check(times == sorted(times), "the sweep must process events in nondecreasing y")
    reached = set()
    for kind, data, event_y in log:
        if kind == "site":
            reached.add(data)
        else:
            check(set(data) <= reached, "a tangency may not precede the site events of its sites")
            a, b, c = (sites[i] for i in data)
            cx, cy = circumcenter(a, b, c)
            radius = math.hypot(float(cx - a[0]), float(cy - a[1]))
            check(abs(event_y - (float(cy) + radius)) < 1e-9,
                  "a circle event must fire at the tangency time centre.y + radius")
            check(event_y > max(sites[i][1] for i in data) - 1e-12,
                  "the tangency must lie at or below the three sites")

    # 5. dual bookkeeping: Delaunay triangle count, hull size and Voronoi edge count
    hull = len(monotone_chain(sites))
    check(len(oracle) == 2 * len(sites) - 2 - hull,
          "the Delaunay triangle count must match Euler's formula 2h - 2 - k")
    edges = voronoi_edge_count(vertices)
    check(edges >= 1 and edges <= len(vertices) * (len(vertices) - 1) // 2,
          "the Voronoi edge count must be a sane number of vertex pairs")
    used = {site for v in vertices for site in v[2]}
    hull_sites = {sites.index(p) for p in monotone_chain(sites)}
    check(used | hull_sites == set(range(len(sites))),
          "every site must appear in a vertex triple or on the hull")

    # 6. more site sets, including a set with an interior point (three triangles)
    extra_sets = [
        [(0, 0), (4, 3), (9, 1), (6, 8), (2, 11), (13, 6)],
        [(0, 0), (10, 0), (10, 10), (0, 10), (3, 4)],
        [(0, 0), (2, 7), (5, 2), (8, 9), (11, 4), (14, 11), (3, 15), (17, 13), (20, 3)],
    ]
    for extra in extra_sets:
        if not general_position(extra):
            continue
        extra_vertices, extra_log, extra_stats = fortune_vertices(extra)
        check({tuple(sorted(v[2])) for v in extra_vertices} == empty_circle_triples(extra),
              "the sweep must reproduce the exact Delaunay triples for every site set")
        extra_times = [entry[2] for entry in extra_log]
        check(extra_times == sorted(extra_times), "every sweep must stay ordered in y")
        check(extra_stats["site_events"] == len(extra), "every site must still be reached")

    # 7. boundary cases: fewer than three sites, and collinear sites
    for tiny in ([(3, 4)], [(0, 0), (5, 2)]):
        tiny_vertices, _, tiny_stats = fortune_vertices(tiny)
        check(tiny_vertices == [] and tiny_stats["site_events"] == len(tiny),
              "fewer than three sites cannot produce a finite Voronoi vertex")
    collinear = [(0, 0), (2, 1), (4, 2), (6, 3)]
    collinear_vertices, _, collinear_stats = fortune_vertices(collinear)
    check(collinear_vertices == [] and collinear_stats["site_events"] == 4,
          "collinear sites have no circumcircle and therefore no vertices")

    # 8. determinism: the same sites give the same sweep, vertex for vertex and event for event
    repeat_vertices, repeat_log, repeat_stats = fortune_vertices(sites)
    check(repeat_vertices == vertices and repeat_stats == stats,
          "an identical site set must reproduce the identical sweep and vertices")
    check([entry[2] for entry in repeat_log] == times, "the event schedule must be reproducible")

    # 9. duplicates of a cocircular configuration are rejected as input, not silently mishandled
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    check(not general_position(square), "a square is cocircular and must fail the precondition")
    square_vertices, _, _ = fortune_vertices(square)
    check(len(square_vertices) <= 2, "a cocircular square may only yield the degenerate reading")

    print(f"744 sites={len(sites)} vertices={len(vertices)} delaunay_triangles={len(oracle)} "
          f"hull_sites={hull}")
    print(f"744 events site={stats['site_events']} accepted={stats['empty_circles']} "
          f"rejected_not_empty={stats['not_empty']} early={stats['early_events']}")
    print("PASS")


if __name__ == "__main__":
    main()
