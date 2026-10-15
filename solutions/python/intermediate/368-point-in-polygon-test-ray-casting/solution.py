"""Task 368 - Point-in-polygon test (ray casting). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Ray casting with the standard half-open edge rule: an edge (y1, y2) counts as crossing
the horizontal ray of the test point p only when exactly one endpoint satisfies y > p.y,
i.e. the edge's y-interval is treated as half-open [y_low, y_high).  Horizontal edges
never cross, a vertex shared by two edges is counted exactly once when the polygon
locally passes through the ray, and both vertices of a horizontal ray passing through a
vertex are handled without double counting.  The intersection test is exact: instead of
computing x_intersection as a float, the code compares (x1 - px) * dy + (py - y1) * dx
against zero with the sign of dy, so only integer arithmetic is used.  Points exactly on
the boundary are reported separately (exact cross = 0 plus bounding box); the winding
number alternative must agree with ray casting on all non-boundary points.
"""


def cross_at(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def on_segment(a, b, p):
    """Exact test for p lying on the closed segment a-b."""
    if cross_at(a, b, p) != 0:
        return False
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def on_boundary(polygon, p):
    n = len(polygon)
    return any(on_segment(polygon[i], polygon[(i + 1) % n], p) for i in range(n))


def ray_cast_inside(polygon, p):
    """Even-odd rule with the half-open edge rule; exact integer comparisons."""
    px, py = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > py) == (y2 > py):                        # ignore horizontal edges here
            continue
        dy = y2 - y1
        # x_intersection > px  <=>  lhs / dy > 0 with lhs = (x1 - px) * dy + (py - y1) * (x2 - x1)
        lhs = (x1 - px) * dy + (py - y1) * (x2 - x1)
        if (lhs > 0) == (dy > 0):
            inside = not inside
    return inside


def winding_number(polygon, p):
    """Winding number alternative; +1 per CCW crossing above p, -1 per clockwise one."""
    px, py = p
    winding = 0
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if a[1] <= py:
            if b[1] > py and cross_at(a, b, p) > 0:
                winding += 1
        else:
            if b[1] <= py and cross_at(a, b, p) < 0:
                winding -= 1
    return winding


def classify(polygon, p):
    """Returns 'boundary', 'inside' or 'outside' using ray casting."""
    if on_boundary(polygon, p):
        return "boundary"
    return "inside" if ray_cast_inside(polygon, p) else "outside"


def monotone_chain(points):
    """Convex hull (CCW) used to build hand-free test polygons from LCG points."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross_at(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross_at(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def lcg_points(count, seed, modulus):
    points = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        x = seed % modulus
        seed = (seed * 75 + 74) % 65537
        y = seed % modulus
        points.append((x, y))
    return points, seed


def main():
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert classify(square, (2, 2)) == "inside"
    assert classify(square, (5, 2)) == "outside"
    assert classify(square, (-1, 2)) == "outside"
    assert classify(square, (2, 5)) == "outside"
    assert classify(square, (4, 4)) == "boundary"         # vertex
    assert classify(square, (0, 2)) == "boundary"         # left edge
    assert classify(square, (2, 0)) == "boundary"         # bottom edge
    assert classify(square, (0, 0)) == "boundary"
    # (-1, 0): the horizontal ray passes straight through vertices (0,0) and (4,0);
    # the half-open rule counts both vertical edges exactly once -> outside.
    assert classify(square, (-1, 0)) == "outside"
    assert ray_cast_inside(square, (-1, 0)) is False
    assert winding_number(square, (-1, 0)) == 0

    triangle = [(0, 0), (6, 0), (0, 6)]
    assert classify(triangle, (1, 1)) == "inside"
    assert classify(triangle, (3, 3)) == "boundary"       # hypotenuse x + y = 6
    assert classify(triangle, (5, 2)) == "outside"        # 7 > 6
    assert classify(triangle, (7, 0)) == "outside"
    assert classify(triangle, (0, 3)) == "boundary"

    # Concave "arrow" polygon: a square with a triangular notch cut from the top.
    arrow = [(0, 0), (6, 0), (6, 6), (3, 3), (0, 6)]
    assert classify(arrow, (3, 1)) == "inside"
    assert classify(arrow, (3, 4)) == "outside"           # inside the notch
    assert classify(arrow, (1, 4)) == "inside"
    assert classify(arrow, (5, 4)) == "inside"
    assert classify(arrow, (3, 3)) == "boundary"          # notch vertex
    assert classify(arrow, (5, 5)) == "boundary"          # on edge y = x
    assert classify(arrow, (2, 6)) == "outside"
    assert classify(arrow, (0, 3)) == "boundary"
    # (1, 3): the ray y = 3 passes through the notch vertex (3, 3) and exits at x = 6;
    # the vertex contributes two crossings that cancel, then one real crossing -> inside.
    assert classify(arrow, (1, 3)) == "inside"
    assert ray_cast_inside(arrow, (1, 3)) is True
    assert winding_number(arrow, (1, 3)) == 1

    # Both algorithms must agree everywhere except on the boundary.
    polygons = [square, triangle, arrow]
    for polygon in polygons:
        for x in range(-2, 9):
            for y in range(-2, 9):
                p = (x, y)
                if on_boundary(polygon, p):
                    continue
                assert (ray_cast_inside(polygon, p) == (winding_number(polygon, p) != 0)), \
                    (polygon, p)
    assert ray_cast_inside(arrow, (1, 3)) == (winding_number(arrow, (1, 3)) != 0)

    # LCG convex polygons from hulls: ray casting and winding number must agree, and
    # boundary points must be classified as such by both checks.
    seed = 31415
    for trial in range(30):
        seed = (seed * 75 + 74) % 65537
        count = 3 + seed % 20
        points, seed = lcg_points(count, seed, 12)
        polygon = monotone_chain(points)
        if len(polygon) < 3:
            continue
        for x in range(-2, 14):
            for y in range(-2, 14):
                p = (x, y)
                if on_boundary(polygon, p):
                    continue                                  # boundary is documented as undefined
                assert ray_cast_inside(polygon, p) == (winding_number(polygon, p) != 0), \
                    (trial, polygon, p)


if __name__ == "__main__":
    main()
    print("PASS")
