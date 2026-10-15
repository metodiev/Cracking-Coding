"""Task 373 - Diameter of a convex polygon by rotating calipers. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

The diameter of a point set is its farthest pair; the farthest pair of a set always
lies on the convex hull, so the problem reduces to the diameter of a convex polygon.
Rotating calipers walk an antipodal vertex j along the hull for each directed edge
(i, i+1): j advances while the triangle area (i, i+1, j+1) exceeds (i, i+1, j).
Every diameter pair is antipodal to some edge, so checking the distance from both
edge endpoints to j for all i finds the exact maximum.  All arithmetic is integer:
the result is the exact SQUARED distance; a float square root is taken only to
compare against the classic square = side*sqrt(2) expectation (relative tolerance
1e-12, the standard double-sqrt tolerance).

The hull is built by Andrew's monotone chain (duplicate points removed, collinear
points dropped).  Degenerate cases are documented: empty/one point -> diameter 0;
two distinct points or a fully collinear set -> hull is the two extreme points.
Cross-checked against the brute-force O(h^2) scan over the hull.

Independent oracle: brute force over ALL input points (not just the hull) - the
farthest pair of any point set is a hull pair, so all three numbers must agree.
"""

import math


def convex_hull(points):
    """Andrew's monotone chain: counter-clockwise hull without collinear points."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def dist_squared(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hull_diameter_squared(hull):
    """Rotating calipers over a CCW convex hull; exact integer result."""
    h = len(hull)
    if h <= 1:
        return 0
    if h == 2:
        return dist_squared(hull[0], hull[1])
    best = 0
    j = 1
    for i in range(h):
        ni = (i + 1) % h
        steps = 0
        while steps <= h:                       # guard against degenerate ties
            nj = (j + 1) % h
            if area2(hull[i], hull[ni], hull[nj]) > area2(hull[i], hull[ni], hull[j]):
                j = nj
                steps += 1
            else:
                break
        best = max(best, dist_squared(hull[i], hull[j]), dist_squared(hull[ni], hull[j]))
    return best


def diameter_squared(points):
    """Returns (diameter^2, hull) for a point set."""
    hull = convex_hull(points)
    return hull_diameter_squared(hull), hull


def brute_hull_diameter_squared(hull):
    """O(h^2) oracle over the hull vertices."""
    best = 0
    for i in range(len(hull)):
        for k in range(i + 1, len(hull)):
            best = max(best, dist_squared(hull[i], hull[k]))
    return best


def brute_all_points_diameter_squared(points):
    """O(n^2) oracle over every input point."""
    best = 0
    for i in range(len(points)):
        for k in range(i + 1, len(points)):
            best = max(best, dist_squared(points[i], points[k]))
    return best


def lcg_points(count, seed, extent=50):
    points = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        x = seed % extent
        seed = (seed * 75 + 74) % 65537
        y = seed % extent
        points.append((x, y))
    return points, seed


def main():
    # classic square: side 3, diagonal 3*sqrt(2)
    side = 3
    square = [(0, 0), (side, 0), (side, side), (0, side)]
    diameter2, hull = diameter_squared(square)
    assert diameter2 == 2 * side * side == 18
    assert len(hull) == 4
    assert math.isclose(math.sqrt(diameter2), side * math.sqrt(2), rel_tol=1e-12)

    # hand-checked small cases
    assert diameter_squared([(7, 7)]) == (0, [(7, 7)])
    assert diameter_squared([(1, 1), (4, 5)])[0] == 25               # 3-4-5 style
    diameter2, hull = diameter_squared([(0, 0), (4, 0), (0, 3)])
    assert diameter2 == 25 and len(hull) == 3                        # 4-3-5 triangle
    assert diameter2 == brute_hull_diameter_squared(hull)
    diameter2, hull = diameter_squared([(0, 0), (2, 0), (2, 5), (0, 5)])
    assert diameter2 == 29                                           # rectangle diagonal
    # collinear input: hull degenerates to the two extreme points
    diameter2, hull = diameter_squared([(0, 0), (1, 1), (3, 3)])
    assert hull == [(0, 0), (3, 3)] and diameter2 == 18
    # interior point is ignored by the hull
    with_interior = [(0, 0), (3, 0), (3, 3), (0, 3), (1, 1)]
    diameter2, hull = diameter_squared(with_interior)
    assert hull == [(0, 0), (3, 0), (3, 3), (0, 3)]
    assert diameter2 == 18

    # LCG cross-checks: calipers == brute force on the hull == brute force on all points
    seed = 37313
    for trial in range(60):
        count = 3 + seed % 10
        points, seed = lcg_points(count, seed)
        diameter2, hull = diameter_squared(points)
        assert diameter2 == brute_hull_diameter_squared(hull), (trial, points)
        assert diameter2 == brute_all_points_diameter_squared(points), (trial, points)

    # larger deterministic set for a slower, wider sweep
    points, seed = lcg_points(120, seed, extent=100)
    diameter2, hull = diameter_squared(points)
    assert diameter2 == brute_all_points_diameter_squared(points)
    assert len(hull) >= 3


if __name__ == "__main__":
    main()
    print("PASS")
