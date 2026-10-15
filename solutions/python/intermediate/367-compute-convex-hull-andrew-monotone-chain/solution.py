"""Task 367 - Compute convex hull (Andrew monotone chain). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Andrew's monotone chain: sort the unique points lexicographically, sweep a lower hull
left-to-right and an upper hull right-to-left, popping whenever the last turn is not a
strict left turn (cross <= 0), which drops collinear points on hull edges.  The
concatenation is counter-clockwise and starts at the lexicographically smallest point;
normalize() re-establishes that canonical form (reverse if the shoelace sign says
clockwise, then rotate to the smallest point) so it can be compared against the Graham
scan from task 366.  Both hulls must be identical on hand-checked sets and on
deterministic LCG point sets.
"""

from functools import cmp_to_key


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def monotone_chain(points):
    """Convex hull, counter-clockwise, starting at the lexicographically smallest point."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def graham_scan(points):
    """The task 366 Graham scan, kept here so both hulls can be compared directly."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]
    pivot = pts[0]
    rest = pts[1:]

    def compare(a, b):
        turn = cross(pivot, a, b)
        if turn > 0:
            return -1
        if turn < 0:
            return 1
        dist_a = (a[0] - pivot[0]) ** 2 + (a[1] - pivot[1]) ** 2
        dist_b = (b[0] - pivot[0]) ** 2 + (b[1] - pivot[1]) ** 2
        return -1 if dist_a < dist_b else (1 if dist_a > dist_b else 0)

    rest.sort(key=cmp_to_key(compare))
    hull = [pivot]
    for point in rest:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return [list(p) for p in hull]


def signed_area2(hull):
    total = 0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def normalize(hull):
    """Counter-clockwise order starting at the lexicographically smallest point."""
    pts = [tuple(p) for p in hull]
    if len(pts) <= 1:
        return [list(p) for p in pts]
    if signed_area2(pts) < 0:
        pts.reverse()
    start = min(range(len(pts)), key=lambda i: pts[i])
    pts = pts[start:] + pts[:start]
    return [list(p) for p in pts]


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
    # Same hand-checked sets as task 366; both algorithms must return the same hull.
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    inside = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)]
    collinear = [(0, 0), (1, 0), (2, 0), (3, 0)]
    mixed = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 3)]
    duplicates = [(0, 0), (0, 0), (2, 0), (2, 0), (2, 2), (0, 2), (2, 2)]
    triangle = [(0, 0), (4, 0), (2, 3)]
    tiny = [[(5, 7)], [(0, 0), (1, 1)], []]

    for name, points, expected in [
        ("square", square, [[0, 0], [2, 0], [2, 2], [0, 2]]),
        ("inside", inside, [[0, 0], [2, 0], [2, 2], [0, 2]]),
        ("collinear", collinear, [[0, 0], [3, 0]]),
        ("mixed", mixed, [[0, 0], [3, 3], [0, 3]]),
        ("duplicates", duplicates, [[0, 0], [2, 0], [2, 2], [0, 2]]),
        ("triangle", triangle, [[0, 0], [4, 0], [2, 3]]),
    ]:
        chain = monotone_chain(points)
        scan = graham_scan(points)
        assert chain == expected, (name, chain)
        assert scan == expected, (name, scan)
        assert normalize(chain) == normalize(scan), name

    for points in tiny:
        assert normalize(monotone_chain(points)) == normalize(graham_scan(points))

    assert signed_area2(monotone_chain(square)) == 8      # CCW, twice the area 4

    # Normalization really canonicalizes: reversed clockwise input flips back.
    clockwise = [[0, 0], [0, 2], [2, 2], [2, 0]]
    assert signed_area2(clockwise) < 0
    assert normalize(clockwise) == [[0, 0], [2, 0], [2, 2], [0, 2]]
    rotated = [[2, 2], [0, 2], [0, 0], [2, 0]]
    assert normalize(rotated) == [[0, 0], [2, 0], [2, 2], [0, 2]]

    # LCG cross-check: monotone chain and Graham scan agree after normalization.
    seed = 90210
    for trial in range(80):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 45
        modulus = 5 + seed % 25                            # small ranges force collinearity
        points, seed = lcg_points(count, seed, modulus)
        chain = normalize(monotone_chain(points))
        scan = normalize(graham_scan(points))
        assert chain == scan, (trial, points)
        assert signed_area2(chain) >= 0
        for i in range(len(chain)):
            a = chain[i - 1]
            b = chain[i]
            c = chain[(i + 1) % len(chain)]
            assert len(chain) <= 2 or cross(a, b, c) > 0, (trial, points)


if __name__ == "__main__":
    main()
    print("PASS")
