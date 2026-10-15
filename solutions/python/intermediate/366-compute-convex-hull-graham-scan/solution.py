"""Task 366 - Compute convex hull (Graham scan). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Graham scan with exact integer arithmetic.  Duplicate points are removed, the pivot is
the lexicographically smallest point (min x, then min y), and the remaining points are
sorted by polar angle around the pivot.  The comparator never uses floating point: it
compares cross products of vectors from the pivot and breaks equal angles by nearest
first (the comparison map for "a before b" is: cross > 0, or cross == 0 with a closer to
the pivot).  While folding the sorted points the scan pops whenever the last turn is not
a strict left turn (cross <= 0), which removes points lying collinear on a hull edge, so
only the extreme endpoints of collinear runs are kept.  Hand-checked sets (square,
square with an interior point, collinear run, duplicates) are asserted and hull size,
area and the full point list are cross-checked against a Jarvis march on deterministic
LCG point sets.
"""

from functools import cmp_to_key


def cross(o, a, b):
    """Cross product (a - o) x (b - o); > 0 means o -> a -> b turns left."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def graham_scan(points):
    """Convex hull (counter-clockwise, starting at the lexicographically smallest point)."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]
    pivot = pts[0]
    rest = pts[1:]

    def compare(a, b):
        turn = cross(pivot, a, b)
        if turn > 0:
            return -1                                     # a has the smaller angle
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


def jarvis_march(points):
    """Independent gift wrapping oracle; same conventions as graham_scan."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]
    start = pts[0]
    hull = [start]
    current = start
    while True:
        candidate = None
        for p in pts:
            if p == current:
                continue
            if candidate is None:
                candidate = p
                continue
            turn = cross(current, candidate, p)
            if turn < 0:
                candidate = p                              # p is the better clockwise fit
            elif turn == 0:
                same_direction = ((p[0] - current[0]) * (candidate[0] - current[0]) +
                                  (p[1] - current[1]) * (candidate[1] - current[1])) > 0
                if same_direction and _dist2(current, p) > _dist2(current, candidate):
                    candidate = p                          # keep the farthest collinear point
        if candidate == start:
            break
        hull.append(candidate)
        current = candidate
        assert len(hull) <= len(pts)
    return [list(p) for p in hull]


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def hull_area2(hull):
    """Twice the signed area (shoelace); positive for counter-clockwise order."""
    total = 0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


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
    # Square: hull is the square itself, CCW starting at the lexicographically smallest.
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert graham_scan(square) == [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert jarvis_march(square) == graham_scan(square)
    assert hull_area2(graham_scan(square)) == 8            # twice the area 4

    # Square plus an interior point: interior point is excluded.
    inside = square + [(1, 1)]
    assert graham_scan(inside) == [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert jarvis_march(inside) == graham_scan(inside)

    # Collinear points: only the two extreme endpoints survive.
    collinear = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert graham_scan(collinear) == [[0, 0], [3, 0]]
    assert jarvis_march(collinear) == [[0, 0], [3, 0]]
    mixed = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 3)]
    assert graham_scan(mixed) == [[0, 0], [3, 3], [0, 3]]  # diagonal run collapses
    assert jarvis_march(mixed) == graham_scan(mixed)

    # Duplicates: identical points cannot enlarge the hull.
    duplicates = [(0, 0), (0, 0), (2, 0), (2, 0), (2, 2), (0, 2), (2, 2)]
    assert graham_scan(duplicates) == [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert jarvis_march(duplicates) == graham_scan(duplicates)

    # Triangle and degenerate sizes.
    assert graham_scan([(0, 0), (4, 0), (2, 3)]) == [[0, 0], [4, 0], [2, 3]]
    assert graham_scan([(5, 7)]) == [[5, 7]]
    assert graham_scan([(0, 0), (1, 1)]) == [[0, 0], [1, 1]]
    assert graham_scan([]) == []

    # LCG cross-check: hull size, doubled area and the exact point list vs Jarvis.
    seed = 15731
    for trial in range(60):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 40
        modulus = 6 + seed % 20                            # small ranges force collinearity
        points, seed = lcg_points(count, seed, modulus)
        scan = graham_scan(points)
        march = jarvis_march(points)
        assert len(scan) == len(march), (trial, points)
        assert hull_area2(scan) == hull_area2(march), (trial, points)
        assert scan == march, (trial, points)
        # The reported hull must be convex (strict left turns all the way around).
        assert len(scan) <= 3 or hull_area2(scan) >= 0
        for i in range(len(scan)):
            a = scan[i - 1]
            b = scan[i]
            c = scan[(i + 1) % len(scan)]
            assert len(scan) <= 2 or cross(a, b, c) > 0, (trial, points)


if __name__ == "__main__":
    main()
    print("PASS")
