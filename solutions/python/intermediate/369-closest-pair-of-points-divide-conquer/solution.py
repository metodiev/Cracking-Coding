"""Task 369 - Closest pair of points (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Classic O(n log n) divide & conquer on points sorted by x.  Each recursive call returns
the best (squared) distance of its half plus the half's points sorted by y, obtained by
merging the two children's y-sorted lists, so the strip is already y-sorted.  Only strip
pairs with dy² < best_so_far can improve the answer, and best keeps shrinking while the
strip is scanned, which is the standard safe pruning.  Distances are squared and exact
integers (no epsilon anywhere).  Assertions compare against a brute-force O(n²) oracle
on many deterministic LCG point sets (sizes up to ~60) plus a hand case with a duplicate
point (distance 0).
"""


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def closest_pair(points):
    """Returns (squared distance, [p, q]) of the closest pair; needs at least 2 points."""
    pts = sorted(map(tuple, points))
    if len(pts) < 2:
        raise ValueError("need at least two points")
    best, pair, _ = _rec(pts)
    return best, [list(pair[0]), list(pair[1])]


def _rec(pts_by_x):
    """(best_sq, pair, pts sorted by y) for the x-sorted slice."""
    n = len(pts_by_x)
    if n <= 3:
        best = None
        pair = None
        for i in range(n):
            for j in range(i + 1, n):
                d = dist2(pts_by_x[i], pts_by_x[j])
                if best is None or d < best:
                    best = d
                    pair = (pts_by_x[i], pts_by_x[j])
        return best, pair, sorted(pts_by_x, key=lambda p: (p[1], p[0]))

    mid = n // 2
    left_best, left_pair, left_y = _rec(pts_by_x[:mid])
    right_best, right_pair, right_y = _rec(pts_by_x[mid:])
    if left_best <= right_best:
        best, pair = left_best, left_pair
    else:
        best, pair = right_best, right_pair

    merged_y = []
    i = j = 0
    while i < len(left_y) or j < len(right_y):
        if j >= len(right_y) or (i < len(left_y) and
                                 (left_y[i][1], left_y[i][0]) <= (right_y[j][1], right_y[j][0])):
            merged_y.append(left_y[i])
            i += 1
        else:
            merged_y.append(right_y[j])
            j += 1

    mid_x = pts_by_x[mid][0]
    strip = [p for p in merged_y if (p[0] - mid_x) ** 2 < best]
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) ** 2 < best:
            d = dist2(strip[i], strip[j])
            if d < best:
                best = d
                pair = (strip[i], strip[j])
            j += 1
    return best, pair, merged_y


def brute_force(points):
    """Independent O(n^2) oracle: same (squared distance, pair) semantics."""
    pts = sorted(map(tuple, points))
    best = None
    pair = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = dist2(pts[i], pts[j])
            if best is None or d < best:
                best = d
                pair = (pts[i], pts[j])
    return best, [list(pair[0]), list(pair[1])]


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
    # Hand case: (0,0)-(3,4) and (3,4)-(7,1) both have squared distance 25.
    hand = [(0, 0), (3, 4), (7, 1)]
    best, pair = closest_pair(hand)
    assert best == 25 and dist2(pair[0], pair[1]) == 25
    assert brute_force(hand)[0] == 25

    # Duplicate point: distance 0.
    dup = [(5, 5), (5, 5), (9, 9)]
    best, pair = closest_pair(dup)
    assert best == 0 and tuple(pair[0]) == (5, 5) and tuple(pair[1]) == (5, 5)
    assert brute_force(dup)[0] == 0

    # Two-point and grid cases.
    best, pair = closest_pair([(0, 0), (-3, -4)])
    assert best == 25
    grid = [(x, y) for x in range(4) for y in range(4)]
    best, pair = closest_pair(grid)
    assert best == 1 and dist2(pair[0], pair[1]) == 1
    assert brute_force(grid)[0] == 1

    # LCG cross-check: sizes up to ~60, sometimes duplicate-heavy coordinate ranges.
    seed = 271828
    for trial in range(30):
        seed = (seed * 75 + 74) % 65537
        size = 2 + seed % 59
        modulus = 10 + seed % 90
        points, seed = lcg_points(size, seed, modulus)
        best, pair = closest_pair(points)
        want_best, _ = brute_force(points)
        assert best == want_best, (trial, size, modulus)
        assert dist2(pair[0], pair[1]) == best, (trial, pair)
        for p in pair:
            assert tuple(p) in set(map(tuple, points)), (trial, p)

    # Concentrated points: many duplicates and very small distances.
    seed = 13579
    for trial in range(10):
        seed = (seed * 75 + 74) % 65537
        size = 2 + seed % 30
        points, seed = lcg_points(size, seed, 5)
        best, pair = closest_pair(points)
        assert best == brute_force(points)[0], (trial, points)

    # Worst-case shape for the strip: a vertical line (all x equal).
    line = [(7, y) for y in range(60)]
    best, _ = closest_pair(line)
    assert best == 1


if __name__ == "__main__":
    main()
    print("PASS")
