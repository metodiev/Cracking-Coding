"""Task 371 - Line-segment intersection detection (Bentley-Ottmann sweep, outline). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Detect all intersecting pairs among n segments.  The full Bentley-Ottmann algorithm
runs in O((n + k) log n) for k intersection points; this task implements the sweep
* skeleton* (event queue + status structure) and handles the small cases this repo
tests (n <= ~12) exactly, plus a plain O(n^2) fallback used as the oracle.

Bentley-Ottmann invariant (the full algorithm this skeleton approximates):
  1. Events: every segment endpoint, ordered lexicographically by (x, y).  A left
     endpoint inserts the segment into the status structure; a right endpoint
     removes it.  Intersection points found during the sweep are also pushed as
     events so that the order of the status structure is restored after a crossing.
  2. Status structure: the segments currently crossing the vertical sweep line,
     kept ordered by their y coordinate at the current sweep x.  In general
     position no two of them cross between events, so the order is well defined
     and stable there.
  3. Crossing detection: only neighbours in the status can become adjacent-and-
     swapped between events, so a crossing is reported when two segments are
     adjacent in the status (on insertion/removal, and after processing a found
     intersection event) and their exact intersection test succeeds.  Such a
     found intersection is reported *once*, at the first event x where the two
     segments become adjacent.
  4. Degeneracies (documented but only partially handled here): vertical
     segments (processed entirely at one event x), several events with equal x,
     endpoint touches, collinear overlaps and overlapping collinear bundles.
     The textbook completion of the algorithm keeps the status structure
     consistent for all of them; this outline instead detects degeneracies by
     the exact predicate below and falls back to checking every pair that is
     simultaneously active at an event x.

Small-case exact handling used here: at every event x, after all left endpoints
of that x have been inserted and finished segments dropped, every *pair* of
segments in the status structure is tested with the exact integer predicate.
Because an intersecting pair always spans the event x = max(left_x1, left_x2)
(the later segment's left endpoint), every intersecting pair is seen at least
once, duplicate reports collapse into a set, and the count matches the O(n^2)
oracle exactly.  The status structure is still maintained in exact y-order at
each event x to mirror the real Bentley-Ottmann skeleton.

All predicates use exact integer arithmetic (cross products); no floating point
is involved anywhere in this task.
"""


def orient(p, q, r):
    """Twice the signed area of triangle (p, q, r): >0 CCW, <0 CW, 0 collinear."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def on_segment(p, q, r):
    """True when collinear point q lies inside segment pr (endpoints included)."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def segments_intersect(s, t):
    """Exact closed-segment intersection test (touches and overlaps count)."""
    (p1, p2), (p3, p4) = s, t
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True                                           # proper crossing
    if o1 == 0 and on_segment(p1, p3, p2):
        return True                                           # p3 touches s
    if o2 == 0 and on_segment(p1, p4, p2):
        return True                                           # p4 touches s
    if o3 == 0 and on_segment(p3, p1, p4):
        return True                                           # p1 touches t
    if o4 == 0 and on_segment(p3, p2, p4):
        return True                                           # p2 touches t
    return False


def y_at(segment, x):
    """Exact y of the segment at sweep x as an integer fraction (numerator, denominator)."""
    (x1, y1), (x2, y2) = segment
    if x2 == x1:
        return (y1, 1)
    return (y1 * (x2 - x1) + (y2 - y1) * (x - x1), x2 - x1)


def compare_y(segment_a, segment_b, x):
    """-1/0/1 comparing the segments' y values at the sweep line x (exact fractions)."""
    na, da = y_at(segment_a, x)
    nb, db = y_at(segment_b, x)
    left, right = na * db, nb * da
    return -1 if left < right else (1 if left > right else 0)


def sweep_intersection_pairs(raw_segments):
    """Sweep skeleton: event queue + status structure; returns the intersecting pairs."""
    segments = []
    for a, b in raw_segments:
        segments.append((a, b) if a <= b else (b, a))         # left endpoint first

    events = []
    for idx, (p, q) in enumerate(segments):
        events.append((p[0], p[1], idx, 0))                   # 0 = left endpoint
        events.append((q[0], q[1], idx, 1))                   # 1 = right endpoint
    events.sort()

    status = []                                               # active ids, y-ordered
    found = set()
    i = 0
    while i < len(events):
        x = events[i][0]
        j = i
        while j < len(events) and events[j][0] == x:
            _, _, idx, kind = events[j]
            if kind == 0:                                     # insert, keeping y-order
                pos = 0
                while pos < len(status) and compare_y(segments[status[pos]], segments[idx], x) <= 0:
                    pos += 1
                status.insert(pos, idx)
            j += 1
        status = [k for k in status if segments[k][1][0] >= x]     # finish segments
        for a in range(len(status)):
            for b in range(a + 1, len(status)):
                if segments_intersect(segments[status[a]], segments[status[b]]):
                    low, high = sorted((status[a], status[b]))
                    found.add((low, high))
        i = j
    return found


def brute_force_intersection_pairs(segments):
    """O(n^2) oracle: test every pair with the same exact predicate."""
    found = set()
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i], segments[j]):
                found.add((i, j))
    return found


def lcg_segments(count, seed):
    """count pseudo-random segments with integer coordinates in [0, 40], endpoints distinct."""
    segments = []
    for _ in range(count):
        while True:
            coords = []
            for _ in range(4):
                seed = (seed * 75 + 74) % 65537
                coords.append(seed % 41)
            if (coords[0], coords[1]) != (coords[2], coords[3]):
                break
        segments.append(((coords[0], coords[1]), (coords[2], coords[3])))
    return segments, seed


def main():
    # --- hand predicates -------------------------------------------------
    assert segments_intersect(((0, 0), (4, 4)), ((0, 4), (4, 0)))          # proper crossing
    assert segments_intersect(((0, 0), (2, 2)), ((2, 2), (4, 0)))          # shared endpoint
    assert segments_intersect(((0, 0), (4, 0)), ((2, 0), (6, 0)))          # collinear overlap
    assert not segments_intersect(((0, 0), (2, 0)), ((0, 1), (2, 1)))      # parallel disjoint
    assert segments_intersect(((0, 0), (4, 0)), ((2, 0), (2, 3)))          # T-touch
    assert not segments_intersect(((0, 0), (1, 1)), ((3, 0), (4, 1)))      # separated
    assert segments_intersect(((0, 0), (1, 1)), ((1, 1), (2, 0)))          # endpoint meet

    # --- hand sweep set: 4 intersecting pairs ----------------------------
    hand = [((0, 0), (4, 4)), ((0, 4), (4, 0)), ((2, 2), (6, 2)),
            ((6, 0), (6, 6)), ((8, 8), (9, 9))]
    expected = {(0, 1), (0, 2), (1, 2), (2, 3)}
    assert sweep_intersection_pairs(hand) == expected
    assert brute_force_intersection_pairs(hand) == expected
    assert len(expected) == 4

    # --- collinear / touch family ----------------------------------------
    collinear = [((0, 0), (4, 0)), ((2, 0), (6, 0)), ((0, 1), (6, 1)), ((0, 0), (0, 1))]
    assert sweep_intersection_pairs(collinear) == {(0, 1), (0, 3), (2, 3)}
    assert sweep_intersection_pairs(collinear) == brute_force_intersection_pairs(collinear)

    # --- no intersections at all ------------------------------------------
    disjoint = [((0, 0), (1, 1)), ((3, 0), (4, 1)), ((5, 5), (6, 6))]
    assert sweep_intersection_pairs(disjoint) == set()
    assert sweep_intersection_pairs([]) == set()

    # --- LCG cross-checks against the O(n^2) oracle -----------------------
    seed = 37113
    for trial in range(40):
        count = 5 + seed % 8
        segments, seed = lcg_segments(count, seed)
        swept = sweep_intersection_pairs(segments)
        oracle = brute_force_intersection_pairs(segments)
        assert swept == oracle, (trial, segments)
    assert seed != 37113


if __name__ == "__main__":
    main()
    print("PASS")
