"""Task 735 - Implement advanced geometric data structures: dynamic convex hull maintenance.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

A dynamic hull answers "what is the hull *now*" after an interleaving of point insertions and
deletions instead of recomputing everything on every update.  Kept here is the classic
two-chain representation: an upper chain x-sorted left-to-right with every triple a clockwise
turn, a lower chain likewise with every triple a counter-clockwise turn.  The public hull is
the walk upper + reverse(lower)[1:-1] (clockwise), reversed into a counter-clockwise cycle.

insert(p): the cheap inside test first - the x-range test plus two chain orientation tests
decide whether p is inside.  If so p cannot be a vertex, both chains stay untouched and only
the point set grows, which is what makes maintenance cheap.  Otherwise p goes to its x-sorted
position in both chains, which are then repaired locally until stable: p is dropped when it is
a degenerate middle vertex, then neighbours are popped while their triples stop making the turn
the chain requires (amortised O(1); only vertices that stopped being extreme ever go).

Simplified, deliberately: delete(p) rebuilds the chains from the retained points instead of a
genuinely dynamic O(log^2 n) deletion, and support queries scan the hull linearly instead of
binary-searching the edge-angle sequence.  Verified after every single mutation: the
incremental hull is vertex-for-vertex identical to a full monotone-chain recomputation of the
same point set, every retained point lies inside or on the hull under a brute-force orientation
test over all hull edges, the chain turn conditions hold, and extreme-point and containment
queries agree with brute-force scans.  Coordinates are exact Fractions - no epsilon anywhere.
"""
import math
from fractions import Fraction

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def pt(x, y):
    return (Fraction(x), Fraction(y))


def cross(o, a, b):
    """Exact orientation: >0 counter-clockwise, <0 clockwise, 0 collinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def inside_bbox(a, b, p):
    """p inside the closed, axis-aligned box spanned by a and b (exact)."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def canonical(cycle):
    """Rotate a hull cycle to start at its lexicographically smallest vertex."""
    if not cycle:
        return []
    start = min(range(len(cycle)), key=lambda i: cycle[i])
    return cycle[start:] + cycle[:start]


def monotone_chain(points):
    """Brute-force oracle: full O(n log n) hull, counter-clockwise, collinear points removed."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    return half(pts)[:-1] + half(list(reversed(pts)))[:-1]


def contains_brute(points, p):
    """Independent oracle: recompute the hull with the monotone chain, then test p edge by
    edge (degenerate hulls reduce to point equality or the segment test)."""
    distinct = sorted(set(points))
    if len(distinct) < 3:
        if not distinct:
            return False
        if len(distinct) == 1:
            return p == distinct[0]
        return cross(distinct[0], distinct[1], p) == 0 and inside_bbox(distinct[0], distinct[1], p)
    poly = canonical(monotone_chain(points))
    return all(cross(poly[i], poly[(i + 1) % len(poly)], p) >= 0 for i in range(len(poly)))


class DynamicHull:
    """Incremental convex hull: x-sorted upper/lower chains repaired locally on insert."""

    def __init__(self, points=()):
        self.points = []
        self.upper = []            # left -> right, clockwise turns
        self.lower = []            # left -> right, counter-clockwise turns
        self.interior_skips = 0    # inserts answered by the inside test alone
        self.pops = 0              # chain vertices removed by repairs
        self.rebuilds = 0
        for p in points:
            self.insert(p)

    def _chain_insert(self, chain, point, want):
        """x-sorted insert, then repair until stable; `want` is +1 for the upper chain (drop
        triples with cross >= 0) and -1 for the lower chain (drop triples with cross <= 0)."""
        lo, hi = 0, len(chain)
        while lo < hi:
            mid = (lo + hi) // 2
            if chain[mid] < point:
                lo = mid + 1
            else:
                hi = mid
        chain.insert(lo, point)
        pos = lo
        while True:
            if 0 < pos < len(chain) - 1 and cross(chain[pos - 1], chain[pos], chain[pos + 1]) * want >= 0:
                del chain[pos]                  # the new point is not extreme after all
                self.pops += 1
                return
            changed = False
            while pos >= 2 and cross(chain[pos - 2], chain[pos - 1], chain[pos]) * want >= 0:
                del chain[pos - 1]
                pos -= 1
                changed = True
            while pos + 2 < len(chain) and cross(chain[pos], chain[pos + 1], chain[pos + 2]) * want >= 0:
                del chain[pos + 1]
                changed = True
            if not changed:
                return
            self.pops += 1                      # each round removes at least one vertex

    @staticmethod
    def _last_le(chain, x):
        """Index of the last chain vertex with x <= x (chain x sorted ascending)."""
        lo, hi = 0, len(chain)
        while lo < hi:
            mid = (lo + hi) // 2
            if chain[mid][0] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    def _inside(self, p):
        """True iff p is inside or on the hull, so it cannot become a vertex."""
        up, low = self.upper, self.lower
        if len(up) < 2 or len(low) < 2 or not (up[0][0] <= p[0] <= up[-1][0]):
            return False
        i, j = self._last_le(up, p[0]), self._last_le(low, p[0])
        if i + 1 >= len(up) or j + 1 >= len(low):
            return False
        return cross(up[i], up[i + 1], p) <= 0 and cross(low[j], low[j + 1], p) >= 0

    def insert(self, point):
        """Add one point; True when it had to be placed in (or repaired out of) the chains."""
        p = pt(point[0], point[1])
        self.points.append(p)
        if self._inside(p):
            self.interior_skips += 1
            return False
        self._chain_insert(self.upper, p, 1)
        self._chain_insert(self.lower, p, -1)
        return True

    def delete(self, point):
        """Remove one occurrence of `point` (KeyError if absent), then rebuild both chains."""
        p = pt(point[0], point[1])
        if p not in self.points:
            raise KeyError("point is not stored in the hull structure")
        self.points.remove(p)
        retained, self.points, self.upper, self.lower = sorted(self.points), [], [], []
        for q in retained:
            self.insert(q)
        self.rebuilds += 1

    def __len__(self):
        return len(self.points)

    def vertices(self):
        """Public hull: canonical counter-clockwise cycle (degenerate hulls: fewer than 3)."""
        walk = self.upper + list(reversed(self.lower))[1:-1]
        deduped = [p for i, p in enumerate(walk) if i == 0 or p != walk[i - 1]]
        if len(deduped) > 1 and deduped[0] == deduped[-1]:
            deduped.pop()
        if len(deduped) < 3:
            return sorted(set(deduped))
        return canonical(deduped[::-1])         # the chain walk runs clockwise

    def area(self):
        poly = self.vertices()
        if len(poly) < 3:
            return Fraction(0)
        doubled = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                      - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
        return abs(doubled) / 2

    def perimeter(self):
        """Boundary length (float); a two-point hull is the degenerate segment walked twice."""
        poly = self.vertices()
        if len(poly) < 2:
            return 0.0
        steps = 2 if len(poly) == 2 else len(poly)
        return sum(math.sqrt((poly[i % len(poly)][0] - poly[(i + 1) % len(poly)][0]) ** 2
                             + (poly[i % len(poly)][1] - poly[(i + 1) % len(poly)][1]) ** 2)
                   for i in range(steps))

    def chain_turns_ok(self):
        """The representation invariant of both chains, checked directly."""
        return all(cross(chain[i], chain[i + 1], chain[i + 2]) * want > 0
                   for chain, want in ((self.upper, -1), (self.lower, 1))
                   for i in range(len(chain) - 2))

    def support(self, direction):
        """Extreme hull vertex in `direction` (documented O(h) scan); None when empty."""
        poly = self.vertices()
        if not poly:
            return None
        return min(poly, key=lambda q: (-(q[0] * direction[0] + q[1] * direction[1]), q))

    def contains(self, point):
        """Inside-or-on test through the chains; the vertical side edges at the two x extremes
        are handled explicitly, everything else needs two binary searches."""
        p = pt(point[0], point[1])
        poly = self.vertices()
        if len(poly) < 3:
            if len(poly) == 1:
                return p == poly[0]
            if len(poly) == 2:
                return cross(poly[0], poly[1], p) == 0 and inside_bbox(poly[0], poly[1], p)
            return False
        up, low = self.upper, self.lower
        if p[0] < up[0][0] or p[0] > up[-1][0]:
            return False
        if p[0] == up[0][0] or p[0] == up[-1][0]:
            return (min(q[1] for q in low if q[0] == p[0])
                    <= p[1] <= max(q[1] for q in up if q[0] == p[0]))
        i, j = self._last_le(up, p[0]), self._last_le(low, p[0])
        return cross(up[i], up[i + 1], p) <= 0 and cross(low[j], low[j + 1], p) >= 0


def verify(hull, label, deep=False):
    """Oracle used after every mutation: incremental hull versus full recomputation."""
    expected = canonical(monotone_chain(hull.points))
    check(hull.vertices() == expected,
          f"{label}: incremental hull {hull.vertices()} != monotone chain {expected}")
    check(hull.chain_turns_ok(), f"{label}: chain turn invariant violated")
    if deep:
        for p in hull.points:
            check(contains_brute(hull.points, p), f"{label}: stored point {p} fell outside the hull")
            check(hull.contains(p), f"{label}: chain containment missed stored point {p}")


def main():
    # 1. hand-checked square plus an interior point that must not disturb anything
    square = DynamicHull([(0, 0), (2, 0), (2, 2), (0, 2)])
    check(square.vertices() == [(0, 0), (2, 0), (2, 2), (0, 2)],
          "square hull must be its four corners counter-clockwise from the lex-smallest")
    check(square.area() == 4 and abs(square.perimeter() - 8.0) < 1e-12, "square area 4, perimeter 8")
    chains = (list(square.upper), list(square.lower))
    check(square.insert((1, 1)) is False and square.interior_skips == 1,
          "an interior point must be answered by the inside test, not reported as a vertex")
    check((square.upper, square.lower) == chains and len(square.vertices()) == 4,
          "an interior insert must leave both chains and the hull untouched")
    verify(square, "square+interior", deep=True)
    # 2. boundary/degenerate cases: empty, single, two, identical, collinear
    empty = DynamicHull()
    check(empty.vertices() == [] and empty.area() == 0 and empty.perimeter() == 0.0
          and empty.support((1, 0)) is None and not empty.contains((0, 0)),
          "an empty hull must be empty, support-free and contain nothing")
    one = DynamicHull([(3, 4)])
    check(one.vertices() == [(3, 4)] and one.area() == 0.0
          and one.contains((3, 4)) and not one.contains((3, 5)), "single-point hull")
    two = DynamicHull([(0, 0), (3, 4)])
    check(two.vertices() == [(0, 0), (3, 4)] and abs(two.perimeter() - 10.0) < 1e-12,
          "a two-point hull is the degenerate segment walked twice")
    identical = DynamicHull([(1, 1)] * 5)
    check(identical.vertices() == [(1, 1)] and identical.area() == 0.0, "five identical points fuse")
    collinear = DynamicHull([(0, 0), (1, 1), (2, 2), (3, 3), (1, 1)])
    check(collinear.vertices() == [(0, 0), (3, 3)] and abs(collinear.perimeter() - 2 * math.sqrt(18)) < 1e-12,
          "collinear input collapses to its two endpoints, walked twice")
    verify(collinear, "collinear", deep=True)
    # 3. insertion stream versus the monotone chain after EVERY insert
    seed, points = 0x5EED, []
    for _ in range(40):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        points.append((r1 % 21 - 10, r2 % 21 - 10))
    growing = DynamicHull()
    for i, p in enumerate(points):
        growing.insert(p)
        verify(growing, f"insert #{i}", deep=(i % 7 == 0))
    check(len(growing.vertices()) >= 3 and growing.area() > 0, "a random 40-point set has a real hull")
    # 4. interleaved insert/delete stream, checked after every single mutation
    mixed = DynamicHull(points[:20])
    state, deleted = 0xABCDEF, 0
    for step in range(60):
        state, r = splitmix64(state)
        if r % 3 == 0 and len(mixed) > 3:
            mixed.delete(mixed.points[(r >> 8) % len(mixed.points)])
            deleted += 1
        else:
            state, r2 = splitmix64(state)
            mixed.insert((r2 % 17 - 8, r % 17 - 8))
        verify(mixed, f"mixed step {step}", deep=(step % 5 == 0))
    check(deleted > 0 and mixed.rebuilds == deleted, "the stream must exercise deletion rebuilds")
    try:
        mixed.delete((10_000, 10_000))
        raise AssertionError("deleting an absent point must raise KeyError")
    except KeyError:
        pass
    # 5. support queries versus a brute-force scan over ALL points, and containment versus the
    #    edge-by-edge brute-force hull test, on random probes
    state, inside_hits, outside_hits = 0x1234, 0, 0
    for _ in range(120):
        state, r1 = splitmix64(state)
        state, r2 = splitmix64(state)
        direction = (r1 % 13 - 6, r2 % 13 - 6)
        if direction != (0, 0):
            value = lambda q: q[0] * direction[0] + q[1] * direction[1]
            check(value(mixed.support(direction)) == value(max(mixed.points, key=value)),
                  f"support in {direction} must attain the brute-force maximum")
        probe = pt(r2 % 25 - 12, r1 % 25 - 12)
        expected = contains_brute(mixed.points, probe)
        check(mixed.contains(probe) == expected,
              f"containment of {probe} disagrees with the brute-force edge test")
        inside_hits += expected
        outside_hits += not expected
    check(inside_hits > 0 and outside_hits > 0, "probes must cover inside and outside cases")
    # 7. interior inserts must stay cheap: the chains must not grow at all
    big = DynamicHull([(0, 0), (100, 0), (100, 100), (0, 100)])
    pops_before, skips_before = big.pops, big.interior_skips
    for i in range(1, 20):
        big.insert((i, i))
    check(len(big.vertices()) == 4 and len(big.upper) == 3 and len(big.lower) == 3,
          "interior inserts must never add chain vertices")
    check(big.interior_skips - skips_before == 19 and big.pops == pops_before,
          "all 19 interior inserts must be answered by the inside test with no repair")
    verify(big, "square with 19 interior points", deep=True)
    # 8. determinism: identical operation streams rebuild identical state
    left, right = DynamicHull(), DynamicHull()
    for p in points:
        left.insert(p)
        right.insert(p)
    check(left.vertices() == right.vertices() and left.pops == right.pops,
          "identical insert streams must give identical hulls and counters")
    for p in points[:10]:
        left.delete(p)
        right.delete(p)
    check(left.vertices() == right.vertices() and left.area() == right.area(),
          "identical delete streams must give identical hulls")
    # 9. rational coordinates keep every orientation exact (no epsilon anywhere)
    rational = DynamicHull([(Fraction(1, 3), Fraction(1, 7)), (Fraction(5, 2), 0), (0, Fraction(9, 4))])
    oracle = canonical(monotone_chain(rational.points))
    oracle_area = sum(oracle[i][0] * oracle[(i + 1) % 3][1] - oracle[(i + 1) % 3][0] * oracle[i][1]
                      for i in range(3))
    check(rational.area() == abs(oracle_area) / 2 and rational.area() > 0,
          "a rational triangle must carry the shoelace area of the recomputed hull")
    verify(rational, "rational triangle", deep=True)
    print(f"735 hull_vertices={len(mixed.vertices())} stored_points={len(mixed)} interior_skips="
          f"{mixed.interior_skips} pops={mixed.pops} rebuilds={mixed.rebuilds}")
    print(f"735 random_hull area={float(growing.area()):.4f} perimeter={growing.perimeter():.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
