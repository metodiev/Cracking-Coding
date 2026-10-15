"""Task 370 - Line sweep for rectangle area union. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Sweep a vertical line over the sorted rectangle x edges.  Between two consecutive
distinct x values the covered y-length is constant, so the union area is the sum of
measure(x) * (next_x - x).  The measure comes from a segment tree over the compressed y
coordinates whose nodes store how many active rectangles fully cover them (count) and
the covered length of their span (length): an elementary interval has length 0 while its
count is 0, an internal node uses its children's lengths when count == 0, and any node
with count > 0 reports its full span.  Counts, not just booleans, are what makes
overlaps of three or more rectangles and exact touching edges work.  The result is
asserted on canonical examples and cross-checked against a coordinate-compression grid
oracle on deterministic LCG inputs (including zero-area and touching rectangles).
"""


class CoverLengthTree:
    """Segment tree over compressed y edges: covered length of the active y-intervals."""

    def __init__(self, ys):
        self.ys = ys                                      # sorted unique y coordinates
        self.m = len(ys) - 1                              # elementary intervals
        self.count = [0] * (4 * self.m) if self.m else []
        self.length = [0] * (4 * self.m) if self.m else []

    def range_add(self, y_lo, y_hi, delta):
        """Adds delta over elementary interval indices [y_lo, y_hi)."""
        if self.m == 0 or y_lo >= y_hi:
            return
        self._update(1, 0, self.m, y_lo, y_hi, delta)

    def _update(self, node, lo, hi, ql, qr, delta):
        if qr <= lo or hi <= ql:
            return
        if ql <= lo and hi <= qr:
            self.count[node] += delta
        else:
            mid = (lo + hi) // 2
            self._update(2 * node, lo, mid, ql, qr, delta)
            self._update(2 * node + 1, mid, hi, ql, qr, delta)
        if self.count[node] > 0:
            self.length[node] = self.ys[hi] - self.ys[lo]
        elif hi - lo == 1:
            self.length[node] = 0
        else:
            self.length[node] = self.length[2 * node] + self.length[2 * node + 1]

    def measure(self):
        return self.length[1] if self.m else 0


def union_area(rectangles):
    """Area of the union of axis-aligned rectangles [x1, y1, x2, y2]."""
    events = []
    ys = set()
    for x1, y1, x2, y2 in rectangles:
        if x1 >= x2 or y1 >= y2:
            continue                                      # zero/negative area: ignored
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))
        ys.add(y1)
        ys.add(y2)
    if not events:
        return 0
    ordered_ys = sorted(ys)
    index = {y: i for i, y in enumerate(ordered_ys)}
    tree = CoverLengthTree(ordered_ys)
    events.sort(key=lambda e: (e[0], e[1]))               # at equal x, removals first
    area = 0
    prev_x = events[0][0]
    i = 0
    while i < len(events):
        x = events[i][0]
        area += tree.measure() * (x - prev_x)
        while i < len(events) and events[i][0] == x:
            _, delta, y1, y2 = events[i]
            tree.range_add(index[y1], index[y2], delta)
            i += 1
        prev_x = x
    return area


def union_area_oracle(rectangles):
    """Independent oracle: sum the fully covered cells of the compressed coordinate grid."""
    rects = [(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles if x1 < x2 and y1 < y2]
    if not rects:
        return 0
    xs = sorted({v for r in rects for v in (r[0], r[2])})
    ys = sorted({v for r in rects for v in (r[1], r[3])})
    area = 0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = xs[i], ys[j]
            for x1, y1, x2, y2 in rects:
                if x1 <= cx and x2 >= xs[i + 1] and y1 <= cy and y2 >= ys[j + 1]:
                    area += (xs[i + 1] - cx) * (ys[j + 1] - cy)
                    break
    return area


def lcg_rectangles(count, seed, modulus):
    rectangles = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        x1 = seed % modulus
        seed = (seed * 75 + 74) % 65537
        width = seed % (modulus // 2)
        seed = (seed * 75 + 74) % 65537
        y1 = seed % modulus
        seed = (seed * 75 + 74) % 65537
        height = seed % (modulus // 2)
        rectangles.append((x1, y1, x1 + width, y1 + height))
    return rectangles, seed


def main():
    # Canonical example: [0,0,2,2] and [1,1,3,3] overlap in a 1x1 square -> 4 + 4 - 1 = 7.
    canonical = [[0, 0, 2, 2], [1, 1, 3, 3]]
    assert union_area(canonical) == 7
    assert union_area_oracle(canonical) == 7

    # Disjoint rectangles: areas simply add up.
    assert union_area([[0, 0, 1, 1], [2, 2, 3, 3]]) == 1 + 1 == 2
    # Nested rectangles: the outer one wins.
    assert union_area([[0, 0, 10, 10], [2, 2, 4, 4]]) == 100
    # Identical rectangles: counted once.
    assert union_area([[1, 1, 3, 3], [1, 1, 3, 3]]) == 4
    # Touching rectangles share an edge / a corner but do not overlap.
    assert union_area([[0, 0, 2, 2], [2, 0, 4, 2]]) == 8
    assert union_area([[0, 0, 1, 1], [1, 1, 2, 2]]) == 2
    # Zero-area rectangles contribute nothing.
    assert union_area([[5, 5, 5, 9]]) == 0
    assert union_area([[5, 5, 5, 5]]) == 0
    assert union_area([[0, 0, 3, 3], [1, 1, 1, 4]]) == 9
    assert union_area([]) == 0
    assert union_area([[2, 3, 7, 9]]) == 30
    assert union_area([[0, 0, 4, 4], [0, 0, 4, 4], [1, 1, 3, 3]]) == 16

    # Triple overlap: y in [0,1) spans x 0..4 (length 4), y in [1,3) also 4, y in [3,4) 3.
    triple = [[0, 0, 3, 3], [1, 0, 4, 3], [0, 1, 3, 4]]
    assert union_area(triple) == 15
    assert union_area_oracle(triple) == 15

    # LCG cross-check, including degenerate rectangles and negative coordinates.
    seed = 1618
    for trial in range(40):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 8
        modulus = 8 + seed % 10
        rectangles, seed = lcg_rectangles(count, seed, modulus)
        offset = -(seed % 7)
        shifted = [(x1 + offset, y1 + offset, x2 + offset, y2 + offset)
                   for x1, y1, x2, y2 in rectangles]
        assert union_area(shifted) == union_area_oracle(shifted), (trial, shifted)


if __name__ == "__main__":
    main()
    print("PASS")
