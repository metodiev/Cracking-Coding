"""Task 418 - Implement online convex hull maintenance (dynamic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Online convex hull trick supporting BOTH signs and BOTH insertion orders:
  * mode 'max' maintains the upper envelope, mode 'min' the lower envelope,
  * order 'inc' inserts lines with non-decreasing slopes, order 'dec' with non-increasing
    ones — both are the orders produced by sorting candidate lines by slope.
The hull is kept as arrays (ms, bs) of lines whose breakpoints increase. When appending
(m, b) after (m1, b1), (m2, b2) the middle line m2 is unnecessary exactly when
    sign * [(b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1)] >= 0,
where sign = +1 for (max, inc) / (min, dec) and -1 for (max, dec) / (min, inc); the
condition compares the two intersection abscissas without ever dividing. Duplicate
slopes keep only the best intercept (larger for max, smaller for min). Queries are
unimodal over the hull, so an O(log n) adjacent-pair binary search finds the peak
(for max) or the valley (for min). Deletions are out of scope: this is add-only.
Everything is exact integer arithmetic — no floating point intersections.
"""


class ConvexHullTrick:
    def __init__(self, mode="max", order="inc"):
        if mode not in ("max", "min") or order not in ("inc", "dec"):
            raise ValueError("mode must be max/min and order inc/dec")
        self.mode = mode
        self.order = order
        self.ms = []
        self.bs = []
        self.sign = (1 if mode == "max" else -1) * (1 if order == "inc" else -1)

    def __len__(self):
        return len(self.ms)

    def add_line(self, m, b):
        if self.ms and m == self.ms[-1]:
            better = b > self.bs[-1] if self.mode == "max" else b < self.bs[-1]
            if not better:
                return
            self.ms.pop()
            self.bs.pop()
        while len(self.ms) >= 2:
            m1, b1 = self.ms[-2], self.bs[-2]
            m2, b2 = self.ms[-1], self.bs[-1]
            expr = (b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1)
            if self.sign * expr >= 0:
                self.ms.pop()
                self.bs.pop()
            else:
                break
        self.ms.append(m)
        self.bs.append(b)

    def query(self, x):
        if not self.ms:
            raise ValueError("query on an empty hull")
        lo, hi = 0, len(self.ms) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            v_mid = self.ms[mid] * x + self.bs[mid]
            v_next = self.ms[mid + 1] * x + self.bs[mid + 1]
            if self.mode == "max":
                if v_mid <= v_next:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                if v_mid >= v_next:
                    lo = mid + 1
                else:
                    hi = mid
        return self.ms[lo] * x + self.bs[lo]


class Lcg:
    def __init__(self, state):
        self.state = state

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def brute_best(lines, x, mode):
    values = [m * x + b for m, b in lines]
    return max(values) if mode == "max" else min(values)


def fixed_case(mode, order):
    hull = ConvexHullTrick(mode, order)
    fixed = [(1, 0), (-1, 3), (0, -2), (5, -10), (-4, 6)]
    fixed.sort(key=lambda mb: mb[0], reverse=(order == "dec"))
    for m, b in fixed:
        hull.add_line(m, b)
    for x in range(-20, 21):
        assert hull.query(x) == brute_best(fixed, x, mode), (mode, order, x)
    return hull


def main():
    # hand case: lines y = x, y = 2x + 3, y = 4x + 5 (max): 5 at 0, 45 at 10, -10 at -10
    hull = ConvexHullTrick("max", "inc")
    for m, b in [(1, 0), (2, 3), (4, 5)]:
        hull.add_line(m, b)
    assert hull.query(0) == 5
    assert hull.query(10) == 45
    assert hull.query(-10) == -10
    # the same lines inserted from the other end of the slope order agree everywhere
    dec = ConvexHullTrick("max", "dec")
    for m, b in [(4, 5), (2, 3), (1, 0)]:
        dec.add_line(m, b)
    for x in range(-10, 11):
        assert dec.query(x) == hull.query(x), x

    # hand case for min: y = 2x + 5, y = x + 1: 1 at 0, 11 at 10, -15 at -10
    mn = ConvexHullTrick("min", "inc")
    mn.add_line(2, 5)
    mn.add_line(1, 1)
    assert mn.query(0) == 1 and mn.query(10) == 11 and mn.query(-10) == -15
    mn_dec = ConvexHullTrick("min", "dec")
    mn_dec.add_line(1, 1)
    mn_dec.add_line(2, 5)
    for x in range(-10, 11):
        assert mn_dec.query(x) == mn.query(x), x

    # duplicate slopes keep only the best intercept: max keeps b = -1 (value 20 at x = 7)
    dup = ConvexHullTrick("max", "inc")
    for _ in range(4):
        dup.add_line(3, -1)
    assert dup.query(7) == 20 and len(dup) == 1
    dup2 = ConvexHullTrick("min", "inc")
    dup2.add_line(3, 10)
    dup2.add_line(3, 5)
    assert dup2.query(7) == 26 and len(dup2) == 1
    dup3 = ConvexHullTrick("max", "inc")
    dup3.add_line(3, -1)
    dup3.add_line(3, -2)          # same slope, worse intercept is discarded
    assert len(dup3) == 1 and dup3.query(7) == 20

    # fixed-line oracle in all four mode/order combinations
    for mode in ("max", "min"):
        for order in ("inc", "dec"):
            fixed_case(mode, order)

    # interleaved add/query scripts from the deterministic LCG, all four combinations
    for index, (mode, order) in enumerate(
            [("max", "inc"), ("max", "dec"), ("min", "inc"), ("min", "dec")]):
        gen = Lcg(4180 + index)
        raw = []
        for _ in range(60):
            m = gen.next() % 141 - 70
            b = gen.next() % 4001 - 2000
            raw.append((m, b))
        raw.sort(key=lambda mb: mb[0], reverse=(order == "dec"))
        hull = ConvexHullTrick(mode, order)
        added = []
        for k, (m, b) in enumerate(raw):
            hull.add_line(m, b)
            added.append((m, b))
            assert len(hull) <= len(added)
            for (hm, hb) in zip(hull.ms, hull.bs):
                assert (hm, hb) in added, (mode, order, hm, hb)
            if k % 3 == 2 or k == len(raw) - 1:
                for _ in range(3):
                    x = gen.next() % 2001 - 1000
                    assert hull.query(x) == brute_best(added, x, mode), (mode, order, k, x)
        # full sweep once the whole envelope is in place
        for x in range(-200, 201):
            assert hull.query(x) == brute_best(raw, x, mode), (mode, order, x)
        # hull line count cannot exceed the number of distinct slopes
        distinct = len(set(m for m, _ in raw))
        assert len(hull) <= distinct

    # querying before the first insertion is rejected
    try:
        ConvexHullTrick("max", "inc").query(3)
        raise AssertionError("expected a ValueError for an empty hull")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
