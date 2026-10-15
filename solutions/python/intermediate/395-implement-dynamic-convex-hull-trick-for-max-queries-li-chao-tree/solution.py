"""Task 395 - Implement the dynamic convex hull trick for max queries (Li Chao tree). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

A Li Chao segment tree stores one line per node of an x-interval tree. Inserting line y=mx+b
keeps the better line at the interval mid and recurses into the only half where the other
line can still win (two lines cross at most once). A query walks root -> leaf and takes the
maximum over the stored lines. Insert O(log X), query O(log X), fully dynamic.
Ties keep the previously stored line (strict > comparisons), which is documented behaviour.
"""

class LiChao:
    def __init__(self, x_lo, x_hi):
        size = 4 * (x_hi - x_lo + 1)
        self.x_lo, self.x_hi = x_lo, x_hi
        self.has = [False] * size
        self.m = [0] * size
        self.b = [0] * size

    def add_line(self, m, b):
        self._add(1, self.x_lo, self.x_hi, m, b)

    def _add(self, node, lo, hi, m, b):
        if not self.has[node]:
            self.has[node] = True
            self.m[node], self.b[node] = m, b
            return
        mid = (lo + hi) // 2
        if m * mid + b > self.m[node] * mid + self.b[node]:
            m, self.m[node] = self.m[node], m
            b, self.b[node] = self.b[node], b
        if lo == hi:
            return
        if m * lo + b > self.m[node] * lo + self.b[node]:
            self._add(2 * node, lo, mid, m, b)
        elif m * hi + b > self.m[node] * hi + self.b[node]:
            self._add(2 * node + 1, mid + 1, hi, m, b)

    def query(self, x):
        node, lo, hi = 1, self.x_lo, self.x_hi
        best = None
        while True:
            if self.has[node]:
                value = self.m[node] * x + self.b[node]
                best = value if best is None else max(best, value)
            if lo == hi:
                break
            mid = (lo + hi) // 2
            if x <= mid:
                node, hi = 2 * node, mid
            else:
                node, lo = 2 * node + 1, mid + 1
        if best is None:
            raise ValueError("query on an empty tree")
        return best


def brute_force(lines, x):
    return max(m * x + b for m, b in lines)


def lcg_lines(count, seed, m_range=50, b_range=1000):
    lines = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        m = seed % (2 * m_range + 1) - m_range
        seed = (seed * 75 + 74) % 65537
        b = seed % (2 * b_range + 1) - b_range
        lines.append((m, b))
    return lines


def main():
    tree = LiChao(-500, 500)
    for m, b in [(1, 0), (-1, 0), (0, 0)]:
        tree.add_line(m, b)
    assert tree.query(5) == 5
    assert tree.query(-5) == 5
    assert tree.query(0) == 0
    assert tree.query(3) == 3

    cross = LiChao(-500, 500)
    cross.add_line(2, 1)
    cross.add_line(-3, 10)
    assert cross.query(2) == 5          # 2*2+1 beats -3*2+10
    assert cross.query(1) == 7          # -3+10 beats 2+1
    assert cross.query(0) == 10
    assert cross.query(5) == 11

    fixed = [(1, 0), (-1, 3), (0, -2), (5, -10), (-4, 6)]
    checker = LiChao(-20, 20)
    for m, b in fixed:
        checker.add_line(m, b)
    for x in range(-20, 21):
        assert checker.query(x) == brute_force(fixed, x), x

    lines = lcg_lines(200, seed=4242)
    big = LiChao(-500, 500)
    for m, b in lines:
        big.add_line(m, b)
    for step in range(101):
        x = -500 + step * 10
        assert big.query(x) == brute_force(lines, x), (x, big.query(x), brute_force(lines, x))

    # duplicate lines are harmless
    duplicate = LiChao(-10, 10)
    for _ in range(4):
        duplicate.add_line(3, -1)
    assert duplicate.query(7) == 20

    empty = LiChao(-10, 10)
    try:
        empty.query(0)
        raise AssertionError("expected a ValueError for an empty tree")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
