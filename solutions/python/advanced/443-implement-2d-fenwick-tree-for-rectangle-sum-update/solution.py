"""Task 443 - Implement 2D Fenwick tree for rectangle sum/update. (Tier: Advanced, CTCI Ch 4)

Two 2D Fenwick trees (BIT of BITs). `Fenwick2D` answers point add + rectangle sum in
O(log n log m) with four 2D prefix queries. `DiffFenwick2D` stores the 2D difference array,
so rectangle add is four corner updates and a point query is a single 2D prefix sum (the 2D
difference trick); this one is the classic way to run many rectangle adds against point asks.
Both structures are checked against a directly maintained brute-force grid.
"""


class Fenwick2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def add(self, row, col, delta):
        i = row + 1
        while i <= self.rows:
            j = col + 1
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i

    def _prefix(self, row, col):
        """Sum of the rectangle (0,0)..(row,col); callers only pass non-negative corners."""
        total = 0
        i = row + 1
        while i > 0:
            j = col + 1
            while j > 0:
                total += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return total

    def rect_sum(self, r1, c1, r2, c2):
        return (self._prefix(r2, c2) - self._prefix(r1 - 1, c2)
                - self._prefix(r2, c1 - 1) + self._prefix(r1 - 1, c1 - 1))


class DiffFenwick2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.inner = Fenwick2D(rows, cols)

    def rect_add(self, r1, c1, r2, c2, delta):
        self.inner.add(r1, c1, delta)
        if r2 + 1 < self.rows:
            self.inner.add(r2 + 1, c1, -delta)
        if c2 + 1 < self.cols:
            self.inner.add(r1, c2 + 1, -delta)
        if r2 + 1 < self.rows and c2 + 1 < self.cols:
            self.inner.add(r2 + 1, c2 + 1, delta)

    def point(self, row, col):
        return self.inner._prefix(row, col)


def lcg_grid_ops(count, seed, n):
    """Op script: kind 0 uses (r1,c1) as a point and `value` as a delta / new value;
    kind 1 uses (r1,c1)..(r2,c2) as the query rectangle."""
    ops = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        r1 = seed % n
        seed = (seed * 75 + 74) % 65537
        c1 = seed % n
        seed = (seed * 75 + 74) % 65537
        r2 = r1 + seed % (n - r1)
        seed = (seed * 75 + 74) % 65537
        c2 = c1 + seed % (n - c1)
        seed = (seed * 75 + 74) % 65537
        kind = seed % 2
        seed = (seed * 75 + 74) % 65537
        value = seed % 21 - 10
        ops.append((kind, r1, c1, r2, c2, value))
    return ops


def brute_rect(grid, r1, c1, r2, c2):
    return sum(grid[r][c] for r in range(r1, r2 + 1) for c in range(c1, c2 + 1))


def main():
    # Hand case: 4x4 point-add / rectangle-sum tree.
    bit = Fenwick2D(4, 4)
    bit.add(0, 0, 1)
    bit.add(1, 2, 5)
    bit.add(3, 3, 2)
    assert bit.rect_sum(0, 0, 3, 3) == 8
    assert bit.rect_sum(1, 2, 1, 2) == 5
    assert bit.rect_sum(0, 1, 2, 1) == 0
    assert bit.rect_sum(0, 0, 0, 0) == 1
    assert bit.rect_sum(1, 2, 3, 3) == 7

    # Hand case: 4x4 difference tree (rectangle add / point query).
    diff = DiffFenwick2D(4, 4)
    diff.rect_add(0, 0, 1, 1, 3)
    assert diff.point(0, 0) == 3
    assert diff.point(1, 1) == 3
    assert diff.point(1, 2) == 0
    assert diff.point(2, 2) == 0
    diff.rect_add(1, 1, 2, 2, 10)
    assert diff.point(1, 1) == 13
    assert diff.point(2, 2) == 10
    assert diff.point(3, 3) == 0
    assert diff.point(0, 0) == 3

    n = 16
    ops = lcg_grid_ops(300, 443, n)

    bit = Fenwick2D(n, n)
    grid = [[0] * n for _ in range(n)]
    for kind, r1, c1, r2, c2, value in ops:
        if kind == 0:
            bit.add(r1, c1, value)
            grid[r1][c1] += value
        else:
            assert bit.rect_sum(r1, c1, r2, c2) == brute_rect(grid, r1, c1, r2, c2)

    diff = DiffFenwick2D(n, n)
    grid = [[0] * n for _ in range(n)]
    for kind, r1, c1, r2, c2, value in ops:
        if kind == 0:
            diff.rect_add(r1, c1, r2, c2, value)
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    grid[r][c] += value
        else:
            assert diff.point(r1, c1) == grid[r1][c1]


if __name__ == "__main__":
    main()
    print("PASS")
