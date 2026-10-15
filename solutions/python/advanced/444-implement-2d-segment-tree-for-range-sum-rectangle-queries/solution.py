"""Task 444 - Implement 2D segment tree for range sum rectangle queries. (Tier: Advanced, CTCI Ch 4)

Nested 2D segment tree: the outer tree runs over rows, and every outer node owns a 1D segment
tree over the columns holding the sums of its row range. Build is O(n*m); a point update walks
O(log n) outer nodes and touches O(log m) inner nodes each, and a rectangle sum decomposes the
row range into O(log n) outer nodes whose column ranges are answered in O(log m) each, so both
operations are O(log n * log m). Verified against a brute-force grid with direct summation.
"""


class SegmentTree2D:
    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.grid = [row[:] for row in grid]
        self.tree = [[0] * (4 * max(1, self.cols)) for _ in range(4 * max(1, self.rows))]
        if self.rows:
            self._build(1, 0, self.rows - 1, grid)

    def _inner_build(self, arr, node, lo, hi, row):
        if lo == hi:
            arr[node] = row[lo]
            return
        mid = (lo + hi) // 2
        self._inner_build(arr, 2 * node, lo, mid, row)
        self._inner_build(arr, 2 * node + 1, mid + 1, hi, row)
        arr[node] = arr[2 * node] + arr[2 * node + 1]

    def _build(self, node, lo, hi, grid):
        if lo == hi:
            self._inner_build(self.tree[node], 1, 0, self.cols - 1, grid[lo])
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, grid)
        self._build(2 * node + 1, mid + 1, hi, grid)
        current = self.tree[node]
        left = self.tree[2 * node]
        right = self.tree[2 * node + 1]
        for k in range(1, len(current)):
            current[k] = left[k] + right[k]

    def set_value(self, row, col, value):
        delta = value - self.grid[row][col]
        self.grid[row][col] = value
        self._outer_update(1, 0, self.rows - 1, row, col, delta)

    def _outer_update(self, node, lo, hi, row, col, delta):
        self._inner_update(self.tree[node], 1, 0, self.cols - 1, col, delta)
        if lo == hi:
            return
        mid = (lo + hi) // 2
        if row <= mid:
            self._outer_update(2 * node, lo, mid, row, col, delta)
        else:
            self._outer_update(2 * node + 1, mid + 1, hi, row, col, delta)

    def _inner_update(self, arr, node, lo, hi, pos, delta):
        arr[node] += delta
        if lo == hi:
            return
        mid = (lo + hi) // 2
        if pos <= mid:
            self._inner_update(arr, 2 * node, lo, mid, pos, delta)
        else:
            self._inner_update(arr, 2 * node + 1, mid + 1, hi, pos, delta)

    def _inner_query(self, arr, node, lo, hi, left, right):
        if left <= lo and hi <= right:
            return arr[node]
        mid = (lo + hi) // 2
        total = 0
        if left <= mid:
            total += self._inner_query(arr, 2 * node, lo, mid, left, right)
        if right > mid:
            total += self._inner_query(arr, 2 * node + 1, mid + 1, hi, left, right)
        return total

    def rect_sum(self, r1, c1, r2, c2):
        return self._rect(1, 0, self.rows - 1, r1, c1, r2, c2)

    def _rect(self, node, lo, hi, r1, c1, r2, c2):
        if r2 < lo or hi < r1:
            return 0
        if r1 <= lo and hi <= r2:
            return self._inner_query(self.tree[node], 1, 0, self.cols - 1, c1, c2)
        mid = (lo + hi) // 2
        return (self._rect(2 * node, lo, mid, r1, c1, r2, c2)
                + self._rect(2 * node + 1, mid + 1, hi, r1, c1, r2, c2))


def lcg_grid_ops(count, seed, n):
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
        value = seed % 41 - 20
        ops.append((kind, r1, c1, r2, c2, value))
    return ops


def main():
    tree = SegmentTree2D([[1, 2, 3], [4, 5, 6]])
    assert tree.rect_sum(0, 0, 0, 1) == 3
    assert tree.rect_sum(0, 0, 1, 2) == 21
    assert tree.rect_sum(1, 0, 1, 1) == 9
    tree.set_value(1, 1, 50)
    assert tree.rect_sum(1, 1, 1, 1) == 50
    assert tree.rect_sum(0, 0, 1, 2) == 66
    assert tree.rect_sum(1, 0, 1, 1) == 54
    tree.set_value(0, 2, 1)
    assert tree.rect_sum(0, 2, 0, 2) == 1
    assert tree.rect_sum(0, 0, 1, 2) == 64

    n = 16
    ops = lcg_grid_ops(300, 444, n)
    seed = 99
    grid = []
    for _ in range(n):
        row = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            row.append(seed % 50 - 25)
        grid.append(row)
    tree = SegmentTree2D(grid)
    brute = [row[:] for row in grid]
    for kind, r1, c1, r2, c2, value in ops:
        if kind == 0:
            tree.set_value(r1, c1, value)
            brute[r1][c1] = value
        else:
            expected = sum(brute[r][c] for r in range(r1, r2 + 1) for c in range(c1, c2 + 1))
            assert tree.rect_sum(r1, c1, r2, c2) == expected, (r1, c1, r2, c2)


if __name__ == "__main__":
    main()
    print("PASS")
