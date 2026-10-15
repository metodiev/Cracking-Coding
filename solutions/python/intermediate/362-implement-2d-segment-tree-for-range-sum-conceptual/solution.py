"""Task 362 - Implement 2D segment tree for range sum (conceptual). (Tier: Intermediate, CTCI Ch 4)

A segment tree over rows whose every node stores an inner segment tree over columns.
An inner node of outer index `node` holds, for each column range, the sum of that column
range across all rows of the outer node's row range; internal outer nodes are the
element-wise sum of their two children, which is exactly the merge a 2D segment tree
needs for sums.  A rectangle sum decomposes the rows into O(log rows) outer nodes, each
answered by an O(log cols) inner query.  Sizes are kept small (8x8) so every queried
rectangle can be cross-checked against a naive double loop on deterministic LCG grids.
"""


class SegmentTree2D:
    """Static 2D range-sum structure (nested segment trees over a fixed grid)."""

    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        size_r = 1
        while size_r < self.rows:
            size_r <<= 1
        size_c = 1
        while size_c < self.cols:
            size_c <<= 1
        self.size_r = size_r
        self.size_c = size_c
        self.trees = [[0] * (2 * size_c) for _ in range(2 * size_r)]
        for r in range(self.rows):
            row_tree = self.trees[size_r + r]
            for c in range(self.cols):
                row_tree[size_c + c] = grid[r][c]
            for node in range(size_c - 1, 0, -1):
                row_tree[node] = row_tree[2 * node] + row_tree[2 * node + 1]
        for node in range(size_r - 1, 0, -1):
            left = self.trees[2 * node]
            right = self.trees[2 * node + 1]
            merged = self.trees[node]
            for j in range(1, 2 * size_c):
                merged[j] = left[j] + right[j]

    def _inner_sum(self, tree, col_lo, col_hi):
        """Sum over half-open column range [col_lo, col_hi) inside one outer node."""
        total = 0
        l = col_lo + self.size_c
        r = col_hi + self.size_c
        while l < r:
            if l & 1:
                total += tree[l]
                l += 1
            if r & 1:
                r -= 1
                total += tree[r]
            l >>= 1
            r >>= 1
        return total

    def range_sum(self, row_lo, row_hi, col_lo, col_hi):
        """Sum over the inclusive rectangle [row_lo..row_hi] x [col_lo..col_hi]."""
        if row_lo > row_hi or col_lo > col_hi:
            return 0
        total = 0
        l = row_lo + self.size_r
        r = row_hi + 1 + self.size_r
        while l < r:
            if l & 1:
                total += self._inner_sum(self.trees[l], col_lo, col_hi + 1)
                l += 1
            if r & 1:
                r -= 1
                total += self._inner_sum(self.trees[r], col_lo, col_hi + 1)
            l >>= 1
            r >>= 1
        return total


def naive_rect_sum(grid, row_lo, row_hi, col_lo, col_hi):
    return sum(grid[r][c] for r in range(row_lo, row_hi + 1)
               for c in range(col_lo, col_hi + 1))


def lcg_grid(rows, cols, seed):
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            seed = (seed * 75 + 74) % 65537
            grid[r][c] = seed % 20 - 5                  # small values, some negative
    return grid, seed


HAND_GRID = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]


def main():
    # Hand-checked sums on the 4x4 grid above.
    tree = SegmentTree2D(HAND_GRID)
    assert tree.range_sum(0, 3, 0, 3) == 136            # 1+2+...+16
    assert tree.range_sum(0, 1, 0, 1) == 1 + 2 + 5 + 6 == 14
    assert tree.range_sum(2, 3, 1, 2) == 10 + 11 + 14 + 15 == 50
    assert tree.range_sum(1, 1, 0, 3) == 5 + 6 + 7 + 8 == 26
    assert tree.range_sum(0, 3, 0, 0) == 1 + 5 + 9 + 13 == 28
    assert tree.range_sum(2, 2, 3, 3) == 12
    assert tree.range_sum(3, 0, 0, 3) == 0              # empty row range
    assert tree.range_sum(0, 3, 3, 2) == 0              # empty column range

    # Degenerate shapes: one cell, one row, one column.
    single = SegmentTree2D([[7]])
    assert single.range_sum(0, 0, 0, 0) == 7
    row = SegmentTree2D([[1, -2, 3, 4, 5]])
    assert row.range_sum(0, 0, 0, 4) == 11 and row.range_sum(0, 0, 1, 3) == 5
    col = SegmentTree2D([[1], [2], [3], [4], [5]])
    assert col.range_sum(0, 4, 0, 0) == 15 and col.range_sum(2, 4, 0, 0) == 12

    # 8x8 LCG grids: every rectangle cross-checked against the naive double loop.
    seed = 31337
    for trial in range(6):
        seed = (seed * 75 + 74) % 65537
        grid, seed = lcg_grid(8, 8, seed)
        tree = SegmentTree2D(grid)
        for row_lo in range(8):
            for row_hi in range(row_lo, 8):
                for col_lo in range(8):
                    for col_hi in range(col_lo, 8):
                        want = naive_rect_sum(grid, row_lo, row_hi, col_lo, col_hi)
                        got = tree.range_sum(row_lo, row_hi, col_lo, col_hi)
                        assert got == want, (trial, row_lo, row_hi, col_lo, col_hi)

    # Non power-of-two shapes (3x5, 5x3, 7x1) still work.
    for rows, cols in ((3, 5), (5, 3), (7, 1)):
        seed = (seed * 75 + 74) % 65537
        grid, seed = lcg_grid(rows, cols, seed)
        tree = SegmentTree2D(grid)
        for row_lo in range(rows):
            for row_hi in range(row_lo, rows):
                for col_lo in range(cols):
                    for col_hi in range(col_lo, cols):
                        assert tree.range_sum(row_lo, row_hi, col_lo, col_hi) == \
                            naive_rect_sum(grid, row_lo, row_hi, col_lo, col_hi)


if __name__ == "__main__":
    main()
    print("PASS")
