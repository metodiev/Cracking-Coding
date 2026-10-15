"""Task 364 - Implement quadtree for 2D region queries. (Tier: Intermediate, CTCI Ch 4)

Quadtree over a square binary grid whose side is a power of two.  Split rule: a node
covering rows [r, r+size) and columns [c, c+size) is a leaf when size == 1 or when all
its cells are equal; otherwise it splits into four equal quadrants (size // 2) and
stores the total number of ones of the whole region.  "Count ones in rectangle"
descends the tree: disjoint nodes contribute 0, fully contained nodes contribute their
stored count, and a partially covered leaf contributes its uniform value times the
cells inside the query.  Counts are asserted against a naive loop for every rectangle
of 8x8 LCG grids.
"""


class QuadTree:
    """Static quadtree over a power-of-two binary grid; counts ones in rectangles."""

    def __init__(self, grid):
        self.size = len(grid)
        assert self.size & (self.size - 1) == 0, "grid side must be a power of two"
        assert all(len(row) == self.size for row in grid), "grid must be square"
        self.grid = grid
        self.root = self._build(0, 0, self.size)

    def _build(self, row, col, size):
        first = self.grid[row][col]
        total = 0
        uniform = True
        for r in range(row, row + size):
            for c in range(col, col + size):
                value = self.grid[r][c]
                total += value
                if value != first:
                    uniform = False
        if uniform or size == 1:
            return [row, col, size, total, first, None]
        half = size // 2
        children = [
            self._build(row, col, half),
            self._build(row, col + half, half),
            self._build(row + half, col, half),
            self._build(row + half, col + half, half),
        ]
        return [row, col, size, total, None, children]

    def count_rect(self, row_lo, row_hi, col_lo, col_hi):
        """Number of ones inside the inclusive rectangle [row_lo..row_hi] x [col_lo..col_hi]."""
        return self._count(self.root, row_lo, row_hi, col_lo, col_hi)

    def _count(self, node, row_lo, row_hi, col_lo, col_hi):
        row, col, size, total, value, children = node
        if (row > row_hi or row + size - 1 < row_lo or
                col > col_hi or col + size - 1 < col_lo):
            return 0                                        # disjoint
        if (row_lo <= row and row + size - 1 <= row_hi and
                col_lo <= col and col + size - 1 <= col_hi):
            return total                                    # fully contained
        if children is None:                                # uniform leaf, partial overlap
            overlap_rows = min(row + size - 1, row_hi) - max(row, row_lo) + 1
            overlap_cols = min(col + size - 1, col_hi) - max(col, col_lo) + 1
            return value * overlap_rows * overlap_cols
        return (self._count(children[0], row_lo, row_hi, col_lo, col_hi) +
                self._count(children[1], row_lo, row_hi, col_lo, col_hi) +
                self._count(children[2], row_lo, row_hi, col_lo, col_hi) +
                self._count(children[3], row_lo, row_hi, col_lo, col_hi))


def naive_rect_count(grid, row_lo, row_hi, col_lo, col_hi):
    return sum(grid[r][c] for r in range(row_lo, row_hi + 1)
               for c in range(col_lo, col_hi + 1))


def lcg_binary_grid(size, seed):
    grid = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            seed = (seed * 75 + 74) % 65537
            grid[r][c] = seed % 2
    return grid, seed


def diagonal_grid(size):
    return [[1 if r == c else 0 for c in range(size)] for r in range(size)]


def main():
    # Hand-checked diagonal grid: ones only on the main diagonal.
    grid = diagonal_grid(8)
    tree = QuadTree(grid)
    assert tree.count_rect(0, 7, 0, 7) == 8
    assert tree.count_rect(0, 3, 0, 3) == 4
    assert tree.count_rect(2, 5, 2, 5) == 4
    assert tree.count_rect(1, 6, 0, 7) == 6
    assert tree.count_rect(0, 7, 7, 7) == 1
    assert tree.count_rect(0, 0, 1, 7) == 0
    assert tree.count_rect(3, 4, 3, 4) == 2
    assert tree.count_rect(2, 2, 0, 1) == 0

    # Uniform grids: all ones and all zeros.
    ones = QuadTree([[1] * 8 for _ in range(8)])
    assert ones.count_rect(0, 7, 0, 7) == 64 and ones.count_rect(2, 4, 1, 3) == 9
    zeros = QuadTree([[0] * 8 for _ in range(8)])
    assert zeros.count_rect(0, 7, 0, 7) == 0

    # Single-cell grid must be a leaf.
    cell = QuadTree([[1]])
    assert cell.count_rect(0, 0, 0, 0) == 1

    # 8x8 LCG grids: every possible rectangle against the naive loop.
    seed = 4919
    for trial in range(8):
        seed = (seed * 75 + 74) % 65537
        grid, seed = lcg_binary_grid(8, seed)
        tree = QuadTree(grid)
        for row_lo in range(8):
            for row_hi in range(row_lo, 8):
                for col_lo in range(8):
                    for col_hi in range(col_lo, 8):
                        want = naive_rect_count(grid, row_lo, row_hi, col_lo, col_hi)
                        assert tree.count_rect(row_lo, row_hi, col_lo, col_hi) == want, \
                            (trial, row_lo, row_hi, col_lo, col_hi)

    # A sparse 16x16 grid (deeper splits) with random rectangles.
    seed = 24680
    for trial in range(3):
        seed = (seed * 75 + 74) % 65537
        grid, seed = lcg_binary_grid(16, seed)
        tree = QuadTree(grid)
        for _ in range(60):
            seed = (seed * 75 + 74) % 65537
            row_lo = seed % 16
            seed = (seed * 75 + 74) % 65537
            row_hi = row_lo + seed % (16 - row_lo)
            seed = (seed * 75 + 74) % 65537
            col_lo = seed % 16
            seed = (seed * 75 + 74) % 65537
            col_hi = col_lo + seed % (16 - col_lo)
            want = naive_rect_count(grid, row_lo, row_hi, col_lo, col_hi)
            assert tree.count_rect(row_lo, row_hi, col_lo, col_hi) == want


if __name__ == "__main__":
    main()
    print("PASS")
