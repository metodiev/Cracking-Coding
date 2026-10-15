/** Task 364 - Implement quadtree for 2D region queries. (Tier: Intermediate, CTCI Ch 4) */
class QuadNode {
    final int row;
    final int col;
    final int size;
    final int count;
    final int value;                 // uniform leaf value; -1 when the node was split
    final QuadNode[] children;       // null for leaves

    QuadNode(int row, int col, int size, int count, int value, QuadNode[] children) {
        this.row = row;
        this.col = col;
        this.size = size;
        this.count = count;
        this.value = value;
        this.children = children;
    }
}

class QuadTree {
    // Split rule: a node is a leaf when size == 1 or all of its cells are equal;
    // otherwise it splits into four quadrants of size / 2 and stores the total count.
    private final QuadNode root;

    QuadTree(int[][] grid) {
        int size = grid.length;
        if (size == 0 || (size & (size - 1)) != 0) {
            throw new IllegalArgumentException("grid side must be a power of two");
        }
        for (int[] row : grid) {
            if (row.length != size) throw new IllegalArgumentException("grid must be square");
        }
        root = build(grid, 0, 0, size);
    }

    private QuadNode build(int[][] grid, int row, int col, int size) {
        int first = grid[row][col];
        int total = 0;
        boolean uniform = true;
        for (int r = row; r < row + size; r++) {
            for (int c = col; c < col + size; c++) {
                int cell = grid[r][c];
                total += cell;
                if (cell != first) uniform = false;
            }
        }
        if (uniform || size == 1) {
            return new QuadNode(row, col, size, total, first, null);
        }
        int half = size / 2;
        QuadNode[] children = {
            build(grid, row, col, half),
            build(grid, row, col + half, half),
            build(grid, row + half, col, half),
            build(grid, row + half, col + half, half),
        };
        return new QuadNode(row, col, size, total, -1, children);
    }

    /** Number of ones in the inclusive rectangle [rowLo..rowHi] x [colLo..colHi]. */
    int countRect(int rowLo, int rowHi, int colLo, int colHi) {
        return count(root, rowLo, rowHi, colLo, colHi);
    }

    private int count(QuadNode node, int rowLo, int rowHi, int colLo, int colHi) {
        if (node.row > rowHi || node.row + node.size - 1 < rowLo ||
                node.col > colHi || node.col + node.size - 1 < colLo) {
            return 0;                                        // disjoint
        }
        if (rowLo <= node.row && node.row + node.size - 1 <= rowHi &&
                colLo <= node.col && node.col + node.size - 1 <= colHi) {
            return node.count;                               // fully contained
        }
        if (node.children == null) {                          // uniform leaf, partial overlap
            int overlapRows = Math.min(node.row + node.size - 1, rowHi) - Math.max(node.row, rowLo) + 1;
            int overlapCols = Math.min(node.col + node.size - 1, colHi) - Math.max(node.col, colLo) + 1;
            return node.value * overlapRows * overlapCols;
        }
        int total = 0;
        for (QuadNode child : node.children) {
            total += count(child, rowLo, rowHi, colLo, colHi);
        }
        return total;
    }
}

public class Solution {
    private static int seed = 4919;

    private static int naiveRectCount(int[][] grid, int rowLo, int rowHi, int colLo, int colHi) {
        int total = 0;
        for (int r = rowLo; r <= rowHi; r++) {
            for (int c = colLo; c <= colHi; c++) total += grid[r][c];
        }
        return total;
    }

    private static int[][] diagonalGrid(int size) {
        int[][] grid = new int[size][size];
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) grid[r][c] = r == c ? 1 : 0;
        }
        return grid;
    }

    private static int[][] lcgBinaryGrid(int size) {
        int[][] grid = new int[size][size];
        for (int r = 0; r < size; r++) {
            for (int c = 0; c < size; c++) {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 2;
            }
        }
        return grid;
    }

    public static void main(String[] args) {
        // Hand-checked diagonal grid: ones only on the main diagonal.
        QuadTree tree = new QuadTree(diagonalGrid(8));
        check(tree.countRect(0, 7, 0, 7) == 8, "diagonal full grid");
        check(tree.countRect(0, 3, 0, 3) == 4, "top-left quadrant");
        check(tree.countRect(2, 5, 2, 5) == 4, "middle 4x4");
        check(tree.countRect(1, 6, 0, 7) == 6, "interior rows");
        check(tree.countRect(0, 7, 7, 7) == 1, "right column");
        check(tree.countRect(0, 0, 1, 7) == 0, "top row minus corner");
        check(tree.countRect(3, 4, 3, 4) == 2, "two diagonal cells");
        check(tree.countRect(2, 2, 0, 1) == 0, "row 2 left of the diagonal");

        // Uniform grids: all ones and all zeros.
        int[][] onesGrid = new int[8][8];
        for (int[] row : onesGrid) java.util.Arrays.fill(row, 1);
        QuadTree ones = new QuadTree(onesGrid);
        check(ones.countRect(0, 7, 0, 7) == 64 && ones.countRect(2, 4, 1, 3) == 9, "all ones");
        QuadTree zeros = new QuadTree(new int[8][8]);
        check(zeros.countRect(0, 7, 0, 7) == 0, "all zeros");

        QuadTree cell = new QuadTree(new int[][]{{1}});
        check(cell.countRect(0, 0, 0, 0) == 1, "single cell");

        // 8x8 LCG grids: every possible rectangle against the naive loop.
        for (int trial = 0; trial < 8; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int[][] grid = lcgBinaryGrid(8);
            QuadTree randomTree = new QuadTree(grid);
            for (int rowLo = 0; rowLo < 8; rowLo++) {
                for (int rowHi = rowLo; rowHi < 8; rowHi++) {
                    for (int colLo = 0; colLo < 8; colLo++) {
                        for (int colHi = colLo; colHi < 8; colHi++) {
                            check(randomTree.countRect(rowLo, rowHi, colLo, colHi) ==
                                            naiveRectCount(grid, rowLo, rowHi, colLo, colHi),
                                    "random trial " + trial + " rectangle");
                        }
                    }
                }
            }
        }

        // A 16x16 grid (deeper splits) with random rectangles.
        for (int trial = 0; trial < 3; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int[][] grid = lcgBinaryGrid(16);
            QuadTree deepTree = new QuadTree(grid);
            for (int q = 0; q < 60; q++) {
                seed = (seed * 75 + 74) % 65537;
                int rowLo = seed % 16;
                seed = (seed * 75 + 74) % 65537;
                int rowHi = rowLo + seed % (16 - rowLo);
                seed = (seed * 75 + 74) % 65537;
                int colLo = seed % 16;
                seed = (seed * 75 + 74) % 65537;
                int colHi = colLo + seed % (16 - colLo);
                check(deepTree.countRect(rowLo, rowHi, colLo, colHi) ==
                                naiveRectCount(grid, rowLo, rowHi, colLo, colHi),
                        "16x16 trial " + trial + " rectangle");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
