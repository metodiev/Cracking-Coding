/** Task 362 - Implement 2D segment tree for range sum (conceptual). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class SegmentTree2D {
    // A segment tree over rows whose every node stores an inner segment tree over
    // columns; internal outer nodes are the element-wise sum of their two children.
    private final int sizeR;
    private final int sizeC;
    private final int[][] trees;

    SegmentTree2D(int[][] grid) {
        int rows = grid.length;
        int cols = rows == 0 ? 0 : grid[0].length;
        sizeR = nextPowerOfTwo(Math.max(rows, 1));
        sizeC = nextPowerOfTwo(Math.max(cols, 1));
        trees = new int[2 * sizeR][2 * sizeC];
        for (int r = 0; r < rows; r++) {
            int[] rowTree = trees[sizeR + r];
            for (int c = 0; c < cols; c++) rowTree[sizeC + c] = grid[r][c];
            for (int node = sizeC - 1; node >= 1; node--) {
                rowTree[node] = rowTree[2 * node] + rowTree[2 * node + 1];
            }
        }
        for (int node = sizeR - 1; node >= 1; node--) {
            int[] left = trees[2 * node];
            int[] right = trees[2 * node + 1];
            int[] merged = trees[node];
            for (int j = 1; j < 2 * sizeC; j++) merged[j] = left[j] + right[j];
        }
    }

    private static int nextPowerOfTwo(int value) {
        int size = 1;
        while (size < value) size <<= 1;
        return size;
    }

    /** Sum over the inclusive rectangle [rowLo..rowHi] x [colLo..colHi]. */
    int rangeSum(int rowLo, int rowHi, int colLo, int colHi) {
        if (rowLo > rowHi || colLo > colHi) return 0;
        int total = 0;
        int l = rowLo + sizeR;
        int r = rowHi + 1 + sizeR;
        while (l < r) {
            if ((l & 1) == 1) {
                total += innerSum(trees[l], colLo, colHi + 1);
                l++;
            }
            if ((r & 1) == 1) {
                r--;
                total += innerSum(trees[r], colLo, colHi + 1);
            }
            l >>= 1;
            r >>= 1;
        }
        return total;
    }

    private int innerSum(int[] tree, int colLo, int colHi) {
        int total = 0;
        int l = colLo + sizeC;
        int r = colHi + sizeC;
        while (l < r) {
            if ((l & 1) == 1) total += tree[l++];
            if ((r & 1) == 1) total += tree[--r];
            l >>= 1;
            r >>= 1;
        }
        return total;
    }
}

public class Solution {
    private static int seed = 31337;

    private static int naiveRectSum(int[][] grid, int rowLo, int rowHi, int colLo, int colHi) {
        int total = 0;
        for (int r = rowLo; r <= rowHi; r++) {
            for (int c = colLo; c <= colHi; c++) total += grid[r][c];
        }
        return total;
    }

    private static int[][] lcgGrid(int rows, int cols) {
        int[][] grid = new int[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 20 - 5;
            }
        }
        return grid;
    }

    public static void main(String[] args) {
        // Hand-checked sums on the 4x4 grid 1..16.
        int[][] hand = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16},
        };
        SegmentTree2D tree = new SegmentTree2D(hand);
        check(tree.rangeSum(0, 3, 0, 3) == 136, "full grid 1+...+16");
        check(tree.rangeSum(0, 1, 0, 1) == 14, "top-left 2x2");
        check(tree.rangeSum(2, 3, 1, 2) == 50, "bottom-right 2x2 offset");
        check(tree.rangeSum(1, 1, 0, 3) == 26, "row 1");
        check(tree.rangeSum(0, 3, 0, 0) == 28, "column 0");
        check(tree.rangeSum(2, 2, 3, 3) == 12, "single cell");
        check(tree.rangeSum(3, 0, 0, 3) == 0, "empty row range");
        check(tree.rangeSum(0, 3, 3, 2) == 0, "empty column range");

        // Degenerate shapes: one cell, one row, one column.
        SegmentTree2D single = new SegmentTree2D(new int[][]{{7}});
        check(single.rangeSum(0, 0, 0, 0) == 7, "single cell grid");
        SegmentTree2D row = new SegmentTree2D(new int[][]{{1, -2, 3, 4, 5}});
        check(row.rangeSum(0, 0, 0, 4) == 11 && row.rangeSum(0, 0, 1, 3) == 5, "single row grid");
        SegmentTree2D col = new SegmentTree2D(new int[][]{{1}, {2}, {3}, {4}, {5}});
        check(col.rangeSum(0, 4, 0, 0) == 15 && col.rangeSum(2, 4, 0, 0) == 12, "single column grid");

        // 8x8 LCG grids: every rectangle cross-checked against the naive double loop.
        for (int trial = 0; trial < 6; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int[][] grid = lcgGrid(8, 8);
            SegmentTree2D randomTree = new SegmentTree2D(grid);
            for (int rowLo = 0; rowLo < 8; rowLo++) {
                for (int rowHi = rowLo; rowHi < 8; rowHi++) {
                    for (int colLo = 0; colLo < 8; colLo++) {
                        for (int colHi = colLo; colHi < 8; colHi++) {
                            check(randomTree.rangeSum(rowLo, rowHi, colLo, colHi) ==
                                            naiveRectSum(grid, rowLo, rowHi, colLo, colHi),
                                    "random trial " + trial + " rectangle");
                        }
                    }
                }
            }
        }

        // Non power-of-two shapes (3x5, 5x3, 7x1) still work.
        int[][] shapes = {{3, 5}, {5, 3}, {7, 1}};
        for (int[] shape : shapes) {
            seed = (seed * 75 + 74) % 65537;
            int rows = shape[0], cols = shape[1];
            int[][] grid = lcgGrid(rows, cols);
            SegmentTree2D shapeTree = new SegmentTree2D(grid);
            for (int rowLo = 0; rowLo < rows; rowLo++) {
                for (int rowHi = rowLo; rowHi < rows; rowHi++) {
                    for (int colLo = 0; colLo < cols; colLo++) {
                        for (int colHi = colLo; colHi < cols; colHi++) {
                            check(shapeTree.rangeSum(rowLo, rowHi, colLo, colHi) ==
                                            naiveRectSum(grid, rowLo, rowHi, colLo, colHi),
                                    "shape " + rows + "x" + cols);
                        }
                    }
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
