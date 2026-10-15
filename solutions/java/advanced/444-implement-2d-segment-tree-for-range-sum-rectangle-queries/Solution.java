/** Task 444 - Implement 2D segment tree for range sum rectangle queries. (Tier: Advanced, CTCI Ch 4) */
class SegmentTree2D {
    private final int rows;
    private final int cols;
    private final long[][] grid;
    private final long[][] tree;

    SegmentTree2D(long[][] grid) {
        this.rows = grid.length;
        this.cols = rows > 0 ? grid[0].length : 0;
        this.grid = new long[rows][];
        for (int r = 0; r < rows; r++) this.grid[r] = grid[r].clone();
        tree = new long[4 * Math.max(1, rows)][4 * Math.max(1, cols)];
        if (rows > 0) build(1, 0, rows - 1, grid);
    }

    private void innerBuild(long[] arr, int node, int lo, int hi, long[] row) {
        if (lo == hi) {
            arr[node] = row[lo];
            return;
        }
        int mid = (lo + hi) >> 1;
        innerBuild(arr, 2 * node, lo, mid, row);
        innerBuild(arr, 2 * node + 1, mid + 1, hi, row);
        arr[node] = arr[2 * node] + arr[2 * node + 1];
    }

    private void build(int node, int lo, int hi, long[][] grid) {
        if (lo == hi) {
            innerBuild(tree[node], 1, 0, cols - 1, grid[lo]);
            return;
        }
        int mid = (lo + hi) >> 1;
        build(2 * node, lo, mid, grid);
        build(2 * node + 1, mid + 1, hi, grid);
        long[] current = tree[node];
        long[] left = tree[2 * node];
        long[] right = tree[2 * node + 1];
        for (int k = 1; k < current.length; k++) current[k] = left[k] + right[k];
    }

    void setValue(int row, int col, long value) {
        long delta = value - grid[row][col];
        grid[row][col] = value;
        outerUpdate(1, 0, rows - 1, row, col, delta);
    }

    private void outerUpdate(int node, int lo, int hi, int row, int col, long delta) {
        innerUpdate(tree[node], 1, 0, cols - 1, col, delta);
        if (lo == hi) return;
        int mid = (lo + hi) >> 1;
        if (row <= mid) outerUpdate(2 * node, lo, mid, row, col, delta);
        else outerUpdate(2 * node + 1, mid + 1, hi, row, col, delta);
    }

    private void innerUpdate(long[] arr, int node, int lo, int hi, int pos, long delta) {
        arr[node] += delta;
        if (lo == hi) return;
        int mid = (lo + hi) >> 1;
        if (pos <= mid) innerUpdate(arr, 2 * node, lo, mid, pos, delta);
        else innerUpdate(arr, 2 * node + 1, mid + 1, hi, pos, delta);
    }

    private long innerQuery(long[] arr, int node, int lo, int hi, int left, int right) {
        if (left <= lo && hi <= right) return arr[node];
        int mid = (lo + hi) >> 1;
        long total = 0;
        if (left <= mid) total += innerQuery(arr, 2 * node, lo, mid, left, right);
        if (right > mid) total += innerQuery(arr, 2 * node + 1, mid + 1, hi, left, right);
        return total;
    }

    long rectSum(int r1, int c1, int r2, int c2) {
        return rect(1, 0, rows - 1, r1, c1, r2, c2);
    }

    private long rect(int node, int lo, int hi, int r1, int c1, int r2, int c2) {
        if (r2 < lo || hi < r1) return 0;
        if (r1 <= lo && hi <= r2) return innerQuery(tree[node], 1, 0, cols - 1, c1, c2);
        int mid = (lo + hi) >> 1;
        return rect(2 * node, lo, mid, r1, c1, r2, c2)
                + rect(2 * node + 1, mid + 1, hi, r1, c1, r2, c2);
    }
}

public class Solution {
    private static int seed;

    private static int[][] lcgGridOps(int count, int n) {
        int[][] ops = new int[count][];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int r1 = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int c1 = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int r2 = r1 + seed % (n - r1);
            seed = (seed * 75 + 74) % 65537;
            int c2 = c1 + seed % (n - c1);
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 41 - 20;
            ops[k] = new int[]{kind, r1, c1, r2, c2, value};
        }
        return ops;
    }

    public static void main(String[] args) {
        SegmentTree2D tree = new SegmentTree2D(new long[][]{{1, 2, 3}, {4, 5, 6}});
        check(tree.rectSum(0, 0, 0, 1) == 3, "top row prefix");
        check(tree.rectSum(0, 0, 1, 2) == 21, "whole grid");
        check(tree.rectSum(1, 0, 1, 1) == 9, "bottom row prefix");
        tree.setValue(1, 1, 50);
        check(tree.rectSum(1, 1, 1, 1) == 50, "updated cell");
        check(tree.rectSum(0, 0, 1, 2) == 66, "whole grid after update");
        check(tree.rectSum(1, 0, 1, 1) == 54, "bottom row after update");
        tree.setValue(0, 2, 1);
        check(tree.rectSum(0, 2, 0, 2) == 1, "second update");
        check(tree.rectSum(0, 0, 1, 2) == 64, "whole grid after second update");

        int n = 16;
        int[][] ops = lcgGridOps(300, n);
        seed = 99;
        long[][] grid = new long[n][n];
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 50 - 25;
            }
        }
        tree = new SegmentTree2D(grid);
        long[][] brute = new long[n][];
        for (int r = 0; r < n; r++) brute[r] = grid[r].clone();
        for (int[] op : ops) {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0) {
                tree.setValue(r1, c1, value);
                brute[r1][c1] = value;
            } else {
                long expected = 0;
                for (int r = r1; r <= r2; r++) {
                    for (int c = c1; c <= c2; c++) expected += brute[r][c];
                }
                check(tree.rectSum(r1, c1, r2, c2) == expected, "random rectangle");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
