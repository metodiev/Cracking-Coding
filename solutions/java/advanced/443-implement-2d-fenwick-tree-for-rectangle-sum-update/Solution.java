/** Task 443 - Implement 2D Fenwick tree for rectangle sum/update. (Tier: Advanced, CTCI Ch 4) */
class Fenwick2D {
    private final int rows;
    private final int cols;
    private final long[][] tree;

    Fenwick2D(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        tree = new long[rows + 1][cols + 1];
    }

    void add(int row, int col, long delta) {
        for (int i = row + 1; i <= rows; i += i & -i) {
            for (int j = col + 1; j <= cols; j += j & -j) {
                tree[i][j] += delta;
            }
        }
    }

    long prefix(int row, int col) {
        long total = 0;
        for (int i = row + 1; i > 0; i -= i & -i) {
            for (int j = col + 1; j > 0; j -= j & -j) {
                total += tree[i][j];
            }
        }
        return total;
    }

    long rectSum(int r1, int c1, int r2, int c2) {
        return prefix(r2, c2) - prefix(r1 - 1, c2) - prefix(r2, c1 - 1) + prefix(r1 - 1, c1 - 1);
    }
}

class DiffFenwick2D {
    private final int rows;
    private final int cols;
    private final Fenwick2D inner;

    DiffFenwick2D(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        inner = new Fenwick2D(rows, cols);
    }

    void rectAdd(int r1, int c1, int r2, int c2, long delta) {
        inner.add(r1, c1, delta);
        if (r2 + 1 < rows) inner.add(r2 + 1, c1, -delta);
        if (c2 + 1 < cols) inner.add(r1, c2 + 1, -delta);
        if (r2 + 1 < rows && c2 + 1 < cols) inner.add(r2 + 1, c2 + 1, delta);
    }

    long point(int row, int col) {
        return inner.prefix(row, col);
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
            int value = seed % 21 - 10;
            ops[k] = new int[]{kind, r1, c1, r2, c2, value};
        }
        return ops;
    }

    public static void main(String[] args) {
        Fenwick2D bit = new Fenwick2D(4, 4);
        bit.add(0, 0, 1);
        bit.add(1, 2, 5);
        bit.add(3, 3, 2);
        check(bit.rectSum(0, 0, 3, 3) == 8, "full rectangle");
        check(bit.rectSum(1, 2, 1, 2) == 5, "single cell");
        check(bit.rectSum(0, 1, 2, 1) == 0, "empty column strip");
        check(bit.rectSum(0, 0, 0, 0) == 1, "corner cell");
        check(bit.rectSum(1, 2, 3, 3) == 7, "offset block");

        DiffFenwick2D diff = new DiffFenwick2D(4, 4);
        diff.rectAdd(0, 0, 1, 1, 3);
        check(diff.point(0, 0) == 3, "covered point");
        check(diff.point(1, 1) == 3, "covered corner");
        check(diff.point(1, 2) == 0, "outside point");
        check(diff.point(2, 2) == 0, "far point");
        diff.rectAdd(1, 1, 2, 2, 10);
        check(diff.point(1, 1) == 13, "overlap sum");
        check(diff.point(2, 2) == 10, "second block");
        check(diff.point(3, 3) == 0, "still outside");
        check(diff.point(0, 0) == 3, "first block intact");

        int n = 16;
        int[][] ops = lcgGridOps(300, n);

        bit = new Fenwick2D(n, n);
        long[][] grid = new long[n][n];
        for (int[] op : ops) {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0) {
                bit.add(r1, c1, value);
                grid[r1][c1] += value;
            } else {
                long expected = 0;
                for (int r = r1; r <= r2; r++) {
                    for (int c = c1; c <= c2; c++) expected += grid[r][c];
                }
                check(bit.rectSum(r1, c1, r2, c2) == expected, "random rectangle");
            }
        }

        diff = new DiffFenwick2D(n, n);
        grid = new long[n][n];
        for (int[] op : ops) {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0) {
                diff.rectAdd(r1, c1, r2, c2, value);
                for (int r = r1; r <= r2; r++) {
                    for (int c = c1; c <= c2; c++) grid[r][c] += value;
                }
            } else {
                check(diff.point(r1, c1) == grid[r1][c1], "random diff point");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
