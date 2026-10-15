/** Task 267 - Minimum path sum in grid (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * Rolling 1D DP: dp[c] = min(dp[c] from above, dp[c-1] from the left) + grid[r][c];
     * the first row and column each have a single predecessor. O(m*n) time, O(n) space.
     * Empty grid / empty first row raises IllegalArgumentException; a single cell
     * returns its value; 1 x n and n x 1 grids have a single path. Negatives allowed.
     */
    static long minPathSum(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) throw new IllegalArgumentException("grid must be a non-empty rectangle");
        int n = grid[0].length;
        long[] dp = new long[n];
        for (int r = 0; r < grid.length; r++)
            for (int c = 0; c < n; c++) {
                if (r == 0 && c == 0) dp[c] = grid[r][c];
                else if (r == 0) dp[c] = dp[c - 1] + grid[r][c];            // only from the left
                else if (c == 0) dp[c] += grid[r][c];                       // only from above
                else dp[c] = Math.min(dp[c], dp[c - 1]) + grid[r][c];
            }
        return dp[n - 1];
    }

    /** Recursive enumeration of all paths; used only for cross-checks. */
    static long minPathSumBrute(int[][] grid) {
        return go(grid, 0, 0);
    }

    private static long go(int[][] grid, int r, int c) {
        if (r == grid.length - 1 && c == grid[0].length - 1) return grid[r][c];
        Long best = null;
        if (r + 1 < grid.length) best = go(grid, r + 1, c);
        if (c + 1 < grid[0].length) {
            long right = go(grid, r, c + 1);
            best = best == null ? right : Math.min(best, right);
        }
        return grid[r][c] + best;
    }

    public static void main(String[] args) {
        check(minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}) == 7, "classic = 7");
        check(minPathSum(new int[][]{{5}}) == 5, "single cell");
        check(minPathSum(new int[][]{{-3}}) == -3, "single negative cell");
        check(minPathSum(new int[][]{{1, 2, 3}}) == 6, "1 x n single path");
        check(minPathSum(new int[][]{{1}, {2}, {3}}) == 6, "n x 1 single path");
        check(minPathSum(new int[][]{{1, 2, 3}, {4, 100, 6}}) == 12, "route above the huge cell");
        check(minPathSum(new int[][]{{1, 100}, {1, 1}}) == 3, "down first, then right");
        check(minPathSum(new int[][]{{1, -3}, {1, -5}}) == -7, "negatives allowed");

        int emptyThrows = 0;
        try { minPathSum(new int[0][]); } catch (IllegalArgumentException e) { emptyThrows++; }
        try { minPathSum(new int[][]{{}}); } catch (IllegalArgumentException e) { emptyThrows++; }
        check(emptyThrows == 2, "empty grid cases raise");

        Random rng = new Random(267);
        for (int test = 0; test < 300; test++) {        // cross-check vs recursive enumeration
            int m = 1 + rng.nextInt(5), n = 1 + rng.nextInt(5);
            int[][] grid = new int[m][n];
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++) grid[r][c] = rng.nextInt(15) - 5;
            check(minPathSum(grid) == minPathSumBrute(grid), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
