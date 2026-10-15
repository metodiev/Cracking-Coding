/** Task 265 - Count number of ways in matrix with obstacles (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * Rolling 1D DP: ways[c] = ways[c] (from above) + ways[c-1] (from the left), zeroed
     * on obstacles (1 marks an obstacle, 0 a free cell). O(m*n) time, O(n) space.
     * Empty grid / empty first row -> 0; blocked start or end -> 0; 1 x n and n x 1
     * grids have the single monotone path, so 1 iff no cell is blocked.
     * On identical grids this equals task 266's unique-path DP; on obstacle-free grids
     * both equal C(m+n-2, m-1) - asserted below for several sizes.
     */
    static long countWays(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int n = grid[0].length;
        long[] ways = new long[n];
        ways[0] = 1;
        for (int[] row : grid)
            for (int c = 0; c < n; c++) {
                if (row[c] == 1) ways[c] = 0;
                else if (c > 0) ways[c] += ways[c - 1];
            }
        return ways[n - 1];
    }

    /** Recursive enumeration used only for cross-checks. */
    static long countWaysBrute(int[][] grid) {
        return go(grid, 0, 0);
    }

    private static long go(int[][] grid, int r, int c) {
        if (grid[r][c] == 1) return 0;
        if (r == grid.length - 1 && c == grid[0].length - 1) return 1;
        long total = 0;
        if (r + 1 < grid.length) total += go(grid, r + 1, c);
        if (c + 1 < grid[0].length) total += go(grid, r, c + 1);
        return total;
    }

    /** Task 266's closed form C(m+n-2, m-1), used for obstacle-free cross-checks. */
    static long uniquePathsNoObstacles(int m, int n) {
        if (m <= 0 || n <= 0) return 0;
        return comb(m + n - 2, m - 1);
    }

    private static long comb(int n, int k) {
        long result = 1;
        for (int i = 1; i <= k; i++) result = result * (n - k + i) / i;
        return result;
    }

    private static int[][] freeGrid(int m, int n) {
        return new int[m][n];
    }

    public static void main(String[] args) {
        check(countWays(new int[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}) == 6, "classic 3x3 free");
        check(countWays(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}) == 2, "classic obstacle case");
        check(countWays(new int[][]{{0, 1}, {0, 0}}) == 1, "only down then right");
        check(countWays(new int[][]{{1, 0}, {0, 0}}) == 0, "start blocked");
        check(countWays(new int[][]{{0, 0}, {0, 1}}) == 0, "end blocked");
        check(countWays(new int[][]{{0, 0, 0, 0, 0}}) == 1, "1 x n");
        check(countWays(new int[][]{{0}, {0}, {0}}) == 1, "n x 1");
        check(countWays(new int[][]{{0, 1, 0}}) == 0, "1 x n blocked");
        check(countWays(new int[][]{{0}, {1}, {0}}) == 0, "n x 1 blocked");
        check(countWays(new int[][]{{0, 1, 0}, {0, 0, 0}}) == 1, "two rows with a hole");
        check(countWays(new int[][]{{0, 1}, {1, 0}}) == 0, "both moves blocked");
        check(countWays(new int[0][]) == 0, "empty grid");
        check(countWays(new int[][]{{}}) == 0, "empty first row");

        int[][] sizeCases = {{1, 1}, {1, 5}, {2, 2}, {3, 3}, {3, 7}, {4, 4}};
        long[] expected = {1, 1, 2, 6, 28, 20};
        for (int i = 0; i < sizeCases.length; i++) {
            int m = sizeCases[i][0], n = sizeCases[i][1];
            check(countWays(freeGrid(m, n)) == expected[i]
                    && uniquePathsNoObstacles(m, n) == expected[i], "closed form m=" + m + " n=" + n);
        }

        Random rng = new Random(265);
        for (int test = 0; test < 300; test++) {        // cross-check vs recursive enumeration
            int m = 1 + rng.nextInt(5), n = 1 + rng.nextInt(5);
            int[][] grid = new int[m][n];
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++) grid[r][c] = rng.nextDouble() < 0.3 ? 1 : 0;
            check(countWays(grid) == countWaysBrute(grid), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
