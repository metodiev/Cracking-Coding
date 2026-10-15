/** Task 266 - Unique paths in grid with obstacles (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * Explicit m x n DP for the number of monotone right/down paths (1 = obstacle).
     * With no obstacles this reduces to the closed form C(m+n-2, m-1)
     * (uniquePathsNoObstacles); obstacles zero out blocked cells, so they can only
     * lower the count. A blocked start or end yields 0. O(m*n) time and space.
     */
    static long uniquePathsWithObstacles2d(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int m = grid.length, n = grid[0].length;
        long[][] dp = new long[m][n];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 1) dp[r][c] = 0;
                else if (r == 0 && c == 0) dp[r][c] = 1;
                else dp[r][c] = (r > 0 ? dp[r - 1][c] : 0) + (c > 0 ? dp[r][c - 1] : 0);
            }
        return dp[m - 1][n - 1];
    }

    /** Task 265's rolling 1D DP; identical grids must give identical counts. */
    static long countWays265(int[][] grid) {
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

    /** Closed form for an obstacle-free m x n grid: C(m+n-2, m-1). */
    static long uniquePathsNoObstacles(int m, int n) {
        if (m <= 0 || n <= 0) return 0;
        long result = 1;
        for (int i = 1; i <= m - 1; i++) result = result * (n - 1 + i) / i;
        return result;
    }

    /** Recursive enumeration used only for cross-checks. */
    static long uniquePathsBrute(int[][] grid) {
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

    private static int[][] freeGrid(int m, int n) {
        return new int[m][n];
    }

    public static void main(String[] args) {
        check(uniquePathsWithObstacles2d(new int[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}) == 6, "classic 3x3 = 6");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}) == 2, "obstacle variant");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 1}, {0, 0}}) == 1, "detour down");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 0}, {0, 1}}) == 0, "goal blocked");
        check(uniquePathsWithObstacles2d(new int[][]{{1, 0}, {0, 0}}) == 0, "start blocked");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 0}, {1, 0}}) == 1, "detour right");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 0, 0, 0, 0}}) == 1, "1 x n");
        check(uniquePathsWithObstacles2d(new int[][]{{0}, {0}, {0}}) == 1, "n x 1");
        check(uniquePathsWithObstacles2d(new int[][]{{0, 1, 0}}) == 0, "1 x n blocked");
        check(uniquePathsWithObstacles2d(new int[0][]) == 0, "empty grid");
        check(uniquePathsWithObstacles2d(new int[][]{{}}) == 0, "empty first row");

        int[][][] shared = {
            {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
            {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
            {{0, 1}, {0, 0}},
            {{0, 0}, {0, 1}},
            {{1, 0}, {0, 0}},
            {{0, 1}, {1, 0}},
            {{0, 1, 0}, {0, 0, 0}},
            {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}},
        };
        for (int[][] grid : shared)                 // 265's counter agrees on identical grids
            check(uniquePathsWithObstacles2d(grid) == countWays265(grid), "agrees with the 265 counter");

        int[][] sizeCases = {{1, 1}, {1, 5}, {2, 2}, {3, 3}, {3, 7}, {4, 4}};
        long[] expected = {1, 1, 2, 6, 28, 20};
        for (int i = 0; i < sizeCases.length; i++) {
            int m = sizeCases[i][0], n = sizeCases[i][1];
            check(uniquePathsWithObstacles2d(freeGrid(m, n)) == expected[i]
                    && uniquePathsNoObstacles(m, n) == expected[i], "closed form m=" + m + " n=" + n);
        }

        Random rng = new Random(266);
        for (int test = 0; test < 300; test++) {    // 2D == 265's rolling DP == brute force
            int m = 1 + rng.nextInt(5), n = 1 + rng.nextInt(5);
            int[][] grid = new int[m][n];
            for (int r = 0; r < m; r++)
                for (int c = 0; c < n; c++) grid[r][c] = rng.nextDouble() < 0.3 ? 1 : 0;
            check(uniquePathsWithObstacles2d(grid) == countWays265(grid), "2D == rolling");
            check(uniquePathsWithObstacles2d(grid) == uniquePathsBrute(grid), "2D == brute");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
