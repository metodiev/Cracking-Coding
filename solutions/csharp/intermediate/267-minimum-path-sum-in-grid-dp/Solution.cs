// Task 267 - Minimum path sum in grid (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>
    /// Rolling 1D DP: dp[c] = min(dp[c] from above, dp[c-1] from the left) + grid[r][c];
    /// the first row and column each have a single predecessor. O(m*n) time, O(n) space.
    /// Empty grid / empty first row raises ArgumentException; a single cell returns its
    /// value; 1 x n and n x 1 grids have a single path. Negatives are allowed.
    /// </summary>
    public static long MinPathSum(int[][] grid)
    {
        if (grid.Length == 0 || grid[0].Length == 0) throw new ArgumentException("grid must be a non-empty rectangle");
        int n = grid[0].Length;
        var dp = new long[n];
        for (int r = 0; r < grid.Length; r++)
            for (int c = 0; c < n; c++)
            {
                if (r == 0 && c == 0) dp[c] = grid[r][c];
                else if (r == 0) dp[c] = dp[c - 1] + grid[r][c];            // only from the left
                else if (c == 0) dp[c] += grid[r][c];                       // only from above
                else dp[c] = Math.Min(dp[c], dp[c - 1]) + grid[r][c];
            }
        return dp[n - 1];
    }

    /// <summary>Recursive enumeration of all paths; used only for cross-checks.</summary>
    public static long MinPathSumBrute(int[][] grid) => Go(grid, 0, 0);

    private static long Go(int[][] grid, int r, int c)
    {
        if (r == grid.Length - 1 && c == grid[0].Length - 1) return grid[r][c];
        long? best = null;
        if (r + 1 < grid.Length) best = Go(grid, r + 1, c);
        if (c + 1 < grid[0].Length)
        {
            long right = Go(grid, r, c + 1);
            best = best == null ? right : Math.Min(best.Value, right);
        }
        return grid[r][c] + best!.Value;
    }

    public static void Main()
    {
        Check(MinPathSum(new[] { new[] { 1, 3, 1 }, new[] { 1, 5, 1 }, new[] { 4, 2, 1 } }) == 7, "classic = 7");
        Check(MinPathSum(new[] { new[] { 5 } }) == 5, "single cell");
        Check(MinPathSum(new[] { new[] { -3 } }) == -3, "single negative cell");
        Check(MinPathSum(new[] { new[] { 1, 2, 3 } }) == 6, "1 x n single path");
        Check(MinPathSum(new[] { new[] { 1 }, new[] { 2 }, new[] { 3 } }) == 6, "n x 1 single path");
        Check(MinPathSum(new[] { new[] { 1, 2, 3 }, new[] { 4, 100, 6 } }) == 12, "route above the huge cell");
        Check(MinPathSum(new[] { new[] { 1, 100 }, new[] { 1, 1 } }) == 3, "down first, then right");
        Check(MinPathSum(new[] { new[] { 1, -3 }, new[] { 1, -5 } }) == -7, "negatives allowed");

        int emptyThrows = 0;
        try { MinPathSum(new int[0][]); } catch (ArgumentException) { emptyThrows++; }
        try { MinPathSum(new[] { new int[0] }); } catch (ArgumentException) { emptyThrows++; }
        Check(emptyThrows == 2, "empty grid cases raise");

        var rng = new Random(267);
        for (int test = 0; test < 300; test++)          // cross-check vs recursive enumeration
        {
            int m = 1 + rng.Next(5), n = 1 + rng.Next(5);
            var grid = new int[m][];
            for (int r = 0; r < m; r++)
            {
                grid[r] = new int[n];
                for (int c = 0; c < n; c++) grid[r][c] = rng.Next(15) - 5;
            }
            Check(MinPathSum(grid) == MinPathSumBrute(grid), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
