// Task 265 - Count number of ways in matrix with obstacles (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>
    /// Rolling 1D DP: ways[c] = ways[c] (from above) + ways[c-1] (from the left), zeroed
    /// on obstacles (1 marks an obstacle, 0 a free cell). O(m*n) time, O(n) space.
    /// Empty grid / empty first row -> 0; blocked start or end -> 0; 1 x n and n x 1
    /// grids have the single monotone path, so 1 iff no cell is blocked.
    /// On identical grids this equals task 266's unique-path DP; on obstacle-free grids
    /// both equal C(m+n-2, m-1) - asserted below for several sizes.
    /// </summary>
    public static long CountWays(int[][] grid)
    {
        if (grid.Length == 0 || grid[0].Length == 0) return 0;
        int n = grid[0].Length;
        var ways = new long[n];
        ways[0] = 1;
        foreach (var row in grid)
            for (int c = 0; c < n; c++)
            {
                if (row[c] == 1) ways[c] = 0;
                else if (c > 0) ways[c] += ways[c - 1];
            }
        return ways[n - 1];
    }

    /// <summary>Recursive enumeration used only for cross-checks.</summary>
    public static long CountWaysBrute(int[][] grid) => Go(grid, 0, 0);

    private static long Go(int[][] grid, int r, int c)
    {
        if (grid[r][c] == 1) return 0;
        if (r == grid.Length - 1 && c == grid[0].Length - 1) return 1;
        long total = 0;
        if (r + 1 < grid.Length) total += Go(grid, r + 1, c);
        if (c + 1 < grid[0].Length) total += Go(grid, r, c + 1);
        return total;
    }

    /// <summary>Task 266's closed form C(m+n-2, m-1), used for obstacle-free cross-checks.</summary>
    public static long UniquePathsNoObstacles(int m, int n)
    {
        if (m <= 0 || n <= 0) return 0;
        return Comb(m + n - 2, m - 1);
    }

    private static long Comb(int n, int k)
    {
        long result = 1;
        for (int i = 1; i <= k; i++) result = result * (n - k + i) / i;
        return result;
    }

    private static int[][] FreeGrid(int m, int n)
    {
        var grid = new int[m][];
        for (int r = 0; r < m; r++) grid[r] = new int[n];
        return grid;
    }

    public static void Main()
    {
        Check(CountWays(new[] { new[] { 0, 0, 0 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 } }) == 6, "classic 3x3 free");
        Check(CountWays(new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 0 }, new[] { 0, 0, 0 } }) == 2, "classic obstacle case");
        Check(CountWays(new[] { new[] { 0, 1 }, new[] { 0, 0 } }) == 1, "only down then right");
        Check(CountWays(new[] { new[] { 1, 0 }, new[] { 0, 0 } }) == 0, "start blocked");
        Check(CountWays(new[] { new[] { 0, 0 }, new[] { 0, 1 } }) == 0, "end blocked");
        Check(CountWays(new[] { new[] { 0, 0, 0, 0, 0 } }) == 1, "1 x n");
        Check(CountWays(new[] { new[] { 0 }, new[] { 0 }, new[] { 0 } }) == 1, "n x 1");
        Check(CountWays(new[] { new[] { 0, 1, 0 } }) == 0, "1 x n blocked");
        Check(CountWays(new[] { new[] { 0 }, new[] { 1 }, new[] { 0 } }) == 0, "n x 1 blocked");
        Check(CountWays(new[] { new[] { 0, 1, 0 }, new[] { 0, 0, 0 } }) == 1, "two rows with a hole");
        Check(CountWays(new[] { new[] { 0, 1 }, new[] { 1, 0 } }) == 0, "both moves blocked");
        Check(CountWays(new int[0][]) == 0, "empty grid");
        Check(CountWays(new[] { new int[0] }) == 0, "empty first row");

        var sizeCases = new[] { (1, 1, 1L), (1, 5, 1L), (2, 2, 2L), (3, 3, 6L), (3, 7, 28L), (4, 4, 20L) };
        foreach (var (m, n, expected) in sizeCases)
            Check(CountWays(FreeGrid(m, n)) == expected && UniquePathsNoObstacles(m, n) == expected,
                $"closed form {m}x{n}");

        var rng = new Random(265);
        for (int test = 0; test < 300; test++)          // cross-check vs recursive enumeration
        {
            int m = 1 + rng.Next(5), n = 1 + rng.Next(5);
            var grid = new int[m][];
            for (int r = 0; r < m; r++)
            {
                grid[r] = new int[n];
                for (int c = 0; c < n; c++) grid[r][c] = rng.NextDouble() < 0.3 ? 1 : 0;
            }
            Check(CountWays(grid) == CountWaysBrute(grid), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
