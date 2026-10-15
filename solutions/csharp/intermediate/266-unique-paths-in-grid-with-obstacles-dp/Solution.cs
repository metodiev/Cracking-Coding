// Task 266 - Unique paths in grid with obstacles (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>
    /// Explicit m x n DP for the number of monotone right/down paths (1 = obstacle).
    /// With no obstacles this reduces to the closed form C(m+n-2, m-1)
    /// (UniquePathsNoObstacles); obstacles zero out blocked cells, so they can only
    /// lower the count. A blocked start or end yields 0. O(m*n) time and space.
    /// </summary>
    public static long UniquePathsWithObstacles2d(int[][] grid)
    {
        if (grid.Length == 0 || grid[0].Length == 0) return 0;
        int m = grid.Length, n = grid[0].Length;
        var dp = new long[m, n];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
            {
                if (grid[r][c] == 1) dp[r, c] = 0;
                else if (r == 0 && c == 0) dp[r, c] = 1;
                else dp[r, c] = (r > 0 ? dp[r - 1, c] : 0) + (c > 0 ? dp[r, c - 1] : 0);
            }
        return dp[m - 1, n - 1];
    }

    /// <summary>Task 265's rolling 1D DP; identical grids must give identical counts.</summary>
    public static long CountWays265(int[][] grid)
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

    /// <summary>Closed form for an obstacle-free m x n grid: C(m+n-2, m-1).</summary>
    public static long UniquePathsNoObstacles(int m, int n)
    {
        if (m <= 0 || n <= 0) return 0;
        long result = 1;
        for (int i = 1; i <= m - 1; i++) result = result * (n - 1 + i) / i;
        return result;
    }

    /// <summary>Recursive enumeration used only for cross-checks.</summary>
    public static long UniquePathsBrute(int[][] grid) => Go(grid, 0, 0);

    private static long Go(int[][] grid, int r, int c)
    {
        if (grid[r][c] == 1) return 0;
        if (r == grid.Length - 1 && c == grid[0].Length - 1) return 1;
        long total = 0;
        if (r + 1 < grid.Length) total += Go(grid, r + 1, c);
        if (c + 1 < grid[0].Length) total += Go(grid, r, c + 1);
        return total;
    }

    private static int[][] FreeGrid(int m, int n)
    {
        var grid = new int[m][];
        for (int r = 0; r < m; r++) grid[r] = new int[n];
        return grid;
    }

    public static void Main()
    {
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 0, 0 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 } }) == 6,
            "classic 3x3 = 6");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 0 }, new[] { 0, 0, 0 } }) == 2,
            "obstacle variant");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 1 }, new[] { 0, 0 } }) == 1, "detour down");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 0 }, new[] { 0, 1 } }) == 0, "goal blocked");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 1, 0 }, new[] { 0, 0 } }) == 0, "start blocked");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 0 }, new[] { 1, 0 } }) == 1, "detour right");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 0, 0, 0, 0 } }) == 1, "1 x n");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0 }, new[] { 0 }, new[] { 0 } }) == 1, "n x 1");
        Check(UniquePathsWithObstacles2d(new[] { new[] { 0, 1, 0 } }) == 0, "1 x n blocked");
        Check(UniquePathsWithObstacles2d(new int[0][]) == 0, "empty grid");
        Check(UniquePathsWithObstacles2d(new[] { new int[0] }) == 0, "empty first row");

        var shared = new[]
        {
            new[] { new[] { 0, 0, 0 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 } },
            new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 0 }, new[] { 0, 0, 0 } },
            new[] { new[] { 0, 1 }, new[] { 0, 0 } },
            new[] { new[] { 0, 0 }, new[] { 0, 1 } },
            new[] { new[] { 1, 0 }, new[] { 0, 0 } },
            new[] { new[] { 0, 1 }, new[] { 1, 0 } },
            new[] { new[] { 0, 1, 0 }, new[] { 0, 0, 0 } },
            new[] { new[] { 0, 0, 0, 0 }, new[] { 1, 0, 0, 0 }, new[] { 0, 0, 1, 0 }, new[] { 0, 0, 0, 0 } },
        };
        foreach (var grid in shared)                // 265's counter agrees on identical grids
            Check(UniquePathsWithObstacles2d(grid) == CountWays265(grid), "agrees with the 265 counter");

        var sizeCases = new[] { (1, 1, 1L), (1, 5, 1L), (2, 2, 2L), (3, 3, 6L), (3, 7, 28L), (4, 4, 20L) };
        foreach (var (m, n, expected) in sizeCases)
            Check(UniquePathsWithObstacles2d(FreeGrid(m, n)) == expected && UniquePathsNoObstacles(m, n) == expected,
                $"closed form {m}x{n}");

        var rng = new Random(266);
        for (int test = 0; test < 300; test++)      // 2D == 265's rolling DP == recursive enumeration
        {
            int m = 1 + rng.Next(5), n = 1 + rng.Next(5);
            var grid = new int[m][];
            for (int r = 0; r < m; r++)
            {
                grid[r] = new int[n];
                for (int c = 0; c < n; c++) grid[r][c] = rng.NextDouble() < 0.3 ? 1 : 0;
            }
            Check(UniquePathsWithObstacles2d(grid) == CountWays265(grid), "2D == rolling");
            Check(UniquePathsWithObstacles2d(grid) == UniquePathsBrute(grid), "2D == brute");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
