// Task 101 - Count islands in boolean grid (DFS). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int CountIslands(int[][] grid)
    {
        if (grid.Length == 0 || grid[0].Length == 0) return 0;
        int rows = grid.Length, cols = grid[0].Length;
        var seen = new int[rows][];
        for (int r = 0; r < rows; r++) seen[r] = (int[])grid[r].Clone();
        int islands = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (seen[r][c] == 1)
                {
                    islands++;
                    Flood(seen, r, c);
                }
            }
        }
        return islands;
    }

    private static void Flood(int[][] grid, int sr, int sc)
    {
        int rows = grid.Length, cols = grid[0].Length;
        var stack = new Stack<(int R, int C)>();
        stack.Push((sr, sc));
        grid[sr][sc] = 0;
        int[] dr = { 1, -1, 0, 0 };
        int[] dc = { 0, 0, 1, -1 };
        while (stack.Count > 0)
        {
            var (r, c) = stack.Pop();
            for (int k = 0; k < 4; k++)
            {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1)
                {
                    grid[nr][nc] = 0;
                    stack.Push((nr, nc));
                }
            }
        }
    }

    public static void Main()
    {
        Check(CountIslands(new int[0][]) == 0, "empty grid");
        Check(CountIslands(new int[][] { new int[0] }) == 0, "empty row");
        Check(CountIslands(new int[][] { new[] { 0, 0 }, new[] { 0, 0 } }) == 0, "all water");
        Check(CountIslands(new int[][] { new[] { 1, 1 }, new[] { 1, 1 } }) == 1, "all land");
        Check(CountIslands(new int[][] { new[] { 1 } }) == 1, "single land");
        Check(CountIslands(new int[][] { new[] { 0 } }) == 0, "single water");
        Check(CountIslands(new int[][] { new[] { 1, 0 }, new[] { 0, 1 } }) == 2, "diagonal not connected");
        int[][] mixed = {
            new[] { 1, 1, 0, 0, 0 },
            new[] { 1, 1, 0, 0, 1 },
            new[] { 0, 0, 0, 1, 1 },
            new[] { 0, 0, 0, 0, 0 },
            new[] { 1, 0, 1, 0, 1 },
        };
        Check(CountIslands(mixed) == 5, "multiple islands");
        Check(mixed[0][0] == 1, "input not mutated");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
