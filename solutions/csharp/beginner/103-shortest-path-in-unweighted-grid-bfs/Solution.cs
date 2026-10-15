// Task 103 - Shortest path in unweighted grid (BFS). (Tier: Beginner, CTCI Ch 11)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int ShortestPath(int[][] grid, (int R, int C) start, (int R, int C) goal)
    {
        if (grid.Length == 0 || grid[0].Length == 0) return -1;
        int rows = grid.Length, cols = grid[0].Length;
        if (grid[start.R][start.C] == 1 || grid[goal.R][goal.C] == 1) return -1;
        if (start == goal) return 0;
        var dist = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            dist[r] = new int[cols];
            Array.Fill(dist[r], -1);
        }
        dist[start.R][start.C] = 0;
        var queue = new Queue<(int R, int C)>();
        queue.Enqueue(start);
        int[] dr = { 1, -1, 0, 0 };
        int[] dc = { 0, 0, 1, -1 };
        while (queue.Count > 0)
        {
            var (r, c) = queue.Dequeue();
            for (int k = 0; k < 4; k++)
            {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 0 && dist[nr][nc] == -1)
                {
                    dist[nr][nc] = dist[r][c] + 1;
                    if (nr == goal.R && nc == goal.C) return dist[nr][nc];
                    queue.Enqueue((nr, nc));
                }
            }
        }
        return -1;
    }

    public static void Main()
    {
        var open3 = new int[][] { new[] { 0, 0, 0 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 } };
        Check(ShortestPath(open3, (0, 0), (2, 2)) == 4, "open grid");
        Check(ShortestPath(open3, (0, 0), (0, 0)) == 0, "start == goal");
        Check(ShortestPath(new int[][] { new[] { 0 } }, (0, 0), (0, 0)) == 0, "single cell");
        var detour = new int[][] { new[] { 0, 0, 0 }, new[] { 1, 1, 0 }, new[] { 0, 0, 0 } };
        Check(ShortestPath(detour, (0, 0), (2, 0)) == 6, "forced detour");
        var split = new int[][] { new[] { 0, 1 }, new[] { 1, 0 } };
        Check(ShortestPath(split, (0, 0), (1, 1)) == -1, "diagonal unreachable");
        var blocked = new int[][] { new[] { 0, 1 }, new[] { 0, 0 } };
        Check(ShortestPath(blocked, (0, 0), (0, 1)) == -1, "blocked goal");
        Check(ShortestPath(blocked, (0, 1), (0, 0)) == -1, "blocked start");
        Check(ShortestPath(new int[0][], (0, 0), (0, 0)) == -1, "empty grid");
        var direct = new int[][] { new[] { 0, 0, 0, 0 }, new[] { 0, 0, 0, 0 }, new[] { 0, 0, 0, 0 }, new[] { 0, 0, 0, 0 } };
        Check(ShortestPath(direct, (0, 0), (3, 3)) == 6, "manhattan distance");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
