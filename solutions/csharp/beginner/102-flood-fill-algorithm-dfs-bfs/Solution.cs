// Task 102 - Flood fill algorithm (DFS/BFS). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[][] FloodFill(int[][] image, int sr, int sc, int newColor)
    {
        if (image.Length == 0 || image[0].Length == 0) return image;
        int rows = image.Length, cols = image[0].Length;
        int startColor = image[sr][sc];
        if (startColor == newColor) return image;
        var queue = new Queue<(int R, int C)>();
        image[sr][sc] = newColor;
        queue.Enqueue((sr, sc));
        int[] dr = { 1, -1, 0, 0 };
        int[] dc = { 0, 0, 1, -1 };
        while (queue.Count > 0)
        {
            var (r, c) = queue.Dequeue();
            for (int k = 0; k < 4; k++)
            {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && image[nr][nc] == startColor)
                {
                    image[nr][nc] = newColor;
                    queue.Enqueue((nr, nc));
                }
            }
        }
        return image;
    }

    public static void Main()
    {
        var classic = new int[][] { new[] { 1, 1, 1 }, new[] { 1, 1, 0 }, new[] { 1, 0, 1 } };
        Check(Same(FloodFill(classic, 1, 1, 2), new int[][] { new[] { 2, 2, 2 }, new[] { 2, 2, 0 }, new[] { 2, 0, 1 } }), "classic fill");
        var noop = new int[][] { new[] { 0, 0 }, new[] { 0, 0 } };
        Check(Same(FloodFill(noop, 0, 0, 0), new int[][] { new[] { 0, 0 }, new[] { 0, 0 } }), "same color no-op");
        var patch = new int[][] { new[] { 1, 1, 0 }, new[] { 0, 1, 0 } };
        Check(Same(FloodFill(patch, 0, 0, 5), new int[][] { new[] { 5, 5, 0 }, new[] { 0, 5, 0 } }), "patch");
        var single = new int[][] { new[] { 7 } };
        Check(Same(FloodFill(single, 0, 0, 9), new int[][] { new[] { 9 } }), "single cell");
        Check(FloodFill(new int[0][], 0, 0, 1).Length == 0, "empty image");
        var mix = new int[][] { new[] { 1, 1 }, new[] { 1, 0 } };
        Check(Same(FloodFill(mix, 0, 0, 0), new int[][] { new[] { 0, 0 }, new[] { 0, 0 } }), "fill with zero");
        Console.WriteLine("PASS");
    }

    private static bool Same(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].Length != b[i].Length) return false;
            for (int j = 0; j < a[i].Length; j++)
                if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
