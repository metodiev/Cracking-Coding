// Task 274 - Check if a graph is bipartite (BFS 2-coloring). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns the colour array (0/1) or null when the graph is not bipartite.</summary>
    public static int[]? TwoColor(int n, int[][] edges)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        var color = new int[n];
        for (int i = 0; i < n; i++) color[i] = -1;
        for (int start = 0; start < n; start++)
        {
            if (color[start] != -1) continue;
            color[start] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                foreach (int v in adjacency[u])
                {
                    if (color[v] == -1)
                    {
                        color[v] = color[u] ^ 1;
                        queue.Enqueue(v);
                    }
                    else if (color[v] == color[u])
                    {
                        return null;
                    }
                }
            }
        }
        return color;
    }

    public static bool IsBipartite(int n, int[][] edges) => TwoColor(n, edges) != null;

    public static void Main()
    {
        Check(IsBipartite(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 } }), "even cycle");
        Check(!IsBipartite(3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } }), "triangle");
        Check(IsBipartite(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }), "path");
        Check(!IsBipartite(1, new[] { new[] { 0, 0 } }), "self-loop");
        Check(IsBipartite(5, Array.Empty<int[]>()), "no edges");
        Check(!IsBipartite(7, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 4, 5 }, new[] { 5, 6 }, new[] { 6, 4 } }), "one odd component");
        Check(IsBipartite(6, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 4, 5 } }), "even plus edge");

        var k33 = new List<int[]>();
        for (int i = 0; i < 3; i++) for (int j = 3; j < 6; j++) k33.Add(new[] { i, j });
        Check(IsBipartite(6, k33.ToArray()), "K3,3");

        var k4 = new List<int[]>();
        for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) k4.Add(new[] { i, j });
        Check(!IsBipartite(4, k4.ToArray()), "K4");

        var colors = TwoColor(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 } });
        Check(colors != null && colors[0] != colors[1] && colors[1] != colors[2], "colours alternate");
        Check(TwoColor(3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } }) == null, "no colouring for triangle");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
