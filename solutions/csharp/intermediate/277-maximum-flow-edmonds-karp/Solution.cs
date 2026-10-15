// Task 277 - Maximum flow (Edmonds-Karp). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>capacity[i][j] is the capacity of the edge i -> j (0 when there is none).</summary>
    public static int EdmondsKarp(int n, int[][] capacity, int source, int sink)
    {
        var residual = new int[n][];
        for (int i = 0; i < n; i++) residual[i] = (int[])capacity[i].Clone();
        int total = 0;
        while (true)
        {
            var parent = Enumerable.Repeat(-1, n).ToArray();
            parent[source] = source;
            var queue = new Queue<int>();
            queue.Enqueue(source);
            while (queue.Count > 0 && parent[sink] == -1)
            {
                int u = queue.Dequeue();
                for (int v = 0; v < n; v++)
                {
                    if (parent[v] == -1 && residual[u][v] > 0)
                    {
                        parent[v] = u;
                        queue.Enqueue(v);
                    }
                }
            }
            if (parent[sink] == -1) return total;

            int bottleneck = int.MaxValue;
            for (int v = sink; v != source; v = parent[v])
                bottleneck = Math.Min(bottleneck, residual[parent[v]][v]);
            for (int v = sink; v != source; v = parent[v])
            {
                int u = parent[v];
                residual[u][v] -= bottleneck;
                residual[v][u] += bottleneck;
            }
            total += bottleneck;
        }
    }

    public static int[][] MatrixFromEdges(int n, int[][] edges)
    {
        var capacity = new int[n][];
        for (int i = 0; i < n; i++) capacity[i] = new int[n];
        foreach (var edge in edges) capacity[edge[0]][edge[1]] += edge[2];
        return capacity;
    }

    public static void Main()
    {
        var clrs = MatrixFromEdges(6, new[]
        {
            new[] { 0, 1, 16 }, new[] { 0, 2, 13 }, new[] { 1, 2, 10 }, new[] { 2, 1, 4 }, new[] { 1, 3, 12 },
            new[] { 3, 2, 9 }, new[] { 2, 4, 14 }, new[] { 4, 3, 7 }, new[] { 3, 5, 20 }, new[] { 4, 5, 4 },
        });
        Check(EdmondsKarp(6, clrs, 0, 5) == 23, "CLRS 23");

        Check(EdmondsKarp(4, MatrixFromEdges(4, new[] { new[] { 0, 1, 3 }, new[] { 0, 2, 2 }, new[] { 1, 3, 2 }, new[] { 2, 3, 3 } }), 0, 3) == 4, "diamond");
        Check(EdmondsKarp(3, MatrixFromEdges(3, new[] { new[] { 0, 1, 3 }, new[] { 1, 2, 5 } }), 0, 2) == 3, "bottleneck path");
        Check(EdmondsKarp(3, MatrixFromEdges(3, new[] { new[] { 0, 1, 1 } }), 0, 2) == 0, "unreachable sink");
        Check(EdmondsKarp(2, MatrixFromEdges(2, Array.Empty<int[]>()), 0, 1) == 0, "no edges");

        var crossing = MatrixFromEdges(4, new[] { new[] { 0, 1, 1 }, new[] { 0, 2, 1 }, new[] { 1, 2, 1 }, new[] { 1, 3, 1 }, new[] { 2, 3, 1 } });
        Check(EdmondsKarp(4, crossing, 0, 3) == 2, "back-edge augmentation");

        Check(EdmondsKarp(3, MatrixFromEdges(3, new[] { new[] { 0, 1, 2 }, new[] { 0, 1, 3 }, new[] { 1, 2, 4 } }), 0, 2) == 4, "parallel edges add up");

        int totalOut = clrs[0].Sum();
        Check(EdmondsKarp(6, clrs, 0, 5) <= totalOut, "bounded by source capacity");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
