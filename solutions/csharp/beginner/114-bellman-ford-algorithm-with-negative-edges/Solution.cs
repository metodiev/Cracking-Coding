// Task 114 - Bellman-Ford algorithm (with negative edges). (Tier: Beginner, CTCI Ch 4)
using System;

public static class Solution
{
    public const long Inf = long.MaxValue / 4;

    /// <summary>edges rows: {u, v, w}. Returns distances or null when a negative cycle is reachable.</summary>
    public static long[]? BellmanFord(int n, int[][] edges, int src)
    {
        var dist = new long[n];
        for (int i = 0; i < n; i++) dist[i] = Inf;
        dist[src] = 0;
        for (int pass = 0; pass < n - 1; pass++)
        {
            bool changed = false;
            foreach (var e in edges)
            {
                if (dist[e[0]] != Inf && dist[e[0]] + e[2] < dist[e[1]])
                {
                    dist[e[1]] = dist[e[0]] + e[2];
                    changed = true;
                }
            }
            if (!changed) break;
        }
        foreach (var e in edges)
            if (dist[e[0]] != Inf && dist[e[0]] + e[2] < dist[e[1]]) return null;
        return dist;
    }

    public static void Main()
    {
        var dist = BellmanFord(4, new[] { new[] { 0, 1, 4 }, new[] { 0, 2, 5 }, new[] { 1, 2, -3 }, new[] { 2, 3, 2 } }, 0);
        Check(dist != null && dist[0] == 0 && dist[1] == 4 && dist[2] == 1 && dist[3] == 3, "negative edge");

        var nonNeg = BellmanFord(3, new[] { new[] { 0, 1, 1 }, new[] { 1, 2, 2 }, new[] { 0, 2, 4 } }, 0);
        Check(nonNeg![1] == 1 && nonNeg[2] == 3, "matches Dijkstra");

        Check(BellmanFord(3, new[] { new[] { 0, 1, 1 }, new[] { 1, 2, -2 }, new[] { 2, 1, -2 } }, 0) == null, "negative cycle");

        var far = BellmanFord(3, new[] { new[] { 1, 2, 1 } }, 0);
        Check(far![2] == Inf, "unreachable");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
