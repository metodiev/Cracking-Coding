// Task 113 - Simple Dijkstra's algorithm (non-negative weights). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public const long Inf = long.MaxValue / 4;

    /// <summary>adjacency[u] = list of (v, weight); returns distances, Inf = unreachable.</summary>
    public static long[] Dijkstra(int n, List<(int to, int weight)>[] adjacency, int src)
    {
        var dist = new long[n];
        for (int i = 0; i < n; i++) dist[i] = Inf;
        dist[src] = 0;
        var heap = new PriorityQueue<(long d, int u), long>();
        heap.Enqueue((0, src), 0);
        while (heap.Count > 0)
        {
            var (d, u) = heap.Dequeue();
            if (d > dist[u]) continue;
            foreach (var (to, w) in adjacency[u])
            {
                long nd = d + w;
                if (nd < dist[to])
                {
                    dist[to] = nd;
                    heap.Enqueue((nd, to), nd);
                }
            }
        }
        return dist;
    }

    public static void Main()
    {
        var adjacency = new List<(int, int)>[4];
        adjacency[0] = new List<(int, int)> { (1, 1), (2, 4) };
        adjacency[1] = new List<(int, int)> { (2, 2), (3, 6) };
        adjacency[2] = new List<(int, int)> { (3, 3) };
        adjacency[3] = new List<(int, int)>();
        var dist = Dijkstra(4, adjacency, 0);
        Check(dist[0] == 0 && dist[1] == 1 && dist[2] == 3 && dist[3] == 6, "classic");

        var adjacency2 = new List<(int, int)>[3];
        adjacency2[0] = new List<(int, int)> { (1, 5) };
        adjacency2[1] = new List<(int, int)> { (0, 5) };
        adjacency2[2] = new List<(int, int)>();
        var dist2 = Dijkstra(3, adjacency2, 0);
        Check(dist2[2] == Inf, "unreachable is Inf");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
