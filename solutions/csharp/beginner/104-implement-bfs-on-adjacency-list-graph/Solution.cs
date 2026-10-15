// Task 104 - Implement BFS on adjacency list graph. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<int> BfsOrder(Dictionary<int, List<int>> graph, int source)
    {
        var order = new List<int>();
        if (!graph.ContainsKey(source)) return order;
        var seen = new HashSet<int> { source };
        var queue = new Queue<int>();
        queue.Enqueue(source);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            order.Add(v);
            if (graph.TryGetValue(v, out var neighbors))
            {
                foreach (int w in neighbors)
                {
                    if (seen.Add(w)) queue.Enqueue(w);
                }
            }
        }
        return order;
    }

    public static Dictionary<int, int> BfsDistances(Dictionary<int, List<int>> graph, int source)
    {
        var dist = new Dictionary<int, int>();
        if (!graph.ContainsKey(source)) return dist;
        dist[source] = 0;
        var queue = new Queue<int>();
        queue.Enqueue(source);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            if (graph.TryGetValue(v, out var neighbors))
            {
                foreach (int w in neighbors)
                {
                    if (!dist.ContainsKey(w))
                    {
                        dist[w] = dist[v] + 1;
                        queue.Enqueue(w);
                    }
                }
            }
        }
        return dist;
    }

    public static void Main()
    {
        var g = MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 3 }, new[] { 3, 3 } });
        Check(SameList(BfsOrder(g, 0), 0, 1, 2, 3), "order with cycle");
        var dist = BfsDistances(g, 0);
        Check(dist.Count == 4 && dist[0] == 0 && dist[1] == 1 && dist[2] == 1 && dist[3] == 2, "distances");

        var parts = MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 2, 3 }, new[] { 3, 2 } });
        parts[4] = new List<int>();
        Check(SameList(BfsOrder(parts, 0), 0, 1), "disconnected order");
        var d0 = BfsDistances(parts, 0);
        Check(d0.Count == 2 && d0[0] == 0 && d0[1] == 1, "disconnected distances");
        Check(SameList(BfsOrder(parts, 4), 4), "loner vertex");
        var d4 = BfsDistances(parts, 4);
        Check(d4.Count == 1 && d4[4] == 0, "loner distance");
        Check(BfsOrder(parts, 99).Count == 0, "source not in graph");
        Check(BfsDistances(parts, 99).Count == 0, "distances for missing source");
        Check(BfsOrder(new Dictionary<int, List<int>>(), 0).Count == 0, "empty graph");
        Check(BfsDistances(new Dictionary<int, List<int>>(), 0).Count == 0, "empty graph distances");
        var twoCycles = MakeGraph(new int[][] { new[] { 5, 6 }, new[] { 6, 5 }, new[] { 6, 7 }, new[] { 7, 5 } });
        Check(SameList(BfsOrder(twoCycles, 5), 5, 6, 7), "two-node cycle");
        Console.WriteLine("PASS");
    }

    private static Dictionary<int, List<int>> MakeGraph(int[][] edges)
    {
        var g = new Dictionary<int, List<int>>();
        foreach (var e in edges)
        {
            if (!g.TryGetValue(e[0], out var list))
            {
                list = new List<int>();
                g[e[0]] = list;
            }
            list.Add(e[1]);
        }
        return g;
    }

    private static bool SameList(List<int> actual, params int[] expected)
    {
        if (actual.Count != expected.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (actual[i] != expected[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
