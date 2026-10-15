// Task 112 - Prim's MST using heap. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>adjacency[u] = list of (v, weight). Returns (total, edgeCount) or null.</summary>
    public static long[]? Prim(int n, List<(int to, int weight)>[] adjacency)
    {
        var visited = new bool[n];
        long total = 0;
        int count = 0;
        var heap = new PriorityQueue<(int weight, int node, int parent), int>();
        heap.Enqueue((0, 0, -1), 0);
        while (heap.Count > 0 && count < Math.Max(0, n - 1))
        {
            var (weight, u, parent) = heap.Dequeue();
            if (visited[u]) continue;
            visited[u] = true;
            total += weight;
            if (parent != -1) count++;
            foreach (var (to, w) in adjacency[u])
                if (!visited[to]) heap.Enqueue((w, to, u), w);
        }
        return count == Math.Max(0, n - 1) ? new long[] { total, count } : null;
    }

    public static void Main()
    {
        var adjacency = new List<(int, int)>[5];
        adjacency[0] = new List<(int, int)> { (1, 4), (2, 3) };
        adjacency[1] = new List<(int, int)> { (0, 4), (2, 1), (3, 2) };
        adjacency[2] = new List<(int, int)> { (0, 3), (1, 1), (3, 4), (4, 5) };
        adjacency[3] = new List<(int, int)> { (1, 2), (2, 4), (4, 2) };
        adjacency[4] = new List<(int, int)> { (3, 2), (2, 5) };

        var result = Prim(5, adjacency);
        Check(result != null && result[0] == 8 && result[1] == 4, "classic graph");

        var single = new List<(int, int)>[1];
        single[0] = new List<(int, int)>();
        Check(Prim(1, single)![0] == 0, "single vertex");

        var disconnected = new List<(int, int)>[3];
        disconnected[0] = new List<(int, int)> { (1, 1) };
        disconnected[1] = new List<(int, int)> { (0, 1) };
        disconnected[2] = new List<(int, int)>();
        Check(Prim(3, disconnected) == null, "disconnected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
