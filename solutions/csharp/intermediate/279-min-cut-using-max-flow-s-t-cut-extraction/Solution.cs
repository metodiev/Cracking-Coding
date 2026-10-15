// Task 279 - Min cut using max flow (s-t cut extraction). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns (value, S, T, cut capacity) for the minimum s-t cut.</summary>
    public static (int Value, List<int> S, List<int> T, int CutCapacity) MinCut(int n, int[][] capacity, int source, int sink)
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
                    if (parent[v] == -1 && residual[u][v] > 0) { parent[v] = u; queue.Enqueue(v); }
                }
            }
            if (parent[sink] == -1) break;
            int bottleneck = int.MaxValue;
            for (int v = sink; v != source; v = parent[v]) bottleneck = Math.Min(bottleneck, residual[parent[v]][v]);
            for (int v = sink; v != source; v = parent[v])
            {
                int u = parent[v];
                residual[u][v] -= bottleneck;
                residual[v][u] += bottleneck;
            }
            total += bottleneck;
        }

        var side = new bool[n];
        side[source] = true;
        var walk = new Queue<int>();
        walk.Enqueue(source);
        while (walk.Count > 0)
        {
            int u = walk.Dequeue();
            for (int v = 0; v < n; v++)
            {
                if (!side[v] && residual[u][v] > 0) { side[v] = true; walk.Enqueue(v); }
            }
        }
        var inS = new List<int>();
        var inT = new List<int>();
        for (int v = 0; v < n; v++) (side[v] ? inS : inT).Add(v);
        int cutCapacity = 0;
        foreach (int u in inS)
            foreach (int v in inT)
                cutCapacity += capacity[u][v];
        return (total, inS, inT, cutCapacity);
    }

    public static int MinCutBruteForce(int n, int[][] capacity, int source, int sink)
    {
        var others = new List<int>();
        for (int v = 0; v < n; v++) if (v != source && v != sink) others.Add(v);
        int best = int.MaxValue;
        for (int mask = 0; mask < (1 << others.Count); mask++)
        {
            var side = new bool[n];
            side[source] = true;
            for (int i = 0; i < others.Count; i++) if ((mask >> i & 1) == 1) side[others[i]] = true;
            int cut = 0;
            for (int u = 0; u < n; u++)
            {
                if (!side[u]) continue;
                for (int v = 0; v < n; v++) if (!side[v]) cut += capacity[u][v];
            }
            best = Math.Min(best, cut);
        }
        return best;
    }

    public static int[][] MatrixFromEdges(int n, int[][] edges)
    {
        var capacity = new int[n][];
        for (int i = 0; i < n; i++) capacity[i] = new int[n];
        foreach (var edge in edges) capacity[edge[0]][edge[1]] += edge[2];
        return capacity;
    }

    private static int seed = 24680;

    private static int[][] LcgCapacities(int n)
    {
        var capacity = new int[n][];
        for (int i = 0; i < n; i++) capacity[i] = new int[n];
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 4 == 0) capacity[u][v] = seed % 10;
            }
        }
        return capacity;
    }

    public static void Main()
    {
        var clrs = MatrixFromEdges(6, new[]
        {
            new[] { 0, 1, 16 }, new[] { 0, 2, 13 }, new[] { 1, 2, 10 }, new[] { 2, 1, 4 }, new[] { 1, 3, 12 },
            new[] { 3, 2, 9 }, new[] { 2, 4, 14 }, new[] { 4, 3, 7 }, new[] { 3, 5, 20 }, new[] { 4, 5, 4 },
        });
        var result = MinCut(6, clrs, 0, 5);
        Check(result.Value == 23, "CLRS flow 23");
        Check(result.CutCapacity == result.Value, "cut capacity equals flow");
        Check(result.S.Contains(0) && result.T.Contains(5), "source in S, sink in T");
        Check(result.Value == MinCutBruteForce(6, clrs, 0, 5), "brute force min cut");

        var diamond = MatrixFromEdges(4, new[] { new[] { 0, 1, 3 }, new[] { 0, 2, 2 }, new[] { 1, 3, 2 }, new[] { 2, 3, 3 } });
        result = MinCut(4, diamond, 0, 3);
        Check(result.Value == 4 && result.CutCapacity == 4, "diamond cut");
        Check(MinCutBruteForce(4, diamond, 0, 3) == 4, "diamond brute force");

        var stranded = MatrixFromEdges(3, new[] { new[] { 0, 1, 7 } });
        result = MinCut(3, stranded, 0, 2);
        Check(result.Value == 0 && result.CutCapacity == 0, "stranded sink flow");
        Check(string.Join(",", result.S) == "0,1" && string.Join(",", result.T) == "2", "stranded partition");

        for (int trial = 0; trial < 60; trial++)
        {
            var capacity = LcgCapacities(6);
            var got = MinCut(6, capacity, 0, 5);
            Check(got.CutCapacity == got.Value, "cut equals flow " + trial);
            Check(got.Value == MinCutBruteForce(6, capacity, 0, 5), "brute force " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
