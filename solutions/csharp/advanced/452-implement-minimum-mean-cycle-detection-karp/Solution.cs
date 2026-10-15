// Task 452 - Implement minimum mean cycle detection (Karp). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal static class Karp
{
    private const int Inf = 1_000_000_000;

    private static int Gcd(int a, int b)
    {
        a = Math.Abs(a);
        while (b != 0)
        {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    public static (int num, int den) ReduceFraction(int num, int den)
    {
        int g = Gcd(num, den);
        return (num / g, den / g);
    }

    private static bool FracLess((int num, int den) a, (int num, int den) b)
        => a.num * b.den < b.num * a.den;

    private static bool FracGreater((int num, int den) a, (int num, int den) b)
        => a.num * b.den > b.num * a.den;

    /// <summary>Min mean cycle weight as a reduced fraction, or null when acyclic.</summary>
    public static (int num, int den)? KarpMinMeanCycle(int n, List<int[]> edges)
    {
        var d = new int[n + 1][];
        for (int k = 0; k <= n; k++)
        {
            d[k] = new int[n];
            for (int v = 0; v < n; v++) d[k][v] = Inf;
        }
        for (int v = 0; v < n; v++) d[0][v] = 0;
        for (int k = 1; k <= n; k++)
        {
            foreach (var e in edges)
            {
                int u = e[0], v = e[1], w = e[2];
                if (d[k - 1][u] < Inf && d[k - 1][u] + w < d[k][v]) d[k][v] = d[k - 1][u] + w;
            }
        }
        (int num, int den)? best = null;
        for (int v = 0; v < n; v++)
        {
            if (d[n][v] >= Inf) continue;
            (int num, int den)? local = null;
            for (int k = 0; k < n; k++)
            {
                if (d[k][v] >= Inf) continue;
                var cand = ReduceFraction(d[n][v] - d[k][v], n - k);
                if (local == null || FracGreater(cand, local.Value)) local = cand;
            }
            if (best == null || FracLess(local.Value, best.Value)) best = local;
        }
        return best;
    }

    /// <summary>Minimum mean over all simple cycles by enumeration (each cycle from its min vertex).</summary>
    public static (int num, int den)? BruteMinMeanCycle(int n, List<int[]> edges)
    {
        var adj = new List<int[]>[n];
        for (int i = 0; i < n; i++) adj[i] = new List<int[]>();
        foreach (var e in edges) adj[e[0]].Add(new[] { e[1], e[2] });
        (int num, int den)? best = null;
        var path = new List<int>();
        for (int s = 0; s < n; s++)
        {
            path.Add(s);
            Walk(adj, path, 0, s, ref best);
            path.RemoveAt(path.Count - 1);
        }
        return best;
    }

    private static void Walk(List<int[]>[] adj, List<int> path, int total, int start,
                             ref (int num, int den)? best)
    {
        int cur = path[path.Count - 1];
        foreach (var e in adj[cur])
        {
            int v = e[0], w = e[1];
            if (v == start)
            {
                var cand = ReduceFraction(total + w, path.Count);
                if (best == null || FracLess(cand, best.Value)) best = cand;
            }
            else if (v > start && !path.Contains(v))
            {
                path.Add(v);
                Walk(adj, path, total + w, start, ref best);
                path.RemoveAt(path.Count - 1);
            }
        }
    }

    public static List<(int n, List<int[]> edges)> LcgDigraphs(int count, int seed, int minN, int maxN)
    {
        var graphs = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = minN + seed % (maxN - minN + 1);
            var edges = new List<int[]>();
            for (int node = 0; node < n - 1; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                edges.Add(new[] { node, node + 1, seed % 15 - 5 });
            }
            seed = (seed * 75 + 74) % 65537;
            edges.Add(new[] { n - 1, 0, seed % 15 - 5 });
            for (int k = 0; k < 2 * n; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                if (seed % 100 < 20)
                {
                    seed = (seed * 75 + 74) % 65537;
                    edges.Add(new[] { u, u, seed % 9 - 4 });
                    continue;
                }
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.Add(new[] { u, v, seed % 15 - 5 });
            }
            graphs.Add((n, edges));
        }
        return graphs;
    }
}

public static class Solution
{
    public static void Main()
    {
        var fourCycle = new List<int[]> { new[] { 0, 1, 1 }, new[] { 1, 2, 1 }, new[] { 2, 3, 1 }, new[] { 3, 0, 1 }, new[] { 2, 0, 5 } };
        Check(Karp.KarpMinMeanCycle(4, fourCycle) == (1, 1), "4-cycle mean 1 beats shortcut");
        Check(Karp.BruteMinMeanCycle(4, fourCycle) == (1, 1), "4-cycle brute");

        var negative = new List<int[]> { new[] { 0, 1, -2 }, new[] { 1, 2, -1 }, new[] { 2, 0, -3 } };
        Check(Karp.KarpMinMeanCycle(3, negative) == (-2, 1), "negative karp");
        Check(Karp.BruteMinMeanCycle(3, negative) == (-2, 1), "negative brute");

        var selfLoop = new List<int[]> { new[] { 0, 0, 7 } };
        Check(Karp.KarpMinMeanCycle(1, selfLoop) == (7, 1), "self loop karp");
        Check(Karp.BruteMinMeanCycle(1, selfLoop) == (7, 1), "self loop brute");

        var twoCycles = new List<int[]> { new[] { 0, 1, 2 }, new[] { 1, 0, 2 }, new[] { 1, 2, -1 }, new[] { 2, 1, -1 } };
        Check(Karp.KarpMinMeanCycle(3, twoCycles) == (-1, 1), "two cycles karp");
        Check(Karp.BruteMinMeanCycle(3, twoCycles) == (-1, 1), "two cycles brute");

        var half = new List<int[]> { new[] { 0, 1, 1 }, new[] { 1, 0, 0 } };
        Check(Karp.KarpMinMeanCycle(2, half) == (1, 2), "fraction karp");
        Check(Karp.BruteMinMeanCycle(2, half) == (1, 2), "fraction brute");

        var dag = new List<int[]> { new[] { 0, 1, 1 }, new[] { 1, 2, 2 } };
        Check(Karp.KarpMinMeanCycle(3, dag) == null, "DAG karp");
        Check(Karp.BruteMinMeanCycle(3, dag) == null, "DAG brute");

        int checkedCount = 0;
        foreach (int seed in new[] { 4521, 4522, 4523, 4524 })
        {
            foreach (var (n, edges) in Karp.LcgDigraphs(5, seed, 3, 6))
            {
                var expected = Karp.BruteMinMeanCycle(n, edges);
                var got = Karp.KarpMinMeanCycle(n, edges);
                Check(expected != null, "LCG has cycle");
                Check(got == expected, "LCG min mean " + got + " vs " + expected);
                checkedCount++;
            }
        }
        Check(checkedCount == 20, "graph count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
