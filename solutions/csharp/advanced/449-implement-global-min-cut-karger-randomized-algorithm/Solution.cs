// Task 449 - Implement global min-cut (Karger randomized algorithm). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal class Lcg
{
    private int state;

    public Lcg(int seed)
    {
        state = seed % 65537;
    }

    public int Next()
    {
        state = (state * 75 + 74) % 65537;
        return state;
    }

    public int Below(int bound) => Next() % bound;
}

internal class CutResult
{
    public readonly int Value;
    public readonly int[] SideA;
    public readonly int[] SideB;

    public CutResult(int value, int[] sideA, int[] sideB)
    {
        Value = value;
        SideA = sideA;
        SideB = sideB;
    }
}

internal static class Karger
{
    public static CutResult MinCut(int n, int[][] edges, int seed)
    {
        var rng = new Lcg(seed);
        var label = new int[n];
        for (int i = 0; i < n; i++) label[i] = i;
        var remaining = new List<int[]>();
        foreach (int[] e in edges) remaining.Add(new[] { e[0], e[1] });
        int components = n;
        while (components > 2)
        {
            int[] pick = remaining[rng.Below(remaining.Count)];
            int a = label[pick[0]];
            int b = label[pick[1]];
            for (int i = 0; i < n; i++)
            {
                if (label[i] == b) label[i] = a;
            }
            var next = new List<int[]>();
            foreach (int[] e in remaining)
            {
                if (label[e[0]] != label[e[1]]) next.Add(e);
            }
            remaining = next;
            components--;
        }
        int first = label[0];
        var sideA = new List<int>();
        var sideB = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (label[i] == first) sideA.Add(i);
            else sideB.Add(i);
        }
        return new CutResult(remaining.Count, sideA.ToArray(), sideB.ToArray());
    }
}

public static class Solution
{
    private static int BruteMinCut(int n, int[][] edges)
    {
        int best = int.MaxValue;
        for (int mask = 1; mask < 1 << (n - 1); mask++)
        {
            int cut = 0;
            foreach (int[] e in edges)
            {
                int bu = e[0] > 0 ? (mask >> (e[0] - 1)) & 1 : 0;
                int bv = e[1] > 0 ? (mask >> (e[1] - 1)) & 1 : 0;
                if (bu != bv) cut++;
            }
            best = Math.Min(best, cut);
        }
        return best;
    }

    private static void CheckWitness(int n, int[][] edges, int value, int[] sideA, int[] sideB)
    {
        var onA = new bool[n];
        foreach (int v in sideA) onA[v] = true;
        foreach (int v in sideB)
        {
            if (onA[v]) throw new Exception("FAILED: sides overlap");
        }
        if (sideA.Length == 0 || sideB.Length == 0) throw new Exception("FAILED: empty side");
        if (sideA.Length + sideB.Length != n) throw new Exception("FAILED: not a partition");
        int crossing = 0;
        foreach (int[] e in edges)
        {
            if (onA[e[0]] != onA[e[1]]) crossing++;
        }
        if (crossing != value) throw new Exception("FAILED: witness crossing " + crossing + " != " + value);
    }

    private static int[] KargerBestOf(int n, int[][] edges, int baseSeed, int trials, int expected)
    {
        int best = int.MaxValue;
        int successes = 0;
        for (int t = 0; t < trials; t++)
        {
            int seed = (baseSeed * 31 + t * 17 + 1) % 65537;
            CutResult result = Karger.MinCut(n, edges, seed);
            CheckWitness(n, edges, result.Value, result.SideA, result.SideB);
            if (result.Value < expected) throw new Exception("FAILED: cut below minimum " + result.Value);
            if (result.Value == expected) successes++;
            best = Math.Min(best, result.Value);
        }
        return new[] { best, successes };
    }

    private static int seed;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<EdgeGraph> LcgGraphs(int count, int maxN, int baseSeed)
    {
        seed = baseSeed;
        var graphs = new List<EdgeGraph>();
        for (int g = 0; g < count; g++)
        {
            int n = 3 + seed % (maxN - 2);
            var edges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                int parent = seed % node;
                edges.Add(new[] { node, parent });
            }
            int extras = seed % (n + 2);
            for (int e = 0; e < extras; e++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                if (u != v) edges.Add(new[] { u, v });
            }
            graphs.Add(new EdgeGraph(n, edges.ToArray()));
        }
        return graphs;
    }

    public static void Main()
    {
        int[][] triangle = { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } };
        Check(BruteMinCut(3, triangle) == 2, "triangle brute");
        foreach (int seedValue in new[] { 1, 2, 3, 65536 })
        {
            CutResult result = Karger.MinCut(3, triangle, seedValue);
            CheckWitness(3, triangle, result.Value, result.SideA, result.SideB);
            Check(result.Value == 2, "triangle seed " + seedValue);
        }

        var k4List = new List<int[]>();
        for (int u = 0; u < 4; u++)
        {
            for (int v = u + 1; v < 4; v++) k4List.Add(new[] { u, v });
        }
        int[][] k4 = k4List.ToArray();
        Check(BruteMinCut(4, k4) == 3, "K4 brute");
        int[] k4Stats = KargerBestOf(4, k4, 42, 60, 3);
        Check(k4Stats[0] == 3, "K4 best");
        Check(k4Stats[1] > 0, "K4 successes");

        int totalTrials = 0;
        foreach (EdgeGraph graph in LcgGraphs(3, 6, 449))
        {
            int expected = BruteMinCut(graph.N, graph.Edges);
            foreach (int baseSeed in new[] { 1, 42, 1234, 65536 })
            {
                int[] stats = KargerBestOf(graph.N, graph.Edges, baseSeed, 400, expected);
                Check(stats[0] == expected, "LCG best seed " + baseSeed);
                Check(stats[1] > 0, "LCG successes seed " + baseSeed);
                totalTrials += 400;
            }
        }
        Check(totalTrials == 4800, "trial coverage");

        var k6List = new List<int[]>();
        for (int u = 0; u < 6; u++)
        {
            for (int v = u + 1; v < 6; v++) k6List.Add(new[] { u, v });
        }
        int[][] k6 = k6List.ToArray();
        Check(BruteMinCut(6, k6) == 5, "K6 brute");
        int[] k6Stats = KargerBestOf(6, k6, 1234, 400, 5);
        Check(k6Stats[0] == 5, "K6 best");
        Check(k6Stats[1] > 0, "K6 successes");

        int[][] parallel = { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 0, 1 } };
        Check(BruteMinCut(2, parallel) == 3, "parallel brute");
        int[] parallelStats = KargerBestOf(2, parallel, 7, 25, 3);
        Check(parallelStats[0] == 3 && parallelStats[1] == 25, "parallel trials");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

internal class EdgeGraph
{
    public readonly int N;
    public readonly int[][] Edges;

    public EdgeGraph(int n, int[][] edges)
    {
        N = n;
        Edges = edges;
    }
}
