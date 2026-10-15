// Task 450 - Implement stoer-wagner minimum cut algorithm. (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal class MinCutInfo
{
    public readonly int Value;
    public readonly int[] Side;

    public MinCutInfo(int value, int[] side)
    {
        Value = value;
        Side = side;
    }
}

internal static class StoerWagner
{
    public static MinCutInfo MinCut(int n, int[][] edges)
    {
        var weight = new long[n][];
        for (int i = 0; i < n; i++) weight[i] = new long[n];
        foreach (int[] e in edges)
        {
            if (e[0] == e[1]) continue;
            weight[e[0]][e[1]] += e[2];
            weight[e[1]][e[0]] += e[2];
        }
        var vertices = new List<int>();
        for (int i = 0; i < n; i++) vertices.Add(i);
        var groups = new List<List<int>>();
        for (int i = 0; i < n; i++) groups.Add(new List<int> { i });
        long best = long.MaxValue;
        List<int> bestSide = null;
        while (vertices.Count > 1)
        {
            var remaining = new List<int>(vertices);
            var connection = new long[n];
            var order = new List<int>();
            while (remaining.Count > 0)
            {
                int pickIndex = 0;
                for (int i = 1; i < remaining.Count; i++)
                {
                    if (connection[remaining[i]] > connection[remaining[pickIndex]]) pickIndex = i;
                }
                int pick = remaining[pickIndex];
                remaining.RemoveAt(pickIndex);
                order.Add(pick);
                foreach (int other in remaining) connection[other] += weight[pick][other];
            }
            int t = order[order.Count - 1];
            int s = order[order.Count - 2];
            long cut = connection[t];
            if (cut < best)
            {
                best = cut;
                bestSide = new List<int>(groups[t]);
            }
            foreach (int v in vertices)
            {
                if (v != s && v != t)
                {
                    weight[s][v] += weight[t][v];
                    weight[v][s] = weight[s][v];
                }
            }
            foreach (int v in vertices)
            {
                weight[t][v] = 0;
                weight[v][t] = 0;
            }
            vertices.Remove(t);
            groups[s].AddRange(groups[t]);
        }
        bestSide.Sort();
        return new MinCutInfo((int)best, bestSide.ToArray());
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
                if (bu != bv) cut += e[2];
            }
            best = Math.Min(best, cut);
        }
        return best;
    }

    private static int CrossingWeight(int[][] edges, int[] side)
    {
        var onSide = new HashSet<int>(side);
        int total = 0;
        foreach (int[] e in edges)
        {
            if (onSide.Contains(e[0]) != onSide.Contains(e[1])) total += e[2];
        }
        return total;
    }

    private static int seed;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<EdgeGraph452> LcgGraphs(int count, int maxN, bool connected, int baseSeed)
    {
        seed = baseSeed;
        var graphs = new List<EdgeGraph452>();
        for (int g = 0; g < count; g++)
        {
            int n = 2 + seed % (maxN - 1);
            var edges = new List<int[]>();
            int start = connected ? 1 : 0;
            for (int node = start; node < n; node++)
            {
                int parent = node > 0 ? NextSeed() % node : 0;
                int w = 1 + NextSeed() % 20;
                edges.Add(new[] { node, parent, w });
            }
            for (int e = 0; e < n + 2; e++)
            {
                int u = NextSeed() % n;
                int v = NextSeed() % n;
                int w = 1 + NextSeed() % 20;
                if (u != v) edges.Add(new[] { u, v, w });
            }
            graphs.Add(new EdgeGraph452(n, edges.ToArray()));
        }
        return graphs;
    }

    public static void Main()
    {
        int[][] triangle = { new[] { 0, 1, 1 }, new[] { 1, 2, 1 }, new[] { 0, 2, 1 } };
        var k4List = new List<int[]>();
        for (int u = 0; u < 4; u++)
        {
            for (int v = u + 1; v < 4; v++) k4List.Add(new[] { u, v, 1 });
        }
        int[][] k4 = k4List.ToArray();
        Check(BruteMinCut(3, triangle) == 2, "triangle brute");
        Check(BruteMinCut(4, k4) == 3, "K4 brute");
        MinCutInfo info = StoerWagner.MinCut(3, triangle);
        Check(info.Value == 2, "triangle value");
        Check(CrossingWeight(triangle, info.Side) == 2, "triangle witness");
        info = StoerWagner.MinCut(4, k4);
        Check(info.Value == 3, "K4 value");
        Check(CrossingWeight(k4, info.Side) == 3, "K4 witness");

        int[][] cycle4 = { new[] { 0, 1, 1 }, new[] { 1, 2, 1 }, new[] { 2, 3, 1 }, new[] { 3, 0, 1 } };
        Check(StoerWagner.MinCut(4, cycle4).Value == 2, "cycle value");
        Check(BruteMinCut(4, cycle4) == 2, "cycle brute");

        int[][] disjoint = { new[] { 0, 1, 5 }, new[] { 1, 2, 5 }, new[] { 0, 2, 5 }, new[] { 3, 4, 7 }, new[] { 4, 5, 7 }, new[] { 3, 5, 7 } };
        Check(BruteMinCut(6, disjoint) == 0, "disjoint brute");
        info = StoerWagner.MinCut(6, disjoint);
        Check(info.Value == 0, "disjoint value");
        Check(CrossingWeight(disjoint, info.Side) == 0, "disjoint witness");

        int[][] parallel = { new[] { 0, 1, 3 }, new[] { 1, 0, 4 }, new[] { 0, 1, 5 } };
        Check(StoerWagner.MinCut(2, parallel).Value == 12, "parallel value");
        int[][] weighted = { new[] { 0, 1, 5 }, new[] { 1, 2, 5 }, new[] { 0, 2, 5 } };
        Check(StoerWagner.MinCut(3, weighted).Value == 10, "weighted triangle");

        int checkedCount = 0;
        foreach (bool connected in new[] { true, false })
        {
            foreach (EdgeGraph452 graph in LcgGraphs(8, 8, connected, connected ? 450 : 451))
            {
                int expected = BruteMinCut(graph.N, graph.Edges);
                MinCutInfo result = StoerWagner.MinCut(graph.N, graph.Edges);
                Check(CrossingWeight(graph.Edges, result.Side) == result.Value, "witness value");
                Check(result.Value == expected, "LCG value n=" + graph.N);
                checkedCount++;
            }
        }
        foreach (int extra in new[] { 452, 453, 454, 455, 456, 457 })
        {
            foreach (EdgeGraph452 graph in LcgGraphs(3, 7, true, extra))
            {
                Check(StoerWagner.MinCut(graph.N, graph.Edges).Value == BruteMinCut(graph.N, graph.Edges),
                        "extra seed " + extra);
                checkedCount++;
            }
        }
        Check(checkedCount == 34, "coverage count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

internal class EdgeGraph452
{
    public readonly int N;
    public readonly int[][] Edges;

    public EdgeGraph452(int n, int[][] edges)
    {
        N = n;
        Edges = edges;
    }
}
