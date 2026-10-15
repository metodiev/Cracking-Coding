// Task 451 - Implement Gomory-Hu tree for all-pairs min-cut (outline). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal static class GomoryHu
{
    private const int Inf = int.MaxValue / 4;

    /// <summary>Undirected LCG graphs: a random tree plus a few extra edges.</summary>
    public static List<(int n, List<int[]> edges)> LcgGraphs(int count, int seed, int minN, int maxN)
    {
        var graphs = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = minN + seed % (maxN - minN + 1);
            var edges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                int parent = seed % node;
                seed = (seed * 75 + 74) % 65537;
                edges.Add(new[] { node, parent, 1 + seed % 9 });
            }
            for (int k = 0; k < n; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.Add(new[] { u, v, 1 + seed % 9 });
            }
            graphs.Add((n, edges));
        }
        return graphs;
    }

    private static int[][] CapacityMatrix(int n, List<int[]> edges)
    {
        var cap = new int[n][];
        for (int i = 0; i < n; i++) cap[i] = new int[n];
        foreach (var e in edges)
        {
            int u = e[0], v = e[1], w = e[2];
            if (u != v)
            {
                cap[u][v] += w;
                cap[v][u] += w;
            }
        }
        return cap;
    }

    /// <summary>Edmonds-Karp on the directed model of an undirected graph: (value, side).</summary>
    public static (int value, bool[] side) MaxFlowUndirected(int n, List<int[]> edges, int s, int t)
    {
        var cap = CapacityMatrix(n, edges);
        int flow = 0;
        while (true)
        {
            var parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = -1;
            parent[s] = s;
            var q = new Queue<int>();
            q.Enqueue(s);
            while (q.Count > 0)
            {
                int u = q.Dequeue();
                for (int v = 0; v < n; v++)
                {
                    if (parent[v] == -1 && cap[u][v] > 0)
                    {
                        parent[v] = u;
                        q.Enqueue(v);
                    }
                }
            }
            var side = new bool[n];
            for (int i = 0; i < n; i++) side[i] = parent[i] != -1;
            if (parent[t] == -1) return (flow, side);
            int aug = Inf;
            for (int v = t; v != s; v = parent[v]) aug = Math.Min(aug, cap[parent[v]][v]);
            for (int v = t; v != s; v = parent[v])
            {
                cap[parent[v]][v] -= aug;
                cap[v][parent[v]] += aug;
            }
            flow += aug;
        }
    }

    /// <summary>Gusfield simplification: n-1 max-flow runs: (parent, weight, runs).</summary>
    public static (int[] parent, int[] weight, int runs) Compute(int n, List<int[]> edges)
    {
        var parent = new int[n];
        var weight = new int[n];
        int runs = 0;
        for (int s = 1; s < n; s++)
        {
            int t = parent[s];
            var result = MaxFlowUndirected(n, edges, s, t);
            runs++;
            weight[s] = result.value;
            for (int i = s + 1; i < n; i++)
            {
                if (parent[i] == t && result.side[i]) parent[i] = s;
            }
        }
        return (parent, weight, runs);
    }

    /// <summary>Minimum edge weight on the tree path u..v.</summary>
    public static int TreeMinCut(int n, int[] parent, int[] weight, int u, int v)
    {
        var tree = new List<int[]>[n];
        for (int i = 0; i < n; i++) tree[i] = new List<int[]>();
        for (int i = 1; i < n; i++)
        {
            tree[i].Add(new[] { parent[i], weight[i] });
            tree[parent[i]].Add(new[] { i, weight[i] });
        }
        var seen = new bool[n];
        seen[u] = true;
        var q = new Queue<int[]>();
        q.Enqueue(new[] { u, Inf });
        while (q.Count > 0)
        {
            var cur = q.Dequeue();
            if (cur[0] == v) return cur[1];
            foreach (var nxt in tree[cur[0]])
            {
                if (!seen[nxt[0]])
                {
                    seen[nxt[0]] = true;
                    q.Enqueue(new[] { nxt[0], Math.Min(cur[1], nxt[1]) });
                }
            }
        }
        return -1;
    }

    /// <summary>Exact min u-v cut: minimum crossing weight over all vertex partitions.</summary>
    public static int BrutePairMinCut(int n, List<int[]> edges, int u, int v)
    {
        int best = Inf;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            if (((mask >> u) & 1) == ((mask >> v) & 1)) continue;
            int cut = 0;
            foreach (var e in edges)
            {
                if (((mask >> e[0]) & 1) != ((mask >> e[1]) & 1)) cut += e[2];
            }
            best = Math.Min(best, cut);
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        var triangle = new List<int[]> { new[] { 0, 1, 1 }, new[] { 1, 2, 1 }, new[] { 0, 2, 1 } };
        var gh = GomoryHu.Compute(3, triangle);
        Check(gh.runs == 2, "triangle flow runs");
        for (int u = 0; u < 3; u++)
        {
            for (int v = u + 1; v < 3; v++)
            {
                Check(GomoryHu.TreeMinCut(3, gh.parent, gh.weight, u, v) == 2, "triangle tree cut");
                Check(GomoryHu.TreeMinCut(3, gh.parent, gh.weight, u, v)
                      == GomoryHu.TreeMinCut(3, gh.parent, gh.weight, v, u), "symmetry");
                Check(GomoryHu.BrutePairMinCut(3, triangle, u, v) == 2, "triangle brute");
            }
        }

        var path = new List<int[]> { new[] { 0, 1, 5 }, new[] { 1, 2, 3 }, new[] { 2, 3, 4 } };
        var expected = new (int u, int v, int w)[]
        {
            (0, 1, 5), (0, 2, 3), (0, 3, 3), (1, 2, 3), (1, 3, 3), (2, 3, 4),
        };
        gh = GomoryHu.Compute(4, path);
        Check(gh.runs == 3, "path flow runs");
        foreach (var (u, v, w) in expected)
        {
            Check(GomoryHu.TreeMinCut(4, gh.parent, gh.weight, u, v) == w, "path tree cut");
            Check(GomoryHu.BrutePairMinCut(4, path, u, v) == w, "path brute");
        }

        var k4 = new List<int[]>();
        for (int u = 0; u < 4; u++)
            for (int v = u + 1; v < 4; v++) k4.Add(new[] { u, v, 1 });
        gh = GomoryHu.Compute(4, k4);
        for (int u = 0; u < 4; u++)
        {
            for (int v = u + 1; v < 4; v++)
            {
                Check(GomoryHu.TreeMinCut(4, gh.parent, gh.weight, u, v) == 3, "K4 tree cut");
                Check(GomoryHu.BrutePairMinCut(4, k4, u, v) == 3, "K4 brute");
            }
        }

        var parallel = new List<int[]> { new[] { 0, 1, 3 }, new[] { 1, 0, 4 } };
        gh = GomoryHu.Compute(2, parallel);
        Check(GomoryHu.TreeMinCut(2, gh.parent, gh.weight, 0, 1) == 7, "parallel edges tree");
        Check(GomoryHu.BrutePairMinCut(2, parallel, 0, 1) == 7, "parallel edges brute");

        int pairChecks = 0, graphCount = 0;
        foreach (int seed in new[] { 4511, 4512, 4513, 4514 })
        {
            foreach (var (n, edges) in GomoryHu.LcgGraphs(4, seed, 4, 7))
            {
                gh = GomoryHu.Compute(n, edges);
                Check(gh.runs == n - 1, "LCG flow runs");
                for (int u = 0; u < n; u++)
                {
                    for (int v = u + 1; v < n; v++)
                    {
                        int got = GomoryHu.TreeMinCut(n, gh.parent, gh.weight, u, v);
                        int want = GomoryHu.BrutePairMinCut(n, edges, u, v);
                        Check(got == want, "LCG pair " + u + "," + v);
                        Check(GomoryHu.TreeMinCut(n, gh.parent, gh.weight, v, u) == got, "LCG symmetry");
                        pairChecks++;
                    }
                }
                var direct = GomoryHu.MaxFlowUndirected(n, edges, 0, n - 1);
                Check(direct.value == GomoryHu.BrutePairMinCut(n, edges, 0, n - 1), "LCG direct oracle");
                graphCount++;
            }
        }
        Check(graphCount == 16, "graph count");
        Check(pairChecks >= 16 * 6, "pair count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
