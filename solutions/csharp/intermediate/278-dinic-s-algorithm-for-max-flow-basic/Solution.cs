// Task 278 - Dinic's algorithm for max flow (basic). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static int[][] residual = Array.Empty<int[]>();
    private static int[] level = Array.Empty<int>();
    private static int[] pointer = Array.Empty<int>();
    private static int sinkNode;

    /// <summary>capacity[i][j] is the capacity of i -> j; returns the max flow value.</summary>
    public static int Dinic(int n, int[][] capacity, int source, int sink)
    {
        residual = new int[n][];
        for (int i = 0; i < n; i++) residual[i] = (int[])capacity[i].Clone();
        sinkNode = sink;
        int total = 0;
        while (true)
        {
            level = Enumerable.Repeat(-1, n).ToArray();
            level[source] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(source);
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                for (int v = 0; v < n; v++)
                {
                    if (level[v] == -1 && residual[u][v] > 0)
                    {
                        level[v] = level[u] + 1;
                        queue.Enqueue(v);
                    }
                }
            }
            if (level[sink] == -1) return total;

            pointer = new int[n];
            while (true)
            {
                int pushed = Dfs(source, int.MaxValue);
                if (pushed == 0) break;
                total += pushed;
            }
        }
    }

    private static int Dfs(int u, int pushed)
    {
        if (u == sinkNode) return pushed;
        while (pointer[u] < residual.Length)
        {
            int v = pointer[u];
            if (residual[u][v] > 0 && level[v] == level[u] + 1)
            {
                int here = Dfs(v, Math.Min(pushed, residual[u][v]));
                if (here > 0)
                {
                    residual[u][v] -= here;
                    residual[v][u] += here;
                    return here;
                }
            }
            pointer[u]++;
        }
        return 0;
    }

    /// <summary>BFS augmenting-path reference implementation used to cross-check Dinic.</summary>
    public static int EdmondsKarp(int n, int[][] capacity, int source, int sink)
    {
        var res = new int[n][];
        for (int i = 0; i < n; i++) res[i] = (int[])capacity[i].Clone();
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
                    if (parent[v] == -1 && res[u][v] > 0) { parent[v] = u; queue.Enqueue(v); }
                }
            }
            if (parent[sink] == -1) return total;
            int bottleneck = int.MaxValue;
            for (int v = sink; v != source; v = parent[v]) bottleneck = Math.Min(bottleneck, res[parent[v]][v]);
            for (int v = sink; v != source; v = parent[v])
            {
                int u = parent[v];
                res[u][v] -= bottleneck;
                res[v][u] += bottleneck;
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

    private static int seed = 97531;

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

    /// <summary>Independent oracle: enumerate every s-t cut (max-flow = min-cut).</summary>
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

    public static void Main()
    {
        var clrs = MatrixFromEdges(6, new[]
        {
            new[] { 0, 1, 16 }, new[] { 0, 2, 13 }, new[] { 1, 2, 10 }, new[] { 2, 1, 4 }, new[] { 1, 3, 12 },
            new[] { 3, 2, 9 }, new[] { 2, 4, 14 }, new[] { 4, 3, 7 }, new[] { 3, 5, 20 }, new[] { 4, 5, 4 },
        });
        Check(Dinic(6, clrs, 0, 5) == 23, "CLRS 23");
        Check(EdmondsKarp(6, clrs, 0, 5) == 23, "EK agrees on CLRS");

        Check(Dinic(4, MatrixFromEdges(4, new[] { new[] { 0, 1, 3 }, new[] { 0, 2, 2 }, new[] { 1, 3, 2 }, new[] { 2, 3, 3 } }), 0, 3) == 4, "diamond");
        var crossing = MatrixFromEdges(4, new[] { new[] { 0, 1, 1 }, new[] { 0, 2, 1 }, new[] { 1, 2, 1 }, new[] { 1, 3, 1 }, new[] { 2, 3, 1 } });
        Check(Dinic(4, crossing, 0, 3) == 2, "back-edge augmentation");
        Check(Dinic(3, MatrixFromEdges(3, new[] { new[] { 0, 1, 1 } }), 0, 2) == 0, "unreachable sink");

        for (int trial = 0; trial < 150; trial++)
        {
            var capacity = LcgCapacities(6);
            Check(Dinic(6, capacity, 0, 5) == EdmondsKarp(6, capacity, 0, 5), "random agreement " + trial);
        }

        var layered = MatrixFromEdges(8, new[]
        {
            new[] { 0, 1, 10 }, new[] { 0, 2, 10 }, new[] { 1, 3, 5 }, new[] { 1, 4, 6 }, new[] { 2, 5, 6 },
            new[] { 2, 6, 5 }, new[] { 3, 7, 4 }, new[] { 4, 7, 7 }, new[] { 5, 7, 5 }, new[] { 6, 7, 4 },
        });
        Check(Dinic(8, layered, 0, 7) == EdmondsKarp(8, layered, 0, 7), "layered agrees with EK");
        Check(Dinic(8, layered, 0, 7) == MinCutBruteForce(8, layered, 0, 7), "layered equals min cut");
        Check(Dinic(6, clrs, 0, 5) == MinCutBruteForce(6, clrs, 0, 5), "CLRS equals min cut");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
