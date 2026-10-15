// Task 457 - Implement blossom algorithm (conceptual/skeleton). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal static class Matching
{
    public static List<List<int>> BuildAdj(int n, List<int[]> edges)
    {
        var adj = new List<List<int>>();
        for (int i = 0; i < n; i++) adj.Add(new List<int>());
        foreach (var e in edges)
        {
            adj[e[0]].Add(e[1]);
            adj[e[1]].Add(e[0]);
        }
        return adj;
    }

    /// <summary>Validates the matching and returns its size.</summary>
    public static int CheckMatching(int n, List<List<int>> adj, int[] mate)
    {
        int size = 0;
        for (int v = 0; v < n; v++)
        {
            if (mate[v] != -1)
            {
                if (mate[mate[v]] != v) throw new Exception("FAILED: symmetric mates");
                if (mate[v] == v) throw new Exception("FAILED: no self match");
                if (!adj[v].Contains(mate[v])) throw new Exception("FAILED: matched pair must be an edge");
                size++;
            }
        }
        if (size % 2 != 0) throw new Exception("FAILED: even matched count");
        return size / 2;
    }

    /// <summary>Exact maximum matching by exhaustive backtracking (first-free-vertex branching).</summary>
    public static int BruteMaxMatching(int n, List<List<int>> adj)
    {
        var mate = new int[n];
        for (int i = 0; i < n; i++) mate[i] = -1;
        return Rec(0, n, adj, mate);
    }

    private static int Rec(int i, int n, List<List<int>> adj, int[] mate)
    {
        while (i < n && mate[i] != -1) i++;
        if (i == n) return 0;
        int best = Rec(i + 1, n, adj, mate);
        foreach (int v in adj[i])
        {
            if (v > i && mate[v] == -1)
            {
                mate[i] = v;
                mate[v] = i;
                best = Math.Max(best, 1 + Rec(i + 1, n, adj, mate));
                mate[i] = -1;
                mate[v] = -1;
            }
        }
        return best;
    }

    public static List<(int n, List<int[]> edges)> LcgGeneralGraphs(int count, int seed)
    {
        var graphs = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 2 + seed % 7;
            var edges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                edges.Add(new[] { node, seed % node });
            }
            for (int k = 0; k < n + 1; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                if (u != v && !HasEdge(edges, u, v)) edges.Add(new[] { u, v });
            }
            graphs.Add((n, edges));
        }
        return graphs;
    }

    private static bool HasEdge(List<int[]> edges, int u, int v)
    {
        foreach (var e in edges)
        {
            if ((e[0] == u && e[1] == v) || (e[0] == v && e[1] == u)) return true;
        }
        return false;
    }
}

/// <summary>Edmonds' algorithm with labelled BFS + blossom contraction.</summary>
internal class Blossom
{
    private readonly int n;
    private readonly List<List<int>> adj;
    private readonly int[] mate, baseArr, p;
    private readonly bool[] used, blossom;
    private int contractions = 0;

    public Blossom(int n, List<List<int>> adj)
    {
        this.n = n;
        this.adj = adj;
        mate = new int[n];
        baseArr = new int[n];
        p = new int[n];
        used = new bool[n];
        blossom = new bool[n];
        for (int i = 0; i < n; i++) mate[i] = -1;
    }

    private int Lca(int a, int b)
    {
        var seen = new bool[n];
        while (true)
        {
            a = baseArr[a];
            seen[a] = true;
            if (mate[a] == -1) break;
            a = p[mate[a]];
        }
        while (true)
        {
            b = baseArr[b];
            if (seen[b]) return b;
            b = p[mate[b]];
        }
    }

    private void MarkPath(int v, int b, int child)
    {
        while (baseArr[v] != b)
        {
            blossom[baseArr[v]] = true;
            blossom[baseArr[mate[v]]] = true;
            p[v] = child;
            child = mate[v];
            v = p[mate[v]];
        }
    }

    private int FindPath(int root)
    {
        for (int i = 0; i < n; i++)
        {
            used[i] = false;
            p[i] = -1;
            baseArr[i] = i;
        }
        used[root] = true;
        var queue = new Queue<int>();
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            foreach (int to in adj[v])
            {
                if (baseArr[v] == baseArr[to] || mate[v] == to) continue;
                if (to == root || (mate[to] != -1 && p[mate[to]] != -1))
                {
                    contractions++;
                    int curbase = Lca(v, to);
                    for (int i = 0; i < n; i++) blossom[i] = false;
                    MarkPath(v, curbase, to);
                    MarkPath(to, curbase, v);
                    for (int i = 0; i < n; i++)
                    {
                        if (blossom[baseArr[i]])
                        {
                            baseArr[i] = curbase;
                            if (!used[i])
                            {
                                used[i] = true;
                                queue.Enqueue(i);
                            }
                        }
                    }
                }
                else if (p[to] == -1)
                {
                    p[to] = v;
                    if (mate[to] == -1) return to;
                    int nxt = mate[to];
                    used[nxt] = true;
                    queue.Enqueue(nxt);
                }
            }
        }
        return -1;
    }

    /// <summary>Runs the algorithm; returns (mate, contractions).</summary>
    public (int[] mate, int contractions) Run()
    {
        for (int i = 0; i < n; i++)
        {
            if (mate[i] == -1)
            {
                int v = FindPath(i);
                while (v != -1)
                {
                    int pv = p[v];
                    int ppv = mate[pv];
                    mate[v] = pv;
                    mate[pv] = v;
                    v = ppv;
                }
            }
        }
        return (mate, contractions);
    }
}

public static class Solution
{
    public static void Main()
    {
        var c5adj = Matching.BuildAdj(5, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 0 } });
        var c5 = new Blossom(5, c5adj);
        var c5result = c5.Run();
        Check(Matching.CheckMatching(5, c5adj, c5result.mate) == 2, "C5 optimum");
        Check(c5result.contractions >= 1, "C5 contractions");
        Check(Matching.BruteMaxMatching(5, c5adj) == 2, "C5 brute");

        var handGraphs = new (string label, int n, List<int[]> edges, int want)[]
        {
            ("triangle", 3, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } }, 1),
            ("C5", 5, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 0 } }, 2),
            ("K4", 4, new List<int[]> { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 1, 2 }, new[] { 1, 3 }, new[] { 2, 3 } }, 2),
            ("star", 5, new List<int[]> { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 } }, 1),
            ("C6", 6, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 0 } }, 3),
            ("C7", 7, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 6 }, new[] { 6, 0 } }, 3),
            ("K3,3", 6, new List<int[]> { new[] { 0, 3 }, new[] { 0, 4 }, new[] { 0, 5 }, new[] { 1, 3 }, new[] { 1, 4 }, new[] { 1, 5 }, new[] { 2, 3 }, new[] { 2, 4 }, new[] { 2, 5 } }, 3),
            ("two triangles", 6, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 3, 5 } }, 2),
            ("diamond", 4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 0, 2 } }, 2),
            ("P4", 4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }, 2),
            ("empty", 5, new List<int[]>(), 0),
            ("single", 2, new List<int[]> { new[] { 0, 1 } }, 1),
        };
        foreach (var (label, n, edges, want) in handGraphs)
        {
            var adj = Matching.BuildAdj(n, edges);
            Check(Matching.BruteMaxMatching(n, adj) == want, label + " brute");
            var blossom = new Blossom(n, adj);
            var result = blossom.Run();
            Check(Matching.CheckMatching(n, adj, result.mate) == want, label + " optimum");
        }

        int checkedCount = 0, totalContractions = 0;
        foreach (int seed in new[] { 4571, 4572, 4573, 4574 })
        {
            foreach (var (n, edges) in Matching.LcgGeneralGraphs(6, seed))
            {
                var adj = Matching.BuildAdj(n, edges);
                var blossom = new Blossom(n, adj);
                var result = blossom.Run();
                totalContractions += result.contractions;
                int size = Matching.CheckMatching(n, adj, result.mate);
                Check(size == Matching.BruteMaxMatching(n, adj), "blossom equals brute");
                checkedCount++;
            }
        }
        Check(checkedCount == 24, "graph count");
        Check(totalContractions >= 1, "contractions exercised");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
