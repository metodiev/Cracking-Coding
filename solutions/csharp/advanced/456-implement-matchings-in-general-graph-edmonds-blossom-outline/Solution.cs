// Task 456 - Implement matchings in general graph (Edmonds blossom outline). (Tier: Advanced, CTCI Ch 4)
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

    public static int[] GreedyMatching(int n, List<List<int>> adj)
    {
        var mate = new int[n];
        for (int i = 0; i < n; i++) mate[i] = -1;
        for (int u = 0; u < n; u++)
        {
            if (mate[u] == -1)
            {
                foreach (int v in adj[u])
                {
                    if (v > u && mate[v] == -1)
                    {
                        mate[u] = v;
                        mate[v] = u;
                        break;
                    }
                }
            }
        }
        return mate;
    }

    /// <summary>DFS over simple alternating paths from a free vertex; returns the edge list or null.</summary>
    public static List<int[]> FindAugmentingPath(int n, List<List<int>> adj, int[] mate, int root)
    {
        var onPath = new bool[n];
        onPath[root] = true;
        var path = new List<int[]>();
        return Dfs(root, adj, mate, onPath, path) ? path : null;
    }

    private static bool Dfs(int u, List<List<int>> adj, int[] mate, bool[] onPath, List<int[]> path)
    {
        foreach (int v in adj[u])
        {
            if (mate[u] == v || onPath[v]) continue;
            if (mate[v] == -1)
            {
                path.Add(new[] { u, v });
                return true;
            }
            int w = mate[v];
            if (onPath[w]) continue;
            onPath[v] = true;
            onPath[w] = true;
            path.Add(new[] { u, v });
            path.Add(new[] { v, w });
            if (Dfs(w, adj, mate, onPath, path)) return true;
            path.RemoveAt(path.Count - 1);
            path.RemoveAt(path.Count - 1);
            onPath[v] = false;
            onPath[w] = false;
        }
        return false;
    }

    public static void ApplyAugmenting(int[] mate, List<int[]> path)
    {
        for (int i = 0; i < path.Count; i++)
        {
            if (i % 2 == 1)
            {
                var e = path[i];
                if (mate[e[0]] != e[1] || mate[e[1]] != e[0]) throw new Exception("FAILED: path alternates");
                mate[e[0]] = -1;
                mate[e[1]] = -1;
            }
        }
        for (int i = 0; i < path.Count; i++)
        {
            if (i % 2 == 0)
            {
                var e = path[i];
                mate[e[0]] = e[1];
                mate[e[1]] = e[0];
            }
        }
    }

    /// <summary>Greedy + augmenting paths; returns (mate, augmentations).</summary>
    public static (int[] mate, int augmentations) MaxMatchingOutline(int n, List<List<int>> adj)
    {
        var mate = GreedyMatching(n, adj);
        int augmentations = 0;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int u = 0; u < n; u++)
            {
                if (mate[u] == -1)
                {
                    var path = FindAugmentingPath(n, adj, mate, u);
                    if (path != null)
                    {
                        ApplyAugmenting(mate, path);
                        augmentations++;
                        changed = true;
                    }
                }
            }
        }
        return (mate, augmentations);
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

public static class Solution
{
    public static void Main()
    {
        var handGraphs = new (string label, int n, List<int[]> edges, int want)[]
        {
            ("triangle", 3, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 } }, 1),
            ("C5", 5, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 0 } }, 2),
            ("K4", 4, new List<int[]> { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 1, 2 }, new[] { 1, 3 }, new[] { 2, 3 } }, 2),
            ("star", 5, new List<int[]> { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 } }, 1),
            ("C6", 6, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 0 } }, 3),
            ("C7", 7, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 6 }, new[] { 6, 0 } }, 3),
            ("two triangles", 6, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 0, 2 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 3, 5 } }, 2),
            ("K3,3", 6, new List<int[]> { new[] { 0, 3 }, new[] { 0, 4 }, new[] { 0, 5 }, new[] { 1, 3 }, new[] { 1, 4 }, new[] { 1, 5 }, new[] { 2, 3 }, new[] { 2, 4 }, new[] { 2, 5 } }, 3),
            ("diamond", 4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 0, 2 } }, 2),
            ("P4", 4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }, 2),
            ("empty", 5, new List<int[]>(), 0),
            ("single", 2, new List<int[]> { new[] { 0, 1 } }, 1),
        };
        foreach (var (label, n, edges, want) in handGraphs)
        {
            var adj = Matching.BuildAdj(n, edges);
            Check(Matching.BruteMaxMatching(n, adj) == want, label + " brute");
            var result = Matching.MaxMatchingOutline(n, adj);
            Check(Matching.CheckMatching(n, adj, result.mate) == want, label + " outline");
        }

        int improvedGraphs = 0, checkedCount = 0, totalAugmentations = 0;
        foreach (int seed in new[] { 4561, 4562, 4563, 4564 })
        {
            foreach (var (n, edges) in Matching.LcgGeneralGraphs(6, seed))
            {
                var adj = Matching.BuildAdj(n, edges);
                int greedySize = Matching.CheckMatching(n, adj, Matching.GreedyMatching(n, adj));
                var result = Matching.MaxMatchingOutline(n, adj);
                int size = Matching.CheckMatching(n, adj, result.mate);
                Check(size >= greedySize, "outline at least greedy");
                if (size > greedySize) improvedGraphs++;
                Check(size == Matching.BruteMaxMatching(n, adj), "outline equals brute");
                totalAugmentations += result.augmentations;
                checkedCount++;
            }
        }
        Check(checkedCount == 24, "graph count");
        Check(improvedGraphs >= 1, "augmenting improved something");
        Check(totalAugmentations >= improvedGraphs, "augmentation count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
