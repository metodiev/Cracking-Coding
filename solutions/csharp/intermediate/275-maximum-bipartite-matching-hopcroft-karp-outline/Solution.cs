// Task 275 - Maximum bipartite matching (Hopcroft-Karp). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static int[] pairU = Array.Empty<int>();
    private static int[] pairV = Array.Empty<int>();
    private static int[] dist = Array.Empty<int>();
    private static List<List<int>> adjacency = new List<List<int>>();

    /// <summary>Returns (size, pairs) for the maximum matching.</summary>
    public static (int Size, List<(int Left, int Right)> Pairs) HopcroftKarp(int nl, List<List<int>> adj, int nr)
    {
        adjacency = adj;
        pairU = Enumerable.Repeat(-1, nl).ToArray();
        pairV = Enumerable.Repeat(-1, nr).ToArray();
        dist = new int[nl];

        int matching = 0;
        while (Bfs(nl))
        {
            for (int u = 0; u < nl; u++)
            {
                if (pairU[u] == -1 && Dfs(u)) matching++;
            }
        }
        var pairs = new List<(int, int)>();
        for (int u = 0; u < nl; u++) if (pairU[u] != -1) pairs.Add((u, pairU[u]));
        return (matching, pairs);
    }

    private static bool Bfs(int nl)
    {
        var queue = new Queue<int>();
        for (int u = 0; u < nl; u++)
        {
            if (pairU[u] == -1) { dist[u] = 0; queue.Enqueue(u); }
            else dist[u] = -1;
        }
        bool reachable = false;
        while (queue.Count > 0)
        {
            int u = queue.Dequeue();
            foreach (int v in adjacency[u])
            {
                int w = pairV[v];
                if (w == -1) reachable = true;
                else if (dist[w] == -1) { dist[w] = dist[u] + 1; queue.Enqueue(w); }
            }
        }
        return reachable;
    }

    private static bool Dfs(int u)
    {
        foreach (int v in adjacency[u])
        {
            int w = pairV[v];
            if (w == -1 || (dist[w] == dist[u] + 1 && Dfs(w)))
            {
                pairU[u] = v;
                pairV[v] = u;
                return true;
            }
        }
        dist[u] = -1;
        return false;
    }

    /// <summary>Simple O(V*E) augmenting-path oracle.</summary>
    public static int Kuhn(int nl, List<List<int>> adj, int nr)
    {
        var matchV = Enumerable.Repeat(-1, nr).ToArray();
        int count = 0;
        for (int u = 0; u < nl; u++)
        {
            if (TryKuhn(u, adj, matchV, new bool[nr])) count++;
        }
        return count;
    }

    private static bool TryKuhn(int u, List<List<int>> adj, int[] matchV, bool[] visited)
    {
        foreach (int v in adj[u])
        {
            if (!visited[v])
            {
                visited[v] = true;
                if (matchV[v] == -1 || TryKuhn(matchV[v], adj, matchV, visited))
                {
                    matchV[v] = u;
                    return true;
                }
            }
        }
        return false;
    }

    public static bool IsValidMatching(List<(int Left, int Right)> pairs, List<List<int>> adj, int nl, int nr)
    {
        var lefts = new HashSet<int>();
        var rights = new HashSet<int>();
        foreach (var (u, v) in pairs)
        {
            if (!(u >= 0 && u < nl && v >= 0 && v < nr)) return false;
            if (!lefts.Add(u) || !rights.Add(v)) return false;
            if (!adj[u].Contains(v)) return false;
        }
        return true;
    }

    private static List<List<int>> Edges(int nl, int[][] pairs)
    {
        var adj = new List<List<int>>();
        for (int i = 0; i < nl; i++) adj.Add(new List<int>());
        foreach (var pair in pairs) adj[pair[0]].Add(pair[1]);
        return adj;
    }

    public static void Main()
    {
        Check(HopcroftKarp(3, Edges(3, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 0 }, new[] { 1, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 1 }, new[] { 2, 2 } }), 3).Size == 3, "K3,3");
        Check(HopcroftKarp(3, Edges(3, Array.Empty<int[]>()), 3).Size == 0, "no edges");
        Check(HopcroftKarp(3, Edges(3, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 } }), 2).Size == 2, "three lefts two rights");
        Check(HopcroftKarp(3, Edges(3, new[] { new[] { 0, 0 }, new[] { 1, 0 } }), 1).Size == 1, "contention for one right");
        Check(HopcroftKarp(3, Edges(3, new[] { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 1, 1 }, new[] { 2, 1 } }), 2).Size == 2, "chain");
        Check(HopcroftKarp(3, Edges(3, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 1 }, new[] { 2, 2 } }), 3).Size == 3, "distinct rights");
        Check(HopcroftKarp(2, Edges(2, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 } }), 2).Size == 2, "greedy trap");
        Check(HopcroftKarp(2, Edges(2, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 }, new[] { 1, 0 } }), 5).Size == 2, "unbalanced");

        var k33 = Edges(3, new[] { new[] { 0, 0 }, new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 0 }, new[] { 1, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 1 }, new[] { 2, 2 } });
        var result = HopcroftKarp(3, k33, 3);
        Check(IsValidMatching(result.Pairs, k33, 3, 3), "matching validity");
        Check(result.Pairs.Count == result.Size, "size equals pair count");

        int seed = 12345;
        for (int trial = 0; trial < 200; trial++)
        {
            var adj = new List<List<int>>();
            for (int u = 0; u < 6; u++) adj.Add(new List<int>());
            for (int u = 0; u < 6; u++)
            {
                for (int v = 0; v < 6; v++)
                {
                    seed = (seed * 75 + 74) % 65537;
                    if (seed % 3 == 0) adj[u].Add(v);
                }
            }
            var hk = HopcroftKarp(6, adj, 6);
            Check(IsValidMatching(hk.Pairs, adj, 6, 6), "random matching validity " + trial);
            Check(hk.Size == Kuhn(6, adj, 6), "random agreement " + trial);
            Check(hk.Size <= 6, "bounded " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
