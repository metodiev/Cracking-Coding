// Task 264 - Find longest path in DAG (DP on topo order). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>
    /// Longest path in an UNWEIGHTED DAG measured in EDGES (documented; no weighted
    /// variant). Kahn's topological order feeds the DP: when u is popped, dist[u] is
    /// final because all its predecessors are done; every edge u->v relaxes
    /// dist[v] = max(dist[v], dist[u] + 1). Isolated vertices stay at 0.
    /// A cyclic input throws ArgumentException (documented): Kahn's queue drains before
    /// every vertex is processed, and a longest path in a cyclic graph is ill-defined.
    /// O(V + E) time and space.
    /// </summary>
    public static int LongestPath(int n, int[][] edges)
    {
        var adj = new List<int>[n];
        for (int i = 0; i < n; i++) adj[i] = new List<int>();
        var indeg = new int[n];
        foreach (var e in edges)
        {
            adj[e[0]].Add(e[1]);
            indeg[e[1]]++;
        }
        var q = new Queue<int>();
        for (int i = 0; i < n; i++) if (indeg[i] == 0) q.Enqueue(i);
        var dist = new int[n];
        int processed = 0;
        while (q.Count > 0)
        {
            int u = q.Dequeue();
            processed++;
            foreach (int v in adj[u])
            {
                if (dist[u] + 1 > dist[v]) dist[v] = dist[u] + 1;
                if (--indeg[v] == 0) q.Enqueue(v);
            }
        }
        if (processed != n) throw new ArgumentException("graph contains a cycle");
        int best = 0;
        foreach (int d in dist) best = Math.Max(best, d);
        return best;
    }

    /// <summary>Memoized DFS reference (valid on acyclic inputs); used only for cross-checks.</summary>
    public static int LongestPathDfs(int n, int[][] edges)
    {
        var adj = new List<int>[n];
        for (int i = 0; i < n; i++) adj[i] = new List<int>();
        foreach (var e in edges) adj[e[0]].Add(e[1]);
        var memo = new int[n];
        Array.Fill(memo, -1);
        int best = 0;
        for (int i = 0; i < n; i++) best = Math.Max(best, Depth(adj, memo, i));
        return best;
    }

    private static int Depth(List<int>[] adj, int[] memo, int u)
    {
        if (memo[u] >= 0) return memo[u];
        int best = 0;
        foreach (int v in adj[u]) best = Math.Max(best, 1 + Depth(adj, memo, v));
        memo[u] = best;
        return best;
    }

    public static void Main()
    {
        var dag = new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 2, 5 } };
        Check(LongestPath(6, dag) == 3, "hand-verified DAG (0->2->3->4)");
        Check(LongestPath(8, dag) == 3, "disconnected vertices 6, 7");
        Check(LongestPath(5, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 } }) == 4, "linear chain");
        Check(LongestPath(1, new int[0][]) == 0, "single vertex");
        Check(LongestPath(0, new int[0][]) == 0, "no vertices");
        Check(LongestPath(5, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 3, 4 } }) == 2, "disconnected chains");
        Check(LongestPath(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2, 3 } }) == 2, "diamond");
        Check(LongestPath(3, new[] { new[] { 0, 2 }, new[] { 2, 1 } }) == 2, "topo order != index order");
        Check(LongestPath(4, new[] { new[] { 3, 2 }, new[] { 2, 1 }, new[] { 1, 0 } }) == 3, "reversed chain");

        foreach (var (badN, badEdges) in new[]
        {
            (3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } }),
            (4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 3 } }),
        })
        {
            bool threw = false;
            try { LongestPath(badN, badEdges); } catch (ArgumentException) { threw = true; }
            Check(threw, "cycle rejected");
        }

        var rng = new Random(264);
        for (int test = 0; test < 300; test++)          // cross-check vs the memoized-DFS oracle
        {
            int n = rng.Next(13);
            var edgesList = new List<int[]>();
            for (int u = 0; u < n; u++)
                for (int v = u + 1; v < n; v++)
                    if (rng.NextDouble() < 0.25) edgesList.Add(new[] { u, v });
            var edges = edgesList.ToArray();
            Check(LongestPath(n, edges) == LongestPathDfs(n, edges), "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
