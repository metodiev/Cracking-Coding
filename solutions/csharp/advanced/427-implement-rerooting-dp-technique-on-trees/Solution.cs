// Task 427 - Implement rerooting DP technique on trees. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static long[] SumOfDistancesFromEachNode(int n, int[][] edges)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = -1;
        parent[0] = 0;
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    parent[v] = u;
                    order.Add(v);
                }
            }
        }
        var size = new long[n];
        var down = new long[n];                     // distances from u into its subtree
        for (int i = 0; i < n; i++) size[i] = 1;
        for (int index = order.Count - 1; index >= 0; index--)
        {
            int u = order[index];
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    size[u] += size[v];
                    down[u] += down[v] + size[v];
                }
            }
        }
        var result = new long[n];
        result[0] = down[0];
        foreach (int u in order)
        {
            foreach (int v in adjacency[u])
            {
                if (v != parent[u]) result[v] = result[u] + n - 2 * size[v];
            }
        }
        return result;
    }

    private static long[] BruteForce(int n, int[][] edges)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        var totals = new long[n];
        for (int source = 0; source < n; source++)
        {
            var distance = new int[n];
            for (int i = 0; i < n; i++) distance[i] = -1;
            distance[source] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(source);
            long total = 0;
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                total += distance[u];
                foreach (int v in adjacency[u])
                {
                    if (distance[v] == -1)
                    {
                        distance[v] = distance[u] + 1;
                        queue.Enqueue(v);
                    }
                }
            }
            totals[source] = total;
        }
        return totals;
    }

    private static int seed = 427;

    private static void NextTree(int maxN, out int n, out int[][] edges)
    {
        n = 2 + seed % (maxN - 1);
        edges = new int[n - 1][];
        for (int node = 1; node < n; node++)
        {
            seed = (seed * 75 + 74) % 65537;
            edges[node - 1] = new[] { seed % node, node };
        }
    }

    private static bool Same(long[] a, long[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        Check(Same(SumOfDistancesFromEachNode(1, Array.Empty<int[]>()), new long[] { 0 }), "single node");
        Check(Same(SumOfDistancesFromEachNode(2, new[] { new[] { 0, 1 } }), new long[] { 1, 1 }), "edge");
        Check(Same(SumOfDistancesFromEachNode(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }), new long[] { 3, 2, 3 }), "path3");
        Check(Same(SumOfDistancesFromEachNode(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }),
                   new long[] { 6, 4, 4, 6 }), "path4");
        Check(Same(SumOfDistancesFromEachNode(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 } }),
                   new long[] { 3, 5, 5, 5 }), "star");
        Check(Same(SumOfDistancesFromEachNode(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 } }),
                   new long[] { 4, 4, 6, 6 }), "balanced");

        for (int trial = 0; trial < 30; trial++)
        {
            NextTree(60, out int n, out int[][] edges);
            long[] got = SumOfDistancesFromEachNode(n, edges);
            long[] expected = BruteForce(n, edges);
            Check(Same(got, expected), "lcg trial " + trial + " n=" + n);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
