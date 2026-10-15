// Task 399 - Implement centroid decomposition for distance queries. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static List<List<int>> adjacency = new List<List<int>>();
    private static bool[] removed = Array.Empty<bool>();
    private static int[] size = Array.Empty<int>();
    private static long total;

    public static long CountPairsAtDistance(int n, int[][] edges, int k)
    {
        adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        removed = new bool[n];
        size = new int[n];
        total = 0;
        if (n > 0) Decompose(0, k);
        return total;
    }

    private static void Sizes(int u, int parent)
    {
        size[u] = 1;
        foreach (int v in adjacency[u])
        {
            if (v != parent && !removed[v])
            {
                Sizes(v, u);
                size[u] += size[v];
            }
        }
    }

    private static int FindCentroid(int u, int parent, int componentSize)
    {
        foreach (int v in adjacency[u])
        {
            if (v != parent && !removed[v] && size[v] > componentSize / 2)
            {
                return FindCentroid(v, u, componentSize);
            }
        }
        return u;
    }

    private static void Collect(int u, int parent, int distance, List<int> outDists)
    {
        outDists.Add(distance);
        foreach (int v in adjacency[u])
        {
            if (v != parent && !removed[v]) Collect(v, u, distance + 1, outDists);
        }
    }

    private static void Decompose(int start, int k)
    {
        Sizes(start, -1);
        int centroid = FindCentroid(start, -1, size[start]);
        removed[centroid] = true;
        var seen = new Dictionary<int, int> { [0] = 1 };   // the centroid itself is at distance 0
        foreach (int v in adjacency[centroid])
        {
            if (removed[v]) continue;
            var subtree = new List<int>();
            Collect(v, centroid, 1, subtree);
            foreach (int d in subtree)
            {
                if (seen.TryGetValue(k - d, out int c)) total += c;
            }
            foreach (int d in subtree)
            {
                seen[d] = seen.TryGetValue(d, out int c) ? c + 1 : 1;
            }
        }
        foreach (int v in adjacency[centroid])
        {
            if (!removed[v]) Decompose(v, k);
        }
    }

    public static long BruteForce(int n, int[][] edges, int k)
    {
        var graph = new List<List<int>>();
        for (int i = 0; i < n; i++) graph.Add(new List<int>());
        foreach (var edge in edges)
        {
            graph[edge[0]].Add(edge[1]);
            graph[edge[1]].Add(edge[0]);
        }
        long count = 0;
        for (int source = 0; source < n; source++)
        {
            var distance = new int[n];
            for (int i = 0; i < n; i++) distance[i] = -1;
            distance[source] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(source);
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                foreach (int v in graph[u])
                {
                    if (distance[v] == -1)
                    {
                        distance[v] = distance[u] + 1;
                        queue.Enqueue(v);
                    }
                }
            }
            for (int target = source + 1; target < n; target++)
            {
                if (distance[target] == k) count++;
            }
        }
        return count;
    }

    private static int seed = 600;

    private static (int N, int[][] Edges) LcgTree(int maxN)
    {
        int n = 2 + seed % (maxN - 1);
        var edges = new int[n - 1][];
        for (int node = 1; node < n; node++)
        {
            seed = (seed * 75 + 74) % 65537;
            edges[node - 1] = new[] { seed % node, node };
        }
        return (n, edges);
    }

    public static void Main()
    {
        Check(CountPairsAtDistance(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }, 1) == 2, "path3 k1");
        Check(CountPairsAtDistance(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }, 2) == 1, "path3 k2");
        Check(CountPairsAtDistance(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }, 3) == 0, "path3 k3");
        Check(CountPairsAtDistance(3, new[] { new[] { 0, 1 }, new[] { 0, 2 } }, 2) == 1, "star k2");
        Check(CountPairsAtDistance(3, new[] { new[] { 0, 1 }, new[] { 0, 2 } }, 1) == 2, "star k1");
        Check(CountPairsAtDistance(1, Array.Empty<int[]>(), 0) == 0, "single node");
        Check(CountPairsAtDistance(2, new[] { new[] { 0, 1 } }, 1) == 1, "two nodes");
        Check(CountPairsAtDistance(2, new[] { new[] { 0, 1 } }, 2) == 0, "two nodes too far");

        var path6 = new int[5][];
        for (int i = 0; i < 5; i++) path6[i] = new[] { i, i + 1 };
        for (int k = 0; k <= 6; k++)
        {
            Check(CountPairsAtDistance(6, path6, k) == BruteForce(6, path6, k), "path6 k=" + k);
        }

        for (int trial = 0; trial < 30; trial++)
        {
            var (n, edges) = LcgTree(60);
            for (int k = 0; k < 6; k++)
            {
                long got = CountPairsAtDistance(n, edges, k);
                long expected = BruteForce(n, edges, k);
                Check(got == expected, "random n=" + n + " k=" + k + " got=" + got + " want=" + expected);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
