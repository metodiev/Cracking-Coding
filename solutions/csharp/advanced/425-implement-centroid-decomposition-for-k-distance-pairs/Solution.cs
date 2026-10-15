// Task 425 - Implement centroid decomposition for k-distance pairs. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static List<List<int>> adjacency;
    private static bool[] removed;
    private static int[] size;
    private static long total;
    private static int target;

    private static long CountPairsAtDistance(int n, int[][] edges, int k)
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
        target = k;
        if (n > 0) Decompose(0);
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

    private static int FindCentroid(int u, int parent, int component)
    {
        foreach (int v in adjacency[u])
        {
            if (v != parent && !removed[v] && size[v] > component / 2)
                return FindCentroid(v, u, component);
        }
        return u;
    }

    private static void Collect(int u, int parent, int distance, List<int> distances)
    {
        distances.Add(distance);
        foreach (int v in adjacency[u])
        {
            if (v != parent && !removed[v]) Collect(v, u, distance + 1, distances);
        }
    }

    private static long PairsWithSum(List<int> distances)
    {
        var seen = new Dictionary<int, int>();
        long pairs = 0;
        foreach (int d in distances)
        {
            if (seen.TryGetValue(target - d, out int matching)) pairs += matching;
            seen.TryGetValue(d, out int current);
            seen[d] = current + 1;
        }
        return pairs;
    }

    private static void Decompose(int start)
    {
        Sizes(start, -1);
        int centroid = FindCentroid(start, -1, size[start]);
        removed[centroid] = true;
        var allDistances = new List<int> { 0 };     // the centroid itself
        var branches = new List<List<int>>();
        foreach (int v in adjacency[centroid])
        {
            if (removed[v]) continue;
            var distances = new List<int>();
            Collect(v, centroid, 1, distances);
            branches.Add(distances);
            allDistances.AddRange(distances);
        }
        total += PairsWithSum(allDistances);
        foreach (var distances in branches)         // remove pairs inside one branch
            total -= PairsWithSum(distances);
        foreach (int v in adjacency[centroid])
        {
            if (!removed[v]) Decompose(v);
        }
    }

    private static long BruteForce(int n, int[][] edges, int k)
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
            for (int other = source + 1; other < n; other++)
                if (distance[other] == k) count++;
        }
        return count;
    }

    private static int seed = 425;

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

    public static void Main()
    {
        Check(CountPairsAtDistance(0, Array.Empty<int[]>(), 0) == 0, "empty tree");
        Check(CountPairsAtDistance(1, Array.Empty<int[]>(), 0) == 0, "single node k0");
        Check(CountPairsAtDistance(1, Array.Empty<int[]>(), 1) == 0, "single node k1");
        Check(CountPairsAtDistance(2, new[] { new[] { 0, 1 } }, 1) == 1, "two nodes k1");
        Check(CountPairsAtDistance(2, new[] { new[] { 0, 1 } }, 0) == 0, "two nodes k0");
        Check(CountPairsAtDistance(2, new[] { new[] { 0, 1 } }, 2) == 0, "two nodes k2");

        var path3 = new[] { new[] { 0, 1 }, new[] { 1, 2 } };
        Check(CountPairsAtDistance(3, path3, 0) == 0, "path3 k0");
        Check(CountPairsAtDistance(3, path3, 1) == 2, "path3 k1");
        Check(CountPairsAtDistance(3, path3, 2) == 1, "path3 k2");
        Check(CountPairsAtDistance(3, path3, 3) == 0, "path3 k3");

        var star = new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 } };
        Check(CountPairsAtDistance(4, star, 1) == 3, "star k1");
        Check(CountPairsAtDistance(4, star, 2) == 3, "star k2");
        Check(CountPairsAtDistance(4, star, 3) == 0, "star k3");

        var path4 = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } };
        Check(CountPairsAtDistance(4, path4, 1) == 3, "path4 k1");
        Check(CountPairsAtDistance(4, path4, 2) == 2, "path4 k2");
        Check(CountPairsAtDistance(4, path4, 3) == 1, "path4 k3");
        Check(CountPairsAtDistance(4, path4, 4) == 0, "path4 k4");

        var path6 = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 } };
        var expectedPath6 = new[] { 0, 5, 4, 3, 2, 1, 0 };
        for (int k = 0; k < expectedPath6.Length; k++)
            Check(CountPairsAtDistance(6, path6, k) == expectedPath6[k], "path6 k=" + k);

        for (int trial = 0; trial < 30; trial++)
        {
            NextTree(60, out int n, out int[][] edges);
            for (int k = 0; k < 13; k++)
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
