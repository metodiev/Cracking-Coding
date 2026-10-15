// Task 447 - Implement dynamic MST under edge insertions (conceptual). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal class DynamicMST
{
    private readonly List<int[]> edges = new List<int[]>();
    private readonly List<int[]> treeEdges = new List<int[]>();
    public int Swaps;

    public void AddEdge(int u, int v, int w)
    {
        edges.Add(new[] { u, v, w });
        int[] heaviest = HeaviestOnPath(u, v);
        if (heaviest == null)
        {
            treeEdges.Add(new[] { u, v, w });
        }
        else if (heaviest[2] > w)
        {
            treeEdges.Remove(heaviest);
            treeEdges.Add(new[] { u, v, w });
            Swaps++;
        }
    }

    private int[] HeaviestOnPath(int u, int v)
    {
        if (u == v) return null;
        var adjacency = new Dictionary<int, List<int[]>>();
        for (int i = 0; i < treeEdges.Count; i++)
        {
            int[] e = treeEdges[i];
            if (!adjacency.ContainsKey(e[0])) adjacency[e[0]] = new List<int[]>();
            if (!adjacency.ContainsKey(e[1])) adjacency[e[1]] = new List<int[]>();
            adjacency[e[0]].Add(new[] { e[1], i });
            adjacency[e[1]].Add(new[] { e[0], i });
        }
        if (!adjacency.ContainsKey(u)) return null;
        var parent = new Dictionary<int, int[]>();
        parent[u] = new[] { -1, -1 };
        var stack = new Stack<int>();
        stack.Push(u);
        while (stack.Count > 0)
        {
            int x = stack.Pop();
            foreach (int[] pair in adjacency[x])
            {
                if (!parent.ContainsKey(pair[0]))
                {
                    parent[pair[0]] = new[] { x, pair[1] };
                    stack.Push(pair[0]);
                }
            }
        }
        if (!parent.ContainsKey(v)) return null;
        int[] heaviest = null;
        int node = v;
        while (parent[node][0] != -1)
        {
            int[] info = parent[node];
            int[] edge = treeEdges[info[1]];
            if (heaviest == null || edge[2] > heaviest[2]) heaviest = edge;
            node = info[0];
        }
        return heaviest;
    }

    public long MstWeight()
    {
        long total = 0;
        foreach (int[] e in treeEdges) total += e[2];
        return total;
    }
}

public static class Solution
{
    private static int seed;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static long KruskalWeight(int n, List<int[]> edges)
    {
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        var sorted = new List<int[]>(edges);
        sorted.Sort((a, b) => a[2].CompareTo(b[2]));
        long total = 0;
        foreach (int[] e in sorted)
        {
            int ru = Find(parent, e[0]);
            int rv = Find(parent, e[1]);
            if (ru != rv)
            {
                parent[ru] = rv;
                total += e[2];
            }
        }
        return total;
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static List<int[]> LcgEdgeStream(int count, int n, int baseSeed)
    {
        seed = baseSeed;
        var edges = new List<int[]>();
        for (int i = 0; i < count; i++)
        {
            int u = NextSeed() % n;
            int v = NextSeed() % n;
            if (u == v) v = (v + 1) % n;
            int w = 1 + NextSeed() % 30;
            edges.Add(new[] { u, v, w });
        }
        return edges;
    }

    public static void Main()
    {
        var dynamic = new DynamicMST();
        dynamic.AddEdge(0, 1, 5);
        Check(dynamic.MstWeight() == 5, "first edge");
        dynamic.AddEdge(1, 2, 3);
        Check(dynamic.MstWeight() == 8, "second edge");
        dynamic.AddEdge(2, 3, 7);
        Check(dynamic.MstWeight() == 15, "third edge");
        dynamic.AddEdge(0, 3, 1);
        Check(dynamic.MstWeight() == 9, "swap in lighter edge");
        Check(dynamic.Swaps == 1, "swap counted");
        dynamic.AddEdge(0, 2, 9);
        Check(dynamic.MstWeight() == 9, "heavier edge ignored");

        int swapsSeen = 0;
        for (int trial = 0; trial < 25; trial++)
        {
            int n = 4 + trial % 5;
            List<int[]> stream = LcgEdgeStream(2 * n, n, 900 + trial * 37);
            dynamic = new DynamicMST();
            var inserted = new List<int[]>();
            foreach (int[] edge in stream)
            {
                dynamic.AddEdge(edge[0], edge[1], edge[2]);
                inserted.Add(edge);
                Check(dynamic.MstWeight() == KruskalWeight(n, inserted), "trial " + trial);
            }
            swapsSeen += dynamic.Swaps;
        }
        Check(swapsSeen > 0, "swap rule exercised");

        int finalN = 6;
        List<int[]> finalStream = LcgEdgeStream(3 * finalN, finalN, 321);
        dynamic = new DynamicMST();
        foreach (int[] edge in finalStream) dynamic.AddEdge(edge[0], edge[1], edge[2]);
        Check(dynamic.MstWeight() == KruskalWeight(finalN, finalStream), "final weight");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
