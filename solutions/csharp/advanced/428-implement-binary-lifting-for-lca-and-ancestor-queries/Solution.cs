// Task 428 - Implement binary lifting for LCA and ancestor queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class BinaryLifting
{
    public readonly int N;
    private readonly int log;
    public readonly int[] Depth;
    public readonly int[] Parent;
    private readonly int[][] up;

    public BinaryLifting(int n, int[][] edges)
    {
        N = n;
        int levels = 1;
        while ((1 << levels) <= n) levels++;
        log = levels;
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        Depth = new int[n];
        Parent = new int[n];
        var visited = new bool[n];
        visited[0] = true;
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (!visited[v])
                {
                    visited[v] = true;
                    Parent[v] = u;
                    Depth[v] = Depth[u] + 1;
                    order.Add(v);
                }
            }
        }
        up = new int[log][];
        up[0] = (int[])Parent.Clone();
        for (int j = 1; j < log; j++)
        {
            var previous = up[j - 1];
            var current = new int[n];
            for (int u = 0; u < n; u++) current[u] = previous[previous[u]];
            up[j] = current;
        }
    }

    public int KthAncestor(int node, int k)
    {
        if (k > Depth[node]) return -1;
        int bit = 0;
        while (k != 0)
        {
            if ((k & 1) != 0) node = up[bit][node];
            k >>= 1;
            bit++;
        }
        return node;
    }

    public int Lca(int a, int b)
    {
        if (Depth[a] < Depth[b]) { int tmp = a; a = b; b = tmp; }
        a = KthAncestor(a, Depth[a] - Depth[b]);
        if (a == b) return a;
        for (int j = log - 1; j >= 0; j--)
        {
            if (up[j][a] != up[j][b])
            {
                a = up[j][a];
                b = up[j][b];
            }
        }
        return Parent[a];
    }
}

public static class Solution
{
    private static void RootedTree(int n, int[][] edges, out int[] parent, out int[] depth)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        parent = new int[n];
        depth = new int[n];
        var visited = new bool[n];
        visited[0] = true;
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (!visited[v])
                {
                    visited[v] = true;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    order.Add(v);
                }
            }
        }
    }

    private static int NaiveLca(int[] parent, int[] depth, int a, int b)
    {
        while (depth[a] > depth[b]) a = parent[a];
        while (depth[b] > depth[a]) b = parent[b];
        while (a != b)
        {
            a = parent[a];
            b = parent[b];
        }
        return a;
    }

    private static int NaiveKthAncestor(int[] parent, int[] depth, int node, int k)
    {
        if (k > depth[node]) return -1;
        for (int i = 0; i < k; i++) node = parent[node];
        return node;
    }

    private static int seed = 428;

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
        var path = new BinaryLifting(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } });
        Check(path.Lca(0, 3) == 0, "path lca root");
        Check(path.Lca(2, 3) == 2, "path lca nested");
        Check(path.Lca(1, 2) == 1, "path lca adjacent");
        Check(path.Lca(3, 3) == 3, "path lca self");
        Check(path.Lca(0, 0) == 0, "path lca root self");
        Check(path.KthAncestor(3, 0) == 3, "path kth 0");
        Check(path.KthAncestor(3, 1) == 2, "path kth 1");
        Check(path.KthAncestor(3, 2) == 1, "path kth 2");
        Check(path.KthAncestor(3, 3) == 0, "path kth 3");
        Check(path.KthAncestor(3, 4) == -1, "path kth beyond root");
        Check(path.KthAncestor(0, 1) == -1, "root has no ancestor");

        var star = new BinaryLifting(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 } });
        Check(star.Lca(1, 2) == 0, "star lca leaves");
        Check(star.Lca(1, 3) == 0, "star lca leaves 2");
        Check(star.Lca(0, 3) == 0, "star lca centre");
        Check(star.Lca(2, 2) == 2, "star self");
        Check(star.KthAncestor(2, 1) == 0, "star kth 1");
        Check(star.KthAncestor(2, 2) == -1, "star kth beyond");

        var single = new BinaryLifting(1, Array.Empty<int[]>());
        Check(single.Lca(0, 0) == 0, "single lca");
        Check(single.KthAncestor(0, 0) == 0, "single kth 0");
        Check(single.KthAncestor(0, 1) == -1, "single kth beyond");

        for (int trial = 0; trial < 25; trial++)
        {
            NextTree(30, out int n, out int[][] edges);
            var bl = new BinaryLifting(n, edges);
            RootedTree(n, edges, out int[] parent, out int[] depth);
            for (int a = 0; a < n; a++)
            {
                for (int b = 0; b < n; b++)
                {
                    int expected = NaiveLca(parent, depth, a, b);
                    Check(bl.Lca(a, b) == expected, "lcg lca " + trial + "/" + a + "/" + b);
                    Check(bl.Lca(a, b) == bl.Lca(b, a), "lcg lca symmetry");
                }
            }
            for (int u = 0; u < n; u++)
            {
                for (int k = 0; k <= depth[u] + 2; k++)
                {
                    int expected = NaiveKthAncestor(parent, depth, u, k);
                    Check(bl.KthAncestor(u, k) == expected, "lcg kth " + trial + "/" + u + "/" + k);
                }
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
