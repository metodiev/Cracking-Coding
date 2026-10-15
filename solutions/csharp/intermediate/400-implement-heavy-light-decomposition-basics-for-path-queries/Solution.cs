// Task 400 - Implement heavy-light decomposition basics for path queries. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal class Fenwick
{
    private readonly long[] tree;

    public Fenwick(int size) { tree = new long[size + 1]; }

    public void Add(int index, long delta)
    {
        int i = index + 1;
        while (i < tree.Length)
        {
            tree[i] += delta;
            i += i & -i;
        }
    }

    private long Prefix(int count)
    {
        long total = 0;
        int i = count;
        while (i > 0)
        {
            total += tree[i];
            i -= i & -i;
        }
        return total;
    }

    public long RangeSum(int left, int right) => Prefix(right + 1) - Prefix(left);
}

internal class HeavyLight
{
    private readonly int[] parent, depth, size, heavy, head, pos;
    private readonly long[] values;
    private readonly Fenwick fenwick;

    public HeavyLight(int n, int[][] edges, int[] nodeValues)
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
        for (int i = 0; i < n; i++) parent[i] = -1;
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    order.Add(v);
                }
            }
        }
        size = new int[n];
        heavy = new int[n];
        for (int i = 0; i < n; i++) { size[i] = 1; heavy[i] = -1; }
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int u = order[i];
            int best = 0;
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    size[u] += size[v];
                    if (size[v] > best)
                    {
                        best = size[v];
                        heavy[u] = v;
                    }
                }
            }
        }
        head = new int[n];
        pos = new int[n];
        int position = 0;
        foreach (int u in order)
        {
            if (u == 0 || heavy[parent[u]] != u)          // starts a new chain
            {
                int x = u;
                while (x != -1)
                {
                    head[x] = u;
                    pos[x] = position++;
                    x = heavy[x];
                }
            }
        }
        values = new long[n];
        for (int i = 0; i < n; i++) values[i] = nodeValues[i];
        fenwick = new Fenwick(n);
        for (int node = 0; node < n; node++) fenwick.Add(pos[node], values[node]);
    }

    public void Update(int node, long value)
    {
        long delta = value - values[node];
        values[node] = value;
        fenwick.Add(pos[node], delta);
    }

    public long PathSum(int u, int v)
    {
        long total = 0;
        while (head[u] != head[v])
        {
            if (depth[head[u]] < depth[head[v]]) (u, v) = (v, u);
            total += fenwick.RangeSum(pos[head[u]], pos[u]);
            u = parent[head[u]];
        }
        if (depth[u] > depth[v]) (u, v) = (v, u);
        total += fenwick.RangeSum(pos[u], pos[v]);
        return total;
    }
}

public static class Solution
{
    public static long BrutePathSum(int n, int[][] edges, long[] values, int u, int v)
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
        parent[u] = u;
        var queue = new Queue<int>();
        queue.Enqueue(u);
        while (queue.Count > 0)
        {
            int x = queue.Dequeue();
            if (x == v) break;
            foreach (int y in adjacency[x])
            {
                if (parent[y] == -1)
                {
                    parent[y] = x;
                    queue.Enqueue(y);
                }
            }
        }
        long total = 0;
        int node = v;
        while (node != u)
        {
            total += values[node];
            node = parent[node];
        }
        return total + values[u];
    }

    private static int seed = 400;

    public static void Main()
    {
        var hld = new HeavyLight(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }, new[] { 1, 2, 3, 4 });
        Check(hld.PathSum(0, 3) == 10, "path graph full");
        Check(hld.PathSum(0, 0) == 1, "single node");
        Check(hld.PathSum(1, 2) == 5, "middle segment");
        hld.Update(2, 10);
        Check(hld.PathSum(0, 3) == 17, "after update");
        Check(hld.PathSum(1, 2) == 12, "middle after update");
        Check(hld.PathSum(2, 2) == 10, "single after update");

        var star = new HeavyLight(3, new[] { new[] { 0, 1 }, new[] { 0, 2 } }, new[] { 5, 1, 1 });
        Check(star.PathSum(1, 2) == 7, "star leaves");
        Check(star.PathSum(0, 0) == 5, "star centre");

        for (int trial = 0; trial < 25; trial++)
        {
            int n = 2 + seed % 49;
            var values = new int[n];
            var edges = new List<int[]>();
            for (int node = 0; node < n; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                values[node] = seed % 20 - 5;
                if (node > 0)
                {
                    seed = (seed * 75 + 74) % 65537;
                    edges.Add(new[] { seed % node, node });
                }
            }
            var edgeArray = edges.ToArray();
            var longValues = new long[n];
            for (int i = 0; i < n; i++) longValues[i] = values[i];
            var random = new HeavyLight(n, edgeArray, values);
            int localSeed = n;
            for (int step = 0; step < 12; step++)
            {
                localSeed = (localSeed * 75 + 74) % 65537;
                int u = localSeed % n;
                localSeed = (localSeed * 75 + 74) % 65537;
                int v = localSeed % n;
                Check(random.PathSum(u, v) == BrutePathSum(n, edgeArray, longValues, u, v), "random path");
                localSeed = (localSeed * 75 + 74) % 65537;
                int node = localSeed % n;
                localSeed = (localSeed * 75 + 74) % 65537;
                long newValue = localSeed % 40 - 20;
                longValues[node] = newValue;
                random.Update(node, newValue);
                Check(random.PathSum(u, v) == BrutePathSum(n, edgeArray, longValues, u, v), "random after update");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
