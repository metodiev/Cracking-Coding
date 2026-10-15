// Task 111 - Kruskal's MST using union-find. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Linq;

public class Dsu
{
    private readonly int[] parent;
    private readonly int[] rank;

    public Dsu(int n)
    {
        parent = Enumerable.Range(0, n).ToArray();
        rank = new int[n];
    }

    public int Find(int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public bool Union(int a, int b)
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) (ra, rb) = (rb, ra);
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        return true;
    }
}

public static class Solution
{
    /// <summary>edges rows: {u, v, w}. Returns {totalWeight, edgeCount} or null when disconnected.</summary>
    public static long[]? Kruskal(int n, int[][] edges)
    {
        var sorted = edges.OrderBy(e => e[2]).ToArray();
        var dsu = new Dsu(n);
        long total = 0;
        int count = 0;
        foreach (var e in sorted)
        {
            if (dsu.Union(e[0], e[1]))
            {
                total += e[2];
                if (++count == Math.Max(0, n - 1)) break;
            }
        }
        return count == Math.Max(0, n - 1) ? new long[] { total, count } : null;
    }

    public static void Main()
    {
        var edges = new[]
        {
            new[] { 0, 1, 4 }, new[] { 0, 2, 3 }, new[] { 1, 2, 1 },
            new[] { 1, 3, 2 }, new[] { 2, 3, 4 }, new[] { 3, 4, 2 }, new[] { 2, 4, 5 },
        };
        var result = Kruskal(5, edges);
        Check(result != null && result[0] == 8 && result[1] == 4, "classic graph");
        Check(Kruskal(1, new int[0][])![0] == 0, "single vertex");
        Check(Kruskal(3, new[] { new[] { 0, 1, 1 } }) == null, "disconnected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
