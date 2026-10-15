// Task 108 - Detect cycle in undirected graph (union-find). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class Dsu
{
    private readonly int[] parent;
    private readonly int[] rank;

    public Dsu(int n)
    {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    public int Find(int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];   // path halving
            x = parent[x];
        }
        return x;
    }

    public bool Union(int a, int b)
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        return true;
    }
}

public static class Solution
{
    public static bool HasCycleUndirected(int n, int[][] edges)
    {
        var dsu = new Dsu(n);
        foreach (var e in edges)
        {
            int u = e[0], v = e[1];
            if (u < 0 || u >= n || v < 0 || v >= n)
                throw new ArgumentException("edge endpoint out of range");
            if (!dsu.Union(u, v)) return true;
        }
        return false;
    }

    public static void Main()
    {
        Check(!HasCycleUndirected(5, new int[][] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 1, 4 } }), "tree");
        Check(HasCycleUndirected(3, new int[][] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } }), "triangle");
        Check(!HasCycleUndirected(6, new int[][] { new[] { 0, 1 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 } }), "forest");
        Check(HasCycleUndirected(6, new int[][] { new[] { 0, 1 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 2 } }), "cycle in one tree");
        Check(!HasCycleUndirected(1, new int[0][]), "single vertex");
        Check(HasCycleUndirected(2, new int[][] { new[] { 0, 0 } }), "self-loop");
        Check(HasCycleUndirected(2, new int[][] { new[] { 0, 1 }, new[] { 1, 0 } }), "parallel edges");

        bool threw = false;
        try
        {
            HasCycleUndirected(2, new int[][] { new[] { 0, 5 } });
        }
        catch (ArgumentException)
        {
            threw = true;
        }
        Check(threw, "out-of-range edge rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
