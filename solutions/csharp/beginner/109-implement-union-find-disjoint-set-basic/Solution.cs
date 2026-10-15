// Task 109 - Implement union-find / disjoint set (basic). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class Dsu
{
    private readonly int[] parent;
    private readonly int[] rank;
    private int components;

    public Dsu(int n)
    {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        components = n;
    }

    public int Find(int x)                  // bare pointer walk (no compression yet)
    {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    public bool Union(int a, int b)         // plain linking
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        parent[ra] = rb;
        components--;
        return true;
    }

    public bool UnionByRank(int a, int b)
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        components--;
        return true;
    }

    public bool Connected(int a, int b) => Find(a) == Find(b);

    public int Count() => components;

    public int Depth(int x)
    {
        int d = 0;
        while (parent[x] != x)
        {
            x = parent[x];
            d++;
        }
        return d;
    }
}

public static class Solution
{
    public static void Main()
    {
        var d = new Dsu(6);
        Check(d.Count() == 6, "fresh components");
        Check(d.Find(3) == 3, "each element its own root");
        Check(d.Connected(0, 0), "element connected to itself");
        Check(d.Union(0, 1), "merge 0-1");
        Check(d.Union(1, 2), "merge 1-2");
        Check(d.Count() == 4, "count after two merges");
        Check(d.Connected(0, 2), "0 and 2 connected");
        Check(d.Depth(0) == 2, "plain linking leaves a chain");
        Check(!d.Union(0, 2), "redundant union is a no-op");
        Check(d.Count() == 4, "redundant union keeps count");
        Check(!d.Union(5, 5), "self union is a no-op");
        Check(d.UnionByRank(3, 4), "rank merge 3-4");
        Check(d.UnionByRank(4, 5), "rank merge 4-5");
        Check(d.Connected(3, 5), "3 and 5 connected");
        Check(!d.Connected(2, 3), "different sets stay apart");
        Check(d.Count() == 2, "final component count");

        var r = new Dsu(8);
        for (int i = 0; i < 7; i++) Check(r.UnionByRank(i, i + 1), "rank merge " + i);
        Check(r.Count() == 1, "rank merges connect everything");
        int maxDepth = 0;
        for (int i = 0; i < 8; i++) maxDepth = Math.Max(maxDepth, r.Depth(i));
        Check(maxDepth <= 1, "union-by-rank keeps the tree shallow");

        var one = new Dsu(1);
        Check(one.Count() == 1 && one.Find(0) == 0 && one.Depth(0) == 0, "single element");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
