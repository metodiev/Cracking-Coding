// Task 110 - Path compression in union-find. (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class Dsu
{
    private readonly int[] parent;
    private int components;

    public Dsu(int n)
    {
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        components = n;
    }

    public int Find(int x)
    {
        int root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root)            // second pass flattens the path
        {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    public bool Union(int a, int b)          // plain linking on purpose
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        parent[ra] = rb;
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
        var d = new Dsu(9);
        Check(d.Count() == 9, "fresh components");
        for (int i = 0; i < 7; i++) Check(d.Union(i, i + 1), "chain link " + i);
        Check(d.Count() == 2, "chain plus loner");
        Check(d.Depth(0) == 7, "chain is deep before compression");
        Check(d.Find(0) == 7, "root of the chain");
        Check(d.Depth(0) == 1, "find flattened the chain");
        for (int i = 1; i <= 6; i++) Check(d.Depth(i) == 1, "node " + i + " points at root");
        Check(d.Depth(7) == 0, "root has depth 0");
        Check(d.Connected(0, 6), "chain members connected");
        Check(d.Find(0) == d.Find(0), "find is idempotent");
        Check(d.Depth(0) == 1, "second find stays flat");
        Check(d.Union(7, 8), "attach loner");
        Check(d.Count() == 1, "everything connected");
        Check(d.Find(0) == 8, "new root");
        Check(d.Depth(0) <= 2, "depth stays near constant");
        Check(d.Find(0) == 8 && d.Depth(0) == 1, "compressed onto new root");
        int maxDepth = 0;
        for (int i = 0; i < 9; i++) maxDepth = Math.Max(maxDepth, d.Depth(i));
        Check(maxDepth <= 2, "bounded depth overall");

        var one = new Dsu(1);
        Check(one.Count() == 1 && one.Find(0) == 0 && one.Depth(0) == 0, "single element");

        var flat = new Dsu(3);
        Check(flat.Union(0, 1) && flat.Union(1, 2), "flat unions");
        Check(flat.Depth(0) == 2, "chain of three");
        Check(flat.Find(0) == 2 && flat.Depth(0) == 1 && flat.Depth(1) == 1, "compression on chain of three");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
