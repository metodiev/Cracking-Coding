// Task 421 - Implement persistent segment tree (versioned). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class PersistentSegTree
{
    public readonly int N;
    private readonly List<int> left = new List<int>();
    private readonly List<int> right = new List<int>();
    private readonly List<long> total = new List<long>();
    public readonly List<int> Roots = new List<int>();

    public PersistentSegTree(int[] values)
    {
        N = values.Length;
        left.Add(0);                       // node 0 acts as the null node
        right.Add(0);
        total.Add(0);
        Roots.Add(Build(values, 0, N - 1));
    }

    private int MakeNode(int l, int r, long t)
    {
        left.Add(l);
        right.Add(r);
        total.Add(t);
        return total.Count - 1;
    }

    private int Build(int[] values, int lo, int hi)
    {
        if (lo == hi) return MakeNode(0, 0, values[lo]);
        int mid = (lo + hi) / 2;
        int l = Build(values, lo, mid);
        int r = Build(values, mid + 1, hi);
        return MakeNode(l, r, total[l] + total[r]);
    }

    public int Update(int version, int pos, long value)
    {
        Roots.Add(UpdateNode(Roots[version], 0, N - 1, pos, value));
        return Roots.Count - 1;
    }

    private int UpdateNode(int node, int lo, int hi, int pos, long value)
    {
        if (lo == hi) return MakeNode(0, 0, value);
        int mid = (lo + hi) / 2;
        if (pos <= mid)
        {
            int l = UpdateNode(left[node], lo, mid, pos, value);
            int r = right[node];
            return MakeNode(l, r, total[l] + total[r]);
        }
        int lo2 = left[node];
        int hi2 = UpdateNode(right[node], mid + 1, hi, pos, value);
        return MakeNode(lo2, hi2, total[lo2] + total[hi2]);
    }

    public long RangeSum(int version, int lo, int hi)
    {
        return RangeSumNode(Roots[version], 0, N - 1, lo, hi);
    }

    private long RangeSumNode(int node, int lo, int hi, int ql, int qr)
    {
        if (node == 0 || qr < lo || hi < ql) return 0;
        if (ql <= lo && hi <= qr) return total[node];
        int mid = (lo + hi) / 2;
        return RangeSumNode(left[node], lo, mid, ql, qr)
             + RangeSumNode(right[node], mid + 1, hi, ql, qr);
    }
}

public static class Solution
{
    private static int seed;

    private static int[] LcgArray(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 41 - 20;
        }
        return values;
    }

    public static void Main()
    {
        var st = new PersistentSegTree(new[] { 1, 2, 3, 4, 5 });
        Check(st.Roots.Count == 1, "one initial version");
        Check(st.RangeSum(0, 0, 4) == 15, "v0 whole");
        Check(st.RangeSum(0, 1, 3) == 9, "v0 middle");
        Check(st.RangeSum(0, 2, 2) == 3, "v0 single");

        int v1 = st.Update(0, 2, 10);                   // [1, 2, 10, 4, 5]
        Check(v1 == 1, "v1 index");
        Check(st.RangeSum(1, 0, 4) == 22, "v1 whole");
        Check(st.RangeSum(1, 2, 2) == 10, "v1 changed cell");
        Check(st.RangeSum(1, 1, 3) == 16, "v1 window");
        Check(st.RangeSum(0, 0, 4) == 15, "v0 untouched");
        Check(st.RangeSum(0, 2, 2) == 3, "v0 cell untouched");

        int v2 = st.Update(1, 0, -1);                   // [-1, 2, 10, 4, 5]
        Check(v2 == 2, "v2 index");
        Check(st.RangeSum(2, 0, 4) == 20, "v2 whole");
        Check(st.RangeSum(2, 0, 2) == 11, "v2 prefix");
        Check(st.RangeSum(1, 0, 2) == 13, "v1 prefix unchanged");

        int v3 = st.Update(0, 4, 7);                    // branch off OLD version: [1, 2, 3, 4, 7]
        Check(v3 == 3, "v3 index");
        Check(st.RangeSum(3, 0, 4) == 17, "v3 whole");
        Check(st.RangeSum(3, 3, 4) == 11, "v3 suffix");
        Check(st.RangeSum(3, 2, 2) == 3, "v3 old cell");
        Check(st.RangeSum(2, 0, 4) == 20, "other branch unchanged");
        Check(st.RangeSum(1, 0, 4) == 22, "v1 still intact");
        Check(st.RangeSum(0, 0, 4) == 15, "v0 still intact");

        foreach (int config in new[] { 11, 421, 9001 })
        {
            seed = config;
            int n = 1 + seed % 29;
            int[] values = LcgArray(n);
            var tree = new PersistentSegTree(values);
            var snapshots = new List<int[]> { (int[])values.Clone() };
            int local = seed;
            for (int step = 0; step < 60; step++)
            {
                local = (local * 75 + 74) % 65537;
                if (local % 3 == 0)
                {
                    int version = local % snapshots.Count;
                    int leftIndex = local % n;
                    int rightIndex = local / n % n;
                    if (leftIndex > rightIndex) { int tmp = leftIndex; leftIndex = rightIndex; rightIndex = tmp; }
                    long expected = 0;
                    for (int i = leftIndex; i <= rightIndex; i++) expected += snapshots[version][i];
                    Check(tree.RangeSum(version, leftIndex, rightIndex) == expected, "random query");
                }
                else
                {
                    int pos = local % n;
                    local = (local * 75 + 74) % 65537;
                    int value = local % 201 - 100;
                    int version = tree.Update(tree.Roots.Count - 1, pos, value);
                    var state = (int[])snapshots[snapshots.Count - 1].Clone();
                    state[pos] = value;
                    snapshots.Add(state);
                    Check(version == snapshots.Count - 1, "version chain");
                    long whole = 0;
                    foreach (int v in state) whole += v;
                    Check(tree.RangeSum(version, 0, n - 1) == whole, "random total");
                }
            }
        }

        var small = new PersistentSegTree(new[] { 42 });
        Check(small.RangeSum(0, 0, 0) == 42, "single value");
        small.Update(0, 0, -8);
        Check(small.RangeSum(1, 0, 0) == -8, "single value updated");
        Check(small.RangeSum(0, 0, 0) == 42, "single value versioned");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
