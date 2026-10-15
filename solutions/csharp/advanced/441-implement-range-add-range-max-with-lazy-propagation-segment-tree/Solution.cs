// Task 441 - Implement range add + range max with lazy propagation. (Tier: Advanced, CTCI Ch 4)
using System;

internal class RangeAddRangeMax
{
    private readonly int n;
    private readonly long[] tree, lazy;

    public RangeAddRangeMax(long[] values)
    {
        n = values.Length;
        tree = new long[4 * Math.Max(1, n)];
        lazy = new long[4 * Math.Max(1, n)];
        if (n > 0) Build(1, 0, n - 1, values);
    }

    private void Build(int node, int lo, int hi, long[] values)
    {
        if (lo == hi)
        {
            tree[node] = values[lo];
            return;
        }
        int mid = (lo + hi) >> 1;
        Build(2 * node, lo, mid, values);
        Build(2 * node + 1, mid + 1, hi, values);
        tree[node] = Math.Max(tree[2 * node], tree[2 * node + 1]);
    }

    private void Push(int node)
    {
        long delta = lazy[node];
        if (delta != 0)
        {
            foreach (int child in new[] { 2 * node, 2 * node + 1 })
            {
                tree[child] += delta;
                lazy[child] += delta;
            }
            lazy[node] = 0;
        }
    }

    public void Add(int left, int right, long value) => AddRange(1, 0, n - 1, left, right, value);

    private void AddRange(int node, int lo, int hi, int left, int right, long value)
    {
        if (right < lo || hi < left) return;
        if (left <= lo && hi <= right)
        {
            tree[node] += value;
            lazy[node] += value;
            return;
        }
        Push(node);
        int mid = (lo + hi) >> 1;
        AddRange(2 * node, lo, mid, left, right, value);
        AddRange(2 * node + 1, mid + 1, hi, left, right, value);
        tree[node] = Math.Max(tree[2 * node], tree[2 * node + 1]);
    }

    public long Max(int left, int right) => QueryMax(1, 0, n - 1, left, right);

    private long QueryMax(int node, int lo, int hi, int left, int right)
    {
        if (left <= lo && hi <= right) return tree[node];
        Push(node);
        int mid = (lo + hi) >> 1;
        long best = long.MinValue;
        if (left <= mid) best = Math.Max(best, QueryMax(2 * node, lo, mid, left, right));
        if (right > mid) best = Math.Max(best, QueryMax(2 * node + 1, mid + 1, hi, left, right));
        return best;
    }
}

public static class Solution
{
    private static int seed = 777;

    private static int[][] LcgOps(int count, int n)
    {
        var ops = new int[count][];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            int lo = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int hi = lo + seed % (n - lo);
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 21 - 10;
            ops[k] = new[] { kind, lo, hi, value };
        }
        return ops;
    }

    public static void Main()
    {
        var tree = new RangeAddRangeMax(new long[] { 1, 3, 2, 5, 4 });
        Check(tree.Max(0, 4) == 5, "initial max");
        Check(tree.Max(3, 3) == 5, "single point");
        tree.Add(1, 3, 2);
        Check(tree.Max(0, 4) == 7, "after add");
        Check(tree.Max(0, 0) == 1, "untouched left");
        Check(tree.Max(1, 1) == 5, "shifted point");
        Check(tree.Max(2, 3) == 7, "window max");
        tree.Add(0, 4, -10);
        Check(tree.Max(0, 4) == -3, "negative add");
        tree.Add(4, 4, 100);
        Check(tree.Max(0, 4) == 94, "point boost");

        int n = 16;
        seed = 1234;
        var initial = new long[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            initial[i] = seed % 50 - 25;
        }
        tree = new RangeAddRangeMax(initial);
        var brute = (long[])initial.Clone();
        foreach (var op in LcgOps(200, n))
        {
            int kind = op[0], lo = op[1], hi = op[2], value = op[3];
            if (kind == 0)
            {
                tree.Add(lo, hi, value);
                for (int i = lo; i <= hi; i++) brute[i] += value;
            }
            else
            {
                long best = long.MinValue;
                for (int i = lo; i <= hi; i++) best = Math.Max(best, brute[i]);
                Check(tree.Max(lo, hi) == best, "random query " + lo + ".." + hi);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
