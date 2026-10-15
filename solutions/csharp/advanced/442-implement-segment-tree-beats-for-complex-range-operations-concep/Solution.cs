// Task 442 - Implement segment tree beats for complex range operations (conceptual). (Tier: Advanced, CTCI Ch 4)
using System;

internal class SegmentTreeBeats
{
    private readonly int n;
    private readonly long[] nodeMax;
    private readonly long[] nodeSum;

    public SegmentTreeBeats(long[] values)
    {
        n = values.Length;
        nodeMax = new long[4 * Math.Max(1, n)];
        nodeSum = new long[4 * Math.Max(1, n)];
        if (n > 0) Build(1, 0, n - 1, values);
    }

    private void Build(int node, int lo, int hi, long[] values)
    {
        if (lo == hi)
        {
            nodeMax[node] = values[lo];
            nodeSum[node] = values[lo];
            return;
        }
        int mid = (lo + hi) >> 1;
        Build(2 * node, lo, mid, values);
        Build(2 * node + 1, mid + 1, hi, values);
        nodeMax[node] = Math.Max(nodeMax[2 * node], nodeMax[2 * node + 1]);
        nodeSum[node] = nodeSum[2 * node] + nodeSum[2 * node + 1];
    }

    public void Chmin(int left, int right, long cap)
    {
        Chmin(1, 0, n - 1, left, right, cap);
    }

    private void Chmin(int node, int lo, int hi, int left, int right, long cap)
    {
        if (right < lo || hi < left || nodeMax[node] <= cap) return;
        if (lo == hi)
        {
            nodeMax[node] = cap;
            nodeSum[node] = cap;
            return;
        }
        int mid = (lo + hi) >> 1;
        Chmin(2 * node, lo, mid, left, right, cap);
        Chmin(2 * node + 1, mid + 1, hi, left, right, cap);
        nodeMax[node] = Math.Max(nodeMax[2 * node], nodeMax[2 * node + 1]);
        nodeSum[node] = nodeSum[2 * node] + nodeSum[2 * node + 1];
    }

    public long RangeMax(int left, int right)
    {
        return RangeMax(1, 0, n - 1, left, right);
    }

    private long RangeMax(int node, int lo, int hi, int left, int right)
    {
        if (left <= lo && hi <= right) return nodeMax[node];
        int mid = (lo + hi) >> 1;
        long best = long.MinValue;
        if (left <= mid) best = Math.Max(best, RangeMax(2 * node, lo, mid, left, right));
        if (right > mid) best = Math.Max(best, RangeMax(2 * node + 1, mid + 1, hi, left, right));
        return best;
    }

    public long RangeSum(int left, int right)
    {
        return RangeSum(1, 0, n - 1, left, right);
    }

    private long RangeSum(int node, int lo, int hi, int left, int right)
    {
        if (left <= lo && hi <= right) return nodeSum[node];
        int mid = (lo + hi) >> 1;
        long total = 0;
        if (left <= mid) total += RangeSum(2 * node, lo, mid, left, right);
        if (right > mid) total += RangeSum(2 * node + 1, mid + 1, hi, left, right);
        return total;
    }
}

public static class Solution
{
    private static int seed;

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
            int kind = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int cap = seed % 41 - 20;
            ops[k] = new[] { kind, lo, hi, cap };
        }
        return ops;
    }

    public static void Main()
    {
        var tree = new SegmentTreeBeats(new long[] { 1, 5, 3, 4, 2 });
        Check(tree.RangeMax(0, 4) == 5, "initial max");
        Check(tree.RangeSum(0, 4) == 15, "initial sum");
        tree.Chmin(1, 3, 3);
        Check(tree.RangeMax(0, 4) == 3, "capped max");
        Check(tree.RangeMax(1, 1) == 3, "capped point");
        Check(tree.RangeSum(0, 4) == 12, "capped sum");
        Check(tree.RangeSum(2, 3) == 6, "window sum");
        tree.Chmin(0, 4, -1);
        Check(tree.RangeMax(0, 4) == -1, "global cap max");
        Check(tree.RangeSum(0, 4) == -5, "global cap sum");
        tree.Chmin(2, 2, 10);
        Check(tree.RangeMax(0, 4) == -1, "no-op cap max");
        Check(tree.RangeSum(0, 4) == -5, "no-op cap sum");

        int n = 16;
        seed = 1234;
        var baseValues = new long[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            baseValues[i] = seed % 50 - 25;
        }
        tree = new SegmentTreeBeats(baseValues);
        var brute = (long[])baseValues.Clone();
        foreach (var op in LcgOps(300, n))
        {
            int kind = op[0], lo = op[1], hi = op[2], cap = op[3];
            if (kind == 0)
            {
                tree.Chmin(lo, hi, cap);
                for (int i = lo; i <= hi; i++) brute[i] = Math.Min(brute[i], cap);
            }
            else if (kind == 1)
            {
                long best = long.MinValue;
                for (int i = lo; i <= hi; i++) best = Math.Max(best, brute[i]);
                Check(tree.RangeMax(lo, hi) == best, "random max");
            }
            else
            {
                long total = 0;
                for (int i = lo; i <= hi; i++) total += brute[i];
                Check(tree.RangeSum(lo, hi) == total, "random sum");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
