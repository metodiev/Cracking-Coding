// Task 182 - Implement segment tree for range sum (build/query/update). (Tier: Beginner, CTCI Ch 4)
using System;

public static class Solution
{
    /// <summary>
    /// Recursive segment tree in a flat array of size 4n; node 1 is the root and covers
    /// [0, n - 1], the children of node i are 2i and 2i + 1. Build O(n), point update and
    /// inclusive range sum O(log n).
    /// </summary>
    private class SegmentTree
    {
        private readonly int n;
        private readonly long[] tree;

        public SegmentTree(int[] values)
        {
            n = values.Length;
            tree = new long[4 * Math.Max(n, 1)];
            if (n > 0) Build(1, 0, n - 1, values);
        }

        private void Build(int node, int lo, int hi, int[] values)
        {
            if (lo == hi)
            {
                tree[node] = values[lo];
                return;
            }
            int mid = (lo + hi) / 2;
            Build(2 * node, lo, mid, values);
            Build(2 * node + 1, mid + 1, hi, values);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }

        public void Update(int index, long value)
        {
            if (index < 0 || index >= n) throw new IndexOutOfRangeException("index out of range");
            Update(1, 0, n - 1, index, value);
        }

        private void Update(int node, int lo, int hi, int index, long value)
        {
            if (lo == hi)
            {
                tree[node] = value;
                return;
            }
            int mid = (lo + hi) / 2;
            if (index <= mid) Update(2 * node, lo, mid, index, value);
            else Update(2 * node + 1, mid + 1, hi, index, value);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }

        public long RangeSum(int lo, int hi)
        {
            if (lo > hi) return 0;
            if (lo < 0 || hi >= n) throw new IndexOutOfRangeException("range out of bounds");
            return Query(1, 0, n - 1, lo, hi);
        }

        private long Query(int node, int lo, int hi, int qlo, int qhi)
        {
            if (qlo <= lo && hi <= qhi) return tree[node];
            int mid = (lo + hi) / 2;
            long total = 0;
            if (qlo <= mid) total += Query(2 * node, lo, mid, qlo, qhi);
            if (qhi > mid) total += Query(2 * node + 1, mid + 1, hi, qlo, qhi);
            return total;
        }
    }

    public static void Main()
    {
        var values = new[] { 1, 3, 5, 7, 9, 11 };
        var st = new SegmentTree(values);
        Check(st.RangeSum(0, 5) == 36, "whole array");
        Check(st.RangeSum(0, 0) == 1, "first element");
        Check(st.RangeSum(5, 5) == 11, "last element");
        Check(st.RangeSum(2, 3) == 12, "middle range");
        Check(st.RangeSum(3, 2) == 0, "empty range");

        st.Update(2, 10);
        values[2] = 10;
        Check(st.RangeSum(1, 3) == 20, "after update");
        st.Update(0, -4);
        values[0] = -4;
        Check(st.RangeSum(0, 5) == 36, "sum unchanged after -5/+5 swap");
        Check(st.RangeSum(0, 0) == -4, "updated single value");

        var single = new SegmentTree(new[] { 7 });
        Check(single.RangeSum(0, 0) == 7, "single element");
        single.Update(0, 2);
        Check(single.RangeSum(0, 0) == 2, "single element updated");

        var empty = new SegmentTree(new int[0]);
        Check(empty.RangeSum(0, -1) == 0, "empty tree");

        CheckThrows(() => st.RangeSum(-1, 2), "negative lo rejected");
        CheckThrows(() => st.Update(6, 0), "bad update index rejected");

        var rng = new Random(21);
        var data = new int[17];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(41) - 20;
        var st2 = new SegmentTree(data);
        for (int t = 0; t < 40; t++)
        {
            int i = rng.Next(data.Length);
            int value = rng.Next(41) - 20;
            data[i] = value;
            st2.Update(i, value);
            for (int lo = 0; lo < data.Length; lo++)
            {
                long expected = 0;
                for (int hi = lo; hi < data.Length; hi++)
                {
                    expected += data[hi];
                    Check(st2.RangeSum(lo, hi) == expected, "cross-check range " + lo + ".." + hi);
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (IndexOutOfRangeException)
        {
            return;
        }
        throw new Exception("FAILED: expected IndexOutOfRangeException: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
