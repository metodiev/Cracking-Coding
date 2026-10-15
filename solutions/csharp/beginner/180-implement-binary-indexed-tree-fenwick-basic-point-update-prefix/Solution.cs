// Task 180 - Implement binary indexed tree (Fenwick) basic. (Tier: Beginner, CTCI Ch 4)
using System;

public static class Solution
{
    private class Fenwick
    {
        private readonly int n;
        private readonly long[] tree;

        public Fenwick(int[] values)
        {
            n = values.Length;
            tree = new long[n + 1];
            for (int i = 0; i < n; i++) UpdateIndex(i + 1, values[i]);
        }

        private void UpdateIndex(int i, long delta)
        {
            while (i <= n)
            {
                tree[i] += delta;
                i += i & -i;
            }
        }

        public void Update(int index, long delta)
        {
            if (index < 0 || index >= n) throw new IndexOutOfRangeException("index out of range");
            UpdateIndex(index + 1, delta);
        }

        public long PrefixSum(int index)
        {
            int i = index + 1;
            if (i < 0 || i > n) throw new IndexOutOfRangeException("index out of range");
            long total = 0;
            while (i > 0)
            {
                total += tree[i];
                i -= i & -i;
            }
            return total;
        }

        public long RangeSum(int lo, int hi) => lo > hi ? 0 : PrefixSum(hi) - PrefixSum(lo - 1);
    }

    public static void Main()
    {
        var values = new[] { 1, 3, 5, 7, 9, 11 };
        var fw = new Fenwick(values);
        Check(fw.PrefixSum(0) == 1, "prefix 0");
        Check(fw.PrefixSum(5) == 36, "prefix 5");
        Check(fw.RangeSum(1, 3) == 15, "range 1..3");
        Check(fw.RangeSum(3, 2) == 0, "empty range");

        fw.Update(2, 5);
        values[2] += 5;
        Check(fw.PrefixSum(5) == 41, "after +5");
        fw.Update(0, -1);
        values[0] -= 1;
        Check(fw.PrefixSum(0) == 0, "after -1");

        var rng = new Random(9);
        var fw2 = new Fenwick(values);
        for (int t = 0; t < 50; t++)
        {
            int i = rng.Next(values.Length);
            int delta = rng.Next(11) - 5;
            values[i] += delta;
            fw2.Update(i, delta);
            for (int end = 0; end < values.Length; end++)
            {
                long expected = 0;
                for (int k = 0; k <= end; k++) expected += values[k];
                Check(fw2.PrefixSum(end) == expected, "cross-check prefix " + end);
            }
            int lo = rng.Next(values.Length);
            int hi = lo + rng.Next(values.Length - lo);
            long expectedRange = 0;
            for (int k = lo; k <= hi; k++) expectedRange += values[k];
            Check(fw2.RangeSum(lo, hi) == expectedRange, "cross-check range");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
