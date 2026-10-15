// Task 181 - Range update, point query with Fenwick tree. (Tier: Beginner, CTCI Ch 4)
using System;

public static class Solution
{
    /// <summary>
    /// Dual of task 180's Fenwick tree: the tree stores a *difference* array, so
    /// RangeAdd costs two point updates and PointQuery is a prefix sum (all O(log n)).
    /// </summary>
    private class RangeUpdateFenwick
    {
        private readonly int n;
        private readonly long[] tree;

        public RangeUpdateFenwick(int size)
        {
            n = size;
            tree = new long[size + 1];
        }

        private void Add(int i, long delta)
        {
            while (i <= n)
            {
                tree[i] += delta;
                i += i & -i;
            }
        }

        public void RangeAdd(int lo, int hi, long value)
        {
            if (lo < 0 || hi >= n || lo > hi) throw new IndexOutOfRangeException("range out of bounds");
            Add(lo + 1, value);
            Add(hi + 2, -value);      // hi + 2 > n is a silent no-op suffix update
        }

        public long PointQuery(int index)
        {
            if (index < 0 || index >= n) throw new IndexOutOfRangeException("index out of range");
            long total = 0;
            for (int i = index + 1; i > 0; i -= i & -i) total += tree[i];
            return total;
        }
    }

    public static void Main()
    {
        var fw = new RangeUpdateFenwick(6);
        fw.RangeAdd(1, 3, 5);
        CheckSnapshot(fw, new long[] { 0, 5, 5, 5, 0, 0 }, "add [1,3] +5");
        fw.RangeAdd(0, 5, -2);
        CheckSnapshot(fw, new long[] { -2, 3, 3, 3, -2, -2 }, "add [0,5] -2");
        fw.RangeAdd(2, 2, 10);
        CheckSnapshot(fw, new long[] { -2, 3, 13, 3, -2, -2 }, "single position");
        fw.RangeAdd(5, 5, 7);
        Check(fw.PointQuery(5) == 5, "range ending at n - 1");
        fw.RangeAdd(0, 0, -3);
        Check(fw.PointQuery(0) == -5, "range starting at 0");

        CheckThrows(() => fw.RangeAdd(3, 2, 1), "lo > hi rejected");
        CheckThrows(() => fw.RangeAdd(0, 6, 1), "hi >= n rejected");
        CheckThrows(() => fw.PointQuery(-1), "negative index rejected");
        CheckThrows(() => new RangeUpdateFenwick(0).PointQuery(0), "empty structure rejected");

        var rng = new Random(11);
        int size = 12;
        var values = new long[size];
        var fw2 = new RangeUpdateFenwick(size);
        for (int t = 0; t < 80; t++)
        {
            int lo = rng.Next(size);
            int hi = lo + rng.Next(size - lo);
            long value = rng.Next(19) - 9;
            for (int k = lo; k <= hi; k++) values[k] += value;
            fw2.RangeAdd(lo, hi, value);
            for (int i = 0; i < size; i++)
            {
                Check(fw2.PointQuery(i) == values[i], "cross-check position " + i);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void CheckSnapshot(RangeUpdateFenwick fw, long[] expected, string label)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            Check(fw.PointQuery(i) == expected[i], label + " at position " + i);
        }
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
