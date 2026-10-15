// Task 183 - Implement iterative segment tree. (Tier: Beginner, CTCI Ch 4)
using System;

public static class Solution
{
    /// <summary>
    /// Bottom-up segment tree in a flat array of size 2n: leaves live at [n, 2n) and
    /// parents at [1, n) with tree[i] = tree[2i] + tree[2i + 1]. Everything is a loop, so
    /// there is no recursion overhead; compared to the recursive tree of task 182 it needs
    /// 2n slots instead of 4n and never recurses, while producing the same answers.
    /// </summary>
    private class IterativeSegmentTree
    {
        private readonly int n;
        private readonly long[] tree;

        public IterativeSegmentTree(int[] values)
        {
            n = values.Length;
            tree = new long[2 * n];
            for (int i = 0; i < n; i++) tree[n + i] = values[i];
            for (int i = n - 1; i >= 1; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];
        }

        public void Update(int index, long value)
        {
            if (index < 0 || index >= n) throw new IndexOutOfRangeException("index out of range");
            int i = n + index;
            tree[i] = value;
            for (i /= 2; i >= 1; i /= 2) tree[i] = tree[2 * i] + tree[2 * i + 1];
        }

        public long RangeSum(int lo, int hi)
        {
            if (lo > hi) return 0;
            if (lo < 0 || hi >= n) throw new IndexOutOfRangeException("range out of bounds");
            long total = 0;
            int left = lo + n;
            int right = hi + 1 + n;               // classic half-open [left, right)
            while (left < right)
            {
                if ((left & 1) == 1) total += tree[left++];
                if ((right & 1) == 1) total += tree[--right];
                left /= 2;
                right /= 2;
            }
            return total;
        }
    }

    public static void Main()
    {
        var values = new[] { 1, 3, 5, 7, 9, 11 };
        var st = new IterativeSegmentTree(values);
        Check(st.RangeSum(0, 5) == 36, "whole array");
        Check(st.RangeSum(0, 0) == 1, "first element");
        Check(st.RangeSum(5, 5) == 11, "last element");
        Check(st.RangeSum(2, 3) == 12, "middle range");
        Check(st.RangeSum(1, 4) == 24, "inner range");
        Check(st.RangeSum(3, 2) == 0, "empty range");

        st.Update(2, 10);
        values[2] = 10;
        Check(st.RangeSum(0, 5) == 41, "after update");
        Check(st.RangeSum(2, 2) == 10, "updated point");
        st.Update(0, -4);
        values[0] = -4;
        Check(st.RangeSum(0, 5) == 36, "after second update");

        var single = new IterativeSegmentTree(new[] { 7 });
        Check(single.RangeSum(0, 0) == 7, "single element");
        single.Update(0, 2);
        Check(single.RangeSum(0, 0) == 2, "single element updated");

        var empty = new IterativeSegmentTree(new int[0]);
        Check(empty.RangeSum(0, -1) == 0, "empty tree");

        CheckThrows(() => st.RangeSum(-1, 2), "negative lo rejected");
        CheckThrows(() => st.Update(6, 0), "bad update index rejected");

        var rng = new Random(33);
        var data = new int[24];
        for (int i = 0; i < data.Length; i++) data[i] = rng.Next(41) - 20;
        var st2 = new IterativeSegmentTree(data);
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
