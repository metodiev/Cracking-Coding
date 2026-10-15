// Task 082 - Range sum query naive using prefix sums. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    private class RangeSumQuery
    {
        private readonly int[] prefix;

        public RangeSumQuery(int[] a)
        {
            prefix = new int[a.Length + 1];
            for (int i = 0; i < a.Length; i++) prefix[i + 1] = prefix[i] + a[i];
        }

        public int RangeSum(int lo, int hi)
        {
            int n = prefix.Length - 1;
            if (lo < 0 || hi >= n || lo > hi)
            {
                throw new ArgumentException("invalid range [" + lo + ", " + hi + "] for n=" + n);
            }
            return prefix[hi + 1] - prefix[lo];
        }
    }

    public static void Main()
    {
        var rs = new RangeSumQuery(new[] { 3, 1, 4, 1, 5 });
        Check(rs.RangeSum(0, 4) == 14, "full range");
        Check(rs.RangeSum(1, 3) == 6, "inner range");
        Check(rs.RangeSum(2, 2) == 4, "single element");
        Check(rs.RangeSum(0, 0) == 3, "first element");
        Check(rs.RangeSum(4, 4) == 5, "last element");
        Check(new RangeSumQuery(new[] { 7 }).RangeSum(0, 0) == 7, "single element array");
        CheckThrows(() => rs.RangeSum(-1, 2), "lo < 0");
        CheckThrows(() => rs.RangeSum(2, 1), "lo > hi");
        CheckThrows(() => rs.RangeSum(1, 5), "hi >= n");
        CheckThrows(() => new RangeSumQuery(new int[0]).RangeSum(0, 0), "empty array");
        Console.WriteLine("PASS");
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected ArgumentException: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
