// Task 261 - Median of two sorted arrays (log min(n,m) algorithm). (Tier: Intermediate, CTCI Ch 10)
using System;

public static class Solution
{
    /// <summary>
    /// Partition-based binary search on the SHORTER array: i = number of a-elements on
    /// the left side, j = (m+n+1)/2 - i follows. Valid when a[i-1] &lt;= b[j] and
    /// b[j-1] &lt;= a[i]; +-infinity sentinels cover empty arrays and the
    /// fully-left/fully-right cases. O(log min(m, n)); two empty arrays throw.
    /// </summary>
    public static double MedianOfTwoSorted(int[] a, int[] b)
    {
        if (a.Length == 0 && b.Length == 0) throw new ArgumentException("at least one array must be non-empty");
        if (a.Length > b.Length) { var t = a; a = b; b = t; }   // binary search the shorter array
        int m = a.Length, n = b.Length;
        int half = (m + n + 1) / 2;
        int lo = 0, hi = m;
        while (lo <= hi)
        {
            int i = (lo + hi) / 2;
            int j = half - i;
            double aLeft = i > 0 ? a[i - 1] : double.NegativeInfinity;
            double aRight = i < m ? a[i] : double.PositiveInfinity;
            double bLeft = j > 0 ? b[j - 1] : double.NegativeInfinity;
            double bRight = j < n ? b[j] : double.PositiveInfinity;
            if (aLeft <= bRight && bLeft <= aRight)             // valid partition found
            {
                if ((m + n) % 2 == 1) return Math.Max(aLeft, bLeft);
                return (Math.Max(aLeft, bLeft) + Math.Min(aRight, bRight)) / 2.0;
            }
            if (aLeft > bRight) hi = i - 1;                     // too many a on the left
            else lo = i + 1;                                    // too few a on the left
        }
        throw new ArgumentException("inputs are not sorted");
    }

    private static int[] MergeSorted(int[] a, int[] b)
    {
        var outArr = new int[a.Length + b.Length];
        int i = 0, j = 0, k = 0;
        while (i < a.Length && j < b.Length) outArr[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.Length) outArr[k++] = a[i++];
        while (j < b.Length) outArr[k++] = b[j++];
        return outArr;
    }

    /// <summary>O(m+n) merged-middle reference used only for cross-checks.</summary>
    public static double MedianOracle(int[] a, int[] b)
    {
        var merged = MergeSorted(a, b);
        int n = merged.Length;
        return n % 2 == 1 ? merged[n / 2] : (merged[n / 2 - 1] + merged[n / 2]) / 2.0;
    }

    private static int[] SortedRandom(Random rng, int len, int lo, int hi)
    {
        var outArr = new int[len];
        for (int i = 0; i < len; i++) outArr[i] = lo + rng.Next(hi - lo + 1);
        Array.Sort(outArr);
        return outArr;
    }

    public static void Main()
    {
        Check(MedianOfTwoSorted(new[] { 1, 3 }, new[] { 2 }) == 2.0, "2+1");
        Check(MedianOfTwoSorted(new[] { 1, 2 }, new[] { 3, 4 }) == 2.5, "even total");
        Check(MedianOfTwoSorted(new int[0], new[] { 1 }) == 1.0, "empty first");
        Check(MedianOfTwoSorted(new[] { 1, 2, 3, 4, 5 }, new int[0]) == 3.0, "empty second");
        Check(MedianOfTwoSorted(new[] { 0, 0 }, new[] { 0, 0 }) == 0.0, "zeros");
        Check(MedianOfTwoSorted(new[] { 2 }, new[] { 1, 3 }) == 2.0, "wraps both sides");
        Check(MedianOfTwoSorted(new[] { 1, 3, 8 }, new[] { 7, 9, 10, 11 }) == 8.0, "classic mixed");
        Check(MedianOfTwoSorted(new[] { 1, 2 }, new[] { 3, 4, 5, 6 }) == 3.5, "fully left");
        Check(MedianOfTwoSorted(new[] { 5, 6, 7, 8 }, new[] { 1, 2, 3, 4 }) == 4.5, "fully right");
        Check(MedianOfTwoSorted(new[] { 1, 1 }, new[] { 1, 1 }) == 1.0, "duplicates");
        Check(MedianOfTwoSorted(new[] { -5, -3, -1 }, new[] { -2 }) == -2.5, "negatives");
        Check(MedianOfTwoSorted(new[] { 1, 2, 3 }, new[] { 4, 5, 6 }) == 3.5, "interleaved halves");

        bool threw = false;
        try { MedianOfTwoSorted(new int[0], new int[0]); } catch (ArgumentException) { threw = true; }
        Check(threw, "both empty raises");

        var rng = new Random(261);
        for (int test = 0; test < 300; test++)      // cross-check vs the merge oracle
        {
            var a = SortedRandom(rng, rng.Next(9), -20, 20);
            var b = SortedRandom(rng, rng.Next(9), -20, 20);
            if (a.Length == 0 && b.Length == 0) continue;
            Check(Math.Abs(MedianOfTwoSorted(a, b) - MedianOracle(a, b)) < 1e-9, "random agreement");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
