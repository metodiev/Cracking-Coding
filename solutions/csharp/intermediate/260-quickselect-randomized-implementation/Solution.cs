// Task 260 - Quickselect randomized implementation. (Tier: Intermediate, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const long LcgMod = 2147483647L;
    private const long LcgMul = 48271L;
    private const long Seed = 260260L;

    // Dutch flag; returns (lt, gt): a[lo..lt-1] < pivot == a[lt..gt], a[gt+1..hi] > pivot.
    private static (int Lt, int Gt) Partition3(int[] a, int lo, int hi, int pivot)
    {
        int lt = lo;
        int i = lo;
        int gt = hi;
        while (i <= gt)
        {
            if (a[i] < pivot)
            {
                (a[lt], a[i]) = (a[i], a[lt]);
                lt++;
                i++;
            }
            else if (a[i] > pivot)
            {
                (a[i], a[gt]) = (a[gt], a[i]);
                gt--;
            }
            else
            {
                i++;
            }
        }
        return (lt, gt);
    }

    // Randomized quickselect with a fixed-seed LCG (reproducible runs); the returned value
    // never depends on the seed. Duplicates collapse into one equal block, so all-equal
    // input finishes after a single partition; the loop is iterative, so no stack blow-up.
    // Expected O(n), worst case O(n^2). k is 1-based; the caller's array is copied.
    private static int KthSmallest(int[] values, int k, long seed)
    {
        if (k < 1 || k > values.Length) throw new Exception("k out of range");
        var a = (int[])values.Clone();
        long state = seed % LcgMod;
        if (state == 0) state = 1;
        int lo = 0;
        int hi = a.Length - 1;
        int target = k - 1;
        while (true)
        {
            state = (state * LcgMul) % LcgMod;
            int pivot = a[lo + (int)(state % (hi - lo + 1))];
            var (lt, gt) = Partition3(a, lo, hi, pivot);
            if (target < lt) hi = lt - 1;
            else if (target > gt) lo = gt + 1;
            else return a[target];
        }
    }

    private static int KthSmallest(int[] values, int k)
    {
        return KthSmallest(values, k, Seed);
    }

    private static int[] LcgValues(long seed, int count, int bound)
    {
        var outArr = new int[count];
        long state = seed;
        for (int i = 0; i < count; i++)
        {
            state = (state * LcgMul) % LcgMod;
            outArr[i] = (int)(state % bound);
        }
        return outArr;
    }

    // Every k agrees with sorted order, and the value is seed-independent.
    private static bool AgreesWithOracle(int[] values)
    {
        var ordered = (int[])values.Clone();
        Array.Sort(ordered);
        for (int k = 1; k <= values.Length; k++)
        {
            if (KthSmallest(values, k) != ordered[k - 1]) return false;
            if (KthSmallest(values, k, 999) != ordered[k - 1]) return false;
        }
        return true;
    }

    private static int[] Range(int n)
    {
        var outArr = new int[n];
        for (int i = 0; i < n; i++) outArr[i] = i;
        return outArr;
    }

    private static int[] ReverseRange(int n)
    {
        var outArr = new int[n];
        for (int i = 0; i < n; i++) outArr[i] = n - 1 - i;
        return outArr;
    }

    private static int[] Repeat(int value, int n)
    {
        var outArr = new int[n];
        for (int i = 0; i < n; i++) outArr[i] = value;
        return outArr;
    }

    private static int[] Sawtooth(int n)
    {
        var outArr = new int[n];
        for (int i = 0; i < n; i++) outArr[i] = i % 17;
        return outArr;
    }

    public static void Main()
    {
        Check(KthSmallest(new[] { 7 }, 1) == 7, "single element");
        Check(KthSmallest(new[] { 3, 1, 2 }, 2) == 2, "k=2");
        Check(KthSmallest(new[] { 5, 5, 5, 5 }, 3) == 5, "all equal small");
        Check(KthSmallest(new[] { 2, 1, 1, 3 }, 2) == 1, "duplicates");

        var original = new[] { 9, 4, 7, 1, 8, 2, 6, 3, 5 };
        Check(KthSmallest(original, 4) == 4, "middle");
        Check(string.Join(",", original) == "9,4,7,1,8,2,6,3,5", "input untouched");

        foreach (int bad in new[] { 0, -1, 10 })
        {
            bool threw = false;
            try { KthSmallest(new[] { 1, 2, 3 }, bad); }
            catch (Exception) { threw = true; }
            Check(threw, "k out of range throws: " + bad);
        }

        var cases = new List<int[]>
        {
            LcgValues(260001, 1, 5),
            LcgValues(260002, 2, 5),
            LcgValues(260003, 3, 7),
            LcgValues(260004, 5, 20),
            LcgValues(260005, 8, 3),
            LcgValues(260006, 16, 1000),
            LcgValues(260007, 33, 10),
            LcgValues(260008, 50, 50),
            Range(64),
            ReverseRange(64),
            Repeat(9, 17),
            LcgValues(260009, 200, 10000),
        };
        foreach (int[] c in cases) Check(AgreesWithOracle(c), "oracle agreement");

        var bigEqual = Repeat(7, 5000);
        foreach (int k in new[] { 1, 2, 2500, 4999, 5000 })
        {
            Check(KthSmallest(bigEqual, k) == 7, "big all-equal k=" + k);
        }
        var ascending = Range(4000);
        var descending = ReverseRange(4000);
        var sawtooth = Sawtooth(1000);
        var sawOrdered = (int[])sawtooth.Clone();
        Array.Sort(sawOrdered);
        foreach (int k in new[] { 1, 2000, 4000 })
        {
            Check(KthSmallest(ascending, k) == k - 1, "ascending k=" + k);
            Check(KthSmallest(descending, k) == k - 1, "descending k=" + k);
        }
        foreach (int k in new[] { 1, 500, 1000 })
        {
            Check(KthSmallest(sawtooth, k) == sawOrdered[k - 1], "sawtooth k=" + k);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
