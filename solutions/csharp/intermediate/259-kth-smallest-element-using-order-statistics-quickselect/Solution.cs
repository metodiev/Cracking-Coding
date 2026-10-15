// Task 259 - Kth smallest element using order statistics / quickselect. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const long LcgMod = 2147483647L;
    private const long LcgMul = 48271L;

    // Lomuto partition with a[hi] as the deterministic pivot; returns its final index.
    private static int Partition(int[] a, int lo, int hi)
    {
        int pivot = a[hi];
        int store = lo;
        for (int i = lo; i < hi; i++)
        {
            if (a[i] <= pivot)
            {
                (a[store], a[i]) = (a[i], a[store]);
                store++;
            }
        }
        (a[store], a[hi]) = (a[hi], a[store]);
        return store;
    }

    // The k-th smallest value (1-based k); the caller's array is copied first.
    // The loop is iterative, so a bad pivot sequence never grows the call stack.
    public static int KthSmallest(int[] values, int k)
    {
        if (k < 1 || k > values.Length) throw new Exception("k out of range");
        var a = (int[])values.Clone();
        int lo = 0;
        int hi = a.Length - 1;
        int target = k - 1;
        while (true)
        {
            int p = Partition(a, lo, hi);
            if (p == target) return a[p];
            if (p < target) lo = p + 1;
            else hi = p - 1;
        }
    }

    // Park-Miller minimal standard generator: identical values in every language.
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

    // Every k agrees with the sorted() oracle.
    private static bool CaseAgainstOracle(int[] values)
    {
        var ordered = (int[])values.Clone();
        Array.Sort(ordered);
        for (int k = 1; k <= values.Length; k++)
        {
            if (KthSmallest(values, k) != ordered[k - 1]) return false;
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

    public static void Main()
    {
        Check(KthSmallest(new[] { 7 }, 1) == 7, "single element");
        Check(KthSmallest(new[] { 3, 1, 2 }, 1) == 1, "k=1");
        Check(KthSmallest(new[] { 3, 1, 2 }, 2) == 2, "k=2");
        Check(KthSmallest(new[] { 3, 1, 2 }, 3) == 3, "k=3");
        Check(KthSmallest(new[] { 5, 5, 5, 5 }, 3) == 5, "all equal");
        Check(KthSmallest(new[] { 2, 1, 1, 3 }, 2) == 1, "duplicates: sorted [1,1,2,3]");

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

        // Fixed cases (duplicates included) cross-checked against the sorted() oracle, all k.
        var cases = new List<int[]>
        {
            LcgValues(259001, 1, 5),
            LcgValues(259002, 2, 5),
            LcgValues(259003, 3, 7),
            LcgValues(259004, 5, 20),
            LcgValues(259005, 8, 3),
            LcgValues(259006, 16, 1000),
            LcgValues(259007, 33, 10),
            LcgValues(259008, 50, 50),
            Range(64),
            ReverseRange(64),
            Repeat(7, 32),
            new[] { 5, 5, 4, 4, 3, 3, 2, 2, 1, 1 },
            LcgValues(259009, 200, 10000),
        };
        foreach (int[] c in cases) Check(CaseAgainstOracle(c), "fixed case vs sorted oracle");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
