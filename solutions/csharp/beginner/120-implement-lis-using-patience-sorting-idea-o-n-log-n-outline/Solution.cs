// Task 120 - Implement LIS using patience sorting idea (O(n log n) outline). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static int LowerBound(int[] tails, int size, int x)
    {
        int lo = 0, hi = size;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (tails[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static int LisLength(int[] a)
    {
        var tails = new int[a.Length];
        int size = 0;
        foreach (var x in a)
        {
            int i = LowerBound(tails, size, x);
            tails[i] = x;
            if (i == size) size++;
        }
        return size;
    }

    public static List<int> LisReconstructed(int[] a)
    {
        if (a.Length == 0) return new List<int>();
        var tails = new int[a.Length];
        var tailIdx = new int[a.Length];
        var parent = new int[a.Length];
        for (int i = 0; i < a.Length; i++) parent[i] = -1;
        int size = 0;
        for (int i = 0; i < a.Length; i++)
        {
            int lo = LowerBound(tails, size, a[i]);
            tails[lo] = a[i];
            tailIdx[lo] = i;
            parent[i] = lo > 0 ? tailIdx[lo - 1] : -1;
            if (lo == size) size++;
        }
        var seq = new List<int>(new int[size]);
        int node = tailIdx[size - 1];
        for (int k = size - 1; k >= 0; k--)
        {
            seq[k] = a[node];
            node = parent[node];
        }
        return seq;
    }

    public static void Main()
    {
        var a = new[] { 10, 9, 2, 5, 3, 7, 101, 18 };
        Check(LisLength(a) == 4, "classic sample");
        Check(LisLength(new int[0]) == 0, "empty");
        Check(LisLength(new[] { 3, 2, 1 }) == 1, "decreasing");
        Check(string.Join(",", LisReconstructed(a)) == "2,3,7,18", "reconstruction");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
