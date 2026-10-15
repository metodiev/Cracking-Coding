// Task 299 - DP over subsets (SOS DP simple task). (Tier: Intermediate, CTCI Beyond — Networking, OS & Real-Time)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>f[mask] = sum of a[sub] for all sub ⊆ mask. Naive work is 3^n; this is n * 2^n.</summary>
    public static int[] SumOverSubsets(int[] a)
    {
        if (a.Length == 0 || (a.Length & (a.Length - 1)) != 0)
            throw new ArgumentException("length must be a power of two");
        int n = 0;
        while ((1 << n) < a.Length) n++;
        var f = (int[])a.Clone();
        for (int i = 0; i < n; i++)
        {
            int bit = 1 << i;
            for (int mask = 0; mask < f.Length; mask++)
            {
                if ((mask & bit) != 0) f[mask] += f[mask ^ bit];
            }
        }
        return f;
    }

    /// <summary>Oracle: enumerate every submask of every mask explicitly (the O(3^n) loop).</summary>
    public static int[] BruteSos(int[] a)
    {
        var result = new int[a.Length];
        for (int mask = 0; mask < a.Length; mask++)
        {
            int total = 0;
            int sub = mask;
            while (true)
            {
                total += a[sub];
                if (sub == 0) break;
                sub = (sub - 1) & mask;
            }
            result[mask] = total;
        }
        return result;
    }

    private static int PopCount(int value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }

    private static int seed = 606;

    private static int[] LcgArray(int size)
    {
        var values = new int[size];
        for (int i = 0; i < size; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 50;
        }
        return values;
    }

    public static void Main()
    {
        Check(Same(SumOverSubsets(new[] { 1, 2, 4, 8 }), new[] { 1, 3, 5, 15 }), "n = 2");
        Check(Same(SumOverSubsets(new[] { 7 }), new[] { 7 }), "n = 0");

        foreach (int n in new[] { 1, 2, 3, 4, 10 })
        {
            var a = LcgArray(1 << n);
            var f = SumOverSubsets(a);
            Check(Same(f, BruteSos(a)), "n = " + n + " vs brute force");
            long expected = 0;
            for (int mask = 0; mask < a.Length; mask++)
            {
                expected += (long)a[mask] << (n - PopCount(mask));
            }
            long total = 0;
            foreach (int value in f) total += value;
            Check(total == expected, "n = " + n + " superset identity");
        }
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
