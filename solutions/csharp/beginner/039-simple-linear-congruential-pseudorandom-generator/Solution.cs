// Task 039 - Simple linear congruential pseudorandom generator. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static long[] Lcg(long seed, int n)
    {
        long a = 1103515245L, c = 12345L, m = 1L << 31;
        long x = seed;
        var outArr = new long[n];
        for (int i = 0; i < n; i++)
        {
            x = (a * x + c) % m;
            outArr[i] = x;
        }
        return outArr;
    }

    public static void Main()
    {
        var a = Lcg(42, 5);
        Check(a.Length == 5, "length");
        Check(string.Join(",", a) == string.Join(",", Lcg(42, 5)), "deterministic");
        Check(string.Join(",", a) != string.Join(",", Lcg(43, 5)), "seed-sensitive");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
