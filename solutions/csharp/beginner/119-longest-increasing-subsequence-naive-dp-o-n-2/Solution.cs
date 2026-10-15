// Task 119 - Longest increasing subsequence (naive DP O(n^2)). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int LisLength(int[] a)
    {
        if (a.Length == 0) return 0;
        var dp = new int[a.Length];
        for (int i = 0; i < a.Length; i++) dp[i] = 1;
        int best = 1;
        for (int i = 1; i < a.Length; i++)
        {
            for (int j = 0; j < i; j++)
                if (a[j] < a[i] && dp[j] + 1 > dp[i]) dp[i] = dp[j] + 1;
            best = Math.Max(best, dp[i]);
        }
        return best;
    }

    public static List<int> LisWithSequence(int[] a)
    {
        if (a.Length == 0) return new List<int>();
        var dp = new int[a.Length];
        var parent = new int[a.Length];
        for (int i = 0; i < a.Length; i++) { dp[i] = 1; parent[i] = -1; }
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            for (int j = 0; j < i; j++)
                if (a[j] < a[i] && dp[j] + 1 > dp[i])
                {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                }
            if (dp[i] >= dp[best]) best = i;   // >= keeps the LAST index
        }
        var seq = new List<int>();
        for (int node = best; node != -1; node = parent[node]) seq.Add(a[node]);
        seq.Reverse();
        return seq;
    }

    public static void Main()
    {
        var a = new[] { 10, 9, 2, 5, 3, 7, 101, 18 };
        Check(LisLength(a) == 4, "classic sample");
        Check(LisLength(new int[0]) == 0, "empty");
        Check(LisLength(new[] { 3, 2, 1 }) == 1, "decreasing");
        Check(string.Join(",", LisWithSequence(a)) == "2,5,7,18", "reconstruction");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
