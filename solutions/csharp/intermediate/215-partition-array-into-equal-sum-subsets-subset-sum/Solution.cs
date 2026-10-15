// Task 215 - Partition array into equal-sum subsets (subset sum). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    public static bool CanPartition(int[] nums)
    {
        int total = 0;
        foreach (var x in nums) total += x;
        if (total % 2 != 0) return false;
        int half = total / 2;
        var dp = new bool[half + 1];
        dp[0] = true;
        foreach (var x in nums)
            for (int s = half; s >= x; s--)
                if (dp[s - x]) dp[s] = true;
        return dp[half];
    }

    public static void Main()
    {
        Check(CanPartition(new[] { 1, 5, 11, 5 }), "classic true");
        Check(!CanPartition(new[] { 1, 2, 3, 5 }), "odd total");
        Check(CanPartition(new[] { 2, 2 }), "pair");
        Check(!CanPartition(new[] { 1 }), "single");
        Check(CanPartition(new int[0]), "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
