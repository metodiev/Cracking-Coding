// Task 218 - Count subsets that sum to target (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    public static long CountSubsets(int[] nums, int target)
    {
        if (target < 0) return 0;
        var dp = new long[target + 1];
        dp[0] = 1;
        foreach (var x in nums)
            for (int s = target; s >= x; s--)
                dp[s] += dp[s - x];
        return dp[target];
    }

    public static void Main()
    {
        Check(CountSubsets(new[] { 1, 1, 1, 1 }, 2) == 6, "four ones");
        Check(CountSubsets(new[] { 1, 2, 3 }, 3) == 2, "two subsets");
        Check(CountSubsets(new int[0], 0) == 1, "empty subset");
        Check(CountSubsets(new[] { 0 }, 0) == 2, "zero doubles");
        Check(CountSubsets(new[] { 2, 3, 5 }, 5) == 2, "two ways");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
