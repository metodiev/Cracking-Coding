// Task 217 - Subset sum decision problem (DP boolean). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool SubsetSum(int[] nums, int target)
    {
        if (target < 0) return false;
        var dp = new bool[target + 1];
        dp[0] = true;
        foreach (var x in nums)
            for (int s = target; s >= x; s--)
                if (dp[s - x]) dp[s] = true;
        return dp[target];
    }

    public static bool SubsetSumRecursive(int[] nums, int target)
        => Go(nums, 0, target, new Dictionary<(int, int), bool>());

    private static bool Go(int[] nums, int i, int remaining, Dictionary<(int, int), bool> memo)
    {
        if (remaining == 0) return true;
        if (i == nums.Length || remaining < 0) return false;
        if (memo.TryGetValue((i, remaining), out var hit)) return hit;
        bool value = Go(nums, i + 1, remaining, memo) || Go(nums, i + 1, remaining - nums[i], memo);
        memo[(i, remaining)] = value;
        return value;
    }

    public static void Main()
    {
        Check(SubsetSum(new[] { 3, 34, 4, 12, 5, 2 }, 9), "classic true");
        Check(!SubsetSum(new[] { 3, 34, 4, 12, 5, 2 }, 30), "classic false");
        Check(SubsetSum(new int[0], 0), "empty target 0");
        Check(!SubsetSum(new int[0], 5), "empty target 5");
        Check(SubsetSum(new[] { 5 }, 5), "single hit");
        Check(!SubsetSum(new[] { 5 }, 3), "single miss");

        var rng = new Random(2);
        for (int t = 0; t < 200; t++)
        {
            int n = rng.Next(9);
            var nums = new int[n];
            for (int i = 0; i < n; i++) nums[i] = 1 + rng.Next(14);
            int target = rng.Next(40);
            Check(SubsetSum(nums, target) == SubsetSumRecursive(nums, target), "cross-check " + t);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
