// Task 209 - Longest subarray with sum equal to K (array of positives). (Tier: Intermediate, CTCI Ch 1)
using System;

public static class Solution
{
    public static int LongestSubarraySumK(int[] nums, int target)
    {
        foreach (var x in nums)
            if (x <= 0) throw new ArgumentException("requires positive values");
        int lo = 0, total = 0, best = 0;
        for (int hi = 0; hi < nums.Length; hi++)
        {
            total += nums[hi];
            while (total > target && lo <= hi)
            {
                total -= nums[lo];
                lo++;
            }
            if (total == target) best = Math.Max(best, hi - lo + 1);
        }
        return best;
    }

    public static void Main()
    {
        Check(LongestSubarraySumK(new[] { 1, 2, 3, 1, 1, 1, 1 }, 6) == 4, "classic");
        Check(LongestSubarraySumK(new[] { 1, 1, 1, 1 }, 2) == 2, "all ones");
        Check(LongestSubarraySumK(new[] { 5 }, 5) == 1, "single");
        Check(LongestSubarraySumK(new[] { 1, 2, 3 }, 100) == 0, "too small");
        Check(LongestSubarraySumK(new[] { 2, 2, 2 }, 6) == 3, "whole array");

        bool threw = false;
        try { LongestSubarraySumK(new[] { 1, -1, 1 }, 0); } catch (ArgumentException) { threw = true; }
        Check(threw, "rejects non-positive input");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
