// Task 210 - Longest subarray with sum equal to K (with negatives) using hashmap. (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int LongestSubarraySumK(int[] nums, int target)
    {
        var firstIndex = new Dictionary<int, int> { [0] = -1 };   // prefix 0 before the array
        int prefix = 0, best = 0;
        for (int j = 0; j < nums.Length; j++)
        {
            prefix += nums[j];
            if (firstIndex.TryGetValue(prefix - target, out var idx))
                best = Math.Max(best, j - idx);
            if (!firstIndex.ContainsKey(prefix)) firstIndex[prefix] = j;
        }
        return best;
    }

    public static void Main()
    {
        Check(LongestSubarraySumK(new[] { 1, 2, 3, 1, 1, 1, 1 }, 6) == 4, "positives");
        Check(LongestSubarraySumK(new[] { -1, 1, 1 }, 1) == 3, "whole array sums to 1");
        Check(LongestSubarraySumK(new[] { 1, -1, 5, -2, 3 }, 3) == 4, "mixed");
        Check(LongestSubarraySumK(new[] { -2, -1, 2, 1 }, 1) == 2, "negatives");
        Check(LongestSubarraySumK(new[] { 0, 0, 0 }, 0) == 3, "zeros");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
