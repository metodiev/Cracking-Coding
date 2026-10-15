// Task 226 - Longest arithmetic progression subsequence (DP). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Length of the longest arithmetic subsequence (not necessarily contiguous).</summary>
    public static int LongestArithmeticSubsequence(int[] nums)
    {
        if (nums.Length == 0) return 0;
        var dp = new List<Dictionary<int, int>>();
        for (int i = 0; i < nums.Length; i++) dp.Add(new Dictionary<int, int>());
        int best = 1;
        for (int i = 1; i < nums.Length; i++)
        {
            for (int j = 0; j < i; j++)
            {
                int d = nums[i] - nums[j];
                int prev;
                int length = (dp[j].TryGetValue(d, out prev) ? prev : 1) + 1;
                int cur;
                if (!dp[i].TryGetValue(d, out cur) || length > cur) dp[i][d] = length;
                best = Math.Max(best, length);
            }
        }
        return best;
    }

    public static void Main()
    {
        Check(LongestArithmeticSubsequence(new[] { 3, 6, 9, 12 }) == 4, "whole array");
        Check(LongestArithmeticSubsequence(new[] { 9, 4, 7, 2, 10 }) == 3, "4, 7, 10");
        Check(LongestArithmeticSubsequence(new[] { 1, 2 }) == 2, "pair");
        Check(LongestArithmeticSubsequence(new[] { 4 }) == 1, "single");
        Check(LongestArithmeticSubsequence(new int[0]) == 0, "empty");
        Check(LongestArithmeticSubsequence(new[] { 1, 1, 1 }) == 3, "difference 0");
        Check(LongestArithmeticSubsequence(new[] { 2, 4, 6, 8, 10 }) == 5, "five terms");
        Check(LongestArithmeticSubsequence(new[] { 1, 5, 9, 13, 2, 6, 10, 14 }) == 4, "two interleaved APs");
        Check(LongestArithmeticSubsequence(new[] { 5, 5, 3, 1 }) == 3, "5, 3, 1 (step -2)");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
