// Task 228 - Maximum sum increasing subsequence. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Largest sum of a strictly increasing subsequence.</summary>
    public static int MaxSumIncreasingSubsequence(int[] nums)
    {
        if (nums.Length == 0) return 0;
        var dp = (int[])nums.Clone();
        int best = dp[0];
        for (int i = 1; i < nums.Length; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (nums[j] < nums[i] && dp[j] + nums[i] > dp[i]) dp[i] = dp[j] + nums[i];
            }
            best = Math.Max(best, dp[i]);
        }
        return best;
    }

    public static void Main()
    {
        Check(MaxSumIncreasingSubsequence(new[] { 1, 101, 2, 3, 100, 4 }) == 106, "1+2+3+100");
        Check(MaxSumIncreasingSubsequence(new[] { 10, 1 }) == 10, "falling pair");
        Check(MaxSumIncreasingSubsequence(new[] { 5, 4, 3, 2, 1 }) == 5, "all falling");
        Check(MaxSumIncreasingSubsequence(new[] { 1, 2, 3, 4 }) == 10, "whole array");
        Check(MaxSumIncreasingSubsequence(new[] { 7 }) == 7, "single");
        Check(MaxSumIncreasingSubsequence(new int[0]) == 0, "empty");
        Check(MaxSumIncreasingSubsequence(new[] { 3, 3 }) == 3, "not strict");
        Check(MaxSumIncreasingSubsequence(new[] { -2, -1, -3 }) == -1, "all negative");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
