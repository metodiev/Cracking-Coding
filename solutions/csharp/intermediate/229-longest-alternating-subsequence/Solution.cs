// Task 229 - Longest alternating subsequence. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Longest up-down (wiggle) subsequence length.</summary>
    public static int LongestAlternatingSubsequence(int[] nums)
    {
        if (nums.Length == 0) return 0;
        int up = 1, down = 1;
        for (int i = 1; i < nums.Length; i++)
        {
            if (nums[i] > nums[i - 1]) up = down + 1;
            else if (nums[i] < nums[i - 1]) down = up + 1;
        }
        return Math.Max(up, down);
    }

    public static void Main()
    {
        Check(LongestAlternatingSubsequence(new[] { 1, 7, 4, 9, 2, 5 }) == 6, "full zigzag");
        Check(LongestAlternatingSubsequence(new[] { 1, 17, 5, 10, 13, 15, 10, 5, 16, 8 }) == 7, "long sample");
        Check(LongestAlternatingSubsequence(new[] { 8 }) == 1, "single");
        Check(LongestAlternatingSubsequence(new int[0]) == 0, "empty");
        Check(LongestAlternatingSubsequence(new[] { 1, 2, 3, 4 }) == 2, "monotone");
        Check(LongestAlternatingSubsequence(new[] { 2, 1 }) == 2, "one down step");
        Check(LongestAlternatingSubsequence(new[] { 3, 3 }) == 1, "equal is not a step");
        Check(LongestAlternatingSubsequence(new[] { 3, 3, 3, 2 }) == 2, "equal then down");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
