// Task 227 - Longest bitonic subarray. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Longest contiguous subarray that strictly rises then strictly falls.</summary>
    public static int LongestBitonicSubarray(int[] nums)
    {
        int n = nums.Length;
        if (n == 0) return 0;
        var inc = new int[n];
        var dec = new int[n];
        inc[0] = 1;
        for (int i = 1; i < n; i++) inc[i] = nums[i] > nums[i - 1] ? inc[i - 1] + 1 : 1;
        dec[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) dec[i] = nums[i] > nums[i + 1] ? dec[i + 1] + 1 : 1;
        int best = 0;
        for (int i = 0; i < n; i++) best = Math.Max(best, inc[i] + dec[i] - 1);
        return best;
    }

    public static void Main()
    {
        Check(LongestBitonicSubarray(new[] { 2, 1, 4, 5, 6, 3, 2 }) == 6, "[1,4,5,6,3,2]");
        Check(LongestBitonicSubarray(new[] { 1, 5, 1 }) == 3, "single peak");
        Check(LongestBitonicSubarray(new[] { 1, 2, 3, 4 }) == 4, "monotone up");
        Check(LongestBitonicSubarray(new[] { 4, 3, 2, 1 }) == 4, "monotone down");
        Check(LongestBitonicSubarray(new[] { 7 }) == 1, "single");
        Check(LongestBitonicSubarray(new int[0]) == 0, "empty");
        Check(LongestBitonicSubarray(new[] { 1, 1 }) == 1, "not strict");
        Check(LongestBitonicSubarray(new[] { 12, 4, 78, 90, 45, 23, 100 }) == 5, "[4,78,90,45,23]");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
