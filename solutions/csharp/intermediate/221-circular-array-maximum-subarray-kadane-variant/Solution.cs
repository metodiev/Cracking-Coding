// Task 221 - Circular array maximum subarray (Kadane variant). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Best subarray sum when the array may wrap past the end.</summary>
    public static int MaxCircularSubarray(int[] nums)
    {
        if (nums.Length == 0) throw new ArgumentException("empty array has no subarray");
        int best = nums[0], cur = nums[0];       // Kadane: maximum subarray
        int worst = nums[0], low = nums[0];      // Kadane: minimum subarray
        int total = 0;
        foreach (int x in nums) total += x;
        for (int i = 1; i < nums.Length; i++)
        {
            cur = Math.Max(nums[i], cur + nums[i]);
            best = Math.Max(best, cur);
            low = Math.Min(nums[i], low + nums[i]);
            worst = Math.Min(worst, low);
        }
        if (best < 0) return best;               // all negative: wrapping picks nothing
        return Math.Max(best, total - worst);
    }

    public static void Main()
    {
        Check(MaxCircularSubarray(new[] { 5, -3, 5 }) == 10, "wrap [5] + [5]");
        Check(MaxCircularSubarray(new[] { -3, -2, -3 }) == -2, "all negative");
        Check(MaxCircularSubarray(new[] { 1, -2, 3, -2 }) == 3, "linear beats wrap");
        Check(MaxCircularSubarray(new[] { 1, 2, 3 }) == 6, "all positive");
        Check(MaxCircularSubarray(new[] { 8, -8, 9 }) == 17, "wrap [9] + [8]");
        Check(MaxCircularSubarray(new[] { 11, -1, 11 }) == 22, "wrap skips the dip");
        Check(MaxCircularSubarray(new[] { 5 }) == 5, "single");
        Check(MaxCircularSubarray(new[] { -5 }) == -5, "single negative");
        Check(MaxCircularSubarray(new[] { 0 }) == 0, "single zero");

        bool threw = false;
        try { MaxCircularSubarray(new int[0]); } catch (ArgumentException) { threw = true; }
        Check(threw, "empty rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
