// Task 213 - Maximum product subarray. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    public static long MaxProduct(int[] nums)
    {
        if (nums.Length == 0) throw new ArgumentException("empty array has no subarray");
        long best = nums[0], curMax = nums[0], curMin = nums[0];
        for (int i = 1; i < nums.Length; i++)
        {
            long x = nums[i];
            long a = x, b = curMax * x, c = curMin * x;
            curMax = Math.Max(a, Math.Max(b, c));
            curMin = Math.Min(a, Math.Min(b, c));
            best = Math.Max(best, curMax);
        }
        return best;
    }

    public static void Main()
    {
        Check(MaxProduct(new[] { 2, 3, -2, 4 }) == 6, "classic");
        Check(MaxProduct(new[] { -2, 0, -1 }) == 0, "zero split");
        Check(MaxProduct(new[] { -2, 3, -4 }) == 24, "double negative");
        Check(MaxProduct(new[] { -2 }) == -2, "single negative");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
