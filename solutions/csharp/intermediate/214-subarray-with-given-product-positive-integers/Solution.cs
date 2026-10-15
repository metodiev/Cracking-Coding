// Task 214 - Subarray with given product (positive integers). (Tier: Intermediate, CTCI Ch 1)
using System;

public static class Solution
{
    /// <summary>Returns (start, end) inclusive, or null.</summary>
    public static (int, int)? SubarrayWithProduct(int[] nums, int target)
    {
        foreach (var x in nums)
            if (x <= 0) throw new ArgumentException("requires positive values");
        if (target < 1) return null;
        int left = 0;
        long product = 1;
        for (int right = 0; right < nums.Length; right++)
        {
            product *= nums[right];
            while (left <= right && product > target)
            {
                product /= nums[left];
                left++;
            }
            if (left <= right && product == target) return (left, right);
        }
        return null;
    }

    public static void Main()
    {
        Check(SubarrayWithProduct(new[] { 2, 3, 1, 2, 4 }, 6) == (0, 1), "first match");
        Check(SubarrayWithProduct(new[] { 1, 2, 3, 4 }, 6) == (0, 2), "later match");
        Check(SubarrayWithProduct(new[] { 10 }, 10) == (0, 0), "single");
        Check(SubarrayWithProduct(new[] { 3, 1, 1 }, 2) == null, "none with ones");
        Check(SubarrayWithProduct(new[] { 1, 1, 1 }, 1) == (0, 0), "ones");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
