// Task 212 - Find subarray with maximum sum (Kadane's algorithm). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Returns (bestSum, start, endExclusive).</summary>
    public static (int sum, int start, int end) MaxSubarray(int[] nums)
    {
        if (nums.Length == 0) throw new ArgumentException("empty array has no subarray");
        int best = nums[0], cur = nums[0];
        int bestStart = 0, bestEnd = 0, start = 0;
        for (int i = 1; i < nums.Length; i++)
        {
            if (cur + nums[i] < nums[i])
            {
                start = i;
                cur = nums[i];
            }
            else
            {
                cur += nums[i];
            }
            if (cur > best)
            {
                best = cur;
                bestStart = start;
                bestEnd = i;
            }
        }
        return (best, bestStart, bestEnd + 1);
    }

    public static void Main()
    {
        var r = MaxSubarray(new[] { -2, 1, -3, 4, -1, 2, 1, -5, 4 });
        Check(r == (6, 3, 7), "classic");
        Check(MaxSubarray(new[] { -3, -1, -2 }).sum == -1, "all negative");
        Check(MaxSubarray(new[] { 5, -1, 5 }).sum == 9, "dip in middle");

        bool threw = false;
        try { MaxSubarray(new int[0]); } catch (ArgumentException) { threw = true; }
        Check(threw, "empty rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
