// Task 211 - Subarray sum equals K count (hashmap). (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int CountSubarraysSumK(int[] nums, int target)
    {
        var counts = new Dictionary<int, int> { [0] = 1 };
        int prefix = 0, total = 0;
        foreach (var x in nums)
        {
            prefix += x;
            total += counts.TryGetValue(prefix - target, out var c) ? c : 0;
            counts.TryGetValue(prefix, out var own);
            counts[prefix] = own + 1;
        }
        return total;
    }

    public static void Main()
    {
        Check(CountSubarraysSumK(new[] { 1, 1, 1 }, 2) == 2, "ones");
        Check(CountSubarraysSumK(new[] { 1, 2, 3 }, 3) == 2, "two");
        Check(CountSubarraysSumK(new[] { -1, -1, 1 }, 0) == 1, "mixed");
        Check(CountSubarraysSumK(new[] { 0, 0 }, 0) == 3, "zeros");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
