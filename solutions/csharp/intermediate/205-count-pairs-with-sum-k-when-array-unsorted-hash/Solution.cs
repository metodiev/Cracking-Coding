// Task 205 - Count pairs with sum K when array unsorted (hash). (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static long CountPairsWithSum(int[] nums, int target)
    {
        var seen = new Dictionary<int, int>();
        long total = 0;
        foreach (var x in nums)
        {
            if (seen.TryGetValue(target - x, out var count)) total += count;
            seen.TryGetValue(x, out var own);
            seen[x] = own + 1;
        }
        return total;
    }

    public static void Main()
    {
        Check(CountPairsWithSum(new[] { 1, 5, 3, 3, 3 }, 6) == 4, "mixed");
        Check(CountPairsWithSum(new[] { 1, 2, 3, 4 }, 5) == 2, "two pairs");
        Check(CountPairsWithSum(new[] { 1, 1, 1, 1 }, 2) == 6, "all ones");
        Check(CountPairsWithSum(new[] { 1, 2, 3 }, 100) == 0, "none");
        Check(CountPairsWithSum(new[] { 0, 0 }, 0) == 1, "zeros");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
