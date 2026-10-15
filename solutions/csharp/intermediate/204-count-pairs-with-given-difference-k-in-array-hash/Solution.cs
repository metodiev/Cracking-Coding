// Task 204 - Count pairs with given difference K in array (hash). (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static long CountPairsWithDifference(int[] nums, int k)
    {
        if (k < 0) throw new ArgumentException("k must be >= 0");
        var freq = new Dictionary<int, int>();
        foreach (var x in nums)
        {
            freq.TryGetValue(x, out var c);
            freq[x] = c + 1;
        }
        long total = 0;
        if (k == 0)
        {
            foreach (var c in freq.Values) total += (long)c * (c - 1) / 2;
            return total;
        }
        foreach (var kv in freq)
            total += (long)kv.Value * (freq.TryGetValue(kv.Key + k, out var higher) ? higher : 0);
        return total;
    }

    public static void Main()
    {
        Check(CountPairsWithDifference(new[] { 1, 5, 3, 4, 2 }, 2) == 3, "classic");
        Check(CountPairsWithDifference(new[] { 1, 1, 1, 1 }, 0) == 6, "k=0 combos");
        Check(CountPairsWithDifference(new[] { 1, 2, 3, 4 }, 10) == 0, "none");
        Check(CountPairsWithDifference(new int[0], 3) == 0, "empty");
        Check(CountPairsWithDifference(new[] { 5, 5, 5, 1 }, 4) == 3, "multiplicity");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
