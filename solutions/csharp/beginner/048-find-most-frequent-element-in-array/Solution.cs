// Task 048 - Find most frequent element in array. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static int MostFrequent(int[] items)
    {
        if (items.Length == 0) throw new ArgumentException("empty array");
        var counts = new Dictionary<int, int>();
        foreach (var x in items)
        {
            counts.TryGetValue(x, out var n);
            counts[x] = n + 1;
        }
        int best = items[0], bestCount = 0;
        foreach (var x in items)
        {
            if (counts[x] > bestCount) { best = x; bestCount = counts[x]; }
        }
        return best;
    }

    public static void Main()
    {
        Check(MostFrequent(new[] { 1, 3, 2, 3, 4, 3 }) == 3, "basic");
        Check(MostFrequent(new[] { 1, 2, 1, 2 }) == 1, "tie first");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
