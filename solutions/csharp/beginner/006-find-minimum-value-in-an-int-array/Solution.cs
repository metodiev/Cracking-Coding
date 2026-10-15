// Task 006 - Find minimum value in an int array. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int MinValue(int[] items)
    {
        if (items.Length == 0) throw new ArgumentException("empty array");
        int best = items[0];
        foreach (var x in items) if (x < best) best = x;
        return best;
    }

    public static void Main()
    {
        Check(MinValue(new[] { 3, 7, 2, 9, 4 }) == 2, "basic");
        Check(MinValue(new[] { -5, -1, -3 }) == -5, "negatives");
        Check(MinValue(new[] { 42 }) == 42, "single");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
