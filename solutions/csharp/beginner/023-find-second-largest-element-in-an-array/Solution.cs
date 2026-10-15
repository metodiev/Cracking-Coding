// Task 023 - Find second largest element in an array. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int SecondLargest(int[] items)
    {
        int? best = null, second = null;
        foreach (var x in items)
        {
            if (best == null || x > best) { second = best; best = x; }
            else if (x != best && (second == null || x > second)) second = x;
        }
        if (second == null) throw new ArgumentException("no second largest");
        return second.Value;
    }

    public static void Main()
    {
        Check(SecondLargest(new[] { 3, 7, 2, 9, 4 }) == 7, "basic");
        Check(SecondLargest(new[] { 5, 5, 4 }) == 4, "duplicate max");
        Check(SecondLargest(new[] { -1, -2, -3 }) == -2, "negatives");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
