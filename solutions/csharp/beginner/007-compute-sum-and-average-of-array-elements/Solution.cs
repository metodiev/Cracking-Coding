// Task 007 - Compute sum and average of array elements. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    /// <summary>Returns {sum, average}; average is 0 for an empty array.</summary>
    public static double[] SumAndAverage(int[] items)
    {
        double total = 0;
        foreach (var x in items) total += x;
        double avg = items.Length == 0 ? 0 : total / items.Length;
        return new[] { total, avg };
    }

    public static void Main()
    {
        var r = SumAndAverage(new[] { 1, 2, 3, 4 });
        Check(r[0] == 10 && r[1] == 2.5, "basic");
        Check(SumAndAverage(new int[0])[1] == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
