// Task 126 - House robber problem (1D DP). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static long Rob(int[] houses)
    {
        long take = 0, skip = 0;
        foreach (var value in houses)
        {
            long newTake = skip + value;
            skip = Math.Max(skip, take);
            take = newTake;
        }
        return Math.Max(take, skip);
    }

    public static void Main()
    {
        Check(Rob(new[] { 2, 7, 9, 3, 1 }) == 12, "classic");
        Check(Rob(new int[0]) == 0, "empty");
        Check(Rob(new[] { 5 }) == 5, "single");
        Check(Rob(new[] { 2, 1, 1, 2 }) == 4, "both ends");
        Check(Rob(new[] { 5, 1, 1, 5 }) == 10, "non-adjacent ends");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
