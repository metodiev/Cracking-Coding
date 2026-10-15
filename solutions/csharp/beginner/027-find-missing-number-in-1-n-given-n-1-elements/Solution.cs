// Task 027 - Find missing number in 1..n given n-1 elements. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int MissingNumber(int[] nums, int n)
    {
        long expected = (long)n * (n + 1) / 2;
        long actual = 0;
        foreach (var x in nums) actual += x;
        return (int)(expected - actual);
    }

    public static void Main()
    {
        Check(MissingNumber(new[] { 1, 2, 4, 5 }, 5) == 3, "middle");
        Check(MissingNumber(new[] { 2 }, 2) == 1, "first");
        Check(MissingNumber(new[] { 1 }, 2) == 2, "last");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
