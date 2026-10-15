// Task 021 - Rotate array right by k positions (create new array). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int[] RotateRight(int[] items, int k)
    {
        if (items.Length == 0) return new int[0];
        k %= items.Length;
        var outArr = new int[items.Length];
        for (int i = 0; i < items.Length; i++)
            outArr[(i + k) % items.Length] = items[i];
        return outArr;
    }

    public static void Main()
    {
        Check(string.Join(",", RotateRight(new[] { 1, 2, 3, 4, 5 }, 2)) == "4,5,1,2,3", "k=2");
        Check(string.Join(",", RotateRight(new[] { 1, 2 }, 3)) == "2,1", "k wraps");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
