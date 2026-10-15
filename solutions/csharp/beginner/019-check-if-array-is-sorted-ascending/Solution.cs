// Task 019 - Check if array is sorted (ascending). (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static bool IsSorted(int[] items)
    {
        for (int i = 0; i + 1 < items.Length; i++)
            if (items[i] > items[i + 1]) return false;
        return true;
    }

    public static void Main()
    {
        Check(IsSorted(new int[0]), "empty");
        Check(IsSorted(new[] { 1, 2, 2, 3 }), "sorted");
        Check(!IsSorted(new[] { 1, 3, 2 }), "unsorted");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
