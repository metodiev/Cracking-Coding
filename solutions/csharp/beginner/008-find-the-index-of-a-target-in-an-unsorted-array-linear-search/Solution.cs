// Task 008 - Find the index of a target in an unsorted array (linear search). (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int LinearSearch(int[] items, int target)
    {
        for (int i = 0; i < items.Length; i++)
            if (items[i] == target) return i;
        return -1;
    }

    public static void Main()
    {
        Check(LinearSearch(new[] { 4, 2, 7, 1 }, 7) == 2, "found");
        Check(LinearSearch(new[] { 4, 2, 7, 1 }, 9) == -1, "missing");
        Check(LinearSearch(new int[0], 1) == -1, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
