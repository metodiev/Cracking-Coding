// Task 024 - Remove element at index from array (in-place). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int RemoveAt(int[] items, int index)
    {
        if (index < 0 || index >= items.Length)
            throw new IndexOutOfRangeException("index out of range");
        for (int i = index; i + 1 < items.Length; i++)
            items[i] = items[i + 1];
        return items.Length - 1;
    }

    public static void Main()
    {
        var a = new[] { 10, 20, 30, 40 };
        Check(RemoveAt(a, 1) == 3 && a[0] == 10 && a[1] == 30 && a[2] == 40, "middle");
        Check(RemoveAt(new[] { 1 }, 0) == 0, "single");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
