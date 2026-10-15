// Task 022 - Rotate array in-place by k positions. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int[] RotateInPlace(int[] items, int k)
    {
        int n = items.Length;
        if (n == 0) return items;
        k %= n;
        Reverse(items, 0, n - 1);
        Reverse(items, 0, k - 1);
        Reverse(items, k, n - 1);
        return items;
    }

    private static void Reverse(int[] a, int lo, int hi)
    {
        while (lo < hi) { int t = a[lo]; a[lo++] = a[hi]; a[hi--] = t; }
    }

    public static void Main()
    {
        Check(string.Join(",", RotateInPlace(new[] { 1, 2, 3, 4, 5 }, 2)) == "4,5,1,2,3", "k=2");
        Check(string.Join(",", RotateInPlace(new[] { 1, 2 }, 0)) == "1,2", "k=0");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
