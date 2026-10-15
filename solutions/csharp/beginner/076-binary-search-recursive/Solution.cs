// Task 076 - Binary search recursive. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int BinarySearchRecursive(int[] a, int target) => Search(a, target, 0, a.Length - 1);

    private static int Search(int[] a, int target, int lo, int hi)
    {
        if (lo > hi) return -1;
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == target) return mid;
        if (a[mid] < target) return Search(a, target, mid + 1, hi);
        return Search(a, target, lo, mid - 1);
    }

    public static void Main()
    {
        var a = new[] { 2, 4, 6, 8, 10 };
        Check(BinarySearchRecursive(a, 8) == 3, "found");
        Check(BinarySearchRecursive(a, 2) == 0, "first");
        Check(BinarySearchRecursive(a, 5) == -1, "absent");
        Check(BinarySearchRecursive(new int[0], 1) == -1, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
