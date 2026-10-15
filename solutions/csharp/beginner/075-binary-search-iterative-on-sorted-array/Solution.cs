// Task 075 - Binary search iterative on sorted array. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int BinarySearch(int[] a, int target)
    {
        int lo = 0, hi = a.Length - 1;
        while (lo <= hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] == target) return mid;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    public static void Main()
    {
        var a = new[] { 1, 3, 5, 7, 9, 11 };
        Check(BinarySearch(a, 7) == 3, "middle");
        Check(BinarySearch(a, 1) == 0, "first");
        Check(BinarySearch(a, 11) == 5, "last");
        Check(BinarySearch(a, 4) == -1, "absent");
        Check(BinarySearch(new int[0], 5) == -1, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
