// Task 079 - Find insert position for target in sorted array. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int SearchInsert(int[] a, int target)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void Main()
    {
        var a = new[] { 1, 3, 5, 6 };
        Check(SearchInsert(a, 5) == 2, "existing");
        Check(SearchInsert(a, 2) == 1, "between");
        Check(SearchInsert(a, 7) == 4, "after end");
        Check(SearchInsert(a, 0) == 0, "before start");
        Check(SearchInsert(new int[0], 42) == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
