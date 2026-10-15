// Task 078 - Find last occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int LastOccurrence(int[] a, int target)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (a[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        int idx = lo - 1;
        return idx >= 0 && a[idx] == target ? idx : -1;
    }

    public static void Main()
    {
        var a = new[] { 1, 2, 2, 2, 3, 4, 4 };
        Check(LastOccurrence(a, 2) == 3, "duplicated middle");
        Check(LastOccurrence(a, 4) == 6, "duplicated tail");
        Check(LastOccurrence(a, 1) == 0, "single head");
        Check(LastOccurrence(a, 9) == -1, "absent");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
