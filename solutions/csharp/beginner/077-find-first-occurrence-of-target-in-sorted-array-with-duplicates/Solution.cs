// Task 077 - Find first occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int FirstOccurrence(int[] a, int target)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo < a.Length && a[lo] == target ? lo : -1;
    }

    public static void Main()
    {
        var a = new[] { 1, 2, 2, 2, 3, 4, 4 };
        Check(FirstOccurrence(a, 2) == 1, "duplicated middle");
        Check(FirstOccurrence(a, 4) == 5, "duplicated tail");
        Check(FirstOccurrence(a, 1) == 0, "single head");
        Check(FirstOccurrence(a, 9) == -1, "absent");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
