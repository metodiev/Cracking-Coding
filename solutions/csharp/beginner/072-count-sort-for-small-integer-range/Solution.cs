// Task 072 - Count sort for small integer range. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int[] CountingSort(int[] a)
    {
        if (a.Length == 0) return new int[0];
        int lo = a[0], hi = a[0];
        foreach (var x in a) { lo = Math.Min(lo, x); hi = Math.Max(hi, x); }
        var counts = new int[hi - lo + 1];
        foreach (var x in a) counts[x - lo]++;
        var outArr = new int[a.Length];
        int k = 0;
        for (int offset = 0; offset < counts.Length; offset++)
            for (int c = 0; c < counts[offset]; c++) outArr[k++] = lo + offset;
        return outArr;
    }

    public static void Main()
    {
        Check(string.Join(",", CountingSort(new[] { 4, 2, 2, 8, 3, 3, 1 })) == "1,2,2,3,3,4,8", "basic");
        Check(CountingSort(new int[0]).Length == 0, "empty");
        Check(string.Join(",", CountingSort(new[] { -3, -1, -3 })) == "-3,-3,-1", "negatives");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
