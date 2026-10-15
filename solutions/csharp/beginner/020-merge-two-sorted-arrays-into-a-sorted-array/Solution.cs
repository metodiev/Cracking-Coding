// Task 020 - Merge two sorted arrays into a sorted array. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int[] MergeSorted(int[] a, int[] b)
    {
        var outArr = new int[a.Length + b.Length];
        int i = 0, j = 0, k = 0;
        while (i < a.Length && j < b.Length)
            outArr[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.Length) outArr[k++] = a[i++];
        while (j < b.Length) outArr[k++] = b[j++];
        return outArr;
    }

    public static void Main()
    {
        Check(string.Join(",", MergeSorted(new[] { 1, 3, 5 }, new[] { 2, 4, 6 })) == "1,2,3,4,5,6", "basic");
        Check(string.Join(",", MergeSorted(new int[0], new[] { 1, 2 })) == "1,2", "empty first");
        Check(string.Join(",", MergeSorted(new[] { 2, 2 }, new[] { 2 })) == "2,2,2", "duplicates");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
