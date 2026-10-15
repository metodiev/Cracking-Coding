// Task 069 - Selection sort. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static void SelectionSort(int[] a)
    {
        int n = a.Length;
        for (int i = 0; i < n - 1; i++)
        {
            int minIdx = i;
            for (int j = i + 1; j < n; j++)
                if (a[j] < a[minIdx]) minIdx = j;
            if (minIdx != i) (a[i], a[minIdx]) = (a[minIdx], a[i]);
        }
    }

    public static void Main()
    {
        var a = new[] { 5, 3, 8, 1, 9, 2 };
        SelectionSort(a);
        Check(string.Join(",", a) == "1,2,3,5,8,9", "basic");
        var b = new[] { 2, 1 };
        SelectionSort(b);
        Check(string.Join(",", b) == "1,2", "two");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
