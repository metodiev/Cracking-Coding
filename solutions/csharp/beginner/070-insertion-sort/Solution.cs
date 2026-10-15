// Task 070 - Insertion sort. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static void InsertionSort(int[] a)
    {
        for (int i = 1; i < a.Length; i++)
        {
            int key = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > key)
            {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    }

    public static void Main()
    {
        var a = new[] { 5, 3, 8, 1, 9, 2 };
        InsertionSort(a);
        Check(string.Join(",", a) == "1,2,3,5,8,9", "basic");
        var b = new[] { 1, 2, 3 };
        InsertionSort(b);
        Check(string.Join(",", b) == "1,2,3", "already sorted");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
