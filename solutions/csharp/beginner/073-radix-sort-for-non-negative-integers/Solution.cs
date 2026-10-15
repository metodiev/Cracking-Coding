// Task 073 - Radix sort for non-negative integers. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int[] RadixSort(int[] a)
    {
        if (a.Length == 0) return new int[0];
        int max = 0;
        foreach (var x in a) max = Math.Max(max, x);
        var current = (int[])a.Clone();
        for (long exp = 1; max / exp > 0; exp *= 10)
        {
            var buckets = new int[10];
            foreach (var x in current) buckets[(int)(x / exp % 10)]++;
            for (int i = 1; i < 10; i++) buckets[i] += buckets[i - 1];
            var next = new int[current.Length];
            for (int i = current.Length - 1; i >= 0; i--)
            {
                int digit = (int)(current[i] / exp % 10);
                next[--buckets[digit]] = current[i];
            }
            current = next;
        }
        return current;
    }

    public static void Main()
    {
        Check(string.Join(",", RadixSort(new[] { 170, 45, 75, 90, 802, 24, 2, 66 })) == "2,24,45,66,75,90,170,802", "classic");
        Check(RadixSort(new int[0]).Length == 0, "empty");
        Check(string.Join(",", RadixSort(new[] { 0, 10, 100 })) == "0,10,100", "powers of ten");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
