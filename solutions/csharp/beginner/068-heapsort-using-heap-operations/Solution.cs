// Task 068 - Heapsort using heap operations. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static void HeapSort(int[] a)
    {
        int n = a.Length;
        for (int i = n / 2 - 1; i >= 0; i--) SiftDown(a, i, n);
        for (int end = n - 1; end > 0; end--)
        {
            (a[0], a[end]) = (a[end], a[0]);
            SiftDown(a, 0, end);
        }
    }

    private static void SiftDown(int[] a, int i, int size)
    {
        while (true)
        {
            int largest = i, left = 2 * i + 1, right = 2 * i + 2;
            if (left < size && a[left] > a[largest]) largest = left;
            if (right < size && a[right] > a[largest]) largest = right;
            if (largest == i) return;
            (a[i], a[largest]) = (a[largest], a[i]);
            i = largest;
        }
    }

    public static void Main()
    {
        var a = new[] { 5, 3, 8, 1, 9, 2 };
        HeapSort(a);
        Check(string.Join(",", a) == "1,2,3,5,8,9", "basic");
        var b = new[] { 3, 3, 1 };
        HeapSort(b);
        Check(string.Join(",", b) == "1,3,3", "duplicates");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
