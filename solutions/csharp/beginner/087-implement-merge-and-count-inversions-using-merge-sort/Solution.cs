// Task 087 - Implement merge and count inversions using merge sort. (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static long CountInversions(int[] nums)
    {
        int[] a = (int[])nums.Clone();
        int[] buffer = new int[a.Length];
        return MergeSortCount(a, buffer, 0, a.Length);
    }

    private static long MergeSortCount(int[] a, int[] buffer, int lo, int hi)
    {
        if (hi - lo <= 1) return 0;
        int mid = lo + (hi - lo) / 2;
        long count = MergeSortCount(a, buffer, lo, mid) + MergeSortCount(a, buffer, mid, hi);
        Array.Copy(a, lo, buffer, lo, hi - lo);
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi)
        {
            if (buffer[i] <= buffer[j])
            {
                a[k++] = buffer[i++];
            }
            else
            {
                a[k++] = buffer[j++];
                count += mid - i;
            }
        }
        while (i < mid) a[k++] = buffer[i++];
        while (j < hi) a[k++] = buffer[j++];
        return count;
    }

    private static long CountInversionsNaive(int[] nums)
    {
        long count = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            for (int j = i + 1; j < nums.Length; j++)
            {
                if (nums[i] > nums[j]) count++;
            }
        }
        return count;
    }

    private static int[] PseudoRandomList(int size)
    {
        int[] values = new int[size];
        long state = 7;
        for (int i = 0; i < size; i++)
        {
            state = state * 48271 % 2147483647L;
            values[i] = (int)(state % 1000);
        }
        return values;
    }

    public static void Main()
    {
        Check(CountInversions(new int[0]) == 0, "empty");
        Check(CountInversions(new[] { 1 }) == 0, "single");
        Check(CountInversions(new[] { 1, 2, 3, 4 }) == 0, "sorted");
        Check(CountInversions(new[] { 3, 2, 1 }) == 3, "reverse 3");
        Check(CountInversions(new[] { 5, 4, 3, 2, 1 }) == 10, "reverse 5");
        Check(CountInversions(new[] { 2, 1, 2, 1 }) == 3, "duplicates");
        Check(CountInversions(new[] { 2, 4, 1, 3, 5 }) == 3, "mixed");
        Check(CountInversions(new[] { 1, 1, 1 }) == 0, "all equal");

        int[] reversed = new int[300];
        for (int i = 0; i < 300; i++) reversed[i] = 300 - i;
        Check(CountInversions(reversed) == 44850L, "reverse 300");

        foreach (int size in new[] { 2, 7, 50, 200 })
        {
            int[] nums = PseudoRandomList(size);
            Check(CountInversions(nums) == CountInversionsNaive(nums), "agrees with naive, n=" + size);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
