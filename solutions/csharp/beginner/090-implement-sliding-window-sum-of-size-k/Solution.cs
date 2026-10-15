// Task 090 - Implement sliding window sum of size k. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] WindowSums(int[] nums, int k)
    {
        if (k < 0) throw new ArgumentException("k must be >= 0");
        int n = nums.Length;
        if (k == 0 || k > n) return new int[0];
        int[] result = new int[n - k + 1];
        int window = 0;
        for (int i = 0; i < k; i++) window += nums[i];
        result[0] = window;
        for (int i = k; i < n; i++)
        {
            window += nums[i] - nums[i - k];
            result[i - k + 1] = window;
        }
        return result;
    }

    public static void Main()
    {
        Check(SeqEqual(WindowSums(new[] { 1, 2, 3, 4, 5 }, 3), new[] { 6, 9, 12 }), "k=3");
        Check(SeqEqual(WindowSums(new[] { 1, 2, 3, 4, 5 }, 1), new[] { 1, 2, 3, 4, 5 }), "k=1");
        Check(SeqEqual(WindowSums(new[] { 1, 2, 3, 4, 5 }, 5), new[] { 15 }), "k=n");
        Check(WindowSums(new[] { 1, 2, 3, 4, 5 }, 6).Length == 0, "k > n");
        Check(WindowSums(new[] { 1, 2, 3, 4, 5 }, 0).Length == 0, "k == 0");
        Check(WindowSums(new int[0], 1).Length == 0, "empty, k=1");
        Check(WindowSums(new int[0], 0).Length == 0, "empty, k=0");
        Check(SeqEqual(WindowSums(new[] { 4 }, 1), new[] { 4 }), "single");
        Check(SeqEqual(WindowSums(new[] { -1, -2, -3, -4 }, 2), new[] { -3, -5, -7 }), "negatives");
        Check(SeqEqual(WindowSums(new[] { 2, 2, 2, 2 }, 2), new[] { 4, 4, 4 }), "duplicates");
        CheckThrows(() => WindowSums(new[] { 1, 2 }, -1), "k < 0");

        int n = 20000, k = 100;
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) nums[i] = i;
        int[] sums = WindowSums(nums, k);
        Check(sums.Length == n - k + 1, "large: window count");
        Check(sums[0] == k * (k - 1) / 2, "large: first window");
        Check(sums[sums.Length - 1] == (n - k) * k + k * (k - 1) / 2, "large: last window");
        Console.WriteLine("PASS");
    }

    private static bool SeqEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected ArgumentException: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
