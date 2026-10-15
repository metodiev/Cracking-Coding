// Task 091 - Max over sliding window naive. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] MaxSlidingWindowNaive(int[] nums, int k)
    {
        int n = nums.Length;
        if (k <= 0 || k > n) return new int[0];
        var result = new int[n - k + 1];
        for (int start = 0; start + k <= n; start++)
        {
            int best = nums[start];
            for (int i = start + 1; i < start + k; i++)
            {
                if (nums[i] > best) best = nums[i];
            }
            result[start] = best;
        }
        return result;
    }

    public static void Main()
    {
        var nums = new[] { 1, 3, -1, -3, 5, 3, 6, 7 };
        Check(Same(MaxSlidingWindowNaive(nums, 3), new[] { 3, 3, 5, 5, 6, 7 }), "k=3");
        Check(Same(MaxSlidingWindowNaive(nums, 1), nums), "k=1");
        Check(Same(MaxSlidingWindowNaive(nums, 8), new[] { 7 }), "k=n");
        Check(Same(MaxSlidingWindowNaive(new[] { 9, 9, 9 }, 2), new[] { 9, 9 }), "duplicates");
        Check(Same(MaxSlidingWindowNaive(new[] { -5, -2, -9 }, 2), new[] { -2, -2 }), "negatives");
        Check(Same(MaxSlidingWindowNaive(new[] { 5 }, 1), new[] { 5 }), "single");
        Check(Same(MaxSlidingWindowNaive(new[] { 1, 2, 3 }, 0), new int[0]), "k=0");
        Check(Same(MaxSlidingWindowNaive(new[] { 1, 2, 3 }, 4), new int[0]), "k>n");
        Check(Same(MaxSlidingWindowNaive(new int[0], 1), new int[0]), "empty");
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
