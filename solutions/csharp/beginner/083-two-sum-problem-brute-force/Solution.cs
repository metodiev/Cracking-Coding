// Task 083 - Two-sum problem (brute force). (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] TwoSumBrute(int[] nums, int target)
    {
        for (int i = 0; i < nums.Length; i++)
        {
            for (int j = i + 1; j < nums.Length; j++)
            {
                if (nums[i] + nums[j] == target) return new[] { i, j };
            }
        }
        return new[] { -1, -1 };
    }

    public static void Main()
    {
        Check(SeqEqual(TwoSumBrute(new[] { 2, 7, 11, 15 }, 9), new[] { 0, 1 }), "classic");
        Check(SeqEqual(TwoSumBrute(new[] { 3, 3 }, 6), new[] { 0, 1 }), "equal values");
        Check(SeqEqual(TwoSumBrute(new[] { 3, 2, 4 }, 6), new[] { 1, 2 }), "later pair");
        Check(SeqEqual(TwoSumBrute(new[] { -3, 4, 3, 90 }, 0), new[] { 0, 2 }), "negatives");
        Check(SeqEqual(TwoSumBrute(new[] { 1, 5, 3, 4 }, 9), new[] { 1, 3 }), "first pair with smallest i");
        Check(SeqEqual(TwoSumBrute(new[] { 2, 2, 2, 2 }, 4), new[] { 0, 1 }), "duplicates");
        Check(SeqEqual(TwoSumBrute(new[] { 1, 2, 3 }, 7), new[] { -1, -1 }), "none");
        Check(SeqEqual(TwoSumBrute(new int[0], 0), new[] { -1, -1 }), "empty");
        Check(SeqEqual(TwoSumBrute(new[] { 5 }, 5), new[] { -1, -1 }), "single");
        Console.WriteLine("PASS");
    }

    private static bool SeqEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
