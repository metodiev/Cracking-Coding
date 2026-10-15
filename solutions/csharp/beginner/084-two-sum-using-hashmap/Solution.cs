// Task 084 - Two-sum using hashmap. (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] TwoSumHash(int[] nums, int target)
    {
        var seen = new Dictionary<int, int>();
        for (int j = 0; j < nums.Length; j++)
        {
            int complement = target - nums[j];
            if (seen.ContainsKey(complement)) return new[] { seen[complement], j };
            if (!seen.ContainsKey(nums[j])) seen[nums[j]] = j;
        }
        return new[] { -1, -1 };
    }

    private static int[] TwoSumBrute(int[] nums, int target)
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
        Check(SeqEqual(TwoSumHash(new[] { 2, 7, 11, 15 }, 9), new[] { 0, 1 }), "classic");
        Check(SeqEqual(TwoSumHash(new[] { 3, 2, 4 }, 6), new[] { 1, 2 }), "later pair");
        Check(SeqEqual(TwoSumHash(new[] { 3, 3 }, 6), new[] { 0, 1 }), "equal values");
        Check(SeqEqual(TwoSumHash(new[] { -3, 4, 3, 90 }, 0), new[] { 0, 2 }), "negatives");
        Check(SeqEqual(TwoSumHash(new[] { 1, 2, 3 }, 7), new[] { -1, -1 }), "none");
        Check(SeqEqual(TwoSumHash(new int[0], 0), new[] { -1, -1 }), "empty");
        Check(SeqEqual(TwoSumHash(new[] { 5 }, 5), new[] { -1, -1 }), "single");

        int[][] cases = { new[] { 2, 7, 11, 15 }, new[] { 3, 2, 4 }, new[] { 3, 3 }, new[] { 0, 0, 1 },
                          new[] { -3, 4, 3, 90 }, new[] { 1, 2, 3 }, new int[0], new[] { 5 }, new[] { 1, 5, 3, 4 } };
        int[] targets = { 9, 6, 6, 0, 0, 7, 0, 5, 9 };
        for (int c = 0; c < cases.Length; c++)
        {
            int[] got = TwoSumHash(cases[c], targets[c]);
            bool exists = TwoSumBrute(cases[c], targets[c])[0] != -1;
            Check((got[0] != -1) == exists, "agrees with brute force on case " + c);
            if (got[0] != -1)
            {
                Check(got[0] >= 0 && got[0] < got[1] && got[1] < cases[c].Length, "valid index order, case " + c);
                Check(cases[c][got[0]] + cases[c][got[1]] == targets[c], "valid sum, case " + c);
            }
        }
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
