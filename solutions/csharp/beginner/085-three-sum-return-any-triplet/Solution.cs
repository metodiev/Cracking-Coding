// Task 085 - Three-sum (return any triplet). (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] ThreeSum(int[] nums)
    {
        if (nums.Length < 3) return new int[0];
        int[] values = (int[])nums.Clone();
        Array.Sort(values);
        int n = values.Length;
        for (int i = 0; i < n - 2; i++)
        {
            if (i > 0 && values[i] == values[i - 1]) continue;
            int lo = i + 1, hi = n - 1;
            while (lo < hi)
            {
                int total = values[i] + values[lo] + values[hi];
                if (total == 0) return new[] { values[i], values[lo], values[hi] };
                if (total < 0) lo++;
                else hi--;
            }
        }
        return new int[0];
    }

    private static bool IsSorted(int[] a)
    {
        for (int i = 1; i < a.Length; i++) if (a[i - 1] > a[i]) return false;
        return true;
    }

    private static bool ContainsTriplet(int[] nums, int[] triplet)
    {
        if (triplet.Length != 3) return false;
        bool[] used = new bool[nums.Length];
        foreach (int v in triplet)
        {
            bool found = false;
            for (int i = 0; i < nums.Length; i++)
            {
                if (!used[i] && nums[i] == v)
                {
                    used[i] = true;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    public static void Main()
    {
        int[] nums = { -1, 0, 1, 2, -1, -4 };
        int[] t = ThreeSum(nums);
        Check(t.Length == 3, "triplet found");
        Check(t[0] + t[1] + t[2] == 0, "sums to zero");
        Check(IsSorted(t), "sorted output");
        Check(ContainsTriplet(nums, t), "three distinct positions");

        Check(SeqEqual(ThreeSum(new[] { 0, 0, 0 }), new[] { 0, 0, 0 }), "0,0,0");
        Check(SeqEqual(ThreeSum(new[] { -3, 1, 2 }), new[] { -3, 1, 2 }), "exact triplet");
        Check(SeqEqual(ThreeSum(new[] { 1, -1, 0 }), new[] { -1, 0, 1 }), "unsorted input");
        Check(SeqEqual(ThreeSum(new[] { -2, 0, 0, 2, 2 }), new[] { -2, 0, 2 }), "duplicates");
        Check(ThreeSum(new[] { 1, 2, 3 }).Length == 0, "none");
        Check(ThreeSum(new[] { 1, 1, 1 }).Length == 0, "no zero sum");
        Check(ThreeSum(new[] { 0, 0 }).Length == 0, "too short");
        Check(ThreeSum(new int[0]).Length == 0, "empty");

        int[] nums2 = { 8, -4, -2, 3, 1, -5, 7 };
        int[] t2 = ThreeSum(nums2);
        Check(t2.Length == 3 && t2[0] + t2[1] + t2[2] == 0, "arbitrary order: sums to zero");
        Check(IsSorted(t2) && ContainsTriplet(nums2, t2), "arbitrary order: valid triplet");
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
