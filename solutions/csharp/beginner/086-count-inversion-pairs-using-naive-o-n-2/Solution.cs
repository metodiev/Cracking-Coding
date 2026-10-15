// Task 086 - Count inversion pairs using naive O(n^2). (Tier: Beginner, CTCI Ch 10)
using System;

public static class Solution
{
    public static int CountInversionsNaive(int[] nums)
    {
        int count = 0;
        for (int i = 0; i < nums.Length; i++)
        {
            for (int j = i + 1; j < nums.Length; j++)
            {
                if (nums[i] > nums[j]) count++;
            }
        }
        return count;
    }

    public static void Main()
    {
        Check(CountInversionsNaive(new int[0]) == 0, "empty");
        Check(CountInversionsNaive(new[] { 1 }) == 0, "single");
        Check(CountInversionsNaive(new[] { 1, 2, 3 }) == 0, "sorted");
        Check(CountInversionsNaive(new[] { 1, 1, 1 }) == 0, "all equal");
        Check(CountInversionsNaive(new[] { 2, 2 }) == 0, "pair of equals");
        Check(CountInversionsNaive(new[] { 1, 3, 2 }) == 1, "one inversion");
        Check(CountInversionsNaive(new[] { 3, 2, 1 }) == 3, "reverse 3");
        Check(CountInversionsNaive(new[] { 2, 4, 1, 3, 5 }) == 3, "mixed");
        Check(CountInversionsNaive(new[] { -1, -2, -3 }) == 3, "negatives");
        Check(CountInversionsNaive(new[] { 5, 4, 3, 2, 1 }) == 10, "reverse 5");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
