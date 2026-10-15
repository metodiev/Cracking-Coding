// Task 292 - Minimum subset sum difference (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns (difference, side): minimum |sum(A) - sum(B)| and one achieving subset.</summary>
    public static (int Difference, int[] Side) MinDifference(int[] nums)
    {
        int total = 0;
        foreach (int value in nums) total += value;
        int half = total / 2;
        int n = nums.Length;
        var reachable = new bool[n + 1, half + 1];
        reachable[0, 0] = true;
        for (int i = 1; i <= n; i++)
        {
            int value = nums[i - 1];
            for (int s = 0; s <= half; s++)
            {
                reachable[i, s] = reachable[i - 1, s] || (s >= value && reachable[i - 1, s - value]);
            }
        }
        int best = half;
        while (best > 0 && !reachable[n, best]) best--;
        var side = new List<int>();
        int rest = best;
        for (int i = n; i >= 1; i--)
        {
            int value = nums[i - 1];
            if (rest >= value && reachable[i - 1, rest - value])
            {
                side.Add(value);
                rest -= value;
            }
        }
        side.Reverse();
        return (total - 2 * best, side.ToArray());
    }

    public static int BruteMinDifference(int[] nums)
    {
        int total = 0;
        foreach (int value in nums) total += value;
        int n = nums.Length;
        int best = total;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int s = 0;
            for (int i = 0; i < n; i++) if (((mask >> i) & 1) != 0) s += nums[i];
            best = Math.Min(best, Math.Abs(total - 2 * s));
        }
        return best;
    }

    private static int seed = 777;

    private static int[] LcgNumbers(int n)
    {
        var nums = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            nums[i] = 1 + seed % 25;
        }
        return nums;
    }

    public static void Main()
    {
        var (diff, side) = MinDifference(new[] { 1, 6, 11, 5 });
        Check(diff == 1 && Sum(side) == 11 && side.Length == 2, "1,6,11,5");

        (diff, side) = MinDifference(new[] { 3, 3, 3 });
        Check(diff == 3 && side.Length == 1 && side[0] == 3, "3,3,3");

        (diff, side) = MinDifference(new[] { 1, 2, 3, 4 });
        Check(diff == 0 && Sum(side) == 5, "1,2,3,4");

        (diff, side) = MinDifference(new[] { 7 });
        Check(diff == 7 && side.Length == 0, "single element");

        (diff, side) = MinDifference(Array.Empty<int>());
        Check(diff == 0 && side.Length == 0, "empty");

        for (int trial = 0; trial < 200; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 14;
            var nums = LcgNumbers(count);
            var (answer, chosen) = MinDifference(nums);
            int total = 0;
            foreach (int value in nums) total += value;
            Check(answer == BruteMinDifference(nums), "random trial " + trial);
            Check(total - 2 * Sum(chosen) == answer, "random reconstruction " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static int Sum(int[] values)
    {
        int total = 0;
        foreach (int value in values) total += value;
        return total;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
