// Task 291 - Partition equal subset (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns one equal-sum side when nums splits in half, else null.</summary>
    public static int[]? Partition(int[] nums)
    {
        int total = 0;
        foreach (int value in nums) total += value;
        if (total % 2 != 0) return null;
        int target = total / 2;
        int n = nums.Length;
        var reachable = new bool[n + 1, target + 1];
        reachable[0, 0] = true;
        for (int i = 1; i <= n; i++)
        {
            int value = nums[i - 1];
            for (int s = 0; s <= target; s++)
            {
                reachable[i, s] = reachable[i - 1, s] || (s >= value && reachable[i - 1, s - value]);
            }
        }
        if (!reachable[n, target]) return null;
        var reversed = new List<int>();
        int rest = target;
        for (int i = n; i >= 1; i--)
        {
            int value = nums[i - 1];
            if (rest >= value && reachable[i - 1, rest - value])
            {
                reversed.Add(value);
                rest -= value;
            }
        }
        reversed.Reverse();
        return reversed.ToArray();
    }

    /// <summary>True when side is a sub-multiset of nums whose complement has the same sum.</summary>
    public static bool Validate(int[] nums, int[] side)
    {
        var used = new bool[nums.Length];
        int sideSum = 0;
        foreach (int value in side)
        {
            sideSum += value;
            bool found = false;
            for (int i = 0; i < nums.Length && !found; i++)
            {
                if (!used[i] && nums[i] == value)
                {
                    used[i] = true;
                    found = true;
                }
            }
            if (!found) return false;
        }
        int restSum = 0;
        for (int i = 0; i < nums.Length; i++) if (!used[i]) restSum += nums[i];
        return sideSum == restSum;
    }

    public static bool BrutePartitionExists(int[] nums)
    {
        int total = 0;
        foreach (int value in nums) total += value;
        if (total % 2 != 0) return false;
        int target = total / 2;
        int n = nums.Length;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int sum = 0;
            for (int i = 0; i < n; i++) if (((mask >> i) & 1) != 0) sum += nums[i];
            if (sum == target) return true;
        }
        return false;
    }

    private static int seed = 1234;

    private static int[] LcgNumbers(int n)
    {
        var nums = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            nums[i] = 1 + seed % 15;
        }
        return nums;
    }

    public static void Main()
    {
        var sample = new[] { 1, 5, 11, 5 };
        var side = Partition(sample);
        Check(side != null && Same(side, new[] { 1, 5, 5 }), "sample split");
        Check(side != null && Validate(sample, side), "sample validation");

        Check(Partition(new[] { 1, 2, 3, 5 }) == null, "odd total");
        Check(Partition(new[] { 1, 2 }) == null, "odd total small");
        Check(Partition(Array.Empty<int>()) is { Length: 0 }, "empty set");
        Check(Partition(new[] { 2, 2, 2 }) == null, "even total but unreachable half");

        var four = new[] { 1, 2, 3, 4 };
        side = Partition(four);
        int sideSum = 0;
        if (side != null) foreach (int value in side) sideSum += value;
        Check(sideSum == 5 && side != null && Validate(four, side), "1..4 split");

        for (int trial = 0; trial < 200; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = 2 + seed % 8;
            var nums = LcgNumbers(count);
            var randomSide = Partition(nums);
            Check((randomSide != null) == BrutePartitionExists(nums), "random trial " + trial);
            if (randomSide != null) Check(Validate(nums, randomSide), "random validation " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
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
