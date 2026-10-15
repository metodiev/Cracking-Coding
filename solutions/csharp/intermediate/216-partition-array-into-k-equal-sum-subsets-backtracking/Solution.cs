// Task 216 - Partition array into k equal-sum subsets (backtracking). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool CanPartitionK(int[] nums, int k)
    {
        if (k <= 0) throw new ArgumentException("k must be positive");
        int total = 0;
        foreach (var x in nums) total += x;
        if (total % k != 0) return false;
        int target = total / k;
        var a = (int[])nums.Clone();
        Array.Sort(a);
        Array.Reverse(a);
        if (a.Length > 0 && a[0] > target) return false;
        return Place(a, 0, new int[k], target);
    }

    private static bool Place(int[] nums, int i, int[] buckets, int target)
    {
        if (i == nums.Length) return true;
        var seen = new HashSet<int>();
        for (int b = 0; b < buckets.Length; b++)
        {
            if (seen.Contains(buckets[b])) continue;
            if (buckets[b] + nums[i] <= target)
            {
                seen.Add(buckets[b]);
                buckets[b] += nums[i];
                if (Place(nums, i + 1, buckets, target)) return true;
                buckets[b] -= nums[i];
            }
            if (buckets[b] == 0) break;
        }
        return false;
    }

    public static void Main()
    {
        Check(CanPartitionK(new[] { 4, 3, 2, 3, 5, 2, 1 }, 4), "classic");
        Check(CanPartitionK(new[] { 1, 2, 3, 4 }, 2), "two halves");
        Check(CanPartitionK(new[] { 1, 1, 1, 1 }, 2), "four ones");
        Check(!CanPartitionK(new[] { 2, 2, 2, 2, 3 }, 3), "not divisible");
        Check(!CanPartitionK(new[] { 2, 2, 2, 2, 2, 2 }, 4), "twos cannot make 3s");
        Check(CanPartitionK(new[] { 10, 10, 10, 10, 10, 10 }, 3), "three tens");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
