// Task 208 - Implement sliding window minimum using deque. (Tier: Intermediate, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<int> SlidingWindowMinimum(int[] nums, int k)
    {
        if (k <= 0) throw new ArgumentException("k must be positive");
        var result = new List<int>();
        var dq = new LinkedList<int>();              // indices, values increasing
        for (int i = 0; i < nums.Length; i++)
        {
            while (dq.Count > 0 && nums[dq.Last!.Value] >= nums[i]) dq.RemoveLast();
            dq.AddLast(i);
            if (dq.First!.Value <= i - k) dq.RemoveFirst();
            if (i >= k - 1) result.Add(nums[dq.First!.Value]);
        }
        return result;
    }

    public static void Main()
    {
        Check(string.Join(",", SlidingWindowMinimum(new[] { 1, 3, -1, -3, 5, 3, 6, 7 }, 3)) == "-1,-3,-3,-3,3,3", "classic");
        Check(string.Join(",", SlidingWindowMinimum(new[] { 1 }, 1)) == "1", "single");
        Check(SlidingWindowMinimum(new[] { 1, 2 }, 5).Count == 0, "k > n");
        Check(string.Join(",", SlidingWindowMinimum(new[] { 5, 5, 5 }, 2)) == "5,5", "duplicates");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
