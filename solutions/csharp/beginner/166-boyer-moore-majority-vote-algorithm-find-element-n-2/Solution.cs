// Task 166 - Boyer-Moore majority vote algorithm (element > n/2). (Tier: Beginner, CTCI Ch 6)
using System;

public static class Solution
{
    /// <summary>Returns the element occurring more than n/2 times, or null.</summary>
    public static int? MajorityElement(int[] nums)
    {
        int? candidate = null;
        int count = 0;
        foreach (var x in nums)
        {
            if (count == 0)
            {
                candidate = x;
                count = 1;
            }
            else if (x == candidate) count++;
            else count--;
        }
        if (candidate == null) return null;
        int occurrences = 0;
        foreach (var x in nums) if (x == candidate) occurrences++;
        return occurrences > nums.Length / 2 ? candidate : null;
    }

    public static void Main()
    {
        Check(MajorityElement(new[] { 3, 2, 3 }) == 3, "3,2,3");
        Check(MajorityElement(new[] { 2, 2, 1, 1, 1, 2, 2 }) == 2, "classic");
        Check(MajorityElement(new[] { 1 }) == 1, "single");
        Check(MajorityElement(new int[0]) == null, "empty");
        Check(MajorityElement(new[] { 1, 2, 3 }) == null, "no majority");
        Check(MajorityElement(new[] { 1, 1, 2, 2 }) == null, "exactly half is not majority");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
