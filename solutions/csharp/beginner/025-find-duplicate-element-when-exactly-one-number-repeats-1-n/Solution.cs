// Task 025 - Find duplicate element when exactly one number repeats (1..n). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static int FindDuplicate(int[] nums)
    {
        int slow = nums[0], fast = nums[0];
        do
        {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = nums[0];
        while (slow != fast) { slow = nums[slow]; fast = nums[fast]; }
        return slow;
    }

    public static void Main()
    {
        Check(FindDuplicate(new[] { 1, 3, 4, 2, 2 }) == 2, "n=4");
        Check(FindDuplicate(new[] { 3, 1, 3, 4, 2 }) == 3, "value 3");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
