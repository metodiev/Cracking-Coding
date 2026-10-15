// Task 004 - Remove duplicates from an array (create new array). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] RemoveDuplicates(int[] items)
    {
        var seen = new HashSet<int>();
        var outList = new List<int>();
        foreach (var x in items)
        {
            if (seen.Add(x)) outList.Add(x);
        }
        return outList.ToArray();
    }

    public static void Main()
    {
        Check(string.Join(",", RemoveDuplicates(new[] { 1, 2, 2, 3, 1 })) == "1,2,3", "basic");
        Check(RemoveDuplicates(new int[0]).Length == 0, "empty");
        Check(RemoveDuplicates(new[] { 5, 5, 5 }).Length == 1, "all same");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
