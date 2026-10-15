// Task 026 - Check if array contains any duplicates (boolean). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    public static bool ContainsDuplicates(int[] items)
    {
        var seen = new HashSet<int>();
        foreach (var x in items) if (!seen.Add(x)) return true;
        return false;
    }

    public static void Main()
    {
        Check(ContainsDuplicates(new[] { 1, 2, 3, 1 }), "has dup");
        Check(!ContainsDuplicates(new[] { 1, 2, 3 }), "unique");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
