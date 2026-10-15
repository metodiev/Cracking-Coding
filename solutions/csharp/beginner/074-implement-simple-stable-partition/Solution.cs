// Task 074 - Implement simple stable partition. (Tier: Beginner, CTCI Ch 10)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] StablePartition(int[] items, Func<int, bool> predicate)
    {
        var outList = new List<int>();
        foreach (var x in items) if (predicate(x)) outList.Add(x);
        foreach (var x in items) if (!predicate(x)) outList.Add(x);
        return outList.ToArray();
    }

    public static void Main()
    {
        Check(string.Join(",", StablePartition(new[] { 1, 2, 3, 4, 5, 6 }, x => x % 2 == 0)) == "2,4,6,1,3,5", "evens first");
        Check(StablePartition(new int[0], _ => true).Length == 0, "empty");
        Check(string.Join(",", StablePartition(new[] { 4, 1, 4, 2 }, x => x == 4)) == "4,4,1,2", "stability");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
