// Task 130 - Generate all subsets (power set) iterative. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<List<int>> SubsetsBitmask(int[] items)
    {
        int n = items.Length;
        var outList = new List<List<int>>();
        for (int mask = 0; mask < (1 << n); mask++)
        {
            var subset = new List<int>();
            for (int i = 0; i < n; i++)
                if ((mask & (1 << i)) != 0) subset.Add(items[i]);
            outList.Add(subset);
        }
        return outList;
    }

    public static List<List<int>> SubsetsIncremental(int[] items)
    {
        var outList = new List<List<int>> { new List<int>() };
        foreach (var x in items)
        {
            int size = outList.Count;
            for (int i = 0; i < size; i++)
            {
                var extended = new List<int>(outList[i]) { x };
                outList.Add(extended);
            }
        }
        return outList;
    }

    public static void Main()
    {
        var mask = SubsetsBitmask(new[] { 1, 2, 3 });
        Check(mask.Count == 8, "eight subsets");
        Check(mask[0].Count == 0, "empty first");
        Check(string.Join(",", mask[1]) == "1", "second is [1]");
        Check(string.Join(",", mask[7]) == "1,2,3", "full set last");
        Check(SubsetsBitmask(new int[0]).Count == 1, "empty input -> one subset");
        var inc = SubsetsIncremental(new[] { 1, 2 });
        Check(inc.Count == 4 && string.Join(",", inc[3]) == "1,2", "incremental");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
