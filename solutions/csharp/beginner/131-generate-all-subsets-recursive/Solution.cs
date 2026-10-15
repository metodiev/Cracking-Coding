// Task 131 - Generate all subsets recursive. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<List<int>> SubsetsRecursive(int[] items)
    {
        var outList = new List<List<int>>();
        Backtrack(items, 0, new List<int>(), outList);
        return outList;
    }

    private static void Backtrack(int[] items, int i, List<int> current, List<List<int>> outList)
    {
        if (i == items.Length)
        {
            outList.Add(new List<int>(current));
            return;
        }
        Backtrack(items, i + 1, current, outList);      // skip
        current.Add(items[i]);
        Backtrack(items, i + 1, current, outList);      // take
        current.RemoveAt(current.Count - 1);
    }

    public static void Main()
    {
        var result = SubsetsRecursive(new[] { 1, 2, 3 });
        Check(result.Count == 8, "eight subsets");
        Check(result[0].Count == 0 && string.Join(",", result[1]) == "3", "skip-first order");
        Check(SubsetsRecursive(new int[0]).Count == 1, "empty input");
        Check(SubsetsRecursive(new[] { 1, 2, 3, 4 }).Count == 16, "2^4");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
