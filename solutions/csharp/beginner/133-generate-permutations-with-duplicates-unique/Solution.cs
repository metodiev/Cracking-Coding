// Task 133 - Generate permutations with duplicates (unique). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<List<int>> UniquePermutations(int[] items)
    {
        var sorted = (int[])items.Clone();
        Array.Sort(sorted);
        var outList = new List<List<int>>();
        Backtrack(sorted, new bool[sorted.Length], new List<int>(), outList);
        return outList;
    }

    private static void Backtrack(int[] items, bool[] used, List<int> current, List<List<int>> outList)
    {
        if (current.Count == items.Length)
        {
            outList.Add(new List<int>(current));
            return;
        }
        for (int i = 0; i < items.Length; i++)
        {
            if (used[i]) continue;
            if (i > 0 && items[i] == items[i - 1] && !used[i - 1]) continue;
            used[i] = true;
            current.Add(items[i]);
            Backtrack(items, used, current, outList);
            current.RemoveAt(current.Count - 1);
            used[i] = false;
        }
    }

    public static void Main()
    {
        var result = UniquePermutations(new[] { 1, 1, 2 });
        Check(result.Count == 3, "three unique perms");
        Check(string.Join(",", result[0]) == "1,1,2", "lexicographic order");
        Check(UniquePermutations(new[] { 1, 1, 1 }).Count == 1, "all same");
        Check(UniquePermutations(new[] { 1, 2, 2, 3 }).Count == 12, "4!/2! = 12");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
