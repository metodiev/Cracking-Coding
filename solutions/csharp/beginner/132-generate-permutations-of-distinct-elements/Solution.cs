// Task 132 - Generate permutations of distinct elements. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<List<int>> Permutations(int[] items)
    {
        var outList = new List<List<int>>();
        Backtrack(items, new bool[items.Length], new List<int>(), outList);
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
            used[i] = true;
            current.Add(items[i]);
            Backtrack(items, used, current, outList);
            current.RemoveAt(current.Count - 1);
            used[i] = false;
        }
    }

    public static void Main()
    {
        var perms = Permutations(new[] { 1, 2, 3 });
        Check(perms.Count == 6, "3! = 6");
        Check(string.Join(",", perms[0]) == "1,2,3" && string.Join(",", perms[5]) == "3,2,1", "lexicographic order");
        Check(Permutations(new int[0]).Count == 1, "empty");
        Check(Permutations(new[] { 1, 2, 3, 4 }).Count == 24, "4! = 24");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
