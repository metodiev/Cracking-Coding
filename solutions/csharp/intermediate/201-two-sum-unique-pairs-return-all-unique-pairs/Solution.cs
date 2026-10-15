// Task 201 - Two-sum unique pairs (return all unique pairs). (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static List<(int, int)> TwoSumUniquePairs(int[] nums, int target)
    {
        var seen = new HashSet<int>();
        var pairs = new SortedSet<(int, int)>();
        foreach (var x in nums)
        {
            int y = target - x;
            if (seen.Contains(y)) pairs.Add((Math.Min(x, y), Math.Max(x, y)));
            seen.Add(x);
        }
        return pairs.ToList();
    }

    public static void Main()
    {
        Check(Render(TwoSumUniquePairs(new[] { 1, 1, 2, 3, 2 }, 4)) == "(1, 3),(2, 2)", "dup-friendly");
        Check(Render(TwoSumUniquePairs(new[] { 1, 5, 3, 3, 5 }, 6)) == "(1, 5),(3, 3)", "second case");
        Check(TwoSumUniquePairs(new[] { 1, 2, 3 }, 100).Count == 0, "no pairs");
        Check(Render(TwoSumUniquePairs(new[] { 0, 0, 0 }, 0)) == "(0, 0)", "zeros");
        Check(Render(TwoSumUniquePairs(new[] { -1, 1, 0 }, 0)) == "(-1, 1)", "negatives");
        Console.WriteLine("PASS");
    }

    private static string Render(List<(int, int)> pairs) =>
        string.Join(",", pairs.Select(p => $"({p.Item1}, {p.Item2})"));

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
