// Task 203 - 4-sum problem (k-sum generalization). (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static List<List<int>> KSum(int[] nums, int target, int k)
    {
        var a = (int[])nums.Clone();
        Array.Sort(a);
        var results = new List<List<int>>();
        Search(a, 0, target, k, new List<int>(), results);
        return results;
    }

    private static void Search(int[] a, int start, int remaining, int kLeft,
                               List<int> current, List<List<int>> results)
    {
        if (kLeft == 2)
        {
            int lo = start, hi = a.Length - 1;
            while (lo < hi)
            {
                int total = a[lo] + a[hi];
                if (total == remaining)
                {
                    var found = new List<int>(current) { a[lo], a[hi] };
                    results.Add(found);
                    while (lo < hi && a[lo] == a[lo + 1]) lo++;
                    while (lo < hi && a[hi] == a[hi - 1]) hi--;
                    lo++;
                    hi--;
                }
                else if (total < remaining) lo++;
                else hi--;
            }
            return;
        }
        for (int i = start; i + kLeft <= a.Length; i++)
        {
            if (i > start && a[i] == a[i - 1]) continue;
            if ((long)a[i] * kLeft > remaining) break;
            current.Add(a[i]);
            Search(a, i + 1, remaining - a[i], kLeft - 1, current, results);
            current.RemoveAt(current.Count - 1);
        }
    }

    private static string Render(List<List<int>> sets) =>
        "[" + string.Join(", ", sets.Select(s => "[" + string.Join(", ", s) + "]")) + "]";

    public static void Main()
    {
        Check(Render(KSum(new[] { 1, 0, -1, 0, -2, 2 }, 0, 4)) == "[[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]", "classic");
        Check(Render(KSum(new[] { 2, 2, 2, 2, 2 }, 8, 4)) == "[[2, 2, 2, 2]]", "all twos");
        Check(KSum(new[] { 1, 2, 3, 4 }, 100, 4).Count == 0, "no match");
        Check(Render(KSum(new[] { 1, 2, 3, 4 }, 5, 2)) == "[[1, 4], [2, 3]]", "k=2");
        Check(Render(KSum(new[] { -1, 0, 1, 2, -1, -4 }, 0, 3)) == "[[-1, -1, 2], [-1, 0, 1]]", "k=3");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
