// Task 202 - 3-sum all unique triplets. (Tier: Intermediate, CTCI Ch 16)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static List<List<int>> ThreeSum(int[] nums)
    {
        var a = (int[])nums.Clone();
        Array.Sort(a);
        var outList = new List<List<int>>();
        for (int i = 0; i + 2 < a.Length; i++)
        {
            if (i > 0 && a[i] == a[i - 1]) continue;
            if (a[i] > 0) break;
            int lo = i + 1, hi = a.Length - 1;
            while (lo < hi)
            {
                int total = a[i] + a[lo] + a[hi];
                if (total == 0)
                {
                    outList.Add(new List<int> { a[i], a[lo], a[hi] });
                    while (lo < hi && a[lo] == a[lo + 1]) lo++;
                    while (lo < hi && a[hi] == a[hi - 1]) hi--;
                    lo++;
                    hi--;
                }
                else if (total < 0) lo++;
                else hi--;
            }
        }
        return outList;
    }

    private static string Render(List<List<int>> triplets) =>
        "[" + string.Join(", ", triplets.Select(t => "[" + string.Join(", ", t) + "]")) + "]";

    public static void Main()
    {
        Check(Render(ThreeSum(new[] { -1, 0, 1, 2, -1, -4 })) == "[[-1, -1, 2], [-1, 0, 1]]", "classic");
        Check(ThreeSum(new[] { 0, 1, 1 }).Count == 0, "no triplet");
        Check(Render(ThreeSum(new[] { 0, 0, 0 })) == "[[0, 0, 0]]", "triple zeros");
        Check(Render(ThreeSum(new[] { -2, 0, 0, 2, 2 })) == "[[-2, 0, 2]]", "dedup");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
