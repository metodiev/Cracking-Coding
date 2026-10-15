// Task 167 - Find all elements appearing > n/3 (generalized). (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns a sorted list of values occurring more than n/3 times.</summary>
    public static List<int> MajorityElementsN3(int[] nums)
    {
        int? c1 = null, c2 = null;
        int n1 = 0, n2 = 0;
        foreach (var x in nums)
        {
            if (c1 != null && x == c1) n1++;
            else if (c2 != null && x == c2) n2++;
            else if (n1 == 0) { c1 = x; n1 = 1; }
            else if (n2 == 0) { c2 = x; n2 = 1; }
            else { n1--; n2--; }
        }
        var outList = new List<int>();
        foreach (var cand in new HashSet<int>(new[] { c1, c2 }.Where(v => v != null).Select(v => v!.Value)))
        {
            int occurrences = 0;
            foreach (var x in nums) if (x == cand) occurrences++;
            if (occurrences * 3 > nums.Length) outList.Add(cand);
        }
        outList.Sort();
        return outList;
    }

    public static void Main()
    {
        Check(string.Join(",", MajorityElementsN3(new[] { 3, 2, 3 })) == "3", "3,2,3");
        Check(string.Join(",", MajorityElementsN3(new[] { 1, 1, 1, 3, 3, 2, 2, 2 })) == "1,2", "two values");
        Check(MajorityElementsN3(new[] { 1, 2, 3 }).Count == 0, "none");
        Check(MajorityElementsN3(new int[0]).Count == 0, "empty");
        Check(string.Join(",", MajorityElementsN3(new[] { 4 })) == "4", "single");
        Check(string.Join(",", MajorityElementsN3(new[] { 1, 1, 2, 2, 3 })) == "1,2", "2/5 = 0.4 > 1/3");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
