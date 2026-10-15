// Task 337 - Design majority vote generalized (> n/k) using count map. (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>
    /// Generalised Boyer-Moore: keep at most k - 1 candidate counters; a new value with no
    /// free slot cancels one vote from every counter. A verification pass then keeps exactly
    /// the values with count * k > n.
    /// </summary>
    public static int[] MajorityElements(int[] values, int k)
    {
        if (k < 2) throw new Exception("k must be >= 2");
        int n = values.Length;
        if (n == 0) return Array.Empty<int>();

        var slots = new Dictionary<int, int>();
        foreach (int value in values)
        {
            if (slots.TryGetValue(value, out int count))
            {
                slots[value] = count + 1;
            }
            else if (slots.Count < k - 1)
            {
                slots[value] = 1;
            }
            else
            {
                foreach (int candidate in new List<int>(slots.Keys))
                {
                    slots[candidate] = slots[candidate] - 1;
                    if (slots[candidate] == 0) slots.Remove(candidate);
                }
            }
        }

        var counts = new Dictionary<int, int>();
        foreach (int value in values)
        {
            counts[value] = (counts.TryGetValue(value, out int c) ? c : 0) + 1;
        }
        var found = new List<int>();
        foreach (int value in slots.Keys)
        {
            if (counts[value] * k > n) found.Add(value);
        }
        found.Sort();
        return found.ToArray();
    }

    public static int[] BruteMajority(int[] values, int k)
    {
        var counts = new Dictionary<int, int>();
        foreach (int value in values)
        {
            counts[value] = (counts.TryGetValue(value, out int c) ? c : 0) + 1;
        }
        var found = new List<int>();
        foreach (KeyValuePair<int, int> entry in counts)
        {
            if (entry.Value * k > values.Length) found.Add(entry.Key);
        }
        found.Sort();
        return found.ToArray();
    }

    private static int seed = 6060;

    private static string Join(int[] values)
    {
        return string.Join(",", values);
    }

    public static void Main()
    {
        // Hand-checked with n = 6, k = 4: the bar is count * 4 > 6, so 1 (3 times) and
        // 2 (2 times) qualify while 3 (once) does not.
        Check(Join(MajorityElements(new[] { 1, 1, 1, 2, 2, 3 }, 4)) == "1,2", "hand n=6 k=4");
        Check(Join(BruteMajority(new[] { 1, 1, 1, 2, 2, 3 }, 4)) == "1,2", "hand brute");

        Check(Join(MajorityElements(new[] { 3, 2, 3 }, 2)) == "3", "classic majority");
        Check(Join(MajorityElements(new[] { 2, 2, 1, 1, 1, 2, 2 }, 2)) == "2", "classic seven");

        Check(MajorityElements(new[] { 1, 2, 3, 4 }, 4).Length == 0, "no majority k=4");
        Check(MajorityElements(new[] { 1, 2, 3, 4 }, 2).Length == 0, "no majority k=2");
        Check(MajorityElements(new[] { 1, 2, 2, 3, 3, 4 }, 3).Length == 0, "no majority n=6 k=3");
        Check(MajorityElements(Array.Empty<int>(), 3).Length == 0, "empty");

        Check(MajorityElements(new[] { 5, 5, 6, 6 }, 2).Length == 0, "boundary count*k == n");
        Check(Join(MajorityElements(new[] { 5, 5, 6, 6 }, 3)) == "5,6", "boundary with k=3");
        Check(Join(MajorityElements(new[] { 7, 8, 9, 7, 8, 9 }, 4)) == "7,8,9", "all pass");

        for (int trial = 0; trial < 400; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = seed % 60;
            seed = (seed * 75 + 74) % 65537;
            int k = 2 + seed % 5;
            seed = (seed * 75 + 74) % 65537;
            int alphabet = 1 + seed % 9;
            var values = new int[n];
            for (int i = 0; i < n; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                values[i] = seed % alphabet;
            }
            int[] found = MajorityElements(values, k);
            Check(Join(found) == Join(BruteMajority(values, k)), "random set " + trial);
            Check(found.Length <= k - 1, "bounded candidates " + trial);
            var counts = new Dictionary<int, int>();
            foreach (int value in values)
            {
                counts[value] = (counts.TryGetValue(value, out int c) ? c : 0) + 1;
            }
            foreach (int value in found)
            {
                Check(counts[value] * k > n, "every report exceeds n/k " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
