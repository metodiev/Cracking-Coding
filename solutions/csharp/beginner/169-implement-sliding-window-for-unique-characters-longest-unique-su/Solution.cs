// Task 169 - Sliding window for unique characters (longest unique substring). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns (length, substring).</summary>
    public static (int Length, string Substring) LongestUniqueSubstring(string s)
    {
        var last = new Dictionary<char, int>();
        int bestLen = 0, bestStart = 0, lo = 0;
        for (int hi = 0; hi < s.Length; hi++)
        {
            char c = s[hi];
            if (last.TryGetValue(c, out int seen) && seen >= lo) lo = seen + 1;
            last[c] = hi;
            if (hi - lo + 1 > bestLen)
            {
                bestLen = hi - lo + 1;
                bestStart = lo;
            }
        }
        return (bestLen, s.Substring(bestStart, bestLen));
    }

    public static void Main()
    {
        Check(LongestUniqueSubstring("abcabcbb") == (3, "abc"), "abcabcbb");
        Check(LongestUniqueSubstring("bbbbb") == (1, "b"), "bbbbb");
        Check(LongestUniqueSubstring("pwwkew") == (3, "wke"), "pwwkew");
        Check(LongestUniqueSubstring("") == (0, ""), "empty");
        Check(LongestUniqueSubstring("dvdf") == (3, "vdf"), "dvdf");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
