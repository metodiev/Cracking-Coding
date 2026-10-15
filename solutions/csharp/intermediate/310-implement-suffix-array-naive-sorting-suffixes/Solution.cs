// Task 310 - Suffix array by naively sorting suffixes. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static int[] SuffixArray(string s)
    {
        var indices = Enumerable.Range(0, s.Length).ToArray();
        Array.Sort(indices, (a, b) => string.CompareOrdinal(s.Substring(a), s.Substring(b)));
        return indices;
    }

    /// <summary>Manual lexicographic comparator; the independent oracle.</summary>
    public static int CompareSuffixes(string s, int i, int j)
    {
        int n = s.Length;
        while (i < n && j < n)
        {
            if (s[i] != s[j]) return s[i] < s[j] ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;                 // the shorter suffix is smaller
    }

    public static bool IsValidSuffixArray(string s, int[] order)
    {
        var sorted = (int[])order.Clone();
        Array.Sort(sorted);
        for (int i = 0; i < sorted.Length; i++) if (sorted[i] != i) return false;
        for (int k = 0; k + 1 < order.Length; k++) if (CompareSuffixes(s, order[k], order[k + 1]) >= 0) return false;
        return true;
    }

    private static int seed = 2718;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "ab"[seed % 2];
        }
        return new string(chars);
    }

    private static string Join(int[] values) => string.Join(",", values);

    public static void Main()
    {
        Check(Join(SuffixArray("banana")) == "5,3,1,0,4,2", "banana");
        var mississippi = SuffixArray("mississippi");
        Check(Join(mississippi) == "10,7,4,1,0,9,8,6,3,5,2", "mississippi");
        Check(IsValidSuffixArray("mississippi", mississippi), "mississippi valid");

        Check(SuffixArray("").Length == 0, "empty");
        Check(Join(SuffixArray("a")) == "0", "single");
        Check(Join(SuffixArray("aaaa")) == "3,2,1,0", "aaaa");
        Check(Join(SuffixArray("abab")) == "2,0,3,1", "abab");

        for (int i = 0; i < 60; i++)
        {
            string s = LcgString(12);
            Check(IsValidSuffixArray(s, SuffixArray(s)), "random " + s);
        }
        Check(IsValidSuffixArray("banana", SuffixArray("banana")), "banana valid");
        Check(IsValidSuffixArray("aaaa", SuffixArray("aaaa")), "aaaa valid");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
