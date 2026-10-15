// Task 185 - KMP search using prefix function. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    /// <summary>Prefix function (LPS array) of `s`, as in task 184.</summary>
    private static int[] PrefixFunction(string s)
    {
        int n = s.Length;
        var pi = new int[n];
        for (int i = 1; i < n; i++)
        {
            int j = pi[i - 1];
            while (j > 0 && s[i] != s[j]) j = pi[j - 1];
            if (s[i] == s[j]) j++;
            pi[i] = j;
        }
        return pi;
    }

    /// <summary>
    /// All start indices of `pattern` in `text`, in increasing order (overlapping matches
    /// included), in O(n + m). An empty pattern returns an empty list by convention:
    /// "no match" avoids an unbounded list of positions.
    /// </summary>
    private static List<int> KmpSearch(string text, string pattern)
    {
        var matches = new List<int>();
        if (pattern.Length == 0) return matches;
        int[] pi = PrefixFunction(pattern);
        int j = 0;
        for (int i = 0; i < text.Length; i++)
        {
            while (j > 0 && text[i] != pattern[j]) j = pi[j - 1];
            if (text[i] == pattern[j]) j++;
            if (j == pattern.Length)
            {
                matches.Add(i - pattern.Length + 1);
                j = pi[j - 1];                // keep going, matches may overlap
            }
        }
        return matches;
    }

    /// <summary>Reference scan used by the self-check.</summary>
    private static List<int> KmpSearchNaive(string text, string pattern)
    {
        var matches = new List<int>();
        if (pattern.Length == 0) return matches;
        for (int i = 0; i + pattern.Length <= text.Length; i++)
        {
            if (text.Substring(i, pattern.Length) == pattern) matches.Add(i);
        }
        return matches;
    }

    private static bool Same(List<int> actual, params int[] expected)
    {
        return Same(actual, new List<int>(expected));
    }

    private static bool Same(List<int> actual, List<int> expected)
    {
        if (actual.Count != expected.Count) return false;
        for (int i = 0; i < actual.Count; i++)
        {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        Check(Same(KmpSearch("aaaaa", "aaaa"), 0, 1), "overlapping matches");
        Check(Same(KmpSearch("abababa", "aba"), 0, 2, 4), "abababa");
        Check(Same(KmpSearch("abcabcabc", "abc"), 0, 3, 6), "repeated block");
        Check(Same(KmpSearch("abracadabra", "cad"), 4), "single match");
        Check(Same(KmpSearch("mississippi", "issi"), 1, 4), "mississippi");
        Check(Same(KmpSearch("aaaa", "b")), "no match");
        Check(Same(KmpSearch("ab", "abc")), "pattern longer than text");
        Check(Same(KmpSearch("", "a")), "empty text");
        Check(Same(KmpSearch("", "")), "empty text and pattern");
        Check(Same(KmpSearch("anything", "")), "empty pattern");
        Check(Same(KmpSearch("a", "a"), 0), "single char");
        Check(Same(KmpSearch("aaa", "aa"), 0, 1), "aaa / aa");
        Check(Same(KmpSearch("aaa", "aaa"), 0), "exact match");

        // agreement with the naive scan on pseudo-random strings
        var rng = new Random(13);
        for (int t = 0; t < 300; t++)
        {
            int textLength = rng.Next(31);
            var textBuilder = new StringBuilder();
            for (int i = 0; i < textLength; i++) textBuilder.Append((char)('a' + rng.Next(2)));
            string text = textBuilder.ToString();
            int patternLength = 1 + rng.Next(5);
            var patternBuilder = new StringBuilder();
            for (int i = 0; i < patternLength; i++) patternBuilder.Append((char)('a' + rng.Next(2)));
            string pattern = patternBuilder.ToString();
            Check(Same(KmpSearch(text, pattern), KmpSearchNaive(text, pattern)), "cross-check " + text + " / " + pattern);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
