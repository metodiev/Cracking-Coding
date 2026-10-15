// Task 184 - Implement prefix function (KMP) for pattern matching. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Text;

public static class Solution
{
    /// <summary>
    /// pi[i] is the length of the longest proper prefix of s[:i+1] that is also a suffix of
    /// s[:i+1]. The fallback loop runs in O(n) overall because j only ever moves back via
    /// earlier prefix-function values.
    /// </summary>
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

    /// <summary>O(n^2) reference: longest proper border of every prefix, brute-forced.</summary>
    private static int[] PrefixFunctionNaive(string s)
    {
        int n = s.Length;
        var pi = new int[n];
        for (int end = 0; end < n; end++)
        {
            int best = 0;
            for (int length = 1; length <= end; length++)
            {
                if (s.Substring(0, length) == s.Substring(end - length + 1, length)) best = length;
            }
            pi[end] = best;
        }
        return pi;
    }

    private static bool Same(int[] left, int[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        Check(Same(PrefixFunction(""), new int[0]), "empty string");
        Check(Same(PrefixFunction("a"), new[] { 0 }), "single char");
        Check(Same(PrefixFunction("ab"), new[] { 0, 0 }), "no border");
        Check(Same(PrefixFunction("ababaca"), new[] { 0, 0, 1, 2, 3, 0, 1 }), "ababaca");
        Check(Same(PrefixFunction("aabaaab"), new[] { 0, 1, 0, 1, 2, 2, 3 }), "aabaaab");
        Check(Same(PrefixFunction("aaaa"), new[] { 0, 1, 2, 3 }), "aaaa");
        Check(Same(PrefixFunction("abcabcd"), new[] { 0, 0, 0, 1, 2, 3, 0 }), "abcabcd");
        Check(Same(PrefixFunction("abababab"), new[] { 0, 0, 1, 2, 3, 4, 5, 6 }), "abababab");
        Check(Same(PrefixFunction("aabaaabaaab"), new[] { 0, 1, 0, 1, 2, 2, 3, 4, 5, 6, 7 }), "aabaaabaaab");

        // agreement with the plain definition on pseudo-random strings
        var rng = new Random(5);
        for (int t = 0; t < 200; t++)
        {
            int length = rng.Next(41);
            var sb = new StringBuilder();
            for (int i = 0; i < length; i++) sb.Append((char)('a' + rng.Next(2)));
            string s = sb.ToString();
            Check(Same(PrefixFunction(s), PrefixFunctionNaive(s)), "cross-check " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
