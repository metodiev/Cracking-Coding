// Task 231 - Manacher's algorithm for longest palindromic substring. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Text;

public static class Solution
{
    /// <summary>Manacher: leftmost longest palindromic substring of s; "" for empty input.</summary>
    public static string LongestPalindrome(string s)
    {
        if (s.Length == 0) return "";
        var sb = new StringBuilder("^");
        foreach (char c in s) sb.Append('#').Append(c);
        sb.Append("#$");
        string t = sb.ToString();
        int n = t.Length;
        var p = new int[n];
        int centre = 0, right = 0, bestLen = 0, bestStart = 0;
        for (int i = 1; i < n - 1; i++)
        {
            int mirror = 2 * centre - i;
            if (i < right) p[i] = Math.Min(right - i, p[mirror]);
            while (t[i + 1 + p[i]] == t[i - 1 - p[i]]) p[i]++;
            if (i + p[i] > right) { centre = i; right = i + p[i]; }
            if (p[i] > bestLen)                        // strict: ties keep the leftmost centre
            {
                bestLen = p[i];
                bestStart = (i - p[i]) / 2;
            }
        }
        return s.Substring(bestStart, bestLen);
    }

    /// <summary>O(n^2) reference helper: length of the longest palindromic substring of s.</summary>
    public static int ExpandAroundCentreLength(string s)
    {
        int best = 0;
        for (int i = 0; i < s.Length; i++)
        {
            best = Math.Max(best, Expand(s, i, i));
            best = Math.Max(best, Expand(s, i, i + 1));
        }
        return best;
    }

    private static int Expand(string s, int lo, int hi)
    {
        while (lo >= 0 && hi < s.Length && s[lo] == s[hi]) { lo--; hi++; }
        return hi - lo - 1;
    }

    public static void Main()
    {
        Check(LongestPalindrome("babad") == "bab", "leftmost of \"bab\"/\"aba\"");
        Check(LongestPalindrome("cbbd") == "bb", "even palindrome");
        Check(LongestPalindrome("forgeeksskeegfor") == "geeksskeeg", "classic");
        Check(LongestPalindrome("abacaba") == "abacaba", "whole string");
        Check(LongestPalindrome("bananas") == "anana", "bananas");
        Check(LongestPalindrome("abcde") == "a", "no repeats");
        Check(LongestPalindrome("ccc") == "ccc", "all same");
        Check(LongestPalindrome("") == "", "empty");
        Check(LongestPalindrome("a") == "a", "single");

        var rng = new Random(231);
        var sb = new StringBuilder();
        for (int t = 0; t < 300; t++)              // cross-check vs expand-around-centre
        {
            int len = rng.Next(41);
            sb.Clear();
            for (int i = 0; i < len; i++) sb.Append(rng.Next(2) == 0 ? 'a' : 'b');
            string s = sb.ToString();
            string got = LongestPalindrome(s);
            Check(got == Reverse(got), "palindromic result");
            Check(got.Length == ExpandAroundCentreLength(s), "length matches expand-around-centre");
            if (s.Length > 0)
            {
                int first = -1;
                for (int i = 0; i + got.Length <= s.Length && first < 0; i++)
                {
                    string sub = s.Substring(i, got.Length);
                    if (sub == Reverse(sub)) first = i;
                }
                Check(s.IndexOf(got, StringComparison.Ordinal) == first, "leftmost among maximal palindromes");
            }
        }
        Console.WriteLine("PASS");
    }

    private static string Reverse(string s)
    {
        var chars = s.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
