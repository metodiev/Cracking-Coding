// Task 230 - Longest palindrome substring (expand center). (Tier: Intermediate, CTCI Ch 1)
using System;

public static class Solution
{
    /// <summary>
    /// Expands around each of the 2n-1 centers. Centers are visited left to
    /// right with the odd center before the even one, and only strictly longer
    /// palindromes replace the answer, so the first longest one wins ties.
    /// </summary>
    public static string LongestPalindromeSubstring(string s)
    {
        if (s.Length == 0) return "";
        int start = 0, length = 1;
        for (int c = 0; c < s.Length; c++)
        {
            var odd = Expand(s, c, c);
            if (odd.length > length) { start = odd.start; length = odd.length; }
            var even = Expand(s, c, c + 1);
            if (even.length > length) { start = even.start; length = even.length; }
        }
        return s.Substring(start, length);
    }

    /// <summary>Returns (start, length) of the longest palindrome around (l, r).</summary>
    private static (int start, int length) Expand(string s, int l, int r)
    {
        while (l >= 0 && r < s.Length && s[l] == s[r]) { l--; r++; }
        return (l + 1, r - l - 1);
    }

    public static void Main()
    {
        Check(LongestPalindromeSubstring("babad") == "bab", "tie goes to the first found");
        Check(LongestPalindromeSubstring("cbbd") == "bb", "even center");
        Check(LongestPalindromeSubstring("a") == "a", "single char");
        Check(LongestPalindromeSubstring("") == "", "empty");
        Check(LongestPalindromeSubstring("aaaa") == "aaaa", "all same");
        Check(LongestPalindromeSubstring("forgeeksskeegfor") == "geeksskeeg", "long even");
        Check(LongestPalindromeSubstring("abaxyzzyxf") == "xyzzyx", "long even 2");
        Check(LongestPalindromeSubstring("ac") == "a", "no palindrome pair");
        Check(LongestPalindromeSubstring("aac") == "aa", "prefix pair");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
