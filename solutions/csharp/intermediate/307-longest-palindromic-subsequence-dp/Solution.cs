// Task 307 - Longest palindromic subsequence (DP). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Linq;

public static class Solution
{
    public static int LongestPalindromicSubsequence(string s)
    {
        int n = s.Length;
        if (n == 0) return 0;
        var dp = new int[n, n];
        for (int i = 0; i < n; i++) dp[i, i] = 1;
        for (int length = 2; length <= n; length++)
        {
            for (int i = 0; i + length - 1 < n; i++)
            {
                int j = i + length - 1;
                if (s[i] == s[j]) dp[i, j] = length == 2 ? 2 : dp[i + 1, j - 1] + 2;
                else dp[i, j] = Math.Max(dp[i + 1, j], dp[i, j - 1]);
            }
        }
        return dp[0, n - 1];
    }

    public static int BruteForce(string s)
    {
        int best = 0;
        for (int mask = 0; mask < (1 << s.Length); mask++)
        {
            var chars = new System.Collections.Generic.List<char>();
            for (int i = 0; i < s.Length; i++) if ((mask >> i & 1) == 1) chars.Add(s[i]);
            var candidate = new string(chars.ToArray());
            if (candidate.Length > best && candidate == new string(candidate.Reverse().ToArray())) best = candidate.Length;
        }
        return best;
    }

    private static int seed = 9753;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "abc"[seed % 3];
        }
        return new string(chars);
    }

    public static void Main()
    {
        Check(LongestPalindromicSubsequence("bbbab") == 4, "bbbab");
        Check(LongestPalindromicSubsequence("cbbd") == 2, "cbbd");
        Check(LongestPalindromicSubsequence("abcde") == 1, "abcde");
        Check(LongestPalindromicSubsequence("racecar") == 7, "racecar");
        Check(LongestPalindromicSubsequence("") == 0, "empty");
        Check(LongestPalindromicSubsequence("a") == 1, "single");

        for (int trial = 0; trial < 60; trial++)
        {
            string s = LcgString(10);
            Check(LongestPalindromicSubsequence(s) == BruteForce(s), "random " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
