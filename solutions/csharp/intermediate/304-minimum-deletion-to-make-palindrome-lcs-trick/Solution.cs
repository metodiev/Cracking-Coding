// Task 304 - Minimum deletions to make a palindrome (LCS trick). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Linq;

public static class Solution
{
    public static int LcsLength(string a, string b)
    {
        var previous = new int[b.Length + 1];
        for (int i = 1; i <= a.Length; i++)
        {
            var current = new int[b.Length + 1];
            for (int j = 1; j <= b.Length; j++)
            {
                current[j] = a[i - 1] == b[j - 1] ? previous[j - 1] + 1 : Math.Max(previous[j], current[j - 1]);
            }
            previous = current;
        }
        return previous[b.Length];
    }

    /// <summary>Direct interval DP, independent of the LCS formulation.</summary>
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

    public static int MinimumDeletions(string s) => s.Length - LongestPalindromicSubsequence(s);

    private static int seed = 3579;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "abcd"[seed % 4];
        }
        return new string(chars);
    }

    public static void Main()
    {
        Check(MinimumDeletions("aebcbda") == 2, "aebcbda");
        Check(MinimumDeletions("abc") == 2, "abc");
        Check(MinimumDeletions("abcd") == 3, "abcd");
        Check(MinimumDeletions("aa") == 0, "aa");
        Check(MinimumDeletions("racecar") == 0, "racecar");
        Check(MinimumDeletions("") == 0, "empty");
        Check(MinimumDeletions("a") == 0, "single");

        foreach (string s in new[] { "aebcbda", "geeksforgeeks", "abca", "abbc", "" })
            Check(LongestPalindromicSubsequence(s) == LcsLength(s, new string(s.Reverse().ToArray())), "lcs trick " + s);

        for (int trial = 0; trial < 60; trial++)
        {
            string s = LcgString(10);
            int lps = LongestPalindromicSubsequence(s);
            Check(lps == LcsLength(s, new string(s.Reverse().ToArray())), "random " + s);
            Check(s.Length - lps >= 0 && s.Length - lps <= s.Length, "bounds " + s);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
