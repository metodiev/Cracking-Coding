// Task 124 - Word break (DP). (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool WordBreak(string s, string[] dictionary)
    {
        var words = new HashSet<string>(dictionary);
        var dp = new bool[s.Length + 1];
        dp[0] = true;
        for (int i = 1; i <= s.Length; i++)
            for (int j = 0; j < i; j++)
                if (dp[j] && words.Contains(s.Substring(j, i - j)))
                {
                    dp[i] = true;
                    break;
                }
        return dp[s.Length];
    }

    public static List<string>? WordBreakSegmentation(string s, string[] dictionary)
    {
        var words = new HashSet<string>(dictionary);
        var dp = new bool[s.Length + 1];
        var prev = new int[s.Length + 1];
        for (int i = 0; i < prev.Length; i++) prev[i] = -1;
        dp[0] = true;
        for (int i = 1; i <= s.Length; i++)
            for (int j = 0; j < i; j++)
                if (dp[j] && words.Contains(s.Substring(j, i - j)))
                {
                    dp[i] = true;
                    prev[i] = j;
                    break;
                }
        if (!dp[s.Length]) return null;
        var parts = new List<string>();
        for (int i = s.Length; i > 0; i = prev[i]) parts.Insert(0, s.Substring(prev[i], i - prev[i]));
        return parts;
    }

    public static void Main()
    {
        Check(WordBreak("leetcode", new[] { "leet", "code" }), "leetcode");
        Check(!WordBreak("catsandog", new[] { "cats", "dog", "sand", "and", "cat" }), "catsandog");
        Check(WordBreak("applepenapple", new[] { "apple", "pen" }), "applepenapple");
        Check(string.Join(",", WordBreakSegmentation("leetcode", new[] { "leet", "code" })!) == "leet,code", "segmentation");
        Check(WordBreakSegmentation("catsandog", new[] { "cats", "dog" }) == null, "no segmentation");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
