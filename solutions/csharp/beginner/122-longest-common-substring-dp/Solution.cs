// Task 122 - Longest common substring (DP). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Returns (length, substring) — substring is taken from a.</summary>
    public static (int Length, string Substring) LongestCommonSubstring(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        int bestLen = 0, bestEnd = 0;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                if (a[i - 1] == b[j - 1])
                {
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                    if (dp[i, j] > bestLen)
                    {
                        bestLen = dp[i, j];
                        bestEnd = i;
                    }
                }
        return (bestLen, a.Substring(bestEnd - bestLen, bestLen));
    }

    public static void Main()
    {
        var r = LongestCommonSubstring("abcdef", "zbcdf");
        Check(r.Length == 3 && r.Substring == "bcd", "classic");
        Check(LongestCommonSubstring("abc", "abc").Length == 3, "identical");
        Check(LongestCommonSubstring("abc", "xyz").Length == 0, "no match");
        Check(LongestCommonSubstring("", "abc").Length == 0, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
