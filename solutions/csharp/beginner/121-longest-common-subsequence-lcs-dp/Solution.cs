// Task 121 - Longest common subsequence (LCS) DP. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Text;

public static class Solution
{
    public static int LcsLength(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i, j] = a[i - 1] == b[j - 1]
                    ? dp[i - 1, j - 1] + 1
                    : Math.Max(dp[i - 1, j], dp[i, j - 1]);
        return dp[n, m];
    }

    public static string LcsString(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i, j] = a[i - 1] == b[j - 1]
                    ? dp[i - 1, j - 1] + 1
                    : Math.Max(dp[i - 1, j], dp[i, j - 1]);
        var sb = new StringBuilder();
        int x = n, y = m;
        while (x > 0 && y > 0)
        {
            if (a[x - 1] == b[y - 1])
            {
                sb.Append(a[x - 1]);
                x--; y--;
            }
            else if (dp[x - 1, y] >= dp[x, y - 1]) x--;
            else y--;
        }
        var chars = sb.ToString().ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }

    private static bool IsSubsequence(string s, string t)
    {
        int j = 0;
        for (int i = 0; i < t.Length && j < s.Length; i++)
            if (t[i] == s[j]) j++;
        return j == s.Length;
    }

    public static void Main()
    {
        Check(LcsLength("ABCBDAB", "BDCABA") == 4, "classic");
        Check(LcsLength("", "abc") == 0, "empty");
        Check(LcsLength("abc", "abc") == 3, "identical");
        var seq = LcsString("ABCBDAB", "BDCABA");
        Check(seq.Length == 4 && IsSubsequence(seq, "ABCBDAB") && IsSubsequence(seq, "BDCABA"),
            "reconstruction is a common subsequence");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
