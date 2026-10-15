// Task 123 - Edit distance (Levenshtein) DP. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static int EditDistance(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = j;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i, j] = a[i - 1] == b[j - 1]
                    ? dp[i - 1, j - 1]
                    : 1 + Math.Min(dp[i - 1, j], Math.Min(dp[i, j - 1], dp[i - 1, j - 1]));
        return dp[n, m];
    }

    public static int EditDistanceRollingRow(string a, string b)
    {
        var prev = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++) prev[j] = j;
        for (int i = 1; i <= a.Length; i++)
        {
            var cur = new int[b.Length + 1];
            cur[0] = i;
            for (int j = 1; j <= b.Length; j++)
                cur[j] = a[i - 1] == b[j - 1]
                    ? prev[j - 1]
                    : 1 + Math.Min(prev[j], Math.Min(cur[j - 1], prev[j - 1]));
            prev = cur;
        }
        return prev[b.Length];
    }

    public static void Main()
    {
        Check(EditDistance("kitten", "sitting") == 3, "kitten/sitting");
        Check(EditDistance("", "abc") == 3, "empty->abc");
        Check(EditDistance("same", "same") == 0, "identical");
        Check(EditDistance("flaw", "lawn") == 2, "flaw/lawn");
        Check(EditDistance("abc", "yabd") == EditDistanceRollingRow("abc", "yabd"), "rolling row agrees");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
