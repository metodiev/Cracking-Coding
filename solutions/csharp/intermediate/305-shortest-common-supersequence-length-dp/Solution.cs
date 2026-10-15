// Task 305 - Shortest common supersequence length (DP). (Tier: Intermediate, CTCI Ch 1)
using System;

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

    /// <summary>Direct DP: shortest string containing both a and b as subsequences.</summary>
    public static int ScsLength(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                dp[i, j] = a[i - 1] == b[j - 1] ? dp[i - 1, j - 1] + 1 : Math.Min(dp[i - 1, j], dp[i, j - 1]) + 1;
            }
        }
        return dp[n, m];
    }

    private static int seed = 4682;

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
        Check(ScsLength("AGGTAB", "GXTXAYB") == 9, "AGGTAB/GXTXAYB");
        Check(ScsLength("ab", "b") == 2, "ab/b");
        Check(ScsLength("", "abc") == 3, "empty left");
        Check(ScsLength("abc", "") == 3, "empty right");
        Check(ScsLength("", "") == 0, "both empty");
        Check(ScsLength("abc", "abc") == 3, "identical");
        Check(ScsLength("geek", "eke") == 5, "geek/eke");

        foreach (var pair in new[] { ("AGGTAB", "GXTXAYB"), ("geek", "eke"), ("ab", "b") })
            Check(ScsLength(pair.Item1, pair.Item2) == pair.Item1.Length + pair.Item2.Length - LcsLength(pair.Item1, pair.Item2), "lcs formula " + pair.Item1);

        var strings = new string[40];
        for (int i = 0; i < strings.Length; i++) strings[i] = LcgString(6);
        for (int i = 0; i + 1 < strings.Length; i += 2)
        {
            string a = strings[i], b = strings[i + 1];
            Check(ScsLength(a, b) == a.Length + b.Length - LcsLength(a, b), "random " + a + "/" + b);
            Check(ScsLength(a, b) >= Math.Max(a.Length, b.Length), "lower bound " + a);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
