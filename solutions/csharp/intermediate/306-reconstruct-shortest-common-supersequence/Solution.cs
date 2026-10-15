// Task 306 - Reconstruct the shortest common supersequence. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static string ShortestCommonSupersequence(string a, string b)
    {
        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = j;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                if (a[i - 1] == b[j - 1]) dp[i, j] = dp[i - 1, j - 1] + 1;
                else dp[i, j] = Math.Min(dp[i - 1, j], dp[i, j - 1]) + 1;
            }
        }

        var reversed = new List<char>();
        int x = n, y = m;
        while (x > 0 && y > 0)
        {
            if (a[x - 1] == b[y - 1])
            {
                reversed.Add(a[x - 1]);
                x--; y--;
            }
            else if (dp[x - 1, y] <= dp[x, y - 1])       // tie prefers a's character
            {
                reversed.Add(a[x - 1]);
                x--;
            }
            else
            {
                reversed.Add(b[y - 1]);
                y--;
            }
        }
        while (x > 0) reversed.Add(a[--x]);
        while (y > 0) reversed.Add(b[--y]);
        reversed.Reverse();
        return new string(reversed.ToArray());
    }

    public static bool IsSubsequence(string candidate, string text)
    {
        int idx = 0;
        for (int i = 0; i < text.Length && idx < candidate.Length; i++)
        {
            if (text[i] == candidate[idx]) idx++;
        }
        return idx == candidate.Length;
    }

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

    private static int seed = 3141;

    private static string LcgString(int length)
    {
        var chars = new char[length];
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = "ab"[seed % 2];
        }
        return new string(chars);
    }

    public static void Main()
    {
        Check(ShortestCommonSupersequence("ab", "b") == "ab", "ab/b");
        Check(ShortestCommonSupersequence("abc", "abc") == "abc", "identical");
        Check(ShortestCommonSupersequence("", "abc") == "abc", "empty left");
        Check(ShortestCommonSupersequence("abc", "") == "abc", "empty right");
        Check(ShortestCommonSupersequence("", "") == "", "both empty");

        string a = "AGGTAB", b = "GXTXAYB";
        string scs = ShortestCommonSupersequence(a, b);
        Check(scs.Length == 9 && scs.Length == a.Length + b.Length - LcsLength(a, b), "classic length");
        Check(IsSubsequence(a, scs) && IsSubsequence(b, scs), "classic contents");

        var strings = new string[40];
        for (int i = 0; i < strings.Length; i++) strings[i] = LcgString(5);
        for (int i = 0; i + 1 < strings.Length; i += 2)
        {
            string x = strings[i], y = strings[i + 1];
            string candidate = ShortestCommonSupersequence(x, y);
            Check(candidate.Length == x.Length + y.Length - LcsLength(x, y), "random length " + x + "/" + y);
            Check(IsSubsequence(x, candidate) && IsSubsequence(y, candidate), "random contents " + x + "/" + y);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
