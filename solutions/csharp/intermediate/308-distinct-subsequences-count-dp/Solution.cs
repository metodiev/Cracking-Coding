// Task 308 - Count distinct subsequences. (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Numerics;

public static class Solution
{
    public static BigInteger CountDistinctSubsequences(string s, string t)
    {
        int n = s.Length, m = t.Length;
        if (m == 0) return BigInteger.One;
        if (n < m) return BigInteger.Zero;
        var dp = new BigInteger[n + 1, m + 1];
        for (int i = 0; i <= n; i++) dp[i, 0] = BigInteger.One;
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                dp[i, j] = dp[i - 1, j];
                if (s[i - 1] == t[j - 1]) dp[i, j] += dp[i - 1, j - 1];
            }
        }
        return dp[n, m];
    }

    public static long BruteForce(string s, string t)
    {
        long count = 0;
        for (int mask = 0; mask < (1 << s.Length); mask++)
        {
            var chars = new System.Collections.Generic.List<char>();
            for (int i = 0; i < s.Length; i++) if ((mask >> i & 1) == 1) chars.Add(s[i]);
            if (new string(chars.ToArray()) == t) count++;
        }
        return count;
    }

    private static int seed = 8642;

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
        Check(CountDistinctSubsequences("rabbbit", "rabbit") == new BigInteger(3), "rabbbit");
        Check(CountDistinctSubsequences("babgbag", "bag") == new BigInteger(5), "babgbag");
        Check(CountDistinctSubsequences("abc", "") == BigInteger.One, "empty t");
        Check(CountDistinctSubsequences("", "a") == BigInteger.Zero, "empty s");
        Check(CountDistinctSubsequences("aaa", "a") == new BigInteger(3), "aaa/a");
        Check(CountDistinctSubsequences("aaa", "aa") == new BigInteger(3), "aaa/aa");
        Check(CountDistinctSubsequences("abc", "abcd") == BigInteger.Zero, "t longer than s");

        var strings = new string[40];
        for (int i = 0; i < strings.Length; i++) strings[i] = LcgString(8);
        for (int i = 0; i + 1 < strings.Length; i += 2)
        {
            string s = strings[i];
            string t = strings[i + 1].Substring(0, 4);
            Check(CountDistinctSubsequences(s, t) == new BigInteger(BruteForce(s, t)), "random " + s + "/" + t);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
