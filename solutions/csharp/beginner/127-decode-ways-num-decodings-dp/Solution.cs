// Task 127 - Decode ways (num decodings) DP. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static int NumDecodings(string s)
    {
        if (s.Length == 0) return 0;
        var dp = new int[s.Length + 1];
        dp[0] = 1;
        dp[1] = s[0] == '0' ? 0 : 1;
        for (int i = 2; i <= s.Length; i++)
        {
            int one = s[i - 1] - '0';
            int two = int.Parse(s.Substring(i - 2, 2));
            if (one >= 1 && one <= 9) dp[i] += dp[i - 1];
            if (two >= 10 && two <= 26) dp[i] += dp[i - 2];
        }
        return dp[s.Length];
    }

    public static void Main()
    {
        Check(NumDecodings("12") == 2, "12");
        Check(NumDecodings("226") == 3, "226");
        Check(NumDecodings("06") == 0, "leading zero");
        Check(NumDecodings("10") == 1, "10");
        Check(NumDecodings("27") == 1, "27");
        Check(NumDecodings("11106") == 2, "11106");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
