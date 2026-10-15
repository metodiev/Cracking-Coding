// Task 125 - Climbing stairs DP (count ways). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    public static long ClimbWays(int n)
    {
        if (n < 0) throw new ArgumentException("n must be >= 0");
        long a = 1, b = 1;
        for (int i = 0; i < n; i++)
        {
            long next = a + b;
            a = b;
            b = next;
        }
        return a;
    }

    public static long ClimbWaysGeneral(int n, int[] steps)
    {
        var dp = new long[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++)
            foreach (var s in steps)
                if (s <= i) dp[i] += dp[i - s];
        return dp[n];
    }

    public static void Main()
    {
        long[] expected = { 1, 1, 2, 3, 5, 8 };
        for (int n = 0; n < expected.Length; n++)
            Check(ClimbWays(n) == expected[n], "n=" + n);
        Check(ClimbWays(35) == 14930352L, "n=35");
        Check(ClimbWaysGeneral(4, new[] { 1, 2, 3 }) == 7, "steps [1,2,3]");
        Check(ClimbWaysGeneral(0, new[] { 1, 2 }) == 1, "zero stairs");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
