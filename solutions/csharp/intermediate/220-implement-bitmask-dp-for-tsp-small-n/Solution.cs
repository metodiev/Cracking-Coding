// Task 220 - Implement bitmask DP for TSP small N. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    public static long Tsp(int[][] dist)
    {
        int n = dist.Length;
        if (n <= 1) return 0;
        long INF = long.MaxValue / 4;
        int full = (1 << n) - 1;
        var dp = new long[1 << n, n];
        for (int m = 0; m <= full; m++)
            for (int i = 0; i < n; i++) dp[m, i] = INF;
        dp[1, 0] = 0;
        for (int mask = 0; mask <= full; mask++)
            for (int last = 0; last < n; last++)
            {
                if (dp[mask, last] == INF) continue;
                for (int nxt = 0; nxt < n; nxt++)
                {
                    if ((mask & (1 << nxt)) != 0) continue;
                    int newMask = mask | (1 << nxt);
                    long candidate = dp[mask, last] + dist[last][nxt];
                    if (candidate < dp[newMask, nxt]) dp[newMask, nxt] = candidate;
                }
            }
        long best = INF;
        for (int last = 0; last < n; last++)
            if (dp[full, last] != INF) best = Math.Min(best, dp[full, last] + dist[last][0]);
        return best;
    }

    public static void Main()
    {
        var square = new[]
        {
            new[] { 0, 1, 10, 10 },
            new[] { 1, 0, 1, 10 },
            new[] { 10, 1, 0, 1 },
            new[] { 1, 10, 1, 0 },
        };
        Check(Tsp(square) == 4, "square loop, all cycle edges 1");
        Check(Tsp(new[] { new[] { 0, 1, 3 }, new[] { 1, 0, 2 }, new[] { 3, 2, 0 } }) == 6, "triangle");
        Check(Tsp(new[] { new[] { 0 } }) == 0, "single city");
        Check(Tsp(new[] { new[] { 0, 7 }, new[] { 7, 0 } }) == 14, "two cities");

        var rng = new Random(6);
        int n = 7;
        var d = new int[n][];
        for (int i = 0; i < n; i++) d[i] = new int[n];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) d[i][j] = d[j][i] = 1 + rng.Next(19);

        var perm = new[] { 1, 2, 3, 4, 5, 6 };
        long brute = long.MaxValue;
        do
        {
            long cost = 0;
            int prev = 0;
            foreach (var city in perm)
            {
                cost += d[prev][city];
                prev = city;
            }
            cost += d[prev][0];
            brute = Math.Min(brute, cost);
        } while (NextPermutation(perm));

        Check(Tsp(d) == brute, "matches brute force");
        Console.WriteLine("PASS");
    }

    private static bool NextPermutation(int[] a)
    {
        int i = a.Length - 2;
        while (i >= 0 && a[i] >= a[i + 1]) i--;
        if (i < 0) return false;
        int j = a.Length - 1;
        while (a[j] <= a[i]) j--;
        (a[i], a[j]) = (a[j], a[i]);
        for (int lo = i + 1, hi = a.Length - 1; lo < hi; lo++, hi--) (a[lo], a[hi]) = (a[hi], a[lo]);
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
