// Task 294 - Coin change: minimum coins (DP). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Fewest coins summing to amount, or -1 when impossible (documented sentinel).</summary>
    public static int MinCoins(int[] coins, int amount)
    {
        int infinity = int.MaxValue / 4;
        var dp = new int[amount + 1];
        for (int i = 0; i <= amount; i++) dp[i] = infinity;
        dp[0] = 0;
        for (int s = 1; s <= amount; s++)
        {
            foreach (int coin in coins)
            {
                if (coin <= s && dp[s - coin] + 1 < dp[s]) dp[s] = dp[s - coin] + 1;
            }
        }
        return dp[amount] == infinity ? -1 : dp[amount];
    }

    /// <summary>Independent oracle: BFS over reachable amounts gives fewest-coin distances.</summary>
    public static int BfsMinCoins(int[] coins, int amount)
    {
        var dist = new int[amount + 1];
        for (int i = 0; i <= amount; i++) dist[i] = -1;
        dist[0] = 0;
        var queue = new int[amount + 1];
        int head = 0;
        int tail = 1;
        while (head < tail)
        {
            int u = queue[head++];
            foreach (int coin in coins)
            {
                int v = u + coin;
                if (v <= amount && dist[v] == -1)
                {
                    dist[v] = dist[u] + 1;
                    queue[tail++] = v;
                }
            }
        }
        return dist[amount];
    }

    private static int seed = 13579;

    private static int[] LcgCoins(int n)
    {
        var values = new List<int>();
        while (values.Count < n)
        {
            seed = (seed * 75 + 74) % 65537;
            int value = 1 + seed % 7;
            if (!values.Contains(value)) values.Add(value);
        }
        values.Sort();
        return values.ToArray();
    }

    public static void Main()
    {
        Check(MinCoins(new[] { 1, 2, 5 }, 11) == 3, "1,2,5 amount 11");
        Check(MinCoins(new[] { 2 }, 3) == -1, "impossible sentinel");
        Check(MinCoins(new[] { 1, 2, 5 }, 0) == 0, "amount 0");
        Check(MinCoins(new[] { 1, 3, 4 }, 6) == 2, "greedy counterexample");
        Check(MinCoins(new[] { 1, 3, 4 }, 9) == 3, "1,3,4 amount 9");
        Check(MinCoins(new[] { 5, 7 }, 11) == -1, "5,7 amount 11");
        Check(MinCoins(Array.Empty<int>(), 0) == 0, "no coins amount 0");

        for (int trial = 0; trial < 150; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 4;
            var coins = LcgCoins(count);
            seed = (seed * 75 + 74) % 65537;
            int amount = seed % 80;
            Check(MinCoins(coins, amount) == BfsMinCoins(coins, amount), "random trial " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
