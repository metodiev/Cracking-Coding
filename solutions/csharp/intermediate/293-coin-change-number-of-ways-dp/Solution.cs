// Task 293 - Coin change: number of ways (DP). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Number of unordered combinations of coins (unlimited supply) summing to amount.</summary>
    public static long CountWays(int[] coins, int amount)
    {
        var ways = new long[amount + 1];
        ways[0] = 1;
        foreach (int coin in coins)
        {
            for (int s = coin; s <= amount; s++) ways[s] += ways[s - coin];
        }
        return ways[amount];
    }

    /// <summary>Independent oracle: recursive count over coin index with descending multiplicities.</summary>
    public static long BruteWays(int[] coins, int amount)
    {
        return Count(coins, 0, amount);
    }

    private static long Count(int[] coins, int index, int remaining)
    {
        if (remaining == 0) return 1;
        if (index == coins.Length) return 0;
        long total = 0;
        for (int take = 0; take * coins[index] <= remaining; take++)
        {
            total += Count(coins, index + 1, remaining - take * coins[index]);
        }
        return total;
    }

    private static int seed = 24680;

    private static int[] LcgCoins(int n)
    {
        var values = new List<int>();
        while (values.Count < n)
        {
            seed = (seed * 75 + 74) % 65537;
            int value = 1 + seed % 8;
            if (!values.Contains(value)) values.Add(value);
        }
        values.Sort();
        return values.ToArray();
    }

    public static void Main()
    {
        Check(CountWays(new[] { 1, 2, 5 }, 5) == 4, "1,2,5 amount 5");
        Check(CountWays(new[] { 2 }, 3) == 0, "2 amount 3");
        Check(CountWays(new[] { 1, 2, 5 }, 0) == 1, "amount 0");
        Check(CountWays(new[] { 1 }, 10) == 1, "1 amount 10");
        Check(CountWays(Array.Empty<int>(), 0) == 1, "no coins amount 0");
        Check(CountWays(new[] { 2, 3, 5 }, 9) == 3, "2,3,5 amount 9");
        Check(CountWays(new[] { 1, 2 }, 3) == 2, "order must not matter");

        for (int trial = 0; trial < 150; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 5;
            var coins = LcgCoins(count);
            seed = (seed * 75 + 74) % 65537;
            int amount = seed % 60;
            Check(CountWays(coins, amount) == BruteWays(coins, amount), "random trial " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
