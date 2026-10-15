// Task 224 - Stock with cooldown (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Unlimited transactions with a mandatory one-day rest after each sale.</summary>
    public static int MaxProfitCooldown(int[] prices)
    {
        if (prices.Length == 0) return 0;
        int held = -prices[0];                 // bought on day 0
        int sold = int.MinValue;               // no sale yet
        int rest = 0;                          // idle on day 0
        for (int i = 1; i < prices.Length; i++)
        {
            int nextHeld = Math.Max(held, rest - prices[i]);
            int nextSold = held + prices[i];
            int nextRest = Math.Max(rest, sold);
            held = nextHeld;
            sold = nextSold;
            rest = nextRest;
        }
        return Math.Max(sold, rest);
    }

    public static void Main()
    {
        Check(MaxProfitCooldown(new[] { 1, 2, 3, 0, 2 }) == 3, "buy 1/sell 2, rest, buy 0/sell 2");
        Check(MaxProfitCooldown(new[] { 1 }) == 0, "single");
        Check(MaxProfitCooldown(new int[0]) == 0, "empty");
        Check(MaxProfitCooldown(new[] { 2, 1, 4 }) == 3, "buy 1 sell 4");
        Check(MaxProfitCooldown(new[] { 6, 1, 3, 2, 4, 7 }) == 6, "hold through, one sale");
        Check(MaxProfitCooldown(new[] { 1, 2 }) == 1, "one up step");
        Check(MaxProfitCooldown(new[] { 2, 2, 2 }) == 0, "flat");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
