// Task 225 - Stock with transaction fee (DP). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Unlimited transactions, each sale pays a fixed fee.</summary>
    public static int MaxProfitFee(int[] prices, int fee)
    {
        if (fee < 0) throw new ArgumentException("fee must be non-negative");
        if (prices.Length == 0) return 0;
        int cash = 0, hold = -prices[0];
        for (int i = 1; i < prices.Length; i++)
        {
            int nextHold = Math.Max(hold, cash - prices[i]);
            int nextCash = Math.Max(cash, hold + prices[i] - fee);
            hold = nextHold;
            cash = nextCash;
        }
        return cash;
    }

    public static void Main()
    {
        Check(MaxProfitFee(new[] { 1, 3, 2, 8, 4, 9 }, 2) == 8, "(8-1-2) + (9-4-2)");
        Check(MaxProfitFee(new[] { 1, 3, 2, 8, 4, 9 }, 0) == 13, "fee 0 matches greedy 223");
        Check(MaxProfitFee(new[] { 1, 2 }, 5) == 0, "fee eats the gain");
        Check(MaxProfitFee(new[] { 1, 2 }, 0) == 1, "free trade");
        Check(MaxProfitFee(new[] { 1, 4 }, 1) == 2, "fee 1");
        Check(MaxProfitFee(new int[0], 2) == 0, "empty");
        Check(MaxProfitFee(new[] { 5 }, 1) == 0, "single");
        Check(MaxProfitFee(new[] { 5 }, 0) == 0, "single with fee 0");

        bool threw = false;
        try { MaxProfitFee(new[] { 1, 2 }, -1); } catch (ArgumentException) { threw = true; }
        Check(threw, "negative fee rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
