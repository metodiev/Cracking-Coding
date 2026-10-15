// Task 223 - Stock buy/sell multiple transactions (greedy). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Unlimited transactions; sum every positive consecutive gain.</summary>
    public static int MaxProfitMultiple(int[] prices)
    {
        int total = 0;
        for (int i = 1; i < prices.Length; i++)
        {
            if (prices[i] > prices[i - 1]) total += prices[i] - prices[i - 1];
        }
        return total;
    }

    public static void Main()
    {
        Check(MaxProfitMultiple(new[] { 7, 1, 5, 3, 6, 4 }) == 7, "(5-1) + (6-3)");
        Check(MaxProfitMultiple(new[] { 7, 6, 4, 3, 1 }) == 0, "monotone decreasing");
        Check(MaxProfitMultiple(new[] { 1, 2, 3, 4, 5 }) == 4, "one long rise");
        Check(MaxProfitMultiple(new[] { 1, 3, 2, 8 }) == 8, "+2 then +6");
        Check(MaxProfitMultiple(new int[0]) == 0, "empty");
        Check(MaxProfitMultiple(new[] { 5 }) == 0, "single");
        Check(MaxProfitMultiple(new[] { 3, 3 }) == 0, "flat");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
