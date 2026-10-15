// Task 222 - Stock buy/sell single transaction max profit. (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>One buy and one sell (buy before sell); 0 when no profit is possible.</summary>
    public static int MaxProfitSingle(int[] prices)
    {
        int best = 0, cheapest = int.MaxValue;
        foreach (int p in prices)
        {
            cheapest = Math.Min(cheapest, p);
            best = Math.Max(best, p - cheapest);
        }
        return best;
    }

    public static void Main()
    {
        Check(MaxProfitSingle(new[] { 7, 1, 5, 3, 6, 4 }) == 5, "buy 1 sell 6");
        Check(MaxProfitSingle(new[] { 7, 6, 4, 3, 1 }) == 0, "only falls");
        Check(MaxProfitSingle(new[] { 1 }) == 0, "single day");
        Check(MaxProfitSingle(new int[0]) == 0, "no days");
        Check(MaxProfitSingle(new[] { 1, 2 }) == 1, "one up step");
        Check(MaxProfitSingle(new[] { 2, 1 }) == 0, "one down step");
        Check(MaxProfitSingle(new[] { 3, 3 }) == 0, "flat");
        Check(MaxProfitSingle(new[] { 5, 4, 6, 2, 7 }) == 5, "buy 2 sell 7");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
