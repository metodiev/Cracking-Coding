// Task 206 - Pair with sum closest to target in sorted array. (Tier: Intermediate, CTCI Ch 16)
using System;

public static class Solution
{
    /// <summary>Returns (x, y, sum); on equal distance the first pair found wins.</summary>
    public static (int x, int y, int sum) ClosestPair(int[] a, int target)
    {
        if (a.Length < 2) throw new ArgumentException("need at least two values");
        int lo = 0, hi = a.Length - 1;
        int bestX = a[lo], bestY = a[hi];
        int bestDiff = Math.Abs(bestX + bestY - target);
        while (lo < hi)
        {
            int total = a[lo] + a[hi];
            int diff = Math.Abs(total - target);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                bestX = a[lo];
                bestY = a[hi];
            }
            if (total < target) lo++;
            else if (total > target) hi--;
            else break;
        }
        return (bestX, bestY, bestX + bestY);
    }

    public static void Main()
    {
        var a = new[] { 1, 3, 4, 5, 7 };
        Check(ClosestPair(a, 10) == (3, 7, 10), "exact");
        Check(ClosestPair(a, 8) == (1, 7, 8), "exact 2");
        Check(ClosestPair(a, 0) == (1, 3, 4), "closest below");
        Check(ClosestPair(new[] { 2, 4 }, 5) == (2, 4, 6), "two values");

        bool threw = false;
        try { ClosestPair(new[] { 1 }, 5); } catch (ArgumentException) { threw = true; }
        Check(threw, "needs two values");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
