// Task 207 - Three-sum closest. (Tier: Intermediate, CTCI Ch 16)
using System;

public static class Solution
{
    public static int ThreeSumClosest(int[] nums, int target)
    {
        if (nums.Length < 3) throw new ArgumentException("need at least three values");
        var a = (int[])nums.Clone();
        Array.Sort(a);
        int best = a[0] + a[1] + a[2];
        for (int i = 0; i + 2 < a.Length; i++)
        {
            int lo = i + 1, hi = a.Length - 1;
            while (lo < hi)
            {
                int total = a[i] + a[lo] + a[hi];
                if (Math.Abs(total - target) < Math.Abs(best - target)) best = total;
                if (total < target) lo++;
                else if (total > target) hi--;
                else return total;
            }
        }
        return best;
    }

    public static void Main()
    {
        Check(ThreeSumClosest(new[] { -1, 2, 1, -4 }, 1) == 2, "classic");
        Check(ThreeSumClosest(new[] { 0, 0, 0 }, 1) == 0, "zeros");
        Check(ThreeSumClosest(new[] { 1, 1, 1, 1 }, 5) == 3, "all ones");
        Check(ThreeSumClosest(new[] { -5, -2, -1 }, -3) == -8, "negatives");

        bool threw = false;
        try { ThreeSumClosest(new[] { 1, 2 }, 3); } catch (ArgumentException) { threw = true; }
        Check(threw, "needs three values");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
