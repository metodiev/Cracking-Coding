// Task 398 - Implement a divide-and-conquer DP optimization example. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const long INF = long.MaxValue / 4;

    public static long PartitionNaive(int[] values, int k)
    {
        int n = values.Length;
        if (k > n) return INF;
        var prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + values[i - 1];
        var previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        for (int t = 1; t <= k; t++)
        {
            var current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            for (int i = t; i <= n; i++)
            {
                for (int j = t - 1; j < i; j++)
                {
                    if (previous[j] == INF) continue;
                    long group = prefix[i] - prefix[j];
                    long cost = previous[j] + group * group;
                    if (cost < current[i]) current[i] = cost;
                }
            }
            previous = current;
        }
        return previous[n];
    }

    private static long[] dcPrevious = Array.Empty<long>();
    private static long[] dcCurrent = Array.Empty<long>();
    private static long[] dcPrefix = Array.Empty<long>();
    private static int[] dcOpt = Array.Empty<int>();

    /// <summary>Returns (answer, opt table per layer).</summary>
    public static (long Answer, List<int[]> AllOpts) PartitionDc(int[] values, int k)
    {
        int n = values.Length;
        if (k > n) return (INF, new List<int[]>());
        dcPrefix = new long[n + 1];
        for (int i = 1; i <= n; i++) dcPrefix[i] = dcPrefix[i - 1] + values[i - 1];
        var previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        var allOpts = new List<int[]>();
        for (int t = 1; t <= k; t++)
        {
            var current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            dcOpt = new int[n + 1];
            // i == t: the only split is j == t-1; the last singleton group costs values[t-1]^2
            current[t] = previous[t - 1] + (long)values[t - 1] * values[t - 1];
            dcOpt[t] = t - 1;
            dcPrevious = previous;
            dcCurrent = current;
            if (t + 1 <= n) Solve(t + 1, n, t - 1, n - 1);
            allOpts.Add((int[])dcOpt.Clone());
            previous = current;
        }
        return (previous[n], allOpts);
    }

    private static void Solve(int l, int r, int optl, int optr)
    {
        if (l > r) return;
        int mid = (l + r) >> 1;
        long best = INF;
        int bestJ = optl;
        int upper = Math.Min(optr, mid - 1);
        for (int j = optl; j <= upper; j++)
        {
            if (dcPrevious[j] == INF) continue;
            long group = dcPrefix[mid] - dcPrefix[j];
            long cost = dcPrevious[j] + group * group;
            if (cost < best)
            {
                best = cost;
                bestJ = j;
            }
        }
        dcCurrent[mid] = best;
        dcOpt[mid] = bestJ;
        Solve(l, mid - 1, optl, bestJ);
        Solve(mid + 1, r, bestJ, optr);
    }

    public static bool OptIsMonotone(int[] opt, int t, int n)
    {
        int last = -1;
        for (int i = t; i <= n; i++)
        {
            if (opt[i] < last) return false;
            last = opt[i];
        }
        return true;
    }

    private static int seed = 804;

    private static int[][] LcgArrays(int count, int maxLen)
    {
        var arrays = new int[count][];
        for (int a = 0; a < count; a++)
        {
            int n = 2 + seed % (maxLen - 1);
            var values = new int[n];
            for (int i = 0; i < n; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                values[i] = 1 + seed % 9;
            }
            arrays[a] = values;
        }
        return arrays;
    }

    public static void Main()
    {
        Check(PartitionNaive(new[] { 1, 2, 3, 4 }, 2) == 52, "naive 52");
        Check(PartitionDc(new[] { 1, 2, 3, 4 }, 2).Answer == 52, "dc 52");
        Check(PartitionNaive(new[] { 1, 2, 3, 4 }, 1) == 100, "single group");
        Check(PartitionNaive(new[] { 1, 2, 3, 4 }, 4) == 30, "all singletons");
        Check(PartitionNaive(new[] { 5 }, 1) == 25, "one element");
        Check(PartitionNaive(new[] { 1, 2 }, 3) == INF, "infeasible");
        Check(PartitionDc(new[] { 1, 2, 3, 4 }, 4).Answer == 30, "dc all singletons");

        foreach (var values in LcgArrays(40, 60))
        {
            int n = values.Length;
            for (int k = 1; k <= Math.Min(n, 8); k++)
            {
                long naive = PartitionNaive(values, k);
                var (answer, allOpts) = PartitionDc(values, k);
                Check(answer == naive, "agreement");
                for (int t = 1; t <= allOpts.Count; t++)
                {
                    Check(OptIsMonotone(allOpts[t - 1], t, n), "monotone layer " + t);
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
