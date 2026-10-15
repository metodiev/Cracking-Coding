// Task 397 - Implement Knuth optimization (DP speedup demonstration). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private const long INF = long.MaxValue / 4;

    /// <summary>weights: n positive weights; returns the minimum weighted search cost.</summary>
    public static long ObstPlain(int[] weights)
    {
        int n = weights.Length;
        if (n == 0) return 0;
        var prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
        var e = new long[n + 2, n + 2];
        for (int i = 1; i <= n; i++) e[i, i] = weights[i - 1];
        for (int length = 2; length <= n; length++)
        {
            for (int i = 1; i + length - 1 <= n; i++)
            {
                int j = i + length - 1;
                long total = prefix[j] - prefix[i - 1];
                long best = INF;
                for (int k = i; k <= j; k++)
                {
                    long left = k - 1 >= i ? e[i, k - 1] : 0;
                    long right = k + 1 <= j ? e[k + 1, j] : 0;
                    best = Math.Min(best, left + right + total);
                }
                e[i, j] = best;
            }
        }
        return e[1, n];
    }

    /// <summary>Knuth-optimised DP; returns (cost, opt table).</summary>
    public static (long Cost, int[,] Opt) ObstKnuth(int[] weights)
    {
        int n = weights.Length;
        var e = new long[n + 2, n + 2];
        var opt = new int[n + 2, n + 2];
        if (n == 0) return (0, opt);
        var prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
        for (int i = 1; i <= n; i++)
        {
            e[i, i] = weights[i - 1];
            opt[i, i] = i;
        }
        for (int length = 2; length <= n; length++)
        {
            for (int i = 1; i + length - 1 <= n; i++)
            {
                int j = i + length - 1;
                long total = prefix[j] - prefix[i - 1];
                int lo = Math.Max(opt[i, j - 1], i);
                int hi = Math.Min(opt[i + 1, j], j);
                long best = INF;
                int bestK = lo;
                for (int k = lo; k <= hi; k++)
                {
                    long left = k - 1 >= i ? e[i, k - 1] : 0;
                    long right = k + 1 <= j ? e[k + 1, j] : 0;
                    long candidate = left + right + total;
                    if (candidate < best)
                    {
                        best = candidate;
                        bestK = k;
                    }
                }
                e[i, j] = best;
                opt[i, j] = bestK;
            }
        }
        return (e[1, n], opt);
    }

    public static bool OptIsMonotone(int[,] opt, int n)
    {
        for (int length = 2; length <= n; length++)
        {
            for (int i = 1; i + length - 1 <= n; i++)
            {
                int j = i + length - 1;
                if (opt[i, j] < opt[i, j - 1] || opt[i, j] > opt[i + 1, j]) return false;
            }
        }
        return true;
    }

    /// <summary>Rebuilds the tree from the opt table and re-scores it by walking the structure.</summary>
    public static long RebuildCost(int[] weights, int[,] opt, int i, int j, int depth)
    {
        if (i > j) return 0;
        int root = opt[i, j];
        return (long)weights[root - 1] * (depth + 1)
               + RebuildCost(weights, opt, i, root - 1, depth + 1)
               + RebuildCost(weights, opt, root + 1, j, depth + 1);
    }

    public static List<int> RebuildInorder(int[,] opt, int i, int j)
    {
        var outList = new List<int>();
        if (i > j) return outList;
        int root = opt[i, j];
        outList.AddRange(RebuildInorder(opt, i, root - 1));
        outList.Add(root);
        outList.AddRange(RebuildInorder(opt, root + 1, j));
        return outList;
    }

    private static int seed = 2026;

    private static int[][] LcgWeightSets(int count, int maxN)
    {
        var sets = new int[count][];
        for (int s = 0; s < count; s++)
        {
            int n = 2 + seed % (maxN - 1);
            var weights = new int[n];
            for (int i = 0; i < n; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                weights[i] = 1 + seed % 50;
            }
            sets[s] = weights;
        }
        return sets;
    }

    public static void Main()
    {
        Check(ObstPlain(new[] { 5 }) == 5, "single key");
        Check(ObstKnuth(new[] { 5 }).Cost == 5, "single key knuth");
        Check(ObstPlain(new[] { 1, 1 }) == 3, "two equal keys");
        Check(ObstPlain(new[] { 3, 2, 1 }) == 10, "three keys");
        Check(ObstKnuth(new[] { 3, 2, 1 }).Cost == 10, "three keys knuth");
        Check(ObstPlain(new[] { 1, 1, 1, 1 }) == 8, "four equal keys");
        Check(ObstKnuth(new[] { 1, 1, 1, 1 }).Cost == 8, "four equal keys knuth");

        foreach (var weights in LcgWeightSets(120, 40))
        {
            long plain = ObstPlain(weights);
            var (cost, opt) = ObstKnuth(weights);
            int n = weights.Length;
            Check(cost == plain, "random agreement");
            Check(OptIsMonotone(opt, n), "monotone opt");
            Check(RebuildCost(weights, opt, 1, n, 0) == cost, "rebuild scoring");
            Check(RebuildInorder(opt, 1, n).SequenceEqual(Enumerable.Range(1, n)), "inorder keys");
        }

        var sample = LcgWeightSets(1, 40)[0];
        int size = sample.Length;
        var sampleOpt = ObstKnuth(sample).Opt;
        long scanned = 0;
        for (int length = 2; length <= size; length++)
        {
            for (int i = 1; i + length - 1 <= size; i++)
            {
                int j = i + length - 1;
                scanned += sampleOpt[i, j] - sampleOpt[i, j - 1] + 1;
            }
        }
        Check(scanned < (long)size * size, "windows stay quadratic");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
