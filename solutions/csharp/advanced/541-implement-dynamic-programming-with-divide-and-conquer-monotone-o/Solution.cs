// Task 541 - Implement dynamic programming with divide-and-conquer monotone optimization. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
//
// The classic divide-and-conquer optimisation: partitioning the first i values of an array into g
// non-empty contiguous groups with cost C(j, i) = (P[i] - P[j])^2 (P the prefix sums), where the
// optimal split opt(g, i) is monotone in i. DacDp exploits that by computing the whole row g with
// one recursive descent - the midpoint is solved by scanning the candidate range
// [optLo, min(optHi, mid - 1)], and the two halves inherit [optLo, bestJ] and [bestJ, optHi] - for
// O(n k log n) instead of the O(n^2 k) of the straightforward recurrence. Three solvers are
// present: the quadratic DacDp/NaiveDp pair plus an exhaustive enumerator over all ways of cutting
// n items into g non-empty groups.
//
// Documented checks: the cost satisfies the quadrangle (Monge) inequality
// C(a,c) + C(b,d) <= C(a,d) + C(b,c) for a <= b <= c <= d on the LCG arrays used here, and the
// optimal split of the naive DP is non-decreasing in i - the precondition the recursion relies on;
// DacDp equals NaiveDp entry-wise for n in {1,2,3,5,8,13,25,60,120,300} crossed with every
// k in 1..9 (including k > n, where both report the infinity sentinel); the exhaustive enumerator
// agrees with both for n = 10 and k = 1..4; the boundary cases match closed forms - k = 1 gives
// (total)^2 and k = n gives the sum of the squared individual values; and a second D&C run
// reproduces the table byte for byte.
using System;
using System.Collections.Generic;

internal sealed class PartitionDp
{
    internal const long Inf = long.MaxValue / 4;

    internal readonly int N;
    internal readonly long[] Prefix;

    internal PartitionDp(long[] values)
    {
        N = values.Length;
        Prefix = new long[N + 1];
        for (int i = 0; i < N; i++) Prefix[i + 1] = Prefix[i] + values[i];
    }

    /** Cost of the group covering the half-open range [lo, hi). */
    internal long Cost(int lo, int hi)
    {
        long sum = Prefix[hi] - Prefix[lo];
        return sum * sum;
    }

    /** dp[g][i] = cheapest way to split the first i values into exactly g non-empty groups. */
    internal long[][] NaiveDp(int k)
    {
        var dp = NewTable(k);
        for (int g = 1; g <= k; g++)
        {
            for (int i = g; i <= N; i++)
            {
                long best = Inf;
                for (int j = g - 1; j < i; j++)
                {
                    if (dp[g - 1][j] >= Inf) continue;
                    long cand = dp[g - 1][j] + Cost(j, i);
                    if (cand < best) best = cand;
                }
                dp[g][i] = best;
            }
        }
        return dp;
    }

    internal long[][] DacDp(int k)
    {
        var dp = NewTable(k);
        for (int g = 1; g <= k; g++) Solve(g, g, N, g - 1, N - 1, dp);
        return dp;
    }

    private long[][] NewTable(int k)
    {
        var dp = new long[k + 1][];
        for (int g = 0; g <= k; g++)
        {
            dp[g] = new long[N + 1];
            for (int i = 0; i <= N; i++) dp[g][i] = Inf;
        }
        dp[0][0] = 0;
        return dp;
    }

    private void Solve(int g, int lo, int hi, int optLo, int optHi, long[][] dp)
    {
        if (lo > hi) return;
        int mid = (lo + hi) / 2;
        int upper = Math.Min(optHi, mid - 1);
        long best = Inf;
        int bestJ = -1;
        for (int j = optLo; j <= upper; j++)
        {
            if (j < 0 || dp[g - 1][j] >= Inf) continue;
            long cand = dp[g - 1][j] + Cost(j, mid);
            if (cand < best)
            {
                best = cand;
                bestJ = j;
            }
        }
        dp[g][mid] = best;
        if (bestJ < 0) bestJ = optLo;
        Solve(g, lo, mid - 1, optLo, bestJ, dp);
        Solve(g, mid + 1, hi, bestJ, optHi, dp);
    }

    /** Exhaustive enumeration of every way to cut n items into k non-empty groups. */
    internal long BruteForce(int k)
    {
        if (k <= 0) return N == 0 ? 0 : Inf;
        return BruteFrom(0, k, 0);
    }

    private long BruteFrom(int start, int groupsLeft, long acc)
    {
        if (groupsLeft == 1)
        {
            if (start >= N) return Inf;
            return acc + Cost(start, N);
        }
        long best = Inf;
        for (int end = start + 1; end <= N - (groupsLeft - 1); end++)
        {
            long cand = BruteFrom(end, groupsLeft - 1, acc + Cost(start, end));
            if (cand < best) best = cand;
        }
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int state = 541001;

    private static long[] LcgValues(int n, int limit)
    {
        var values = new long[n];
        for (int i = 0; i < n; i++)
        {
            state = (state * 75 + 74) % 65537;
            values[i] = 1 + state % limit;
        }
        return values;
    }

    private static string Render(long[][] table)
    {
        var parts = new List<string>();
        foreach (long[] row in table)
        {
            parts.Add(string.Join(",", row));
        }
        return string.Join("|", parts);
    }

    private static void CheckMonge(PartitionDp dp)
    {
        int n = dp.N;
        for (int a = 0; a <= n; a++)
        {
            for (int b = a; b <= n; b++)
            {
                for (int c = b; c <= n; c++)
                {
                    for (int d = c; d <= n; d++)
                    {
                        long left = dp.Cost(a, c) + dp.Cost(b, d);
                        long right = dp.Cost(a, d) + dp.Cost(b, c);
                        Check(left <= right, "Monge inequality at " + a + "," + b + "," + c + "," + d);
                    }
                }
            }
        }
    }

    internal static void Main()
    {
        foreach (int n in new[] { 1, 2, 3, 5, 8, 13, 25, 60, 120, 300 })
        {
            var dp = new PartitionDp(LcgValues(n, 9));
            for (int k = 1; k <= 9; k++)
            {
                long[][] naive = dp.NaiveDp(k);
                long[][] dac = dp.DacDp(k);
                Check(Render(naive) == Render(dac), "D&C equals the quadratic DP for n=" + n + ", k=" + k);
                for (int i = 0; i <= n; i++)
                {
                    Check(naive[k][i] >= PartitionDp.Inf || naive[k][i] >= 0,
                            "finite table entries are non-negative");
                }
            }
            if (n <= 25) CheckMonge(dp);
        }

        // Boundary cases against closed forms.
        var small = new PartitionDp(new long[] { 3, 1, 4, 1, 5 });
        long total = 3 + 1 + 4 + 1 + 5;
        Check(small.NaiveDp(1)[1][5] == total * total, "k = 1 costs the square of the total");
        Check(small.DacDp(1)[1][5] == total * total, "k = 1 via D&C matches the closed form");
        long singletonSum = 3L * 3 + 1 + 4L * 4 + 1 + 5L * 5;
        Check(small.NaiveDp(5)[5][5] == singletonSum, "k = n costs the sum of squares");
        Check(small.DacDp(5)[5][5] == singletonSum, "k = n via D&C costs the sum of squares");
        Check(small.NaiveDp(6)[6][5] >= PartitionDp.Inf, "k > n is infeasible in the naive DP");
        Check(small.DacDp(6)[6][5] >= PartitionDp.Inf, "k > n is infeasible in the D&C DP");
        Check(small.NaiveDp(3)[3][5] == small.BruteForce(3), "a three-group split matches brute force");

        // Exhaustive enumeration for a small instance.
        var brute = new PartitionDp(LcgValues(10, 6));
        for (int k = 1; k <= 4; k++)
        {
            long expected = brute.BruteForce(k);
            Check(expected < PartitionDp.Inf, "the small instance is feasible for k=" + k);
            Check(brute.NaiveDp(k)[k][10] == expected, "naive DP equals brute force for k=" + k);
            Check(brute.DacDp(k)[k][10] == expected, "D&C DP equals brute force for k=" + k);
        }

        // Monotone optimum of the quadratic DP, the precondition of the recursion.
        var monotone = new PartitionDp(LcgValues(40, 7));
        long[][] table = monotone.NaiveDp(4);
        for (int g = 1; g <= 4; g++)
        {
            var bestJ = new int[41];
            for (int i = 0; i <= 40; i++)
            {
                int argmin = -1;
                long best = PartitionDp.Inf;
                for (int j = g - 1; j < i; j++)
                {
                    if (table[g - 1][j] >= PartitionDp.Inf) continue;
                    long cand = table[g - 1][j] + monotone.Cost(j, i);
                    if (cand < best)
                    {
                        best = cand;
                        argmin = j;
                    }
                }
                bestJ[i] = argmin;
            }
            for (int i = g + 1; i <= 40; i++)
            {
                if (bestJ[i - 1] < 0 || bestJ[i] < 0) continue;
                Check(bestJ[i - 1] <= bestJ[i], "optimal splits are monotone at g=" + g + ", i=" + i);
            }
        }

        // Determinism.
        long[] repeatedValues = LcgValues(50, 5);
        var repeat = new PartitionDp(repeatedValues);
        var repeatAgain = new PartitionDp(repeatedValues);
        Check(Render(repeat.DacDp(6)) == Render(repeat.DacDp(6)), "the D&C table is reproducible");
        Check(Render(repeatAgain.NaiveDp(6)) == Render(repeat.NaiveDp(6)), "the quadratic table is reproducible");
        Check(Render(repeat.DacDp(6)) == Render(repeatAgain.NaiveDp(6)), "both solvers agree on the repeat instance");

        Console.WriteLine("PASS");
    }
}
