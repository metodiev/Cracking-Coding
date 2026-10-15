// Task 396 - Implement the offline convex hull trick with divide-and-conquer DP optimization. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class MinHull
{
    private readonly List<long> ms = new List<long>();
    private readonly List<long> bs = new List<long>();

    /// <summary>Insert lines with NON-INCREASING slopes; query with non-decreasing x.</summary>
    public void AddLine(long m, long b)
    {
        if (ms.Count > 0 && m == ms[^1])
        {
            if (b >= bs[^1]) return;
            ms.RemoveAt(ms.Count - 1);
            bs.RemoveAt(bs.Count - 1);
        }
        while (ms.Count >= 2)
        {
            long m1 = ms[^2], b1 = bs[^2];
            long m2 = ms[^1], b2 = bs[^1];
            if ((b - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m))
            {
                ms.RemoveAt(ms.Count - 1);
                bs.RemoveAt(bs.Count - 1);
            }
            else break;
        }
        ms.Add(m);
        bs.Add(b);
    }

    public long Query(long x)
    {
        if (ms.Count == 0) throw new InvalidOperationException("query on an empty hull");
        int lo = 0, hi = ms.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (ms[mid] * x + bs[mid] <= ms[mid + 1] * x + bs[mid + 1]) hi = mid;
            else lo = mid + 1;
        }
        return ms[lo] * x + bs[lo];
    }
}

public static class Solution
{
    private const long INF = long.MaxValue / 4;

    public static long PartitionBruteforce(long[] xs, int k)
    {
        int n = xs.Length;
        if (k > n) return INF;
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
                    long span = xs[i - 1] - xs[j];
                    long cost = previous[j] + span * span;
                    if (cost < current[i]) current[i] = cost;
                }
            }
            previous = current;
        }
        return previous[n];
    }

    public static long PartitionOfflineCht(long[] xs, int k)
    {
        int n = xs.Length;
        if (k > n) return INF;
        var previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        for (int t = 1; t <= k; t++)
        {
            var current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            var hull = new MinHull();
            for (int i = t; i <= n; i++)
            {
                int j = i - 1;
                if (previous[j] != INF) hull.AddLine(-2 * xs[j], previous[j] + xs[j] * xs[j]);
                long x = xs[i - 1];
                current[i] = x * x + hull.Query(x);
            }
            previous = current;
        }
        return previous[n];
    }

    private static long[] dcPrevious = Array.Empty<long>();
    private static long[] dcCurrent = Array.Empty<long>();
    private static long[] dcXs = Array.Empty<long>();

    public static long PartitionDivideConquer(long[] xs, int k)
    {
        int n = xs.Length;
        if (k > n) return INF;
        var previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        for (int t = 1; t <= k; t++)
        {
            var current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            dcPrevious = previous;
            dcCurrent = current;
            dcXs = xs;
            current[t] = previous[t - 1];
            if (t + 1 <= n) SolveDc(t + 1, n, t - 1, n - 1);
            previous = current;
        }
        return previous[n];
    }

    private static void SolveDc(int l, int r, int optl, int optr)
    {
        if (l > r) return;
        int mid = (l + r) >> 1;
        long best = INF;
        int bestJ = -1;
        int upper = Math.Min(optr, mid - 1);
        for (int j = optl; j <= upper; j++)
        {
            if (dcPrevious[j] == INF) continue;
            long span = dcXs[mid - 1] - dcXs[j];
            long cost = dcPrevious[j] + span * span;
            if (cost < best)
            {
                best = cost;
                bestJ = j;
            }
        }
        dcCurrent[mid] = best;
        if (bestJ == -1) bestJ = optl;
        SolveDc(l, mid - 1, optl, bestJ);
        SolveDc(mid + 1, r, bestJ, optr);
    }

    private static int seed = 99;

    private static long[][] LcgArrays(int count, int length)
    {
        var arrays = new long[count][];
        for (int a = 0; a < count; a++)
        {
            var xs = new long[length];
            for (int i = 1; i < length; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                xs[i] = xs[i - 1] + 1 + seed % 20;
            }
            arrays[a] = xs;
        }
        return arrays;
    }

    public static void Main()
    {
        Check(PartitionBruteforce(new long[] { 1, 3, 5 }, 2) == 4, "brute 4");
        Check(PartitionBruteforce(new long[] { 1, 3, 5 }, 1) == 16, "brute 16");
        Check(PartitionBruteforce(new long[] { 1, 3, 5 }, 3) == 0, "brute 0");
        Check(PartitionBruteforce(new long[] { 1, 3 }, 2) == 0, "two points two segments");
        Check(PartitionBruteforce(new long[] { 1, 3 }, 3) == INF, "infeasible");
        Check(PartitionBruteforce(new long[] { 1 }, 1) == 0, "single point");

        Check(PartitionOfflineCht(new long[] { 1, 3, 5 }, 2) == 4, "cht hand");
        Check(PartitionDivideConquer(new long[] { 1, 3, 5 }, 2) == 4, "dc hand");

        var hull = new MinHull();
        foreach (var line in new[] { new long[] { 1, 0 }, new long[] { 0, 0 }, new long[] { -1, 0 } })
            hull.AddLine(line[0], line[1]);   // slopes non-increasing
        for (int x = -6; x <= 6; x++)
        {
            Check(hull.Query(x) == Math.Min(x, Math.Min(-x, 0)), "hull hand " + x);
        }

        foreach (var xs in LcgArrays(8, 40))
        {
            for (int k = 1; k <= 6; k++)
            {
                long brute = PartitionBruteforce(xs, k);
                Check(PartitionOfflineCht(xs, k) == brute, "cht random k=" + k);
                Check(PartitionDivideConquer(xs, k) == brute, "dc random k=" + k);
            }
        }

        bool threw = false;
        try { new MinHull().Query(3); }
        catch (InvalidOperationException) { threw = true; }
        Check(threw, "empty hull rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
