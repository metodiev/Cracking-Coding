// Task 418 - Implement online convex hull maintenance (dynamic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Online convex hull trick supporting BOTH signs and BOTH insertion orders:
//   * mode 'max' maintains the upper envelope, mode 'min' the lower envelope,
//   * order 'inc' inserts lines with non-decreasing slopes, order 'dec' with non-increasing
//     ones — both are the orders produced by sorting candidate lines by slope.
// The hull is kept as arrays (ms, bs) of lines whose breakpoints increase. When appending
// (m, b) after (m1, b1), (m2, b2) the middle line m2 is unnecessary exactly when
//     sign * [(b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1)] >= 0,
// where sign = +1 for (max, inc) / (min, dec) and -1 for (max, dec) / (min, inc); the
// condition compares the two intersection abscissas without ever dividing. Duplicate
// slopes keep only the best intercept (larger for max, smaller for min). Queries are
// unimodal over the hull, so an O(log n) adjacent-pair binary search finds the peak
// (for max) or the valley (for min). Deletions are out of scope: this is add-only.
// Everything is exact 64-bit integer arithmetic — no floating point intersections.
using System;
using System.Collections.Generic;
using System.Linq;

internal class ConvexHullTrick
{
    private readonly bool forMax;
    private readonly long sign;
    private readonly long[] ms = new long[1024];
    private readonly long[] bs = new long[1024];
    private int count;

    public ConvexHullTrick(bool forMax, bool increasing)
    {
        this.forMax = forMax;
        sign = (forMax ? 1 : -1) * (increasing ? 1 : -1);
    }

    public int Size() => count;

    public long MAt(int i) => ms[i];

    public long BAt(int i) => bs[i];

    public void AddLine(long m, long b)
    {
        if (count > 0 && m == ms[count - 1])
        {
            bool better = forMax ? b > bs[count - 1] : b < bs[count - 1];
            if (!better) return;
            count--;
        }
        while (count >= 2)
        {
            long m1 = ms[count - 2], b1 = bs[count - 2];
            long m2 = ms[count - 1], b2 = bs[count - 1];
            long expr = (b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1);
            if (sign * expr >= 0) count--;
            else break;
        }
        ms[count] = m;
        bs[count] = b;
        count++;
    }

    public long Query(long x)
    {
        if (count == 0) throw new InvalidOperationException("query on an empty hull");
        int lo = 0, hi = count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            long vMid = ms[mid] * x + bs[mid];
            long vNext = ms[mid + 1] * x + bs[mid + 1];
            if (forMax)
            {
                if (vMid <= vNext) lo = mid + 1;
                else hi = mid;
            }
            else
            {
                if (vMid >= vNext) lo = mid + 1;
                else hi = mid;
            }
        }
        return ms[lo] * x + bs[lo];
    }
}

public static class Solution
{
    private static int seed;

    private static int NextRand()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static long BruteBest(long[][] lines, long x, bool forMax)
    {
        long best = forMax ? long.MinValue : long.MaxValue;
        foreach (var line in lines)
        {
            long value = line[0] * x + line[1];
            best = forMax ? Math.Max(best, value) : Math.Min(best, value);
        }
        return best;
    }

    private static long[][] LcgLines(int count)
    {
        var lines = new long[count][];
        for (int k = 0; k < count; k++)
        {
            long m = NextRand() % 141 - 70;
            long b = NextRand() % 4001 - 2000;
            lines[k] = new long[] { m, b };
        }
        return lines;
    }

    private static void SortBySlope(long[][] lines, bool increasing)
    {
        Array.Sort(lines, (p, q) => increasing ? p[0].CompareTo(q[0]) : q[0].CompareTo(p[0]));
    }

    public static void Main()
    {
        // hand case: lines y = x, y = 2x + 3, y = 4x + 5 (max): 5 at 0, 45 at 10, -10 at -10
        var hull = new ConvexHullTrick(true, true);
        foreach (var line in new[] { new long[] { 1, 0 }, new long[] { 2, 3 }, new long[] { 4, 5 } })
            hull.AddLine(line[0], line[1]);
        Check(hull.Query(0) == 5, "hand 0");
        Check(hull.Query(10) == 45, "hand 10");
        Check(hull.Query(-10) == -10, "hand -10");
        // the same lines inserted from the other end of the slope order agree everywhere
        var dec = new ConvexHullTrick(true, false);
        foreach (var line in new[] { new long[] { 4, 5 }, new long[] { 2, 3 }, new long[] { 1, 0 } })
            dec.AddLine(line[0], line[1]);
        for (long x = -10; x <= 10; x++) Check(dec.Query(x) == hull.Query(x), "order agreement " + x);

        // hand case for min: y = 2x + 5, y = x + 1: 1 at 0, 11 at 10, -15 at -10
        var mn = new ConvexHullTrick(false, true);
        mn.AddLine(2, 5);
        mn.AddLine(1, 1);
        Check(mn.Query(0) == 1 && mn.Query(10) == 11 && mn.Query(-10) == -15, "min hand");
        var mnDec = new ConvexHullTrick(false, false);
        mnDec.AddLine(1, 1);
        mnDec.AddLine(2, 5);
        for (long x = -10; x <= 10; x++) Check(mnDec.Query(x) == mn.Query(x), "min order agreement " + x);

        // duplicate slopes keep only the best intercept: max keeps b = -1 (value 20 at x = 7)
        var dup = new ConvexHullTrick(true, true);
        for (int i = 0; i < 4; i++) dup.AddLine(3, -1);
        Check(dup.Query(7) == 20 && dup.Size() == 1, "dup max");
        var dup2 = new ConvexHullTrick(false, true);
        dup2.AddLine(3, 10);
        dup2.AddLine(3, 5);
        Check(dup2.Query(7) == 26 && dup2.Size() == 1, "dup min");
        var dup3 = new ConvexHullTrick(true, true);
        dup3.AddLine(3, -1);
        dup3.AddLine(3, -2);
        Check(dup3.Size() == 1 && dup3.Query(7) == 20, "dup worse discarded");

        // fixed-line oracle in all four mode/order combinations
        var fixedSrc = new[] { new long[] { 1, 0 }, new long[] { -1, 3 }, new long[] { 0, -2 }, new long[] { 5, -10 }, new long[] { -4, 6 } };
        foreach (bool forMax in new[] { true, false })
        {
            foreach (bool increasing in new[] { true, false })
            {
                var fixedLines = fixedSrc.Select(line => new long[] { line[0], line[1] }).ToArray();
                SortBySlope(fixedLines, increasing);
                var probe = new ConvexHullTrick(forMax, increasing);
                foreach (var line in fixedLines) probe.AddLine(line[0], line[1]);
                for (long x = -20; x <= 20; x++)
                {
                    Check(probe.Query(x) == BruteBest(fixedLines, x, forMax),
                          "fixed " + forMax + " " + increasing + " " + x);
                }
            }
        }

        // interleaved add/query scripts from the deterministic LCG, all four combinations
        int index = 0;
        foreach (bool forMax in new[] { true, false })
        {
            foreach (bool increasing in new[] { true, false })
            {
                seed = 4180 + index++;
                var raw = LcgLines(60);
                SortBySlope(raw, increasing);
                var h = new ConvexHullTrick(forMax, increasing);
                var added = new List<long[]>();
                for (int k = 0; k < raw.Length; k++)
                {
                    h.AddLine(raw[k][0], raw[k][1]);
                    added.Add(raw[k]);
                    Check(h.Size() <= added.Count, "hull size bound");
                    for (int t = 0; t < h.Size(); t++)
                    {
                        bool present = added.Any(line => line[0] == h.MAt(t) && line[1] == h.BAt(t));
                        Check(present, "hull line present");
                    }
                    if (k % 3 == 2 || k == raw.Length - 1)
                    {
                        for (int q = 0; q < 3; q++)
                        {
                            long x = NextRand() % 2001 - 1000;
                            Check(h.Query(x) == BruteBest(added.ToArray(), x, forMax),
                                  "script query " + index + " " + k + " " + x);
                        }
                    }
                }
                for (long x = -200; x <= 200; x++)
                {
                    Check(h.Query(x) == BruteBest(raw, x, forMax), "sweep " + index + " " + x);
                }
                var distinct = new HashSet<long>(raw.Select(line => line[0]));
                Check(h.Size() <= distinct.Count, "distinct slope bound");
            }
        }

        // querying before the first insertion is rejected
        bool threw = false;
        try
        {
            new ConvexHullTrick(true, true).Query(3);
        }
        catch (InvalidOperationException)
        {
            threw = true;
        }
        Check(threw, "empty hull rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
