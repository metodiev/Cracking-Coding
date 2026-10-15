// Task 395 - Implement the dynamic convex hull trick for max queries (Li Chao tree). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Linq;

internal class LiChao
{
    private readonly int xLo, xHi;
    private readonly bool[] has;
    private readonly long[] m, b;

    public LiChao(int xLo, int xHi)
    {
        this.xLo = xLo;
        this.xHi = xHi;
        int size = 4 * (xHi - xLo + 1);
        has = new bool[size];
        m = new long[size];
        b = new long[size];
    }

    public void AddLine(long slope, long intercept) => Add(1, xLo, xHi, slope, intercept);

    private void Add(int node, int lo, int hi, long slope, long intercept)
    {
        if (!has[node])
        {
            has[node] = true;
            m[node] = slope;
            b[node] = intercept;
            return;
        }
        int mid = (lo + hi) >> 1;   // floor division; / truncates toward zero and breaks negative intervals
        if (slope * mid + intercept > m[node] * mid + b[node])
        {
            (m[node], slope) = (slope, m[node]);
            (b[node], intercept) = (intercept, b[node]);
        }
        if (lo == hi) return;
        if (slope * lo + intercept > m[node] * lo + b[node])
        {
            Add(2 * node, lo, mid, slope, intercept);
        }
        else if (slope * hi + intercept > m[node] * hi + b[node])
        {
            Add(2 * node + 1, mid + 1, hi, slope, intercept);
        }
    }

    public long Query(int x)
    {
        int node = 1, lo = xLo, hi = xHi;
        long? best = null;
        while (true)
        {
            if (has[node])
            {
                long value = m[node] * x + b[node];
                if (best == null || value > best) best = value;
            }
            if (lo == hi) break;
            int mid = (lo + hi) >> 1;   // floor division; / truncates toward zero and breaks negative intervals
            if (x <= mid) { node = 2 * node; hi = mid; }
            else { node = 2 * node + 1; lo = mid + 1; }
        }
        if (best == null) throw new InvalidOperationException("query on an empty tree");
        return best.Value;
    }
}

public static class Solution
{
    public static long BruteForce(long[][] lines, int x)
    {
        long best = long.MinValue;
        foreach (var line in lines) best = Math.Max(best, line[0] * x + line[1]);
        return best;
    }

    private static int seed = 4242;

    private static long[][] LcgLines(int count, int mRange, int bRange)
    {
        var lines = new long[count][];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            long slope = seed % (2 * mRange + 1) - mRange;
            seed = (seed * 75 + 74) % 65537;
            long intercept = seed % (2 * bRange + 1) - bRange;
            lines[k] = new[] { slope, intercept };
        }
        return lines;
    }

    public static void Main()
    {
        var tree = new LiChao(-500, 500);
        foreach (var line in new[] { new long[] { 1, 0 }, new long[] { -1, 0 }, new long[] { 0, 0 } })
            tree.AddLine(line[0], line[1]);
        Check(tree.Query(5) == 5, "v shape right");
        Check(tree.Query(-5) == 5, "v shape left");
        Check(tree.Query(0) == 0, "v shape middle");
        Check(tree.Query(3) == 3, "v shape 3");

        var cross = new LiChao(-500, 500);
        cross.AddLine(2, 1);
        cross.AddLine(-3, 10);
        Check(cross.Query(2) == 5, "crossing at 2");
        Check(cross.Query(1) == 7, "crossing at 1");
        Check(cross.Query(0) == 10, "crossing at 0");
        Check(cross.Query(5) == 11, "crossing at 5");

        var fixedLines = new[] { new long[] { 1, 0 }, new long[] { -1, 3 }, new long[] { 0, -2 }, new long[] { 5, -10 }, new long[] { -4, 6 } };
        var checker = new LiChao(-20, 20);
        foreach (var line in fixedLines) checker.AddLine(line[0], line[1]);
        for (int x = -20; x <= 20; x++)
        {
            Check(checker.Query(x) == BruteForce(fixedLines, x), "fixed line oracle " + x);
        }

        var lines = LcgLines(200, 50, 1000);
        var big = new LiChao(-500, 500);
        foreach (var line in lines) big.AddLine(line[0], line[1]);
        for (int step = 0; step <= 100; step++)
        {
            int x = -500 + step * 10;
            Check(big.Query(x) == BruteForce(lines, x), "random oracle " + x);
        }

        var duplicate = new LiChao(-10, 10);
        for (int i = 0; i < 4; i++) duplicate.AddLine(3, -1);
        Check(duplicate.Query(7) == 20, "duplicate lines");

        bool threw = false;
        try { new LiChao(-10, 10).Query(0); }
        catch (InvalidOperationException) { threw = true; }
        Check(threw, "empty tree rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
