// Task 468 - Implement streaming quantiles (Greenwald-Khanna outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Greenwald-Khanna ε-approximate quantile summary (SIGMOD 2001), batch merge-compress
// variant: inserted values are buffered and every int(1/eps)+1 arrivals are merged
// into the summary entries {value, g, delta} with band floor(2*eps*(n-1)); a fresh
// value landing before an existing entry gets delta = entry.g + entry.delta - 1,
// otherwise neighbours are absorbed while g_left + g_right + delta_right <= band.
// QUANTILE(φ) accumulates g and returns the last entry with g_sum + delta within
// int(φ*(n-1)+1) + int(eps*(n-1)), falling back to the tracked minimum.
//
// Checks against the exact sorted oracle: for every returned value v,
//   rl = #{x < v} <= φ*N + ε*N   and   ru = #{x <= v} >= φ*N − ε*N,
// with eps = 0.05, N = 2000, φ in {0, 0.05, ..., 1.0} over six stream patterns
// (random, increasing, decreasing, constant, heavy duplicates, sawtooth); query(0)
// returns the true minimum and the value is always a stream member. Documented
// summary size bound (1/(2ε)+1)*(log2(2εn)+2) = 106 tuples, measured maximum 82.
using System;
using System.Collections.Generic;

internal class GreenwaldKhanna
{
    private readonly double eps;
    private readonly List<int[]> entries = new(); // {value, g, delta}
    private readonly List<int> pending = new();
    private int count;
    private int minimum;

    public GreenwaldKhanna(double eps)
    {
        this.eps = eps;
    }

    public int Size => entries.Count;

    public void Insert(int value)
    {
        pending.Add(value);
        count++;
        if (count == 1 || value < minimum) minimum = value;
        if (count % ((int)(1.0 / eps) + 1) == 0) Compress();
    }

    public void Compress()
    {
        int threshold = (int)(2.0 * eps * (count - 1));
        pending.Sort();
        var fresh = new List<int[]>();
        foreach (int value in pending) fresh.Add(new[] { value, 1, 0 });
        var old = entries;
        var outList = new List<int[]>();
        int i = 0, j = 0;
        int nf = fresh.Count, no = old.Count;
        while (i < nf || j < no)
        {
            if (i == nf)
            {
                int[] t = old[j];
                j++;
                if (j < no && t[1] + old[j][1] + old[j][2] <= threshold)
                {
                    old[j][1] += t[1];
                    continue;
                }
                outList.Add(t);
            }
            else if (j == no)
            {
                int[] t = fresh[i];
                i++;
                if (i < nf && t[1] + fresh[i][1] + fresh[i][2] <= threshold)
                {
                    fresh[i][1] += t[1];
                    continue;
                }
                outList.Add(t);
            }
            else if (fresh[i][0] < old[j][0])
            {
                int[] t = fresh[i];
                int[] u = old[j];
                if (t[1] + u[1] + u[2] <= threshold)
                {
                    u[1] += t[1];
                }
                else
                {
                    t[2] = u[1] + u[2] - t[1];
                    outList.Add(t);
                }
                i++;
            }
            else
            {
                int[] t = old[j];
                j++;
                if (j < no && t[1] + old[j][1] + old[j][2] <= threshold)
                {
                    old[j][1] += t[1];
                    continue;
                }
                outList.Add(t);
            }
        }
        entries.Clear();
        entries.AddRange(outList);
        pending.Clear();
    }

    public int? Query(double phi)
    {
        if (count == 0) return null;
        if (pending.Count > 0) Compress();
        int rank = (int)(phi * (count - 1) + 1);
        int spread = (int)(eps * (count - 1));
        int total = 0;
        int i = 0;
        int n = entries.Count;
        while (i < n)
        {
            total += entries[i][1];
            if (total + entries[i][2] > rank + spread) break;
            i++;
        }
        if (i == 0) return minimum;
        return entries[i - 1][0];
    }
}

public static class Solution
{
    private const double Eps = 0.05;
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    private static List<(string Name, int[] Data)> Streams()
    {
        seed = 20260116;
        var randomStream = new int[2000];
        for (int i = 0; i < 2000; i++) randomStream[i] = Lcg() % 1000;
        var dupes = new int[2000];
        for (int i = 0; i < 2000; i++) dupes[i] = Lcg() % 5;
        var increasing = new int[2000];
        var decreasing = new int[2000];
        var constantStream = new int[2000];
        var sawtooth = new int[2000];
        for (int i = 0; i < 2000; i++)
        {
            increasing[i] = i;
            decreasing[i] = 1999 - i;
            constantStream[i] = 42;
            sawtooth[i] = i % 50;
        }
        return new List<(string, int[])>
        {
            ("random", randomStream),
            ("increasing", increasing),
            ("decreasing", decreasing),
            ("constant", constantStream),
            ("dupes", dupes),
            ("sawtooth", sawtooth),
        };
    }

    private static int LowerBound(int[] sorted, int value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (sorted[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(int[] sorted, int value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (sorted[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void Main()
    {
        int maxSize = 0;
        foreach (var (name, stream) in Streams())
        {
            int n = stream.Length;
            var sketch = new GreenwaldKhanna(Eps);
            foreach (int value in stream) sketch.Insert(value);
            var exactly = (int[])stream.Clone();
            Array.Sort(exactly);
            for (int q = 0; q <= 20; q++)
            {
                double phi = q / 20.0;
                int value = sketch.Query(phi)!.Value;
                int lo = LowerBound(exactly, value);
                int hi = UpperBound(exactly, value);
                double target = phi * n;
                double slack = Eps * n;
                Check(lo <= target + slack + 1e-9, name + " upper rank " + phi);
                Check(hi >= target - slack - 1e-9, name + " lower rank " + phi);
            }
            Check(sketch.Query(0.0) == exactly[0], name + " min");
            maxSize = Math.Max(maxSize, sketch.Size); // after queries force compression
        }
        double bound = (1.0 / (2 * Eps) + 1) * (Math.Log2(2 * Eps * 2000) + 2);
        Check(maxSize <= bound, "size bound " + maxSize);
        Check(maxSize == 82, "measured max size " + maxSize);

        var empty = new GreenwaldKhanna(0.01);
        Check(empty.Query(0.5) == null, "empty");
        var tiny = new GreenwaldKhanna(0.01);
        foreach (int value in new[] { 5, 1, 9, 3, 7 }) tiny.Insert(value);
        Check(tiny.Query(0.0) == 1, "tiny min");
        Check(tiny.Query(1.0) == 9, "tiny max");
        int median = tiny.Query(0.5)!.Value;
        Check(median == 3 || median == 5, "tiny median");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
