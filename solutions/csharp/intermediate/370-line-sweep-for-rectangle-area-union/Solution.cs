// Task 370 - Line sweep for rectangle area union. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537.</summary>
internal sealed class Lcg
{
    private int seed;

    public Lcg(int seed) => this.seed = seed;

    public int Next(int modulus)
    {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/// <summary>
/// Segment tree over compressed y edges.  Count == how many active rectangles fully cover
/// the node, Length == covered length of its span.  Counts (not booleans) are what make
/// overlaps of three or more rectangles and exact touching edges work.
/// </summary>
internal sealed class CoverLengthTree
{
    private readonly int[] ys;
    private readonly int m;
    private readonly int[] count;
    private readonly int[] length;

    public CoverLengthTree(int[] sortedYs)
    {
        ys = sortedYs;
        m = sortedYs.Length - 1;
        count = new int[4 * Math.Max(m, 1)];
        length = new int[4 * Math.Max(m, 1)];
    }

    public void RangeAdd(int yLo, int yHi, int delta)
    {
        if (m == 0 || yLo >= yHi) return;
        Update(1, 0, m, yLo, yHi, delta);
    }

    private void Update(int node, int lo, int hi, int ql, int qr, int delta)
    {
        if (qr <= lo || hi <= ql) return;
        if (ql <= lo && hi <= qr)
        {
            count[node] += delta;
        }
        else
        {
            int mid = (lo + hi) / 2;
            Update(2 * node, lo, mid, ql, qr, delta);
            Update(2 * node + 1, mid, hi, ql, qr, delta);
        }
        if (count[node] > 0)
        {
            length[node] = ys[hi] - ys[lo];
        }
        else if (hi - lo == 1)
        {
            length[node] = 0;
        }
        else
        {
            length[node] = length[2 * node] + length[2 * node + 1];
        }
    }

    public int Measure() => m == 0 ? 0 : length[1];
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(1618);

    /// <summary>Area of the union of axis-aligned rectangles [x1, y1, x2, y2].</summary>
    public static long UnionArea(int[][] rectangles)
    {
        var events = new List<int[]>();
        var ySet = new SortedSet<int>();
        foreach (var r in rectangles)
        {
            if (r[0] >= r[2] || r[1] >= r[3]) continue;      // zero/negative area: ignored
            events.Add(new[] { r[0], 1, r[1], r[3] });
            events.Add(new[] { r[2], -1, r[1], r[3] });
            ySet.Add(r[1]);
            ySet.Add(r[3]);
        }
        if (events.Count == 0) return 0;
        int[] ys = ySet.ToArray();
        var index = new Dictionary<int, int>();
        for (int i = 0; i < ys.Length; i++) index[ys[i]] = i;
        events.Sort((a, b) => a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
        var tree = new CoverLengthTree(ys);
        long area = 0;
        int prevX = events[0][0];
        int k = 0;
        while (k < events.Count)
        {
            int x = events[k][0];
            area += (long)tree.Measure() * (x - prevX);
            while (k < events.Count && events[k][0] == x)
            {
                var evt = events[k];
                tree.RangeAdd(index[evt[2]], index[evt[3]], evt[1]);
                k++;
            }
            prevX = x;
        }
        return area;
    }

    /// <summary>Independent oracle: sum the fully covered cells of the compressed coordinate grid.</summary>
    public static long UnionAreaOracle(int[][] rectangles)
    {
        var rects = rectangles.Where(r => r[0] < r[2] && r[1] < r[3]).ToArray();
        if (rects.Length == 0) return 0;
        int[] xs = rects.SelectMany(r => new[] { r[0], r[2] }).Distinct().OrderBy(v => v).ToArray();
        int[] ys = rects.SelectMany(r => new[] { r[1], r[3] }).Distinct().OrderBy(v => v).ToArray();
        long area = 0;
        for (int i = 0; i + 1 < xs.Length; i++)
        {
            for (int j = 0; j + 1 < ys.Length; j++)
            {
                foreach (var r in rects)
                {
                    if (r[0] <= xs[i] && r[2] >= xs[i + 1] && r[1] <= ys[j] && r[3] >= ys[j + 1])
                    {
                        area += (long)(xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]);
                        break;
                    }
                }
            }
        }
        return area;
    }

    public static void Main()
    {
        // Canonical example: [0,0,2,2] and [1,1,3,3] overlap in a 1x1 square -> 7.
        var canonical = new[] { new[] { 0, 0, 2, 2 }, new[] { 1, 1, 3, 3 } };
        Check(UnionArea(canonical) == 7, "canonical 7");
        Check(UnionAreaOracle(canonical) == 7, "canonical oracle 7");

        Check(UnionArea(new[] { new[] { 0, 0, 1, 1 }, new[] { 2, 2, 3, 3 } }) == 2, "disjoint");
        Check(UnionArea(new[] { new[] { 0, 0, 10, 10 }, new[] { 2, 2, 4, 4 } }) == 100, "nested");
        Check(UnionArea(new[] { new[] { 1, 1, 3, 3 }, new[] { 1, 1, 3, 3 } }) == 4, "identical");
        Check(UnionArea(new[] { new[] { 0, 0, 2, 2 }, new[] { 2, 0, 4, 2 } }) == 8, "touching edge");
        Check(UnionArea(new[] { new[] { 0, 0, 1, 1 }, new[] { 1, 1, 2, 2 } }) == 2, "touching corner");
        Check(UnionArea(new[] { new[] { 5, 5, 5, 9 } }) == 0, "zero width");
        Check(UnionArea(new[] { new[] { 5, 5, 5, 5 } }) == 0, "point rectangle");
        Check(UnionArea(new[] { new[] { 0, 0, 3, 3 }, new[] { 1, 1, 1, 4 } }) == 9, "zero-area among valid");
        Check(UnionArea(Array.Empty<int[]>()) == 0, "empty");
        Check(UnionArea(new[] { new[] { 2, 3, 7, 9 } }) == 30, "single rectangle");
        Check(UnionArea(new[] { new[] { 0, 0, 4, 4 }, new[] { 0, 0, 4, 4 }, new[] { 1, 1, 3, 3 } }) == 16,
            "repeats");

        // Triple overlap: y in [0,1) spans x 0..4 (length 4), y in [1,3) also 4, y in [3,4) 3.
        var triple = new[] { new[] { 0, 0, 3, 3 }, new[] { 1, 0, 4, 3 }, new[] { 0, 1, 3, 4 } };
        Check(UnionArea(triple) == 15, "triple union 15");
        Check(UnionAreaOracle(triple) == 15, "triple union oracle 15");

        // LCG cross-check, including degenerate rectangles and negative coordinates.
        for (int trial = 0; trial < 40; trial++)
        {
            int count = 1 + Random.Next(8);
            int modulus = 8 + Random.Next(10);
            var rectangles = new int[count][];
            for (int i = 0; i < count; i++)
            {
                int x1 = Random.Next(modulus);
                int width = Random.Next(modulus / 2);
                int y1 = Random.Next(modulus);
                int height = Random.Next(modulus / 2);
                rectangles[i] = new[] { x1, y1, x1 + width, y1 + height };
            }
            int offset = -Random.Next(7);
            var shifted = rectangles.Select(r => new[] { r[0] + offset, r[1] + offset,
                r[2] + offset, r[3] + offset }).ToArray();
            Check(UnionArea(shifted) == UnionAreaOracle(shifted), "random trial " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
