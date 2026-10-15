// Task 367 - Compute convex hull (Andrew monotone chain). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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

public static class Solution
{
    private static readonly Lcg Random = new Lcg(90210);

    public static long Cross(int[] o, int[] a, int[] b) =>
        (long)(a[0] - o[0]) * (b[1] - o[1]) - (long)(a[1] - o[1]) * (b[0] - o[0]);

    private static long Dist2(int[] a, int[] b)
    {
        long dx = a[0] - b[0];
        long dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    private static List<int[]> UniqueSortedPoints(int[][] points)
    {
        var copy = points.Select(p => (int[])p.Clone()).ToList();
        copy.Sort((a, b) => a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
        var unique = new List<int[]>();
        foreach (var p in copy)
        {
            if (unique.Count == 0 || unique[unique.Count - 1][0] != p[0] ||
                    unique[unique.Count - 1][1] != p[1])
            {
                unique.Add(p);
            }
        }
        return unique;
    }

    /// <summary>Convex hull (CCW, from the smallest point); collinear points dropped.</summary>
    public static int[][] MonotoneChain(int[][] points)
    {
        var pts = UniqueSortedPoints(points);
        if (pts.Count <= 2) return pts.ToArray();
        var lower = new List<int[]>();
        foreach (var p in pts)
        {
            while (lower.Count >= 2 && Cross(lower[lower.Count - 2], lower[lower.Count - 1], p) <= 0)
            {
                lower.RemoveAt(lower.Count - 1);
            }
            lower.Add(p);
        }
        var upper = new List<int[]>();
        for (int i = pts.Count - 1; i >= 0; i--)
        {
            int[] p = pts[i];
            while (upper.Count >= 2 && Cross(upper[upper.Count - 2], upper[upper.Count - 1], p) <= 0)
            {
                upper.RemoveAt(upper.Count - 1);
            }
            upper.Add(p);
        }
        var hull = lower.GetRange(0, lower.Count - 1);
        hull.AddRange(upper.GetRange(0, upper.Count - 1));
        return hull.ToArray();
    }

    /// <summary>The task 366 Graham scan, kept here so both hulls can be compared directly.</summary>
    public static int[][] GrahamScan(int[][] points)
    {
        var pts = UniqueSortedPoints(points);
        if (pts.Count <= 2) return pts.ToArray();
        int[] pivot = pts[0];
        var rest = pts.Skip(1).ToList();
        rest.Sort((a, b) =>
        {
            long turn = Cross(pivot, a, b);
            if (turn > 0) return -1;
            if (turn < 0) return 1;
            return Dist2(pivot, a).CompareTo(Dist2(pivot, b));
        });
        var hull = new List<int[]> { pivot };
        foreach (var point in rest)
        {
            while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], point) <= 0)
            {
                hull.RemoveAt(hull.Count - 1);
            }
            hull.Add(point);
        }
        return hull.ToArray();
    }

    public static long SignedArea2(int[][] hull)
    {
        long total = 0;
        for (int i = 0; i < hull.Length; i++)
        {
            int[] a = hull[i];
            int[] b = hull[(i + 1) % hull.Length];
            total += (long)a[0] * b[1] - (long)b[0] * a[1];
        }
        return total;
    }

    /// <summary>Counter-clockwise order starting at the lexicographically smallest point.</summary>
    public static int[][] Normalize(int[][] hull)
    {
        if (hull.Length <= 1) return (int[][])hull.Clone();
        var pts = hull.Select(p => (int[])p.Clone()).ToList();
        if (SignedArea2(hull) < 0) pts.Reverse();
        int start = 0;
        for (int i = 1; i < pts.Count; i++)
        {
            if (pts[i][0] < pts[start][0] ||
                    (pts[i][0] == pts[start][0] && pts[i][1] < pts[start][1]))
            {
                start = i;
            }
        }
        var rotated = pts.GetRange(start, pts.Count - start);
        rotated.AddRange(pts.GetRange(0, start));
        return rotated.ToArray();
    }

    private static bool SameHull(int[][] a, int[][] b) =>
        a.Length == b.Length && a.Zip(b).All(pair =>
            pair.First[0] == pair.Second[0] && pair.First[1] == pair.Second[1]);

    public static void Main()
    {
        // Same hand-checked sets as task 366; both algorithms must return the same hull.
        var cases = new (string Name, int[][] Points, int[][] Expected)[]
        {
            ("square", new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } },
                new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } }),
            ("inside", new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 }, new[] { 1, 1 } },
                new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } }),
            ("collinear", new[] { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 2, 0 }, new[] { 3, 0 } },
                new[] { new[] { 0, 0 }, new[] { 3, 0 } }),
            ("mixed", new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 2, 2 }, new[] { 3, 3 }, new[] { 0, 3 } },
                new[] { new[] { 0, 0 }, new[] { 3, 3 }, new[] { 0, 3 } }),
            ("duplicates", new[] { new[] { 0, 0 }, new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 0 },
                new[] { 2, 2 }, new[] { 0, 2 }, new[] { 2, 2 } },
                new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } }),
            ("triangle", new[] { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 2, 3 } },
                new[] { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 2, 3 } }),
        };
        foreach (var testCase in cases)
        {
            Check(SameHull(MonotoneChain(testCase.Points), testCase.Expected),
                testCase.Name + " monotone chain");
            Check(SameHull(GrahamScan(testCase.Points), testCase.Expected),
                testCase.Name + " graham scan");
            Check(SameHull(Normalize(MonotoneChain(testCase.Points)),
                Normalize(GrahamScan(testCase.Points))), testCase.Name);
        }
        Check(SameHull(Normalize(MonotoneChain(new[] { new[] { 5, 7 } })),
            Normalize(GrahamScan(new[] { new[] { 5, 7 } }))), "single");
        Check(SameHull(Normalize(MonotoneChain(new[] { new[] { 0, 0 }, new[] { 1, 1 } })),
            Normalize(GrahamScan(new[] { new[] { 0, 0 }, new[] { 1, 1 } }))), "pair");
        Check(MonotoneChain(Array.Empty<int[]>()).Length == 0, "empty");
        Check(SignedArea2(MonotoneChain(new[] { new[] { 0, 0 }, new[] { 2, 0 },
            new[] { 2, 2 }, new[] { 0, 2 } })) == 8, "square twice-area 8");

        // Normalization really canonicalizes: reversed clockwise input flips back.
        var clockwise = new[] { new[] { 0, 0 }, new[] { 0, 2 }, new[] { 2, 2 }, new[] { 2, 0 } };
        var canonical = new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } };
        Check(SignedArea2(clockwise) < 0, "clockwise signed area negative");
        Check(SameHull(Normalize(clockwise), canonical), "normalize flips clockwise");
        Check(SameHull(Normalize(new[] { new[] { 2, 2 }, new[] { 0, 2 },
            new[] { 0, 0 }, new[] { 2, 0 } }), canonical), "normalize rotates to smallest");

        // LCG cross-check: monotone chain and Graham scan agree after normalization.
        for (int trial = 0; trial < 80; trial++)
        {
            int count = 1 + Random.Next(45);
            int modulus = 5 + Random.Next(25);       // small ranges force collinearity
            var points = new int[count][];
            for (int i = 0; i < count; i++)
            {
                points[i] = new[] { Random.Next(modulus), Random.Next(modulus) };
            }
            int[][] chain = Normalize(MonotoneChain(points));
            int[][] scan = Normalize(GrahamScan(points));
            Check(SameHull(chain, scan), "random trial " + trial);
            Check(SignedArea2(chain) >= 0, "random CCW trial " + trial);
            if (chain.Length > 2)
            {
                for (int i = 0; i < chain.Length; i++)
                {
                    int[] a = chain[(i + chain.Length - 1) % chain.Length];
                    int[] b = chain[i];
                    int[] c = chain[(i + 1) % chain.Length];
                    Check(Cross(a, b, c) > 0, "convex trial " + trial);
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
