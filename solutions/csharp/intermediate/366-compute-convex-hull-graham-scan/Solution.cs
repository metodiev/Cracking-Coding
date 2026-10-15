// Task 366 - Compute convex hull (Graham scan). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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
    private static readonly Lcg Random = new Lcg(15731);

    /// <summary>Cross product (a - o) x (b - o); > 0 means o -> a -> b turns left.</summary>
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

    /// <summary>Convex hull, counter-clockwise, starting at the lexicographically smallest point.</summary>
    public static int[][] GrahamScan(int[][] points)
    {
        var pts = UniqueSortedPoints(points);
        if (pts.Count <= 2) return pts.ToArray();
        int[] pivot = pts[0];
        var rest = pts.Skip(1).ToList();
        rest.Sort((a, b) =>
        {
            // Equal angles keep the nearest point first; collinear hull-edge points are
            // later removed by the cross <= 0 pops.
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

    /// <summary>Independent gift wrapping oracle with the same conventions as GrahamScan.</summary>
    public static int[][] JarvisMarch(int[][] points)
    {
        var pts = UniqueSortedPoints(points);
        if (pts.Count <= 2) return pts.ToArray();
        int[] start = pts[0];
        var hull = new List<int[]> { start };
        int[] current = start;
        while (true)
        {
            int[] candidate = null;
            foreach (var p in pts)
            {
                if (ReferenceEquals(p, current)) continue;
                if (candidate == null)
                {
                    candidate = p;
                    continue;
                }
                long turn = Cross(current, candidate, p);
                if (turn < 0)
                {
                    candidate = p;
                }
                else if (turn == 0)
                {
                    bool sameDirection = (long)(p[0] - current[0]) * (candidate[0] - current[0]) +
                        (long)(p[1] - current[1]) * (candidate[1] - current[1]) > 0;
                    if (sameDirection && Dist2(current, p) > Dist2(current, candidate))
                    {
                        candidate = p;                 // keep the farthest collinear point
                    }
                }
            }
            if (ReferenceEquals(candidate, start)) break;
            hull.Add(candidate);
            current = candidate;
            if (hull.Count > pts.Count) throw new Exception("degenerate input");
        }
        return hull.ToArray();
    }

    /// <summary>Twice the signed area (shoelace); positive for counter-clockwise order.</summary>
    public static long HullArea2(int[][] hull)
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

    private static bool SameHull(int[][] a, int[][] b) =>
        a.Length == b.Length && a.Zip(b).All(pair =>
            pair.First[0] == pair.Second[0] && pair.First[1] == pair.Second[1]);

    private static string Show(int[][] hull) =>
        string.Join("", hull.Select(p => "[" + p[0] + "," + p[1] + "]"));

    public static void Main()
    {
        var square = new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } };
        var squareHull = new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 } };
        Check(SameHull(GrahamScan(square), squareHull), "square");
        Check(SameHull(JarvisMarch(square), squareHull), "square jarvis");
        Check(HullArea2(GrahamScan(square)) == 8, "square twice-area 8");

        var inside = new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 2 }, new[] { 0, 2 }, new[] { 1, 1 } };
        Check(SameHull(GrahamScan(inside), squareHull), "interior point excluded");
        Check(SameHull(JarvisMarch(inside), squareHull), "interior point excluded jarvis");

        var collinear = new[] { new[] { 0, 0 }, new[] { 1, 0 }, new[] { 2, 0 }, new[] { 3, 0 } };
        Check(SameHull(GrahamScan(collinear), new[] { new[] { 0, 0 }, new[] { 3, 0 } }), "collinear run");
        Check(SameHull(JarvisMarch(collinear), new[] { new[] { 0, 0 }, new[] { 3, 0 } }), "collinear jarvis");

        var mixed = new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 2, 2 }, new[] { 3, 3 }, new[] { 0, 3 } };
        Check(SameHull(GrahamScan(mixed), new[] { new[] { 0, 0 }, new[] { 3, 3 }, new[] { 0, 3 } }),
            "diagonal run");
        Check(SameHull(JarvisMarch(mixed), GrahamScan(mixed)), "diagonal jarvis");

        var duplicates = new[] { new[] { 0, 0 }, new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 0 },
            new[] { 2, 2 }, new[] { 0, 2 }, new[] { 2, 2 } };
        Check(SameHull(GrahamScan(duplicates), squareHull), "duplicates");
        Check(SameHull(JarvisMarch(duplicates), squareHull), "duplicates jarvis");

        Check(SameHull(GrahamScan(new[] { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 2, 3 } }),
            new[] { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 2, 3 } }), "triangle");
        Check(SameHull(GrahamScan(new[] { new[] { 5, 7 } }), new[] { new[] { 5, 7 } }), "single point");
        Check(SameHull(GrahamScan(new[] { new[] { 0, 0 }, new[] { 1, 1 } }),
            new[] { new[] { 0, 0 }, new[] { 1, 1 } }), "two points");
        Check(GrahamScan(Array.Empty<int[]>()).Length == 0, "no points");

        // LCG cross-check: hull size, doubled area and the exact point list vs Jarvis.
        for (int trial = 0; trial < 60; trial++)
        {
            int count = 1 + Random.Next(40);
            int modulus = 6 + Random.Next(20);       // small ranges force collinearity
            var points = new int[count][];
            for (int i = 0; i < count; i++)
            {
                points[i] = new[] { Random.Next(modulus), Random.Next(modulus) };
            }
            int[][] scan = GrahamScan(points);
            int[][] march = JarvisMarch(points);
            Check(scan.Length == march.Length, "random size trial " + trial);
            Check(HullArea2(scan) == HullArea2(march), "random area trial " + trial);
            Check(SameHull(scan, march), "random hull trial " + trial);
            if (scan.Length > 2)
            {
                for (int i = 0; i < scan.Length; i++)
                {
                    int[] a = scan[(i + scan.Length - 1) % scan.Length];
                    int[] b = scan[i];
                    int[] c = scan[(i + 1) % scan.Length];
                    Check(Cross(a, b, c) > 0, "convex at " + Show(scan) + " from " + Show(points));
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
