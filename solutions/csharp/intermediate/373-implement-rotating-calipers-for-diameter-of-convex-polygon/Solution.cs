// Task 373 - Diameter of a convex polygon by rotating calipers. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;
using System.Collections.Generic;

/// <summary>
/// The diameter of a point set is its farthest pair; the farthest pair of a set always
/// lies on the convex hull, so the problem reduces to the diameter of a convex polygon.
/// Rotating calipers walk an antipodal vertex j along the hull for each directed edge
/// (i, i+1): j advances while the triangle area (i, i+1, j+1) exceeds (i, i+1, j).  Every
/// diameter pair is antipodal to some edge, so checking the distance from both edge
/// endpoints to j for all i finds the exact maximum.  All arithmetic is integer: the
/// result is the exact SQUARED distance; a float square root is taken only to compare
/// against the classic square = side*sqrt(2) expectation (relative tolerance 1e-12, the
/// standard double-sqrt tolerance).
///
/// The hull is built by Andrew's monotone chain (duplicate points removed, collinear
/// points dropped).  Degenerate cases are documented: empty/one point -> diameter 0; two
/// distinct points or a fully collinear set -> hull is the two extreme points.
/// Cross-checked against the brute-force O(h^2) scan over the hull.
///
/// Independent oracle: brute force over ALL input points (not just the hull) - the
/// farthest pair of any point set is a hull pair, so all three numbers must agree.
/// </summary>
public static class Solution
{
    private sealed class Lcg
    {
        private int seed;

        public Lcg(int seed)
        {
            this.seed = seed;
        }

        public int Next(int modulus)
        {
            seed = (seed * 75 + 74) % 65537;
            return seed % modulus;
        }
    }

    /// <summary>Andrew's monotone chain: counter-clockwise hull without collinear points.</summary>
    public static int[][] ConvexHull(int[][] points)
    {
        var unique = new SortedSet<long>();
        foreach (int[] p in points) unique.Add(((long)p[0] << 32) ^ (uint)p[1]);
        var sorted = new List<int[]>();
        foreach (long key in unique) sorted.Add(new[] { (int)(key >> 32), (int)(key & 0xFFFFFFFFL) });
        if (sorted.Count <= 2) return sorted.ToArray();

        var lower = new List<int[]>();
        foreach (int[] p in sorted)
        {
            while (lower.Count >= 2 && Cross(lower[lower.Count - 2], lower[lower.Count - 1], p) <= 0)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(p);
        }
        var upper = new List<int[]>();
        for (int i = sorted.Count - 1; i >= 0; i--)
        {
            int[] p = sorted[i];
            while (upper.Count >= 2 && Cross(upper[upper.Count - 2], upper[upper.Count - 1], p) <= 0)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(p);
        }
        var hull = new List<int[]>();
        for (int i = 0; i + 1 < lower.Count; i++) hull.Add(lower[i]);
        for (int i = 0; i + 1 < upper.Count; i++) hull.Add(upper[i]);
        return hull.ToArray();
    }

    private static long Cross(int[] o, int[] a, int[] b)
    {
        return (long)(a[0] - o[0]) * (b[1] - o[1]) - (long)(a[1] - o[1]) * (b[0] - o[0]);
    }

    private static long DistSquared(int[] a, int[] b)
    {
        long dx = a[0] - b[0], dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    private static long Area2(int[] a, int[] b, int[] c)
    {
        return (long)(b[0] - a[0]) * (c[1] - a[1]) - (long)(b[1] - a[1]) * (c[0] - a[0]);
    }

    /// <summary>Rotating calipers over a CCW convex hull; exact integer result.</summary>
    public static long HullDiameterSquared(int[][] hull)
    {
        int h = hull.Length;
        if (h <= 1) return 0;
        if (h == 2) return DistSquared(hull[0], hull[1]);
        long best = 0;
        int j = 1;
        for (int i = 0; i < h; i++)
        {
            int ni = (i + 1) % h;
            int steps = 0;
            while (steps <= h)                                 // guard against degenerate ties
            {
                int nj = (j + 1) % h;
                if (Area2(hull[i], hull[ni], hull[nj]) > Area2(hull[i], hull[ni], hull[j]))
                {
                    j = nj;
                    steps++;
                }
                else
                {
                    break;
                }
            }
            best = Math.Max(best, DistSquared(hull[i], hull[j]));
            best = Math.Max(best, DistSquared(hull[ni], hull[j]));
        }
        return best;
    }

    /// <summary>O(h^2) oracle over the hull vertices.</summary>
    public static long BruteHullDiameterSquared(int[][] hull)
    {
        long best = 0;
        for (int i = 0; i < hull.Length; i++)
        {
            for (int k = i + 1; k < hull.Length; k++) best = Math.Max(best, DistSquared(hull[i], hull[k]));
        }
        return best;
    }

    /// <summary>O(n^2) oracle over every input point.</summary>
    public static long BruteAllPointsDiameterSquared(int[][] points)
    {
        long best = 0;
        for (int i = 0; i < points.Length; i++)
        {
            for (int k = i + 1; k < points.Length; k++) best = Math.Max(best, DistSquared(points[i], points[k]));
        }
        return best;
    }

    private static int[][] LcgPoints(int count, Lcg lcg, int extent)
    {
        var points = new int[count][];
        for (int i = 0; i < count; i++) points[i] = new[] { lcg.Next(extent), lcg.Next(extent) };
        return points;
    }

    public static void Main()
    {
        // classic square: side 3, diagonal 3*sqrt(2)
        int side = 3;
        int[][] square = { new[] { 0, 0 }, new[] { side, 0 }, new[] { side, side }, new[] { 0, side } };
        int[][] hull = ConvexHull(square);
        long diameter2 = HullDiameterSquared(hull);
        Check(diameter2 == 2L * side * side && diameter2 == 18, "square diameter^2 = 18");
        Check(hull.Length == 4, "square hull has 4 vertices");
        Check(Math.Abs(Math.Sqrt(diameter2) - side * Math.Sqrt(2)) <= 1e-12 * side * Math.Sqrt(2),
            "square diagonal is side*sqrt(2)");

        // hand-checked small cases
        Check(HullDiameterSquared(ConvexHull(new[] { new[] { 7, 7 } })) == 0, "single point");
        Check(HullDiameterSquared(ConvexHull(new[] { new[] { 1, 1 }, new[] { 4, 5 } })) == 25, "two points");
        int[][] triangle = { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 0, 3 } };
        hull = ConvexHull(triangle);
        Check(hull.Length == 3 && HullDiameterSquared(hull) == 25, "3-4-5 triangle");
        Check(HullDiameterSquared(hull) == BruteHullDiameterSquared(hull), "triangle vs oracle");
        int[][] rectangle = { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 2, 5 }, new[] { 0, 5 } };
        Check(HullDiameterSquared(ConvexHull(rectangle)) == 29, "rectangle diagonal");
        int[][] collinear = { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 3, 3 } };
        hull = ConvexHull(collinear);
        Check(hull.Length == 2 && hull[0][0] == 0 && hull[1][0] == 3, "collinear hull degenerates");
        Check(HullDiameterSquared(hull) == 18, "collinear diameter");
        int[][] withInterior = { new[] { 0, 0 }, new[] { 3, 0 }, new[] { 3, 3 }, new[] { 0, 3 }, new[] { 1, 1 } };
        hull = ConvexHull(withInterior);
        Check(hull.Length == 4 && HullDiameterSquared(hull) == 18, "interior point ignored");

        // LCG cross-checks: calipers == brute force on the hull == brute force on all points
        var lcg = new Lcg(37313);
        for (int trial = 0; trial < 60; trial++)
        {
            int count = 3 + lcg.Next(10);
            int[][] points = LcgPoints(count, lcg, 50);
            hull = ConvexHull(points);
            long calipers = HullDiameterSquared(hull);
            Check(calipers == BruteHullDiameterSquared(hull), "calipers vs hull oracle " + trial);
            Check(calipers == BruteAllPointsDiameterSquared(points), "calipers vs all-points oracle " + trial);
        }

        // larger deterministic set for a wider sweep
        int[][] bigPoints = LcgPoints(120, lcg, 100);
        hull = ConvexHull(bigPoints);
        Check(hull.Length >= 3, "large hull size");
        Check(HullDiameterSquared(hull) == BruteAllPointsDiameterSquared(bigPoints), "large set diameter");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
