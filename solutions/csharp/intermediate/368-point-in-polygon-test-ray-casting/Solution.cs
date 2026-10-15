// Task 368 - Point-in-polygon test (ray casting). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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
    private static readonly Lcg Random = new Lcg(31415);

    public static long CrossAt(int[] o, int[] a, int[] b) =>
        (long)(a[0] - o[0]) * (b[1] - o[1]) - (long)(a[1] - o[1]) * (b[0] - o[0]);

    /// <summary>Exact test for p lying on the closed segment a-b.</summary>
    public static bool OnSegment(int[] a, int[] b, int[] p)
    {
        if (CrossAt(a, b, p) != 0) return false;
        return Math.Min(a[0], b[0]) <= p[0] && p[0] <= Math.Max(a[0], b[0]) &&
            Math.Min(a[1], b[1]) <= p[1] && p[1] <= Math.Max(a[1], b[1]);
    }

    public static bool OnBoundary(int[][] polygon, int[] p)
    {
        for (int i = 0; i < polygon.Length; i++)
        {
            if (OnSegment(polygon[i], polygon[(i + 1) % polygon.Length], p)) return true;
        }
        return false;
    }

    /// <summary>
    /// Even-odd ray casting with the half-open edge rule: an edge counts only when exactly
    /// one endpoint has y > p.y, so horizontal edges never cross and a vertex shared by two
    /// edges is counted once.  The intersection test is exact integer arithmetic: instead of
    /// computing x_intersection, it compares (x1-px)*dy + (py-y1)*(x2-x1) against zero with
    /// the sign of dy.
    /// </summary>
    public static bool RayCastInside(int[][] polygon, int[] p)
    {
        int px = p[0], py = p[1];
        bool inside = false;
        for (int i = 0; i < polygon.Length; i++)
        {
            int x1 = polygon[i][0], y1 = polygon[i][1];
            int x2 = polygon[(i + 1) % polygon.Length][0], y2 = polygon[(i + 1) % polygon.Length][1];
            if ((y1 > py) == (y2 > py)) continue;            // ignore horizontal edges
            long dy = y2 - y1;
            long lhs = (long)(x1 - px) * dy + (long)(py - y1) * (x2 - x1);
            if ((lhs > 0) == (dy > 0)) inside = !inside;
        }
        return inside;
    }

    /// <summary>Winding number alternative: +1 per CCW crossing above p, -1 per clockwise one.</summary>
    public static int WindingNumber(int[][] polygon, int[] p)
    {
        int px = p[0], py = p[1];
        int winding = 0;
        for (int i = 0; i < polygon.Length; i++)
        {
            int[] a = polygon[i];
            int[] b = polygon[(i + 1) % polygon.Length];
            if (a[1] <= py)
            {
                if (b[1] > py && CrossAt(a, b, p) > 0) winding++;
            }
            else if (b[1] <= py && CrossAt(a, b, p) < 0)
            {
                winding--;
            }
        }
        return winding;
    }

    /// <summary>Convex hull (CCW) used to build test polygons from LCG points.</summary>
    public static int[][] MonotoneChain(int[][] points)
    {
        var copy = points.Select(p => (int[])p.Clone()).ToList();
        copy.Sort((a, b) => a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
        var pts = new List<int[]>();
        foreach (var p in copy)
        {
            if (pts.Count == 0 || pts[pts.Count - 1][0] != p[0] || pts[pts.Count - 1][1] != p[1])
            {
                pts.Add(p);
            }
        }
        if (pts.Count <= 2) return pts.ToArray();
        var lower = new List<int[]>();
        foreach (var p in pts)
        {
            while (lower.Count >= 2 && CrossAt(lower[lower.Count - 2], lower[lower.Count - 1], p) <= 0)
            {
                lower.RemoveAt(lower.Count - 1);
            }
            lower.Add(p);
        }
        var upper = new List<int[]>();
        for (int i = pts.Count - 1; i >= 0; i--)
        {
            int[] p = pts[i];
            while (upper.Count >= 2 && CrossAt(upper[upper.Count - 2], upper[upper.Count - 1], p) <= 0)
            {
                upper.RemoveAt(upper.Count - 1);
            }
            upper.Add(p);
        }
        var hull = lower.GetRange(0, lower.Count - 1);
        hull.AddRange(upper.GetRange(0, upper.Count - 1));
        return hull.ToArray();
    }

    public static void Main()
    {
        var square = new[] { new[] { 0, 0 }, new[] { 4, 0 }, new[] { 4, 4 }, new[] { 0, 4 } };
        Check(!OnBoundary(square, new[] { 2, 2 }) && RayCastInside(square, new[] { 2, 2 }), "square inside");
        Check(!OnBoundary(square, new[] { 5, 2 }) && !RayCastInside(square, new[] { 5, 2 }), "square right outside");
        Check(!RayCastInside(square, new[] { -1, 2 }), "square left outside");
        Check(!RayCastInside(square, new[] { 2, 5 }), "square above outside");
        Check(OnBoundary(square, new[] { 4, 4 }), "square vertex boundary");
        Check(OnBoundary(square, new[] { 0, 2 }), "square left edge boundary");
        Check(OnBoundary(square, new[] { 2, 0 }), "square bottom edge boundary");
        Check(OnBoundary(square, new[] { 0, 0 }), "square corner boundary");
        // (-1, 0): the horizontal ray passes straight through vertices (0,0) and (4,0);
        // the half-open rule counts both vertical edges exactly once -> outside.
        Check(!RayCastInside(square, new[] { -1, 0 }) && WindingNumber(square, new[] { -1, 0 }) == 0,
            "ray through two vertices");

        var triangle = new[] { new[] { 0, 0 }, new[] { 6, 0 }, new[] { 0, 6 } };
        Check(!OnBoundary(triangle, new[] { 1, 1 }) && RayCastInside(triangle, new[] { 1, 1 }),
            "triangle inside");
        Check(OnBoundary(triangle, new[] { 3, 3 }), "triangle hypotenuse boundary");
        Check(!RayCastInside(triangle, new[] { 5, 2 }), "triangle outside above hypotenuse");
        Check(!RayCastInside(triangle, new[] { 7, 0 }), "triangle outside right of x = 6");
        Check(OnBoundary(triangle, new[] { 0, 3 }), "triangle left edge boundary");

        // Concave "arrow" polygon: a square with a triangular notch cut from the top.
        var arrow = new[] { new[] { 0, 0 }, new[] { 6, 0 }, new[] { 6, 6 }, new[] { 3, 3 }, new[] { 0, 6 } };
        Check(!OnBoundary(arrow, new[] { 3, 1 }) && RayCastInside(arrow, new[] { 3, 1 }), "arrow inside bottom");
        Check(!RayCastInside(arrow, new[] { 3, 4 }), "arrow notch outside");
        Check(!OnBoundary(arrow, new[] { 1, 4 }) && RayCastInside(arrow, new[] { 1, 4 }), "arrow inside left");
        Check(!OnBoundary(arrow, new[] { 5, 4 }) && RayCastInside(arrow, new[] { 5, 4 }), "arrow inside right");
        Check(OnBoundary(arrow, new[] { 3, 3 }), "arrow notch vertex boundary");
        Check(OnBoundary(arrow, new[] { 5, 5 }), "arrow on edge y = x");
        Check(!RayCastInside(arrow, new[] { 2, 6 }), "arrow above notch outside");
        Check(OnBoundary(arrow, new[] { 0, 3 }), "arrow left edge boundary");
        // (1, 3): the ray y = 3 passes through the notch vertex (3, 3) and exits at x = 6;
        // the vertex contributes two crossings that cancel, then one real crossing -> inside.
        Check(!OnBoundary(arrow, new[] { 1, 3 }), "point (1,3) not on boundary");
        Check(RayCastInside(arrow, new[] { 1, 3 }), "ray through notch vertex");
        Check(WindingNumber(arrow, new[] { 1, 3 }) == 1, "winding number of (1,3)");

        // Both algorithms must agree everywhere except on the boundary.
        var polygons = new[] { square, triangle, arrow };
        foreach (var polygon in polygons)
        {
            for (int x = -2; x <= 9; x++)
            {
                for (int y = -2; y <= 9; y++)
                {
                    var p = new[] { x, y };
                    if (OnBoundary(polygon, p)) continue;
                    Check(RayCastInside(polygon, p) == (WindingNumber(polygon, p) != 0),
                        "agreement at " + x + "," + y);
                }
            }
        }

        // LCG convex polygons from hulls: ray casting and winding number must agree.
        for (int trial = 0; trial < 30; trial++)
        {
            int count = 3 + Random.Next(20);
            var points = new int[count][];
            for (int i = 0; i < count; i++)
            {
                points[i] = new[] { Random.Next(12), Random.Next(12) };
            }
            int[][] polygon = MonotoneChain(points);
            if (polygon.Length < 3) continue;
            for (int x = -2; x <= 14; x++)
            {
                for (int y = -2; y <= 14; y++)
                {
                    var p = new[] { x, y };
                    if (OnBoundary(polygon, p)) continue;    // boundary is documented as undefined
                    Check(RayCastInside(polygon, p) == (WindingNumber(polygon, p) != 0),
                        "LCG agreement trial " + trial);
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
