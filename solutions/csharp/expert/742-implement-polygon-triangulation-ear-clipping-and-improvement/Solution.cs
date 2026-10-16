// Task 742 - Implement polygon triangulation (ear clipping) and improvement. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// Ear clipping for a simple polygon. The polygon is oriented counter-clockwise and cleaned of collinear
// vertices, then a doubly linked list of its vertices is walked: a vertex is an ear when its triangle turns
// left (strictly, so reflex and collinear vertices are never ears) and no other remaining vertex lies inside
// its triangle. Clipping an ear emits a triangle and unlinks the vertex, so every clip removes exactly one
// vertex and the scan resumes at the successor - the linked list keeps each pass O(n) and the whole clip O(n^2).
// The improvement stage is the classic Lawson edge flip: whenever a diagonal is shared by two triangles, the
// quadrilateral is convex and replacing the diagonal by the other one raises the minimum angle of the four
// candidate triangles, the flip is performed; the loop repeats until no flip helps, which leaves the
// triangulation at a local maximum of the minimum angle (the Delaunay triangulation restricted to the polygon).
// Invariants verified: exactly n-2 triangles, every polygon edge belongs to exactly one triangle and every
// diagonal to exactly two (so the triangles partition the polygon), the summed doubled area equals the exact
// integer shoelace area of the polygon, every triangle centroid lies inside the polygon, and after the
// improvement the exact integer in-circle predicate confirms the local Delaunay condition on every convex
// interior edge while the areas stay untouched. All predicates are exact integer arithmetic; only the angle
// comparison uses doubles.
// Simplification: the ear test is the O(n) point-in-triangle scan (no acceleration structure), self-touching or
// hole-carrying inputs are not supported, and a clip failure is reported as an exception rather than repaired.
using System;
using System.Collections.Generic;

internal sealed class SimplePolygon
{
    internal readonly List<(long X, long Y)> Points = new List<(long X, long Y)>();

    /// <summary>Orients the input counter-clockwise and drops collinear vertices (which keep the area).</summary>
    internal SimplePolygon(IReadOnlyList<(long X, long Y)> input)
    {
        var points = new List<(long X, long Y)>(input);
        if (Shoelace(points) < 0) points.Reverse();
        bool removed = true;
        while (removed && points.Count > 3)
        {
            removed = false;
            for (int i = 0; i < points.Count; i++)
            {
                if (ShoelaceCross(points[(i - 1 + points.Count) % points.Count], points[i], points[(i + 1) % points.Count]) == 0)
                {
                    points.RemoveAt(i);
                    removed = true;
                    break;
                }
            }
        }
        Points = points;
    }

    private static long ShoelaceCross((long X, long Y) a, (long X, long Y) b, (long X, long Y) c)
        => (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);

    internal int Count => Points.Count;
    internal (long X, long Y) this[int i] => Points[i];

    internal long Area2 => Math.Abs(Shoelace(Points));

    private static long Shoelace(List<(long X, long Y)> points)
    {
        long sum = 0;
        for (int i = 0; i < points.Count; i++)
        {
            var a = points[i];
            var b = points[(i + 1) % points.Count];
            sum += a.X * b.Y - b.X * a.Y;
        }
        return sum;
    }

    internal long Cross(int a, int b, int c)
        => (Points[b].X - Points[a].X) * (Points[c].Y - Points[a].Y)
         - (Points[b].Y - Points[a].Y) * (Points[c].X - Points[a].X);

    internal long Cross(int a, int b, long px, long py)
        => (Points[b].X - Points[a].X) * (py - Points[a].Y) - (Points[b].Y - Points[a].Y) * (px - Points[a].X);

    /// <summary>Exact crossing-number test; points on the boundary count as inside.</summary>
    internal bool Contains(long px, long py)
    {
        bool inside = false;
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            if (Cross(i, (i + 1) % Points.Count, px, py) == 0
                && px >= Math.Min(a.X, b.X) && px <= Math.Max(a.X, b.X)
                && py >= Math.Min(a.Y, b.Y) && py <= Math.Max(a.Y, b.Y)) return true;
            if ((a.Y > py) != (b.Y > py))
            {
                double x = a.X + (double)(b.X - a.X) * (py - a.Y) / (b.Y - a.Y);
                if (px < x) inside = !inside;
            }
        }
        return inside;
    }

    /// <summary>Strictly inside a double point: inside the polygon and at least 1e-9 away from every edge.</summary>
    internal bool StrictlyContains(double px, double py)
    {
        bool inside = false;
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            double ex = b.X - a.X, ey = b.Y - a.Y;
            double t = ((px - a.X) * ex + (py - a.Y) * ey) / (ex * ex + ey * ey);
            t = Math.Max(0, Math.Min(1, t));
            double dx = px - (a.X + ex * t), dy = py - (a.Y + ey * t);
            if (Math.Sqrt(dx * dx + dy * dy) < 1e-9) return false;      // on the edge, not strictly inside
            if ((a.Y > py) != (b.Y > py))
            {
                double x = a.X + ex * (py - a.Y) / ey;
                if (px < x) inside = !inside;
            }
        }
        return inside;
    }
}

internal static class EarClip
{
    private static bool IsEar(SimplePolygon polygon, int[] prev, int[] next, int a, int b, int c)
    {
        if (polygon.Cross(a, b, c) <= 0) return false;                 // reflex or collinear vertex
        long px = polygon[b].X, py = polygon[b].Y;
        for (int i = next[c]; i != a; i = next[i])
        {
            long qx = polygon[i].X, qy = polygon[i].Y;
            if (polygon.Cross(a, b, qx, qy) >= 0 && polygon.Cross(b, c, qx, qy) >= 0 && polygon.Cross(c, a, qx, qy) >= 0)
                return false;                                          // a remaining vertex blocks the ear
        }
        return true;
    }

    internal static List<(int A, int B, int C)> Triangulate(SimplePolygon polygon)
    {
        int n = polygon.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = (i - 1 + n) % n; next[i] = (i + 1) % n; }
        var triangles = new List<(int A, int B, int C)>();
        int remaining = n, current = 0, guard = 0;
        while (remaining > 3)
        {
            if (++guard > 4 * n * n) throw new InvalidOperationException("no ear found: the polygon is not simple");
            int a = prev[current], b = current, c = next[current];
            if (IsEar(polygon, prev, next, a, b, c))
            {
                triangles.Add((a, b, c));
                next[a] = c;
                prev[c] = a;
                remaining--;
                current = c;
            }
            else current = next[current];
        }
        triangles.Add((prev[current], current, next[current]));
        return triangles;
    }

    internal static Dictionary<(int, int), List<int>> EdgeMap(List<(int A, int B, int C)> triangles)
    {
        var map = new Dictionary<(int, int), List<int>>();
        for (int t = 0; t < triangles.Count; t++)
        {
            var (a, b, c) = triangles[t];
            Add(map, a, b, t);
            Add(map, b, c, t);
            Add(map, c, a, t);
        }
        return map;
    }

    private static void Add(Dictionary<(int, int), List<int>> map, int a, int b, int t)
    {
        var key = a < b ? (a, b) : (b, a);
        if (!map.TryGetValue(key, out List<int>? list)) { list = new List<int>(); map[key] = list; }
        list.Add(t);
    }
}

internal static class Quality
{
    internal static double MinAngle(SimplePolygon polygon, List<(int A, int B, int C)> triangles)
    {
        double best = double.MaxValue;
        foreach (var (a, b, c) in triangles) best = Math.Min(best, MinAngle(polygon, a, b, c));
        return best;
    }

    internal static double MinAngle(SimplePolygon polygon, (int A, int B, int C) triangle)
        => MinAngle(polygon, triangle.A, triangle.B, triangle.C);

    internal static double MinAngle(SimplePolygon polygon, int a, int b, int c)
    {
        double ab = Length(polygon, a, b), bc = Length(polygon, b, c), ca = Length(polygon, c, a);
        double alpha = Math.Acos(Clamp((ab * ab + ca * ca - bc * bc) / (2 * ab * ca)));
        double beta = Math.Acos(Clamp((ab * ab + bc * bc - ca * ca) / (2 * ab * bc)));
        return Math.Min(alpha, Math.Min(beta, Math.PI - alpha - beta));
    }

    private static double Clamp(double v) => Math.Max(-1.0, Math.Min(1.0, v));

    private static double Length(SimplePolygon polygon, int a, int b)
    {
        double dx = polygon[a].X - polygon[b].X, dy = polygon[a].Y - polygon[b].Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>Exact in-circle: &gt; 0 iff d is strictly inside the circumcircle of the counter-clockwise a,b,c.</summary>
    internal static long InCircle(SimplePolygon polygon, int a, int b, int c, int d)
    {
        long ax = polygon[a].X - polygon[d].X, ay = polygon[a].Y - polygon[d].Y;
        long bx = polygon[b].X - polygon[d].X, by = polygon[b].Y - polygon[d].Y;
        long cx = polygon[c].X - polygon[d].X, cy = polygon[c].Y - polygon[d].Y;
        return (ax * ax + ay * ay) * (bx * cy - by * cx)
             - (bx * bx + by * by) * (ax * cy - ay * cx)
             + (cx * cx + cy * cy) * (ax * by - ay * bx);
    }

    /// <summary>Lawson flips until the minimum angle stops improving; returns the number of flips.</summary>
    internal static long Improve(SimplePolygon polygon, List<(int A, int B, int C)> triangles)
    {
        long flips = 0;
        bool improved = true;
        while (improved)
        {
            improved = false;
            var map = EarClip.EdgeMap(triangles);
            var keys = new List<(int, int)>(map.Keys);
            keys.Sort((x, y) => x.Item1 != y.Item1 ? x.Item1.CompareTo(y.Item1) : x.Item2.CompareTo(y.Item2));
            foreach (var key in keys)
            {
                List<int> owners = map[key];
                if (owners.Count != 2) continue;                        // boundary edge or already flipped
                int t1 = owners[0], t2 = owners[1];
                int apex1 = Third(triangles[t1], key.Item1, key.Item2);
                int apex2 = Third(triangles[t2], key.Item1, key.Item2);
                if (!ConvexQuad(polygon, key.Item1, key.Item2, apex1, apex2)) continue;
                double before = Math.Min(MinAngle(polygon, triangles[t1]), MinAngle(polygon, triangles[t2]));
                int[] candidate1 = { key.Item1, apex1, apex2 };
                int[] candidate2 = { key.Item2, apex1, apex2 };
                double after = Math.Min(MinAngle(polygon, candidate1[0], candidate1[1], candidate1[2]),
                                        MinAngle(polygon, candidate2[0], candidate2[1], candidate2[2]));
                if (after <= before + 1e-12) continue;
                triangles[t1] = Ordered(polygon, candidate1[0], candidate1[1], candidate1[2]);
                triangles[t2] = Ordered(polygon, candidate2[0], candidate2[1], candidate2[2]);
                flips++;
                improved = true;
                break;
            }
        }
        return flips;
    }

    private static (int A, int B, int C) Ordered(SimplePolygon polygon, int a, int b, int c)
        => polygon.Cross(a, b, c) > 0 ? (a, b, c) : (a, c, b);

    private static bool ConvexQuad(SimplePolygon polygon, int a, int b, int apex1, int apex2)
        => (polygon.Cross(apex1, apex2, a) > 0) != (polygon.Cross(apex1, apex2, b) > 0);

    private static int Third((int A, int B, int C) triangle, int u, int v)
    {
        if (triangle.A != u && triangle.A != v) return triangle.A;
        if (triangle.B != u && triangle.B != v) return triangle.B;
        return triangle.C;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static SimplePolygon Comb() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (0, 30), (20, 30), (20, 10), (30, 10), (30, 30), (50, 30),
        (50, 10), (60, 10), (60, 30), (80, 30), (80, 0)
    });

    private static SimplePolygon ConvexPentagon() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (10, 0), (14, 8), (5, 14), (-4, 8)
    });

    private static SimplePolygon ThinComb() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (0, 40), (10, 40), (10, 5), (20, 5), (20, 40), (30, 40), (30, 5),
        (40, 5), (40, 40), (50, 40), (50, 0)
    });

    private static long SumArea2(SimplePolygon polygon, List<(int A, int B, int C)> triangles)
    {
        long total = 0;
        foreach (var (a, b, c) in triangles) total += Math.Abs(polygon.Cross(a, b, c));
        return total;
    }

    private static (double X, double Y) Centroid(SimplePolygon polygon, (int A, int B, int C) triangle)
        => ((polygon[triangle.A].X + polygon[triangle.B].X + polygon[triangle.C].X) / 3.0,
            (polygon[triangle.A].Y + polygon[triangle.B].Y + polygon[triangle.C].Y) / 3.0);

    private static string? EdgeMultiplicityViolation(SimplePolygon polygon, List<(int A, int B, int C)> triangles)
    {
        var map = EarClip.EdgeMap(triangles);
        var boundary = new HashSet<(int, int)>(BoundaryEdges(polygon));
        foreach (var (x, y) in boundary)
        {
            if (!map.TryGetValue((x, y), out List<int>? list)) return "boundary edge " + x + "-" + y + " is missing";
            if (list.Count != 1) return "boundary edge " + x + "-" + y + " is used by " + list.Count + " triangles";
        }
        foreach (var pair in map)
            if (!boundary.Contains(pair.Key) && pair.Value.Count != 2)
                return "diagonal " + pair.Key.Item1 + "-" + pair.Key.Item2 + " is used by " + pair.Value.Count + " triangles";
        return null;
    }

    private static IEnumerable<(int, int)> BoundaryEdges(SimplePolygon polygon)
    {
        for (int i = 0; i < polygon.Count; i++)
        {
            int a = i, b = (i + 1) % polygon.Count;
            yield return a < b ? (a, b) : (b, a);
        }
    }

    public static void Main()
    {
        // ---- degenerate polygons ---------------------------------------------------------------------
        var triangle = new SimplePolygon(new List<(long X, long Y)> { (0, 0), (10, 0), (0, 10) });
        var triangleTriangles = EarClip.Triangulate(triangle);
        Check(triangleTriangles.Count == 1 && SumArea2(triangle, triangleTriangles) == triangle.Area2,
            "a triangle triangulates to itself");
        Check(Quality.Improve(triangle, triangleTriangles) == 0, "a single triangle has no interior diagonal to flip");

        var clockwise = new SimplePolygon(new List<(long X, long Y)> { (0, 0), (0, 10), (10, 10), (10, 0) });
        Check(clockwise.Area2 == 200 && EarClip.Triangulate(clockwise).Count == 2,
            "a clockwise input is re-oriented and still yields n-2 triangles");

        var collinearEdges = new SimplePolygon(new List<(long X, long Y)>
        {
            (0, 0), (5, 0), (10, 0), (10, 10), (0, 10)
        });
        Check(collinearEdges.Count == 4 && EarClip.Triangulate(collinearEdges).Count == 2
            && SumArea2(collinearEdges, EarClip.Triangulate(collinearEdges)) == collinearEdges.Area2,
            "a collinear vertex on an edge is cleaned away, keeping the area and the triangle count");

        // ---- the concave comb: the real ear-clipping test ----------------------------------------------
        bool allGood = true, insideAll = true, multiplicitous = true;
        string? firstFailure = null;
        foreach (var polygon in new[] { Comb(), ThinComb(), ConvexPentagon() })
        {
            var triangles = EarClip.Triangulate(polygon);
            if (triangles.Count != polygon.Count - 2)
            {
                allGood = false;
                firstFailure ??= "triangle count " + triangles.Count + " != " + (polygon.Count - 2);
            }
            if (SumArea2(polygon, triangles) != polygon.Area2)
            {
                allGood = false;
                firstFailure ??= "area " + SumArea2(polygon, triangles) + " != " + polygon.Area2;
            }
            if (EdgeMultiplicityViolation(polygon, triangles) is string violation)
            {
                multiplicitous = false;
                firstFailure ??= violation;
            }
            var seenTriangles = new HashSet<(int, int, int)>();
            foreach (var t in triangles)
            {
                var ordered = t.A < t.B ? (t.A, t.B, t.C) : (t.B, t.A, t.C);
                if (!seenTriangles.Add(ordered))
                {
                    allGood = false;
                    firstFailure ??= "a triangle is repeated";
                }
                var (cx, cy) = Centroid(polygon, t);
                if (!polygon.StrictlyContains(cx, cy))
                {
                    insideAll = false;
                    firstFailure ??= "centroid (" + cx + "," + cy + ") of triangle " + t.A + "," + t.B + "," + t.C + " is not inside";
                }
            }
        }
        Check(allGood, "every polygon yields n-2 triangles with the exact polygon area: " + (firstFailure ?? "ok"));
        Check(multiplicitous, "each polygon edge belongs to one triangle and each diagonal to two");
        Check(insideAll, "every triangle centroid lies strictly inside the polygon: " + (firstFailure ?? "ok"));

        // ---- the improvement stage ---------------------------------------------------------------------
        var comb = Comb();
        var earOnly = EarClip.Triangulate(comb);
        double before = Quality.MinAngle(comb, earOnly);
        long flips = Quality.Improve(comb, earOnly);
        double after = Quality.MinAngle(comb, earOnly);
        Check(flips > 0 && after > before, "Lawson flips raised the minimum angle from "
            + Math.Round(before * 180 / Math.PI, 3) + " to " + Math.Round(after * 180 / Math.PI, 3)
            + " degrees in " + flips + " flips");
        Check(SumArea2(comb, earOnly) == comb.Area2 && earOnly.Count == comb.Count - 2,
            "flipping diagonals preserves the area and the triangle count");

        bool delaunay = true;
        var edgeMap = EarClip.EdgeMap(earOnly);
        foreach (var pair in edgeMap)
        {
            if (pair.Value.Count != 2) continue;
            var t1 = earOnly[pair.Value[0]];
            var t2 = earOnly[pair.Value[1]];
            int apex1 = t1.A != pair.Key.Item1 && t1.A != pair.Key.Item2 ? t1.A : (t1.B != pair.Key.Item1 && t1.B != pair.Key.Item2 ? t1.B : t1.C);
            int apex2 = t2.A != pair.Key.Item1 && t2.A != pair.Key.Item2 ? t2.A : (t2.B != pair.Key.Item1 && t2.B != pair.Key.Item2 ? t2.B : t2.C);
            bool convex = (comb.Cross(apex1, apex2, pair.Key.Item1) > 0) != (comb.Cross(apex1, apex2, pair.Key.Item2) > 0);
            if (!convex) continue;
            var (a, b, c) = comb.Cross(t1.A, t1.B, t1.C) > 0 ? t1 : (t1.A, t1.C, t1.B);
            int d = t2.A == a || t2.A == b || t2.A == c ? (t2.B == a || t2.B == b || t2.B == c ? t2.C : t2.B) : t2.A;
            if (Quality.InCircle(comb, a, b, c, d) > 0) delaunay = false;
        }
        Check(delaunay, "no convex interior edge violates the exact in-circle condition after the improvement");

        var repeat = Comb();
        var repeatTriangles = EarClip.Triangulate(repeat);
        Quality.Improve(repeat, repeatTriangles);
        Check(repeatTriangles.Count == earOnly.Count && Math.Abs(Quality.MinAngle(repeat, repeatTriangles) - after) < 1e-12,
            "the clipping and flipping are deterministic for the same polygon");

        Console.WriteLine("combVertices=" + comb.Count + " triangles=" + earOnly.Count
            + " area2=" + comb.Area2 + " flips=" + flips
            + " minAngleBefore=" + Math.Round(before * 180 / Math.PI, 4)
            + " minAngleAfter=" + Math.Round(after * 180 / Math.PI, 4));
        Console.WriteLine("PASS");
    }
}
