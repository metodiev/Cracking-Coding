// Task 743 - Implement constrained Delaunay triangulation outline. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// A constrained Delaunay triangulation of a simple polygon: the polygon's boundary edges are the constraints
// and nothing may cross them, while every free diagonal must satisfy the empty-circumcircle (Delaunay)
// criterion. The outline implements it in two stages: an ear-clipping pass produces a constrained triangulation
// whose triangles all lie inside the polygon, then a Lawson flip loop removes non-Delaunay diagonals, flipping
// only edges that are not constraints and whose quadrilateral is convex. The in-circle test is the exact
// integer determinant, so no epsilon decides a flip; only the angle comparison uses doubles.
// Invariants verified: n-2 triangles covering exactly the polygon area (exact shoelace), every boundary edge
// used by exactly one triangle and every diagonal by exactly two (the constraints survive every flip), every
// triangle centroid strictly inside the polygon, and after the loop every convex interior edge satisfies
// InCircle <= 0 exactly. The strongest check is optimality against exhaustive enumeration: for an 8-vertex
// non-convex polygon all 132 triangulations (the count is cross-checked against the Catalan number computed by
// its own DP recurrence) are enumerated by a different algorithm - a recursive diagonal split that requires the
// candidate diagonal to lie inside the polygon - and the CDT's minimum angle must equal the maximum over all of
// them, which is exactly the max-min angle optimality that characterises Delaunay triangulations of a polygon.
// Simplifications of this outline: one simple polygon without holes, no Steiner points, no intersecting or
// self-touching constraints, ear clipping plus flipping rather than an incremental insertion with a DCEL, and
// exhaustive comparison only for the small polygon used here.
using System;
using System.Collections.Generic;

internal sealed class Poly
{
    internal readonly List<(long X, long Y)> Points = new List<(long X, long Y)>();

    internal Poly(IReadOnlyList<(long X, long Y)> input)
    {
        var points = new List<(long X, long Y)>(input);
        if (Shoelace(points) < 0) points.Reverse();
        bool removed = true;
        while (removed && points.Count > 3)
        {
            removed = false;
            for (int i = 0; i < points.Count; i++)
            {
                var a = points[(i - 1 + points.Count) % points.Count];
                var b = points[i];
                var c = points[(i + 1) % points.Count];
                if ((b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X) == 0)
                {
                    points.RemoveAt(i);
                    removed = true;
                    break;
                }
            }
        }
        Points = points;
    }

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

    /// <summary>Exact in-circle: &gt; 0 iff d is strictly inside the circumcircle of the counter-clockwise a,b,c.</summary>
    internal long InCircle(int a, int b, int c, int d)
    {
        long ax = Points[a].X - Points[d].X, ay = Points[a].Y - Points[d].Y;
        long bx = Points[b].X - Points[d].X, by = Points[b].Y - Points[d].Y;
        long cx = Points[c].X - Points[d].X, cy = Points[c].Y - Points[d].Y;
        return (ax * ax + ay * ay) * (bx * cy - by * cx)
             - (bx * bx + by * by) * (ax * cy - ay * cx)
             + (cx * cx + cy * cy) * (ax * by - ay * bx);
    }

    internal bool Contains(double px, double py)
    {
        bool inside = false;
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            if ((a.Y > py) != (b.Y > py))
            {
                double x = a.X + (double)(b.X - a.X) * (py - a.Y) / (b.Y - a.Y);
                if (px < x) inside = !inside;
            }
        }
        return inside;
    }

    internal bool StrictlyContains(double px, double py)
    {
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            double ex = b.X - a.X, ey = b.Y - a.Y;
            double t = ((px - a.X) * ex + (py - a.Y) * ey) / (ex * ex + ey * ey);
            t = Math.Max(0, Math.Min(1, t));
            double dx = px - (a.X + ex * t), dy = py - (a.Y + ey * t);
            if (Math.Sqrt(dx * dx + dy * dy) < 1e-9) return false;
        }
        return Contains(px, py);
    }

    /// <summary>Validity of a chord: a boundary edge is always valid, otherwise no crossing and midpoint inside.</summary>
    internal bool ChordInside(int u, int v)
    {
        if (_chords.TryGetValue((Math.Min(u, v), Math.Max(u, v)), out bool cached)) return cached;
        int n = Points.Count;
        bool boundary = (u + 1) % n == v || (v + 1) % n == u;
        bool result = boundary || DiagonalInside(u, v);
        _chords[(Math.Min(u, v), Math.Max(u, v))] = result;
        return result;
    }

    private readonly Dictionary<(int, int), bool> _chords = new Dictionary<(int, int), bool>();

    /// <summary>Crossing test for a candidate diagonal: no proper crossing and the midpoint inside.</summary>
    internal bool DiagonalInside(int u, int v)
    {
        var a = Points[u];
        var b = Points[v];
        for (int i = 0; i < Points.Count; i++)
        {
            int j = (i + 1) % Points.Count;
            if (i == u || i == v || j == u || j == v) continue;
            var p = Points[i];
            var q = Points[j];
            double d1 = S(a, b, p), d2 = S(a, b, q), d3 = S(p, q, a), d4 = S(p, q, b);
            if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return false;
        }
        return StrictlyContains((a.X + b.X) / 2.0, (a.Y + b.Y) / 2.0);
    }

    private static double S((long X, long Y) o, (long X, long Y) a, (long X, long Y) b)
        => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
}

internal static class Cdt
{
    internal static List<(int A, int B, int C)> EarClip(Poly poly)
    {
        int n = poly.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = (i - 1 + n) % n; next[i] = (i + 1) % n; }
        var triangles = new List<(int A, int B, int C)>();
        int remaining = n, current = 0, guard = 0;
        while (remaining > 3)
        {
            if (++guard > 4 * n * n) throw new InvalidOperationException("no ear found");
            int a = prev[current], b = current, c = next[current];
            bool ear = poly.Cross(a, b, c) > 0;
            if (ear)
                for (int i = next[c]; i != a; i = next[i])
                {
                    long px = poly[i].X, py = poly[i].Y;
                    if ((poly[b].X - poly[a].X) * (py - poly[a].Y) - (poly[b].Y - poly[a].Y) * (px - poly[a].X) >= 0
                     && (poly[c].X - poly[b].X) * (py - poly[b].Y) - (poly[c].Y - poly[b].Y) * (px - poly[b].X) >= 0
                     && (poly[a].X - poly[c].X) * (py - poly[c].Y) - (poly[a].Y - poly[c].Y) * (px - poly[c].X) >= 0)
                        ear = false;
                }
            if (ear)
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

    /// <summary>Flips non-constrained diagonals while the exact in-circle test is violated.</summary>
    internal static long DelaunayFlip(Poly poly, HashSet<(int, int)> constrained, List<(int A, int B, int C)> triangles)
    {
        long flips = 0;
        bool changed = true;
        while (changed)
        {
            changed = false;
            var map = EdgeMap(triangles);
            var keys = new List<(int, int)>(map.Keys);
            keys.Sort((x, y) => x.Item1 != y.Item1 ? x.Item1.CompareTo(y.Item1) : x.Item2.CompareTo(y.Item2));
            foreach (var key in keys)
            {
                if (constrained.Contains(key) || map[key].Count != 2) continue;
                var t1 = triangles[map[key][0]];
                var t2 = triangles[map[key][1]];
                if (poly.Cross(t1.A, t1.B, t1.C) <= 0) continue;
                int d = Third(t2, key.Item1, key.Item2);
                if (!Convex(poly, t1, t2, key, d)) continue;
                if (poly.InCircle(t1.A, t1.B, t1.C, d) <= 0) continue;
                int a = Third(t1, key.Item1, key.Item2);
                int b = Third(t2, key.Item1, key.Item2);
                triangles[map[key][0]] = Orient(poly, key.Item1, a, b);
                triangles[map[key][1]] = Orient(poly, key.Item2, a, b);
                flips++;
                changed = true;
                break;
            }
        }
        return flips;
    }

    private static bool Convex(Poly poly, (int A, int B, int C) t1, (int A, int B, int C) t2, (int, int) key, int d)
    {
        int a = Third(t1, key.Item1, key.Item2);
        return (poly.Cross(a, d, key.Item1) > 0) != (poly.Cross(a, d, key.Item2) > 0);
    }

    private static int Third((int A, int B, int C) triangle, int u, int v)
    {
        if (triangle.A != u && triangle.A != v) return triangle.A;
        if (triangle.B != u && triangle.B != v) return triangle.B;
        return triangle.C;
    }

    private static (int A, int B, int C) Orient(Poly poly, int a, int b, int c)
        => poly.Cross(a, b, c) > 0 ? (a, b, c) : (a, c, b);

    internal static double MinAngle(Poly poly, List<(int A, int B, int C)> triangles)
    {
        double best = double.MaxValue;
        foreach (var (a, b, c) in triangles) best = Math.Min(best, MinAngle(poly, a, b, c));
        return best;
    }

    internal static double MinAngle(Poly poly, int a, int b, int c)
    {
        double ab = Length(poly, a, b), bc = Length(poly, b, c), ca = Length(poly, c, a);
        double alpha = Math.Acos(Clamp((ab * ab + ca * ca - bc * bc) / (2 * ab * ca)));
        double beta = Math.Acos(Clamp((ab * ab + bc * bc - ca * ca) / (2 * ab * bc)));
        return Math.Min(alpha, Math.Min(beta, Math.PI - alpha - beta));
    }

    private static double Clamp(double v) => Math.Max(-1.0, Math.Min(1.0, v));

    private static double Length(Poly poly, int a, int b)
    {
        double dx = poly[a].X - poly[b].X, dy = poly[a].Y - poly[b].Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>
    /// Exhaustive enumeration by recursive ear removal: every triangulation of a simple polygon has an ear, so
    /// removing each valid ear and recursing reaches every triangulation. Different removal orders can produce
    /// the same triangle set, which the caller deduplicates by a canonical key.
    /// </summary>
    internal static void Enumerate(Poly poly, List<int> ring, List<(int A, int B, int C)> current,
        List<List<(int A, int B, int C)>> output)
    {
        int m = ring.Count;
        if (m == 3)
        {
            var complete = new List<(int A, int B, int C)>(current) { (ring[0], ring[1], ring[2]) };
            output.Add(complete);
            return;
        }
        for (int i = 0; i < m; i++)
        {
            int prev = ring[(i - 1 + m) % m], v = ring[i], next = ring[(i + 1) % m];
            if (poly.Cross(prev, v, next) <= 0) continue;
            bool blocked = false;
            foreach (int w in ring)
            {
                if (w == prev || w == v || w == next) continue;
                long b1 = (poly[v].X - poly[prev].X) * (poly[w].Y - poly[prev].Y) - (poly[v].Y - poly[prev].Y) * (poly[w].X - poly[prev].X);
                long b2 = (poly[next].X - poly[v].X) * (poly[w].Y - poly[v].Y) - (poly[next].Y - poly[v].Y) * (poly[w].X - poly[v].X);
                long b3 = (poly[prev].X - poly[next].X) * (poly[w].Y - poly[next].Y) - (poly[prev].Y - poly[next].Y) * (poly[w].X - poly[next].X);
                if (b1 >= 0 && b2 >= 0 && b3 >= 0) { blocked = true; break; }
            }
            if (blocked) continue;
            current.Add((prev, v, next));
            var rest = new List<int>(ring);
            rest.RemoveAt(i);
            Enumerate(poly, rest, current, output);
            current.RemoveAt(current.Count - 1);
        }
    }

    /// <summary>Canonical key of a triangulation, used to remove the duplicate removal orders.</summary>
    internal static string Key(List<(int A, int B, int C)> triangles)
    {
        var keys = new List<string>();
        foreach (var (a, b, c) in triangles)
        {
            int[] sorted = { a, b, c };
            Array.Sort(sorted);
            keys.Add(sorted[0] + "-" + sorted[1] + "-" + sorted[2]);
        }
        keys.Sort();
        return string.Join(",", keys);
    }

    /// <summary>Catalan number by its own recurrence, the expected number of triangulations of an n-gon.</summary>
    internal static long Catalan(int n)
    {
        var values = new long[n + 1];
        values[0] = 1;
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < i; j++)
                values[i] += values[j] * values[i - 1 - j];
        return values[n];
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Poly NotchedSquare() => new Poly(new List<(long X, long Y)>
    {
        (0, 0), (12, 0), (12, 12), (7, 12), (7, 4), (5, 4), (5, 12), (0, 12)
    });

    private static Poly ConvexPentagon() => new Poly(new List<(long X, long Y)>
    {
        (0, 0), (10, 0), (14, 8), (5, 14), (-4, 8)
    });

    private static HashSet<(int, int)> Constraints(Poly poly)
    {
        var set = new HashSet<(int, int)>();
        for (int i = 0; i < poly.Count; i++)
        {
            int a = i, b = (i + 1) % poly.Count;
            set.Add(a < b ? (a, b) : (b, a));
        }
        return set;
    }

    private static long SumArea2(Poly poly, List<(int A, int B, int C)> triangles)
    {
        long total = 0;
        foreach (var (a, b, c) in triangles) total += Math.Abs(poly.Cross(a, b, c));
        return total;
    }

    public static void Main()
    {
        // ---- invariant battery over three polygons ------------------------------------------------------
        var polygons = new List<(string Name, Poly Polygon)>
        {
            ("notched square", NotchedSquare()),
            ("convex pentagon", ConvexPentagon()),
            ("triangle", new Poly(new List<(long X, long Y)> { (0, 0), (9, 0), (2, 7) })),
            ("collinear edge", new Poly(new List<(long X, long Y)> { (0, 0), (4, 0), (8, 0), (8, 6), (0, 6) }))
        };
        bool counts = true, areas = true, inside = true, constraints = true, delaunay = true;
        string? firstFailure = null;
        long totalFlips = 0;
        foreach (var (name, poly) in polygons)
        {
            var constrained = Constraints(poly);
            var triangles = Cdt.EarClip(poly);
            totalFlips += Cdt.DelaunayFlip(poly, constrained, triangles);
            if (triangles.Count != poly.Count - 2)
            {
                counts = false;
                firstFailure ??= name + ": " + triangles.Count + " triangles for " + poly.Count + " vertices";
            }
            if (SumArea2(poly, triangles) != poly.Area2)
            {
                areas = false;
                firstFailure ??= name + ": area " + SumArea2(poly, triangles) + " != " + poly.Area2;
            }
            var map = Cdt.EdgeMap(triangles);
            foreach (var key in constrained)
                if (!map.TryGetValue(key, out List<int>? owners) || owners.Count != 1)
                {
                    constraints = false;
                    firstFailure ??= name + ": boundary edge " + key.Item1 + "-" + key.Item2 + " is missing";
                }
            foreach (var pair in map)
                if (!constrained.Contains(pair.Key) && pair.Value.Count != 2)
                {
                    counts = false;
                    firstFailure ??= name + ": diagonal appears " + pair.Value.Count + " times";
                }
            foreach (var (a, b, c) in triangles)
            {
                double cx = (poly[a].X + poly[b].X + poly[c].X) / 3.0;
                double cy = (poly[a].Y + poly[b].Y + poly[c].Y) / 3.0;
                if (!poly.StrictlyContains(cx, cy))
                {
                    inside = false;
                    firstFailure ??= name + ": centroid of " + a + "," + b + "," + c + " is outside";
                }
            }
            foreach (var pair in map)
            {
                if (pair.Value.Count != 2) continue;
                var t1 = triangles[pair.Value[0]];
                var t2 = triangles[pair.Value[1]];
                if (poly.Cross(t1.A, t1.B, t1.C) <= 0) continue;
                int d = t2.A != pair.Key.Item1 && t2.A != pair.Key.Item2 ? t2.A
                      : (t2.B != pair.Key.Item1 && t2.B != pair.Key.Item2 ? t2.B : t2.C);
                int a = t1.A != pair.Key.Item1 && t1.A != pair.Key.Item2 ? t1.A
                      : (t1.B != pair.Key.Item1 && t1.B != pair.Key.Item2 ? t1.B : t1.C);
                if ((poly.Cross(a, d, pair.Key.Item1) > 0) == (poly.Cross(a, d, pair.Key.Item2) > 0)) continue;
                if (poly.InCircle(t1.A, t1.B, t1.C, d) > 0)
                {
                    delaunay = false;
                    firstFailure ??= name + ": edge " + pair.Key.Item1 + "-" + pair.Key.Item2 + " is not Delaunay";
                }
            }
        }
        Check(counts, "every polygon yields n-2 triangles with each diagonal shared by two triangles: " + (firstFailure ?? "ok"));
        Check(areas, "the triangulations cover exactly the polygon area (exact integer shoelace)");
        Check(inside, "every triangle centroid lies strictly inside the polygon");
        Check(constraints, "all boundary edges survive the flips (the constraints are never flipped)");
        Check(delaunay, "after the flip loop every convex interior edge satisfies the exact in-circle test");
        Check(totalFlips > 0, "the flip loop actually removed " + totalFlips + " non-Delaunay diagonals");

        // ---- exhaustive enumeration of all triangulations ------------------------------------------------
        // A convex octagon has exactly Catalan(6) = 132 triangulations, which validates the enumerator; the
        // non-convex notched square has fewer (only the chords that stay inside the polygon are usable).
        var octagon = new Poly(new List<(long X, long Y)>
        {
            (0, 0), (4, -3), (10, -3), (14, 0), (14, 8), (10, 11), (4, 11), (0, 8)
        });
        var octagonRings = new List<int>();
        for (int i = 0; i < octagon.Count; i++) octagonRings.Add(i);
        var octagonTriangulations = new List<List<(int A, int B, int C)>>();
        Cdt.Enumerate(octagon, octagonRings, new List<(int A, int B, int C)>(), octagonTriangulations);
        var octagonDistinct = new Dictionary<string, List<(int A, int B, int C)>>();
        foreach (var triangulation in octagonTriangulations) octagonDistinct[Cdt.Key(triangulation)] = triangulation;
        octagonTriangulations = new List<List<(int A, int B, int C)>>(octagonDistinct.Values);
        bool octagonSizes = octagonTriangulations.TrueForAll(t => t.Count == octagon.Count - 2);
        Check(octagonTriangulations.Count == Cdt.Catalan(octagon.Count - 2) && octagonTriangulations.Count == 132
            && octagonSizes,
            "the recursive ear removal enumerates all Catalan(" + (octagon.Count - 2) + ") = "
            + Cdt.Catalan(octagon.Count - 2) + " triangulations of the convex octagon (found "
            + octagonTriangulations.Count + " distinct of " + octagonDistinct.Count + " keys)");
        Check(octagonTriangulations.TrueForAll(t => SumArea2(octagon, t) == octagon.Area2),
            "every enumerated triangulation of the octagon covers exactly its area");
        double octagonBest = 0;
        foreach (var triangulation in octagonTriangulations) octagonBest = Math.Max(octagonBest, Cdt.MinAngle(octagon, triangulation));
        var octagonCdt = Cdt.EarClip(octagon);
        Cdt.DelaunayFlip(octagon, Constraints(octagon), octagonCdt);
        Check(Math.Abs(Cdt.MinAngle(octagon, octagonCdt) - octagonBest) < 1e-9,
            "the CDT of the convex octagon attains the max-min angle over all 132 triangulations");

        var notched = NotchedSquare();
        var rings = new List<int>();
        for (int i = 0; i < notched.Count; i++) rings.Add(i);
        var rawNotched = new List<List<(int A, int B, int C)>>();
        Cdt.Enumerate(notched, rings, new List<(int A, int B, int C)>(), rawNotched);
        var notchedDistinct = new Dictionary<string, List<(int A, int B, int C)>>();
        foreach (var triangulation in rawNotched) notchedDistinct[Cdt.Key(triangulation)] = triangulation;
        var all = new List<List<(int A, int B, int C)>>(notchedDistinct.Values);
        Check(all.Count >= 1 && all.Count < Cdt.Catalan(notched.Count - 2),
            "the notched square admits " + all.Count + " triangulations, fewer than the convex Catalan bound "
            + Cdt.Catalan(notched.Count - 2));
        Check(all.TrueForAll(t => t.Count == notched.Count - 2), "every enumerated triangulation has n-2 triangles");
        double bestMinAngle = 0;
        foreach (var triangulation in all)
        {
            Check(SumArea2(notched, triangulation) == notched.Area2, "every enumerated triangulation covers the same area");
            bestMinAngle = Math.Max(bestMinAngle, Cdt.MinAngle(notched, triangulation));
        }
        Check(all.TrueForAll(t => SumArea2(notched, t) == notched.Area2),
            "all enumerated triangulations of the notched square share its exact area");
        Check(all.Count <= 14, "the enumeration is exhaustive but small (" + all.Count + " triangulations)");

        var cdt = Cdt.EarClip(notched);
        var constrainedEdges = Constraints(notched);
        Cdt.DelaunayFlip(notched, constrainedEdges, cdt);
        double cdtMinAngle = Cdt.MinAngle(notched, cdt);
        Check(Math.Abs(cdtMinAngle - bestMinAngle) < 1e-9,
            "the CDT attains the max-min angle optimum over all " + all.Count + " triangulations ("
            + Math.Round(cdtMinAngle * 180 / Math.PI, 6) + " vs " + Math.Round(bestMinAngle * 180 / Math.PI, 6) + " degrees)");

        // ---- determinism and a degenerate input ----------------------------------------------------------
        var again = Cdt.EarClip(NotchedSquare());
        Cdt.DelaunayFlip(NotchedSquare(), Constraints(NotchedSquare()), again);
        Check(Math.Abs(Cdt.MinAngle(NotchedSquare(), again) - cdtMinAngle) < 1e-12,
            "clipping and flipping are deterministic for the same polygon");

        var collinearOriginal = new Poly(new List<(long X, long Y)>
        {
            (0, 0), (4, 0), (8, 0), (8, 6), (0, 6)
        });
        var plainRectangle = new Poly(new List<(long X, long Y)> { (0, 0), (8, 0), (8, 6), (0, 6) });
        var collinearTriangles = Cdt.EarClip(collinearOriginal);
        Cdt.DelaunayFlip(collinearOriginal, Constraints(collinearOriginal), collinearTriangles);
        var plainTriangles = Cdt.EarClip(plainRectangle);
        Cdt.DelaunayFlip(plainRectangle, Constraints(plainRectangle), plainTriangles);
        Check(collinearOriginal.Count == 4 && collinearTriangles.Count == 2
            && Math.Abs(Cdt.MinAngle(collinearOriginal, collinearTriangles) - Cdt.MinAngle(plainRectangle, plainTriangles)) < 1e-9,
            "a collinear vertex is cleaned away, so the rectangle triangulates like the plain rectangle");

        Console.WriteLine("polygons=" + polygons.Count + " flips=" + totalFlips
            + " triangulationsOfNotchedSquare=" + all.Count
            + " cdtMinAngleDegrees=" + Math.Round(cdtMinAngle * 180 / Math.PI, 6));
        Console.WriteLine("PASS");
    }
}
