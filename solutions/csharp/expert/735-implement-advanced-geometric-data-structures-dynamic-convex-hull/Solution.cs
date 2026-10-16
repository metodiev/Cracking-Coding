// Task 735 - Implement advanced geometric data structures: dynamic convex hull maintenance. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// The live point set is kept in a sorted map x -> y. The hull is *not* recomputed after an insertion: it is
// maintained as the two monotone chains of the hull, each stored as the ascending list of the x coordinates of
// its vertices. An insertion is repaired locally:
//   1. the new point p is bracketed by its chain neighbours L < p < R; a chain is convex, hence it lies below
//      its own chords, so if p is on or above the chord L-R it is interior and the chain stays untouched;
//   2. otherwise p joins the chain and the runs of now non-extreme neighbours are dropped by the usual
//      left-turn (lower chain) / right-turn (upper chain) loop, followed by one final test that removes p
//      itself should it turn out to be interior after the repairs.
// Orientation is decided by exact `long` cross products, so no epsilon appears anywhere and collinear triples
// are resolved by dropping the middle vertex (strict hulls only).
// Invariants verified after every mutation: both chains are strictly convex, they agree on the two x-extremes
// and hold only live points, and - against an independent Andrew monotone-chain rebuild - the vertex list, the
// doubled shoelace area and every point-in-hull answer agree. Deletion is the non-incremental path (rebuild
// from the surviving points, counted by Rebuilds) so the test can prove which path ran.
// Simplifications: points with equal x are not supported as distinct vertices - a repeated x replaces the
// stored y and rebuilds; the chains are sorted arrays, not balanced trees, so a splice is O(n) (fine for the
// tens of points exercised here, and the asymptotics are not the subject of the exercise). All coordinates
// come from a hand-written splitmix64 stream, so every run is identical.
using System;
using System.Collections.Generic;

internal static class SplitMix
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class DynamicHull
{
    private readonly SortedDictionary<int, int> _points = new SortedDictionary<int, int>();
    private readonly List<int> _lower = new List<int>();   // ascending x of the lower-hull vertices
    private readonly List<int> _upper = new List<int>();   // ascending x of the upper-hull vertices

    internal long Inserts, Splices, Rebuilds;

    internal int PointCount => _points.Count;

    internal bool Insert(int x, int y)
    {
        Inserts++;
        if (_points.TryGetValue(x, out int previous))
        {
            if (previous == y) return false;
            _points.Remove(x);
            Rebuild();                        // that x already owned a hull slot: restart from a clean chain pair
        }
        _points[x] = y;
        Splice(_lower, x, lower: true);
        Splice(_upper, x, lower: false);
        return true;
    }

    internal bool Remove(int x)
    {
        if (!_points.Remove(x)) return false;
        Rebuild();
        return true;
    }

    /// <summary>Hull vertices in counter-clockwise order, starting at the smallest x.</summary>
    internal List<(int X, int Y)> Hull()
    {
        var result = new List<(int X, int Y)>();
        foreach (int x in _lower) result.Add((x, _points[x]));
        for (int i = _upper.Count - 2; i >= 1; i--) result.Add((_upper[i], _points[_upper[i]]));
        return result;
    }

    /// <summary>Twice the hull area (exact shoelace over the integer vertices).</summary>
    internal long Area2()
    {
        var hull = Hull();
        if (hull.Count < 3) return 0;
        long sum = 0;
        for (int i = 0; i < hull.Count; i++)
        {
            var a = hull[i];
            var b = hull[(i + 1) % hull.Count];
            sum += (long)a.X * b.Y - (long)b.X * a.Y;
        }
        return sum < 0 ? -sum : sum;
    }

    internal bool Contains(int x, int y)
    {
        if (_points.Count == 0) return false;
        int minX = _lower[0], maxX = _lower[_lower.Count - 1];
        if (x < minX || x > maxX) return false;
        if (x == minX || x == maxX) return y == _points[x];   // the hull pinches to one point at an x-extreme
        return AboveChain(_lower, x, y) >= 0 && AboveChain(_upper, x, y) <= 0;
    }

    /// <summary>
    /// Cross product of (the hull edge bracketing x, the point). Because the query uses the two *adjacent*
    /// chain vertices around x, the chord is the hull edge itself, so the sign is exact; when x coincides with
    /// a chain vertex the directed-line test still answers "on which side of that vertex" correctly.
    /// </summary>
    private long AboveChain(List<int> chain, int x, int y)
    {
        int right = chain.BinarySearch(x);
        if (right < 0) right = ~right;
        int lx = chain[right - 1], rx = chain[right];
        return Cross(lx, _points[lx], rx, _points[rx], x, y);
    }

    /// <summary>Rebuilds both chains by replaying the live points in ascending x (Andrew's sweep).</summary>
    private void Rebuild()
    {
        Rebuilds++;
        _lower.Clear();
        _upper.Clear();
        foreach (var kv in _points)
        {
            Splice(_lower, kv.Key, lower: true);
            Splice(_upper, kv.Key, lower: false);
        }
    }

    private void Splice(List<int> chain, int x, bool lower)
    {
        int slot = chain.BinarySearch(x);
        if (slot >= 0) throw new InvalidOperationException("chain already contains x = " + x);
        int pos = ~slot;
        if (pos > 0 && pos < chain.Count && Side(chain[pos - 1], x, chain[pos], lower) <= 0) return;
        chain.Insert(pos, x);
        Splices++;
        int i = pos;
        while (i + 2 < chain.Count && Turn(chain[i], chain[i + 1], chain[i + 2], lower) <= 0) chain.RemoveAt(i + 1);
        while (i - 2 >= 0 && Turn(chain[i - 2], chain[i - 1], chain[i], lower) <= 0) { chain.RemoveAt(i - 1); i--; }
        if (i > 0 && i + 1 < chain.Count && Turn(chain[i - 1], chain[i], chain[i + 1], lower) <= 0) chain.RemoveAt(i);
    }

    /// <summary>cross(L, p, R) with the upper chain's sign flipped, so "&gt; 0" means "p belongs to this chain".</summary>
    private long Side(int lx, int px, int rx, bool lower)
    {
        long c = Cross(lx, _points[lx], px, _points[px], rx, _points[rx]);
        return lower ? c : -c;
    }

    /// <summary>cross(a, b, c) with the upper chain's sign flipped; a strictly convex chain needs "&gt; 0".</summary>
    private long Turn(int a, int b, int c, bool lower)
    {
        long t = Cross(a, _points[a], b, _points[b], c, _points[c]);
        return lower ? t : -t;
    }

    private static long Cross(int ax, int ay, int bx, int by, int cx, int cy)
        => (long)(bx - ax) * (cy - ay) - (long)(by - ay) * (cx - ax);

    /// <summary>Structural invariant: null when the stored chains really are the hull of the stored points.</summary>
    internal string? InvariantViolation()
    {
        // The two chains only have to agree on their x-extremes: the lower and upper hull legitimately carry
        // different numbers of intermediate vertices.
        if (_lower.Count == 0) return _points.Count == 0 ? null : "live points without a hull";
        for (int i = 1; i < _lower.Count; i++) if (_lower[i] <= _lower[i - 1]) return "lower chain x not ascending";
        foreach (int x in _lower) if (!_points.ContainsKey(x)) return "lower chain vertex is not a live point";
        foreach (int x in _upper) if (!_points.ContainsKey(x)) return "upper chain vertex is not a live point";
        if (_lower[0] != _upper[0] || _lower[_lower.Count - 1] != _upper[_upper.Count - 1])
            return "chains disagree on the x extremes";
        foreach (int x in _points.Keys)
        {
            if (x < _lower[0]) return "chain starts after the smallest live x";
            if (x > _lower[_lower.Count - 1]) return "chain ends before the largest live x";
        }
        for (int i = 0; i + 2 < _lower.Count; i++) if (Turn(_lower[i], _lower[i + 1], _lower[i + 2], true) <= 0) return "lower chain not strictly convex";
        for (int i = 0; i + 2 < _upper.Count; i++) if (Turn(_upper[i], _upper[i + 1], _upper[i + 2], false) <= 0) return "upper chain not strictly convex";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static long CrossOf((int X, int Y) a, (int X, int Y) b, (int X, int Y) c)
        => (long)(b.X - a.X) * (c.Y - a.Y) - (long)(b.Y - a.Y) * (c.X - a.X);

    /// <summary>Independent oracle: Andrew's monotone chain over the same live points.</summary>
    private static List<(int X, int Y)> BruteHull(List<(int X, int Y)> points)
    {
        var sorted = new List<(int X, int Y)>(points);
        sorted.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));
        if (sorted.Count <= 2) return sorted;
        var chain = new List<(int X, int Y)>();
        foreach (var p in sorted)
        {
            while (chain.Count >= 2 && CrossOf(chain[chain.Count - 2], chain[chain.Count - 1], p) <= 0) chain.RemoveAt(chain.Count - 1);
            chain.Add(p);
        }
        int floor = chain.Count + 1;
        for (int i = sorted.Count - 2; i >= 0; i--)
        {
            var p = sorted[i];
            while (chain.Count >= floor && CrossOf(chain[chain.Count - 2], chain[chain.Count - 1], p) <= 0) chain.RemoveAt(chain.Count - 1);
            chain.Add(p);
        }
        chain.RemoveAt(chain.Count - 1);
        return chain;
    }

    private static bool BruteContains(List<(int X, int Y)> hull, int x, int y)
    {
        if (hull.Count == 0) return false;
        if (hull.Count == 1) return hull[0].X == x && hull[0].Y == y;
        if (hull.Count == 2)
            return CrossOf(hull[0], hull[1], (x, y)) == 0
                && x >= Math.Min(hull[0].X, hull[1].X) && x <= Math.Max(hull[0].X, hull[1].X)
                && y >= Math.Min(hull[0].Y, hull[1].Y) && y <= Math.Max(hull[0].Y, hull[1].Y);
        for (int i = 0; i < hull.Count; i++)
            if (CrossOf(hull[i], hull[(i + 1) % hull.Count], (x, y)) < 0) return false;
        return true;
    }

    private static long ScaledArea(List<(int X, int Y)> hull)
    {
        if (hull.Count < 3) return 0;
        long sum = 0;
        for (int i = 0; i < hull.Count; i++)
        {
            var a = hull[i];
            var b = hull[(i + 1) % hull.Count];
            sum += (long)a.X * b.Y - (long)b.X * a.Y;
        }
        return sum < 0 ? -sum : sum;
    }

    private static bool SameHull(List<(int X, int Y)> a, List<(int X, int Y)> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        const int Span = 61, YRange = 201;
        ulong state = 0x7350CAFE5EED1234UL;
        var live = new List<(int X, int Y)>();
        bool verticesAgree = true, areaAgree = true, containsAgree = true, invariantHeld = true;
        string? firstFailure = null;
        int probes = 0;

        void Verify(DynamicHull structure, string stage, List<(int X, int Y)> points)
        {
            var expected = BruteHull(points);
            var got = structure.Hull();
            if (!SameHull(got, expected))
            {
                verticesAgree = false;
                firstFailure ??= stage + ": " + got.Count + " vertices, expected " + expected.Count;
            }
            if (ScaledArea(got) != ScaledArea(expected))
            {
                areaAgree = false;
                firstFailure ??= stage + ": area2 " + ScaledArea(got) + " != " + ScaledArea(expected);
            }
            if (structure.InvariantViolation() is string violation)
            {
                invariantHeld = false;
                firstFailure ??= stage + ": " + violation;
            }
            for (int gx = -3; gx <= Span + 3; gx += 4)
            {
                for (int gy = -120; gy <= 120; gy += 12)
                {
                    probes++;
                    if (structure.Contains(gx, gy) != BruteContains(expected, gx, gy))
                    {
                        containsAgree = false;
                        firstFailure ??= stage + ": containment of (" + gx + "," + gy + ") differs";
                    }
                }
            }
        }

        // ---- degenerate inputs ---------------------------------------------------------------------
        var empty = new DynamicHull();
        Check(empty.PointCount == 0 && empty.Hull().Count == 0 && empty.Area2() == 0 && !empty.Contains(0, 0),
            "an empty structure has no hull, no area and contains nothing");
        Check(!empty.Remove(4), "removing a point that was never inserted reports no change");
        Check(empty.Insert(5, 5) && !empty.Insert(5, 5), "a repeated identical point is a no-op");
        Check(empty.Insert(5, 7) && empty.PointCount == 1 && empty.Hull().Count == 1 && empty.Contains(5, 7) && !empty.Contains(5, 5),
            "the same x with a new y replaces the stored point");

        var collinear = new DynamicHull();
        collinear.Insert(0, 0);
        collinear.Insert(1, 1);
        Check(collinear.Insert(2, 2) && collinear.Hull().Count == 2 && collinear.Area2() == 0,
            "three collinear points collapse to the two extreme hull vertices");
        Check(collinear.Contains(1, 1) && collinear.Contains(0, 0) && !collinear.Contains(1, 2),
            "a point on the degenerate hull segment is inside, off the segment it is not");
        Check(collinear.InvariantViolation() == null, "the degenerate two-vertex hull satisfies the chain invariant");

        // ---- a known shape: trapezoid + interior point + boundary point ------------------------------
        // The vertices need pairwise distinct x, which is the structure's documented limitation, so a rotated
        // convex quadrilateral is used instead of an axis-aligned square.
        var trapezoid = new DynamicHull();
        trapezoid.Insert(0, 0);
        trapezoid.Insert(1, 3);
        trapezoid.Insert(3, 3);
        trapezoid.Insert(4, 0);
        Check(trapezoid.Hull().Count == 4 && trapezoid.Area2() == 18 && trapezoid.Contains(2, 1),
            "the trapezoid hull has four vertices, double area 18 and contains its centre");
        Check(trapezoid.Insert(2, 2) && trapezoid.Hull().Count == 4, "an interior point leaves the hull untouched");
        Check(trapezoid.Contains(2, 0) && trapezoid.Contains(2, 3) && !trapezoid.Contains(2, -1),
            "points on the lower and upper hull edges are inside, one below the hull is not");

        // ---- incremental workload, verified after every single insertion ----------------------------
        var work = new DynamicHull();
        for (int x = 0; x < Span; x++)
        {
            int y = SplitMix.NextInt(ref state, YRange) - 100;
            work.Insert(x, y);
            live.Add((x, y));
            Verify(work, "insert x=" + x, live);
        }
        Check(verticesAgree, "incremental chain repair matches the Andrew rebuild after every insertion: " + (firstFailure ?? "ok"));
        Check(areaAgree, "the shoelace area of the maintained hull matches the rebuilt one after every insertion");
        Check(containsAgree, "point-in-hull agrees with the brute-force oracle on all " + probes + " probes after every mutation");
        Check(invariantHeld, "both chains stay strictly convex and consistent after every insertion: " + (firstFailure ?? "ok"));
        Check(work.Rebuilds == 0 && work.Splices >= Span / 2 && work.Splices <= 2 * Span,
            "the " + Span + " insertions never fell back to a rebuild (" + work.Splices + " splices, 0 rebuilds)");
        Check(work.PointCount == Span && work.Hull().Count >= 3 && work.Hull().Count < Span,
            "the hull is a proper subset of the " + Span + " live points");

        // ---- a replaced y forces exactly one rebuild ------------------------------------------------
        var swap = new DynamicHull();
        foreach (var p in live) swap.Insert(p.X, p.Y);
        long before = swap.Rebuilds;
        Check(swap.Insert(live[0].X, live[0].Y + 1000) && swap.Rebuilds == before + 1,
            "replacing the y of an existing x rebuilds the chains exactly once");
        Check(!swap.Insert(live[0].X, live[0].Y + 1000) && swap.Rebuilds == before + 1,
            "repeating the identical point afterwards changes nothing");

        // ---- deletion is the rebuild path -----------------------------------------------------------
        int removed = 0;
        for (int r = 0; r < 15; r++)
        {
            int x = SplitMix.NextInt(ref state, Span);
            int pos = live.FindIndex(p => p.X == x);
            if (pos < 0) continue;
            live.RemoveAt(pos);
            Check(work.Remove(x), "removing a live point reports a change");
            removed++;
            Verify(work, "remove x=" + x, live);
        }
        Check(removed >= 10 && work.Rebuilds == removed,
            "each of the " + removed + " deletions triggered exactly one rebuild and nothing else");
        Check(!work.Remove(-1), "deleting an unknown point is a no-op");
        Check(verticesAgree && areaAgree && containsAgree && invariantHeld,
            "hull, area, containment and the invariant still agree after deletions: " + (firstFailure ?? "ok"));

        foreach (var p in live) work.Remove(p.X);
        Check(work.PointCount == 0 && work.Hull().Count == 0 && work.Area2() == 0 && work.InvariantViolation() == null,
            "deleting every point leaves an empty, invariant-clean structure");

        Console.WriteLine("inserts=" + work.Inserts + " splices=" + work.Splices + " rebuilds=" + work.Rebuilds
            + " deletions=" + removed + " probes=" + probes + " livePoints=" + Span);
        Console.WriteLine("PASS");
    }
}
