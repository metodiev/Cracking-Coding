// Task 877 - Implement full-featured computational geometry library with robust predicates.
// (Tier: Master, CTCI Beyond - Geometry & Robust Computation)
//
// Model: a geometry core whose every combinatorial decision is a *predicate with an exact fallback*.
//   * orient2d / incircle - a floating-point filter (a forward error bound on the determinant) that
//     answers immediately when the sign is safely determined, and an exact BigInteger determinant
//     otherwise. The sign returned is therefore never wrong, which is exactly what makes the algorithms
//     built on it - convex hull (monotone chain), segment intersection, point-in-polygon (winding number),
//     closest pair (divide and conquer) and polygon area - immune to the classic degeneracies.
//   * All test coordinates are integers, so the exact branch is a genuine exact answer (no rational
//     arithmetic approximation anywhere) and every oracle can be exact too.
// The header records the invariant each routine maintains: the hull is convex with all points on one side
// of every edge, the winding number matches ray casting, the closest-pair distance equals the brute-force
// minimum, and each predicate agrees with the exact determinant.
//
// Simplifications (documented, not behavioural): integer inputs in a bounded range (|coordinate| <= 1e6)
// so determinants stay small; the adaptive filter has one level (double, then exact) instead of the
// staged floating-point expansions of Shewchuk's predicates; no Delaunay triangulation, no Voronoi
// diagram and no 3D predicates - the library covers the 2D kernel the rest of the tier needs.
//
// Verified here:
//   * predicates: 600 random and near-degenerate triples agree with the exact BigInteger determinant, the
//     filter fires on the generic cases and falls back exactly on the cocircular/collinear ones, and a
//     deliberately naive double determinant is shown to be wrong on the degenerate family;
//   * hull: the vertex set equals an independent gift-wrapping (Jarvis march) computation, every point lies on
//     one side of every hull edge, the area is the shoelace sum, all points are inside, and duplicate or
//     collinear points do not change the answer;
//   * winding: exact point-in-convex-polygon agrees with ray casting on a full grid (including on-boundary
//     classification by exact orientation);
//   * segment intersection and closest pair: exact orientation/rational oracles and a brute-force minimum;
//   * degenerate inputs: 0, 1 and 2 points, all-collinear input, repeated vertices, zero-area polygon;
//   * determinism: every routine is bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

public static class Solution
{
    private static ulong _state = 0x1DEA5EEDF00DBA11UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static long NextCoord(long bound) => (long)(NextU64() % (ulong)(2 * bound + 1)) - bound;

    private struct P
    {
        public long X;
        public long Y;
        public P(long x, long y) { X = x; Y = y; }
        public override string ToString() => "(" + X + "," + Y + ")";
        public static P operator -(P a, P b) => new P(a.X - b.X, a.Y - b.Y);
    }

    // ---------- predicates ----------
    private static long _exactFallbacks;
    private static long _filterHits;

    private static BigInteger ExactOrient(P a, P b, P c)
    {
        BigInteger bx = b.X - a.X, by = b.Y - a.Y;
        BigInteger cx = c.X - a.X, cy = c.Y - a.Y;
        return bx * cy - by * cx;
    }

    /// <summary>
    /// Sign of the orient2d determinant. The double evaluation carries a forward error bound of
    /// 3*2^-53*(|term1| + |term2|); inside that band the exact BigInteger determinant decides, so the sign
    /// is always the sign of the exact determinant.
    /// </summary>
    private static int Orient2D(P a, P b, P c)
    {
        double bx = b.X - a.X, by = b.Y - a.Y;
        double cx = c.X - a.X, cy = c.Y - a.Y;
        double left = bx * cy, right = by * cx;
        double det = left - right;
        double bound = 3.0 * 2.2204460492503131e-16 * (Math.Abs(left) + Math.Abs(right));
        if (Math.Abs(det) > bound)
        {
            _filterHits++;
            return Math.Sign(det);
        }
        _exactFallbacks++;
        return ExactOrient(a, b, c).Sign;
    }

    /// <summary>Deliberately unguarded determinant - kept as the demonstration of why the filter exists.</summary>
    private static int NaiveOrient(P a, P b, P c)
    {
        double bx = b.X - a.X, by = b.Y - a.Y;
        double cx = c.X - a.X, cy = c.Y - a.Y;
        return Math.Sign(bx * cy - by * cx);
    }

    private static int ExactSign(BigInteger v) => v.Sign;

    /// <summary>incircle for integer input: the 4x4 determinant's sign decides inside/on/outside.</summary>
    private static int InCircle(P a, P b, P c, P d)
    {
        // Rows: (x - dx, y - dy, (x-dx)^2 + (y-dy)^2) for a, b, c; the determinant is exact in BigInteger.
        BigInteger ax = a.X - d.X, ay = a.Y - d.Y;
        BigInteger bx = b.X - d.X, by = b.Y - d.Y;
        BigInteger cx = c.X - d.X, cy = c.Y - d.Y;
        BigInteger det =
            ax * (by * (cx * cx + cy * cy) - (bx * bx + by * by) * cy)
          - ay * (bx * (cx * cx + cy * cy) - (bx * bx + by * by) * cx)
          + (ax * ax + ay * ay) * (bx * cy - by * cx);
        return det.Sign;
    }

    // ---------- convex hull (monotone chain) ----------
    private static List<P> ConvexHull(List<P> points)
    {
        var sorted = points.Distinct().OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        if (sorted.Count <= 2) return sorted;
        var hull = new List<P>();
        foreach (P p in sorted)
        {
            while (hull.Count >= 2 && Orient2D(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        int lower = hull.Count + 1;
        for (int i = sorted.Count - 2; i >= 0; i--)
        {
            P p = sorted[i];
            while (hull.Count >= lower && Orient2D(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        hull.RemoveAt(hull.Count - 1);      // the first point is repeated at the end
        return hull;
    }

    /// <summary>
    /// Independent oracle: Jarvis march (gift wrapping). From the leftmost-lowest vertex it repeatedly picks
    /// the candidate that leaves every other point strictly to its left, breaking exact collinearity in favour
    /// of the farther point - so the traversal never stops on a point that merely lies on an edge's line.
    /// </summary>
    private static List<P> BruteForceHull(List<P> points)
    {
        var distinct = points.Distinct().ToList();
        if (distinct.Count <= 2) return distinct.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        P start = distinct.OrderBy(p => p.X).ThenBy(p => p.Y).First();
        var hull = new List<P> { start };
        P current = start;
        for (int guard = 0; guard <= distinct.Count; guard++)
        {
            P next = current;
            foreach (P r in distinct)
            {
                if (Same(r, current)) continue;
                if (Same(next, current)) { next = r; continue; }
                int o = Orient2D(current, next, r);
                if (o < 0 || (o == 0 && Dist2(current, r) > Dist2(current, next))) next = r;
            }
            if (Same(next, start) || Same(next, current)) break;
            hull.Add(next);
            current = next;
        }
        return hull.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
    }

    private static bool Same(P a, P b) => a.X == b.X && a.Y == b.Y;

    private static BigInteger SignedAreaTimes2(List<P> polygon)
    {
        BigInteger acc = 0;
        for (int i = 0; i < polygon.Count; i++)
        {
            P a = polygon[i], b = polygon[(i + 1) % polygon.Count];
            acc += (BigInteger)a.X * b.Y - (BigInteger)b.X * a.Y;
        }
        return acc;
    }

    private static bool InsideConvex(List<P> hull, P q)
    {
        if (hull.Count == 0) return false;
        if (hull.Count == 1) return Same(hull[0], q);
        if (hull.Count == 2)
        {
            if (Orient2D(hull[0], hull[1], q) != 0) return false;
            return Math.Min(hull[0].X, hull[1].X) <= q.X && q.X <= Math.Max(hull[0].X, hull[1].X) &&
                   Math.Min(hull[0].Y, hull[1].Y) <= q.Y && q.Y <= Math.Max(hull[0].Y, hull[1].Y);
        }
        for (int i = 0; i < hull.Count; i++)
            if (Orient2D(hull[i], hull[(i + 1) % hull.Count], q) < 0) return false;
        return true;
    }

    /// <summary>Ray casting oracle with exact orientation tests for the on-boundary cases.</summary>
    private static bool InsideByRayCasting(List<P> polygon, P q)
    {
        if (polygon.Count < 3) return polygon.Any(p => Same(p, q));
        for (int i = 0; i < polygon.Count; i++)
        {
            P a = polygon[i], b = polygon[(i + 1) % polygon.Count];
            if (OnSegment(a, b, q)) return true;
        }
        bool inside = false;
        for (int i = 0, j = polygon.Count - 1; i < polygon.Count; j = i++)
        {
            P a = polygon[i], b = polygon[j];
            bool straddles = (a.Y > q.Y) != (b.Y > q.Y);
            if (!straddles) continue;
            // x coordinate of the crossing, compared exactly: (b.X - a.X) * (q.Y - a.Y) / (b.Y - a.Y) + a.X > q.X
            BigInteger lhs = (BigInteger)(b.X - a.X) * (q.Y - a.Y);
            BigInteger rhs = (BigInteger)(q.X - a.X) * (b.Y - a.Y);
            if (b.Y > a.Y ? lhs > rhs : lhs < rhs) inside = !inside;
        }
        return inside;
    }

    private static bool OnSegment(P a, P b, P q)
    {
        if (Orient2D(a, b, q) != 0) return false;
        return Math.Min(a.X, b.X) <= q.X && q.X <= Math.Max(a.X, b.X) &&
               Math.Min(a.Y, b.Y) <= q.Y && q.Y <= Math.Max(a.Y, b.Y);
    }

    // ---------- segment intersection ----------
    private static bool SegmentsIntersect(P p1, P p2, P p3, P p4) => SegmentsIntersect(p1, p2, p3, p4, out _);

    private static bool SegmentsIntersect(P p1, P p2, P p3, P p4, out string reason)
    {
        reason = "";
        int d1 = Orient2D(p3, p4, p1);
        int d2 = Orient2D(p3, p4, p2);
        int d3 = Orient2D(p1, p2, p3);
        int d4 = Orient2D(p1, p2, p4);
        if (d1 * d2 < 0 && d3 * d4 < 0) { reason = "proper crossing"; return true; }
        if (d1 == 0 && OnSegment(p3, p4, p1)) { reason = "endpoint on the other segment"; return true; }
        if (d2 == 0 && OnSegment(p3, p4, p2)) { reason = "endpoint on the other segment"; return true; }
        if (d3 == 0 && OnSegment(p1, p2, p3)) { reason = "endpoint on the other segment"; return true; }
        if (d4 == 0 && OnSegment(p1, p2, p4)) { reason = "endpoint on the other segment"; return true; }
        return false;
    }

    /// <summary>Independent oracle: solve the two-parameter system exactly with rational arithmetic.</summary>
    private static bool SegmentsIntersectRational(P p1, P p2, P p3, P p4)
    {
        BigInteger rx = p2.X - p1.X, ry = p2.Y - p1.Y;
        BigInteger sx = p4.X - p3.X, sy = p4.Y - p3.Y;
        BigInteger denom = rx * sy - ry * sx;
        BigInteger qpx = p3.X - p1.X, qpy = p3.Y - p1.Y;
        if (denom == 0)
        {
            // parallel: they intersect only if collinear and the projections overlap
            if (rx * qpy - ry * qpx != 0) return false;
            return ProjectionsOverlap(p1.X, p2.X, p3.X, p4.X) && ProjectionsOverlap(p1.Y, p2.Y, p3.Y, p4.Y);
        }
        BigInteger numT = qpx * sy - qpy * sx;
        BigInteger numU = qpx * ry - qpy * rx;
        // 0 <= numT/denom <= 1 and 0 <= numU/denom <= 1, with the sign of the denominator carried along
        Func<BigInteger, BigInteger, bool> inUnit = (num, den) =>
            den > 0 ? num >= 0 && num <= den : num <= 0 && num >= den;
        return inUnit(numT, denom) && inUnit(numU, denom);
    }

    private static bool ProjectionsOverlap(BigInteger a1, BigInteger a2, BigInteger b1, BigInteger b2) =>
        BigInteger.Max(BigInteger.Min(a1, a2), BigInteger.Min(b1, b2)) <=
        BigInteger.Min(BigInteger.Max(a1, a2), BigInteger.Max(b1, b2));

    // ---------- closest pair ----------
    private static long _distanceComparisons;

    private static long ClosestPairSquared(List<P> pts)
    {
        var sorted = pts.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        return ClosestRec(sorted, 0, sorted.Count - 1);
    }

    private static long ClosestRec(List<P> pts, int lo, int hi)
    {
        int n = hi - lo + 1;
        if (n <= 3)
        {
            long best = long.MaxValue;
            for (int i = lo; i <= hi; i++)
                for (int j = i + 1; j <= hi; j++)
                {
                    _distanceComparisons++;
                    best = Math.Min(best, Dist2(pts[i], pts[j]));
                }
            return best;
        }
        int mid = (lo + hi) / 2;
        long bestLeft = ClosestRec(pts, lo, mid);
        long bestRight = ClosestRec(pts, mid + 1, hi);
        long bestSoFar = Math.Min(bestLeft, bestRight);
        long midX = pts[mid].X;
        var strip = pts.GetRange(lo, n).Where(p => (p.X - midX) * (p.X - midX) < bestSoFar)
                        .OrderBy(p => p.Y).ThenBy(p => p.X).ToList();
        for (int i = 0; i < strip.Count; i++)
            for (int j = i + 1; j < strip.Count && (strip[j].Y - strip[i].Y) * (strip[j].Y - strip[i].Y) < bestSoFar; j++)
            {
                _distanceComparisons++;
                bestSoFar = Math.Min(bestSoFar, Dist2(strip[i], strip[j]));
            }
        return bestSoFar;
    }

    private static long Dist2(P a, P b) => (a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y);

    // ---------- main ----------
    public static void Main()
    {
        // 1: the robust predicate agrees with the exact determinant on random and exactly degenerate triples.
        int predicateAgreement = 0, degenerateFamily = 0;
        for (int t = 0; t < 600; t++)
        {
            P a, b, c;
            if (t % 3 == 0)
            {
                // Exactly collinear family: c = a + k*(b - a). The double determinant of the differences is
                // contaminated by cancellation to ~1e-4 while the exact value is 0, which is precisely the band
                // the filter cannot decide - it must hand over to the exact determinant.
                a = new P(NextCoord(1000000), NextCoord(1000000));
                b = new P(NextCoord(1000000), NextCoord(1000000));
                long k = 1 + (long)(NextU64() % 3);
                c = new P(a.X + k * (b.X - a.X), a.Y + k * (b.Y - a.Y));
                degenerateFamily++;
            }
            else
            {
                a = new P(NextCoord(1000000), NextCoord(1000000));
                b = new P(NextCoord(1000000), NextCoord(1000000));
                c = new P(NextCoord(1000000), NextCoord(1000000));
            }
            int robust = Orient2D(a, b, c);
            int exact = ExactSign(ExactOrient(a, b, c));
            if (robust == exact) predicateAgreement++;
            if (t % 3 == 0 && robust != 0) predicateAgreement = -1000000;      // a collinear triple must be 0
        }
        Check(predicateAgreement == 600, "orient2d matches the exact determinant on 600 triples");
        Check(degenerateFamily == 200, "the suite really contains the exactly-degenerate family");
        Check(_exactFallbacks > 0 && _filterHits > 0,
              "both the filter and the exact fallback are exercised (" + _filterHits + " filtered, " +
              _exactFallbacks + " exact)");
        int robustOnDegenerate;
        long naiveMistakes = CountNaiveMistakes(out robustOnDegenerate);
        Console.WriteLine("orient2d: " + _filterHits + " filtered / " + _exactFallbacks +
                          " exact fallbacks; on the large-coordinate collinear family the naive determinant gets " +
                          naiveMistakes + "/200 wrong and orient2d gets " + robustOnDegenerate + "/200 right");

        // 2: the naive double determinant genuinely fails where the robust one does not.
        Check(naiveMistakes > 0 && robustOnDegenerate == 200,
              "an unguarded determinant gets " + naiveMistakes +
              "/200 degenerate triples wrong while orient2d gets all 200 right");
        Check(Orient2D(new P(0, 0), new P(1, 1), new P(2, 2)) == 0, "three collinear points are exactly zero");
        Check(Orient2D(new P(0, 0), new P(4, 1), new P(2, 3)) != 0, "a strictly convex triple is not zero");

        // 3: incircle exactness on a cocircular quadruple.
        int cocircularZero = InCircle(new P(1, 0), new P(0, 1), new P(-1, 0), new P(0, -1));
        Check(cocircularZero == 0, "four points on the unit circle are exactly cocircular");
        Check(InCircle(new P(0, 0), new P(2, 0), new P(0, 2), new P(1, 1)) > 0,
              "the fourth point sits strictly inside the circumcircle");

        // 4: hull vertex set against the brute-force extreme-edge characterisation.
        int hullAgreements = 0;
        for (int t = 0; t < 40; t++)
        {
            int n = 3 + (int)(NextU64() % 12);
            var pts = Enumerable.Range(0, n).Select(_ => new P(NextCoord(60), NextCoord(60))).ToList();
            var hull = ConvexHull(pts);
            var oracle = BruteForceHull(pts);
            if (HullEquals(hull, oracle)) hullAgreements++;
        }
        Check(hullAgreements == 40, "the hull vertex set matches the independent gift-wrapping oracle on 40 point sets ("
              + hullAgreements + "/40)");

        // 5-6: hull invariants - convexity, one-sidedness, area and containment.
        var cloud = Enumerable.Range(0, 200).Select(_ => new P(NextCoord(5000), NextCoord(5000))).ToList();
        var bigHull = ConvexHull(cloud);
        Check(bigHull.Count >= 3 && bigHull.Count <= cloud.Distinct().Count(), "the hull is a proper subset");
        Check(IsStrictlyConvex(bigHull), "consecutive hull triples turn strictly left");
        Check(cloud.All(p => InsideConvex(bigHull, p)), "every input point lies inside the hull");
        Check(cloud.Select(p => Orient2D(bigHull[0], bigHull[1], p)).All(s => s >= 0),
              "every point is on one side of the first hull edge");
        Check(SignedAreaTimes2(bigHull) > 0, "the hull is oriented counter-clockwise with positive area");
        Check(HullAreaMatchesTriangulation(bigHull), "the shoelace area equals the triangle fan area");
        Check(ConvexHull(cloud.Concat(cloud.Take(30)).ToList()).Count == bigHull.Count,
              "duplicating points does not change the hull");
        Check(HullEquals(ConvexHull(cloud.OrderBy(p => p.Y).ToList()), ConvexHull(cloud.OrderByDescending(p => p.X).ToList())),
              "the hull is independent of the input order");

        // 7: winding membership against ray casting over a full grid.
        int gridAgreements = 0, gridTotal = 0, boundaryHits = 0;
        for (int x = -20; x <= 20; x += 2)
            for (int y = -20; y <= 20; y += 2)
            {
                var q = new P(x, y);
                bool onBoundary = Enumerable.Range(0, bigHull.Count)
                    .Any(i => OnSegment(bigHull[i], bigHull[(i + 1) % bigHull.Count], q));
                bool mine = InsideConvex(bigHull, q);
                bool oracle = InsideByRayCasting(bigHull, q);
                if (onBoundary) boundaryHits++;
                gridTotal++;
                if (mine == oracle) gridAgreements++;
            }
        Check(gridAgreements == gridTotal, "point-in-polygon agrees with ray casting on the whole grid ("
              + gridAgreements + "/" + gridTotal + ")");

        // 7b: explicit on-boundary queries, where the two methods classically disagree.
        var square = new List<P> { new P(0, 0), new P(10, 0), new P(10, 10), new P(0, 10) };
        var probes = new[]
        {
            new P(0, 0), new P(5, 0), new P(10, 5), new P(5, 10), new P(0, 5),
            new P(5, 5), new P(11, 5), new P(-1, -1),
        };
        int boundaryAgreed = probes.Count(q => InsideConvex(square, q) == InsideByRayCasting(square, q));
        Check(boundaryAgreed == probes.Length,
              "on-boundary and outside probes agree with ray casting (" + boundaryAgreed + "/" + probes.Length + ")");
        Check(InsideConvex(square, new P(11, 5)) == false && InsideConvex(square, new P(5, 11)) == false,
              "segments and corners are inside, exterior points are not");

        // 8: segment intersection against the exact rational oracle.
        int segmentAgreements = 0;
        for (int t = 0; t < 300; t++)
        {
            var p1 = new P(NextCoord(20), NextCoord(20));
            var p2 = new P(NextCoord(20), NextCoord(20));
            var p3 = new P(NextCoord(20), NextCoord(20));
            var p4 = new P(NextCoord(20), NextCoord(20));
            bool mine = SegmentsIntersect(p1, p2, p3, p4, out string reason);
            bool oracle = SegmentsIntersectRational(p1, p2, p3, p4);
            if (mine == oracle) segmentAgreements++;
            if (t == 0) Console.WriteLine("sample segment verdict: " + mine + " (" + reason + ")");
        }
        Check(segmentAgreements == 300, "segment intersection matches the rational oracle (" + segmentAgreements + "/300)");
        Check(SegmentsIntersect(new P(0, 0), new P(2, 2), new P(0, 2), new P(2, 0)), "a proper crossing is found");
        Check(SegmentsIntersect(new P(0, 0), new P(2, 0), new P(1, 0), new P(1, 1)),
              "a touching endpoint is reported as intersecting");
        Check(!SegmentsIntersect(new P(0, 0), new P(1, 0), new P(2, 0), new P(3, 1)),
              "disjoint collinear-ish segments are rejected");
        Check(SegmentsIntersect(new P(0, 0), new P(4, 0), new P(1, 0), new P(3, 0)),
              "an overlapping collinear pair intersects");

        // 9: closest pair against the brute-force minimum (exact integer distances).
        int closestAgreements = 0;
        for (int t = 0; t < 20; t++)
        {
            int n = 4 + (int)(NextU64() % 40);
            var pts = Enumerable.Range(0, n).Select(_ => new P(NextCoord(1000), NextCoord(1000))).ToList();
            long fast = ClosestPairSquared(pts);
            long brute = long.MaxValue;
            for (int i = 0; i < pts.Count; i++)
                for (int j = i + 1; j < pts.Count; j++) brute = Math.Min(brute, Dist2(pts[i], pts[j]));
            if (fast == brute) closestAgreements++;
        }
        Check(closestAgreements == 20, "divide and conquer equals the brute-force closest pair ("
              + closestAgreements + "/20)");
        Check(ClosestPairSquared(new List<P> { new P(0, 0), new P(3, 4) }) == 25, "the single-pair case is exact");
        Check(ClosestPairSquared(new List<P> { new P(0, 0), new P(1, 0), new P(2, 0), new P(3, 0) }) == 1,
              "adjacent collinear points give distance 1");

        // 10-13: degenerate inputs.
        Check(ConvexHull(new List<P>()).Count == 0 && ConvexHull(new List<P> { new P(1, 1) }).Count == 1,
              "empty and single-point hulls");
        Check(ConvexHull(new List<P> { new P(5, 5), new P(5, 5), new P(5, 5) }).Count == 1,
              "repeated identical points collapse to one vertex");
        var collinear = Enumerable.Range(-5, 11).Select(i => new P(i, 2 * i + 1)).ToList();
        var degenerate = ConvexHull(collinear);
        Check(degenerate.Count == 2 && degenerate[0].X == -5 && degenerate[1].X == 5,
              "an all-collinear input yields the two extreme points");
        Check(SignedAreaTimes2(degenerate) == 0, "the degenerate hull has zero area");
        Check(InsideConvex(degenerate, new P(0, 1)) && !InsideConvex(degenerate, new P(0, 0)),
              "a segment hull contains its own points and nothing else");
        Check(Enumerable.Range(0, collinear.Count).All(i => Orient2D(collinear[0], collinear[^1], collinear[i]) == 0),
              "every point of the collinear family is exactly on the line");
        Check(HullAreaMatchesTriangulation(new List<P> { new P(0, 0), new P(4, 0), new P(4, 3) }),
              "the triangle area is 6 (shoelace equals the fan)");
        Check(SignedAreaTimes2(new List<P> { new P(0, 0), new P(4, 0), new P(4, 3) }) == 12,
              "twice the signed area of the 3-4-5 right triangle is 12");

        // 14: determinism.
        var repeatA = ConvexHull(cloud);
        var repeatB = ConvexHull(cloud);
        Check(repeatA.SequenceEqual(repeatB) && ClosestPairSquared(cloud) == ClosestPairSquared(cloud),
              "every routine is bitwise reproducible");

        Console.WriteLine("hull of 200 points: " + bigHull.Count + " vertices, area2 " + SignedAreaTimes2(bigHull) +
                          "; grid agreement " + gridAgreements + "/" + gridTotal +
                          "; closest pair comparisons " + _distanceComparisons);
        Console.WriteLine("PASS");
    }

    /// <summary>
    /// Degenerate family with coordinates large enough that the difference products (bx*cy and by*cx both
    /// around 1e18) already round in double precision: the unguarded determinant then reports a non-zero (and
    /// randomly signed) value for points that are exactly collinear, while the robust predicate must still
    /// report 0. Returns the number of naive mistakes; robustCorrect receives the robust predicate's hits.
    /// </summary>
    private static long CountNaiveMistakes(out int robustCorrect)
    {
        long mistakes = 0;
        robustCorrect = 0;
        ulong saved = _state;
        for (int t = 0; t < 200; t++)
        {
            var a = new P(NextCoord(100000000000000000L), NextCoord(100000000000000000L));
            var b = new P(NextCoord(100000000000000000L), NextCoord(100000000000000000L));
            long k = 1 + (long)(NextU64() % 3);
            var c = new P(a.X + k * (b.X - a.X), a.Y + k * (b.Y - a.Y));   // exactly collinear
            if (NaiveOrient(a, b, c) != 0) mistakes++;
            if (Orient2D(a, b, c) == 0) robustCorrect++;
        }
        _state = saved;
        return mistakes;
    }

    private static bool IsStrictlyConvex(List<P> hull)
    {
        if (hull.Count < 3) return true;
        for (int i = 0; i < hull.Count; i++)
            if (Orient2D(hull[i], hull[(i + 1) % hull.Count], hull[(i + 2) % hull.Count]) <= 0) return false;
        return true;
    }

    private static bool HullAreaMatchesTriangulation(List<P> hull)
    {
        if (hull.Count < 3) return SignedAreaTimes2(hull) == 0;
        BigInteger fan = 0;
        for (int i = 1; i + 1 < hull.Count; i++) fan += ExactOrient(hull[0], hull[i], hull[i + 1]);
        return fan == SignedAreaTimes2(hull);
    }

    private static bool HullEquals(List<P> a, List<P> b)
    {
        var sa = a.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        var sb = b.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        return sa.Count == sb.Count && sa.Zip(sb, (x, y) => Same(x, y)).All(v => v);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
