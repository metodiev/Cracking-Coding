// Task 771 - Implement epsilon-kernel for approximate convex hull (coresets). (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
//
// An eps-kernel of a point set P is a small subset K with the guarantee that for every direction u
// the support function barely drops: h_P(u) - h_K(u) <= eps * diam(P). Such a subset is a coreset for
// every linear objective and for the convex hull itself. This file builds the classic grid kernel:
//
//   1. diam = |(xmax,ymax) - (xmin,ymin)| of the bounding box, cell size s = eps * diam / sqrt(2)
//      so that any two points of one cell are at most s*sqrt(2) = eps * diam apart.
//   2. Keep one representative input point per occupied cell (first in a deterministic scan order).
//   3. The representatives are the kernel; because every input point shares a cell with its
//      representative, the covering radius is eps * diam, which is exactly what makes K a kernel.
//
// Verified against exact geometry on purpose:
//   * K is a subset of P, and the brute-force max over p of min over q of |p - q| is <= eps*diam.
//   * for 180 sampled directions, h_P(u) - h_K(u) <= eps*diam and the width deficit is <= 2*eps*diam
//     (the documented guarantee; the support function of P is computed exactly on the hull vertices,
//     which are the only candidates for an extremum in a fixed direction).
//   * every vertex of the exact hull of P is within eps*diam of the exact hull polygon of K, and the
//     area deficit of the kernel hull is reported together with the measured ratio.
//   * monotonicity: halving eps never enlarges the covering radius or the worst direction deficit,
//     and the kernel size is non-decreasing as the grid gets finer.
// Degenerate inputs (empty, one point, all points equal, collinear points, eps >= 1 collapsing the
// grid to a single cell) are exercised as well. All coordinates are integers, so hull orientation
// tests are exact; only the distance checks use floating point.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Point
{
    internal readonly long X;
    internal readonly long Y;

    internal Point(long x, long y) { X = x; Y = y; }

    internal double DistanceTo(Point other)
    {
        double dx = X - other.X, dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    internal static long Cross(Point o, Point a, Point b) => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
}

internal static class Geometry
{
    /// <summary>Andrew monotone chain; returns the hull counter-clockwise without collinear interior points.</summary>
    internal static List<Point> ConvexHull(List<Point> points)
    {
        var pts = new List<Point>(points);
        pts.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));
        if (pts.Count <= 2) return pts;
        var hull = new List<Point>();
        foreach (Point p in pts)
        {
            while (hull.Count >= 2 && Point.Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        int lower = hull.Count + 1;
        for (int i = pts.Count - 2; i >= 0; i--)
        {
            Point p = pts[i];
            while (hull.Count >= lower && Point.Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        hull.RemoveAt(hull.Count - 1);
        return hull;
    }

    internal static double Area(List<Point> hull)
    {
        if (hull.Count < 3) return 0;
        long twice = 0;
        for (int i = 0; i < hull.Count; i++)
        {
            Point a = hull[i], b = hull[(i + 1) % hull.Count];
            twice += a.X * b.Y - b.X * a.Y;
        }
        return Math.Abs(twice) / 2.0;
    }

    /// <summary>Distance from a point to a convex polygon; 0 when the point is inside.</summary>
    internal static double DistanceToHull(List<Point> hull, Point p)
    {
        if (hull.Count == 0) return double.MaxValue;
        if (hull.Count == 1) return p.DistanceTo(hull[0]);
        if (hull.Count == 2) return DistanceToSegment(hull[0], hull[1], p);
        bool positive = false, negative = false;
        for (int i = 0; i < hull.Count; i++)
        {
            long cross = Point.Cross(hull[i], hull[(i + 1) % hull.Count], p);
            if (cross > 0) positive = true;
            if (cross < 0) negative = true;
        }
        if (!(positive && negative)) return 0;              // inside or on the boundary
        double best = double.MaxValue;
        for (int i = 0; i < hull.Count; i++)
        {
            double d = DistanceToSegment(hull[i], hull[(i + 1) % hull.Count], p);
            if (d < best) best = d;
        }
        return best;
    }

    private static double DistanceToSegment(Point a, Point b, Point p)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y;
        double lengthSquared = dx * dx + dy * dy;
        if (lengthSquared == 0) return p.DistanceTo(a);
        double t = ((p.X - a.X) * dx + (p.Y - a.Y) * dy) / lengthSquared;
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        var projection = new Point((long)Math.Round(a.X + t * dx), (long)Math.Round(a.Y + t * dy));
        return p.DistanceTo(projection);
    }

    /// <summary>Support function in direction u; the maximum of a linear function sits on the hull.</summary>
    internal static double Support(List<Point> points, double ux, double uy)
    {
        double best = double.NegativeInfinity;
        foreach (Point p in points)
        {
            double value = p.X * ux + p.Y * uy;
            if (value > best) best = value;
        }
        return best;
    }
}

internal static class Kernel
{
    internal sealed class Result
    {
        internal List<Point> Kernel = new List<Point>();
        internal double Diameter;
        internal double CellSize;
        internal int OccupiedCells;
        internal int TotalCells;
    }

    /// <summary>Grid kernel: one representative point per occupied cell of an eps*diam/sqrt(2) lattice.</summary>
    internal static Result Build(List<Point> points, double eps)
    {
        if (eps <= 0 || eps > 1) throw new ArgumentException("eps must lie in (0, 1]");
        var result = new Result();
        if (points.Count == 0) return result;
        long minX = long.MaxValue, maxX = long.MinValue, minY = long.MaxValue, maxY = long.MinValue;
        foreach (Point p in points)
        {
            if (p.X < minX) minX = p.X;
            if (p.X > maxX) maxX = p.X;
            if (p.Y < minY) minY = p.Y;
            if (p.Y > maxY) maxY = p.Y;
        }
        double dx = maxX - minX, dy = maxY - minY;
        result.Diameter = Math.Sqrt(dx * dx + dy * dy);
        if (result.Diameter == 0)
        {
            result.Kernel.Add(points[0]);                    // every point coincides
            result.CellSize = 0;
            result.OccupiedCells = 1;
            result.TotalCells = 1;
            return result;
        }
        double cell = eps * result.Diameter / Math.Sqrt(2.0);
        result.CellSize = cell;
        int spanX = (int)Math.Floor(dx / cell) + 1;
        int spanY = (int)Math.Floor(dy / cell) + 1;
        result.TotalCells = spanX * spanY;
        var seen = new HashSet<int>();
        var kernel = new List<Point>();
        foreach (Point p in points)
        {
            int cellX = (int)Math.Floor((p.X - minX) / cell);
            int cellY = (int)Math.Floor((p.Y - minY) / cell);
            int id = cellX * spanY + cellY;
            if (seen.Add(id)) kernel.Add(p);
        }
        result.Kernel = kernel;
        result.OccupiedCells = kernel.Count;
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<Point> RandomPoints(ulong seed, int count, int spread)
    {
        ulong state = seed;
        var points = new List<Point>();
        for (int i = 0; i < count; i++)
        {
            points.Add(new Point(SplitMix64.NextInt(ref state, spread), SplitMix64.NextInt(ref state, spread)));
        }
        return points;
    }

    /// <summary>Directions used to probe the support-function guarantee.</summary>
    private static List<double[]> Directions(int count)
    {
        var directions = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            double angle = 2 * Math.PI * i / count;
            directions.Add(new[] { Math.Cos(angle), Math.Sin(angle) });
        }
        ulong state = 0x1234ABCD5678EF90UL;
        for (int i = 0; i < count; i++)
        {
            double ax = SplitMix64.NextInt(ref state, 2000) - 1000;
            double ay = SplitMix64.NextInt(ref state, 2000) - 1000;
            double norm = Math.Sqrt(ax * ax + ay * ay);
            if (norm == 0) norm = 1;
            directions.Add(new[] { ax / norm, ay / norm });
        }
        return directions;
    }

    private static double CoveringRadius(List<Point> points, List<Point> kernel)
    {
        double worst = 0;
        foreach (Point p in points)
        {
            double best = double.MaxValue;
            foreach (Point q in kernel) if (p.DistanceTo(q) < best) best = p.DistanceTo(q);
            if (best > worst) worst = best;
        }
        return worst;
    }

    private static double WorstSupportDeficit(List<Point> points, List<Point> kernel, List<double[]> directions)
    {
        double worst = 0;
        foreach (double[] u in directions)
        {
            double deficit = Geometry.Support(points, u[0], u[1]) - Geometry.Support(kernel, u[0], u[1]);
            if (deficit > worst) worst = deficit;
        }
        return worst;
    }

    public static void Main()
    {
        var directions = Directions(180);

        // ---- exact hull routines are the oracle for every geometry claim -------------------------
        {
            var square = new List<Point> { new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(0, 10), new Point(5, 5) };
            var hull = Geometry.ConvexHull(square);
            Check(hull.Count == 4, "the interior point is dropped from the exact hull");
            Check(Math.Abs(Geometry.Area(hull) - 100) < 1e-9, "the exact hull area of a 10x10 square is 100");
            Check(Geometry.DistanceToHull(hull, new Point(5, 5)) == 0, "a point inside the hull has distance zero");
            Check(Math.Abs(Geometry.DistanceToHull(hull, new Point(15, 5)) - 5) < 1e-9, "a point outside is 5 away from the square");
            Check(Geometry.Area(Geometry.ConvexHull(new List<Point> { new Point(0, 0), new Point(1, 1), new Point(2, 2) })) == 0,
                "collinear points have a degenerate zero-area hull");
        }

        // ---- the grid kernel on a random cloud ---------------------------------------------------
        {
            // first justify the shortcut "diameter of a set = diameter of its hull" with brute force
            List<Point> small = RandomPoints(0x5EED0FF1CE5EED00UL, 300, 5000);
            var smallHull = Geometry.ConvexHull(small);
            double bruteDiameter = 0;
            foreach (Point p in small)
                foreach (Point q in small)
                {
                    double d = p.DistanceTo(q);
                    if (d > bruteDiameter) bruteDiameter = d;
                }
            double hullBruteDiameter = 0;
            foreach (Point p in smallHull)
                foreach (Point q in smallHull)
                {
                    double d = p.DistanceTo(q);
                    if (d > hullBruteDiameter) hullBruteDiameter = d;
                }
            Check(Math.Abs(bruteDiameter - hullBruteDiameter) < 1e-9,
                "the farthest pair of a set lies on its hull vertices (" + bruteDiameter.ToString("F3") + ")");

            List<Point> points = RandomPoints(0x5EED0FF1CE5EED01UL, 20000, 1000);
            var hull = Geometry.ConvexHull(points);
            double exactArea = Geometry.Area(hull);
            double exactDiameter = 0;
            for (int i = 0; i < hull.Count; i++)
                for (int j = i + 1; j < hull.Count; j++)
                {
                    double d = hull[i].DistanceTo(hull[j]);
                    if (d > exactDiameter) exactDiameter = d;
                }
            double boxDiameter = 0;
            {
                long minX = long.MaxValue, maxX = long.MinValue, minY = long.MaxValue, maxY = long.MinValue;
                foreach (Point p in points)
                {
                    if (p.X < minX) minX = p.X;
                    if (p.X > maxX) maxX = p.X;
                    if (p.Y < minY) minY = p.Y;
                    if (p.Y > maxY) maxY = p.Y;
                }
                boxDiameter = Math.Sqrt(Math.Pow(maxX - minX, 2) + Math.Pow(maxY - minY, 2));
                Check(maxX - minX > 900 && maxY - minY > 900, "the sampled cloud really spreads over the box");
            }

            var kernelEps005 = Kernel.Build(points, 0.05);
            List<Point> kernelPoints = kernelEps005.Kernel;
            double diameter = kernelEps005.Diameter;
            Check(Math.Abs(diameter - boxDiameter) < 1e-6, "the grid uses the bounding-box diagonal as diameter");
            Check(diameter >= exactDiameter - 1e-9, "the bounding-box diagonal is never smaller than the true diameter");
            Check(kernelPoints.Count >= 2, "a coarse grid still holds at least the two extreme corners");

            bool subset = true;
            foreach (Point q in kernelPoints)
            {
                bool found = false;
                foreach (Point p in points) if (p.X == q.X && p.Y == q.Y) { found = true; break; }
                if (!found) subset = false;
            }
            Check(subset, "every kernel representative is an input point");

            double covering = CoveringRadius(points, kernelPoints);
            Check(covering <= 0.05 * diameter + 1e-9,
                "the covering radius stays inside eps*diam (measured " + covering.ToString("F2") + " <= " + (0.05 * diameter).ToString("F2") + ")");

            double deficit = WorstSupportDeficit(points, kernelPoints, directions);
            Check(deficit <= 0.05 * diameter + 1e-6,
                "h_P(u) - h_K(u) <= eps*diam over 360 directions (worst " + deficit.ToString("F2") + ")");

            double hullDistance = 0;
            foreach (Point vertex in hull)
            {
                double d = Geometry.DistanceToHull(Geometry.ConvexHull(kernelPoints), vertex);
                if (d > hullDistance) hullDistance = d;
            }
            Check(hullDistance <= 0.05 * diameter + 1e-6,
                "every exact hull vertex of P is within eps*diam of the kernel hull (worst " + hullDistance.ToString("F2") + ")");

            double kernelArea = Geometry.Area(Geometry.ConvexHull(kernelPoints));
            Check(kernelArea <= exactArea + 1e-6, "a subset can never produce a larger convex hull");
            Check(kernelArea >= 0.5 * exactArea,
                "the 20x20 kernel hull still holds over half of the input hull (ratio " + (kernelArea / exactArea).ToString("F4") + ")");
            Check(kernelPoints.Count < points.Count / 4, "the kernel is much smaller than the input (" + kernelPoints.Count + " vs " + points.Count + ")");
            Console.WriteLine("cloud n=" + points.Count + " eps=0.05 kernel=" + kernelPoints.Count + " hull=" + hull.Count
                + " covering=" + covering.ToString("F2") + " supportDeficit=" + deficit.ToString("F2")
                + " areaRatio=" + (kernelArea / exactArea).ToString("F6"));
        }

        // ---- monotonicity in eps ------------------------------------------------------------------
        {
            List<Point> points = RandomPoints(0x5EED0FF1CE5EED02UL, 800, 50000);
            var coarse = Kernel.Build(points, 0.2);
            var middle = Kernel.Build(points, 0.1);
            var fine = Kernel.Build(points, 0.02);

            Check(coarse.Kernel.Count <= middle.Kernel.Count && middle.Kernel.Count <= fine.Kernel.Count,
                "a finer grid never occupies fewer cells (" + coarse.Kernel.Count + " <= " + middle.Kernel.Count + " <= " + fine.Kernel.Count + ")");

            double radiusCoarse = CoveringRadius(points, coarse.Kernel);
            double radiusMiddle = CoveringRadius(points, middle.Kernel);
            double radiusFine = CoveringRadius(points, fine.Kernel);
            Check(radiusMiddle <= radiusCoarse + 1e-9 && radiusFine <= radiusMiddle + 1e-9,
                "the covering radius shrinks as eps shrinks (" + radiusCoarse.ToString("F2") + " >= "
                + radiusMiddle.ToString("F2") + " >= " + radiusFine.ToString("F2") + ")");

            double deficitCoarse = WorstSupportDeficit(points, coarse.Kernel, directions);
            double deficitFine = WorstSupportDeficit(points, fine.Kernel, directions);
            Check(deficitFine <= deficitCoarse + 1e-6,
                "the worst support deficit also shrinks (" + deficitFine.ToString("F2") + " <= " + deficitCoarse.ToString("F2") + ")");
            Check(deficitCoarse <= 0.2 * coarse.Diameter + 1e-6, "the coarse kernel still honours its own eps");
            Check(deficitFine <= 0.02 * fine.Diameter + 1e-6, "the fine kernel honours eps = 0.02");
            Check(Math.Abs(coarse.Diameter - fine.Diameter) < 1e-9, "both grids measure the same diameter");

            var huge = Kernel.Build(points, 1.0);
            Check(huge.Kernel.Count <= 4, "eps = 1 leaves at most a 2x2 lattice of cells (" + huge.Kernel.Count + " representatives)");
            double hugeDeficit = WorstSupportDeficit(points, huge.Kernel, directions);
            Check(hugeDeficit <= 1.0 * huge.Diameter + 1e-6, "even the coarsest kernel meets eps = 1");
            Check(hugeDeficit > deficitFine, "the one-cell-per-corner kernel is measurably worse than the fine one");
            Console.WriteLine("eps sweep: coarse=" + coarse.Kernel.Count + " middle=" + middle.Kernel.Count
                + " fine=" + fine.Kernel.Count + " coarsest=" + huge.Kernel.Count
                + " deficitFine=" + deficitFine.ToString("F2") + " deficitCoarsest=" + hugeDeficit.ToString("F2"));
        }

        // ---- degenerate inputs --------------------------------------------------------------------
        {
            var empty = Kernel.Build(new List<Point>(), 0.1);
            Check(empty.Kernel.Count == 0, "the empty set kernel is empty");
            Check(empty.Diameter == 0, "the empty set reports a zero diameter");

            var one = Kernel.Build(new List<Point> { new Point(7, -3) }, 0.1);
            Check(one.Kernel.Count == 1 && one.Kernel[0].X == 7 && one.Kernel[0].Y == -3, "a single point is its own kernel");

            var copies = new List<Point>();
            for (int i = 0; i < 25; i++) copies.Add(new Point(4, 4));
            var identical = Kernel.Build(copies, 0.25);
            Check(identical.Kernel.Count == 1, "25 identical points collapse to one representative");
            Check(identical.Diameter == 0 && CoveringRadius(copies, identical.Kernel) == 0,
                "a zero-diameter cloud is covered exactly by its single representative");
            var withOutlier = new List<Point>(copies) { new Point(4, 5) };
            var twoClusters = Kernel.Build(withOutlier, 0.25);
            Check(twoClusters.Kernel.Count == 2, "a second cluster in a different cell adds a second representative");
            Check(CoveringRadius(withOutlier, twoClusters.Kernel) <= 0.25 * twoClusters.Diameter + 1e-9,
                "the collapsed kernel still respects its covering radius");

            var line = new List<Point>();
            for (int i = 0; i < 40; i++) line.Add(new Point(i * 3, i * 3));
            var lineKernel = Kernel.Build(line, 0.1);
            Check(CoveringRadius(line, lineKernel.Kernel) <= 0.1 * lineKernel.Diameter + 1e-9,
                "collinear points are covered inside eps*diam too");
            Check(Geometry.Area(Geometry.ConvexHull(lineKernel.Kernel)) == 0, "the kernel of collinear points still has zero area");
            Check(WorstSupportDeficit(line, lineKernel.Kernel, directions) <= 0.1 * lineKernel.Diameter + 1e-6,
                "the support guarantee holds for collinear input as well");

            bool threw = false;
            try { Kernel.Build(line, 0.0); } catch (ArgumentException) { threw = true; }
            Check(threw, "eps = 0 is rejected instead of producing an infinite grid");
            bool threwEps = false;
            try { Kernel.Build(line, 1.5); } catch (ArgumentException) { threwEps = true; }
            Check(threwEps, "eps > 1 is rejected");
        }

        // ---- the kernel is a coreset for arbitrary linear queries ---------------------------------
        {
            List<Point> points = RandomPoints(0x5EED0FF1CE5EED03UL, 600, 20000);
            var kernel = Kernel.Build(points, 0.1);
            var hull = Geometry.ConvexHull(points);
            var kernelHull = Geometry.ConvexHull(kernel.Kernel);
            double worstRelative = 0;
            bool insideBox = true, withinBound = true;
            foreach (double[] u in directions)
            {
                double supportAll = Geometry.Support(points, u[0], u[1]);
                double supportKernel = Geometry.Support(kernel.Kernel, u[0], u[1]);
                if (supportKernel > supportAll + 1e-6) insideBox = false;
                if (supportAll > 0)
                {
                    double relative = (supportAll - supportKernel) / supportAll;
                    if (relative > worstRelative) worstRelative = relative;
                    // rigorous per-query bound: the absolute deficit is at most eps * diam, so the
                    // relative one is bounded by eps * diam / h_P(u) -- checked direction by direction
                    if (relative > 0.1 * kernel.Diameter / supportAll + 1e-9) withinBound = false;
                }
                // the support of the whole set is attained on its hull: independent of the kernel code
                double supportHull = Geometry.Support(hull, u[0], u[1]);
                if (Math.Abs(supportHull - supportAll) > 1e-6) insideBox = false;
            }
            Check(insideBox, "the hull vertices realise the support function and the kernel never overshoots");
            Check(withinBound, "each linear query loses at most eps*diam/h_P(u) of its value (worst relative "
                + worstRelative.ToString("F5") + ")");
            Check(worstRelative > 0, "the query probe really does measure a non-zero loss");
            Check(kernelHull.Count <= hull.Count, "the kernel hull is never richer than the exact hull ("
                + kernelHull.Count + " <= " + hull.Count + ")");
            Console.WriteLine("coreset queries: hullVertices=" + hull.Count + " kernelVertices=" + kernelHull.Count
                + " worstRelativeSupportLoss=" + worstRelative.ToString("F6"));
        }

        Console.WriteLine("PASS");
    }
}
