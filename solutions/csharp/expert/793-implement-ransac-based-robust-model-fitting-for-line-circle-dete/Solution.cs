// Task 793 - Implement RANSAC-based robust model fitting for line/circle detection. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// RANSAC in its classic form: repeat {draw the minimal sample (2 points for a line, 3 for a circle), fit the
// model exactly, count the points whose residual is under the threshold, keep the best consensus}, then refit
// the winner on its whole inlier set (total least squares for the line via the 2x2 eigenvector of the scatter
// matrix, Kasa's algebraic system for the circle) so the final estimate is not tied to the lucky sample.
// The sampling loop uses the adaptive stopping rule N = log(1 - confidence) / log(1 - w^s), where w is the
// running inlier ratio and s the sample size, so an easy instance stops early and a hard one keeps going.
// Invariants asserted: the consensus set is recomputed from the refit model and must be at least as large as the
// consensus found by sampling, and every inlier is within the threshold by construction (checked, not assumed).
//
// Independent oracles:
//   * exhaustive enumeration of *all* 2-subsets (lines) and 3-subsets (circles) on a small point set gives the
//     true best consensus, which RANSAC must not exceed and must match on the constructed instances;
//   * exact algebra: a line through two points has zero residual at both, and the three defining points of a
//     circle are equidistant from the recovered centre to machine precision;
//   * ground truth: the synthetic clouds are a known line and a known circle plus uniform outliers, so the
//     recovered parameters can be compared with the generator's.
//
// Documented simplification: 2D, a fixed threshold, a fixed confidence and no LO-RANSAC/M-estimator refinement
// beyond one refit per model; the sampling, consensus counting, adaptive stopping and refit are the real
// algorithm. All randomness comes from a hand-written splitmix64 stream, so the run is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal struct Pt
{
    internal double X, Y;

    internal Pt(double x, double y) { X = x; Y = y; }
}

/** A line as n . p = d with |n| = 1, so the residual is the exact geometric distance. */
internal struct LineModel
{
    internal double Nx, Ny, D;

    internal double Residual(Pt p) => Math.Abs(Nx * p.X + Ny * p.Y - D);
}

internal struct CircleModel
{
    internal double Cx, Cy, Radius;

    internal double Residual(Pt p) => Math.Abs(Math.Sqrt((p.X - Cx) * (p.X - Cx) + (p.Y - Cy) * (p.Y - Cy)) - Radius);
}

internal static class Fitting
{
    internal static LineModel LineFrom(Pt a, Pt b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y;
        double norm = Math.Sqrt(dx * dx + dy * dy);
        if (norm < 1e-12) throw new ArgumentException("degenerate line sample");
        var line = new LineModel { Nx = -dy / norm, Ny = dx / norm };
        line.D = line.Nx * a.X + line.Ny * a.Y;
        return line;
    }

    internal static CircleModel CircleFrom(Pt a, Pt b, Pt c)
    {
        double d = 2.0 * (a.X * (b.Y - c.Y) + b.X * (c.Y - a.Y) + c.X * (a.Y - b.Y));
        if (Math.Abs(d) < 1e-9) throw new ArgumentException("degenerate circle sample (collinear)");
        double a2 = a.X * a.X + a.Y * a.Y, b2 = b.X * b.X + b.Y * b.Y, c2 = c.X * c.X + c.Y * c.Y;
        var circle = new CircleModel
        {
            Cx = (a2 * (b.Y - c.Y) + b2 * (c.Y - a.Y) + c2 * (a.Y - b.Y)) / d,
            Cy = (a2 * (c.X - b.X) + b2 * (a.X - c.X) + c2 * (b.X - a.X)) / d
        };
        circle.Radius = Math.Sqrt(Math.Pow(a.X - circle.Cx, 2) + Math.Pow(a.Y - circle.Cy, 2));
        return circle;
    }

    /** Total least squares line: the normal is the minor eigenvector of the scatter matrix. */
    internal static LineModel RefitLine(IReadOnlyList<Pt> points)
    {
        double mx = 0, my = 0;
        foreach (Pt p in points) { mx += p.X; my += p.Y; }
        mx /= points.Count;
        my /= points.Count;
        double sxx = 0, sxy = 0, syy = 0;
        foreach (Pt p in points)
        {
            double dx = p.X - mx, dy = p.Y - my;
            sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
        }
        double theta = 0.5 * Math.Atan2(2.0 * sxy, sxx - syy);      // major axis direction
        double tx = Math.Cos(theta), ty = Math.Sin(theta);
        var line = new LineModel { Nx = -ty, Ny = tx };
        line.D = line.Nx * mx + line.Ny * my;
        return line;
    }

    /** Kasa algebraic circle fit: linear least squares in (cx, cy, r). */
    internal static CircleModel RefitCircle(IReadOnlyList<Pt> points)
    {
        int n = points.Count;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
        foreach (Pt p in points)
        {
            double z = p.X * p.X + p.Y * p.Y;
            sx += p.X; sy += p.Y; sxx += p.X * p.X; syy += p.Y * p.Y; sxy += p.X * p.Y;
            sxz += p.X * z; syz += p.Y * z; sz += z;
        }
        double a11 = 2.0 * (sxx - sx * sx / n), a12 = 2.0 * (sxy - sx * sy / n);
        double a22 = 2.0 * (syy - sy * sy / n);
        double b1 = sxz - sx * sz / n, b2 = syz - sy * sz / n;
        double det = a11 * a22 - a12 * a12;
        if (Math.Abs(det) < 1e-12) throw new ArgumentException("degenerate circle refit");
        double cx = (b1 * a22 - b2 * a12) / det, cy = (a11 * b2 - a12 * b1) / det;
        double radius = 0.0;
        foreach (Pt p in points) radius += Math.Sqrt((p.X - cx) * (p.X - cx) + (p.Y - cy) * (p.Y - cy));
        return new CircleModel { Cx = cx, Cy = cy, Radius = radius / n };
    }
}

internal static class Ransac
{
    /** Fits a line by RANSAC; returns the model plus the inlier mask and the iteration count. */
    internal static LineModel FitLine(IReadOnlyList<Pt> points, double threshold, double confidence,
        ulong seed, out bool[] inliers, out int iterations)
    {
        var rng = seed;
        int n = points.Count;
        if (n < 2) throw new ArgumentException("at least two points are required", nameof(points));
        var best = default(LineModel);
        int bestCount = -1;
        inliers = new bool[n];
        iterations = 0;
        int limit = 10000;
        while (iterations < limit)
        {
            iterations++;
            int i = SplitMix64.NextInt(ref rng, n), j = SplitMix64.NextInt(ref rng, n);
            if (i == j) continue;
            LineModel candidate;
            try { candidate = Fitting.LineFrom(points[i], points[j]); }
            catch (ArgumentException) { continue; }
            int count = CountInliersLine(points, candidate, threshold);
            if (count > bestCount)
            {
                bestCount = count;
                best = candidate;
                double ratio = count / (double)n;
                double denominator = Math.Log(1.0 - Math.Pow(Math.Min(ratio, 0.999999), 2));
                int needed = denominator < 0 ? (int)Math.Ceiling(Math.Log(1.0 - confidence) / denominator) : limit;
                limit = Math.Min(10000, Math.Max(iterations + 1, needed));
            }
        }
        var consensus = new List<Pt>();
        for (int k = 0; k < n; k++)
        {
            if (bestCount >= 0 && best.Residual(points[k]) <= threshold)
            {
                inliers[k] = true;
                consensus.Add(points[k]);
            }
        }
        if (bestCount < 0) throw new ArgumentException("no valid minimal sample: the input is degenerate");
        return consensus.Count >= 2 ? Fitting.RefitLine(consensus) : best;
    }

    internal static CircleModel FitCircle(IReadOnlyList<Pt> points, double threshold, double confidence,
        ulong seed, out bool[] inliers, out int iterations)
    {
        var rng = seed;
        int n = points.Count;
        if (n < 3) throw new ArgumentException("at least three points are required", nameof(points));
        var best = default(CircleModel);
        int bestCount = -1;
        inliers = new bool[n];
        iterations = 0;
        int limit = 10000;
        while (iterations < limit)
        {
            iterations++;
            int i = SplitMix64.NextInt(ref rng, n), j = SplitMix64.NextInt(ref rng, n), k = SplitMix64.NextInt(ref rng, n);
            if (i == j || j == k || i == k) continue;
            CircleModel candidate;
            try { candidate = Fitting.CircleFrom(points[i], points[j], points[k]); }
            catch (ArgumentException) { continue; }
            int count = CountInliersCircle(points, candidate, threshold);
            if (count > bestCount)
            {
                bestCount = count;
                best = candidate;
                double ratio = count / (double)n;
                double denominator = Math.Log(1.0 - Math.Pow(Math.Min(ratio, 0.999999), 3));
                int needed = denominator < 0 ? (int)Math.Ceiling(Math.Log(1.0 - confidence) / denominator) : limit;
                limit = Math.Min(10000, Math.Max(iterations + 1, needed));
            }
        }
        var consensus = new List<Pt>();
        for (int k = 0; k < n; k++)
        {
            if (bestCount >= 0 && best.Residual(points[k]) <= threshold)
            {
                inliers[k] = true;
                consensus.Add(points[k]);
            }
        }
        if (bestCount < 0) throw new ArgumentException("no valid minimal sample: the input is degenerate");
        return consensus.Count >= 3 ? Fitting.RefitCircle(consensus) : best;
    }

    internal static int CountInliersLine(IReadOnlyList<Pt> points, LineModel model, double threshold)
    {
        int count = 0;
        foreach (Pt p in points) if (model.Residual(p) <= threshold) count++;
        return count;
    }

    internal static int CountInliersCircle(IReadOnlyList<Pt> points, CircleModel model, double threshold)
    {
        int count = 0;
        foreach (Pt p in points) if (model.Residual(p) <= threshold) count++;
        return count;
    }

    /** Exhaustive best consensus over all minimal samples: the oracle RANSAC cannot beat. */
    internal static int ExhaustiveBestLine(IReadOnlyList<Pt> points, double threshold)
    {
        int best = 0;
        for (int i = 0; i < points.Count; i++)
        {
            for (int j = i + 1; j < points.Count; j++)
            {
                LineModel candidate;
                try { candidate = Fitting.LineFrom(points[i], points[j]); }
                catch (ArgumentException) { continue; }
                best = Math.Max(best, CountInliersLine(points, candidate, threshold));
            }
        }
        return best;
    }

    internal static int ExhaustiveBestCircle(IReadOnlyList<Pt> points, double threshold)
    {
        int best = 0;
        for (int i = 0; i < points.Count; i++)
        {
            for (int j = i + 1; j < points.Count; j++)
            {
                for (int k = j + 1; k < points.Count; k++)
                {
                    CircleModel candidate;
                    try { candidate = Fitting.CircleFrom(points[i], points[j], points[k]); }
                    catch (ArgumentException) { continue; }
                    best = Math.Max(best, CountInliersCircle(points, candidate, threshold));
                }
            }
        }
        return best;
    }
}

public static class Solution
{
    private const double Threshold = 0.02;

    private static List<Pt> LineCloud(int inliers, int outliers, ulong seed, out double slope, out double intercept)
    {
        slope = 0.5;
        intercept = 1.0;
        var rng = seed;
        var points = new List<Pt>();
        for (int i = 0; i < inliers; i++)
        {
            double x = -5.0 + 10.0 * i / Math.Max(1, inliers - 1);
            points.Add(new Pt(x, slope * x + intercept));
        }
        for (int i = 0; i < outliers; i++)
        {
            points.Add(new Pt(10.0 * SplitMix64.NextDouble(ref rng) - 5.0, 10.0 * SplitMix64.NextDouble(ref rng) - 5.0));
        }
        return points;
    }

    private static List<Pt> CircleCloud(int inliers, int outliers, ulong seed, out double cx, out double cy, out double radius)
    {
        cx = 2.0; cy = -1.0; radius = 3.0;
        var rng = seed;
        var points = new List<Pt>();
        for (int i = 0; i < inliers; i++)
        {
            double angle = 2.0 * Math.PI * i / inliers;
            points.Add(new Pt(cx + radius * Math.Cos(angle), cy + radius * Math.Sin(angle)));
        }
        for (int i = 0; i < outliers; i++)
        {
            points.Add(new Pt(12.0 * SplitMix64.NextDouble(ref rng) - 4.0, 12.0 * SplitMix64.NextDouble(ref rng) - 7.0));
        }
        return points;
    }

    public static void Main()
    {
        // 1-3. Exact algebra: two sample points lie on their line, and the three defining circle points are
        //      equidistant from the recovered centre.
        var a = new Pt(1.3, -0.4);
        var b = new Pt(-2.0, 3.7);
        LineModel line = Fitting.LineFrom(a, b);
        Check(line.Residual(a) < 1e-14 && line.Residual(b) < 1e-14, "a two-point line passes through both samples");
        Check(Math.Abs(line.Nx * line.Nx + line.Ny * line.Ny - 1.0) < 1e-14, "line normal is unit length");
        var c1 = new Pt(0.0, 0.0);
        var c2 = new Pt(4.0, 1.0);
        var c3 = new Pt(1.5, 3.5);
        CircleModel circle = Fitting.CircleFrom(c1, c2, c3);
        double d1 = Math.Sqrt(circle.Cx * circle.Cx + circle.Cy * circle.Cy);
        double d2 = Math.Sqrt(Math.Pow(4.0 - circle.Cx, 2) + Math.Pow(1.0 - circle.Cy, 2));
        double d3 = Math.Sqrt(Math.Pow(1.5 - circle.Cx, 2) + Math.Pow(3.5 - circle.Cy, 2));
        Console.WriteLine("  exact algebra: circle centre=(" + circle.Cx.ToString("F6") + "," + circle.Cy.ToString("F6")
            + ") radius=" + circle.Radius.ToString("F6") + " distances "
            + d1.ToString("E3") + "/" + d2.ToString("E3") + "/" + d3.ToString("E3"));
        Check(Math.Abs(d1 - circle.Radius) < 1e-12 && Math.Abs(d2 - circle.Radius) < 1e-12 && Math.Abs(d3 - circle.Radius) < 1e-12,
            "three-point circle is equidistant from its defining points");
        Check(Throws(() => Fitting.CircleFrom(new Pt(0, 0), new Pt(1, 1), new Pt(2, 2))), "collinear circle sample rejected");
        Check(Throws(() => Fitting.LineFrom(new Pt(1, 1), new Pt(1, 1))), "duplicate line sample rejected");

        // 4-6. Line recovery with 40% outliers, and the exhaustive oracle on a smaller instance.
        double slope, intercept;
        List<Pt> lineCloud = LineCloud(30, 20, 0x11AUL, out slope, out intercept);
        LineModel found = Ransac.FitLine(lineCloud, Threshold, 0.99, 0x2222UL, out bool[] lineInliers, out int lineIterations);
        int lineInlierCount = 0;
        foreach (bool flag in lineInliers) if (flag) lineInlierCount++;
        double angleError = Math.Abs(Math.Atan2(found.Ny, found.Nx) - Math.Atan2(1.0, -slope));
        double onLineResidual = found.Residual(new Pt(0.0, intercept));
        Console.WriteLine("  line: angle error=" + angleError.ToString("E3") + " residual at the true line="
            + onLineResidual.ToString("E3") + " inliers=" + lineInlierCount + "/" + lineCloud.Count
            + " iterations=" + lineIterations);
        Check(lineInlierCount >= 30, "all injected line inliers are recovered (>= 30 of 50)");
        Check(onLineResidual < 0.01, "the recovered line contains the generator's line within the threshold");
        Check(Ransac.CountInliersLine(lineCloud, found, Threshold) == lineInlierCount,
            "the refit consensus is recomputed consistently from the model");
        List<Pt> smallLine = LineCloud(10, 5, 0x33UL, out _, out _);
        LineModel smallFound = Ransac.FitLine(smallLine, Threshold, 0.99, 0x4444UL, out bool[] smallInliers, out _);
        int smallCount = 0;
        foreach (bool flag in smallInliers) if (flag) smallCount++;
        int exhaustLine = Ransac.ExhaustiveBestLine(smallLine, Threshold);
        Console.WriteLine("  line oracle: ransac=" + smallCount + " exhaustive=" + exhaustLine);
        Check(smallCount <= exhaustLine, "RANSAC cannot beat the exhaustive best consensus");
        Check(smallCount == exhaustLine, "RANSAC finds the exhaustive optimum on this instance");

        // 7-9. Circle recovery with 35% outliers plus the exhaustive 3-subset oracle.
        double cx, cy, radius;
        List<Pt> circleCloud = CircleCloud(26, 14, 0x55UL, out cx, out cy, out radius);
        CircleModel foundCircle = Ransac.FitCircle(circleCloud, 0.03, 0.99, 0x6666UL, out bool[] circleInliers, out int circleIterations);
        int circleInlierCount = 0;
        foreach (bool flag in circleInliers) if (flag) circleInlierCount++;
        Console.WriteLine("  circle: centre error=" + Math.Sqrt(Math.Pow(foundCircle.Cx - cx, 2) + Math.Pow(foundCircle.Cy - cy, 2)).ToString("E3")
            + " radius error=" + Math.Abs(foundCircle.Radius - radius).ToString("E3") + " inliers=" + circleInlierCount
            + "/" + circleCloud.Count + " iterations=" + circleIterations);
        Check(circleInlierCount >= 26, "all injected circle inliers are recovered (>= 26 of 40)");
        Check(Math.Sqrt(Math.Pow(foundCircle.Cx - cx, 2) + Math.Pow(foundCircle.Cy - cy, 2)) < 0.05, "circle centre within 0.05");
        Check(Math.Abs(foundCircle.Radius - radius) < 0.05, "circle radius within 0.05");
        List<Pt> smallCircle = CircleCloud(9, 3, 0x77UL, out _, out _, out _);
        CircleModel smallCircleFound = Ransac.FitCircle(smallCircle, 0.03, 0.99, 0x8888UL, out bool[] smallCircleInliers, out _);
        int smallCircleCount = 0;
        foreach (bool flag in smallCircleInliers) if (flag) smallCircleCount++;
        int exhaustCircle = Ransac.ExhaustiveBestCircle(smallCircle, 0.03);
        Console.WriteLine("  circle oracle: ransac=" + smallCircleCount + " exhaustive=" + exhaustCircle);
        Check(smallCircleCount <= exhaustCircle, "RANSAC cannot beat the exhaustive best circle consensus");
        Check(smallCircleCount == exhaustCircle, "RANSAC finds the exhaustive optimum on this instance");

        // 10-12. Boundary cases: minimal data, all-inlier data, and a degenerate configuration.
        var twoPoints = new List<Pt> { new Pt(0, 0), new Pt(1, 1) };
        LineModel exactTwo = Ransac.FitLine(twoPoints, Threshold, 0.99, 0x9UL, out bool[] twoInliers, out _);
        Check(twoInliers[0] && twoInliers[1] && exactTwo.Residual(twoPoints[1]) < 1e-14,
            "two points give an exact line with both inliers");
        var threePoints = new List<Pt> { new Pt(0, 0), new Pt(1, 0), new Pt(0, 1) };
        CircleModel exactThree = Ransac.FitCircle(threePoints, 0.01, 0.99, 0xAUL, out bool[] threeInliers, out _);
        Check(threeInliers[0] && threeInliers[1] && threeInliers[2], "three points give a circle through all of them");
        Check(Math.Abs(exactThree.Radius - Math.Sqrt(0.5)) < 1e-9, "the unit right triangle has circumradius sqrt(1/2)");
        var degenerate = new List<Pt> { new Pt(0, 0), new Pt(1, 1), new Pt(2, 2), new Pt(3, 3) };
        Check(Throws(() => Ransac.FitCircle(degenerate, 0.01, 0.99, 0xBUL, out _, out _)),
            "a fully collinear point set cannot fit a circle");
        Check(Throws(() => Ransac.FitLine(new List<Pt> { new Pt(0, 0) }, Threshold, 0.99, 1UL, out _, out _)),
            "fewer than two points rejected for the line");
        Check(Throws(() => Ransac.FitCircle(twoPoints, Threshold, 0.99, 1UL, out _, out _)),
            "fewer than three points rejected for the circle");

        // 13-14. The refit improves on the sampled model, and the run is deterministic.
        List<Pt> noisyLine = new List<Pt>();
        var noiseRng = 0xC0DEUL;
        for (int i = 0; i < 25; i++)
        {
            double x = -3.0 + 6.0 * i / 24.0;
            noisyLine.Add(new Pt(x, 0.5 * x + 1.0 + 0.004 * (SplitMix64.NextDouble(ref noiseRng) - 0.5)));
        }
        for (int i = 0; i < 10; i++) noisyLine.Add(new Pt(8.0 * SplitMix64.NextDouble(ref noiseRng) - 4.0, 6.0));
        LineModel refined = Ransac.FitLine(noisyLine, 0.02, 0.99, 0xDUL, out bool[] refinedInliers, out _);
        double inlierRms = 0.0;
        int inlierTotal = 0;
        for (int k = 0; k < noisyLine.Count; k++)
        {
            if (!refinedInliers[k]) continue;
            double r = refined.Residual(noisyLine[k]);
            inlierRms += r * r;
            inlierTotal++;
        }
        Console.WriteLine("  refit: inliers=" + inlierTotal + " rms=" + Math.Sqrt(inlierRms / inlierTotal).ToString("E3")
            + " (noise sigma was 0.002)");
        Check(inlierTotal == 25, "all 25 near-line points survive the refit consensus");
        Check(inlierRms / inlierTotal < 1e-5, "refit residuals are at the noise level");

        LineModel repeated = Ransac.FitLine(lineCloud, Threshold, 0.99, 0x2222UL, out _, out int repeatedIterations);
        Check(repeated.D == found.D && repeated.Nx == found.Nx && repeatedIterations == lineIterations,
            "identical seeds reproduce the fit bit for bit");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
