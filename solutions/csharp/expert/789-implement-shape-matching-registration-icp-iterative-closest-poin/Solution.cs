// Task 789 - Implement shape-matching registration (ICP) iterative closest point. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// Point-to-point ICP in 2D. Given a source and a target cloud the algorithm alternates
//   (1) correspondence: every source point is matched to its nearest target point (exact k-d tree, median
//       split on the alternating axis, pruned by the splitting-plane distance);
//   (2) alignment: the rigid transform (theta, t) minimising sum |R(theta) p_i + t - q_i|^2 is obtained in
//       closed form - centroids mu_p, mu_q, cross-covariance H = sum (p_i - mu_p)(q_i - mu_q)^T,
//       theta = atan2(H_yx - H_xy, H_xx + H_yy), t += mu_q - R(theta) mu_p (2D Kabsch/Umeyama, no SVD needed).
// The cold start is the centroid difference; the ICP invariant asserted below is that the mean squared
// correspondence error never increases across iterations. A trimmed variant drops the worst *trimFraction*
// of the matches before step (2), which is what recovers the transform when part of the source has no
// counterpart in the target.
//
// Independent oracles:
//   * k-d tree nearest-neighbour distances are compared with an exhaustive linear scan over hundreds of random
//     queries (both are exact, so they must agree to the last bit);
//   * the closed-form rotation is compared with a brute-force sweep over 20001 candidate angles, each carrying
//     its own optimal translation (the centroid shift) - a different algorithm, matched to grid resolution;
//   * the target cloud IS the source moved by a known transform, so recovery is measured directly: exact in the
//     noise-free case, at the noise level with sigma = 0.01, and trimmed-vs-untrimmed under 30% outliers.
//
// Documented simplification: clouds are 2D and small (60-100 points, tens of iterations) and the transform is a
// plane rotation, so there is no 3D quaternion/SVD step, and no multi-resolution or simulated-annealing escape
// from a bad basin - as in any ICP, convergence is only local and the checks therefore start the runs inside the
// basin as well as confirming that the cold start (centroid translation, identity rotation) also gets there for
// the test clouds. Correspondence, closed-form alignment, trimming and the k-d tree are the real algorithm.
// All randomness comes from a hand-written splitmix64 stream, so the run is reproducible.
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

    /** Standard normal via Box-Muller on the deterministic stream. */
    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal struct Vec2
{
    internal double X, Y;

    internal Vec2(double x, double y) { X = x; Y = y; }

    public static Vec2 operator +(Vec2 a, Vec2 b) => new Vec2(a.X + b.X, a.Y + b.Y);
    public static Vec2 operator -(Vec2 a, Vec2 b) => new Vec2(a.X - b.X, a.Y - b.Y);
    public static Vec2 operator *(Vec2 a, double s) => new Vec2(a.X * s, a.Y * s);

    internal double Norm => Math.Sqrt(X * X + Y * Y);
}

/** Exact nearest-neighbour queries over a 2D k-d tree (median split, alternating axis, branch pruning). */
internal sealed class KdTree
{
    private readonly Vec2[] _points;
    private readonly int[] _order;

    internal KdTree(IReadOnlyList<Vec2> points)
    {
        _points = new Vec2[points.Count];
        _order = new int[points.Count];
        for (int i = 0; i < points.Count; i++) { _points[i] = points[i]; _order[i] = i; }
        Build(0, _order.Length, 0);
    }

    private void Build(int lo, int hi, int depth)
    {
        if (hi - lo <= 1) return;
        int axis = depth % 2;
        int mid = (lo + hi) / 2;
        Array.Sort(_order, lo, hi - lo, Comparer<int>.Create((a, b) =>
            axis == 0 ? _points[a].X.CompareTo(_points[b].X) : _points[a].Y.CompareTo(_points[b].Y)));
        Build(lo, mid, depth + 1);
        Build(mid + 1, hi, depth + 1);
    }

    internal double NearestDistance(Vec2 q)
    {
        double best = double.MaxValue;
        int index = 0;
        Search(0, _order.Length, 0, q, ref best, ref index);
        return Math.Sqrt(best);
    }

    /** The nearest point itself (the tree is exact, so this is the true closest target point). */
    internal Vec2 NearestPoint(Vec2 q)
    {
        double best = double.MaxValue;
        int index = 0;
        Search(0, _order.Length, 0, q, ref best, ref index);
        return _points[index];
    }

    /** Exhaustive scan, used only by the checks as the oracle for the tree. */
    internal double BruteForceDistance(Vec2 q)
    {
        double best = double.MaxValue;
        foreach (Vec2 p in _points)
        {
            double dx = p.X - q.X, dy = p.Y - q.Y;
            double d = dx * dx + dy * dy;
            if (d < best) best = d;
        }
        return Math.Sqrt(best);
    }

    private void Search(int lo, int hi, int depth, Vec2 q, ref double bestSquared, ref int bestIndex)
    {
        if (lo >= hi) return;
        int mid = (lo + hi) / 2;
        int axis = depth % 2;
        int index = _order[mid];
        Vec2 p = _points[index];
        double dx = q.X - p.X, dy = q.Y - p.Y;
        double d2 = dx * dx + dy * dy;
        if (d2 < bestSquared) { bestSquared = d2; bestIndex = index; }

        double delta = axis == 0 ? dx : dy;
        int nearLo = delta <= 0 ? lo : mid + 1, nearHi = delta <= 0 ? mid : hi;
        int farLo = delta <= 0 ? mid + 1 : lo, farHi = delta <= 0 ? hi : mid;
        Search(nearLo, nearHi, depth + 1, q, ref bestSquared, ref bestIndex);
        if (delta * delta < bestSquared) Search(farLo, farHi, depth + 1, q, ref bestSquared, ref bestIndex);
    }
}

internal sealed class Icp
{
    internal int Iterations { get; private set; }
    internal double MeanSquaredError { get; private set; }
    internal double Theta { get; private set; }
    internal Vec2 Translation { get; private set; }
    internal bool Monotone { get; private set; } = true;

    /**
     * Runs ICP. The initial translation defaults to the centroid difference (the usual cold start); pass
     * `initialShift` and/or `initialTheta` to resume from a known pose. `trimFraction` of the worst matches are
     * dropped when estimating the transform (0 = plain ICP); `maxIterations` bounds the alternation.
     */
    internal void Align(IReadOnlyList<Vec2> source, IReadOnlyList<Vec2> target, int maxIterations = 60,
        double trimFraction = 0.0, double tolerance = 1e-13, double initialTheta = 0.0, Vec2? initialShift = null)
    {
        if (source.Count == 0 || target.Count == 0) throw new ArgumentException("clouds must not be empty");
        if (trimFraction < 0.0 || trimFraction >= 1.0) throw new ArgumentException("trimFraction out of range", nameof(trimFraction));
        var tree = new KdTree(target);
        double theta = initialTheta;
        Vec2 t = initialShift ?? (Mean(target) - Mean(source));
        var moved = new Vec2[source.Count];
        var matched = new Vec2[source.Count];
        var distances = new double[source.Count];
        double previous = double.MaxValue;
        Iterations = 0;
        MeanSquaredError = double.MaxValue;

        for (int it = 0; it < maxIterations; it++)
        {
            Iterations++;
            for (int i = 0; i < source.Count; i++)
            {
                moved[i] = Rotate(source[i], theta) + t;
                matched[i] = tree.NearestPoint(moved[i]);
                Vec2 diff = moved[i] - matched[i];
                distances[i] = diff.X * diff.X + diff.Y * diff.Y;
            }

            var keep = new List<int>();
            for (int i = 0; i < source.Count; i++) keep.Add(i);
            keep.Sort((a, b) => distances[a].CompareTo(distances[b]));
            int kept = Math.Max(Math.Min(2, source.Count), (int)Math.Round(source.Count * (1.0 - trimFraction)));

            double sx = 0, sy = 0, tx = 0, ty = 0;
            for (int k = 0; k < kept; k++)
            {
                Vec2 p = moved[keep[k]], q = matched[keep[k]];
                sx += p.X; sy += p.Y; tx += q.X; ty += q.Y;
            }
            Vec2 muP = new Vec2(sx / kept, sy / kept), muQ = new Vec2(tx / kept, ty / kept);
            double hxx = 0, hxy = 0, hyx = 0, hyy = 0;
            for (int k = 0; k < kept; k++)
            {
                Vec2 p = moved[keep[k]] - muP, q = matched[keep[k]] - muQ;
                hxx += p.X * q.X; hxy += p.X * q.Y; hyx += p.Y * q.X; hyy += p.Y * q.Y;
            }
            // H = sum p q^T with p on the moved (source) side, so the optimal angle is atan2(H_xy - H_yx, H_xx + H_yy).
            double delta = Math.Atan2(hxy - hyx, hxx + hyy);
            theta += delta;
            t = t + (muQ - Rotate(muP, delta));

            double error = 0.0;
            for (int i = 0; i < kept; i++)
            {
                double d = distances[keep[i]];
                error += d;
            }
            error /= source.Count;
            MeanSquaredError = error;
            if (error > previous + 1e-15 * Math.Max(1.0, Math.Abs(previous))) Monotone = false;
            if (Math.Abs(previous - error) <= tolerance * Math.Max(1.0, previous) || error == 0.0) break;
            previous = error;
        }
        Theta = theta;
        Translation = t;
    }

    internal static Vec2 Rotate(Vec2 p, double theta)
    {
        double c = Math.Cos(theta), s = Math.Sin(theta);
        return new Vec2(c * p.X - s * p.Y, s * p.X + c * p.Y);
    }

    internal static Vec2 Mean(IReadOnlyList<Vec2> points)
    {
        double sx = 0.0, sy = 0.0;
        foreach (Vec2 p in points) { sx += p.X; sy += p.Y; }
        return new Vec2(sx / points.Count, sy / points.Count);
    }
}

public static class Solution
{
    private const double TrueTheta = 0.2;
    private static readonly Vec2 TrueShift = new Vec2(2.0, -1.0);

    private static Vec2[] Blob(int count, ulong seed, double spread)
    {
        var rng = seed;
        var points = new Vec2[count];
        for (int i = 0; i < count; i++)
        {
            points[i] = new Vec2(spread * (SplitMix64.NextDouble(ref rng) - 0.5),
                spread * (SplitMix64.NextDouble(ref rng) - 0.5));
        }
        return points;
    }

    private static Vec2[] Transform(IReadOnlyList<Vec2> points, double theta, Vec2 t, ulong noiseSeed, double sigma)
    {
        var rng = noiseSeed;
        var result = new Vec2[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            Vec2 p = Icp.Rotate(points[i], theta) + t;
            if (sigma > 0.0)
            {
                p = new Vec2(p.X + sigma * SplitMix64.NextGaussian(ref rng), p.Y + sigma * SplitMix64.NextGaussian(ref rng));
            }
            result[i] = p;
        }
        return result;
    }

    /** Root-mean-square nearest-neighbour residual of a cloud under an externally supplied transform. */
    private static double Rms(IReadOnlyList<Vec2> source, IReadOnlyList<Vec2> target, double theta, Vec2 t)
    {
        var tree = new KdTree(target);
        double sum = 0.0;
        foreach (Vec2 p in source)
        {
            double d = tree.NearestDistance(Icp.Rotate(p, theta) + t);
            sum += d * d;
        }
        return Math.Sqrt(sum / source.Count);
    }

    /** Brute force: 20001 candidate angles, each with the optimal translation (centroid shift). */
    private static double BruteForceTheta(IReadOnlyList<Vec2> source, IReadOnlyList<Vec2> target, int samples = 20001)
    {
        var tree = new KdTree(target);
        double bestTheta = 0.0, bestCost = double.MaxValue;
        for (int s = 0; s < samples; s++)
        {
            double theta = 2.0 * Math.PI * s / samples;
            double shiftedXs = 0.0, shiftedYs = 0.0;
            foreach (Vec2 p in source)
            {
                Vec2 r = Icp.Rotate(p, theta);
                shiftedXs += r.X;
                shiftedYs += r.Y;
            }
            Vec2 muQ = Icp.Mean(target);
            var shift = new Vec2(muQ.X - shiftedXs / source.Count, muQ.Y - shiftedYs / source.Count);
            double cost = 0.0;
            foreach (Vec2 p in source)
            {
                double d = tree.NearestDistance(Icp.Rotate(p, theta) + shift);
                cost += d * d;
            }
            if (cost < bestCost) { bestCost = cost; bestTheta = theta; }
        }
        return bestTheta;
    }

    private static double AngleDifference(double a, double b)
    {
        double d = Math.Abs(a - b) % (2.0 * Math.PI);
        return Math.Min(d, 2.0 * Math.PI - d);
    }

    public static void Main()
    {
        Vec2[] shape = Blob(60, 0x1CE5UL, 4.0);
        Vec2[] targetExact = Transform(shape, TrueTheta, TrueShift, 0UL, 0.0);

        // 1. k-d tree oracle: exact nearest-neighbour distances versus the exhaustive scan.
        var tree = new KdTree(shape);
        var queryRng = 0x777UL;
        double worstTree = 0.0;
        for (int i = 0; i < 400; i++)
        {
            var q = new Vec2(4.0 * (SplitMix64.NextDouble(ref queryRng) - 0.5), 4.0 * (SplitMix64.NextDouble(ref queryRng) - 0.5));
            worstTree = Math.Max(worstTree, Math.Abs(tree.NearestDistance(q) - tree.BruteForceDistance(q)));
        }
        Console.WriteLine("  k-d tree vs brute force: max deviation " + worstTree.ToString("E3") + " over 400 queries");
        Check(worstTree < 1e-12, "k-d tree nearest-neighbour distances equal the brute-force scan");

        // 2-5. Noise-free registration from a start inside the basin (a fraction of the point spacing apart).
        var icp = new Icp();
        icp.Align(shape, targetExact, 60, 0.0, 1e-13, TrueTheta - 0.06, TrueShift + new Vec2(0.10, -0.08));
        double thetaError = AngleDifference(icp.Theta, TrueTheta);
        double shiftError = (icp.Translation - TrueShift).Norm;
        Console.WriteLine("  noise-free: theta error=" + thetaError.ToString("E3") + " shift error=" + shiftError.ToString("E3")
            + " rms=" + Math.Sqrt(icp.MeanSquaredError).ToString("E3") + " iterations=" + icp.Iterations);
        Check(thetaError < 1e-9, "rotation recovered exactly from a nearby start");
        Check(shiftError < 1e-9, "translation recovered exactly from a nearby start");
        Check(icp.MeanSquaredError < 1e-20, "correspondence error driven to zero");
        Check(icp.Monotone, "ICP invariant: mean squared error never increases");
        Check(Iterations(shape, targetExact, TrueTheta, TrueShift) == 1, "the exact pose is a fixed point of the iteration");

        // 6. Brute-force oracle for the closed-form rotation on the same pair.
        double bruteTheta = BruteForceTheta(shape, targetExact);
        Console.WriteLine("  brute-force sweep theta=" + bruteTheta.ToString("F6") + " vs closed form " + TrueTheta.ToString("F6")
            + " (grid step " + (2.0 * Math.PI / 20001).ToString("E2") + ")");
        Check(AngleDifference(bruteTheta, TrueTheta) < 3e-3, "brute-force angle sweep agrees with the closed-form rotation");

        // 7. Cold start (identity rotation plus the centroid translation) reaches the same exact solution; the
        //    printed start residual shows how far the iteration had to travel.
        var cold = new Icp();
        cold.Align(shape, targetExact);
        double coldStartRms = Rms(shape, targetExact, 0.0, TrueShift);
        Console.WriteLine("  cold start: rms " + coldStartRms.ToString("E3") + " -> " + Math.Sqrt(cold.MeanSquaredError).ToString("E3")
            + " theta error=" + AngleDifference(cold.Theta, TrueTheta).ToString("E3") + " iterations=" + cold.Iterations);
        Check(coldStartRms > 0.1, "the cold start really begins away from the solution");
        Check(Math.Sqrt(cold.MeanSquaredError) < 1e-12 && AngleDifference(cold.Theta, TrueTheta) < 1e-9,
            "cold start converges to the exact pose from the centroid initialisation");

        // 8-10. Noisy registration: transform within the noise level, residual on the order of sigma.
        const double sigma = 0.01;
        Vec2[] targetNoisy = Transform(shape, TrueTheta, TrueShift, 0xBEEFUL, sigma);
        var noisy = new Icp();
        noisy.Align(shape, targetNoisy, 60, 0.0, 1e-13, TrueTheta - 0.06, TrueShift + new Vec2(0.10, -0.08));
        double noisyRms = Math.Sqrt(noisy.MeanSquaredError);
        Console.WriteLine("  sigma=0.01: theta error=" + AngleDifference(noisy.Theta, TrueTheta).ToString("E3")
            + " shift error=" + (noisy.Translation - TrueShift).Norm.ToString("E3") + " rms=" + noisyRms.ToString("E4")
            + " (2D noise rms " + (sigma * Math.Sqrt(2.0)).ToString("E4") + ")");
        Check(AngleDifference(noisy.Theta, TrueTheta) < 0.01, "rotation within the noise level");
        Check((noisy.Translation - TrueShift).Norm < 0.02, "translation within the noise level");
        Check(noisyRms > 0.5 * sigma && noisyRms < 1.6 * sigma * Math.Sqrt(2.0), "residual is on the order of sigma");

        // 11-12. Partial overlap: 30% of the source has no counterpart, so plain ICP is dragged off while the
        //        trimmed variant (same start) still recovers the transform.
        var partialTarget = Transform(shape, TrueTheta, new Vec2(1.0, -0.5), 0x5A5AUL, 0.0);
        var partialSource = new List<Vec2>(shape);
        var outlierRng = 0xDEADUL;
        for (int i = 0; i < 25; i++)
        {
            partialSource.Add(new Vec2(8.0 + 4.0 * SplitMix64.NextDouble(ref outlierRng),
                8.0 + 4.0 * SplitMix64.NextDouble(ref outlierRng)));
        }
        var partialShift = new Vec2(1.0, -0.5);
        var plain = new Icp();
        plain.Align(partialSource, partialTarget, 60, 0.0, 1e-13, TrueTheta - 0.06, partialShift + new Vec2(0.10, -0.08));
        var trimmed = new Icp();
        trimmed.Align(partialSource, partialTarget, 60, 30.0 / partialSource.Count, 1e-13, TrueTheta - 0.06,
            partialShift + new Vec2(0.10, -0.08));
        double plainError = AngleDifference(plain.Theta, TrueTheta) + (plain.Translation - partialShift).Norm;
        double trimmedError = AngleDifference(trimmed.Theta, TrueTheta) + (trimmed.Translation - partialShift).Norm;
        Console.WriteLine("  70% overlap: plain error=" + plainError.ToString("E3") + " (theta "
            + AngleDifference(plain.Theta, TrueTheta).ToString("E3") + ") trimmed error=" + trimmedError.ToString("E3")
            + " (theta " + AngleDifference(trimmed.Theta, TrueTheta).ToString("E3") + ")");
        Check(trimmedError < 0.05, "trimmed ICP recovers the transform under 30% outliers");
        Check(trimmedError < plainError, "trimming beats the plain run ("
            + trimmedError.ToString("E3") + " < " + plainError.ToString("E3") + ")");

        // 13-15. Boundary cases: identical clouds, single point, minimal triangle from a perturbed start.
        var identity = new Icp();
        identity.Align(shape, shape);
        Check(identity.Iterations == 1 && identity.Theta == 0.0 && identity.Translation.Norm < 1e-15,
            "identical clouds give the identity on the first iteration");
        var single = new Icp();
        single.Align(new[] { new Vec2(1.0, 2.0) }, new[] { new Vec2(3.0, 5.0) });
        Check(single.Theta == 0.0 && (single.Translation - new Vec2(2.0, 3.0)).Norm < 1e-12,
            "single point yields the translation only");
        var triangle = new[] { new Vec2(0.0, 0.0), new Vec2(2.0, 0.0), new Vec2(0.0, 2.0) };
        var triangleTarget = Transform(triangle, 0.37, new Vec2(2.0, -1.0), 0UL, 0.0);
        var tri = new Icp();
        tri.Align(triangle, triangleTarget, 60, 0.0, 1e-13, 0.30, new Vec2(1.8, -0.9));
        Check(AngleDifference(tri.Theta, 0.37) < 1e-12 && (tri.Translation - new Vec2(2.0, -1.0)).Norm < 1e-12,
            "three points converge exactly from a perturbed initial guess");

        // 16-17. Degenerate collinear shape and rejected input.
        var line = new Vec2[10];
        for (int i = 0; i < line.Length; i++) line[i] = new Vec2(i, 2.5);
        var lineTarget = Transform(line, -1.1, new Vec2(-3.0, 4.0), 0UL, 0.0);
        var lineIcp = new Icp();
        lineIcp.Align(line, lineTarget, 60, 0.0, 1e-13, -1.0, new Vec2(-2.7, 3.6));
        Check(AngleDifference(lineIcp.Theta, -1.1) < 1e-12 && (lineIcp.Translation - new Vec2(-3.0, 4.0)).Norm < 1e-12,
            "collinear shape registers exactly from a nearby start");
        Check(Throws(() => new Icp().Align(Array.Empty<Vec2>(), shape)), "empty source cloud rejected");
        Check(Throws(() => new Icp().Align(shape, shape, 60, 1.5)), "trim fraction >= 1 rejected");

        // 18. Round trip: the inverse transform recovers the original cloud exactly.
        double roundTrip = 0.0;
        for (int i = 0; i < shape.Length; i++)
        {
            Vec2 recovered = Icp.Rotate(targetExact[i] - TrueShift, -TrueTheta);
            roundTrip = Math.Max(roundTrip, (recovered - shape[i]).Norm);
        }
        Check(roundTrip < 1e-12, "forward then inverse transform is the identity");

        // 19. Determinism.
        var again = new Icp();
        again.Align(shape, targetNoisy, 60, 0.0, 1e-13, TrueTheta - 0.06, TrueShift + new Vec2(0.10, -0.08));
        Check(again.Theta == noisy.Theta && again.MeanSquaredError == noisy.MeanSquaredError,
            "identical inputs reproduce the registration bit for bit");

        Console.WriteLine("PASS");
    }

    /** Number of iterations needed when ICP is restarted exactly at its own solution (fixed-point check). */
    private static int Iterations(Vec2[] source, Vec2[] target, double theta, Vec2 shift)
    {
        var probe = new Icp();
        probe.Align(source, target, 60, 0.0, 1e-13, theta, shift);
        return probe.Iterations;
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
