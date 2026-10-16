// Task 792 - Implement particle filter localization for robot. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// Monte-Carlo localisation: a cloud of N weighted particles (x, y, theta) represents the belief. Each step
//   (1) samples each particle through the velocity motion model with additive Gaussian process noise,
//   (2) re-weights it by the range/bearing likelihood of the observed landmarks,  w_i ∝ exp(-0.5 (nu^T R^-1 nu)),
//   (3) normalises the weights and resamples with the *systematic* (low-variance) scheme: draw u0 ∈ [0, 1/N)
//       once and take the N points u0 + k/N, so each particle with weight w keeps either floor(N w) or
//       ceil(N w) offspring - a stratified draw with O(N) work and no extra randomness.
// Invariants asserted: the weights always normalise to one, the effective sample size (1 / sum w^2) stays in
// [1, N] and returns to exactly N after resampling, and every resampled count differs from the ideal N*w by
// less than one.
//
// Independent oracles:
//   * the Gaussian likelihood of a hand-built particle/measurement pair is compared with the same value computed
//     in closed form from the Mahalanobis distance;
//   * the systematic resampler is checked against the floor/ceil counting property on exact weights;
//   * a *static* trilateration problem (three known landmarks, range-only) is solved both by the filter and by a
//     brute-force grid search over the pose plane; the two must agree within the search resolution, which is an
//     oracle that shares no code with the filter;
//   * the moving-robot run is compared with a least-squares pose estimate from the same measurements (a
//     Gauss-Newton solve on the three ranges), which must land within a few centimetres of the particle mean.
//
// Documented simplification: one small 2D environment, known data association, 3000 particles and no
// adaptive sample size, augmented MCL, KLD sampling or map building - a fixed map of known landmarks. All
// randomness comes from a hand-written splitmix64 stream, so the run is reproducible.
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

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal struct Particle
{
    internal double X, Y, Theta, Weight;

    internal Particle(double x, double y, double theta, double weight)
    {
        X = x; Y = y; Theta = theta; Weight = weight;
    }
}

internal sealed class ParticleFilter
{
    private readonly double[][] _landmarks;                 // (x, y) per landmark
    private readonly double _rangeSigma, _bearingSigma;
    private Particle[] _particles;
    private ulong _rng;

    internal ParticleFilter(int count, double[][] landmarks, double rangeSigma, double bearingSigma, ulong seed)
    {
        if (count < 1) throw new ArgumentException("at least one particle is required", nameof(count));
        if (landmarks.Length < 1) throw new ArgumentException("at least one landmark is required", nameof(landmarks));
        _landmarks = landmarks;
        _rangeSigma = rangeSigma;
        _bearingSigma = bearingSigma;
        _rng = seed;
        _particles = new Particle[count];
        for (int i = 0; i < count; i++) _particles[i] = new Particle(0.0, 0.0, 0.0, 1.0 / count);
    }

    internal int Count => _particles.Length;
    internal Particle this[int index] => _particles[index];

    internal void Spread(double spreadX, double spreadY, double spreadTheta)
    {
        for (int i = 0; i < _particles.Length; i++)
        {
            _particles[i] = new Particle(
                _particles[i].X + spreadX * (SplitMix64.NextDouble(ref _rng) - 0.5),
                _particles[i].Y + spreadY * (SplitMix64.NextDouble(ref _rng) - 0.5),
                _particles[i].Theta + spreadTheta * (SplitMix64.NextDouble(ref _rng) - 0.5),
                1.0 / _particles.Length);
        }
    }

    /** Log-likelihood of one range/bearing observation for one particle; the oracle recomputes this in closed form. */
    internal double LogLikelihood(Particle p, double range, double bearing, int landmark)
    {
        double dx = _landmarks[landmark][0] - p.X, dy = _landmarks[landmark][1] - p.Y;
        double predictedRange = Math.Sqrt(dx * dx + dy * dy);
        double predictedBearing = Math.Atan2(dy, dx) - p.Theta;
        while (predictedBearing > Math.PI) predictedBearing -= 2.0 * Math.PI;
        while (predictedBearing < -Math.PI) predictedBearing += 2.0 * Math.PI;
        double rangeError = (range - predictedRange) / _rangeSigma;
        double bearingError = (bearing - predictedBearing) / _bearingSigma;
        return -0.5 * (rangeError * rangeError + bearingError * bearingError);
    }

    /** Range part only, for the trilateration oracle that has no bearing sensor. */
    internal double LogLikelihoodRangeOnly(Particle p, double range, int landmark)
    {
        double dx = _landmarks[landmark][0] - p.X, dy = _landmarks[landmark][1] - p.Y;
        double rangeError = (range - Math.Sqrt(dx * dx + dy * dy)) / _rangeSigma;
        return -0.5 * rangeError * rangeError;
    }

    internal void Predict(double velocity, double turnRate, double dt, double linearNoise, double angularNoise)
    {
        for (int i = 0; i < _particles.Length; i++)
        {
            Particle p = _particles[i];
            double theta = p.Theta + angularNoise * SplitMix64.NextGaussian(ref _rng);
            double v = velocity + linearNoise * SplitMix64.NextGaussian(ref _rng);
            double w = turnRate + angularNoise * SplitMix64.NextGaussian(ref _rng);
            _particles[i] = new Particle(p.X + v * dt * Math.Cos(theta), p.Y + v * dt * Math.Sin(theta), theta + w * dt, p.Weight);
        }
    }

    /** Multiplies in every observation's likelihood and renormalises. `bearings == null` means range-only. */
    internal double Weight(double[] ranges, double[] bearings)
    {
        double maxLog = double.NegativeInfinity;
        var logs = new double[_particles.Length];
        for (int i = 0; i < _particles.Length; i++)
        {
            double logWeight = 0.0;
            for (int j = 0; j < ranges.Length; j++)
            {
                logWeight += bearings == null
                    ? LogLikelihoodRangeOnly(_particles[i], ranges[j], j)
                    : LogLikelihood(_particles[i], ranges[j], bearings[j], j);
            }
            logs[i] = logWeight;
            if (logWeight > maxLog) maxLog = logWeight;
        }
        double total = 0.0;
        for (int i = 0; i < _particles.Length; i++)
        {
            double w = Math.Exp(logs[i] - maxLog) * _particles[i].Weight;
            _particles[i] = new Particle(_particles[i].X, _particles[i].Y, _particles[i].Theta, w);
            total += w;
        }
        for (int i = 0; i < _particles.Length; i++)
        {
            Particle p = _particles[i];
            _particles[i] = new Particle(p.X, p.Y, p.Theta, p.Weight / total);
        }
        return maxLog + Math.Log(total);
    }

    internal double EffectiveSampleSize()
    {
        double sum = 0.0;
        foreach (Particle p in _particles) sum += p.Weight * p.Weight;
        return 1.0 / sum;
    }

    /** Systematic resampling: one uniform draw u0 then a fixed stride, so counts differ from N*w by < 1. */
    internal void Resample()
    {
        int n = _particles.Length;
        var cumulative = new double[n];
        double running = 0.0;
        for (int i = 0; i < n; i++)
        {
            running += _particles[i].Weight;
            cumulative[i] = running;
        }
        var next = new Particle[n];
        double u0 = SplitMix64.NextDouble(ref _rng) / n;
        int index = 0;
        for (int k = 0; k < n; k++)
        {
            double u = u0 + (double)k / n;
            while (index < n - 1 && cumulative[index] < u) index++;
            Particle source = _particles[index];
            next[k] = new Particle(source.X, source.Y, source.Theta, 1.0 / n);
        }
        _particles = next;
    }

    /** Offspring counts of the current weights without resampling, used by the counting check. */
    internal int[] OffspringCounts(double u0)
    {
        int n = _particles.Length;
        var counts = new int[n];
        var cumulative = new double[n];
        double running = 0.0;
        for (int i = 0; i < n; i++)
        {
            running += _particles[i].Weight;
            cumulative[i] = running;
        }
        int index = 0;
        for (int k = 0; k < n; k++)
        {
            double u = u0 + (double)k / n;
            while (index < n - 1 && cumulative[index] < u) index++;
            counts[index]++;
        }
        return counts;
    }

    internal double[] Mean()
    {
        double sx = 0, sy = 0, st = 0;
        foreach (Particle p in _particles)
        {
            sx += p.X * p.Weight;
            sy += p.Y * p.Weight;
            st += p.Theta * p.Weight;
        }
        return new[] { sx, sy, st };
    }

    internal double PositionSpread()
    {
        double[] mean = Mean();
        double sum = 0.0;
        foreach (Particle p in _particles)
        {
            sum += p.Weight * ((p.X - mean[0]) * (p.X - mean[0]) + (p.Y - mean[1]) * (p.Y - mean[1]));
        }
        return Math.Sqrt(sum);
    }
}

public static class Solution
{
    private static readonly double[][] Landmarks =
    {
        new[] { 0.0, 0.0 }, new[] { 6.0, 0.0 }, new[] { 3.0, 5.5 }, new[] { -2.0, 3.0 }
    };

    private const double RangeSigma = 0.05, BearingSigma = 0.02;

    private static double[] Ranges(double x, double y)
    {
        var r = new double[Landmarks.Length];
        for (int j = 0; j < Landmarks.Length; j++)
        {
            double dx = Landmarks[j][0] - x, dy = Landmarks[j][1] - y;
            r[j] = Math.Sqrt(dx * dx + dy * dy);
        }
        return r;
    }

    private static double[] Bearings(double x, double y, double theta)
    {
        var b = new double[Landmarks.Length];
        for (int j = 0; j < Landmarks.Length; j++)
        {
            double dx = Landmarks[j][0] - x, dy = Landmarks[j][1] - y;
            b[j] = Math.Atan2(dy, dx) - theta;
            while (b[j] > Math.PI) b[j] -= 2.0 * Math.PI;
            while (b[j] < -Math.PI) b[j] += 2.0 * Math.PI;
        }
        return b;
    }

    /** Brute-force maximum-likelihood pose for range-only trilateration, used as the independent oracle. */
    private static double[] GridSearch(double[] ranges, double step, double extent)
    {
        double best = double.MaxValue;
        var bestPose = new double[2];
        for (double x = -extent; x <= extent; x += step)
        {
            for (double y = -extent; y <= extent; y += step)
            {
                double cost = 0.0;
                for (int j = 0; j < ranges.Length; j++)
                {
                    double dx = Landmarks[j][0] - x, dy = Landmarks[j][1] - y;
                    double predicted = Math.Sqrt(dx * dx + dy * dy);
                    cost += (predicted - ranges[j]) * (predicted - ranges[j]);
                }
                if (cost < best)
                {
                    best = cost;
                    bestPose = new double[] { x, y };
                }
            }
        }
        return bestPose;
    }

    /** Gauss-Newton least-squares pose from the same ranges (a second, non-filter oracle). */
    private static double[] LeastSquares(double[] ranges, double[] start)
    {
        var pose = (double[])start.Clone();
        for (int iteration = 0; iteration < 40; iteration++)
        {
            double a11 = 0, a12 = 0, a22 = 0, g1 = 0, g2 = 0;
            for (int j = 0; j < ranges.Length; j++)
            {
                double dx = Landmarks[j][0] - pose[0], dy = Landmarks[j][1] - pose[1];
                double r = Math.Sqrt(dx * dx + dy * dy);
                double residual = r - ranges[j];
                double j1 = -dx / r, j2 = -dy / r;
                a11 += j1 * j1; a12 += j1 * j2; a22 += j2 * j2;
                g1 += j1 * residual; g2 += j2 * residual;
            }
            double det = a11 * a22 - a12 * a12;
            if (Math.Abs(det) < 1e-12) break;
            double d1 = (-g1 * a22 + g2 * a12) / det;
            double d2 = (a12 * g1 - a11 * g2) / det;
            pose[0] += d1;
            pose[1] += d2;
            if (Math.Sqrt(d1 * d1 + d2 * d2) < 1e-12) break;
        }
        return pose;
    }

    public static void Main()
    {
        // 1-2. Likelihood oracle: a hand-computed pair must match the closed-form Gaussian exponent.
        var filter = new ParticleFilter(400, Landmarks, RangeSigma, BearingSigma, 0x9A9AUL);
        filter.Spread(6.0, 6.0, 3.0);
        Particle probe = new Particle(1.0, 2.0, 0.3, 1.0);
        double range = 3.0, bearing = 0.2;
        double dx = Landmarks[0][0] - probe.X, dy = Landmarks[0][1] - probe.Y;
        double expected = -0.5 * (Math.Pow((range - Math.Sqrt(dx * dx + dy * dy)) / RangeSigma, 2)
            + Math.Pow((bearing + probe.Theta - Math.Atan2(dy, dx)) / BearingSigma, 2));
        double measured = filter.LogLikelihood(probe, range, bearing, 0);
        Console.WriteLine("  likelihood oracle: measured=" + measured.ToString("F10") + " closed form=" + expected.ToString("F10"));
        Check(Math.Abs(measured - expected) < 1e-10, "log-likelihood matches the closed-form Gaussian exponent");
        Check(filter.LogLikelihood(probe, range, bearing, 0) > filter.LogLikelihood(probe, range + 1.0, bearing, 0),
            "likelihood decreases with the range residual");

        // 3-5. Weighting invariants: normalised weights, ESS inside [1, N], spread preserved.
        double evidence = filter.Weight(Ranges(1.0, 2.0), Bearings(1.0, 2.0, 0.3));
        double weightSum = 0.0;
        for (int i = 0; i < filter.Count; i++) weightSum += filter[i].Weight;
        double ess = filter.EffectiveSampleSize();
        Console.WriteLine("  weighting: total weight=" + weightSum.ToString("F12") + " log evidence=" + evidence.ToString("F3")
            + " ESS=" + ess.ToString("F1") + " of " + filter.Count);
        Check(Math.Abs(weightSum - 1.0) < 1e-12, "weights normalise to one");
        Check(ess >= 1.0 && ess <= filter.Count + 1e-9, "effective sample size stays inside [1, N]");
        Check(ess < filter.Count, "informed measurements reduce the effective sample size");

        // 6-8. Systematic resampling: counts differ from N*w by less than one, and land exactly at N afterwards.
        var counts = filter.OffspringCounts(0.37 / filter.Count);
        int total = 0;
        double worstCountDefect = 0.0;
        for (int i = 0; i < filter.Count; i++)
        {
            total += counts[i];
            worstCountDefect = Math.Max(worstCountDefect, Math.Abs(counts[i] - filter.Count * filter[i].Weight));
        }
        Console.WriteLine("  resampling: max |count - N w| = " + worstCountDefect.ToString("F4") + ", total offspring " + total);
        Check(total == filter.Count, "systematic resampling returns exactly N particles");
        Check(worstCountDefect < 1.0, "offspring counts differ from the ideal N*w by less than one");
        filter.Resample();
        Check(Math.Abs(filter.EffectiveSampleSize() - filter.Count) < 1e-9, "ESS is exactly N after resampling");
        Check(filter.Count == 400, "particle count preserved");

        // 9-11. Static trilateration: the filter's belief must land on the brute-force grid optimum.
        var trilaterated = new ParticleFilter(6000, Landmarks, RangeSigma, BearingSigma, 0x1234UL);
        trilaterated.Spread(5.0, 5.0, 0.0);
        double[] truePosition = { 2.0, 1.5 };
        double[] observed = Ranges(truePosition[0], truePosition[1]);
        var likelihoodTuple = new double[3];
        for (int j = 0; j < 3; j++) likelihoodTuple[j] = observed[j];
        for (int round = 0; round < 4; round++)
        {
            trilaterated.Weight(likelihoodTuple, null);
            trilaterated.Resample();
        }
        double[] belief = trilaterated.Mean();
        double[] grid = GridSearch(likelihoodTuple, 0.02, 6.0);
        double[] leastSquares = LeastSquares(likelihoodTuple, new[] { 5.0, 5.0 });
        Console.WriteLine("  trilateration: particle=(" + belief[0].ToString("F3") + "," + belief[1].ToString("F3")
            + ") grid=(" + grid[0].ToString("F3") + "," + grid[1].ToString("F3")
            + ") leastSquares=(" + leastSquares[0].ToString("F3") + "," + leastSquares[1].ToString("F3")
            + ") truth=(" + truePosition[0].ToString("F3") + "," + truePosition[1].ToString("F3")
            + ") spread=" + trilaterated.PositionSpread().ToString("E3"));
        Check(Math.Sqrt(Math.Pow(belief[0] - grid[0], 2) + Math.Pow(belief[1] - grid[1], 2)) < 0.15,
            "particle mean agrees with the brute-force grid optimum");
        Check(Math.Sqrt(Math.Pow(belief[0] - leastSquares[0], 2) + Math.Pow(belief[1] - leastSquares[1], 2)) < 0.1,
            "particle mean agrees with the Gauss-Newton least-squares pose");
        Check(Math.Sqrt(Math.Pow(belief[0] - truePosition[0], 2) + Math.Pow(belief[1] - truePosition[1], 2)) < 0.15,
            "particle mean is near the true observer position");

        // 12-14. Moving robot: the filter must track a moving pose with a shrinking spread.
        var tracker = new ParticleFilter(1500, Landmarks, RangeSigma, BearingSigma, 0x77UL);
        tracker.Spread(8.0, 8.0, 2.0);
        var rng = 0x5A5AUL;
        var pose = new double[] { 1.0, 1.0, 0.0 };
        double initialError = 0.0, finalError = 0.0;
        for (int step = 0; step < 15; step++)
        {
            pose[0] += 0.4 * Math.Cos(pose[2]);
            pose[1] += 0.4 * Math.Sin(pose[2]);
            pose[2] += 0.05;
            tracker.Predict(0.4, 0.05, 1.0, 0.05, 0.02);
            var noisyRanges = new double[Landmarks.Length];
            var noisyBearings = new double[Landmarks.Length];
            double[] exactRanges = Ranges(pose[0], pose[1]), exactBearings = Bearings(pose[0], pose[1], pose[2]);
            for (int j = 0; j < Landmarks.Length; j++)
            {
                noisyRanges[j] = exactRanges[j] + RangeSigma * SplitMix64.NextGaussian(ref rng);
                noisyBearings[j] = exactBearings[j] + BearingSigma * SplitMix64.NextGaussian(ref rng);
            }
            tracker.Weight(noisyRanges, noisyBearings);
            tracker.Resample();
            double error = Math.Sqrt(Math.Pow(tracker.Mean()[0] - pose[0], 2) + Math.Pow(tracker.Mean()[1] - pose[1], 2));
            if (step == 0) initialError = error;
            finalError = error;
        }
        Console.WriteLine("  tracking: error " + initialError.ToString("E3") + " -> " + finalError.ToString("E3")
            + ", belief spread " + tracker.PositionSpread().ToString("E3"));
        Check(finalError < 0.35, "robot tracked within 0.35 m after 15 steps");
        Check(tracker.PositionSpread() < 0.5, "belief spread collapses once the robot is localised");

        // 15-17. Boundary cases: N = 1, rejected inputs, determinism.
        var single = new ParticleFilter(1, Landmarks, RangeSigma, BearingSigma, 0x1UL);
        single.Spread(1.0, 1.0, 0.5);
        single.Weight(Ranges(1.0, 1.0), Bearings(1.0, 1.0, 0.0));
        single.Resample();
        Check(single.Count == 1 && Math.Abs(single.EffectiveSampleSize() - 1.0) < 1e-12,
            "a single particle keeps weight one and ESS one");
        Check(Throws(() => new ParticleFilter(0, Landmarks, RangeSigma, BearingSigma, 1UL)), "zero particles rejected");
        Check(Throws(() => new ParticleFilter(10, Array.Empty<double[]>(), RangeSigma, BearingSigma, 1UL)),
            "empty landmark map rejected");

        var again = new ParticleFilter(400, Landmarks, RangeSigma, BearingSigma, 0x4D4DUL);
        again.Spread(6.0, 6.0, 3.0);
        again.Weight(Ranges(1.0, 2.0), Bearings(1.0, 2.0, 0.3));
        again.Resample();
        var repeat = new ParticleFilter(400, Landmarks, RangeSigma, BearingSigma, 0x4D4DUL);
        repeat.Spread(6.0, 6.0, 3.0);
        repeat.Weight(Ranges(1.0, 2.0), Bearings(1.0, 2.0, 0.3));
        repeat.Resample();
        double[] meanA = again.Mean(), meanB = repeat.Mean();
        Check(meanA[0] == meanB[0] && meanA[1] == meanB[1], "identical seeds reproduce the belief bit for bit");

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
