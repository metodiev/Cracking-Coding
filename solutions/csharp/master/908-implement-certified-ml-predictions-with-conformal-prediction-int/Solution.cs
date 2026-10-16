// Task 908 - Implement certified ML predictions with conformal prediction intervals.
// (Tier: Master, Beyond - Machine Learning & Simulation)
//
// Split conformal prediction wraps any point predictor with a distribution-free interval.  A linear
// least-squares model is fitted on a training split, absolute residuals |y - yhat| are collected on a
// *disjoint* calibration split, and the interval half-width is the order statistic
//
//     q = the ceil((n + 1)(1 - alpha))-th smallest calibration score,
//
// which by exchangeability gives the finite-sample guarantee P(y_new in [yhat - q, yhat + q]) >= 1 - alpha
// with no assumption on the noise.  Two consequences the checks exercise directly:
//
//  * when ceil((n + 1)(1 - alpha)) exceeds the calibration size the rule degenerates to q = +infinity - the
//    honest answer for an under-sampled tail, not a silent under-coverage;
//  * even a deliberately bad predictor keeps the guarantee - the intervals simply grow.
//
// Simplifications: one-dimensional inputs, deterministic pseudo-random data, split (not full) conformal, and
// absolute-residual scores.  Verified: the order statistic against a brute-force "smallest value covering
// ceil((n+1)(1-alpha)) scores" search, empirical test coverage of the fitted and of a constant predictor,
// monotone widths in alpha, permutation invariance, the single-point and infinite-width boundaries, and
// reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUniform() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    /** Twelve uniforms minus six: a cheap deterministic approximation of a standard normal. */
    private static double NextNormal()
    {
        double sum = 0.0;
        for (int i = 0; i < 12; i++) sum += NextUniform();
        return sum - 6.0;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    /** Ground truth plus heteroscedastic noise: the spread grows with x, so one width is never right. */
    private static double Truth(double x) => 1.5 + 0.8 * x + 0.4 * Math.Sin(2.0 * x);

    private static void Generate(int count, int seed, out double[] xs, out double[] ys)
    {
        _state = (ulong)seed;
        xs = new double[count];
        ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            double x = 4.0 * NextUniform();
            double sigma = 0.3 + 0.2 * x;
            xs[i] = x;
            ys[i] = Truth(x) + sigma * NextNormal();
        }
    }

    /** Least-squares fit of y = a + b x on the training split. */
    private static (double A, double B) Fit(double[] xs, double[] ys)
    {
        int n = xs.Length;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++)
        {
            sx += xs[i];
            sy += ys[i];
            sxx += xs[i] * xs[i];
            sxy += xs[i] * ys[i];
        }
        double denominator = n * sxx - sx * sx;
        double b = denominator == 0.0 ? 0.0 : (n * sxy - sx * sy) / denominator;
        return ((sy - b * sx) / n, b);
    }

    private static List<double> CalibrationScores(double[] xs, double[] ys, Func<double, double> predict)
    {
        List<double> scores = new List<double>();
        for (int i = 0; i < xs.Length; i++) scores.Add(Math.Abs(ys[i] - predict(xs[i])));
        return scores;
    }

    /** The conformal order statistic: the smallest value whose rank is ceil((n+1)(1-alpha)). */
    private static double Quantile(List<double> scores, double alpha)
    {
        int n = scores.Count;
        int rank = (int)Math.Ceiling((n + 1) * (1.0 - alpha));
        if (rank > n) return double.PositiveInfinity;
        List<double> sorted = new List<double>(scores);
        sorted.Sort();
        return sorted[rank - 1];
    }

    /** Independent oracle: brute-force the smallest score that covers ceil((n+1)(1-alpha)) calibration points. */
    private static double OracleQuantile(List<double> scores, double alpha)
    {
        int need = (int)Math.Ceiling((scores.Count + 1) * (1.0 - alpha));
        List<double> sorted = new List<double>(scores);
        sorted.Sort();
        for (int i = 0; i < sorted.Count; i++)
        {
            int covered = 0;
            foreach (double score in scores)
            {
                if (score <= sorted[i]) covered++;
            }
            if (covered >= need) return sorted[i];
        }
        return double.PositiveInfinity;
    }

    private static double Coverage(double[] xs, double[] ys, Func<double, double> predict, double q)
    {
        int covered = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            if (Math.Abs(ys[i] - predict(xs[i])) <= q) covered++;
        }
        return covered / (double)xs.Length;
    }

    public static void Main()
    {
        Generate(120, 11, out double[] trainX, out double[] trainY);
        Generate(200, 22, out double[] calX, out double[] calY);
        Generate(600, 33, out double[] testX, out double[] testY);
        Generate(600, 77, out double[] test2X, out double[] test2Y);
        (double a, double b) = Fit(trainX, trainY);
        Func<double, double> model = x => a + b * x;
        List<double> scores = CalibrationScores(calX, calY, model);
        const double alpha = 0.1;
        double q = Quantile(scores, alpha);

        // ---- 1. the order statistic against the brute-force oracle ----------------------------------
        double oracle = OracleQuantile(scores, alpha);
        Check(q == oracle, "the order statistic equals the brute-force covering-value search (" +
            q.ToString("F6", CultureInfo.InvariantCulture) + ")");
        Check(scores.Contains(q), "the half-width is always one of the calibration residuals");
        int rank = (int)Math.Ceiling((scores.Count + 1) * (1.0 - alpha));
        Check(rank == 181, "a calibration set of 200 at alpha=0.1 uses rank " + rank);
        Check(q > 0.0 && !double.IsInfinity(q), "the half-width is finite and positive for a 10% target");

        // ---- 2. the coverage guarantee on unseen data -------------------------------------------------
        double coverage = Coverage(testX, testY, model, q);
        Check(Math.Abs(coverage - (1.0 - alpha)) <= 0.04,
            "test coverage " + coverage.ToString("F4", CultureInfo.InvariantCulture) + " sits within 4 points of 90%");
        Check(coverage >= 0.85, "coverage never falls materially below the target");
        double coverage2 = Coverage(test2X, test2Y, model, q);
        Check(Math.Abs(coverage2 - (1.0 - alpha)) <= 0.04,
            "a second independent test split is covered the same way (" + coverage2.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 3. monotone widths as alpha shrinks -------------------------------------------------------
        double[] alphas = { 0.2, 0.1, 0.05, 0.02 };
        double previous = 0.0;
        int rungs = 0;
        foreach (double level in alphas)
        {
            double width = Quantile(scores, level);
            Check(width >= previous, "alpha=" + level + " gives at least the width of the previous level (" +
                width.ToString("F4", CultureInfo.InvariantCulture) + ")");
            previous = width;
            rungs++;
        }
        Check(rungs == 4 && previous > q, "a stricter target widens the interval");

        // ---- 4. a deliberately bad predictor keeps the guarantee ---------------------------------------
        Func<double, double> constant = _ => 0.0;
        List<double> biased = CalibrationScores(calX, calY, constant);
        double biasedQ = Quantile(biased, alpha);
        double biasedCoverage = Coverage(testX, testY, constant, biasedQ);
        Check(Math.Abs(biasedCoverage - (1.0 - alpha)) <= 0.06,
            "a constant predictor still hits the target coverage (" + biasedCoverage.ToString("F4", CultureInfo.InvariantCulture) + ")");
        Check(biasedQ > q, "and pays for it with a wider interval (" + biasedQ.ToString("F4", CultureInfo.InvariantCulture) +
            " > " + q.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 5. exchangeability: permuting the calibration set changes nothing -------------------------
        List<double> shuffled = new List<double>(scores);
        _state = 4242UL;
        for (int i = shuffled.Count - 1; i > 0; i--)
        {
            int j = (int)(NextUInt64() % (ulong)(i + 1));
            double swap = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = swap;
        }
        Check(!SameOrder(scores, shuffled), "the permutation really reordered the residuals");
        Check(Quantile(shuffled, alpha) == q, "the order statistic is exchangeable, so the permutation changes nothing");

        // ---- 6. boundaries -----------------------------------------------------------------------------
        double infinite = Quantile(scores, 0.002);
        Check(double.IsInfinity(infinite), "asking for rank 201 of 200 calibration points degenerates to +infinity");
        Check(Coverage(testX, testY, model, infinite) == 1.0, "an infinite interval covers every test point");
        List<double> single = new List<double> { 0.75 };
        Check(Quantile(single, 0.5) == 0.75 && Quantile(single, 0.6) == 0.75,
            "one calibration point at alpha>=0.5 gives exactly its own residual");
        Check(double.IsInfinity(Quantile(single, 0.1)), "one calibration point cannot certify a 90% interval");
        Check(double.IsInfinity(Quantile(new List<double>(), 0.1)), "an empty calibration set yields an infinite width");

        // ---- 7. reproducibility ------------------------------------------------------------------------
        Generate(200, 22, out double[] calX2, out double[] calY2);
        Check(Quantile(CalibrationScores(calX2, calY2, model), alpha) == q, "the same data reproduces the same half-width");

        Console.WriteLine("model: y = " + a.ToString("F6", CultureInfo.InvariantCulture) + " + " +
            b.ToString("F6", CultureInfo.InvariantCulture) + " x, calibration n=200, alpha=0.1, width " +
            q.ToString("F6", CultureInfo.InvariantCulture));
        Console.WriteLine("coverage: test " + coverage.ToString("F4", CultureInfo.InvariantCulture) + ", second split " +
            coverage2.ToString("F4", CultureInfo.InvariantCulture) + ", constant predictor " +
            biasedCoverage.ToString("F4", CultureInfo.InvariantCulture) + " at width " + biasedQ.ToString("F4", CultureInfo.InvariantCulture));
        Console.WriteLine("certificate: rank " + rank + " of 200, brute-force oracle agreement, infinite-width boundary below alpha=0.005");
        Console.WriteLine("PASS");
    }

    private static bool SameOrder(List<double> left, List<double> right)
    {
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }
}
