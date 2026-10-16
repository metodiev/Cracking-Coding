// Task 709 - Implement differential privacy noise addition and utility tradeoff demo.
// (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// Three standard differentially private mechanisms, all fed by a hand-written splitmix64 stream so
// the whole demo is deterministic:
//   * Laplace mechanism: releasing f(D) + Lap(sensitivity / epsilon) is epsilon-DP because the
//     Laplace density ratio between neighbouring databases is at most exp(epsilon). Samples come
//     from the inverse CDF, sign * b * ln(1 - 2|u|), which is exact in double precision.
//   * Gaussian mechanism: noise scale sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon gives
//     (epsilon, delta)-DP; the demo checks the formula, the ordering in epsilon and in delta, and
//     the empirical spread.
//   * Exponential mechanism: sampling an item with probability proportional to
//     exp(epsilon * u(item) / (2 * sensitivity)) keeps the ratio between any two output
//     probabilities below exp(epsilon * (u(a) - u(b)) / (2 * sensitivity)).
//
// The statistical checks are honest statistics rather than tautologies: the sampled Laplace noise
// must have the theoretical mean, median, variance (2b^2), mean absolute error (b) and sign balance,
// the utility tradeoff must be monotone in epsilon, and - the actual privacy statement - the ratio
// of two *empirical* output-distribution probabilities of two neighbouring databases, measured over
// 40 000 samples per database, must stay inside the exp(+-epsilon) envelope the definition demands,
// both for a single query and for a two-query composition split of the budget.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>Uniform double in (0,1): the 52-bit draw is nudged away from the open ends.</summary>
    internal static double OpenUnit(ref ulong state)
    {
        double u = (Next(ref state) >> 12) * (1.0 / 4503599627370496.0);   // [0,1)
        return u <= 0.0 ? 0.5 : u;                                        // the 0.5 fallback has probability 2^-52
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal static class Mechanisms
{
    /// <summary>Laplace sample with the given scale, from the exact inverse CDF.</summary>
    internal static double Laplace(ref ulong state, double scale)
    {
        if (scale < 0) throw new ArgumentOutOfRangeException(nameof(scale));
        if (scale == 0) return 0;
        double u = Rng.OpenUnit(ref state) - 0.5;               // in (-0.5, 0.5)
        double magnitude = -Math.Log(1 - 2 * Math.Abs(u));
        return u < 0 ? -scale * magnitude : scale * magnitude;
    }

    internal static double LaplaceMechanism(double value, double sensitivity, double epsilon, ref ulong state)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive");
        if (sensitivity < 0) throw new ArgumentOutOfRangeException(nameof(sensitivity));
        return value + Laplace(ref state, sensitivity / epsilon);
    }

    /// <summary>Standard deviation of the Gaussian mechanism for (epsilon, delta)-DP.</summary>
    internal static double GaussianSigma(double sensitivity, double epsilon, double delta)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive");
        if (delta <= 0 || delta >= 1) throw new ArgumentOutOfRangeException(nameof(delta), "delta must be in (0,1)");
        if (sensitivity < 0) throw new ArgumentOutOfRangeException(nameof(sensitivity));
        return Math.Sqrt(2 * Math.Log(1.25 / delta)) * sensitivity / epsilon;
    }

    /// <summary>Box-Muller draw from the deterministic stream (used only for the Gaussian demo).</summary>
    internal static double Gaussian(ref ulong state, double sigma)
    {
        double u1 = Rng.OpenUnit(ref state), u2 = Rng.OpenUnit(ref state);
        return sigma * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    internal static double GaussianMechanism(double value, double sensitivity, double epsilon, double delta, ref ulong state)
        => value + Gaussian(ref state, GaussianSigma(sensitivity, epsilon, delta));

    /// <summary>Exponential mechanism: pick index i with probability proportional to exp(eps u_i / (2 sens)).</summary>
    internal static int ExponentialMechanism(double[] utilities, double sensitivity, double epsilon, ref ulong state)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive");
        if (sensitivity <= 0) throw new ArgumentOutOfRangeException(nameof(sensitivity));
        if (utilities.Length == 0) throw new ArgumentException("no candidates to choose from", nameof(utilities));
        var weights = new double[utilities.Length];
        double peak = double.NegativeInfinity;
        for (int i = 0; i < utilities.Length; i++)
        {
            if (double.IsNaN(utilities[i])) throw new ArgumentException("utility is not a number", nameof(utilities));
            weights[i] = epsilon * utilities[i] / (2 * sensitivity);
            if (weights[i] > peak) peak = weights[i];
        }
        double sum = 0;
        for (int i = 0; i < weights.Length; i++) { weights[i] = Math.Exp(weights[i] - peak); sum += weights[i]; }
        double pick = Rng.OpenUnit(ref state) * sum;
        double running = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            running += weights[i];
            if (pick <= running) return i;
        }
        return weights.Length - 1;
    }

    /// <summary>k count queries with budget eps each: the composed release is (k*eps)-DP.</summary>
    internal static long[] PrivateHistogram(long[] counts, double epsilon, ref ulong state)
    {
        var noisy = new long[counts.Length];
        for (int i = 0; i < counts.Length; i++)
        {
            // a single record changes one bucket by one, so the sensitivity of each query is 1
            noisy[i] = (long)Math.Round(LaplaceMechanism(counts[i], 1.0, epsilon, ref state));
        }
        return noisy;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double Mean(double[] values)
    {
        double sum = 0;
        foreach (double v in values) sum += v;
        return sum / values.Length;
    }

    private static double MeanAbs(double[] values)
    {
        double sum = 0;
        foreach (double v in values) sum += Math.Abs(v);
        return sum / values.Length;
    }

    private static double Variance(double[] values)
    {
        double mean = Mean(values), sum = 0;
        foreach (double v in values) sum += (v - mean) * (v - mean);
        return sum / values.Length;
    }

    private static double Median(double[] values)
    {
        var copy = (double[])values.Clone();
        Array.Sort(copy);
        return copy[copy.Length / 2];
    }

    /// <summary>Empirical mean absolute error of the Laplace mechanism on a unit-sensitivity count.</summary>
    private static double MeasuredError(double epsilon, int samples, ulong seed)
    {
        ulong state = seed;
        double sum = 0;
        for (int i = 0; i < samples; i++) sum += Math.Abs(Mechanisms.LaplaceMechanism(1000.0, 1.0, epsilon, ref state) - 1000.0);
        return sum / samples;
    }

    public static void Main()
    {
        // ---- the Laplace sampler has to be a real Laplace distribution -------------------------
        {
            const int Samples = 200000;
            const double Scale = 2.0;
            ulong state = 0xD15EA5EUL;
            var draws = new double[Samples];
            for (int i = 0; i < Samples; i++) draws[i] = Mechanisms.Laplace(ref state, Scale);

            double mean = Mean(draws), variance = Variance(draws), median = Median(draws);
            double standardError = Math.Sqrt(2 * Scale * Scale / Samples);
            Check(Math.Abs(mean) < 5 * standardError,
                "the Laplace mean is 0 (measured " + mean.ToString("F4") + ", standard error " + standardError.ToString("F4") + ")");
            Check(Math.Abs(median) < 0.1 * Scale, "the Laplace median is 0 (measured " + median.ToString("F4") + ")");
            Check(Math.Abs(variance - 2 * Scale * Scale) < 0.1 * 2 * Scale * Scale,
                "the Laplace variance is 2b^2 = " + (2 * Scale * Scale) + " (measured " + variance.ToString("F4") + ")");
            int negatives = 0;
            foreach (double d in draws) if (d < 0) negatives++;
            double share = (double)negatives / Samples;
            Check(Math.Abs(share - 0.5) < 0.01, "the noise is symmetric: " + share.ToString("F4") + " negative");
            Check(Mechanisms.Laplace(ref state, 0) == 0, "zero scale produces exactly zero noise");
            Check(Mechanisms.LaplaceMechanism(42.0, 0, 1.0, ref state) == 42.0,
                "zero sensitivity means the answer is exact");
            Console.WriteLine("laplaceSamples=" + Samples + " scale=" + Scale + " mean=" + mean.ToString("F4")
                + " variance=" + variance.ToString("F4") + " median=" + median.ToString("F4"));
        }

        // ---- the utility tradeoff: mean absolute error is sensitivity / epsilon ----------------
        {
            const int Samples = 200000;
            var epsilons = new[] { 0.25, 1.0, 4.0, 16.0 };
            var errors = new double[epsilons.Length];
            for (int i = 0; i < epsilons.Length; i++) errors[i] = MeasuredError(epsilons[i], Samples, 0x5EED0000UL + (ulong)i);
            for (int i = 0; i < epsilons.Length; i++)
            {
                double theoretical = 1.0 / epsilons[i];
                Check(Math.Abs(errors[i] - theoretical) < 0.08 * theoretical,
                    "epsilon=" + epsilons[i] + " gives mean absolute error ~1/epsilon (measured "
                    + errors[i].ToString("F4") + " vs " + theoretical.ToString("F4") + ")");
            }
            for (int i = 1; i < epsilons.Length; i++)
                Check(errors[i] < errors[i - 1], "more privacy budget means less error at epsilon=" + epsilons[i]);
            Check(errors[3] < errors[0] / 10, "the 64-fold budget increase improves the error by more than 10x");

            // 95% of the Laplace error is inside 3b, so this is a real distributional check
            ulong state = 0xC0FFEEUL;
            int inside = 0;
            const int Trials = 40000;
            for (int i = 0; i < Trials; i++)
            {
                double noisy = Mechanisms.LaplaceMechanism(500.0, 1.0, 1.0, ref state);
                if (Math.Abs(noisy - 500.0) <= 3.0) inside++;
            }
            double share = (double)inside / Trials;
            Check(Math.Abs(share - (1 - Math.Exp(-3))) < 0.01,
                "P(|error| <= 3b) is 1 - e^-3 (measured " + share.ToString("F4") + ")");
            Console.WriteLine("mae(0.25,1,4,16)=" + errors[0].ToString("F4") + "/" + errors[1].ToString("F4")
                + "/" + errors[2].ToString("F4") + "/" + errors[3].ToString("F4"));
        }

        // ---- the privacy statement: empirical output ratios stay inside exp(epsilon) ----------
        {
            // D and D' differ in one record: the count query returns 1000 and 1001.
            const int Samples = 40000;
            double epsilon = 1.0;
            ulong stateA = 0xAAAAUL, stateB = 0xBBBBUL;
            const int Low = 997, High = 999;
            int hitsA = 0, hitsB = 0;
            for (int i = 0; i < Samples; i++)
            {
                double a = Mechanisms.LaplaceMechanism(1000.0, 1.0, epsilon, ref stateA);
                double b = Mechanisms.LaplaceMechanism(1001.0, 1.0, epsilon, ref stateB);
                if (a >= Low && a <= High) hitsA++;
                if (b >= Low && b <= High) hitsB++;
            }
            double pA = (double)hitsA / Samples, pB = (double)hitsB / Samples;
            Check(pA > 0 && pB > 0, "the chosen output interval has mass under both databases");
            double ratio = pA / pB;
            Check(ratio <= Math.Exp(epsilon) * 1.05 && ratio >= Math.Exp(-epsilon) * 0.95,
                "the empirical probability ratio " + ratio.ToString("F4") + " respects exp(+-" + epsilon + ")");

            // Composition: two queries that each spend epsilon/2 satisfy the same envelope for the
            // concatenated release, and the tighter per-query budget shrinks the envelope.
            double half = epsilon / 2;
            ulong stateC = 0xCCCCUL, stateD = 0xDDDDUL;
            int hitsC = 0, hitsD = 0;
            for (int i = 0; i < Samples; i++)
            {
                double a1 = Mechanisms.LaplaceMechanism(1000.0, 1.0, half, ref stateC);
                double a2 = Mechanisms.LaplaceMechanism(2000.0, 1.0, half, ref stateC);
                double b1 = Mechanisms.LaplaceMechanism(1001.0, 1.0, half, ref stateD);
                double b2 = Mechanisms.LaplaceMechanism(2000.0, 1.0, half, ref stateD);
                if (a1 >= Low && a1 <= High && a2 >= 1997 && a2 <= 1999) hitsC++;
                if (b1 >= Low && b1 <= High && b2 >= 1997 && b2 <= 1999) hitsD++;
            }
            double qC = (double)hitsC / Samples, qD = (double)hitsD / Samples;
            Check(qC > 0 && qD > 0, "the composed interval has mass under both databases");
            double composedRatio = qC / qD;
            Check(composedRatio <= Math.Exp(epsilon) * 1.15,
                "two queries of epsilon/2 compose to the published epsilon (" + composedRatio.ToString("F4") + ")");
            Console.WriteLine("singleQueryRatio=" + ratio.ToString("F4") + " composedRatio=" + composedRatio.ToString("F4")
                + " expEps=" + Math.Exp(epsilon).ToString("F4"));
        }

        // ---- the Gaussian mechanism ------------------------------------------------------------
        {
            Check(Math.Abs(Mechanisms.GaussianSigma(1, 1, 1e-5) - Math.Sqrt(2 * Math.Log(125000))) < 1e-9,
                "sigma = sqrt(2 ln(1.25/delta))/epsilon matches the closed form");
            Check(Mechanisms.GaussianSigma(1, 2, 1e-5) < Mechanisms.GaussianSigma(1, 1, 1e-5),
                "a larger epsilon needs less Gaussian noise");
            Check(Mechanisms.GaussianSigma(1, 1, 1e-9) > Mechanisms.GaussianSigma(1, 1, 1e-5),
                "a smaller delta needs more Gaussian noise");
            Check(Mechanisms.GaussianSigma(3, 1, 1e-5) > Mechanisms.GaussianSigma(1, 1, 1e-5),
                "noise scales linearly with the sensitivity");

            ulong state = 0x1CEB00DAUL;
            const int Samples = 100000;
            var draws = new double[Samples];
            double sigma = Mechanisms.GaussianSigma(1, 1, 1e-5);
            for (int i = 0; i < Samples; i++) draws[i] = Mechanisms.GaussianMechanism(0, 1, 1, 1e-5, ref state);
            double measured = Math.Sqrt(Variance(draws));
            Check(Math.Abs(measured - sigma) < 0.03 * sigma,
                "the Gaussian sampler has the requested spread (measured " + measured.ToString("F4")
                + " vs " + sigma.ToString("F4") + ")");

            bool thrown = false;
            try { Mechanisms.GaussianSigma(1, 1, 1.5); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "delta outside (0,1) is rejected");
            thrown = false;
            try { Mechanisms.LaplaceMechanism(0, 1, 0, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "epsilon = 0 is rejected instead of dividing by zero");
        }

        // ---- the exponential mechanism ---------------------------------------------------------
        {
            var utilities = new double[] { 0, 1, 2, 3 };
            const int Samples = 200000;
            ulong state = 0xE4E4UL;
            var counts = new int[utilities.Length];
            for (int i = 0; i < Samples; i++) counts[Mechanisms.ExponentialMechanism(utilities, 1.0, 1.0, ref state)]++;
            var observed = new double[utilities.Length];
            for (int i = 0; i < utilities.Length; i++) observed[i] = (double)counts[i] / Samples;

            double normaliser = 0;
            var expected = new double[utilities.Length];
            for (int i = 0; i < utilities.Length; i++) expected[i] = Math.Exp(0.5 * utilities[i]);
            foreach (double w in expected) normaliser += w;
            for (int i = 0; i < utilities.Length; i++) expected[i] /= normaliser;

            bool close = true;
            for (int i = 0; i < utilities.Length; i++) if (Math.Abs(observed[i] - expected[i]) > 0.01) close = false;
            Check(close, "the exponential mechanism matches exp(eps u / 2 sens) normalised ("
                + observed[0].ToString("F4") + "/" + observed[1].ToString("F4") + "/"
                + observed[2].ToString("F4") + "/" + observed[3].ToString("F4") + ")");

            bool ratioBound = true;
            for (int a = 0; a < utilities.Length; a++)
            {
                for (int b = 0; b < utilities.Length; b++)
                {
                    double bound = Math.Exp(1.0 * (utilities[a] - utilities[b]) / (2 * 1.0)) * 1.05;
                    if (observed[a] / observed[b] > bound) ratioBound = false;
                }
            }
            Check(ratioBound, "every pair of output probabilities respects the exp(eps du / 2 sens) ratio bound");

            ulong loud = 0xFAFAUL;
            int picked = 0;
            const int LoudTrials = 5000;
            for (int i = 0; i < LoudTrials; i++) if (Mechanisms.ExponentialMechanism(utilities, 1.0, 60.0, ref loud) == 3) picked++;
            Check(picked >= LoudTrials - 1, "a huge epsilon selects the best candidate almost surely ("
                + picked + "/" + LoudTrials + ")");

            bool thrown = false;
            try { Mechanisms.ExponentialMechanism(Array.Empty<double>(), 1, 1, ref state); }
            catch (ArgumentException) { thrown = true; }
            Check(thrown, "the exponential mechanism rejects an empty candidate list");
            Console.WriteLine("expMechObserved=" + observed[0].ToString("F4") + "/" + observed[1].ToString("F4")
                + "/" + observed[2].ToString("F4") + "/" + observed[3].ToString("F4") + " loudPicks=" + picked);
        }

        // ---- a private histogram and the composition budget ------------------------------------
        {
            var counts = new long[] { 120, 90, 30, 200, 5 };
            double budget = 1.0;
            int queries = counts.Length;
            ulong state = 0xABCDUL;
            var noisy = Mechanisms.PrivateHistogram(counts, budget / queries, ref state);
            Check(noisy.Length == counts.Length, "every bucket is released");
            long totalNoisy = 0, totalTrue = 0;
            for (int i = 0; i < counts.Length; i++) { totalNoisy += noisy[i]; totalTrue += counts[i]; }
            Check(Math.Abs(totalNoisy - totalTrue) <= 60,
                "the composed release keeps the total roughly right (" + totalNoisy + " vs " + totalTrue + ")");
            double perQueryEpsilon = budget / queries;
            Check(Math.Abs(perQueryEpsilon * queries - budget) < 1e-12,
                "the per-query budgets add up to the total budget");

            // The total noise is bigger when the budget is split, which is the composition cost.
            ulong tight = 0x1111UL, loose = 0x2222UL;
            double splitError = 0, wholeError = 0;
            const int Runs = 4000;
            for (int r = 0; r < Runs; r++)
            {
                var a = Mechanisms.PrivateHistogram(counts, budget / queries, ref tight);
                var b = Mechanisms.PrivateHistogram(counts, budget, ref loose);
                for (int i = 0; i < counts.Length; i++)
                {
                    splitError += Math.Abs(a[i] - counts[i]);
                    wholeError += Math.Abs(b[i] - counts[i]);
                }
            }
            Check(splitError > wholeError,
                "splitting the budget over the histogram costs accuracy (" + (splitError / Runs).ToString("F2")
                + " vs " + (wholeError / Runs).ToString("F2") + " per run)");

            var empty = Mechanisms.PrivateHistogram(Array.Empty<long>(), 1.0, ref state);
            Check(empty.Length == 0, "an empty histogram releases nothing");
            Console.WriteLine("histogramSplitError=" + (splitError / Runs).ToString("F3")
                + " histogramWholeError=" + (wholeError / Runs).ToString("F3"));
        }

        // ---- determinism ----------------------------------------------------------------------
        {
            ulong a = 0x777UL, b = 0x777UL;
            double first = Mechanisms.LaplaceMechanism(10, 1, 1, ref a);
            double second = Mechanisms.LaplaceMechanism(10, 1, 1, ref b);
            Check(first == second, "the same seed reproduces the same release");
            Check(a == b, "and the same stream position");
        }

        Console.WriteLine("PASS");
    }
}
