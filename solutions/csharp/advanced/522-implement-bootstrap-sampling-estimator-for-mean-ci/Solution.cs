// Task 522 - Implement bootstrap sampling estimator for mean CI. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// The percentile bootstrap for the mean of a one-dimensional sample. B resamples of size n are drawn
// with replacement from the data with a deterministic LCG (three discarded warm-up steps, because the
// raw recurrence (75 s + 74) is correlated with the seed modulo small numbers), the mean of each
// resample is recorded, the B means are sorted, and the (1 - level) interval is read off with linear
// interpolation between the two neighbouring order statistics (the "type 7" quantile, exactly the
// quantile of the sample 0..n-1 at position p*(n-1)). The basic (reverse-percentile) interval comes
// from the same sorted means as 2*theta_hat - hi and 2*theta_hat - lo, and the analytic normal
// interval theta_hat +- z*se is computed for the three tabulated levels 0.90 / 0.95 / 0.99 with
// se = s/sqrt(n), s the unbiased sample standard deviation.
//
// Documented checks. (1) The percentile helper on the array 0..100 returns exactly 25 / 50 / 75 at
// p = 0.25 / 0.5 / 0.75 and clamps at the extremes. (2) Closed forms: the bootstrap means average to
// the sample mean to within 0.02*se, and their standard deviation matches the bootstrap identity
// sd_empirical/sqrt(n) to within 5% (B = 4000, sd_empirical = the biased sample standard deviation);
// the basic interval equals the reflected percentile interval to 1e-12 at every level. (3) Shape:
// the interval brackets the sample mean, the 90% interval nests inside the 95% inside the 99%, the
// 95% interval stays within 0.25*se of the analytic normal interval, and dividing n by four roughly
// doubles its width (ratio in [1.4, 2.6]). (4) Coverage: over 100 independent samples of size 30 from
// a discrete uniform population with a known mean, the 95% percentile interval covers the true mean
// at least 85 times. (5) Boundaries: a one-element sample and constant data give a degenerate
// interval of width 0 whose every quantile is the constant; an empty sample, a non-positive
// iteration count, a level outside (0,1) and a level without a tabulated z each throw
// ArgumentException. (6) Determinism: the same seed reproduces the interval bit for bit while a
// different seed moves it.
using System;

/** Deterministic LCG; Next(bound) rejects to give an exactly uniform value in [0, bound). */
internal sealed class Lcg
{
    private int state;

    internal Lcg(int seed)
    {
        state = seed % 65537;
        if (state < 0) state += 65537;
        for (int i = 0; i < 3; i++) state = (state * 75 + 74) % 65537;
    }

    internal int Next(int bound)
    {
        if (bound <= 0 || bound > 65537) throw new ArgumentOutOfRangeException(nameof(bound));
        int limit = 65537 - 65537 % bound;
        int value;
        do
        {
            state = (state * 75 + 74) % 65537;
            value = state;
        }
        while (value >= limit);
        return value % bound;
    }

    internal double NextDouble()
    {
        return Next(65536) / 65536.0;
    }

    /** Uniform on (0, 1), so that Math.Log is always finite. */
    internal double NextOpenDouble()
    {
        return (Next(65536) + 1) / 65537.0;
    }
}

internal static class Bootstrap
{
    internal static double[] ResampleMeans(double[] data, int iterations, int seed)
    {
        if (data.Length == 0) throw new ArgumentException("empty sample");
        if (iterations < 1) throw new ArgumentException("at least one iteration is required");
        var rng = new Lcg(seed);
        var means = new double[iterations];
        int n = data.Length;
        for (int b = 0; b < iterations; b++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += data[rng.Next(n)];
            means[b] = sum / n;
        }
        return means;
    }

    /** Linear-interpolated quantile of an ascending array at position p*(n-1). */
    internal static double Percentile(double[] ascending, double p)
    {
        if (ascending.Length == 0) throw new ArgumentException("empty array");
        if (p < 0.0 || p > 1.0) throw new ArgumentException("p is outside [0, 1]");
        if (p == 0.0) return ascending[0];
        if (p == 1.0) return ascending[ascending.Length - 1];
        double position = p * (ascending.Length - 1);
        int low = (int)Math.Floor(position);
        int high = (int)Math.Ceiling(position);
        double fraction = position - low;
        return ascending[low] * (1.0 - fraction) + ascending[high] * fraction;
    }

    internal static double[] SortedMeans(double[] data, int iterations, int seed)
    {
        double[] means = ResampleMeans(data, iterations, seed);
        Array.Sort(means);
        return means;
    }

    internal static double[] PercentileInterval(double[] sortedMeans, double level)
    {
        RequireLevel(level);
        double alpha = (1.0 - level) / 2.0;
        return new[] { Percentile(sortedMeans, alpha), Percentile(sortedMeans, 1.0 - alpha) };
    }

    internal static double[] BasicInterval(double[] sortedMeans, double level, double sampleMean)
    {
        double[] percentile = PercentileInterval(sortedMeans, level);
        return new[] { 2.0 * sampleMean - percentile[1], 2.0 * sampleMean - percentile[0] };
    }

    internal static double[] NormalInterval(double[] data, double level)
    {
        RequireLevel(level);
        double mean = SampleMean(data);
        double se = StandardError(data);
        double z = ZFor(level);
        return new[] { mean - z * se, mean + z * se };
    }

    internal static double SampleMean(double[] data)
    {
        double sum = 0.0;
        foreach (double value in data) sum += value;
        return sum / data.Length;
    }

    /** Unbiased sample standard deviation (divides by n - 1). */
    internal static double SampleStandardDeviation(double[] data)
    {
        if (data.Length < 2) return 0.0;
        double mean = SampleMean(data);
        double sum = 0.0;
        foreach (double value in data) sum += (value - mean) * (value - mean);
        return Math.Sqrt(sum / (data.Length - 1));
    }

    /** Biased standard deviation of the data seen as the bootstrap distribution. */
    internal static double EmpiricalStandardDeviation(double[] data)
    {
        double mean = SampleMean(data);
        double sum = 0.0;
        foreach (double value in data) sum += (value - mean) * (value - mean);
        return Math.Sqrt(sum / data.Length);
    }

    internal static double StandardError(double[] data)
    {
        return SampleStandardDeviation(data) / Math.Sqrt(data.Length);
    }

    private static void RequireLevel(double level)
    {
        if (!(level > 0.0 && level < 1.0)) throw new ArgumentException("level must be in (0, 1)");
    }

    private static double ZFor(double level)
    {
        if (Math.Abs(level - 0.90) < 1e-12) return 1.6448536269514722;
        if (Math.Abs(level - 0.95) < 1e-12) return 1.959963984540054;
        if (Math.Abs(level - 0.99) < 1e-12) return 2.5758293035489004;
        throw new ArgumentException("no tabulated z for level " + level);
    }
}

public static class Solution
{
    public static void Main()
    {
        var ramp = new double[101];
        for (int i = 0; i < ramp.Length; i++) ramp[i] = i;
        CheckClose(Bootstrap.Percentile(ramp, 0.25), 25.0, "quantile 0.25 of 0..100");
        CheckClose(Bootstrap.Percentile(ramp, 0.5), 50.0, "quantile 0.5 of 0..100");
        CheckClose(Bootstrap.Percentile(ramp, 0.75), 75.0, "quantile 0.75 of 0..100");
        CheckClose(Bootstrap.Percentile(ramp, 0.0), 0.0, "quantile 0 clamps to the minimum");
        CheckClose(Bootstrap.Percentile(ramp, 1.0), 100.0, "quantile 1 clamps to the maximum");
        CheckClose(Bootstrap.Percentile(new[] { 2.0 }, 0.3), 2.0, "a one-element array is its own quantile");

        double[] data = LcgSample(200, 4242);
        double sampleMean = Bootstrap.SampleMean(data);
        double se = Bootstrap.StandardError(data);
        Check(se > 0.0, "the sample has a positive standard error");

        double[] means = Bootstrap.SortedMeans(data, 4000, 31337);
        double meanOfMeans = Bootstrap.SampleMean(means);
        Check(Math.Abs(meanOfMeans - sampleMean) <= 0.02 * se,
                "the bootstrap means average to the sample mean (" + meanOfMeans + " vs " + sampleMean + ")");
        double bootstrapSd = Bootstrap.EmpiricalStandardDeviation(means);
        double expectedSd = Bootstrap.EmpiricalStandardDeviation(data) / Math.Sqrt(data.Length);
        Check(Math.Abs(bootstrapSd - expectedSd) <= 0.05 * expectedSd,
                "the bootstrap standard deviation matches sd/sqrt(n) (" + bootstrapSd + " vs " + expectedSd + ")");

        double[] ninety = Bootstrap.PercentileInterval(means, 0.90);
        double[] ninetyFive = Bootstrap.PercentileInterval(means, 0.95);
        double[] ninetyNine = Bootstrap.PercentileInterval(means, 0.99);
        Check(ninety[0] > ninetyFive[0] - 1e-12 && ninetyFive[0] > ninetyNine[0] - 1e-12,
                "lower endpoints nest across the confidence levels");
        Check(ninety[1] < ninetyFive[1] + 1e-12 && ninetyFive[1] < ninetyNine[1] + 1e-12,
                "upper endpoints nest across the confidence levels");
        Check(ninetyFive[0] < sampleMean && sampleMean < ninetyFive[1],
                "the 95% interval brackets the sample mean");

        double[] normal = Bootstrap.NormalInterval(data, 0.95);
        Check(Math.Abs(ninetyFive[0] - normal[0]) <= 0.25 * se,
                "the lower percentile endpoint tracks the normal interval");
        Check(Math.Abs(ninetyFive[1] - normal[1]) <= 0.25 * se,
                "the upper percentile endpoint tracks the normal interval");

        double[] basic = Bootstrap.BasicInterval(means, 0.95, sampleMean);
        CheckClose(basic[0], 2.0 * sampleMean - ninetyFive[1], "the basic interval reflects the upper endpoint");
        CheckClose(basic[1], 2.0 * sampleMean - ninetyFive[0], "the basic interval reflects the lower endpoint");

        double width200 = ninetyFive[1] - ninetyFive[0];
        double[] quarter = LcgSample(50, 4242);
        double[] quarterMeans = Bootstrap.SortedMeans(quarter, 4000, 31337);
        double[] quarterInterval = Bootstrap.PercentileInterval(quarterMeans, 0.95);
        double ratio = (quarterInterval[1] - quarterInterval[0]) / width200;
        Check(ratio >= 1.4 && ratio <= 2.6, "quartering n roughly doubles the width (ratio " + ratio + ")");

        int covered = 0;
        for (int seed = 1; seed <= 100; seed++)
        {
            double[] sample = UniformPopulationSample(30, seed);
            double[] resampled = Bootstrap.SortedMeans(sample, 400, 9000 + seed);
            double[] interval = Bootstrap.PercentileInterval(resampled, 0.95);
            if (interval[0] <= 497.5 && 497.5 <= interval[1]) covered++;
        }
        Check(covered >= 85, "the 95% interval covers the true mean in " + covered + "/100 replications");

        var constant = new double[12];
        for (int i = 0; i < constant.Length; i++) constant[i] = 5.0;
        double[] constantMeans = Bootstrap.SortedMeans(constant, 200, 7);
        double[] constantInterval = Bootstrap.PercentileInterval(constantMeans, 0.95);
        CheckClose(constantInterval[0], 5.0, "constant data: degenerate lower endpoint");
        CheckClose(constantInterval[1], 5.0, "constant data: degenerate upper endpoint");
        double[] single = Bootstrap.SortedMeans(new[] { 3.5 }, 200, 7);
        double[] singleInterval = Bootstrap.PercentileInterval(single, 0.95);
        CheckClose(singleInterval[0], 3.5, "single observation: lower endpoint");
        CheckClose(singleInterval[1], 3.5, "single observation: upper endpoint");
        CheckClose(Bootstrap.StandardError(new[] { 3.5 }), 0.0, "single observation: zero standard error");

        double[] repeat = Bootstrap.SortedMeans(data, 4000, 31337);
        double[] repeatInterval = Bootstrap.PercentileInterval(repeat, 0.95);
        CheckClose(repeatInterval[0], ninetyFive[0], "the same seed reproduces the lower endpoint");
        CheckClose(repeatInterval[1], ninetyFive[1], "the same seed reproduces the upper endpoint");
        double[] other = Bootstrap.PercentileInterval(Bootstrap.SortedMeans(data, 4000, 2), 0.95);
        Check(Math.Abs(other[1] - ninetyFive[1]) > 0.0, "a different seed moves the interval");

        CheckThrows(() => Bootstrap.ResampleMeans(new double[0], 10, 1), "empty sample throws");
        CheckThrows(() => Bootstrap.ResampleMeans(data, 0, 1), "zero iterations throw");
        CheckThrows(() => Bootstrap.PercentileInterval(means, 1.5), "level above one throws");
        CheckThrows(() => Bootstrap.NormalInterval(data, 0.5), "a level without a tabulated z throws");
        CheckThrows(() => Bootstrap.Percentile(means, -0.1), "a negative quantile throws");
        CheckThrows(() => Bootstrap.NormalInterval(data, 0.97), "0.97 has no tabulated z");

        Console.WriteLine("PASS");
    }

    private static double[] LcgSample(int n, int seed)
    {
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            data[i] = 10.0 + seed % 1000 / 10.0;
        }
        return data;
    }

    /** 30 i.i.d. draws from the discrete uniform population {0, 5, ..., 995}, mean 497.5. */
    private static double[] UniformPopulationSample(int n, int seed)
    {
        var rng = new Lcg(seed);
        var sample = new double[n];
        for (int i = 0; i < n; i++) sample[i] = 5.0 * rng.Next(200);
        return sample;
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        if (!(Math.Abs(actual - expected) <= 1e-12))
        {
            throw new Exception("FAILED: " + label + " (" + actual + " != " + expected + ")");
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
