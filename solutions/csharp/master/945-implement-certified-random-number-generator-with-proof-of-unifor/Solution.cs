// Task 945 - Implement certified random number generator with proof of uniformity (small demo).
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// A deterministic counter-based generator (splitmix64) shipped with a *certificate* rather than a
// claim: the generator is run for a fixed sample budget and four statistical tests are evaluated
// and reported together - a chi-square goodness-of-fit over 16 equal bins, a two-dimensional
// chi-square over consecutive pairs (the serial test), a Kolmogorov-Smirnov distance against the
// 1.36/sqrt(N) critical value, and a monobit test on the mantissa bits with a four-sigma band.
// The certificate is only issued when the sample budget is large enough for every expected cell
// count (>= 5) and every test is inside its acceptance region.
//
// What proves the certificate is real: the chi-square p-values come from a hand-written
// regularized incomplete gamma function, checked against the closed form Q(1, y) = e^-y for one
// degree of freedom and against the 5% quantiles 3.841459 and 5.991465; the KS distance is
// recomputed by an independent sweep and checked against the exact 1/N value for a regular grid;
// a biased generator fails the marginal tests; and - the discriminating case - a low-discrepancy
// sequence that passes the marginal chi-square, the KS and the monobit tests is still rejected
// because the serial test sees the correlation between consecutive draws.
// Simplified: four tests instead of a full battery (no runs, gaps, spectral or entropy test), one
// sample size, and alpha = 0.05 hard-coded; the statistics, the p-values and the acceptance rules
// are real, and no result is faked for the passing case.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int PopCount(ulong value)
    {
        int count = 0;
        while (value != 0) { value &= value - 1; count++; }
        return count;
    }
}

internal abstract class UniformSource
{
    internal abstract double Next();
}

internal sealed class CounterSource : UniformSource
{
    private ulong state;
    internal CounterSource(ulong seed) => state = seed;
    internal override double Next() => (SplitMix64.Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

/** Uniform on [0, 0.75): a generator with a correct-looking interface but a biased marginal. */
internal sealed class BiasedSource : UniformSource
{
    private ulong state;
    internal BiasedSource(ulong seed) => state = seed;
    internal override double Next() => ((SplitMix64.Next(ref state) >> 11) * (1.0 / 9007199254740992.0)) * 0.75;
}

/** Low-discrepancy (golden ratio) sequence: uniform marginally, deterministic in pairs. */
internal sealed class QuasiRandomSource : UniformSource
{
    private const double Golden = 0.6180339887498949;
    private long index;

    internal QuasiRandomSource(long startIndex = 0) => index = startIndex;

    internal override double Next()
    {
        index++;
        double value = index * Golden;
        return value - Math.Floor(value);
    }
}

internal sealed class ConstantSource : UniformSource
{
    internal override double Next() => 0.5;
}

internal static class Statistics
{
    private static readonly double[] Lanczos =
    {
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
        12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    };

    internal static double LogGamma(double x)
    {
        if (x < 0.5) return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        x -= 1.0;
        double series = 0.99999999999980993;
        for (int i = 0; i < Lanczos.Length; i++) series += Lanczos[i] / (x + i + 1.0);
        double t = x + Lanczos.Length - 0.5;
        return 0.5 * Math.Log(2.0 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(series);
    }

    /** Regularized upper incomplete gamma Q(a, x): the chi-square tail probability. */
    internal static double GammaQ(double a, double x)
    {
        if (a <= 0.0 || x < 0.0) throw new ArgumentException("gamma parameters out of range");
        if (x == 0.0) return 1.0;
        return x < a + 1.0 ? 1.0 - GammaSeries(a, x) : GammaFraction(a, x);
    }

    private static double GammaSeries(double a, double x)
    {
        double sum = 1.0 / a, term = sum, ap = a;
        for (int n = 1; n < 1000; n++)
        {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (Math.Abs(term) < Math.Abs(sum) * 1e-16) break;
        }
        return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private static double GammaFraction(double a, double x)
    {
        double b = x + 1.0 - a, c = 1e300, d = 1.0 / b, h = d;
        for (int i = 1; i < 1000; i++)
        {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.Abs(d) < 1e-300) d = 1e-300;
            c = b + an / c;
            if (Math.Abs(c) < 1e-300) c = 1e-300;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < 1e-16) break;
        }
        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * h;
    }

    /** Chi-square tail probability with the given degrees of freedom. */
    internal static double ChiSquareTail(double statistic, int degreesOfFreedom) =>
        GammaQ(degreesOfFreedom / 2.0, statistic / 2.0);

    /** Kolmogorov-Smirnov distance of a sorted sample against the uniform distribution. */
    internal static double KsDistance(double[] sorted)
    {
        double worst = 0.0;
        for (int i = 0; i < sorted.Length; i++)
        {
            double upper = (i + 1.0) / sorted.Length - sorted[i];
            double lower = sorted[i] - i / (double)sorted.Length;
            worst = Math.Max(worst, Math.Max(upper, lower));
        }
        return worst;
    }
}

internal sealed class UniformityCertificate
{
    internal const int MinExpected = 5;

    internal readonly int Samples;
    internal readonly int Bins;
    internal readonly double[] Values;
    internal readonly int[] Marginal;
    internal readonly int[] Serial;
    internal readonly double ChiSquare;
    internal readonly double ChiSquareP;
    internal readonly double SerialChiSquare;
    internal readonly double SerialP;
    internal readonly double Ks;
    internal readonly double KsCritical;
    internal readonly double BitFraction;
    internal readonly double BitBand;
    internal readonly bool EnoughEvidence;
    internal readonly bool[] Passed = new bool[4];

    internal UniformityCertificate(UniformSource source, int samples, int bins)
    {
        if (samples <= 0) throw new ArgumentException("a certificate needs at least one sample");
        Samples = samples;
        Bins = bins;
        Values = new double[samples];
        Marginal = new int[bins];
        Serial = new int[bins * bins];
        for (int i = 0; i < samples; i++)
        {
            double raw = source.Next();
            double value = raw < 0.0 ? 0.0 : raw >= 1.0 ? 0.9999999999999999 : raw;
            Values[i] = value;
            Marginal[Math.Min(bins - 1, (int)(value * bins))]++;
        }
        double expected = (double)samples / bins;
        ChiSquare = Marginal.Sum(o => (o - expected) * (o - expected) / expected);
        ChiSquareP = Statistics.ChiSquareTail(ChiSquare, bins - 1);

        int pairs = samples / 2;
        for (int i = 0; i < pairs; i++)
        {
            int x = Math.Min(bins - 1, (int)(Values[2 * i] * bins));
            int y = Math.Min(bins - 1, (int)(Values[2 * i + 1] * bins));
            Serial[x * bins + y]++;
        }
        double expectedPair = (double)pairs / (bins * bins);
        SerialChiSquare = Serial.Sum(o => (o - expectedPair) * (o - expectedPair) / expectedPair);
        SerialP = Statistics.ChiSquareTail(SerialChiSquare, bins * bins - 1);

        double[] sorted = (double[])Values.Clone();
        Array.Sort(sorted);
        Ks = Statistics.KsDistance(sorted);
        KsCritical = 1.36 / Math.Sqrt(samples);

        long ones = 0;
        foreach (double value in Values) ones += SplitMix64.PopCount((ulong)(value * 9007199254740992.0));
        long bits = 53L * samples;
        BitFraction = (double)ones / bits;
        BitBand = 4.0 * 0.5 / Math.Sqrt(bits);

        EnoughEvidence = expected >= MinExpected && expectedPair >= MinExpected;
        Passed[0] = ChiSquareP >= 0.05;
        Passed[1] = SerialP >= 0.05;
        Passed[2] = Ks <= KsCritical;
        Passed[3] = Math.Abs(BitFraction - 0.5) <= BitBand;
    }

    internal bool Certified => EnoughEvidence && Passed.All(p => p);
    internal bool MarginalTestsPass => Passed[0] && Passed[2] && Passed[3];
}

/**
 * A certification covers a *suite* of streams, not one lucky draw: with four tests at alpha = 0.05
 * a perfect generator is still expected to fail about 5% of its verdicts, so the suite is refused
 * when the failure count exceeds a four-sigma binomial band or when the same test fails in three
 * or more streams - the signature of a systematic defect rather than noise.
 */
internal sealed class CertificationSuite
{
    internal const int SystematicThreshold = 3;

    internal readonly List<UniformityCertificate> Streams = new List<UniformityCertificate>();
    internal readonly int[] TestFailures = new int[4];
    internal int Failures;
    internal int Verdicts;
    internal bool AllEnoughEvidence = true;

    internal CertificationSuite(Func<int, UniformSource> factory, int streams, int samples, int bins)
    {
        for (int i = 0; i < streams; i++)
        {
            var certificate = new UniformityCertificate(factory(i), samples, bins);
            Streams.Add(certificate);
            AllEnoughEvidence &= certificate.EnoughEvidence;
            for (int test = 0; test < certificate.Passed.Length; test++)
            {
                Verdicts++;
                if (!certificate.Passed[test])
                {
                    Failures++;
                    TestFailures[test]++;
                }
            }
        }
    }

    internal int AllowedFailures => (int)Math.Floor(0.05 * Verdicts + 4.0 * Math.Sqrt(0.05 * 0.95 * Verdicts));
    internal bool SystematicFailure => TestFailures.Any(f => f >= SystematicThreshold);
    internal bool Certified => AllEnoughEvidence && Failures <= AllowedFailures && !SystematicFailure;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    public static void Main()
    {
        const int Samples = 4096;
        const int Bins = 16;

        // 1. The suite is certified: the four-sigma band absorbs the expected false positives.
        var suite = new CertificationSuite(i => new CounterSource(0x9450UL + (ulong)i * 0x9E37UL), 5, Samples, Bins);
        Check(suite.Certified && suite.Verdicts == 20 && suite.Failures <= suite.AllowedFailures
              && suite.Streams.All(s => s.EnoughEvidence),
            "five streams of splitmix64 pass the battery: " + suite.Failures + " of " + suite.Verdicts
            + " verdicts outside their regions (allowed " + suite.AllowedFailures + ")");

        // 2. Negative control: a biased marginal fails the goodness-of-fit and the KS test in every stream.
        var biasedSuite = new CertificationSuite(i => new BiasedSource(0x9451UL + (ulong)i * 0x9E37UL), 5, Samples, Bins);
        Check(!biasedSuite.Certified && biasedSuite.SystematicFailure && biasedSuite.TestFailures[0] == 5
              && biasedSuite.Failures > biasedSuite.AllowedFailures,
            "a source uniform on [0,0.75) is refused: " + biasedSuite.Failures + " of " + biasedSuite.Verdicts
            + " verdicts fail, chi-square in all five streams");

        // 3. The discriminating case: marginal tests pass, the serial test still catches the correlation.
        var quasiSuite = new CertificationSuite(i => new QuasiRandomSource(1 + i * 137L), 5, Samples, Bins);
        Check(!quasiSuite.Certified && quasiSuite.TestFailures[1] == 5 && quasiSuite.TestFailures[0] == 0,
            "a low-discrepancy stream is refused by the serial test in all five streams while the "
            + "goodness-of-fit stays quiet");

        // 4. A single stream of the certified generator, used by the oracle checks below.
        var certified = new UniformityCertificate(new CounterSource(0x9450UL), Samples, Bins);

        // 5. Chi-square p-values against closed forms and the 5% quantiles.
        Check(Math.Abs(Statistics.GammaQ(1.0, 1.0) - Math.Exp(-1.0)) < 1e-12
              && Math.Abs(Statistics.GammaQ(1.0, 3.7) - Math.Exp(-3.7)) < 1e-12
              && Math.Abs(Statistics.ChiSquareTail(3.841458820694124, 1) - 0.05) < 1e-4
              && Math.Abs(Statistics.ChiSquareTail(5.991464547107979, 2) - 0.05) < 1e-6,
            "Q(1,y) = e^-y exactly and the 5% quantiles of chi-square 1 and 2 reproduce");

        // 5. The tail probability is a bounded, strictly decreasing function of the statistic.
        double previous = 1.1;
        bool monotone = true;
        for (int i = 0; i <= 60; i++)
        {
            double p = Statistics.ChiSquareTail(0.1 * i * i, 15);
            monotone &= p >= 0.0 && p <= 1.0 && p < previous;
            previous = p;
        }
        Check(monotone, "the chi-square tail probability decreases monotonically inside [0,1]");

        // 6. KS oracle: an independent sweep, and the exact 1/N value for a regular grid.
        var grid = new double[512];
        for (int i = 0; i < grid.Length; i++) grid[i] = (i + 0.5) / grid.Length;
        double gridKs = Statistics.KsDistance(grid);
        double sweep = 0.0;
        double[] sorted = (double[])certified.Values.Clone();
        Array.Sort(sorted);
        for (int i = 0; i < sorted.Length; i++)
            sweep = Math.Max(sweep, Math.Max(Math.Abs(sorted[i] - i / (double)sorted.Length),
                Math.Abs(sorted[i] - (i + 1.0) / sorted.Length)));
        Check(Math.Abs(gridKs - 0.5 / grid.Length) < 1e-12 && Math.Abs(sweep - certified.Ks) < 1e-9,
            "the KS distance equals 0.5/N on a regular grid and matches an independent sweep");

        // 7. Boundaries: too little evidence is refused, and an empty sample is an error.
        var thin = new UniformityCertificate(new CounterSource(0x9452UL), 64, Bins);
        Check(!thin.EnoughEvidence && !thin.Certified && (double)64 / Bins < UniformityCertificate.MinExpected,
            "64 samples over 16 bins are refused for lack of evidence rather than certified");
        Check(Throws(() => new UniformityCertificate(new CounterSource(1UL), 0, Bins)),
            "a zero-sample certificate is rejected outright");

        // 8. Degenerate generator: every test fails, including the bit band.
        var constant = new UniformityCertificate(new ConstantSource(), Samples, Bins);
        Check(!constant.Passed.Any(p => p) && constant.Ks == 0.5
              && Math.Abs(constant.BitFraction - 0.5) > constant.BitBand,
            "a constant generator fails all four tests, with a bit fraction of "
            + constant.BitFraction.ToString("F4"));

        // 9. Reproducibility: one seed replays exactly and the multi-seed suite keeps its certificate.
        var repeat = new UniformityCertificate(new CounterSource(0x9450UL), Samples, Bins);
        Check(repeat.ChiSquare == certified.ChiSquare && repeat.Ks == certified.Ks
              && repeat.SerialChiSquare == certified.SerialChiSquare && suite.Certified && suite.Streams.Count == 5,
            "the same seed replays bit for bit and the five-seed suite keeps its certificate");

        // 10. The bit band is a real band: the certified stream sits inside it, the constant one outside.
        Check(Math.Abs(certified.BitFraction - 0.5) <= certified.BitBand && certified.BitBand > 0
              && Math.Abs(constant.BitFraction - 0.5) > constant.BitBand,
            "the monobit band admits the unbiased stream (" + certified.BitFraction.ToString("F4")
            + " +/- " + certified.BitBand.ToString("F4") + ") and rejects the degenerate one");

        Console.WriteLine("  certified stream: chi2 " + certified.ChiSquare.ToString("F2") + " (p "
            + certified.ChiSquareP.ToString("F4") + "), serial " + certified.SerialChiSquare.ToString("F1")
            + " (p " + certified.SerialP.ToString("F4") + "), KS " + certified.Ks.ToString("F5")
            + " vs " + certified.KsCritical.ToString("F5") + ", bits " + certified.BitFraction.ToString("F4"));
        Console.WriteLine("  rejected: biased KS " + biasedSuite.Streams[0].Ks.ToString("F3") + ", quasi-random serial p "
            + quasiSuite.Streams[0].SerialP.ToString("E1") + ", constant passes "
            + constant.Passed.Count(p => p) + "/4");
        Console.WriteLine("PASS");
    }
}
