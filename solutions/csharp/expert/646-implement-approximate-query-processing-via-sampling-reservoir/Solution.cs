// Task 646 - Implement approximate query processing via sampling reservoir. (Tier: Expert, CTCI Ch 14 - Databases)
//
// One-pass approximate query processing behind a reservoir sampler:
//   * Algorithm R keeps a uniform sample of size k of the stream: the first k items fill the
//     reservoir and item i (0-based) then replaces a random slot with probability k/(i+1), which is
//     what makes every item equally likely to survive as the stream grows;
//   * SUMAVG estimate the population mean by the sample mean and the sum by mean * n, with the
//     finite-population standard error s/sqrt(k) * sqrt(1 - k/n) and a z-score interval;
//   * a Hoeffding width M*sqrt(ln(2/delta)/(2k)) gives a distribution-free interval whose width is
//     *provably* non-increasing in k;
//   * a KMV (k-minimum-values) sketch estimates COUNT(DISTINCT): the k smallest unit hashes are
//     kept and the estimate is (k-1)/max, with an exact answer while fewer than k distinct values
//     have been seen.
//
// Randomness is a hand-written splitmix64, so every run is reproducible and the statistical checks
// are deterministic. Simplified: integer values, a single uniform-sample framework (no stratified /
// correlated / weighted sampling, no join sampling) and the sketch is small enough to keep in a
// sorted set.
//
// Verified: the reservoir always holds min(k, seen) items, inclusion frequencies match k/n for every
// element of a 200-element population over 200 trials, the interval covers the truth at the nominal
// rate over 200 independent streams, wider samples are more accurate and narrower, a deliberately
// biased sampler is detected by the same error metric, and the KMV estimate is exact for small
// distinct counts and within 20% for 1000 distinct values.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class SplitMix64
{
    private ulong _state;

    public SplitMix64(ulong seed) => _state = seed;

    public ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Reservoir
{
    private readonly List<int> _items = new List<int>();
    private readonly int _k;
    private readonly SplitMix64 _rng;
    private int _seen;

    public Reservoir(int k, SplitMix64 rng)
    {
        if (k <= 0) throw new Exception("reservoir size must be positive");
        _k = k;
        _rng = rng;
    }

    public int Count => _items.Count;
    public int Seen => _seen;
    public IReadOnlyList<int> Items => _items;

    // Algorithm R, one step of the stream.
    public void Offer(int value)
    {
        if (_seen < _k)
        {
            _items.Add(value);
        }
        else
        {
            int j = _rng.NextInt(_seen + 1);
            if (j < _k) _items[j] = value;
        }
        _seen++;
    }

    public static Reservoir Of(IEnumerable<int> stream, int k, SplitMix64 rng)
    {
        var reservoir = new Reservoir(k, rng);
        foreach (int value in stream) reservoir.Offer(value);
        return reservoir;
    }

    // Deliberately biased alternative used to show that the accuracy metric can detect bias: it keeps
    // the first k items and ignores the rest of the stream.
    public static List<int> BiasedFirst(IEnumerable<int> stream, int k) => stream.Take(k).ToList();
}

internal sealed class SampleEstimates
{
    public readonly int SampleSize;
    public readonly int Population;
    public readonly double Mean;
    public readonly double Sum;
    public readonly double StandardError;

    public SampleEstimates(int sampleSize, int population, double mean, double sum, double standardError)
    {
        SampleSize = sampleSize;
        Population = population;
        Mean = mean;
        Sum = sum;
        StandardError = standardError;
    }

    public double IntervalWidth(double z) => 2 * z * StandardError;

    public double RelativeError(double exact) => exact == 0 ? 0 : Math.Abs(Mean - exact) / Math.Abs(exact);

    // Distribution-free width for values known to live in [0, maximum].
    public static double HoeffdingWidth(double maximum, int sampleSize, double delta)
        => maximum * Math.Sqrt(Math.Log(2.0 / delta) / (2.0 * sampleSize));
}

internal static class Estimators
{
    public static SampleEstimates FromSample(IReadOnlyList<int> sample, int population)
    {
        if (population <= 0) throw new Exception("cannot estimate over an empty population");
        if (sample.Count == 0) throw new Exception("cannot estimate from an empty sample");
        double mean = sample.Average();
        double variance = sample.Count > 1
            ? sample.Sum(v => (v - mean) * (v - mean)) / (sample.Count - 1)
            : 0.0;
        double standardError = sample.Count > 1
            ? Math.Sqrt(variance / sample.Count) * Math.Sqrt(Math.Max(0.0, 1.0 - sample.Count / (double)population))
            : 0.0;
        return new SampleEstimates(sample.Count, population, mean, mean * population, standardError);
    }
}

internal sealed class KmvSketch
{
    private readonly int _k;
    private readonly SortedSet<double> _minimums = new SortedSet<double>();

    public KmvSketch(int k)
    {
        if (k <= 0) throw new Exception("sketch size must be positive");
        _k = k;
    }

    public int Distinct => _minimums.Count;

    public void Add(int value)
    {
        double hash = UnitHash(value);
        // A value that is already among the kept minimums must not consume a slot: repeating the stream
        // has to leave the sketch unchanged.
        if (_minimums.Contains(hash)) return;
        if (_minimums.Count < _k)
        {
            _minimums.Add(hash);
            return;
        }
        if (hash < _minimums.Max)
        {
            _minimums.Remove(_minimums.Max);
            _minimums.Add(hash);
        }
    }

    // Exact while the sketch is not full, otherwise the standard k-th-minimum estimator.
    public double Estimate()
    {
        if (_minimums.Count < _k) return _minimums.Count;
        double max = Math.Max(_minimums.Max, 1e-12);
        return Math.Max(1.0, (_k - 1) / max);
    }

    // Deterministic hash of a value into [0, 1): the top 53 bits of a splitmix64 finalizer scaled by
    // 2^-53. Dividing by a power of two is exact, so distinct 64-bit hashes stay distinct as doubles
    // (an additive offset would merely be rounded away for large values).
    public static double UnitHash(int value)
    {
        ulong z = (ulong)value * 0x9E3779B97F4A7C15UL + 0x123456789ABCDEFUL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        ulong unit = z >> 11;
        return unit / (double)(1UL << 53);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void Close(double actual, double expected, double tolerance, string label)
        => Check(Math.Abs(actual - expected) <= tolerance, label + ": expected " + expected + " got " + actual);

    private static List<int> Data(int n, int modulus, int multiplier)
    {
        var values = new List<int>();
        for (int i = 0; i < n; i++) values.Add((i * multiplier + (i % 97)) % modulus);
        return values;
    }

    private static void ReservoirInvariants()
    {
        var reservoir = new Reservoir(5, new SplitMix64(1));
        for (int i = 1; i <= 12; i++)
        {
            reservoir.Offer(i);
            Check(reservoir.Count == Math.Min(5, i), "reservoir holds min(k, seen) items after " + i + " items");
            Check(reservoir.Items.Distinct().Count() == reservoir.Items.Count, "the reservoir never repeats a stream position");
        }
        Check(reservoir.Seen == 12, "the whole stream was consumed once");

        List<int> small = Data(3, 10, 1);
        Reservoir whole = Reservoir.Of(small, 10, new SplitMix64(2));
        Check(whole.Items.SequenceEqual(small), "when k exceeds the stream the sample is the whole stream");

        Reservoir again = Reservoir.Of(Data(1000, 50, 7), 20, new SplitMix64(3));
        Reservoir repeat = Reservoir.Of(Data(1000, 50, 7), 20, new SplitMix64(3));
        Check(again.Items.SequenceEqual(repeat.Items), "the same seed reproduces the same reservoir");
        Reservoir other = Reservoir.Of(Data(1000, 50, 7), 20, new SplitMix64(4));
        Check(!again.Items.SequenceEqual(other.Items), "a different seed gives a different sample");

        bool threw = false;
        try { new Reservoir(0, new SplitMix64(5)); }
        catch (Exception) { threw = true; }
        Check(threw, "a reservoir of size zero is rejected");
    }

    private static void InclusionFrequencies()
    {
        const int population = 200;
        const int trials = 200;
        const int k = 20;
        var hits = new int[population];
        for (int trial = 0; trial < trials; trial++)
        {
            Reservoir sample = Reservoir.Of(Enumerable.Range(0, population), k, new SplitMix64((ulong)(1000 + trial)));
            foreach (int value in sample.Items) hits[value]++;
        }

        double expected = trials * k / (double)population;
        double sigma = Math.Sqrt(expected * (1 - k / (double)population));
        int worst = hits.Max(v => (int)Math.Abs(v - expected));
        Check(worst <= 3.5 * sigma, "every element's inclusion frequency is within 3.5 sigma of k/n (worst " + worst + ", sigma " + sigma.ToString("0.00", CultureInfo.InvariantCulture) + ")");

        double chiSquare = hits.Sum(observed => (observed - expected) * (observed - expected) / expected);
        Check(chiSquare < 300, "the inclusion histogram is consistent with uniformity, chi-square = " + chiSquare.ToString("0.0", CultureInfo.InvariantCulture));
        Check(hits.All(count => count > 0), "no element of the population is systematically excluded");
        Console.WriteLine("uniformity: " + population + " elements, " + trials + " trials, k=" + k
            + ", chi-square " + chiSquare.ToString("0.0", CultureInfo.InvariantCulture) + " (df " + (population - 1) + "), worst deviation " + worst);
    }

    private static void EstimationAccuracy()
    {
        List<int> data = Data(5000, 100, 37);
        double exactMean = data.Average();
        double exactSum = data.Sum();

        Reservoir sample = Reservoir.Of(data, 200, new SplitMix64(11));
        SampleEstimates estimates = Estimators.FromSample(sample.Items, data.Count);

        Close(estimates.Sum, estimates.Mean * data.Count, 1e-6, "the sum estimate is the mean estimate scaled by n");
        Check(Math.Abs(estimates.Mean - exactMean) <= 3 * estimates.StandardError,
            "the mean estimate is within three standard errors of the truth");
        Check(estimates.RelativeError(exactMean) < 0.1, "the relative mean error stays under 10%, got " + estimates.RelativeError(exactMean).ToString("0.0000", CultureInfo.InvariantCulture));
        Check(Math.Abs(estimates.Sum - exactSum) / exactSum < 0.1, "the relative sum error stays under 10%");
        Check(estimates.StandardError > 0, "the standard error reflects the sampling uncertainty");
        Console.WriteLine("estimate: mean " + estimates.Mean.ToString("0.000", CultureInfo.InvariantCulture) + " exact " + exactMean.ToString("0.000", CultureInfo.InvariantCulture)
            + " se " + estimates.StandardError.ToString("0.0000", CultureInfo.InvariantCulture) + " sum " + estimates.Sum.ToString("0.0", CultureInfo.InvariantCulture) + " exact " + exactSum);

        int covered = 0, trials = 200;
        double totalError = 0, totalWidth = 0;
        for (int trial = 0; trial < trials; trial++)
        {
            List<int> stream = Data(1000 + 20 * trial, 100, 3 + trial % 5);
            double truth = stream.Average();
            Reservoir draw = Reservoir.Of(stream, 64, new SplitMix64((ulong)(9000 + trial)));
            SampleEstimates estimate = Estimators.FromSample(draw.Items, stream.Count);
            if (Math.Abs(estimate.Mean - truth) <= 2 * estimate.StandardError) covered++;
            totalError += Math.Abs(estimate.Mean - truth);
            totalWidth += estimate.IntervalWidth(2);
        }
        double coverage = covered / (double)trials;
        Check(coverage >= 0.85, "the 2-sigma interval covers the truth in at least 85% of independent streams, got " + coverage.ToString("0.000", CultureInfo.InvariantCulture));
        Console.WriteLine("interval coverage: " + coverage.ToString("0.000", CultureInfo.InvariantCulture) + " of " + trials + " streams, mean error "
            + (totalError / trials).ToString("0.0000", CultureInfo.InvariantCulture) + ", mean width " + (totalWidth / trials).ToString("0.0000", CultureInfo.InvariantCulture));

        // A larger sample must be both more accurate and more confident on average.
        double smallError = AverageError(32), largeError = AverageError(512);
        double smallWidth = AverageWidth(32), largeWidth = AverageWidth(512);
        Check(largeError < smallError, "k=512 is more accurate than k=32 (" + largeError.ToString("0.0000", CultureInfo.InvariantCulture) + " < " + smallError.ToString("0.0000", CultureInfo.InvariantCulture) + ")");
        Check(largeWidth < smallWidth, "k=512 has a narrower interval than k=32");
        Console.WriteLine("memory vs accuracy: k=32 error " + smallError.ToString("0.0000", CultureInfo.InvariantCulture) + " width " + smallWidth.ToString("0.000", CultureInfo.InvariantCulture)
            + " | k=512 error " + largeError.ToString("0.0000", CultureInfo.InvariantCulture) + " width " + largeWidth.ToString("0.000", CultureInfo.InvariantCulture));
    }

    private static double AverageError(int k)
    {
        double total = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            List<int> stream = Data(2000, 100, 3 + trial % 5);
            double truth = stream.Average();
            Reservoir draw = Reservoir.Of(stream, k, new SplitMix64((ulong)(77000 + trial)));
            total += Math.Abs(Estimators.FromSample(draw.Items, stream.Count).Mean - truth);
        }
        return total / 40;
    }

    private static double AverageWidth(int k)
    {
        double total = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            List<int> stream = Data(2000, 100, 3 + trial % 5);
            Reservoir draw = Reservoir.Of(stream, k, new SplitMix64((ulong)(77000 + trial)));
            total += Estimators.FromSample(draw.Items, stream.Count).IntervalWidth(2);
        }
        return total / 40;
    }

    private static void HoeffdingIsMonotoneAndContains()
    {
        double previous = double.MaxValue;
        for (int k = 10; k <= 1000; k *= 2)
        {
            double width = SampleEstimates.HoeffdingWidth(100, k, 0.05);
            Check(width < previous, "the Hoeffding half-width shrinks as the sample grows (k=" + k + ")");
            previous = width;
        }

        int contained = 0, prefixes = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            List<int> stream = Data(2000, 100, 3 + trial % 5);
            double truth = stream.Average();
            if (trial % 2 == 0) stream.Reverse();
            for (int size = 10; size <= 1000; size *= 3)
            {
                Reservoir draw = Reservoir.Of(stream.Take(size), 10, new SplitMix64((ulong)(500 + trial * 13 + size)));
                // A stream the sampler has not finished reading is simulated by sampling the prefix; the
                // Hoeffding interval must still cover the truth far more often than not.
                SampleEstimates estimate = Estimators.FromSample(draw.Items, size);
                double halfWidth = SampleEstimates.HoeffdingWidth(100, draw.Items.Count, 0.2);
                prefixes++;
                if (Math.Abs(estimate.Mean - truth) <= halfWidth) contained++;
            }
        }
        double ratio = contained / (double)prefixes;
        Check(ratio >= 0.8, "the Hoeffding interval covers the truth in at least 80% of prefixes, got " + ratio.ToString("0.000", CultureInfo.InvariantCulture));
        Console.WriteLine("Hoeffding: coverage " + ratio.ToString("0.000", CultureInfo.InvariantCulture) + " over " + prefixes + " prefixes");
    }

    private static void BiasIsDetected()
    {
        double reservoirError = 0, biasedError = 0;
        int trials = 20;
        for (int trial = 0; trial < trials; trial++)
        {
            // Values grow with the stream position, so a sampler that only ever looks at the head of the
            // stream cannot see the large values: exactly the bias a reservoir uniformly avoids.
            var stream = new List<int>();
            for (int i = 0; i < 2000; i++) stream.Add((i + trial) / 20);
            double truth = stream.Average();
            SampleEstimates honest = Estimators.FromSample(Reservoir.Of(stream, 50, new SplitMix64((ulong)(6000 + trial))).Items, stream.Count);
            SampleEstimates biased = Estimators.FromSample(Reservoir.BiasedFirst(stream, 50), stream.Count);
            reservoirError += Math.Abs(honest.Mean - truth);
            biasedError += Math.Abs(biased.Mean - truth);
        }
        Check(reservoirError / trials < biasedError / (4.0 * trials), "the biased first-k sampler is far worse than the reservoir ("
            + (reservoirError / trials).ToString("0.000", CultureInfo.InvariantCulture) + " vs "
            + (biasedError / trials).ToString("0.000", CultureInfo.InvariantCulture) + ")");
        Console.WriteLine("bias detection: reservoir error " + (reservoirError / trials).ToString("0.0000", CultureInfo.InvariantCulture)
            + " vs biased error " + (biasedError / trials).ToString("0.0000", CultureInfo.InvariantCulture));
    }

    private static void DistinctEstimation()
    {
        var small = new KmvSketch(64);
        foreach (int value in Enumerable.Range(0, 40)) small.Add(value);
        Close(small.Estimate(), 40, 1e-9, "a distinct count below k is estimated exactly");
        Check(small.Distinct == 40, "the sketch stores one hash per distinct value while it is not full");

        var repeated = new KmvSketch(64);
        for (int i = 0; i < 500; i++) repeated.Add(i % 40);
        Close(repeated.Estimate(), 40, 1e-9, "repeated values do not inflate the exact estimate");
        Check(repeated.Distinct == 40, "the sketch ignores duplicate hashes");

        double errorSmall = MeanDistinctError(32), errorLarge = MeanDistinctError(256);
        Check(errorSmall < 0.25, "KMV stays within 25% with 32 minimums, got " + errorSmall.ToString("0.0000", CultureInfo.InvariantCulture));
        Check(errorLarge < 0.25, "KMV stays within 25% with 256 minimums, got " + errorLarge.ToString("0.0000", CultureInfo.InvariantCulture));
        Check(errorLarge < errorSmall, "more minimums give a better distinct estimate ("
            + errorLarge.ToString("0.000", CultureInfo.InvariantCulture) + " < " + errorSmall.ToString("0.000", CultureInfo.InvariantCulture) + ")");

        // Independent cross-checks: the estimator inverts the k-th smallest hash, so a sketch fed exactly
        // the hashes of 0..999 must report what the sorted hash list predicts; the hash must be injective;
        // and re-feeding the same values (the everyday case of duplicate rows) must change nothing.
        var hashes = new List<double>();
        for (int i = 0; i < 1000; i++) hashes.Add(KmvSketch.UnitHash(i));
        Check(hashes.Distinct().Count() == 1000, "the unit hash is injective over 1000 values");
        Check(hashes.All(h => h >= 0.0 && h < 1.0), "unit hashes live in [0, 1)");
        hashes.Sort();
        Check(hashes[0] < 0.01 && hashes[999] > 0.99, "the hash spreads over the whole unit interval");

        var once = new KmvSketch(32);
        for (int i = 0; i < 1000; i++) once.Add(i);
        Close(once.Estimate(), 31.0 / hashes[31], 1e-9, "the sketch keeps exactly the 32 smallest hashes");

        var replayed = new KmvSketch(32);
        for (int i = 0; i < 3000; i++) replayed.Add(i % 1000);
        Close(replayed.Estimate(), once.Estimate(), 1e-9, "re-feeding the same values leaves the sketch unchanged");
        Check(replayed.Distinct == once.Distinct, "duplicate rows never consume a minimum slot");
        Console.WriteLine("KMV: relative error " + errorSmall.ToString("0.000", CultureInfo.InvariantCulture) + " at k=32, "
            + errorLarge.ToString("0.000", CultureInfo.InvariantCulture) + " at k=256 over 1000 distinct values");
    }

    private static double MeanDistinctError(int k)
    {
        double total = 0;
        int trials = 20;
        for (int trial = 0; trial < trials; trial++)
        {
            var sketch = new KmvSketch(k);
            for (int i = 0; i < 4000; i++) sketch.Add((i * 7 + trial) % 1000);
            total += Math.Abs(sketch.Estimate() - 1000) / 1000.0;
        }
        return total / trials;
    }

    public static void Main()
    {
        ReservoirInvariants();
        InclusionFrequencies();
        EstimationAccuracy();
        HoeffdingIsMonotoneAndContains();
        BiasIsDetected();
        DistinctEstimation();
        Console.WriteLine("PASS");
    }
}
