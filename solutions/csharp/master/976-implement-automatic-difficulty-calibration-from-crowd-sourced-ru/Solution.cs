// Task 976 - Implement automatic difficulty calibration from crowd-sourced runtimes.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Crowd runtimes per problem are cleaned with a Tukey fence (values outside q1 - f*IQR and
// q3 + f*IQR are discarded), summarised by a median and a trimmed mean, and then mapped onto a
// five-step difficulty ladder with decade boundaries: median < 10ms is level 1, < 100ms level 2,
// < 1000ms level 3, < 10000ms level 4, everything slower level 5.
//
// Verified: hand-declared median/quartile/trimmed-mean values, the median from a quickselect oracle
// against the sort-based median, the Tukey fence rejecting exactly the brute-force-detected
// outliers with the median left unchanged, the "fewer than three samples is insufficient" boundary,
// the decade monotonicity that multiplying every runtime by ten raises the level by exactly one, and
// recovery of an injected four-step difficulty ladder from seeded jittered samples.
// Simplification: runtimes are integer milliseconds supplied in memory, one sample set per problem,
// no language/runtime normalisation and no machine-speed regression of the crowd data.
using System;
using System.Collections.Generic;

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

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Calibration
    {
        public long Median, LowerFence, UpperFence;
        public int Level, Rejected, Accepted;
        public double TrimmedMean;
        public bool Sufficient;
    }

    private static long Median(List<long> values)
    {
        var sorted = new List<long>(values);
        sorted.Sort();
        int n = sorted.Count;
        if (n == 0) throw new ArgumentException("no samples");
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }

    /** Oracle: k-th order statistic by Hoare quickselect, independent of the sort-based median. */
    private static long SelectKth(long[] data, int k)
    {
        int lo = 0, hi = data.Length - 1;
        while (lo < hi)
        {
            long pivot = data[(lo + hi) / 2];
            int i = lo, j = hi;
            while (i <= j)
            {
                while (data[i] < pivot) i++;
                while (data[j] > pivot) j--;
                if (i <= j)
                {
                    long swap = data[i];
                    data[i] = data[j];
                    data[j] = swap;
                    i++;
                    j--;
                }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else break;
        }
        return data[k];
    }

    private static long MedianByQuickSelect(List<long> values)
    {
        var data = values.ToArray();
        int n = data.Length;
        if (n == 0) throw new ArgumentException("no samples");
        if (n % 2 == 1) return SelectKth(data, n / 2);
        long lower = SelectKth((long[])data.Clone(), n / 2 - 1), upper = SelectKth(data, n / 2);
        return (lower + upper) / 2;
    }

    private static double TrimmedMean(List<long> sorted, double trimFraction)
    {
        int cut = (int)(sorted.Count * trimFraction);
        if (sorted.Count - 2 * cut < 1) return sorted.Count == 0 ? 0.0 : sorted[sorted.Count / 2];
        double sum = 0.0;
        for (int i = cut; i < sorted.Count - cut; i++) sum += sorted[i];
        return sum / (sorted.Count - 2 * cut);
    }

    /** Decade ladder: level 1 below 10ms, then one level per power of ten, capped at 5. */
    private static int Level(long medianMilliseconds)
    {
        if (medianMilliseconds <= 0) return 0;
        int level = 1;
        long threshold = 10;
        while (level < 5 && medianMilliseconds >= threshold)
        {
            level++;
            threshold *= 10;
        }
        return level;
    }

    private static Calibration Calibrate(List<long> samples, double iqrFactor, double trimFraction)
    {
        var result = new Calibration();
        if (samples.Count < 3) return result;
        var sorted = new List<long>(samples);
        sorted.Sort();
        long q1 = sorted[sorted.Count / 4];
        long q3 = sorted[(3 * sorted.Count) / 4];
        double iqr = q3 - q1;
        double lower = q1 - iqrFactor * iqr;
        double upper = q3 + iqrFactor * iqr;
        result.LowerFence = (long)Math.Floor(lower);
        result.UpperFence = (long)Math.Ceiling(upper);
        var kept = new List<long>();
        foreach (long value in sorted)
        {
            if (value < lower || value > upper) result.Rejected++;
            else kept.Add(value);
        }
        result.Accepted = kept.Count;
        if (kept.Count < 2) return result;
        result.Sufficient = true;
        result.Median = Median(kept);
        result.TrimmedMean = TrimmedMean(kept, trimFraction);
        result.Level = Level(result.Median);
        return result;
    }

    private static List<long> Range(long from, long to)
    {
        var values = new List<long>();
        for (long value = from; value <= to; value++) values.Add(value);
        return values;
    }

    private static List<long> Ladder(ulong seed, int decades, int samplesPerStep)
    {
        _state = seed;
        var values = new List<long>();
        long centre = 5;   // well inside the first decade, so +/-10% jitter cannot cross a boundary
        for (int step = 0; step < decades; step++)
        {
            for (int i = 0; i < samplesPerStep; i++)
            {
                double jitter = 0.9 + 0.2 * NextUnit();
                long value = (long)Math.Round(centre * jitter);
                values.Add(Math.Max(1, value));
            }
            centre *= 10;
        }
        return values;
    }

    public static void Main()
    {
        // Happy path against hand-declared quartiles, median and trimmed mean.
        List<long> ten = Range(1, 10);
        Calibration tenResult = Calibrate(ten, 1.5, 0.2);
        Check(tenResult.Sufficient && tenResult.Accepted == 10 && tenResult.Rejected == 0
              && tenResult.Median == 5 && tenResult.LowerFence == -5 && tenResult.UpperFence == 16,
            "the 1..10 sample is accepted in full with median 5 and Tukey fences -5 and 16");
        Check(Math.Abs(tenResult.TrimmedMean - 5.5) < 1e-12 && tenResult.Level == 1,
            "the 20% trimmed mean of 1..10 is 5.5 and a 5ms median is the easiest level");

        // Oracle: quickselect median equals sort-based median on seeded values.
        var random = new List<long>();
        _state = 976UL;
        for (int i = 0; i < 40; i++) random.Add((long)Math.Round(1.0 + NextUnit() * 900000.0));
        Check(MedianByQuickSelect(random) == Median(random) && MedianByQuickSelect(ten) == Median(ten),
            "the quickselect oracle reproduces the sort-based median on 40 seeded runtimes and on 1..10");

        // Outlier handling: one 100000ms sample in a 12-value run must be fenced off.
        var polluted = Range(10, 21);
        polluted.Add(100000);
        Calibration pollutedResult = Calibrate(polluted, 1.5, 0.2);
        Check(pollutedResult.Rejected == 1 && pollutedResult.Accepted == 12 && pollutedResult.Median == 15
              && Calibrate(Range(10, 21), 1.5, 0.2).Median == 15,
            "the Tukey fence rejects exactly the single huge outlier and the median ignores it");
        double rawMean = 0.0;
        foreach (long value in polluted) rawMean += value;
        rawMean /= polluted.Count;
        Check(rawMean > pollutedResult.TrimmedMean * 10.0,
            "the raw mean is inflated more than tenfold while the trimmed mean stays sane");
        int bruteForceRejects = 0;
        foreach (long value in polluted)
            if (value < pollutedResult.LowerFence || value > pollutedResult.UpperFence) bruteForceRejects++;
        Check(bruteForceRejects == pollutedResult.Rejected,
            "the reported rejection count matches a brute-force fence scan");

        // Boundary: fewer than three samples, and a degenerate identical sample.
        Check(!Calibrate(new List<long>(), 1.5, 0.2).Sufficient && !Calibrate(new List<long> { 42 }, 1.5, 0.2).Sufficient,
            "no samples and a single sample are both insufficient for calibration");
        Check(!Calibrate(new List<long> { 1, 1000000 }, 1.5, 0.2).Sufficient, "two samples are insufficient");
        Calibration flat = Calibrate(new List<long> { 250, 250, 250, 250 }, 1.5, 0.2);
        Check(flat.Sufficient && flat.Rejected == 0 && flat.Median == 250 && flat.Level == 3,
            "an identical sample set has zero interquartile spread and still calibrates");
        // Decade monotonicity: x10 runtimes raise the level by exactly one until the cap.
        long[] decades = { 1, 10, 100, 1000, 10000, 1000000 };
        int[] expected = { 1, 2, 3, 4, 5, 5 };
        bool monotone = true;
        for (int i = 0; i < decades.Length; i++) monotone &= Level(decades[i]) == expected[i];
        for (long value = 1; value < 100000; value *= 3)
            monotone &= Level(value * 10) == Math.Min(5, Level(value) + 1);
        Check(monotone, "the level ladder is monotone and each decade adds exactly one level");

        // Recovery of an injected four-step difficulty ladder from seeded jittered samples.
        const int perStep = 15;
        List<long> ladder = Ladder(424242UL, 4, perStep);
        int recovered = 0;
        for (int step = 0; step < 4; step++)
        {
            List<long> slice = ladder.GetRange(step * perStep, perStep);
            if (Calibrate(slice, 1.5, 0.2).Level == step + 1) recovered++;
        }
        Check(recovered == 4, "all four injected difficulty steps are recovered from jittered runtimes ("
            + recovered + "/4)");

        // Monotone mapping over random medians: no pair of medians is inverted by the ladder.
        _state = 112233UL;
        var medians = new List<long>();
        for (int i = 0; i < 12; i++) medians.Add((long)Math.Round(1.0 + NextUnit() * 20000.0));
        int inversions = 0;
        for (int i = 0; i < medians.Count; i++)
            for (int j = 0; j < medians.Count; j++)
                if (medians[i] < medians[j] && Level(medians[i]) > Level(medians[j])) inversions++;
        Check(inversions == 0, "no pair of the 12 seeded medians inverts the difficulty order");

        Console.WriteLine("ladder recovery: " + recovered + "/4 injected steps, inversions " + inversions);
        Console.WriteLine("PASS");
    }
}
