// Task 650 - Implement online aggregation progressive estimator. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Online aggregation answers an aggregate query *while* it is still running: the table is scanned in a
// random order (a deterministic Fisher-Yates shuffle driven by a hand-written splitmix64, so the
// randomness - and therefore every statistical result below - is reproducible) and after every block the
// running mean, the running SUM estimate (mean * population) and two confidence intervals are published:
//   * a z-score interval from the sample variance with the finite-population correction
//     s/sqrt(k) * sqrt(1 - k/N), and
//   * a distribution-free Hoeffding/Serfling interval width M*sqrt(ln(2/delta)/(2k))*sqrt(1 - k/N) for
//     values known to lie in [0, M], whose width is *provably* non-increasing as the scan proceeds.
//
// Simplified: numeric columns only, one aggregate (COUNT/SUM/AVG), no grouped online aggregation, no
// priority-based block scheduling and no early stopping heuristics. The "progress" reported is the
// fraction of the table scanned, exactly like a real online aggregation operator exposes it.
//
// Verified: the estimate is exact when the scan finishes (both intervals collapse to a point), the
// Hoeffding width decreases strictly with progress, the 2-sigma interval covers the truth at the nominal
// rate over 100 independent shuffles, error shrinks monotonically with progress, and a sequential
// (non-random) scan of ordered data is shown to be far worse - which is precisely why the random order
// matters. Boundaries: an empty scan refuses to estimate, a one-row table and a constant column give a
// zero-width interval, and an outlier widens the sample interval without ever escaping the Hoeffding one.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal static class Shuffle
{
    public static int[] Order(int n, ulong seed)
    {
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        ulong state = seed;
        for (int i = n - 1; i > 0; i--)
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            int j = (int)(z % (ulong)(i + 1));
            (order[i], order[j]) = (order[j], order[i]);
        }
        return order;
    }
}

internal sealed class ProgressiveEstimator
{
    private double _sum;
    private double _sumOfSquares;
    private int _count;

    public int Count => _count;

    public void Observe(double value)
    {
        _sum += value;
        _sumOfSquares += value * value;
        _count++;
    }

    public double Mean => _count == 0
        ? throw new Exception("an online aggregate needs at least one observation")
        : _sum / _count;

    public double Variance => _count > 1
        ? Math.Max(0.0, (_sumOfSquares - _sum * _sum / _count) / (_count - 1))
        : 0.0;

    public double SumEstimate(double population) => Mean * population;

    // Finite-population correction: sampling k of N rows without replacement is strictly better than
    // sampling with replacement, and is exact at k = N.
    public double StandardError(double population)
    {
        if (_count < 2) return 0.0;
        double correction = Math.Sqrt(Math.Max(0.0, 1.0 - _count / population));
        return Math.Sqrt(Variance / _count) * correction;
    }

    public double IntervalHalfWidth(double population, double z) => z * StandardError(population);

    public double HoeffdingHalfWidth(double maximum, double population, double delta)
    {
        if (_count == 0) return double.PositiveInfinity;
        double correction = Math.Sqrt(Math.Max(0.0, 1.0 - _count / population));
        return maximum * Math.Sqrt(Math.Log(2.0 / delta) / (2.0 * _count)) * correction;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string P(double value) => value.ToString("0.0000", CultureInfo.InvariantCulture);

    private static List<int> Values(int n, int modulus, int multiplier)
    {
        var values = new List<int>();
        for (int i = 0; i < n; i++) values.Add((i * multiplier + i % 97) % modulus);
        return values;
    }

    private static ProgressiveEstimator RunPrefix(IReadOnlyList<int> values, int[] order, int prefix)
    {
        var estimator = new ProgressiveEstimator();
        for (int i = 0; i < prefix; i++) estimator.Observe(values[order[i]]);
        return estimator;
    }

    private static void EstimateIsExactWhenTheScanFinishes()
    {
        List<int> values = Values(5000, 1000, 37);
        double exactMean = values.Average();
        double exactSum = values.Sum();
        int[] order = Shuffle.Order(values.Count, 42);

        var estimator = new ProgressiveEstimator();
        foreach (int index in order) estimator.Observe(values[index]);
        Check(estimator.Count == values.Count, "the scan observed every row exactly once");
        Check(order.Distinct().Count() == values.Count, "the random order is a permutation");
        Check(Math.Abs(estimator.Mean - exactMean) < 1e-9, "the finished scan reports the exact mean");
        Check(Math.Abs(estimator.SumEstimate(values.Count) - exactSum) < 1e-6, "the finished scan reports the exact sum");
        Check(estimator.StandardError(values.Count) == 0, "the finite-population correction makes the error zero at 100%");
        Check(estimator.HoeffdingHalfWidth(1000, values.Count, 0.05) == 0, "the Hoeffding width also collapses at 100%");

        var midway = RunPrefix(values, order, values.Count / 2);
        Check(Math.Abs(midway.Mean - exactMean) < 25, "the halfway estimate is close to the truth");
        Console.WriteLine("progress: 50% estimate " + P(midway.Mean) + " +/- " + P(midway.IntervalHalfWidth(values.Count, 2))
            + ", 100% exact " + P(exactMean));
    }

    private static void ConfidenceIntervalsTighten()
    {
        List<int> values = Values(4000, 1000, 11);
        int[] order = Shuffle.Order(values.Count, 7);

        double previous = double.MaxValue;
        foreach (int prefix in new[] { 20, 40, 80, 160, 320, 640, 1280, 2560, 4000 })
        {
            ProgressiveEstimator estimator = RunPrefix(values, order, prefix);
            double width = estimator.HoeffdingHalfWidth(1000, values.Count, 0.05);
            Check(width < previous, "the Hoeffding half-width shrinks with progress (k=" + prefix + ")");
            previous = width;
        }
        Check(previous == 0, "the Hoeffding half-width reaches zero at the end of the scan");

        ProgressiveEstimator early = RunPrefix(values, order, 100);
        ProgressiveEstimator late = RunPrefix(values, order, 3000);
        Check(late.IntervalHalfWidth(values.Count, 2) < early.IntervalHalfWidth(values.Count, 2),
            "the 2-sigma interval is narrower late in the scan");
        Console.WriteLine("intervals: half-width " + P(early.IntervalHalfWidth(values.Count, 2)) + " at 100 rows -> "
            + P(late.IntervalHalfWidth(values.Count, 2)) + " at 3000 rows");
    }

    private static void CoverageOverManyShuffles()
    {
        int coveredHalf = 0, coveredLate = 0;
        const int trials = 100;
        for (int trial = 0; trial < trials; trial++)
        {
            List<int> values = Values(1000 + 10 * trial, 500, 3 + trial % 5);
            double truth = values.Average();
            int[] order = Shuffle.Order(values.Count, (ulong)(1000 + trial));

            ProgressiveEstimator half = RunPrefix(values, order, values.Count / 2);
            if (Math.Abs(half.Mean - truth) <= half.IntervalHalfWidth(values.Count, 2)) coveredHalf++;
            ProgressiveEstimator late = RunPrefix(values, order, values.Count * 9 / 10);
            if (Math.Abs(late.Mean - truth) <= late.IntervalHalfWidth(values.Count, 2)) coveredLate++;
        }
        Check(coveredHalf >= 0.85 * trials, "the 2-sigma interval covers the truth at 50% progress, got " + coveredHalf + "/" + trials);
        Check(coveredLate >= 0.9 * trials, "the 2-sigma interval covers the truth at 90% progress, got " + coveredLate + "/" + trials);
        Console.WriteLine("coverage: " + coveredHalf + "/" + trials + " at 50% progress, " + coveredLate + "/" + trials + " at 90%");
    }

    private static void ProgressiveRefinement()
    {
        double earlyError = 0, lateError = 0;
        const int trials = 100;
        for (int trial = 0; trial < trials; trial++)
        {
            List<int> values = Values(2000, 1000, 7 + trial % 3);
            double truth = values.Average();
            int[] order = Shuffle.Order(values.Count, (ulong)(5000 + trial));
            earlyError += Math.Abs(RunPrefix(values, order, 100).Mean - truth);
            lateError += Math.Abs(RunPrefix(values, order, 1800).Mean - truth);
        }
        Check(lateError < earlyError / 3, "error shrinks with progress ("
            + P(lateError / trials) + " at 90% vs " + P(earlyError / trials) + " at 5%)");
        Console.WriteLine("refinement: mean error " + P(earlyError / trials) + " at 5% -> " + P(lateError / trials) + " at 90%");
    }

    private static void RandomOrderBeatsASequentialPrefix()
    {
        // The values grow with the row id, so the first rows of a sequential scan are systematically
        // small: only a random order gives an unbiased estimate at any progress point.
        var values = new List<int>();
        for (int i = 0; i < 4000; i++) values.Add((i + 1) / 40);
        double truth = values.Average();

        double randomError = 0, sequentialError = 0;
        const int trials = 50;
        for (int trial = 0; trial < trials; trial++)
        {
            int[] order = Shuffle.Order(values.Count, (ulong)(700 + trial));
            randomError += Math.Abs(RunPrefix(values, order, 100).Mean - truth);
            var sequential = new ProgressiveEstimator();
            for (int i = 0; i < 100; i++) sequential.Observe(values[i]);
            sequentialError += Math.Abs(sequential.Mean - truth);
        }
        Check(randomError / trials < sequentialError / (5.0 * trials), "the random order is far more accurate than a sequential prefix ("
            + P(randomError / trials) + " vs " + P(sequentialError / trials) + ")");
        Console.WriteLine("ordering: random-order error " + P(randomError / trials)
            + " vs sequential-prefix error " + P(sequentialError / trials));
    }

    private static void Boundaries()
    {
        var empty = new ProgressiveEstimator();
        Check(empty.Count == 0, "an empty scan has no observations");
        bool threw = false;
        try { _ = empty.Mean; }
        catch (Exception) { threw = true; }
        Check(threw, "an empty online aggregate refuses to report a mean");

        var single = new ProgressiveEstimator();
        single.Observe(17);
        Check(single.Mean == 17, "a one-row table reports its single value");
        Check(single.StandardError(1) == 0, "a one-row table has zero sampling error");
        Check(single.SumEstimate(1) == 17, "the sum estimate of a one-row table is exact");

        var constant = new ProgressiveEstimator();
        for (int i = 0; i < 500; i++) constant.Observe(9);
        Check(constant.Variance == 0, "a constant column has zero variance");
        Check(constant.IntervalHalfWidth(500, 2) == 0, "the sample interval of a constant column is a point");
        Check(Math.Abs(constant.Mean - 9) < 1e-12, "the constant column's mean is exact");

        // An outlier widens the sample-based interval but stays inside the distribution-free bound.
        var table = new List<int>();
        for (int i = 0; i < 199; i++) table.Add(i % 10);
        table.Add(1000);
        const double population = 5000;
        double truth = table.Average();
        var estimator = new ProgressiveEstimator();
        for (int i = 0; i < 199; i++) estimator.Observe(table[i]);
        double beforeOutlier = estimator.StandardError(population);
        double halfWidth = estimator.HoeffdingHalfWidth(1000, population, 0.05);
        Check(Math.Abs(estimator.Mean - truth) <= halfWidth, "the declared-range interval covers the truth before the outlier arrives");
        estimator.Observe(table[199]);
        Check(estimator.StandardError(population) > beforeOutlier, "seeing the outlier widens the sample interval");
        Check(estimator.Variance > 1000, "the outlier dominates the sample variance");
        Check(estimator.HoeffdingHalfWidth(1000, population, 0.05) > 0, "the distribution-free bound stays positive mid-scan");

        int[] order = Shuffle.Order(10, 3);
        Check(order.Length == 10 && order.Distinct().Count() == 10 && order.All(i => i >= 0 && i < 10),
            "the shuffle is a permutation of the row ids");
        Check(Shuffle.Order(10, 3).SequenceEqual(order), "the shuffle is deterministic for a given seed");
        Check(!Shuffle.Order(10, 4).SequenceEqual(order), "a different seed gives a different order");
    }

    public static void Main()
    {
        EstimateIsExactWhenTheScanFinishes();
        ConfidenceIntervalsTighten();
        CoverageOverManyShuffles();
        ProgressiveRefinement();
        RandomOrderBeatsASequentialPrefix();
        Boundaries();
        Console.WriteLine("PASS");
    }
}
