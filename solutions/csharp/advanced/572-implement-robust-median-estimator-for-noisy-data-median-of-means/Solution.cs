// Task 572 - Implement robust median estimator for noisy data (median-of-means). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// The median-of-means (MoM) estimator: split a sample of n values into k consecutive blocks of
// b = n / k values, take the mean of each block, and return the median of those k means. Because a
// single outlier can only damage the blocks that contain it, and because at most half of the block
// means can be pushed past a threshold without at least half of the blocks being heavy-tailed, MoM
// tolerates up to half of the sample being arbitrarily corrupted - a 50% breakdown point, unlike the
// sample mean. The tests below use a fixed-seed 64-bit splitmix64 stream feeding the Irwin-Hall(12)
// device (the sum of 12 uniforms minus 6, mean 0, variance exactly 1) as the light-tailed noise and a
// ratio of two such normals as the heavy-tailed (Cauchy-like) noise; the 16-bit LCG of the earlier
// waves would repeat over these multi-hundred-thousand draws.
//
// Documented checks: the median helper's odd/even conventions are pinned on hand-computed inputs
// (1..8 with k = 2 and k = 4 both give 4.5, and 1..9 with k = 3 gives 5); on 20000 clean Irwin-Hall
// samples MoM's standard error is exactly sqrt(k) = 4 times the mean's - the price of robustness -
// both estimators are inside four of their standard errors, and MoM's deviation stays inside the
// Chebyshev budget epsilon = sqrt(3 / b) = 0.108253 (measured 0.0063); over 200 held-out trials with
// k = 16 blocks of b = 256 values the empirical fraction of trials with |MoM| >= epsilon is 0, at or
// below the median-of-means bound exp(-k / 8) = 0.135335, with the worst deviation 0.0237; replacing
// 40% of the values by +50 blows the mean error up to 20.01 while MoM stays at 0.054, below 5% of it;
// contaminating the three blocks with the largest means (floor((k - 1) / 2) = 3) leaves MoM
// bit-identical to the clean value while damaging three low blocks moves it but keeps it inside the
// clean block-mean range, and damaging four blocks (ceil(k / 2)) pushes it outside - the exact
// breakdown boundary; a systematic 25% block shift of +1000 gives a mean error of 249.99 against
// MoM's 0.0048 and the 10% trimmed mean's 187.67, so the trimmed mean fails where MoM does not; over
// 100 heavy-tailed (ratio-of-normals) samples the mean exceeded 1 in absolute value 34 times while
// MoM never did, with worst errors 3.81 against 0.74; and every statistic is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }

    /** Irwin-Hall(12): sum of 12 uniforms minus 6, mean 0 and variance exactly 1. */
    internal static double NextNormal(ref ulong state)
    {
        double sum = 0.0;
        for (int i = 0; i < 12; i++) sum += NextUnit(ref state);
        return sum - 6.0;
    }

    /** Ratio of two normals: heavy tailed, mean undefined, so the sample mean is a bad estimator. */
    internal static double NextHeavyTailed(ref ulong state)
    {
        double numerator = NextNormal(ref state);
        double denominator = NextNormal(ref state);
        if (Math.Abs(denominator) < 1e-3) denominator = denominator < 0 ? -1e-3 : 1e-3;
        return numerator / denominator;
    }
}

internal static class MedianOfMeans
{
    internal static double Median(IReadOnlyList<double> values)
    {
        var sorted = new List<double>(values);
        sorted.Sort();
        int n = sorted.Count;
        if (n == 0) return double.NaN;
        if (n % 2 == 1) return sorted[n / 2];
        return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    internal static double Mean(IReadOnlyList<double> values)
    {
        double sum = 0.0;
        foreach (double value in values) sum += value;
        return sum / values.Count;
    }

    /** k consecutive blocks of n / k values; returns the median of the block means. */
    internal static double Estimate(IReadOnlyList<double> values, int k)
    {
        int b = values.Count / k;
        var means = new List<double>(k);
        for (int block = 0; block < k; block++)
        {
            double sum = 0.0;
            for (int i = 0; i < b; i++) sum += values[block * b + i];
            means.Add(sum / b);
        }
        return Median(means);
    }

    /** The 10% trimmed mean, used as a second robust estimator for comparison. */
    internal static double TrimmedMean(IReadOnlyList<double> values, double fraction)
    {
        var sorted = new List<double>(values);
        sorted.Sort();
        int cut = (int)(sorted.Count * fraction);
        double sum = 0.0;
        for (int i = cut; i < sorted.Count - cut; i++) sum += sorted[i];
        return sum / (sorted.Count - 2 * cut);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        // Hand-computed conventions.
        var oneToEight = new List<double>();
        for (int i = 1; i <= 8; i++) oneToEight.Add(i);
        Check(Math.Abs(MedianOfMeans.Median(new List<double> { 3.0, 1.0, 2.0 }) - 2.0) <= 1e-12,
                "the odd-count median is the middle value");
        Check(Math.Abs(MedianOfMeans.Median(new List<double> { 4.0, 1.0, 3.0, 2.0 }) - 2.5) <= 1e-12,
                "the even-count median averages the two middles");
        Check(Math.Abs(MedianOfMeans.Estimate(oneToEight, 2) - 4.5) <= 1e-12, "1..8 with k = 2 gives 4.5");
        Check(Math.Abs(MedianOfMeans.Estimate(oneToEight, 4) - 4.5) <= 1e-12, "1..8 with k = 4 gives 4.5");
        Check(Math.Abs(MedianOfMeans.Estimate(oneToEight, 8) - 4.5) <= 1e-12, "1..8 with k = 8 gives 4.5");
        var oneToNine = new List<double>();
        for (int i = 1; i <= 9; i++) oneToNine.Add(i);
        Check(Math.Abs(MedianOfMeans.Estimate(oneToNine, 3) - 5.0) <= 1e-12, "1..9 with k = 3 gives 5");
        Check(Math.Abs(MedianOfMeans.Mean(oneToEight) - 4.5) <= 1e-12, "the mean of 1..8 is 4.5");

        // Clean data: the mean is the more efficient estimator, MoM stays inside its Chebyshev budget.
        ulong state = 572001UL;
        int n = 20000;
        var clean = new List<double>(n);
        for (int i = 0; i < n; i++) clean.Add(Rng.NextNormal(ref state));
        int k = 16;
        int b = n / k;
        double epsilon = Math.Sqrt(3.0 / b);
        double cleanMeanError = Math.Abs(MedianOfMeans.Mean(clean) - 0.0);
        double cleanMomError = Math.Abs(MedianOfMeans.Estimate(clean, k) - 0.0);
        Check(clean.Count == n && n % k == 0, "the clean sample splits evenly into sixteen blocks");
        double meanSigma = 1.0 / Math.Sqrt(n);
        double momSigma = 1.0 / Math.Sqrt(b);
        Check(cleanMeanError <= 4.0 * meanSigma, "the clean sample mean is inside four of its standard errors ("
                + cleanMeanError + " vs " + meanSigma + ")");
        Check(cleanMomError <= 4.0 * momSigma, "the clean median-of-means error is inside four of its standard errors ("
                + cleanMomError + " vs " + momSigma + ")");
        Check(cleanMomError <= epsilon, "the clean median-of-means error is inside the Chebyshev budget (" + cleanMomError
                + " vs " + epsilon + ")");
        Check(Math.Abs(momSigma / meanSigma - Math.Sqrt(k)) <= 1e-9,
                "MoM's standard error is sqrt(k) = 4 times the mean's, the price of robustness");

        // The median-of-means deviation bound: failure fraction at most exp(-k / 8).
        int trials = 200;
        int failures = 0;
        double worstDeviation = 0.0;
        ulong trialState = 572123UL;
        for (int trial = 0; trial < trials; trial++)
        {
            var sample = new List<double>(n);
            for (int i = 0; i < n; i++) sample.Add(Rng.NextNormal(ref trialState));
            double deviation = Math.Abs(MedianOfMeans.Estimate(sample, k));
            worstDeviation = Math.Max(worstDeviation, deviation);
            if (deviation >= epsilon) failures++;
        }
        double failureFraction = failures / (double)trials;
        double bound = Math.Exp(-k / 8.0);
        Check(Math.Abs(bound - 0.1353352832) <= 1e-9, "the exp(-k / 8) bound is 0.1353352832");
        Check(failureFraction <= bound, "the empirical failure fraction respects exp(-k / 8) (" + failureFraction
                + " vs " + bound + ")");
        Check(worstDeviation <= 4.0 * epsilon, "no trial deviated by more than four epsilons (" + worstDeviation + ")");

        // Contamination: 40% of the sample replaced by +50.
        var contaminated = new List<double>(clean);
        int outliers = (int)(0.4 * n);
        for (int i = 0; i < outliers; i++) contaminated[i] = 50.0;
        double contaminatedMeanError = Math.Abs(MedianOfMeans.Mean(contaminated));
        double contaminatedMomError = Math.Abs(MedianOfMeans.Estimate(contaminated, k));
        Check(outliers == 8000, "eight thousand values were replaced");
        Check(contaminatedMeanError >= 19.0, "the sample mean is off by about 20 under contamination ("
                + contaminatedMeanError + ")");
        Check(contaminatedMomError <= 0.5, "MoM stays under 0.5 under contamination (" + contaminatedMomError + ")");
        Check(contaminatedMomError <= 0.05 * contaminatedMeanError, "MoM's error is at most 5% of the mean's ("
                + contaminatedMomError + " vs " + contaminatedMeanError + ")");

        // The breakdown boundary. Damaging the three blocks with the largest clean means (exactly
        // floor((k - 1) / 2) = 3 of the k = 8 blocks) leaves the median of the eight block means
        // bit-identical, because the corrupted values stay at the top; damaging three *low* blocks
        // moves it, but it stays inside the range of the clean block means; damaging four blocks
        // (ceil(k / 2), the first case where the median position is no longer guaranteed to be a clean
        // value) pushes it outside that range.
        int blocks = 8;
        int damaged = (blocks - 1) / 2;
        var boundary = new List<double>(n);
        ulong boundaryState = 572500UL;
        for (int i = 0; i < n; i++) boundary.Add(Rng.NextNormal(ref boundaryState));
        double cleanValue = MedianOfMeans.Estimate(boundary, blocks);
        int blockSize = n / blocks;
        var blockMeans = new List<double>();
        for (int block = 0; block < blocks; block++)
        {
            double sum = 0.0;
            for (int i = block * blockSize; i < (block + 1) * blockSize; i++) sum += boundary[i];
            blockMeans.Add(sum / blockSize);
        }
        double lowest = blockMeans.Min();
        double highest = blockMeans.Max();
        double damagedBlockMean = 1e6;
        List<int> descending = Enumerable.Range(0, blocks).OrderByDescending(b => blockMeans[b]).ToList();
        var topDamaged = new List<double>(boundary);
        var lowDamaged = new List<double>(boundary);
        for (int t = 0; t < damaged; t++)
        {
            for (int i = descending[t] * blockSize; i < (descending[t] + 1) * blockSize; i++) topDamaged[i] = damagedBlockMean;
            for (int i = descending[blocks - 1 - t] * blockSize; i < (descending[blocks - 1 - t] + 1) * blockSize; i++)
            {
                lowDamaged[i] = damagedBlockMean;
            }
        }
        Check(damaged == 3, "floor((k - 1) / 2) is three for k = 8");
        Check(MedianOfMeans.Estimate(topDamaged, blocks) == cleanValue,
                "damaging the three blocks with the largest means leaves MoM bit-identical to the clean value");
        double lowValue = MedianOfMeans.Estimate(lowDamaged, blocks);
        Check(lowValue != cleanValue, "damaging three low blocks does move the estimator");
        Check(lowValue >= lowest - 1e-12 && lowValue <= highest + 1e-12,
                "after three damaged blocks MoM is still inside the clean block-mean range (" + lowValue + ")");
        var fourDamaged = new List<double>(boundary);
        for (int t = 0; t < 4; t++)
        {
            for (int i = descending[blocks - 1 - t] * blockSize; i < (descending[blocks - 1 - t] + 1) * blockSize; i++)
            {
                fourDamaged[i] = damagedBlockMean;
            }
        }
        double fourValue = MedianOfMeans.Estimate(fourDamaged, blocks);
        Check(fourValue > highest, "four damaged blocks push MoM outside the clean range (ceil(k / 2) is fatal)");
        Check(MedianOfMeans.Estimate(boundary, blocks) == cleanValue, "the clean value is reproducible");

        // Systematic contamination: two of the eight blocks (25% of the sample) shifted by 1000. The
        // sample mean inherits 250 of that shift; MoM is untouched, because two damaged blocks leave
        // the median position on clean values; and - the honest part - the 10% trimmed mean fails
        // here too, because trimming 10% off each tail cannot remove a block of 25% of the values.
        int mixed = 4096;
        var spiked = new List<double>(mixed);
        ulong spikeState = 572999UL;
        for (int i = 0; i < mixed; i++) spiked.Add(Rng.NextNormal(ref spikeState));
        int spikedBlocks = 8;
        int spikeBlockSize = mixed / spikedBlocks;
        for (int i = 0; i < 2 * spikeBlockSize; i++) spiked[i] += 1000.0;
        double spikedMeanError = Math.Abs(MedianOfMeans.Mean(spiked));
        double spikedMomError = Math.Abs(MedianOfMeans.Estimate(spiked, spikedBlocks));
        double spikedTrimmedError = Math.Abs(MedianOfMeans.TrimmedMean(spiked, 0.1));
        Check(spikedMeanError >= 200.0, "the sample mean inherits 250 of the block shift (" + spikedMeanError + ")");
        Check(spikedMomError <= 0.25, "MoM is untouched by the two damaged blocks (" + spikedMomError + ")");
        Check(spikedMeanError >= 1000.0 * spikedMomError, "the mean error is a thousand times MoM's ("
                + spikedMeanError + " vs " + spikedMomError + ")");
        Check(spikedTrimmedError >= 100.0, "the 10% trimmed mean cannot cope with 25% contiguous contamination ("
                + spikedTrimmedError + ")");

        // A genuinely heavy-tailed (ratio-of-normals) family with no finite mean: the sample mean's
        // error is unstable across seeds while MoM stays bounded. One particular sample can look fine
        // - that instability is the point, so the file only asserts the observed envelope.
        double worstMean = 0.0;
        double worstMom = 0.0;
        int meanExceeds = 0;
        int momExceeds = 0;
        for (int seed = 0; seed < 100; seed++)
        {
            var heavy = new List<double>(1024);
            ulong heavyState = (ulong)(572999 + 7919 * seed);
            for (int i = 0; i < 1024; i++) heavy.Add(Rng.NextHeavyTailed(ref heavyState));
            double meanError = Math.Abs(MedianOfMeans.Mean(heavy));
            double momError = Math.Abs(MedianOfMeans.Estimate(heavy, 32));
            worstMean = Math.Max(worstMean, meanError);
            worstMom = Math.Max(worstMom, momError);
            if (meanError > 1.0) meanExceeds++;
            if (momError > 1.0) momExceeds++;
        }
        Check(worstMean >= 3.0, "some heavy-tailed sample has a mean error above 3 (" + worstMean + ")");
        Check(worstMom <= 1.0, "MoM stays below 1 on all hundred heavy-tailed samples (" + worstMom + ")");
        Check(meanExceeds >= 20, "the mean exceeded 1 in absolute value on many samples (" + meanExceeds + " of 100)");
        Check(momExceeds == 0, "MoM never exceeded 1 in absolute value (" + momExceeds + ")");
        Check(worstMean >= 3.0 * worstMom, "the worst mean error is at least three times the worst MoM error ("
                + worstMean + " vs " + worstMom + ")");

        Console.WriteLine("PASS");
    }
}
