// Task 521 - Implement reservoir-sampling-based stratified sampling. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Two layered samplers over a deterministic LCG (state in [0, 65537), three discarded warm-up steps
// because the raw recurrence (75 s + 74) mod 10 has only two residues, and rejection sampling for an
// exactly uniform integer in [0, bound) so no modulo bias creeps into the statistics).
//   * Reservoir (Algorithm R): capacity k; the first k items fill the slots, and item i (1-based,
//     i > k) replaces slot j = uniform(0 .. i-1) when j < k. This keeps a uniformly random k-subset
//     regardless of the stream length, and it is the same code path used for simple random sampling
//     without replacement when the whole population is fed in.
//   * Stratified sampling: one reservoir per stratum, driven by a single shared RNG stream, fed in
//     population index order, with a per-stratum quota. A quota larger than the stratum yields the
//     whole stratum; a zero quota consumes no randomness at all.
// Quotas come from a deterministic largest-remainder proportional allocation (ties resolved by
// stratum order, every quota capped by the stratum size).
//
// Documented checks. (1) Sampling validity: each stratum returns exactly min(quota, size) distinct
// members of that stratum, arrays sorted for canonical comparison, total = sum of quotas.
// (2) Uniformity of Algorithm R: with n = 1000, k = 8 and 2000 seeds every item is selected at least
// once, the chi-square statistic of the selection counts against the exact expectation kT/n = 16
// (df = 999, sigma ~ 44.7) must land in [800, 1200], no count exceeds 45, and the counts of the first
// and the second half of the stream agree to within 2% (position independence).
// (3) Determinism: equal seeds give identical samples, a single-stratum stratified run reproduces the
// plain reservoir bit for bit, and distinct seeds change the sample. (4) Boundaries: k = 0 gives an
// empty sample, k >= n gives the whole population, k = n - 1 leaves exactly one item out and every
// item is left out for at least one of 200 seeds. (5) Allocation: the largest-remainder quotas sum to
// the requested budget, never exceed the stratum sizes and stay within one of the exact proportional
// share (checked for several budgets, including 0 and a budget above the population).
// (6) Statistics: for a three-stratum population with means 0/10/100 and within-stratum variance 2/3
// the stratified estimator of the population mean is unbiased (300 replications, tolerance 4 sample
// standard errors) and its empirical variance is at least ten times smaller than that of a simple
// random sample of the same size, while both estimators concentrate on the true population mean.
using System;
using System.Collections.Generic;

/** Deterministic LCG; Next(bound) rejects to give an exactly uniform value in [0, bound). */
internal sealed class Lcg
{
    private int state;

    internal Lcg(int seed)
    {
        state = seed % 65537;
        if (state < 0) state += 65537;
        // Three discarded steps, documented: the raw recurrence satisfies
        // (75 s + 74) mod 10 == (5 s + 4) mod 10, so the very first draw from a small seed can only
        // take two residues modulo a small bound. The warm-up removes that correlation.
        for (int i = 0; i < 3; i++) state = (state * 75 + 74) % 65537;
    }

    internal int Next(int bound)
    {
        if (bound <= 0 || bound > 65537)
        {
            throw new ArgumentOutOfRangeException(nameof(bound));
        }
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
}

/** Algorithm R reservoir: after feeding the whole stream the slots hold a uniform random k-subset. */
internal sealed class Reservoir
{
    private readonly int[] slots;
    private int filled;

    internal Reservoir(int capacity)
    {
        if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        slots = new int[capacity];
    }

    internal int Capacity => slots.Length;
    internal int Seen { get; private set; }

    internal void Offer(int item, Lcg rng)
    {
        Seen++;
        if (filled < slots.Length)
        {
            slots[filled++] = item;
            return;
        }
        if (slots.Length == 0) return;
        int j = rng.Next(Seen);
        if (j < slots.Length) slots[j] = item;
    }

    internal int[] Sorted()
    {
        var copy = new int[filled];
        Array.Copy(slots, copy, filled);
        Array.Sort(copy);
        return copy;
    }
}

internal static class Stratified
{
    internal static Dictionary<int, int> Positions(int[] stratumIds)
    {
        var positions = new Dictionary<int, int>();
        for (int s = 0; s < stratumIds.Length; s++)
        {
            if (positions.ContainsKey(stratumIds[s]))
            {
                throw new ArgumentException("duplicate stratum id " + stratumIds[s]);
            }
            positions[stratumIds[s]] = s;
        }
        return positions;
    }

    internal static int[] Sizes(int[] stratumOf, int[] stratumIds)
    {
        Dictionary<int, int> positions = Positions(stratumIds);
        var sizes = new int[stratumIds.Length];
        foreach (int id in stratumOf)
        {
            if (!positions.TryGetValue(id, out int s))
            {
                throw new ArgumentException("unknown stratum " + id);
            }
            sizes[s]++;
        }
        return sizes;
    }

    /** One shared RNG stream, one reservoir per stratum, population order; sorted canonical output. */
    internal static int[][] Sample(int[] stratumOf, int[] stratumIds, int[] quotas, int seed)
    {
        if (stratumOf.Length == 0) throw new ArgumentException("empty population");
        if (quotas.Length != stratumIds.Length)
        {
            throw new ArgumentException("one quota per stratum is required");
        }
        Dictionary<int, int> positions = Positions(stratumIds);
        var reservoirs = new Reservoir[stratumIds.Length];
        for (int s = 0; s < stratumIds.Length; s++)
        {
            if (quotas[s] < 0) throw new ArgumentException("negative quota for stratum " + stratumIds[s]);
            reservoirs[s] = new Reservoir(quotas[s]);
        }
        var rng = new Lcg(seed);
        for (int i = 0; i < stratumOf.Length; i++)
        {
            if (!positions.TryGetValue(stratumOf[i], out int s))
            {
                throw new ArgumentException("unknown stratum " + stratumOf[i]);
            }
            if (reservoirs[s].Capacity == 0) continue;
            reservoirs[s].Offer(i, rng);
        }
        var result = new int[stratumIds.Length][];
        for (int s = 0; s < stratumIds.Length; s++) result[s] = reservoirs[s].Sorted();
        return result;
    }

    /** Deterministic largest-remainder proportional allocation, capped by the stratum sizes. */
    internal static int[] ProportionalQuotas(int[] stratumSizes, int total)
    {
        int population = 0;
        foreach (int size in stratumSizes) population += size;
        if (population == 0) throw new ArgumentException("empty population");
        if (total < 0) throw new ArgumentException("negative budget");
        total = Math.Min(total, population);
        var quotas = new int[stratumSizes.Length];
        var remainders = new double[stratumSizes.Length];
        int assigned = 0;
        for (int s = 0; s < stratumSizes.Length; s++)
        {
            if (stratumSizes[s] < 0) throw new ArgumentException("negative stratum size");
            double exact = (double)stratumSizes[s] * total / population;
            quotas[s] = Math.Min((int)Math.Floor(exact), stratumSizes[s]);
            remainders[s] = exact - Math.Floor(exact);
            assigned += quotas[s];
        }
        while (assigned < total)
        {
            int best = -1;
            for (int s = 0; s < quotas.Length; s++)
            {
                if (quotas[s] >= stratumSizes[s]) continue;
                if (best < 0 || remainders[s] > remainders[best]) best = s;
            }
            if (best < 0) break;
            quotas[best]++;
            remainders[best] = -1.0;
            assigned++;
        }
        return quotas;
    }

    /** The stratified expansion estimator N_h/N-weighted mean of the per-stratum sample means. */
    internal static double EstimateMean(int[] values, int[][] samples, int[] sizes)
    {
        double estimate = 0.0;
        for (int s = 0; s < samples.Length; s++)
        {
            if (samples[s].Length == 0) continue;
            double sum = 0.0;
            foreach (int item in samples[s]) sum += values[item];
            estimate += (double)sizes[s] / values.Length * (sum / samples[s].Length);
        }
        return estimate;
    }

    /** Simple random sample without replacement of the requested size, via the same reservoir. */
    internal static int[] SimpleRandomSample(int population, int size, int seed)
    {
        var reservoir = new Reservoir(size);
        var rng = new Lcg(seed);
        for (int i = 0; i < population; i++) reservoir.Offer(i, rng);
        return reservoir.Sorted();
    }

    internal static double SampleMean(int[] values, int[] sample)
    {
        double sum = 0.0;
        foreach (int item in sample) sum += values[item];
        return sum / sample.Length;
    }
}

public static class Solution
{
    public static void Main()
    {
        // (4) boundaries of the reservoir itself
        var small = new Reservoir(3);
        var rng = new Lcg(7);
        for (int i = 0; i < 10; i++) small.Offer(i, rng);
        int[] smallSample = small.Sorted();
        Check(smallSample.Length == 3, "reservoir of capacity 3 keeps 3 items");
        Check(small.Seen == 10, "every offered item is counted");
        Check(Distinct(smallSample), "the reservoir never repeats an item");
        Check(WithinRange(smallSample, 0, 10), "the reservoir only returns offered items");

        var everything = new Reservoir(10);
        var rng2 = new Lcg(7);
        for (int i = 0; i < 10; i++) everything.Offer(i, rng2);
        int[] full = everything.Sorted();
        Check(full.Length == 10 && full[0] == 0 && full[9] == 9, "capacity n returns the whole stream");

        var oversized = new Reservoir(15);
        var rng3 = new Lcg(7);
        for (int i = 0; i < 10; i++) oversized.Offer(i, rng3);
        Check(oversized.Sorted().Length == 10, "capacity above the stream length returns the stream");

        var empty = new Reservoir(0);
        var rng4 = new Lcg(7);
        for (int i = 0; i < 10; i++) empty.Offer(i, rng4);
        Check(empty.Sorted().Length == 0, "capacity 0 returns nothing");

        var missing = new Reservoir(9);
        var rng5 = new Lcg(11);
        for (int i = 0; i < 10; i++) missing.Offer(i, rng5);
        Check(missing.Sorted().Length == 9, "capacity n - 1 keeps nine items");

        var omitted = new bool[10];
        for (int seed = 1; seed <= 200; seed++)
        {
            var reservoir = new Reservoir(9);
            var local = new Lcg(seed);
            for (int i = 0; i < 10; i++) reservoir.Offer(i, local);
            int[] sample = reservoir.Sorted();
            Check(Distinct(sample) && sample.Length == 9, "capacity n - 1 sample is a 9-subset");
            for (int i = 0; i < 10; i++)
            {
                if (Array.IndexOf(sample, i) < 0) omitted[i] = true;
            }
        }
        foreach (bool seen in omitted) Check(seen, "every item is left out by at least one seed");

        // (2) uniformity of Algorithm R over 2000 replications
        const int population = 1000;
        const int capacity = 8;
        const int trials = 2000;
        var counts = new int[population];
        for (int seed = 1; seed <= trials; seed++)
        {
            var reservoir = new Reservoir(capacity);
            var local = new Lcg(seed);
            for (int i = 0; i < population; i++) reservoir.Offer(i, local);
            int[] sample = reservoir.Sorted();
            Check(sample.Length == capacity && Distinct(sample), "every replicate yields a k-subset");
            foreach (int item in sample) counts[item]++;
        }
        double expected = (double)capacity * trials / population;
        double chiSquare = 0.0;
        int minCount = int.MaxValue;
        int maxCount = int.MinValue;
        int firstHalf = 0;
        int secondHalf = 0;
        for (int i = 0; i < population; i++)
        {
            double difference = counts[i] - expected;
            chiSquare += difference * difference / expected;
            minCount = Math.Min(minCount, counts[i]);
            maxCount = Math.Max(maxCount, counts[i]);
            if (i < population / 2) firstHalf += counts[i];
            else secondHalf += counts[i];
        }
        Check(minCount >= 1, "every item is selected at least once in 2000 trials");
        Check(maxCount <= 45, "no item is selected suspiciously often (max " + maxCount + ")");
        Check(chiSquare >= 800.0 && chiSquare <= 1200.0,
                "chi-square " + chiSquare + " is consistent with a uniform k-subset (df 999)");
        double halfSpread = Math.Abs(firstHalf - secondHalf) / (double)(firstHalf + secondHalf);
        Check(halfSpread <= 0.02, "selection is independent of the stream position (" + halfSpread + ")");
        Check(Math.Abs(firstHalf + secondHalf - capacity * trials) <= 1e-9, "counts add up to k*T");

        // (3) determinism and seed sensitivity
        int[] first = Stratified.SimpleRandomSample(200, 12, 4242);
        int[] second = Stratified.SimpleRandomSample(200, 12, 4242);
        CheckSame(first, second, "the same seed reproduces the sample");
        int[] other = Stratified.SimpleRandomSample(200, 12, 4243);
        Check(!Same(first, other), "a different seed changes the sample");

        // (1) stratified sampling validity on a three-stratum population
        int[] sizes = { 600, 300, 100 };
        int[] stratumIds = { 0, 1, 2 };
        int[] means = { 0, 10, 100 };
        int[] values = BuildPopulation(sizes, means, 2024);
        int[] stratumOf = BuildStratumOf(sizes);
        Check(Stratified.Sizes(stratumOf, stratumIds)[0] == 600, "stratum 0 size");
        Check(Stratified.Sizes(stratumOf, stratumIds)[1] == 300, "stratum 1 size");
        Check(Stratified.Sizes(stratumOf, stratumIds)[2] == 100, "stratum 2 size");

        int[] quotas = Stratified.ProportionalQuotas(sizes, 60);
        Check(quotas[0] == 36 && quotas[1] == 18 && quotas[2] == 6, "exact proportional quotas");
        int[][] stratifiedSample = Stratified.Sample(stratumOf, stratumIds, quotas, 4242);
        for (int s = 0; s < stratumIds.Length; s++)
        {
            Check(stratifiedSample[s].Length == quotas[s], "stratum " + s + " returns exactly its quota");
            Check(Distinct(stratifiedSample[s]), "stratum " + s + " has no duplicates");
            Check(WithinRange(stratifiedSample[s], 0, values.Length), "stratum " + s + " stays in range");
            foreach (int item in stratifiedSample[s])
            {
                Check(stratumOf[item] == stratumIds[s], "stratum " + s + " only returns its own members");
            }
        }
        CheckTotal(stratifiedSample, quotas);

        int[][] repeated = Stratified.Sample(stratumOf, stratumIds, quotas, 4242);
        for (int s = 0; s < stratumIds.Length; s++)
        {
            CheckSame(stratifiedSample[s], repeated[s], "stratified sampling is deterministic, stratum " + s);
        }
        int[][] otherStratified = Stratified.Sample(stratumOf, stratumIds, quotas, 777);
        Check(!Same(stratifiedSample[0], otherStratified[0]), "a different seed changes a stratum");

        int[][] generous = Stratified.Sample(stratumOf, stratumIds, new[] { 700, 400, 200 }, 99);
        Check(generous[0].Length == 600 && generous[1].Length == 300 && generous[2].Length == 100,
                "quotas above the stratum size return the whole stratum");
        int[][] zeroQuota = Stratified.Sample(stratumOf, stratumIds, new[] { 0, 0, 0 }, 99);
        Check(zeroQuota[0].Length == 0 && zeroQuota[1].Length == 0 && zeroQuota[2].Length == 0,
                "a zero quota returns nothing");

        int[] single = new int[values.Length];
        int[][] singleStratum = Stratified.Sample(single, new[] { 0 }, new[] { 25 }, 31337);
        int[] plainReservoir = Stratified.SimpleRandomSample(values.Length, 25, 31337);
        CheckSame(singleStratum[0], plainReservoir, "one stratum with quota k equals a plain reservoir");
        Check(singleStratum[0].Length == 25, "one stratum returns the quota");

        CheckThrows(() => Stratified.Sample(new int[0], stratumIds, quotas, 1), "empty population throws");
        CheckThrows(() => Stratified.Sample(stratumOf, stratumIds, new[] { 1, 2 }, 1), "quota count throws");
        CheckThrows(() => Stratified.Sample(stratumOf, stratumIds, new[] { -1, 2, 3 }, 1), "negative quota throws");
        CheckThrows(() => Stratified.Sample(new[] { 0, 7 }, stratumIds, new[] { 1, 1, 1 }, 1),
                "unknown stratum throws");
        CheckThrows(() => new Reservoir(-1), "negative reservoir capacity throws");
        CheckThrows(() => new Lcg(1).Next(0), "LCG bound 0 throws");
        CheckThrows(() => Stratified.ProportionalQuotas(new[] { 0, 0 }, 1), "empty quotas throws");
        CheckThrows(() => Stratified.ProportionalQuotas(new[] { 5, 5 }, -1), "negative budget throws");

        // (5) allocation properties
        int[][] budgets = { new[] { 0 }, new[] { 1 }, new[] { 37 }, new[] { 60 }, new[] { 999 },
            new[] { 1000 }, new[] { 5000 } };
        foreach (int[] budget in budgets)
        {
            int[] allocation = Stratified.ProportionalQuotas(sizes, budget[0]);
            int sum = 0;
            for (int s = 0; s < sizes.Length; s++)
            {
                Check(allocation[s] <= sizes[s], "quota capped by the stratum size at budget " + budget[0]);
                double exact = (double)sizes[s] * Math.Min(budget[0], 1000) / 1000;
                Check(Math.Abs(allocation[s] - exact) < 1.0,
                        "quota stays within one of the exact share at budget " + budget[0]);
                sum += allocation[s];
            }
            Check(sum == Math.Min(budget[0], 1000), "quotas sum to the budget " + budget[0]);
        }
        int[] equalSizes = Stratified.ProportionalQuotas(new[] { 10, 10, 10 }, 4);
        Check(equalSizes[0] + equalSizes[1] + equalSizes[2] == 4, "tie-broken remainders still sum");

        // (6) stratified versus simple random sampling of the same size
        const int replications = 300;
        double trueMean = PopulationMean(values);
        double stratifiedSum = 0.0;
        double stratifiedSquareSum = 0.0;
        double simpleSum = 0.0;
        double simpleSquareSum = 0.0;
        for (int seed = 1; seed <= replications; seed++)
        {
            int[][] sample = Stratified.Sample(stratumOf, stratumIds, quotas, seed);
            double stratifiedEstimate = Stratified.EstimateMean(values, sample, sizes);
            stratifiedSum += stratifiedEstimate;
            stratifiedSquareSum += stratifiedEstimate * stratifiedEstimate;
            int[] simpleSample = Stratified.SimpleRandomSample(values.Length, 60, seed);
            double simpleEstimate = Stratified.SampleMean(values, simpleSample);
            Check(Distinct(simpleSample) && simpleSample.Length == 60, "the simple random sample is a 60-subset");
            simpleSum += simpleEstimate;
            simpleSquareSum += simpleEstimate * simpleEstimate;
        }
        double stratifiedMean = stratifiedSum / replications;
        double simpleMean = simpleSum / replications;
        double stratifiedVariance = Variance(stratifiedSquareSum / replications, stratifiedMean, replications);
        double simpleVariance = Variance(simpleSquareSum / replications, simpleMean, replications);
        double stratifiedSe = Math.Sqrt(stratifiedVariance / replications);
        double simpleSe = Math.Sqrt(simpleVariance / replications);
        double expectedMean = 0.0;
        for (int s = 0; s < sizes.Length; s++)
        {
            double stratumSum = 0.0;
            int stratumCount = 0;
            for (int i = 0; i < values.Length; i++)
            {
                if (stratumOf[i] == s)
                {
                    stratumSum += values[i];
                    stratumCount++;
                }
            }
            expectedMean += (double)stratumCount / values.Length * (stratumSum / stratumCount);
        }
        CheckClose(trueMean, expectedMean, "the population mean equals the N_h/N-weighted stratum means");
        Check(Math.Abs(trueMean - 13.0) <= 0.1, "the constructed population has mean near 13");
        Check(Math.Abs(stratifiedMean - trueMean) <= 4.0 * stratifiedSe,
                "the stratified estimator is unbiased (" + stratifiedMean + " vs " + trueMean + ")");
        Check(Math.Abs(simpleMean - trueMean) <= 4.0 * simpleSe,
                "the simple random estimator is unbiased (" + simpleMean + " vs " + trueMean + ")");
        Check(stratifiedVariance * 10.0 <= simpleVariance,
                "stratification cuts the variance by more than 10x (" + stratifiedVariance + " vs "
                        + simpleVariance + ")");
        Check(stratifiedSe < simpleSe, "stratification cuts the standard error");

        Console.WriteLine("PASS");
    }

    private static int[] BuildStratumOf(int[] sizes)
    {
        int total = 0;
        foreach (int size in sizes) total += size;
        var stratumOf = new int[total];
        int index = 0;
        for (int s = 0; s < sizes.Length; s++)
        {
            for (int i = 0; i < sizes[s]; i++) stratumOf[index++] = s;
        }
        return stratumOf;
    }

    private static int[] BuildPopulation(int[] sizes, int[] means, int seed)
    {
        int total = 0;
        foreach (int size in sizes) total += size;
        var values = new int[total];
        int index = 0;
        for (int s = 0; s < sizes.Length; s++)
        {
            for (int i = 0; i < sizes[s]; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                values[index++] = means[s] + seed % 3 - 1;
            }
        }
        return values;
    }

    private static double PopulationMean(int[] values)
    {
        double sum = 0.0;
        foreach (int value in values) sum += value;
        return sum / values.Length;
    }

    private static double Variance(double meanOfSquares, double mean, int count)
    {
        return (meanOfSquares - mean * mean) * count / (count - 1.0);
    }

    private static bool Distinct(int[] items)
    {
        var seen = new HashSet<int>();
        foreach (int item in items)
        {
            if (!seen.Add(item)) return false;
        }
        return true;
    }

    private static bool WithinRange(int[] items, int lo, int hi)
    {
        foreach (int item in items)
        {
            if (item < lo || item >= hi) return false;
        }
        return true;
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void CheckSame(int[] a, int[] b, string label)
    {
        Check(Same(a, b), label);
    }

    private static void CheckTotal(int[][] sample, int[] quotas)
    {
        int total = 0;
        for (int s = 0; s < sample.Length; s++) total += sample[s].Length;
        int expected = 0;
        foreach (int quota in quotas) expected += quota;
        Check(total == expected, "the sample holds exactly the sum of the quotas");
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
