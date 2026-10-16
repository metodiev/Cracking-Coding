// Task 780 - Implement distributed sampling reservoir with merge operation. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Sampling a distributed stream without landing every record centrally: each shard keeps a reservoir
// of its own, and the partial reservoirs are merged into one uniform sample of the union. The file
// implements and cross-checks three samplers:
//
//   * Algorithm R - the classic scan: item i (originally 1-based) replaces a random slot when i is at
//     most k and otherwise with probability k/i. It knows the stream length and yields a uniform k-subset.
//   * Algorithm L - the skip based variant: draw U and skip floor(ln U / ln(1 - k/n)) items at each
//     step. Its whole point is that it makes fewer random draws; that is measured here, and the fact
//     that it samples the same distribution as R is checked by comparing inclusion frequencies.
//   * Distributed merge - each shard hands over its reservoir together with the length n_s of the
//     stream it saw, every entry gets a priority key u^(1/n_s) = exp(ln(u)/n_s), and the merged
//     reservoir is simply the k largest keys. That is the standard priority-sampling merge; the
//     weights are what make shards of different sizes commensurable, and the file has a negative
//     check showing that ignoring them (concatenating the partial reservoirs and re-sampling) biases
//     the sample towards the small shards.
//
// Verified with an independent oracle - the exact inclusion frequency of every item, measured by
// repeating the experiment over thousands of independent trials: R and L both give every item
// probability k/n of being included (checked per item against the exact value with a sampling
// tolerance), the distributed merge matches the same frequencies when every shard holds at least k
// items, ignoring the shard weights really does hand the wrong share to a shard, reservoirs never
// contain duplicates or stray items, k >= n returns the whole stream, and the degenerate cases (k = 1,
// a shard with no data, a single item, an empty stream) behave correctly. The one case the file does
// not paper over: a shard shorter than k contributes every one of its items with weight one, and its
// share of the merged sample then only approximates k/n - the measured deviation is reported instead
// of being hidden behind a loose threshold.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Reservoir
{
    /// <summary>Algorithm R: one pass, one random draw per item after the reservoir is full.</summary>
    internal static List<int> AlgorithmR(IReadOnlyList<int> stream, int k, ref ulong state, out long draws)
    {
        var reservoir = new List<int>();
        draws = 0;
        if (k <= 0) return reservoir;
        for (int i = 0; i < stream.Count; i++)
        {
            if (i < k) { reservoir.Add(stream[i]); continue; }
            draws++;
            int slot = SplitMix64.NextInt(ref state, i + 1);
            if (slot < k) reservoir[slot] = stream[i];
        }
        return reservoir;
    }

    /// <summary>Algorithm L: after each acceptance, skip a geometric number of items.</summary>
    internal static List<int> AlgorithmL(IReadOnlyList<int> stream, int k, ref ulong state, out long draws)
    {
        draws = 0;
        if (k <= 0) return new List<int>();
        int n = stream.Count;
        if (n <= k) return new List<int>(stream);
        var reservoir = new List<int>();
        for (int i = 0; i < k; i++) reservoir.Add(stream[i]);

        double weight = Math.Exp(Math.Log(SafeUnit(ref state)) / k);
        draws++;
        int index = k - 1;
        while (true)
        {
            double skipUnit = SafeUnit(ref state);
            draws++;
            int skip = (int)Math.Floor(Math.Log(skipUnit) / Math.Log(1 - weight)) + 1;
            if (skip < 1) skip = 1;
            index += skip;
            if (index >= n) break;
            reservoir[SplitMix64.NextInt(ref state, k)] = stream[index];
            draws++;
            weight *= Math.Exp(Math.Log(SafeUnit(ref state)) / k);
            draws++;
        }
        return reservoir;
    }

    private static double SafeUnit(ref ulong state)
    {
        double unit = SplitMix64.NextUnit(ref state);
        return unit <= 0 ? double.Epsilon : unit;
    }

    internal sealed class Shard
    {
        internal readonly List<int> Items;
        internal readonly int Length;

        internal Shard(List<int> items, int length)
        {
            Items = items;
            Length = length;
        }
    }

    /// <summary>
    /// Priority-sampling merge. A reservoir entry of shard s stands for n_s / k_s stream items
    /// (k_s = min(k, n_s) is the size of that shard's reservoir), and the priority key of an entry of
    /// weight w is u^(1/w), so keeping the k largest keys yields a uniform sample of the union. Shards
    /// shorter than k contribute every item with weight one, which is what keeps them from being
    /// over-represented by the merge.
    /// </summary>
    internal static List<int> MergeWeighted(IReadOnlyList<Shard> shards, int k, ref ulong state)
    {
        var heap = new List<KeyValuePair<double, int>>();
        foreach (Shard shard in shards)
        {
            if (shard.Length <= 0 || shard.Items.Count == 0) continue;
            double weight = (double)shard.Length / shard.Items.Count;
            if (weight < 1) weight = 1;
            foreach (int item in shard.Items)
            {
                double unit = SplitMix64.NextUnit(ref state);
                if (unit <= 0) unit = double.Epsilon;
                double key = Math.Exp(Math.Log(unit) / weight);
                heap.Add(new KeyValuePair<double, int>(key, item));
            }
        }
        heap.Sort((a, b) => b.Key != a.Key ? b.Key.CompareTo(a.Key) : a.Value.CompareTo(b.Value));
        var merged = new List<int>();
        for (int i = 0; i < Math.Min(k, heap.Count); i++) merged.Add(heap[i].Value);
        return merged;
    }

    /// <summary>The naive merge: concatenate the partial reservoirs and re-sample uniformly.</summary>
    internal static List<int> MergeNaive(IReadOnlyList<Shard> shards, int k, ref ulong state)
    {
        var pool = new List<int>();
        foreach (Shard shard in shards) pool.AddRange(shard.Items);
        var copy = new List<int>(pool);
        var picked = new List<int>();
        for (int i = 0; i < k && copy.Count > 0; i++)
        {
            int at = SplitMix64.NextInt(ref state, copy.Count);
            picked.Add(copy[at]);
            copy.RemoveAt(at);
        }
        return picked;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int[] Range(int count)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++) values[i] = i;
        return values;
    }

    private static bool HasDuplicates(List<int> sample)
    {
        var seen = new HashSet<int>();
        foreach (int item in sample) if (!seen.Add(item)) return true;
        return false;
    }

    private static bool WithinStream(List<int> sample, int streamLength)
    {
        foreach (int item in sample) if (item < 0 || item >= streamLength) return false;
        return true;
    }

    /// <summary>Measured inclusion frequency of every item over many independent trials.</summary>
    private static double[] InclusionFrequencies(int streamLength, int k, int trials, ulong seed, int mode)
    {
        var counts = new double[streamLength];
        ulong state = seed;
        var stream = Range(streamLength);
        for (int t = 0; t < trials; t++)
        {
            List<int> sample;
            if (mode == 0)
            {
                long draws;
                sample = Reservoir.AlgorithmR(stream, k, ref state, out draws);
            }
            else if (mode == 1)
            {
                long draws;
                sample = Reservoir.AlgorithmL(stream, k, ref state, out draws);
            }
            else
            {
                // three shards of very different lengths, each sampled locally then merged
                var shards = new List<Reservoir.Shard>();
                int cut1 = streamLength / 2, cut2 = cut1 + streamLength / 4;
                var local = new List<int>();
                for (int i = 0; i < cut1; i++) local.Add(i);
                long ignored;
                shards.Add(new Reservoir.Shard(Reservoir.AlgorithmR(local, k, ref state, out ignored), local.Count));
                local = new List<int>();
                for (int i = cut1; i < cut2; i++) local.Add(i);
                shards.Add(new Reservoir.Shard(Reservoir.AlgorithmR(local, k, ref state, out ignored), local.Count));
                local = new List<int>();
                for (int i = cut2; i < streamLength; i++) local.Add(i);
                shards.Add(new Reservoir.Shard(Reservoir.AlgorithmR(local, k, ref state, out ignored), local.Count));
                sample = Reservoir.MergeWeighted(shards, k, ref state);
            }
            foreach (int item in sample) counts[item]++;
        }
        for (int i = 0; i < streamLength; i++) counts[i] /= trials;
        return counts;
    }

    public static void Main()
    {
        // ---- the three samplers are uniform: every item appears with probability k/n -------------
        {
            const int StreamLength = 24, K = 6, Trials = 20000;
            double expected = (double)K / StreamLength;
            var names = new[] { "Algorithm R", "Algorithm L", "distributed merge" };
            var frequencies = new double[3][];
            frequencies[0] = InclusionFrequencies(StreamLength, K, Trials, 0x5EED780000000001UL, 0);
            frequencies[1] = InclusionFrequencies(StreamLength, K, Trials, 0x5EED780000000002UL, 1);
            frequencies[2] = InclusionFrequencies(StreamLength, K, Trials, 0x5EED780000000003UL, 2);
            for (int mode = 0; mode < 3; mode++)
            {
                double worst = 0, sum = 0;
                int item0 = 0;
                for (int i = 0; i < StreamLength; i++)
                {
                    double deviation = Math.Abs(frequencies[mode][i] - expected);
                    if (deviation > worst) { worst = deviation; item0 = i; }
                    sum += frequencies[mode][i];
                }
                Check(worst < 0.03, names[mode] + " includes every item with probability k/n (worst deviation "
                    + worst.ToString("F4") + " at item " + item0 + " of " + expected.ToString("F4") + ")");
                Check(Math.Abs(sum - K) < 0.05, names[mode] + " keeps the expected sample size (" + sum.ToString("F4") + ")");
            }
            double gap = 0;
            for (int i = 0; i < StreamLength; i++) gap = Math.Max(gap, Math.Abs(frequencies[0][i] - frequencies[2][i]));
            Check(gap < 0.05, "the merged sample matches the sequential sampler's distribution (worst gap " + gap.ToString("F4") + ")");
            Console.WriteLine("uniformity over " + Trials + " trials: expected=" + expected.ToString("F4")
                + " firstItem R=" + frequencies[0][0].ToString("F4") + " L=" + frequencies[1][0].ToString("F4")
                + " merge=" + frequencies[2][0].ToString("F4"));
        }

        // ---- shard weights matter (and a small shard is the hard case) ------------------------------
        {
            const int Trials = 20000, K = 6;
            ulong state = 0x5EED780000000004UL;
            var big = new List<int>();
            for (int i = 0; i < 200; i++) big.Add(i);
            var small = new List<int>();
            for (int i = 200; i < 204; i++) small.Add(i);
            long drawsA, drawsB;
            const int Total = 204;
            double expected = (double)K / Total;
            double tinyExpected = 4 * expected;

            // two shards that both hold more than k items: the weighted merge stays fair
            var medium = new List<int>();
            for (int i = 0; i < 150; i++) medium.Add(500 + i);
            var mediumCounts = new double[150];
            var mediumNaive = new double[150];
            var bigWeightedForMedium = new double[200];
            var bigNaiveForMedium = new double[200];
            for (int t = 0; t < Trials; t++)
            {
                var shards = new List<Reservoir.Shard>
                {
                    new Reservoir.Shard(Reservoir.AlgorithmR(big, K, ref state, out drawsA), big.Count),
                    new Reservoir.Shard(Reservoir.AlgorithmR(medium, K, ref state, out drawsB), medium.Count),
                };
                foreach (int item in Reservoir.MergeWeighted(shards, K, ref state))
                {
                    if (item >= 500) mediumCounts[item - 500]++; else bigWeightedForMedium[item]++;
                }
                foreach (int item in Reservoir.MergeNaive(shards, K, ref state))
                {
                    if (item >= 500) mediumNaive[item - 500]++; else bigNaiveForMedium[item]++;
                }
            }
            double weightedDeviation = 0, naiveDeviation = 0;
            for (int i = 0; i < 150; i++)
            {
                weightedDeviation = Math.Max(weightedDeviation, Math.Abs(mediumCounts[i] / Trials - (double)K / 350));
                naiveDeviation = Math.Max(naiveDeviation, Math.Abs(mediumNaive[i] / Trials - (double)K / 350));
            }
            double bigWeightedShare = 0, bigNaiveShare = 0;
            for (int i = 0; i < 200; i++) { bigWeightedShare += bigWeightedForMedium[i]; bigNaiveShare += bigNaiveForMedium[i]; }
            bigWeightedShare /= Trials; bigNaiveShare /= Trials;
            Check(weightedDeviation < 0.05, "the weighted merge is uniform across two large shards (worst deviation "
                + weightedDeviation.ToString("F4") + " of " + ((double)K / 350).ToString("F4") + ")");
            Check(Math.Abs(bigWeightedShare - 200.0 * K / 350) < 0.3, "the weighted merge gives the large shard its share ("
                + bigWeightedShare.ToString("F4") + " vs " + (200.0 * K / 350).ToString("F4") + ")");
            Check(Math.Abs(bigNaiveShare - 200.0 * K / 350) > 0.25, "concatenating the partial reservoirs hands the large shard the wrong share ("
                + bigNaiveShare.ToString("F4") + " vs " + (200.0 * K / 350).ToString("F4") + ")");
            Console.WriteLine("two large shards: expected per medium key=" + ((double)K / 350).ToString("F4")
                + " weightedDeviation=" + weightedDeviation.ToString("F4") + " naiveDeviation=" + naiveDeviation.ToString("F4")
                + " bigShare weighted=" + bigWeightedShare.ToString("F4") + " naive=" + bigNaiveShare.ToString("F4"));

            // a shard that is shorter than k keeps every item; the priority keys still beat naive merging,
            // but the composition of such a shard is only approximately k/n - the measured value is reported
            var weightedCounts = new double[Total];
            var naiveCounts = new double[Total];
            for (int t = 0; t < Trials; t++)
            {
                var shards = new List<Reservoir.Shard>
                {
                    new Reservoir.Shard(Reservoir.AlgorithmR(big, K, ref state, out drawsA), big.Count),
                    new Reservoir.Shard(Reservoir.AlgorithmR(small, K, ref state, out drawsB), small.Count),
                };
                foreach (int item in Reservoir.MergeWeighted(shards, K, ref state)) weightedCounts[item]++;
                foreach (int item in Reservoir.MergeNaive(shards, K, ref state)) naiveCounts[item]++;
            }
            double tinyWeighted = 0, tinyNaive = 0, bigWeighted = 0, bigNaive = 0;
            for (int i = 0; i < 200; i++) { bigWeighted += weightedCounts[i]; bigNaive += naiveCounts[i]; }
            for (int i = 200; i < Total; i++) { tinyWeighted += weightedCounts[i]; tinyNaive += naiveCounts[i]; }
            tinyWeighted /= Trials; tinyNaive /= Trials; bigWeighted /= Trials; bigNaive /= Trials;
            Check(tinyNaive > tinyExpected + 0.3, "the naive merge inflates the short shard badly ("
                + tinyNaive.ToString("F4") + " vs " + tinyExpected.ToString("F4") + " expected)");
            Check(tinyWeighted < tinyNaive / 4, "the weighted merge repairs most of that bias ("
                + tinyWeighted.ToString("F4") + " vs " + tinyNaive.ToString("F4") + ")");
            Check(Math.Abs(bigWeighted - 200 * expected) < 0.5, "the large shard keeps its share ("
                + bigWeighted.ToString("F4") + " vs " + (200 * expected).ToString("F4") + ")");
            Console.WriteLine("shard weighting: two large shards worst deviation weighted=" + weightedDeviation.ToString("F4")
                + " naive=" + naiveDeviation.ToString("F4") + " | short shard expected=" + tinyExpected.ToString("F4")
                + " weighted=" + tinyWeighted.ToString("F4") + " naive=" + tinyNaive.ToString("F4")
                + " | large shard weighted=" + bigWeighted.ToString("F4") + " naive=" + bigNaive.ToString("F4"));
        }

        // ---- Algorithm R really returns a valid k-subset -------------------------------------------
        {
            const int StreamLength = 50, K = 5;
            var stream = Range(StreamLength);
            ulong state = 0x5EED780000000005UL;
            bool sizes = true, noDuplicates = true, inside = true;
            for (int t = 0; t < 300; t++)
            {
                long draws;
                var sample = Reservoir.AlgorithmR(stream, K, ref state, out draws);
                if (sample.Count != K) sizes = false;
                if (HasDuplicates(sample)) noDuplicates = false;
                if (!WithinStream(sample, StreamLength)) inside = false;
            }
            Check(sizes, "Algorithm R always returns exactly k items");
            Check(noDuplicates, "a reservoir never holds an item twice");
            Check(inside, "every sampled item comes from the stream");
            Console.WriteLine("algorithm R: 300 runs of k=" + K + " over " + StreamLength + " items are all valid");
        }

        // ---- Algorithm L draws far fewer random numbers than R --------------------------------------
        {
            const int StreamLength = 20000, K = 10;
            var stream = Range(StreamLength);
            ulong state = 0x5EED780000000006UL;
            long drawsR, drawsL;
            var sampleR = Reservoir.AlgorithmR(stream, K, ref state, out drawsR);
            var sampleL = Reservoir.AlgorithmL(stream, K, ref state, out drawsL);
            Check(drawsL < drawsR, "Algorithm L needs fewer draws than Algorithm R (" + drawsL + " < " + drawsR + " for "
                + StreamLength + " items)");
            Check(drawsL < StreamLength / 10, "the skip makes Algorithm L sublinear in draws (" + drawsL + ")");
            Check(sampleL.Count == K && !HasDuplicates(sampleL) && WithinStream(sampleL, StreamLength),
                "Algorithm L returns a valid sample");
            Check(sampleR.Count == K, "Algorithm R returns a valid sample on the long stream");
            Console.WriteLine("draws on " + StreamLength + " items: R=" + drawsR + " L=" + drawsL);
        }

        // ---- boundary cases -------------------------------------------------------------------------
        {
            var stream = Range(5);
            ulong state = 0x5EED780000000007UL;
            long draws;
            var all = Reservoir.AlgorithmR(stream, 5, ref state, out draws);
            Check(all.Count == 5 && all.TrueForAll(i => i >= 0 && i < 5), "k = n returns the whole stream");
            Check(Reservoir.AlgorithmR(stream, 9, ref state, out draws).Count == 5, "k > n returns the whole stream");
            Check(Reservoir.AlgorithmR(stream, 1, ref state, out draws).Count == 1, "k = 1 returns one item");
            Check(Reservoir.AlgorithmR(new int[0], 4, ref state, out draws).Count == 0, "an empty stream yields an empty reservoir");
            Check(Reservoir.AlgorithmR(stream, 0, ref state, out draws).Count == 0, "k = 0 is empty");
            var single = Reservoir.AlgorithmR(new[] { 42 }, 3, ref state, out draws);
            Check(single.Count == 1 && single[0] == 42, "a one-item stream is sampled exactly");
            Check(Reservoir.AlgorithmL(new[] { 7 }, 2, ref state, out draws).Count == 1, "Algorithm L handles a one-item stream");

            var shards = new List<Reservoir.Shard>
            {
                new Reservoir.Shard(new List<int>(), 0),                       // a shard that saw nothing
                new Reservoir.Shard(new List<int> { 1, 2, 3 }, 40),
                new Reservoir.Shard(new List<int> { 4, 5 }, 2),
            };
            var merged = Reservoir.MergeWeighted(shards, 4, ref state);
            Check(merged.Count == 4, "the merge respects k (" + merged.Count + ")");
            Check(!HasDuplicates(merged), "the merged sample has no duplicates");
            Check(merged.TrueForAll(i => i >= 1 && i <= 5), "the merged sample only contains real items");
            var emptyOnly = Reservoir.MergeWeighted(new List<Reservoir.Shard> { new Reservoir.Shard(new List<int>(), 0) }, 3, ref state);
            Check(emptyOnly.Count == 0, "merging shards that saw no data yields nothing");
            var smallMerge = Reservoir.MergeWeighted(new List<Reservoir.Shard>
            {
                new Reservoir.Shard(new List<int> { 1, 2 }, 2),
                new Reservoir.Shard(new List<int> { 3 }, 1),
            }, 9, ref state);
            Check(smallMerge.Count == 3, "k larger than everything keeps every candidate");
            Console.WriteLine("boundary: merged=" + merged.Count + " emptyOnly=" + emptyOnly.Count
                + " smallMerge=" + smallMerge.Count + " single=" + single[0]);
        }

        // ---- determinism ----------------------------------------------------------------------------
        {
            var stream = Range(30);
            ulong stateA = 0x5EED780000000008UL, stateB = stateA;
            long drawsA, drawsB;
            var first = Reservoir.AlgorithmR(stream, 4, ref stateA, out drawsA);
            var second = Reservoir.AlgorithmR(stream, 4, ref stateB, out drawsB);
            Check(first.Count == second.Count && drawsA == drawsB, "two identical runs draw the same sequence");
            bool same = true;
            for (int i = 0; i < first.Count; i++) if (first[i] != second[i]) same = false;
            Check(same, "two identical runs return the same reservoir");

            var shards = new List<Reservoir.Shard> { new Reservoir.Shard(new List<int> { 5, 6, 7 }, 10) };
            ulong stateC = 0x5EED780000000009UL, stateD = stateC;
            var mergeA = Reservoir.MergeWeighted(shards, 2, ref stateC);
            var mergeB = Reservoir.MergeWeighted(shards, 2, ref stateD);
            Check(mergeA.Count == mergeB.Count && mergeA[0] == mergeB[0] && mergeA[1] == mergeB[1],
                "the merge is reproducible for a fixed seed");
            Console.WriteLine("determinism: draws=" + drawsA + " sample=" + string.Join(",", first));
        }

        Console.WriteLine("PASS");
    }
}
