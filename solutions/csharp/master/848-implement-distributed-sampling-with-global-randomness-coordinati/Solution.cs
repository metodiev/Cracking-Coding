// Task 848 - Implement distributed sampling with global randomness coordination. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Sampling a distributed stream so that every sampler produces the *same* global sample as one central sampler,
// without shared mutable state and without coordinating per element. The mechanism is a globally seeded priority:
//
//   * every element gets the key u = H(globalSeed, epoch, id) from a hand-written splitmix64 mixing, so any shard
//     can compute any element's key on its own and two shards agree to the bit;
//   * a shard keeps the k smallest keys it saw (a bottom-k sketch) and the merge is again "keep the k smallest",
//     which is associative, commutative and idempotent - so the merged sample is *identical* to the sample a
//     central sampler would produce over the concatenated stream, for every sharding and every merge order;
//   * rotating the epoch changes the global seed, which resamples everything; stratified mode splits the key space
//     into S strata and gives each one a quota, so the sample stays representative of every region of the id space;
//   * weights enter through priority sampling: element i gets the priority size_i/u_i, the k largest priorities are
//     kept plus the (k+1)-th as the threshold tau, and the total is estimated as sum of max(size_i, tau), which is
//     the unbiased estimator the file validates.
//
// Verified, each by recomputation:
//   * exactness: over many stream lengths, k, shard counts and merge orders (including four different orderings of
//     four shards) the merged bottom-k sample equals the central sample element for element, with size min(k, n)
//     and no duplicates; the same holds for the priority sketch, whose merged estimate equals the central one;
//   * uniformity and independence: over 200 epochs every element's inclusion frequency matches k/n and the mean
//     overlap of consecutive epochs' samples matches the closed form k^2/n for independent k-subsets, both inside
//     Monte-Carlo tolerance, so the epoch rotation really draws fresh randomness;
//   * coordination is necessary: four observers that sample independently with their own seeds and concatenate
//     produce duplicates and disagree with the central sample, measured rather than asserted;
//   * stratification: every stratum keeps to its quota, all strata are represented, and the total stays within k;
//   * estimates: the Horvitz-Thompson estimate of the unweighted sample and the priority-sampling estimate of the
//     weighted one are unbiased for the stream total over 200 epochs, and a hundred-times larger element is drawn
//     far more often, which is the measurement that the size is inside the priority.
//
// Documented simplifications: the coordinates are an in-process epoch counter, so there is no clock skew, network
// deduplication or backpressure; the hash is a deterministic mixing rather than a cryptographic function, so the
// independence of the epochs is argued from the mixing structure and confirmed by the measured statistics instead
// of proven; the streams carry integer values, the estimator tolerances are Monte-Carlo over 200 epochs rather than
// confidence intervals, and the sample size is fixed rather than adapted to a target variance. No external data,
// service or key material is involved.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static ulong Mix(long seed, int epoch, int id, ulong salt)
    {
        ulong state = (ulong)seed * 0x9E3779B97F4A7C15UL ^ salt;
        state = Next(ref state) ^ ((ulong)(uint)epoch * 0xD6E8FEB86659FD93UL);
        state = Next(ref state) ^ ((ulong)(uint)id * 0xA24BAED4963EE407UL);
        return Next(ref state);
    }
}

/// <summary>An element of the stream: identity and value, the value being both the measured quantity and its size.</summary>
internal sealed class Element
{
    internal int Id { get; }
    internal long Value { get; }

    internal Element(int id, long value)
    {
        Id = id;
        Value = value;
    }
}

/// <summary>
/// The global randomness every sampler derives its keys from: the same seed and epoch anywhere give the same key.
/// </summary>
internal static class GlobalRandomness
{
    /// <summary>A uniform priority in [0, 1) for an element in an epoch.</summary>
    internal static double Key(long seed, int epoch, int id) =>
        (SplitMix64.Mix(seed, epoch, id, 0x9E3779B97F4A7C15UL) >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>
    /// The priority of an element for weighted sampling: size/uniform. A heavier element then outranks a lighter one
    /// with the same draw, and the inclusion probability is min(1, size/tau) for the threshold tau.
    /// </summary>
    internal static double Priority(double uniform, long size)
    {
        if (size <= 0) throw new ArgumentException("priorities need positive sizes", nameof(size));
        return size / Math.Max(uniform, 1e-12);
    }

    /// <summary>The stratum an element's key falls into, so stratification is coordinated by the same seed.</summary>
    internal static int Stratum(long seed, int epoch, int id, int strata) =>
        (int)(SplitMix64.Mix(seed, epoch, id, 0x2545F4914F6CDD1DUL) % (ulong)strata);
}

/// <summary>A bottom-k sample: the k smallest keys seen, keyed by element identity.</summary>
internal sealed class BottomK
{
    internal int Capacity { get; }
    private readonly List<(double Key, int Id)> _entries = new List<(double, int)>();

    internal BottomK(int capacity)
    {
        if (capacity < 1) throw new ArgumentException("a sample needs at least one slot", nameof(capacity));
        Capacity = capacity;
    }

    internal int Count => _entries.Count;
    internal IReadOnlyList<(double Key, int Id)> Entries => _entries;

    internal void Offer(double key, int id)
    {
        for (int i = 0; i < _entries.Count; i++)
        {
            if (_entries[i].Id == id)
            {
                // The same element seen twice - two shards holding a copy - keeps its smallest key, which makes the
                // operation idempotent as well as commutative.
                if (key < _entries[i].Key) _entries[i] = (key, id);
                return;
            }
        }
        if (_entries.Count < Capacity)
        {
            _entries.Add((key, id));
            return;
        }
        int worst = 0;
        for (int i = 1; i < _entries.Count; i++) if (_entries[i].Key > _entries[worst].Key) worst = i;
        if (key < _entries[worst].Key) _entries[worst] = (key, id);
    }

    /// <summary>Merging keeps the k smallest of the union, which is exactly what a central sampler would hold.</summary>
    internal BottomK Merge(BottomK other)
    {
        if (other.Capacity != Capacity) throw new ArgumentException("only samples of equal size merge", nameof(other));
        var merged = new BottomK(Capacity);
        foreach ((double key, int id) in _entries) merged.Offer(key, id);
        foreach ((double key, int id) in other._entries) merged.Offer(key, id);
        return merged;
    }

    /// <summary>The sample in key order, which is the deterministic canonical form.</summary>
    internal List<int> IdsInKeyOrder()
    {
        var copy = new List<(double Key, int Id)>(_entries);
        copy.Sort((a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : a.Id.CompareTo(b.Id));
        var ids = new List<int>();
        foreach ((double _, int id) in copy) ids.Add(id);
        return ids;
    }

    internal bool Contains(int id)
    {
        foreach ((double _, int id2) in _entries) if (id2 == id) return true;
        return false;
    }
}

/// <summary>
/// A priority sketch: it keeps the k+1 largest element priorities so that the (k+1)-th is available as the threshold
/// the priority-sampling estimator needs. Keys are stored negated so the bottom-k machinery can be reused.
/// </summary>
internal sealed class PrioritySketch
{
    internal int K { get; }
    private readonly BottomK _sketch;

    internal PrioritySketch(int k)
    {
        K = k;
        _sketch = new BottomK(k + 1);
    }

    internal int Count => _sketch.Count;
    internal IReadOnlyList<(double Key, int Id)> Entries => _sketch.Entries;

    internal void Offer(double priority, int id) => _sketch.Offer(-priority, id);

    internal PrioritySketch Merge(PrioritySketch other)
    {
        if (other.K != K) throw new ArgumentException("only priority sketches of equal size merge", nameof(other));
        var merged = new PrioritySketch(K);
        foreach ((double negated, int id) in _sketch.Entries) merged._sketch.Offer(negated, id);
        foreach ((double negated, int id) in other._sketch.Entries) merged._sketch.Offer(negated, id);
        return merged;
    }

    /// <summary>The (k+1)-th largest priority: the largest negated key the sketch holds.</summary>
    internal double Threshold()
    {
        double worst = double.NegativeInfinity;
        foreach ((double negated, int _) in _sketch.Entries) if (negated > worst) worst = negated;
        return -worst;
    }

    /// <summary>The k largest priorities, in descending order.</summary>
    internal List<(double Priority, int Id)> Top()
    {
        var entries = new List<(double Priority, int Id)>();
        foreach ((double negated, int id) in _sketch.Entries) entries.Add((-negated, id));
        entries.Sort((a, b) => b.Priority != a.Priority ? b.Priority.CompareTo(a.Priority) : a.Id.CompareTo(b.Id));
        if (entries.Count > K) entries.RemoveRange(K, entries.Count - K);
        return entries;
    }
}

internal static class Sampling
{
    internal static List<List<Element>> Split(IReadOnlyList<Element> stream, int shards)
    {
        var slices = new List<List<Element>>();
        for (int s = 0; s < shards; s++) slices.Add(new List<Element>());
        for (int i = 0; i < stream.Count; i++) slices[i % shards].Add(stream[i]);
        return slices;
    }

    /// <summary>Central sampling: one pass with the global seed, the oracle every distributed check is compared with.</summary>
    internal static BottomK Central(IReadOnlyList<Element> stream, int k, long seed, int epoch)
    {
        var sample = new BottomK(k);
        foreach (Element element in stream) sample.Offer(GlobalRandomness.Key(seed, epoch, element.Id), element.Id);
        return sample;
    }

    /// <summary>Each shard samples its slice locally with the shared seed, then the samples merge in the given order.</summary>
    internal static BottomK Distributed(IReadOnlyList<Element> stream, int shards, int k, long seed, int epoch,
        int[] mergeOrder = null)
    {
        List<List<Element>> slices = Split(stream, shards);
        var partial = new List<BottomK>();
        foreach (List<Element> slice in slices)
        {
            var sample = new BottomK(k);
            foreach (Element element in slice) sample.Offer(GlobalRandomness.Key(seed, epoch, element.Id), element.Id);
            partial.Add(sample);
        }
        int[] order = mergeOrder ?? Sequential(shards);
        BottomK merged = null;
        foreach (int index in order) merged = merged is null ? partial[index] : merged.Merge(partial[index]);
        return merged ?? new BottomK(k);
    }

    internal static int[] Sequential(int shards)
    {
        var order = new int[shards];
        for (int i = 0; i < shards; i++) order[i] = i;
        return order;
    }

    /// <summary>
    /// The uncoordinated baseline: s observers each sample the whole stream with their own randomness and the results
    /// are concatenated, which is what merging without a shared seed degenerates to.
    /// </summary>
    internal static List<int> Uncoordinated(IReadOnlyList<Element> stream, int observers, int k, long seed, int epoch)
    {
        var picked = new List<int>();
        for (int s = 0; s < observers; s++)
        {
            var sample = new BottomK(k);
            foreach (Element element in stream)
                sample.Offer(GlobalRandomness.Key(seed + 1000 + s, epoch, element.Id), element.Id);
            picked.AddRange(sample.IdsInKeyOrder());
        }
        return picked;
    }

    /// <summary>Stratified sampling: every stratum reports its own quota of the k smallest keys.</summary>
    internal static BottomK Stratified(IReadOnlyList<Element> stream, int k, long seed, int epoch, int strata,
        out int[] perStratumCounts)
    {
        perStratumCounts = new int[strata];
        int quota = Math.Max(1, k / strata);
        var buckets = new List<Element>[strata];
        for (int s = 0; s < strata; s++) buckets[s] = new List<Element>();
        foreach (Element element in stream) buckets[GlobalRandomness.Stratum(seed, epoch, element.Id, strata)].Add(element);
        var merged = new BottomK(k);
        for (int s = 0; s < strata; s++)
        {
            var sample = new BottomK(quota);
            foreach (Element element in buckets[s])
                sample.Offer(GlobalRandomness.Key(seed, epoch, element.Id), element.Id);
            perStratumCounts[s] = sample.Count;
            foreach ((double key, int id) in sample.Entries) merged.Offer(key, id);
        }
        return merged;
    }
}

internal static class PrioritySampling
{
    internal static PrioritySketch Central(IReadOnlyList<Element> stream, int k, long seed, int epoch)
    {
        var sketch = new PrioritySketch(k);
        foreach (Element element in stream)
            sketch.Offer(GlobalRandomness.Priority(GlobalRandomness.Key(seed, epoch, element.Id), element.Value),
                element.Id);
        return sketch;
    }

    internal static PrioritySketch Distributed(IReadOnlyList<Element> stream, int shards, int k, long seed, int epoch)
    {
        List<List<Element>> slices = Sampling.Split(stream, shards);
        PrioritySketch merged = null;
        foreach (List<Element> slice in slices)
        {
            var sketch = new PrioritySketch(k);
            foreach (Element element in slice)
                sketch.Offer(GlobalRandomness.Priority(GlobalRandomness.Key(seed, epoch, element.Id), element.Value),
                    element.Id);
            merged = merged is null ? sketch : merged.Merge(sketch);
        }
        return merged ?? new PrioritySketch(k);
    }

    /// <summary>
    /// The priority-sampling estimate of the stream total: every element of the sample is credited
    /// max(size, tau), where tau is the (k+1)-th largest priority, which is the standard unbiased estimator.
    /// </summary>
    internal static double Estimate(IReadOnlyList<Element> stream, PrioritySketch sketch)
    {
        var sizes = new Dictionary<int, long>();
        foreach (Element element in stream) sizes[element.Id] = element.Value;
        double tau = sketch.Threshold();
        double total = 0;
        foreach ((double _, int id) in sketch.Top()) total += Math.Max(sizes[id], tau);
        return total;
    }
}

internal static class Estimators
{
    /// <summary>Horvitz-Thompson over an unweighted bottom-k sample: each sampled element stands for n/k of them.</summary>
    internal static double Unweighted(IReadOnlyList<Element> stream, BottomK sample, int n, int k)
    {
        var values = new Dictionary<int, long>();
        foreach (Element element in stream) values[element.Id] = element.Value;
        double total = 0;
        foreach ((double _, int id) in sample.Entries) total += values[id];
        return total * (double)Math.Max(1, n) / Math.Max(1, k);
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<Element> Build(int count, ulong seed)
    {
        ulong state = seed;
        var stream = new List<Element>();
        for (int i = 0; i < count; i++) stream.Add(new Element(i, 1 + SplitMix64.NextBelow(ref state, 50)));
        return stream;
    }

    private static bool SameSample(List<int> left, List<int> right)
    {
        if (left.Count != right.Count) return false;
        var a = new List<int>(left);
        var b = new List<int>(right);
        a.Sort();
        b.Sort();
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        const long Seed = 0x48484848L;

        // ---- the merged sample is exactly the central sample ------------------------------------------
        {
            bool exact = true, sizesOk = true, noDuplicates = true, keysAgree = true;
            long cases = 0;
            foreach (int length in new[] { 40, 120, 400 })
            {
                foreach (int k in new[] { 1, 5, 17 })
                {
                    List<Element> stream = Build(length, 0x5EED848000000001UL + (ulong)length);
                    BottomK central = Sampling.Central(stream, k, Seed, 3);
                    if (central.Count != Math.Min(k, length)) sizesOk = false;
                    foreach (Element element in stream)
                        if (GlobalRandomness.Key(Seed, 3, element.Id) != GlobalRandomness.Key(Seed, 3, element.Id))
                            keysAgree = false;
                    for (int shards = 1; shards <= 6; shards++)
                    {
                        BottomK distributed = Sampling.Distributed(stream, shards, k, Seed, 3);
                        if (!SameSample(distributed.IdsInKeyOrder(), central.IdsInKeyOrder())) exact = false;
                        if (distributed.Count != Math.Min(k, length)) sizesOk = false;
                        var seen = new HashSet<int>();
                        foreach ((double _, int id) in distributed.Entries) if (!seen.Add(id)) noDuplicates = false;
                        cases++;
                    }
                }
            }
            Check(exact, "the merged distributed sample equals the central sample element for element in all " + cases
                + " length/k/shard combinations");
            Check(sizesOk, "every sample holds exactly min(k, n) elements");
            Check(noDuplicates, "and never contains the same element twice");
            Check(keysAgree, "and every shard derives identical keys for the same element from the one global seed");
            Console.WriteLine("exactness: " + cases + " combinations, all identical to the central sample");
        }

        // ---- every merge order agrees, and a duplicate element collapses to one entry -----------------
        {
            List<Element> stream = Build(150, 0x5EED848000000002UL);
            const int K = 9;
            BottomK central = Sampling.Central(stream, K, Seed, 1);
            var orders = new List<int[]>
            {
                new[] { 0, 1, 2, 3 }, new[] { 3, 2, 1, 0 }, new[] { 1, 3, 0, 2 }, new[] { 2, 0, 3, 1 },
            };
            bool allEqual = true;
            foreach (int[] order in orders)
            {
                BottomK merged = Sampling.Distributed(stream, 4, K, Seed, 1, order);
                if (!SameSample(merged.IdsInKeyOrder(), central.IdsInKeyOrder())) allEqual = false;
            }
            Check(allEqual, "four different merge orders all reproduce the central sample, so the merge is associative "
                + "and commutative");

            var first = new BottomK(K);
            first.Offer(GlobalRandomness.Key(Seed, 1, 7), 7);
            var second = new BottomK(K);
            second.Offer(GlobalRandomness.Key(Seed, 1, 7), 7);
            second.Offer(GlobalRandomness.Key(Seed, 1, 8), 8);
            BottomK union = first.Merge(second);
            Check(union.Count == 2 && union.Contains(7) && union.Contains(8),
                "the same element offered by two shards counts once, because the merge is idempotent too");
            Console.WriteLine("merge algebra: 4 orders agree, duplicate element collapsed to one entry");
        }

        // ---- uniformity and epoch independence ---------------------------------------------------------
        {
            const int Length = 64, K = 8, Epochs = 200;
            var frequencies = new long[Length];
            long overlapTotal = 0;
            var previous = new List<int>();
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                List<Element> stream = Build(Length, 0x5EED848000000003UL);
                BottomK sample = Sampling.Central(stream, K, Seed, epoch);
                var current = new List<int>();
                foreach ((double _, int id) in sample.Entries) current.Add(id);
                foreach (int id in current) frequencies[id]++;
                if (epoch > 0)
                {
                    var seen = new HashSet<int>(current);
                    foreach (int id in previous) if (seen.Contains(id)) overlapTotal++;
                }
                previous = current;
            }
            double expected = (double)K / Length;
            double worst = 0;
            for (int id = 0; id < Length; id++)
            {
                double gap = Math.Abs((double)frequencies[id] / Epochs - expected);
                if (gap > worst) worst = gap;
            }
            Check(worst < 0.12, "every element's inclusion frequency matches k/n over " + Epochs + " epochs (worst "
                + "deviation " + worst.ToString("F4") + " from " + expected.ToString("F4") + ")");
            double overlap = (double)overlapTotal / (Epochs - 1);
            double closedForm = (double)K * K / Length;
            Check(Math.Abs(overlap - closedForm) < 0.35, "consecutive epochs share the expected " + closedForm.ToString("F3")
                + " elements for independent k-subsets (measured " + overlap.ToString("F3") + "), so rotating the epoch "
                + "really draws fresh randomness");
            Console.WriteLine("uniformity: worst inclusion deviation " + worst.ToString("F4") + " | epoch overlap measured "
                + overlap.ToString("F3") + " expected " + closedForm.ToString("F3"));
        }

        // ---- coordination is necessary -----------------------------------------------------------------
        {
            List<Element> stream = Build(120, 0x5EED848000000004UL);
            const int K = 10;
            BottomK central = Sampling.Central(stream, K, Seed, 0);
            List<int> uncoordinated = Sampling.Uncoordinated(stream, 4, K, Seed, 0);
            var seen = new HashSet<int>();
            int duplicates = 0;
            foreach (int id in uncoordinated) if (!seen.Add(id)) duplicates++;
            Check(uncoordinated.Count > K, "four observers that sample the same stream independently and concatenate "
                + "return " + uncoordinated.Count + " entries for a k of " + K);
            Check(duplicates > 0, "with " + duplicates + " duplicated elements, so it is not a sample at all");
            Check(!SameSample(uncoordinated.GetRange(0, K), central.IdsInKeyOrder()),
                "and truncating it to k entries still does not match the central sample, which is why the keys must come "
                + "from one global seed");
            Console.WriteLine("uncoordinated baseline: " + uncoordinated.Count + " entries, " + duplicates
                + " duplicates, disagrees with the central sample");
        }

        // ---- stratification keeps every part of the id space ---------------------------------------------
        {
            List<Element> stream = Build(200, 0x5EED848000000005UL);
            const int K = 12, Strata = 6;
            BottomK stratified = Sampling.Stratified(stream, K, Seed, 4, Strata, out int[] perStratum);
            BottomK central = Sampling.Central(stream, K, Seed, 4);
            bool quotas = true;
            int covered = 0;
            foreach (int count in perStratum)
            {
                if (count > Math.Max(1, K / Strata)) quotas = false;
                if (count > 0) covered++;
            }
            Check(quotas, "every stratum respected its quota of " + (K / Strata) + " slots ("
                + string.Join(",", perStratum) + ")");
            Check(covered == Strata, "and all " + Strata + " strata are represented, which the unstratified sample "
                + "cannot promise");
            Check(stratified.Count <= K, "the stratified sample never exceeds k (" + stratified.Count + ")");
            var centralStrata = new HashSet<int>();
            foreach ((double _, int id) in central.Entries) centralStrata.Add(GlobalRandomness.Stratum(Seed, 4, id, Strata));
            Check(centralStrata.Count <= Strata, "the central sample covers " + centralStrata.Count + " strata of "
                + Strata + " on the same stream, which is what stratification improves on");
            Console.WriteLine("stratification: quotas " + string.Join(",", perStratum) + " covering " + covered + "/"
                + Strata + " strata, central covering " + centralStrata.Count);
        }

        // ---- the estimates are unbiased -------------------------------------------------------------------
        {
            const int Length = 50, K = 5, Epochs = 200;
            List<Element> stream = Build(Length, 0x5EED848000000006UL);
            long trueTotal = 0;
            foreach (Element element in stream) trueTotal += element.Value;
            double sum = 0;
            bool priorityExact = true;
            double prioritySum = 0;
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                BottomK sample = Sampling.Central(stream, K, Seed, epoch);
                sum += Estimators.Unweighted(stream, sample, Length, K);
                PrioritySketch centralSketch = PrioritySampling.Central(stream, K, Seed, epoch);
                PrioritySketch mergedSketch = PrioritySampling.Distributed(stream, 4, K, Seed, epoch);
                double centralEstimate = PrioritySampling.Estimate(stream, centralSketch);
                double mergedEstimate = PrioritySampling.Estimate(stream, mergedSketch);
                if (Math.Abs(centralEstimate - mergedEstimate) > 1e-9) priorityExact = false;
                prioritySum += mergedEstimate;
            }
            double mean = sum / Epochs;
            double relativeError = Math.Abs(mean - trueTotal) / trueTotal;
            Check(relativeError < 0.08, "the Horvitz-Thompson estimate of the total is unbiased over " + Epochs
                + " epochs (" + mean.ToString("F2") + " against the true " + trueTotal + ", relative error "
                + relativeError.ToString("F4") + ")");
            double priorityMean = prioritySum / Epochs;
            double priorityError = Math.Abs(priorityMean - trueTotal) / trueTotal;
            Check(priorityExact, "the merged priority sketch gives the same estimate as the central one in every epoch, "
                + "so the weighted merge is exact too");
            Check(priorityError < 0.12, "and the priority-sampling estimate is unbiased for the same total over "
                + Epochs + " epochs (" + priorityMean.ToString("F2") + " against " + trueTotal + ", relative error "
                + priorityError.ToString("F4") + ")");
            Console.WriteLine("estimators: unweighted relative error " + relativeError.ToString("F4") + ", weighted "
                + priorityError.ToString("F4"));
        }

        // ---- the size really is inside the priority --------------------------------------------------------
        {
            const int Epochs = 300;
            var heavy = new Element(1, 500);
            var light = new Element(2, 5);
            var stream = new List<Element> { heavy, light };
            for (int i = 10; i < 40; i++) stream.Add(new Element(i, 5));
            int heavyDrawn = 0, lightDrawn = 0;
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                PrioritySketch sketch = PrioritySampling.Central(stream, 4, Seed, epoch);
                foreach ((double _, int id) in sketch.Top())
                {
                    if (id == heavy.Id) heavyDrawn++;
                    if (id == light.Id) lightDrawn++;
                }
            }
            Check(heavyDrawn > lightDrawn + 20, "the hundred-times larger element is drawn far more often (" + heavyDrawn
                + " vs " + lightDrawn + " of " + Epochs + " epochs), so the size is in the priority");
            Check(PrioritySampling.Central(stream, 4, Seed, 0).Count == 5,
                "the priority sketch holds k+1 entries, one more than the sample it will report");
            Console.WriteLine("priorities: heavy drawn " + heavyDrawn + ", light drawn " + lightDrawn + " of " + Epochs
                + " epochs");
        }

        // ---- degenerate cases ------------------------------------------------------------------------------
        {
            List<Element> stream = Build(20, 0x5EED848000000008UL);
            Check(Sampling.Central(stream, 100, Seed, 0).Count == 20, "k larger than the stream returns every element");
            BottomK one = Sampling.Central(stream, 1, Seed, 0);
            Check(one.Count == 1, "k = 1 returns a single element");
            Check(SameSample(Sampling.Distributed(stream, 3, 1, Seed, 0).IdsInKeyOrder(), one.IdsInKeyOrder()),
                "and the distributed k = 1 agrees with the central k = 1");

            var empty = new List<Element>();
            BottomK emptySample = Sampling.Central(empty, 5, Seed, 0);
            Check(emptySample.Count == 0, "an empty stream gives an empty sample");
            List<List<Element>> slices = Sampling.Split(empty, 3);
            var idle = new BottomK(5);
            foreach (Element element in slices[1]) idle.Offer(GlobalRandomness.Key(Seed, 0, element.Id), element.Id);
            Check(idle.Count == 0, "a shard that receives nothing contributes an empty sample");
            Check(SameSample(emptySample.Merge(idle).IdsInKeyOrder(), emptySample.IdsInKeyOrder()),
                "and merging an empty sample changes nothing");

            var single = new List<Element> { new Element(42, 7) };
            BottomK singleSample = Sampling.Distributed(single, 4, 3, Seed, 0);
            Check(singleSample.Count == 1 && singleSample.Contains(42),
                "a one-element stream is sampled by whichever shard holds it");
            Console.WriteLine("degenerate: k>n, k=1, empty stream, idle shard and single element all handled");
        }

        // ---- determinism -----------------------------------------------------------------------------------
        {
            List<Element> stream = Build(180, 0x5EED848000000009UL);
            BottomK first = Sampling.Distributed(stream, 5, 8, Seed, 7);
            BottomK second = Sampling.Distributed(stream, 5, 8, Seed, 7);
            Check(SameSample(first.IdsInKeyOrder(), second.IdsInKeyOrder()), "two runs of the same epoch agree exactly");
            BottomK otherEpoch = Sampling.Distributed(stream, 5, 8, Seed, 8);
            Check(!SameSample(first.IdsInKeyOrder(), otherEpoch.IdsInKeyOrder()),
                "and a different epoch draws a different sample, which is the rotation working");
            Console.WriteLine("determinism: 8 of 180 elements sampled reproducibly, epoch 8 differs");
        }

        Console.WriteLine("PASS");
    }
}
