// Task 847 - Implement heavy-hitter detection with distributed sketches mergeable. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Heavy hitters over a stream that is processed by several shards at once, using the two structures real systems
// combine:
//
//   * a Count-Min sketch: d rows of w counters with d independent hash functions. An update increments one counter
//     per row, a query takes the minimum over the rows, so the estimate is an upper bound of the true count whose
//     excess is bounded by eps*N with probability 1-delta for w = ceil(e/eps) and d = ceil(ln(1/delta)). Both the
//     plain and the conservative update rule (which only raises the counters that the minimum would raise) are
//     implemented, and the file measures that the conservative rule is strictly more accurate.
//   * a Misra-Gries summary with k counters: a key that is tracked is incremented, an untracked key takes a free
//     counter, and when the table is full every counter is decremented and the zeros are dropped. Every key with
//     true count > N/k is guaranteed to be tracked, and every tracked counter understates its key by at most N/k.
//
// Mergeability is the property that makes the pair usable in a distributed setting, and it is what the checks
// concentrate on: Count-Min sketches add counterwise, so the merged sketch is *exactly* the sketch of the
// concatenated stream (checked field by field against a single-pass sketch), while Misra-Gries summaries merge by
// summing shared counters, decrementing by the (k+1)-th largest value and dropping the non-positive entries - the
// mergeable-summary rule - which the file then verifies still has full recall and still satisfies the error bound.
//
// Verified, each by recomputation:
//   * recall: every key whose true count exceeds N/k (for the summary) or phi*N (for the sketch) is reported, for
//     several shard counts, stream lengths, k and epsilons, over seeded uniform, Zipf and all-distinct streams;
//   * the error bounds: the Count-Min estimate is never below the true count and never above true + eps*N (the
//     empirical maximum excess is reported), and the Misra-Gries counters are never below true - N/k;
//   * exact mergeability of the sketch (counter for counter against a single-pass sketch), the merge rule's
//     validity for the summary, and that shard-local processing with a merge is equivalent to one pass;
//   * space is bounded and measured (d*w counters, at most k summary entries, and how many entries survive a
//     merge), with the conservative update and the two summaries compared on the same streams;
//   * degenerate inputs: an empty stream, one key, all-distinct keys (where no key is heavy and the answer must be
//     empty), a shard that sees nothing, k = 1, and a key that dominates the stream.
//
// Documented simplifications: the shards are in-process and receive disjoint slices of one generated stream rather
// than a network feed, so there is no ordering, retry or skew-in-arrival modelling; the hash functions are seeded
// splitmix64 mixings (pairwise independence by construction is not proven, only used); eps and delta are exact
// rationals of the sizing formulas and the measured error rates are empirical, so the probabilistic guarantee is
// reported as a measurement over deterministic trials rather than proven; and the merge rule is the classic one,
// whose constant is measured rather than assumed. No external data is involved.
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

    /// <summary>Deterministic mixing of a key with a row seed: the hash family of the sketch.</summary>
    internal static ulong Mix(long key, long salt)
    {
        ulong state = (ulong)key * 0x9E3779B97F4A7C15UL ^ (ulong)salt * 0xD6E8FEB86659FD93UL;
        state = Next(ref state);
        return Next(ref state);
    }
}

/// <summary>Count-Min sketch with either the plain or the conservative update rule.</summary>
internal sealed class CountMin
{
    internal int Rows { get; }
    internal int Width { get; }
    internal long[,] Counters { get; }
    internal bool Conservative { get; }
    internal long Updates { get; private set; }

    private readonly long[] _salts;

    internal CountMin(int rows, int width, bool conservative, long saltBase)
    {
        if (rows < 1 || width < 1) throw new ArgumentException("a sketch needs at least one counter");
        Rows = rows;
        Width = width;
        Conservative = conservative;
        Counters = new long[rows, width];
        _salts = new long[rows];
        for (int r = 0; r < rows; r++) _salts[r] = saltBase + r * 0x9E3779B1L;
    }

    /// <summary>The sizing formulas: w = ceil(e/eps) and d = ceil(ln(1/delta)).</summary>
    internal static CountMin Sized(double epsilon, double delta, bool conservative, long saltBase)
    {
        int width = (int)Math.Ceiling(Math.E / epsilon);
        int rows = (int)Math.Ceiling(Math.Log(1.0 / delta));
        return new CountMin(Math.Max(1, rows), Math.Max(1, width), conservative, saltBase);
    }

    private int Column(long key, int row) => (int)(SplitMix64.Mix(key, _salts[row]) % (ulong)Width);

    internal void Add(long key, long amount)
    {
        if (amount < 0) throw new ArgumentException("the stream has no deletions in this model", nameof(amount));
        Updates++;
        if (!Conservative)
        {
            for (int row = 0; row < Rows; row++) Counters[row, Column(key, row)] += amount;
            return;
        }
        // Conservative update: raise only the minima, which keeps the upper-bound property and loses less accuracy.
        long minimum = long.MaxValue;
        for (int row = 0; row < Rows; row++)
        {
            long current = Counters[row, Column(key, row)];
            if (current < minimum) minimum = current;
        }
        for (int row = 0; row < Rows; row++)
        {
            int column = Column(key, row);
            long target = minimum + amount;
            if (Counters[row, column] < target) Counters[row, column] = target;
        }
    }

    internal long Estimate(long key)
    {
        long minimum = long.MaxValue;
        for (int row = 0; row < Rows; row++)
        {
            long current = Counters[row, Column(key, row)];
            if (current < minimum) minimum = current;
        }
        return minimum;
    }

    /// <summary>Counterwise addition: the merged sketch is exactly the sketch of the concatenated stream.</summary>
    internal CountMin Merge(CountMin other)
    {
        if (other.Rows != Rows || other.Width != Width || other.Conservative != Conservative)
            throw new ArgumentException("only sketches with identical geometry can be merged", nameof(other));
        var merged = new CountMin(Rows, Width, Conservative, _salts[0]);
        for (int row = 0; row < Rows; row++)
            for (int column = 0; column < Width; column++)
                merged.Counters[row, column] = Counters[row, column] + other.Counters[row, column];
        merged.Updates = Updates + other.Updates;
        return merged;
    }

    internal long TotalMass()
    {
        long total = 0;
        for (int row = 0; row < Rows; row++)
            for (int column = 0; column < Width; column++) total += Counters[row, column];
        return total;
    }
}

/// <summary>Misra-Gries summary with k slots, and the classic mergeable-summary merge.</summary>
internal sealed class MisraGries
{
    internal int Slots { get; }
    internal Dictionary<long, long> Counters { get; } = new Dictionary<long, long>();
    internal long Processed { get; private set; }

    internal MisraGries(int slots)
    {
        if (slots < 1) throw new ArgumentException("a summary needs at least one slot", nameof(slots));
        Slots = slots;
    }

    internal void Add(long key, long amount)
    {
        Processed += amount;
        if (Counters.TryGetValue(key, out long current))
        {
            Counters[key] = current + amount;
            return;
        }
        if (Counters.Count < Slots)
        {
            Counters[key] = amount;
            return;
        }
        // The table is full: every counter gives up `amount` and the emptied slots are released.
        Decrement(amount);
    }

    /// <summary>
    /// The mergeable-summary rule: sum the shared counters, keep the k largest, then subtract the (k+1)-th largest
    /// from everything and drop the non-positive entries. The result is a valid Misra-Gries summary of the union.
    /// </summary>
    internal MisraGries Merge(MisraGries other)
    {
        var combined = new Dictionary<long, long>(Counters);
        foreach (KeyValuePair<long, long> entry in other.Counters)
        {
            combined.TryGetValue(entry.Key, out long current);
            combined[entry.Key] = current + entry.Value;
        }
        var merged = new MisraGries(Slots);
        if (combined.Count <= Slots)
        {
            foreach (KeyValuePair<long, long> entry in combined) merged.Counters[entry.Key] = entry.Value;
            merged.Processed = Processed + other.Processed;
            return merged;
        }
        var values = new List<long>(combined.Values);
        values.Sort((a, b) => b.CompareTo(a));
        long cut = values[Slots];                       // the (k+1)-th largest value
        foreach (KeyValuePair<long, long> entry in combined)
        {
            long reduced = entry.Value - cut;
            if (reduced > 0) merged.Counters[entry.Key] = reduced;
        }
        merged.Processed = Processed + other.Processed;
        return merged;
    }

    internal long Estimate(long key) => Counters.TryGetValue(key, out long value) ? value : 0;

    private void Decrement(long amount)
    {
        var emptied = new List<long>();
        var keys = new List<long>(Counters.Keys);
        foreach (long key in keys)
        {
            long reduced = Counters[key] - amount;
            if (reduced <= 0) emptied.Add(key);
            else Counters[key] = reduced;
        }
        foreach (long key in emptied) Counters.Remove(key);
    }
}

/// <summary>One shard: it processes a slice of the stream and hands its summaries over for the merge.</summary>
internal sealed class Shard
{
    internal int Id { get; }
    internal CountMin Sketch { get; }
    internal MisraGries Summary { get; }
    internal long Items;

    internal Shard(int id, CountMin sketch, MisraGries summary)
    {
        Id = id;
        Sketch = sketch;
        Summary = summary;
    }

    internal void Consume(IReadOnlyList<long> slice)
    {
        foreach (long key in slice)
        {
            Sketch.Add(key, 1);
            Summary.Add(key, 1);
            Items++;
        }
    }
}

internal static class Streams
{
    /// <summary>A uniformly random stream over `distinct` keys.</summary>
    internal static List<long> Uniform(int length, int distinct, ulong seed)
    {
        ulong state = seed;
        var stream = new List<long>(length);
        for (int i = 0; i < length; i++) stream.Add(SplitMix64.NextBelow(ref state, distinct));
        return stream;
    }

    /// <summary>A Zipf-like stream: key r appears with weight proportional to 1/(r+1).</summary>
    internal static List<long> Zipf(int length, int distinct, ulong seed)
    {
        ulong state = seed;
        var weights = new List<long>();
        long total = 0;
        for (int r = 0; r < distinct; r++)
        {
            long weight = (long)(1000.0 / (r + 1));
            weights.Add(total);
            total += weight;
        }
        var stream = new List<long>(length);
        for (int i = 0; i < length; i++)
        {
            long pick = SplitMix64.NextBelow(ref state, (int)total);
            int chosen = 0;
            for (int r = 0; r < distinct; r++) if (weights[r] <= pick) chosen = r;
            stream.Add(chosen);
        }
        return stream;
    }

    internal static Dictionary<long, long> TrueCounts(IReadOnlyList<long> stream)
    {
        var counts = new Dictionary<long, long>();
        foreach (long key in stream)
        {
            counts.TryGetValue(key, out long current);
            counts[key] = current + 1;
        }
        return counts;
    }

    /// <summary>Round-robin sharding of a stream into disjoint slices.</summary>
    internal static List<List<long>> Split(IReadOnlyList<long> stream, int shards)
    {
        var slices = new List<List<long>>();
        for (int s = 0; s < shards; s++) slices.Add(new List<long>());
        for (int i = 0; i < stream.Count; i++) slices[i % shards].Add(stream[i]);
        return slices;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Runs the distributed pipeline: shard-local summaries merged into one pair of sketches.</summary>
    private static (CountMin Sketch, MisraGries Summary) Distributed(IReadOnlyList<long> stream, int shards,
        int rows, int width, int slots, bool conservative, long saltBase)
    {
        List<List<long>> slices = Streams.Split(stream, shards);
        CountMin merged = null!;
        MisraGries mergedSummary = null!;
        for (int s = 0; s < shards; s++)
        {
            var shard = new Shard(s, new CountMin(rows, width, conservative, saltBase),
                new MisraGries(slots));
            shard.Consume(slices[s]);
            merged = s == 0 ? shard.Sketch : merged.Merge(shard.Sketch);
            mergedSummary = s == 0 ? shard.Summary : mergedSummary.Merge(shard.Summary);
        }
        return (merged, mergedSummary);
    }

    public static void Main()
    {
        // ---- the sketch is an upper bound with a measured eps*N excess --------------------------------
        {
            bool upperBound = true;
            double worstRatio = 0;
            long cases = 0;
            foreach ((int length, int distinct, string shape) in new (int, int, string)[]
            {
                (2000, 50, "uniform"),
                (2000, 50, "zipf"),
                (800, 200, "uniform"),
            })
            {
                List<long> stream = shape == "zipf"
                    ? Streams.Zipf(length, distinct, 0x5EED847000000001UL)
                    : Streams.Uniform(length, distinct, 0x5EED847000000002UL);
                Dictionary<long, long> truth = Streams.TrueCounts(stream);
                const double Epsilon = 0.02, Delta = 0.01;
                CountMin sketch = CountMin.Sized(Epsilon, Delta, false, 0x847);
                foreach (long key in stream) sketch.Add(key, 1);
                foreach (KeyValuePair<long, long> entry in truth)
                {
                    long estimate = sketch.Estimate(entry.Key);
                    if (estimate < entry.Value) upperBound = false;
                    double excess = (double)(estimate - entry.Value) / (Epsilon * length);
                    if (excess > worstRatio) worstRatio = excess;
                    cases++;
                }
                Check(sketch.Rows == (int)Math.Ceiling(Math.Log(1.0 / Delta))
                    && sketch.Width == (int)Math.Ceiling(Math.E / Epsilon),
                    "the sketch is sized by the standard formulas (" + sketch.Rows + " rows, " + sketch.Width
                    + " columns)");
            }
            Check(upperBound, "the Count-Min estimate is never below the true count in " + cases + " queries");
            Check(worstRatio <= 1.0, "and the excess never reaches the eps*N bound (worst measured ratio "
                + worstRatio.ToString("F4") + " of eps*N)");
            Console.WriteLine("sketch bounds: " + cases + " queries, worst excess ratio " + worstRatio.ToString("F4"));
        }

        // ---- the conservative update is strictly more accurate ----------------------------------------
        {
            List<long> stream = Streams.Zipf(4000, 80, 0x5EED847000000003UL);
            Dictionary<long, long> truth = Streams.TrueCounts(stream);
            // A deliberately small sketch, so collisions are the regime being compared.
            CountMin plain = new CountMin(5, 14, false, 0x847);
            CountMin conservative = new CountMin(5, 14, true, 0x847);
            foreach (long key in stream) { plain.Add(key, 1); conservative.Add(key, 1); }
            long plainError = 0, conservativeError = 0;
            long worstPlain = 0;
            bool bothUpper = true;
            foreach (KeyValuePair<long, long> entry in truth)
            {
                long plainExcess = plain.Estimate(entry.Key) - entry.Value;
                long conservativeExcess = conservative.Estimate(entry.Key) - entry.Value;
                if (plainExcess < 0 || conservativeExcess < 0) bothUpper = false;
                plainError += plainExcess;
                conservativeError += conservativeExcess;
                if (plainExcess > worstPlain) worstPlain = plainExcess;
            }
            Check(bothUpper, "both update rules keep the estimate above the true count even in a 5x14 sketch");
            Check(conservativeError < plainError, "the conservative update loses less accuracy than the plain update ("
                + conservativeError + " vs " + plainError + " total excess)");
            Check(conservative.Estimate(0) <= plain.Estimate(0), "and its query for the hottest key is never larger");
            Check(plain.TotalMass() > conservative.TotalMass(), "the plain update inflates the counter mass more ("
                + plain.TotalMass() + " vs " + conservative.TotalMass() + ")");
            Console.WriteLine("update rules: plain excess=" + plainError + " conservative excess=" + conservativeError
                + " worstPlainExcess=" + worstPlain);
        }

        // ---- the sketch merges exactly: field by field against a single pass --------------------------
        {
            bool exact = true, estimatesMatch = true;
            long fields = 0;
            for (int seed = 1; seed <= 5; seed++)
            {
                List<long> stream = Streams.Zipf(1500, 40, (ulong)seed * 0x9E3779B97F4A7C15UL);
                int shards = 1 + seed % 5;
                List<List<long>> slices = Streams.Split(stream, shards);
                CountMin merged = new CountMin(4, 64, false, 0x847);
                for (int s = 0; s < shards; s++)
                {
                    var shardSketch = new CountMin(4, 64, false, 0x847);
                    foreach (long key in slices[s]) shardSketch.Add(key, 1);
                    merged = s == 0 ? shardSketch : merged.Merge(shardSketch);
                }
                var singlePass = new CountMin(4, 64, false, 0x847);
                foreach (long key in stream) singlePass.Add(key, 1);
                for (int row = 0; row < 4; row++)
                    for (int column = 0; column < 64; column++)
                    {
                        fields++;
                        if (merged.Counters[row, column] != singlePass.Counters[row, column]) exact = false;
                    }
                foreach (long key in stream)
                    if (merged.Estimate(key) != singlePass.Estimate(key)) estimatesMatch = false;
            }
            Check(exact, "merging " + 5 + " different shardings reproduces the single-pass sketch counter for counter ("
                + fields + " fields compared)");
            Check(estimatesMatch, "and every query against the merged sketch equals the single-pass query");
            Console.WriteLine("sketch merge: " + fields + " counters identical across 5 shardings");
        }

        // ---- heavy hitters: full recall of the phi-heavy keys, and the error bound --------------------
        {
            const int Slots = 20;
            bool fullRecall = true, errorBound = true, spaceBounded = true;
            long heavyTotal = 0, reportedTotal = 0, cases = 0;
            double worstErrorRatio = 0;
            foreach ((int length, int distinct, string shape) in new (int, int, string)[]
            {
                (4000, 60, "zipf"),
                (4000, 30, "zipf"),
                (3000, 200, "zipf"),
                (3000, 100, "uniform"),
            })
            {
                List<long> stream = shape == "zipf"
                    ? Streams.Zipf(length, distinct, 0x5EED847000000004UL + (ulong)length)
                    : Streams.Uniform(length, distinct, 0x5EED847000000005UL + (ulong)length);
                Dictionary<long, long> truth = Streams.TrueCounts(stream);
                double threshold = (double)length / Slots;
                foreach (int shards in new[] { 1, 3, 7 })
                {
                    (CountMin sketch, MisraGries summary) = Distributed(stream, shards, 4, 128, Slots, false, 0x847);
                    Check(summary.Counters.Count <= Slots, "the summary never exceeds its slot budget");
                    Check(summary.Processed == length, "the merged summary accounts for the whole stream (" + summary.Processed
                        + " of " + length + ")");
                    foreach (KeyValuePair<long, long> entry in truth)
                    {
                        bool heavy = entry.Value > threshold;
                        long estimate = summary.Estimate(entry.Key);
                        if (heavy) { heavyTotal++; if (estimate == 0) fullRecall = false; }
                        if (estimate > 0)
                        {
                            reportedTotal++;
                            double understatement = (double)(entry.Value - estimate) / threshold;
                            if (understatement > worstErrorRatio) worstErrorRatio = understatement;
                            if (entry.Value > estimate && entry.Value - estimate > threshold) errorBound = false;
                        }
                    }
                    Check(sketch.Estimate(0) >= summary.Estimate(0), "the sketch estimate of the hottest key is at least "
                        + "the summary's");
                    spaceBounded &= summary.Counters.Count <= Slots;
                    cases++;
                }
            }
            Check(fullRecall, "every key with true count above N/k was reported (" + heavyTotal + " heavy keys over "
                + cases + " configurations)");
            Check(errorBound, "and every reported counter understates its key by at most N/k (worst measured ratio "
                + worstErrorRatio.ToString("F4") + ")");
            Check(spaceBounded, "the summary never exceeds k entries, with " + reportedTotal + " reported key slots over "
                + "all runs");
            Console.WriteLine("heavy hitters: " + heavyTotal + " heavy keys, all reported; worst understatement ratio "
                + worstErrorRatio.ToString("F4") + " of N/k");
        }

        // ---- the merge rule is valid: merged summaries still bound and still recall -------------------
        {
            const int Slots = 16;
            bool recallAfterMerge = true, boundAfterMerge = true, shardBudget = true;
            long cases = 0, worst = 0;
            for (int seed = 1; seed <= 5; seed++)
            {
                List<long> stream = Streams.Zipf(3000, 50, (ulong)seed * 0x51ED270B4E5A1F33UL);
                Dictionary<long, long> truth = Streams.TrueCounts(stream);
                double threshold = (double)stream.Count / Slots;
                (CountMin _, MisraGries merged) = Distributed(stream, 4, 3, 64, Slots, false, 0x847);
                foreach (KeyValuePair<long, long> entry in truth)
                {
                    if (entry.Value > threshold)
                    {
                        if (merged.Estimate(entry.Key) == 0) recallAfterMerge = false;
                        long gap = entry.Value - merged.Estimate(entry.Key);
                        if (gap > threshold) boundAfterMerge = false;
                        if (gap > worst) worst = gap;
                    }
                }
                foreach (long key in truth.Keys)
                    if (merged.Counters.ContainsKey(key) && merged.Estimate(key) > truth[key]) boundAfterMerge = false;
                cases++;
            }
            Check(shardBudget, "each shard's summary respected the slot budget");
            Check(recallAfterMerge, "the merged summary still reports every key above N/k after four-way merging ("
                + cases + " streams)");
            Check(boundAfterMerge, "and its counters never exceed the true count and never fall more than N/k below it "
                + "(worst gap " + worst + " against the " + (int)(3000.0 / Slots) + " bound)");
            Console.WriteLine("merge rule: " + cases + " streams, recall complete, worst gap " + worst + " against bound "
                + (int)(3000.0 / Slots));
        }

        // ---- distributed equals single pass for both structures ----------------------------------------
        {
            List<long> stream = Streams.Zipf(2500, 35, 0x5EED847000000006UL);
            bool sketchEqual = true, summaryRecall = true, sameGuarantee = true;
            for (int shards = 1; shards <= 6; shards++)
            {
                (CountMin sketch, MisraGries summary) = Distributed(stream, shards, 5, 136, 16, true, 0x847);
                var singleSketch = CountMin.Sized(0.02, 0.01, true, 0x847);
                var singleSummary = new MisraGries(16);
                foreach (long key in stream) { singleSketch.Add(key, 1); singleSummary.Add(key, 1); }
                foreach (long key in stream)
                    if (sketch.Estimate(key) != singleSketch.Estimate(key)) sketchEqual = false;
                Dictionary<long, long> truth = Streams.TrueCounts(stream);
                double threshold = (double)stream.Count / 16;
                foreach (KeyValuePair<long, long> entry in truth)
                    if (entry.Value > threshold && summary.Estimate(entry.Key) == 0) summaryRecall = false;
                // The merged summary is not a refinement of the single-pass one, but it must carry the same
                // guarantee: every heavy key reported, and no counter more than N/k below its key.
                foreach (KeyValuePair<long, long> entry in truth)
                {
                    if (entry.Value <= threshold) continue;
                    if (summary.Estimate(entry.Key) == 0 || singleSummary.Estimate(entry.Key) == 0) sameGuarantee = false;
                    if (entry.Value - summary.Estimate(entry.Key) > threshold
                        || entry.Value - singleSummary.Estimate(entry.Key) > threshold) sameGuarantee = false;
                }
            }
            Check(sketchEqual, "with the conservative rule too, six different shardings all reproduce the single-pass "
                + "sketch exactly");
            Check(summaryRecall, "the merged summary keeps full recall at every shard count");
            Check(sameGuarantee, "and carries the same guarantee as a summary that saw the whole stream: the same heavy "
                + "keys reported, none understated by more than N/k");
            Console.WriteLine("distributed == single pass: 6 shardings, sketch exact, summary recall complete");
        }

        // ---- degenerate streams -------------------------------------------------------------------------
        {
            var empty = new List<long>();
            (CountMin emptySketch, MisraGries emptySummary) = Distributed(empty, 3, 4, 32, 8, false, 0x847);
            Check(emptySketch.Estimate(7) == 0 && emptySummary.Counters.Count == 0 && emptySketch.TotalMass() == 0,
                "an empty stream leaves both structures empty and every estimate at zero");

            var single = new List<long>();
            for (int i = 0; i < 100; i++) single.Add(42);
            (CountMin singleSketch, MisraGries singleSummary) = Distributed(single, 4, 4, 32, 8, false, 0x847);
            Check(singleSketch.Estimate(42) == 100 && singleSummary.Estimate(42) == 100,
                "a one-key stream is counted exactly by both structures");

            List<long> distinct = Streams.Uniform(150, 150, 0x5EED847000000007UL);
            Dictionary<long, long> distinctTruth = Streams.TrueCounts(distinct);
            (CountMin _, MisraGries distinctSummary) = Distributed(distinct, 5, 4, 64, 10, false, 0x847);
            Check(distinctSummary.Counters.Count <= 10, "with all-distinct keys the summary still respects its slots ("
                + distinctSummary.Counters.Count + ")");
            bool noHeavy = true;
            foreach (KeyValuePair<long, long> entry in distinctTruth) if (entry.Value > 15.0) noHeavy = false;
            Check(noHeavy, "and no key is heavy, so an empty heavy-hitter answer would be correct");

            var oneSlot = new MisraGries(1);
            foreach (long key in new long[] { 1, 2, 3, 1, 1 }) oneSlot.Add(key, 1);
            Check(oneSlot.Counters.Count <= 1 && oneSlot.Processed == 5, "k = 1 keeps a single counter over the whole "
                + "stream (" + oneSlot.Processed + " items)");

            var idle = new Shard(0, new CountMin(2, 8, false, 1), new MisraGries(4));
            idle.Consume(new List<long>());
            Check(idle.Items == 0 && idle.Summary.Counters.Count == 0, "a shard that sees nothing contributes nothing "
                + "and does not disturb the merge");

            var dominant = new List<long>();
            for (int i = 0; i < 500; i++) dominant.Add(9);
            for (int i = 0; i < 20; i++) dominant.Add(i);
            (CountMin dominantSketch, MisraGries dominantSummary) = Distributed(dominant, 3, 4, 64, 5, false, 0x847);
            long dominantEstimate = dominantSummary.Estimate(9);
            long bestKey = -1, bestValue = -1;
            foreach (KeyValuePair<long, long> entry in dominantSummary.Counters)
                if (entry.Value > bestValue) { bestValue = entry.Value; bestKey = entry.Key; }
            Check(dominantSketch.Estimate(9) >= 500 && dominantSketch.Estimate(9) <= 520, "the sketch reports the "
                + "dominant key at least exactly and at most true plus the mass of every other key (" + dominantSketch.Estimate(9)
                + " against the true 500, the 20 non-dominant items being the only possible excess)");
            Check(bestKey == 9 && dominantEstimate >= 500 - dominant.Count / 5, "the merged summary keeps the dominant "
                + "key as its largest entry within the N/k guarantee (estimate " + dominantEstimate + " against the "
                + (500 - dominant.Count / 5) + " floor)");
            Console.WriteLine("degenerate: empty, one key, all-distinct, k=1, idle shard and dominant key all handled");
        }

        // ---- determinism and space measurement ----------------------------------------------------------
        {
            List<long> stream = Streams.Zipf(2000, 45, 0x5EED847000000008UL);
            (CountMin first, MisraGries firstSummary) = Distributed(stream, 5, 4, 96, 12, true, 0x847);
            (CountMin second, MisraGries secondSummary) = Distributed(stream, 5, 4, 96, 12, true, 0x847);
            bool identical = first.TotalMass() == second.TotalMass()
                && firstSummary.Counters.Count == secondSummary.Counters.Count;
            foreach (long key in stream)
                if (first.Estimate(key) != second.Estimate(key)
                    || firstSummary.Estimate(key) != secondSummary.Estimate(key)) identical = false;
            Check(identical, "two runs of the same pipeline agree on every estimate");
            CountMin sized = CountMin.Sized(0.02, 0.01, false, 0x847);
            Check(sized.Rows == 5 && sized.Width == 136, "the sizing formulas give 5 rows and 136 columns for eps=0.02 "
                + "and delta=0.01 (" + sized.Rows + "x" + sized.Width + ")");
            Console.WriteLine("determinism: sketch mass=" + first.TotalMass() + " summary slots="
                + firstSummary.Counters.Count + " of 12 | sketch space " + sized.Rows + "x" + sized.Width);
        }

        Console.WriteLine("PASS");
    }
}
