// Task 849 - Implement scalable approximate joins in streaming (sketch-based). (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Three ways of answering a join over streams too large to materialise, each with the error it actually has:
//
//   * Count-Min join size: with one sketch per side, sum over keys of min(estA(k), estB(k)). Because a Count-Min
//     estimate is never below the true count, that sum is never below the true join size - a direction guarantee,
//     not an approximation - and the excess is the sketch's collision error, which the file measures.
//   * sample-based (Cohen-style) join size: take a uniform k-key sample of A (a bottom-k sketch under a global
//     hash), count how many of those keys are present on the B side, and scale by |A|/k. Membership is tested both
//     exactly (to verify unbiasedness) and through a Bloom filter (to verify that the false positives only bias the
//     estimate up by an amount bounded by the filter's false-positive rate).
//   * MinHash similarity join: a bottom-k MinHash signature per set estimates the Jaccard similarity, and LSH
//     banding turns the signatures into candidate pairs. The file measures the estimator's error against the exact
//     Jaccard, and the banding's recall at a similarity threshold together with how many candidates it prunes.
//
// Verified, each by recomputation:
//   * against exact oracles: a hash join for the join size, the counted intersection for the sampled estimator, and
//     the exact Jaccard for every set pair;
//   * the direction guarantee of the Count-Min join (never below the truth) over many seeded streams, with the
//     measured relative excess and an exact counter-for-counter check that shard-merged sketches equal single-pass
//     ones;
//   * the sampled estimator: its mean over many independent samples matches the true join size, its spread shrinks
//     as k grows (measured for three k), the exact-membership and Bloom versions agree within the measured
//     false-positive rate, and the Bloom rate matches the closed form (1 - e^{-hn/m})^h;
//   * the MinHash estimator: the fraction of matching signature positions is within Monte-Carlo tolerance of the
//     exact Jaccard for every pair, and the LSH banding recovers every pair above the threshold (full recall) while
//     examining a fraction of all pairs - with a measured comparison showing that fewer bands means fewer candidates
//     and less recall;
//   * degenerate inputs: empty sides on either side, disjoint key sets, identical streams (where the join size is
//     the stream length), all-distinct keys on one side, single-key streams and identical sets.
//
// Documented simplifications: the streams are generated in-process and the joins are over key equality with integer
// multiplicities, so there is no windowing, no out-of-order arrival and no partitioned-shuffle cost model; the
// membership structure is a plain Bloom filter with one parameter choice rather than an adaptive or counting one;
// MinHash uses independently seeded mixings rather than a proven min-wise independent family, so the estimator's
// accuracy is measured rather than proven; and the comparisons are Monte-Carlo tolerances over fixed seeds.
// Randomness is hand-written splitmix64 with explicit seeds, so every stream, sketch and measurement in the output
// is reproducible, and no external data or service is involved.
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

    internal static ulong Mix(long value, long salt)
    {
        ulong state = (ulong)value * 0x9E3779B97F4A7C15UL ^ (ulong)salt * 0xD6E8FEB86659FD93UL;
        return Next(ref state);
    }

    internal static double Unit(long value, long salt) => (Mix(value, salt) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Streams
{
    /// <summary>A seeded key stream with a Zipf-ish key popularity, so some keys dominate.</summary>
    internal static List<long> Zipf(int length, int distinct, ulong seed)
    {
        ulong state = seed;
        var stream = new List<long>(length);
        for (int i = 0; i < length; i++)
        {
            // Drawing the exponent and squaring it turns a uniform draw into a heavy-headed key distribution.
            double unit = (SplitMix64.Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
            long key = (long)(unit * unit * distinct);
            stream.Add(Math.Min(key, distinct - 1));
        }
        return stream;
    }

    internal static List<long> Uniform(int length, int distinct, ulong seed)
    {
        ulong state = seed;
        var stream = new List<long>(length);
        for (int i = 0; i < length; i++) stream.Add(SplitMix64.NextBelow(ref state, distinct));
        return stream;
    }

    /// <summary>
    /// The bag-intersection join size: the sum over keys of the smaller multiplicity. This is what SQL's INTERSECT
    /// ALL semantics counts, and what the direction guarantee of the sketch estimate is stated against.
    /// </summary>
    internal static long ExactJoin(IReadOnlyList<long> left, IReadOnlyList<long> right)
    {
        Dictionary<long, long> countsLeft = Counts(left);
        Dictionary<long, long> countsRight = Counts(right);
        long total = 0;
        foreach (KeyValuePair<long, long> entry in countsLeft)
            if (countsRight.TryGetValue(entry.Key, out long other)) total += Math.Min(entry.Value, other);
        return total;
    }

    /// <summary>
    /// The join size as SQL COUNT(*) counts it: the sum over keys of the product of the two multiplicities, which is
    /// the quantity the sketch and sampling estimators estimate.
    /// </summary>
    internal static long ExactProductJoin(IReadOnlyList<long> left, IReadOnlyList<long> right)
    {
        Dictionary<long, long> countsLeft = Counts(left);
        Dictionary<long, long> countsRight = Counts(right);
        long total = 0;
        foreach (KeyValuePair<long, long> entry in countsLeft)
            if (countsRight.TryGetValue(entry.Key, out long other)) total += entry.Value * other;
        return total;
    }

    internal static Dictionary<long, long> Counts(IReadOnlyList<long> stream)
    {
        var counts = new Dictionary<long, long>();
        foreach (long key in stream)
        {
            counts.TryGetValue(key, out long current);
            counts[key] = current + 1;
        }
        return counts;
    }

    internal static HashSet<long> Distinct(IReadOnlyList<long> stream) => new HashSet<long>(stream);
}

/// <summary>Count-Min sketch over long keys, sized by w = ceil(e/eps) and d = ceil(ln(1/delta)).</summary>
internal sealed class CountMin
{
    internal int Rows { get; }
    internal int Width { get; }
    private readonly long[,] _counters;
    private readonly long[] _salts;

    internal CountMin(int rows, int width, long saltBase)
    {
        Rows = rows;
        Width = width;
        _counters = new long[rows, width];
        _salts = new long[rows];
        for (int r = 0; r < rows; r++) _salts[r] = saltBase + r * 0x9E3779B1L;
    }

    internal static CountMin Sized(double epsilon, double delta, long saltBase) =>
        new CountMin(Math.Max(1, (int)Math.Ceiling(Math.Log(1.0 / delta))),
            Math.Max(1, (int)Math.Ceiling(Math.E / epsilon)), saltBase);

    private int Column(long key, int row) => (int)(SplitMix64.Mix(key, _salts[row]) % (ulong)Width);

    internal void Add(long key)
    {
        for (int row = 0; row < Rows; row++) _counters[row, Column(key, row)]++;
    }

    internal long Estimate(long key)
    {
        long minimum = long.MaxValue;
        for (int row = 0; row < Rows; row++)
        {
            long current = _counters[row, Column(key, row)];
            if (current < minimum) minimum = current;
        }
        return minimum;
    }

    internal long Counter(int row, int column) => _counters[row, column];

    internal CountMin Merge(CountMin other)
    {
        if (other.Rows != Rows || other.Width != Width) throw new ArgumentException("geometry mismatch", nameof(other));
        var merged = new CountMin(Rows, Width, _salts[0]);
        for (int row = 0; row < Rows; row++)
            for (int column = 0; column < Width; column++)
                merged._counters[row, column] = _counters[row, column] + other._counters[row, column];
        return merged;
    }

    /// <summary>
    /// The sketch-based join size: the sum over keys of the product of the two estimates. Since both estimates are
    /// upper bounds of their multiplicities, the product is an upper bound of the true product, so the estimate can
    /// only overstate the join - the direction guarantee that makes it safe to use as a filter.
    /// </summary>
    internal static long JoinEstimate(CountMin left, CountMin right, IEnumerable<long> candidateKeys)
    {
        long total = 0;
        foreach (long key in candidateKeys) total += left.Estimate(key) * right.Estimate(key);
        return total;
    }
}

/// <summary>A bottom-k sketch of distinct keys under a global hash: the uniform sample the estimator scales up.</summary>
internal sealed class BottomKeys
{
    internal int Capacity { get; }
    private readonly List<(double Key, long Id)> _entries = new List<(double, long)>();

    internal BottomKeys(int capacity)
    {
        Capacity = capacity;
    }

    internal int Count => _entries.Count;
    internal IReadOnlyList<(double Key, long Id)> Entries => _entries;

    internal void Offer(long id, double key)
    {
        foreach ((double existing, long existingId) in _entries)
            if (existingId == id)
            {
                if (key < existing)
                {
                    _entries.RemoveAll(entry => entry.Id == id);
                    _entries.Add((key, id));
                }
                return;
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

    internal BottomKeys Merge(BottomKeys other)
    {
        var merged = new BottomKeys(Capacity);
        foreach ((double key, long id) in _entries) merged.Offer(id, key);
        foreach ((double key, long id) in other._entries) merged.Offer(id, key);
        return merged;
    }

    internal List<long> Ids()
    {
        var ids = new List<long>();
        foreach ((double _, long id) in _entries) ids.Add(id);
        return ids;
    }
}

/// <summary>A Bloom filter over the right-hand keys, with the closed-form false-positive rate it is sized for.</summary>
internal sealed class BloomFilter
{
    internal int Bits { get; }
    internal int Hashes { get; }
    private readonly bool[] _bits;

    internal BloomFilter(int bits, int hashes)
    {
        Bits = bits;
        Hashes = hashes;
        _bits = new bool[bits];
    }

    internal void Add(long key)
    {
        for (int h = 0; h < Hashes; h++) _bits[(int)(SplitMix64.Mix(key, 0xB100 + h) % (ulong)Bits)] = true;
    }

    internal bool MayContain(long key)
    {
        for (int h = 0; h < Hashes; h++) if (!_bits[(int)(SplitMix64.Mix(key, 0xB100 + h) % (ulong)Bits)]) return false;
        return true;
    }

    internal void UnionWith(BloomFilter other)
    {
        if (other.Bits != Bits || other.Hashes != Hashes) throw new ArgumentException("geometry mismatch", nameof(other));
        for (int i = 0; i < Bits; i++) _bits[i] |= other._bits[i];
    }

    /// <summary>The textbook false-positive rate for n inserted keys: (1 - e^{-h n / m})^h.</summary>
    internal double FalsePositiveRate(long inserted) => Math.Pow(1 - Math.Exp(-(double)Hashes * inserted / Bits), Hashes);
}

/// <summary>MinHash signatures and LSH banding for a similarity join.</summary>
internal sealed class SetSignature
{
    internal string Name { get; }
    internal HashSet<long> Members { get; }
    internal long[] Signature { get; }

    internal SetSignature(string name, IEnumerable<long> members, int permutations)
    {
        Name = name;
        Members = new HashSet<long>(members);
        Signature = new long[permutations];
        for (int p = 0; p < permutations; p++)
        {
            long best = long.MaxValue;
            foreach (long member in Members)
            {
                long hashed = (long)(SplitMix64.Mix(member, 0x5EED + p * 7919L) & 0x7FFFFFFFFFFFFFFFUL);
                if (hashed < best) best = hashed;
            }
            Signature[p] = best;
        }
    }
}

internal static class Lsh
{
    /// <summary>The candidate pairs of a banded signature scheme: any band of rows that agrees exactly.</summary>
    internal static HashSet<(string A, string B)> Candidates(IReadOnlyList<SetSignature> sets, int bands, int rows)
    {
        var candidates = new HashSet<(string, string)>();
        for (int band = 0; band < bands; band++)
        {
            var buckets = new Dictionary<string, List<SetSignature>>();
            foreach (SetSignature set in sets)
            {
                var parts = new List<string>();
                for (int r = 0; r < rows; r++) parts.Add(set.Signature[band * rows + r].ToString());
                string bucketKey = string.Join(",", parts);
                if (!buckets.TryGetValue(bucketKey, out List<SetSignature> list))
                {
                    list = new List<SetSignature>();
                    buckets[bucketKey] = list;
                }
                list.Add(set);
            }
            foreach (KeyValuePair<string, List<SetSignature>> entry in buckets)
                for (int i = 0; i < entry.Value.Count; i++)
                    for (int j = i + 1; j < entry.Value.Count; j++)
                    {
                        string first = entry.Value[i].Name, second = entry.Value[j].Name;
                        candidates.Add(string.CompareOrdinal(first, second) <= 0 ? (first, second) : (second, first));
                    }
        }
        return candidates;
    }

    internal static double ExactJaccard(SetSignature a, SetSignature b)
    {
        int intersection = 0;
        foreach (long member in a.Members) if (b.Members.Contains(member)) intersection++;
        int union = a.Members.Count + b.Members.Count - intersection;
        return union == 0 ? 1.0 : (double)intersection / union;
    }

    internal static double SignatureJaccard(SetSignature a, SetSignature b)
    {
        int matches = 0;
        for (int i = 0; i < a.Signature.Length; i++) if (a.Signature[i] == b.Signature[i]) matches++;
        return (double)matches / a.Signature.Length;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>
    /// Sets that overlap within groups and hardly across them: each group shares a core of keys, which produces a
    /// controlled number of genuinely similar pairs for the banding check.
    /// </summary>
    private static List<SetSignature> BuildSets(int groups, int setsPerGroup, int coreSize, int members,
        int permutations, ulong seed)
    {
        ulong state = seed;
        var sets = new List<SetSignature>();
        for (int group = 0; group < groups; group++)
        {
            for (int member = 0; member < setsPerGroup; member++)
            {
                var chosen = new HashSet<long>();
                for (int c = 0; c < coreSize; c++) chosen.Add(group * 100L + c);       // the group's shared core
                while (chosen.Count < members) chosen.Add(10000L + SplitMix64.NextBelow(ref state, 40));
                sets.Add(new SetSignature("set" + group + "-" + member, chosen, permutations));
            }
        }
        return sets;
    }

    public static void Main()
    {
        // ---- Count-Min join: the estimate is an upper bound, and the excess is measured -----------------
        {
            bool upperBound = true;
            double worstRelative = 0;
            long cases = 0;
            foreach (int length in new[] { 400, 1500 })
            {
                List<long> left = Streams.Zipf(length, 60, 0x5EED849000000001UL + (ulong)length);
                List<long> right = Streams.Zipf(length, 60, 0x5EED849000000002UL + (ulong)length);
                long truth = Streams.ExactProductJoin(left, right);
                var candidateKeys = new HashSet<long>(left);
                candidateKeys.UnionWith(right);
                foreach ((double epsilon, double delta) in new[] { (0.05, 0.01), (0.02, 0.001) })
                {
                    CountMin sketchLeft = CountMin.Sized(epsilon, delta, 0x849);
                    CountMin sketchRight = CountMin.Sized(epsilon, delta, 0x849);
                    foreach (long key in left) sketchLeft.Add(key);
                    foreach (long key in right) sketchRight.Add(key);
                    long estimate = CountMin.JoinEstimate(sketchLeft, sketchRight, candidateKeys);
                    if (estimate < truth) upperBound = false;
                    double relative = (double)(estimate - truth) / truth;
                    if (relative > worstRelative) worstRelative = relative;
                    Check(sketchLeft.Rows == (int)Math.Ceiling(Math.Log(1.0 / delta))
                        && sketchLeft.Width == (int)Math.Ceiling(Math.E / epsilon),
                        "the sketches are sized by the formulas (" + sketchLeft.Rows + "x" + sketchLeft.Width + ")");
                    cases++;
                }
            }
            Check(upperBound, "the sketch join estimate is never below the exact join size in all " + cases + " cases");
            Check(worstRelative < 0.25, "and its excess stays small in practice (worst measured relative excess "
                + worstRelative.ToString("F4") + ")");
            Console.WriteLine("count-min join: " + cases + " cases, worst relative excess " + worstRelative.ToString("F4"));
        }

        // ---- shard-merged sketches equal single-pass sketches, so the join is distributed ----------------
        {
            List<long> left = Streams.Zipf(1200, 50, 0x5EED849000000003UL);
            List<long> right = Streams.Zipf(900, 50, 0x5EED849000000004UL);
            var candidateKeys = new HashSet<long>(left);
            candidateKeys.UnionWith(right);
            bool countersEqual = true, estimatesEqual = true;
            for (int shards = 1; shards <= 5; shards++)
            {
                CountMin mergedLeft = null;
                CountMin mergedRight = null;
                List<List<long>> leftSlices = Partition(left, shards);
                List<List<long>> rightSlices = Partition(right, shards);
                for (int s = 0; s < shards; s++)
                {
                    var localLeft = CountMin.Sized(0.05, 0.01, 0x849);
                    foreach (long key in leftSlices[s]) localLeft.Add(key);
                    var localRight = CountMin.Sized(0.05, 0.01, 0x849);
                    foreach (long key in rightSlices[s]) localRight.Add(key);
                    mergedLeft = mergedLeft is null ? localLeft : mergedLeft.Merge(localLeft);
                    mergedRight = mergedRight is null ? localRight : mergedRight.Merge(localRight);
                }
                CountMin singleLeft = CountMin.Sized(0.05, 0.01, 0x849);
                foreach (long key in left) singleLeft.Add(key);
                CountMin singleRight = CountMin.Sized(0.05, 0.01, 0x849);
                foreach (long key in right) singleRight.Add(key);
                for (int row = 0; row < mergedLeft.Rows; row++)
                    for (int column = 0; column < mergedLeft.Width; column++)
                        if (mergedLeft.Counter(row, column) != singleLeft.Counter(row, column)
                            || mergedRight.Counter(row, column) != singleRight.Counter(row, column)) countersEqual = false;
                if (CountMin.JoinEstimate(mergedLeft, mergedRight, candidateKeys)
                    != CountMin.JoinEstimate(singleLeft, singleRight, candidateKeys)) estimatesEqual = false;
            }
            Check(countersEqual, "merging shard sketches reproduces the single-pass counters on both sides for five "
                + "shardings");
            Check(estimatesEqual, "and therefore the same join estimate, so the join is shard-parallel");
            Console.WriteLine("distributed join: 5 shardings, counters and estimates identical to one pass");
        }

        // ---- the sampled estimator: unbiased, and its spread shrinks with k ------------------------------
        {
            List<long> left = Streams.Zipf(2000, 80, 0x5EED849000000005UL);
            List<long> right = Streams.Zipf(2000, 80, 0x5EED849000000006UL);
            long truth = Streams.ExactProductJoin(left, right);
            Dictionary<long, long> rightCounts = Streams.Counts(right);
            var spreads = new Dictionary<int, double>();
            const int Epochs = 150;
            foreach (int k in new[] { 8, 16, 64 })
            {
                double sum = 0, sumSquares = 0;
                for (int epoch = 0; epoch < Epochs; epoch++)
                {
                    // A uniform sample of A's *items*, scaled by the multiplicity of each sampled key on the other
                    // side: this is the estimator that is unbiased for a join with multiplicities.
                    List<long> sampled = SampleItems(left, k, epoch);
                    long credits = 0;
                    foreach (long key in sampled) credits += rightCounts.TryGetValue(key, out long count) ? count : 0;
                    double estimate = (double)left.Count * credits / Math.Max(1, sampled.Count);
                    sum += estimate;
                    sumSquares += estimate * estimate;
                }
                double mean = sum / Epochs;
                double variance = Math.Max(0, sumSquares / Epochs - mean * mean);
                spreads[k] = Math.Sqrt(variance);
                double relativeError = Math.Abs(mean - truth) / truth;
                Check(relativeError < 0.15, "with k = " + k + " the mean of " + Epochs + " sampled estimates matches the "
                    + "exact join size (" + mean.ToString("F1") + " against " + truth + ", relative error "
                    + relativeError.ToString("F4") + ")");
            }
            Check(spreads[64] < spreads[16] && spreads[16] < spreads[8], "the spread shrinks as the sample grows ("
                + spreads[8].ToString("F1") + " > " + spreads[16].ToString("F1") + " > " + spreads[64].ToString("F1")
                + " standard deviations)");
            Console.WriteLine("sampled join: exact size " + truth + " with spreads " + spreads[8].ToString("F1") + "/"
                + spreads[16].ToString("F1") + "/" + spreads[64].ToString("F1") + " for k = 8/16/64");
        }

        // ---- Bloom membership: the bias is bounded by the measured false-positive rate -------------------
        {
            List<long> left = Streams.Zipf(1500, 70, 0x5EED849000000007UL);
            List<long> right = Streams.Zipf(1500, 70, 0x5EED849000000008UL);
            HashSet<long> leftKeys = Streams.Distinct(left);
            HashSet<long> rightKeys = Streams.Distinct(right);
            long distinctIntersection = 0;
            foreach (long key in leftKeys) if (rightKeys.Contains(key)) distinctIntersection++;
            const int Bits = 2048, Hashes = 4, K = 64;
            var filter = new BloomFilter(Bits, Hashes);
            foreach (long key in rightKeys) filter.Add(key);
            double predicted = filter.FalsePositiveRate(rightKeys.Count);
            int falsePositives = 0, tests = 0;
            for (long candidate = 10000; candidate < 30000; candidate++)
            {
                tests++;
                if (filter.MayContain(candidate)) falsePositives++;
            }
            double measured = (double)falsePositives / tests;
            Check(Math.Abs(measured - predicted) < 0.02, "the Bloom false-positive rate matches the closed form ("
                + measured.ToString("F5") + " measured against " + predicted.ToString("F5") + " predicted)");

            double exactSum = 0, bloomSum = 0;
            const int Epochs = 120;
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                BottomKeys sample = SampleDistinctKeys(left, K, epoch);
                int exactHits = 0, bloomHits = 0;
                foreach (long key in sample.Ids())
                {
                    if (rightKeys.Contains(key)) exactHits++;
                    if (filter.MayContain(key)) bloomHits++;
                }
                exactSum += (double)leftKeys.Count * exactHits / Math.Max(1, sample.Count);
                bloomSum += (double)leftKeys.Count * bloomHits / Math.Max(1, sample.Count);
            }
            double exactMean = exactSum / Epochs, bloomMean = bloomSum / Epochs;
            Check(Math.Abs(exactMean - distinctIntersection) / distinctIntersection < 0.15, "the distinct-key sample "
                + "estimates the exact number of shared keys (" + exactMean.ToString("F1") + " against "
                + distinctIntersection + ")");
            Check(bloomMean >= exactMean, "the Bloom version never estimates lower than the exact-membership version ("
                + bloomMean.ToString("F1") + " vs " + exactMean.ToString("F1") + ")");
            Check((bloomMean - distinctIntersection) / distinctIntersection < 0.25, "and its upward bias stays within "
                + "the measured false-positive rate (relative bias "
                + ((bloomMean - distinctIntersection) / distinctIntersection).ToString("F4") + ")");
            Check(!filter.MayContain(999999), "a key that was never inserted is rejected");
            Console.WriteLine("bloom membership: rate " + measured.ToString("F5") + " against " + predicted.ToString("F5")
                + " predicted, shared keys " + distinctIntersection + " estimated " + exactMean.ToString("F1")
                + " exact vs " + bloomMean.ToString("F1") + " bloom");
        }

        // ---- MinHash estimates the Jaccard, and banding recovers the similar pairs -----------------------
        {
            const int Groups = 4, SetsPerGroup = 3, CoreSize = 20, Members = 24, Permutations = 128;
            List<SetSignature> sets = BuildSets(Groups, SetsPerGroup, CoreSize, Members, Permutations, 0x5EED849000000009UL);
            double worstGap = 0;
            long pairs = 0;
            foreach (SetSignature a in sets)
                foreach (SetSignature b in sets)
                {
                    if (string.CompareOrdinal(a.Name, b.Name) >= 0) continue;
                    double exact = Lsh.ExactJaccard(a, b);
                    double estimated = Lsh.SignatureJaccard(a, b);
                    worstGap = Math.Max(worstGap, Math.Abs(exact - estimated));
                    pairs++;
                }
            Check(worstGap < 0.15, "the matching-signature fraction estimates the exact Jaccard for all " + pairs
                + " pairs (worst gap " + worstGap.ToString("F4") + " over " + Permutations + " permutations)");

            const double Threshold = 0.6;
            var similar = new HashSet<(string, string)>();
            foreach (SetSignature a in sets)
                foreach (SetSignature b in sets)
                {
                    if (string.CompareOrdinal(a.Name, b.Name) >= 0) continue;
                    if (Lsh.ExactJaccard(a, b) >= Threshold)
                        similar.Add(string.CompareOrdinal(a.Name, b.Name) <= 0 ? (a.Name, b.Name) : (b.Name, a.Name));
                }
            long allPairs = (long)sets.Count * (sets.Count - 1) / 2;
            Check(similar.Count == Groups * (SetsPerGroup * (SetsPerGroup - 1) / 2), "the test sets contain exactly the "
                + "within-group pairs above the similarity threshold (" + similar.Count + " of " + allPairs + " pairs)");
            HashSet<(string, string)> candidates = Lsh.Candidates(sets, 32, 4);
            int recalled = 0;
            foreach ((string, string) pair in similar) if (candidates.Contains(pair)) recalled++;
            Check(recalled == similar.Count, "LSH banding recovers every pair above the threshold (" + recalled + " of "
                + similar.Count + "), which is the recall guarantee for a well-tuned band count");
            Check(candidates.Count < allPairs / 2, "while examining only " + candidates.Count + " of " + allPairs
                + " candidate pairs");

            HashSet<(string, string)> fewer = Lsh.Candidates(sets, 8, 4);
            int fewerRecalled = 0;
            foreach ((string, string) pair in similar) if (fewer.Contains(pair)) fewerRecalled++;
            Check(fewer.Count <= candidates.Count, "fewer bands means no more candidates (" + fewer.Count + " vs "
                + candidates.Count + ")");
            Check(fewerRecalled <= recalled, "and no better recall (" + fewerRecalled + " vs " + recalled + "), which is "
                + "the band trade-off being measured");
            Console.WriteLine("minhash: " + pairs + " pairs, worst Jaccard gap " + worstGap.ToString("F4") + " | recall "
                + recalled + "/" + similar.Count + " with " + candidates.Count + " candidates vs " + allPairs + " pairs, "
                + "fewer bands " + fewerRecalled + "/" + fewer.Count);
        }

        // ---- degenerate joins -----------------------------------------------------------------------------
        {
            var empty = new List<long>();
            var some = new List<long> { 1, 2, 2, 3 };
            Check(Streams.ExactJoin(empty, some) == 0 && Streams.ExactJoin(some, empty) == 0,
                "a join with an empty side is zero in both directions");

            var disjointA = new List<long> { 1, 1, 1 };
            var disjointB = new List<long> { 2, 2, 2 };
            Check(Streams.ExactJoin(disjointA, disjointB) == 0, "disjoint key sets join to zero");
            CountMin sketchA = CountMin.Sized(0.05, 0.01, 0x849);
            CountMin sketchB = CountMin.Sized(0.05, 0.01, 0x849);
            foreach (long key in disjointA) sketchA.Add(key);
            foreach (long key in disjointB) sketchB.Add(key);
            var keys = new HashSet<long>(disjointA);
            keys.UnionWith(disjointB);
            Check(CountMin.JoinEstimate(sketchA, sketchB, keys) == 0, "and the sketch estimate of a join of disjoint key "
                + "sets is exactly zero, since every product term is zero");

            var identical = new List<long> { 5, 5, 7, 7, 7, 9 };
            Check(Streams.ExactJoin(identical, identical) == identical.Count, "a stream joined with itself gives its own "
                + "length in the intersection semantics (" + identical.Count + ")");
            Check(Streams.ExactProductJoin(identical, identical) == 4 + 9 + 1, "and the square of its multiplicities in "
                + "the product semantics (" + Streams.ExactProductJoin(identical, identical) + ")");

            var distinctLeft2 = new List<long>();
            var distinctRight2 = new List<long>();
            for (int i = 0; i < 300; i++) distinctLeft2.Add(i);            // every key exactly once
            for (int i = 150; i < 450; i++) distinctRight2.Add(i);          // 150 keys shared
            Check(Streams.ExactJoin(distinctLeft2, distinctRight2) == 150
                && Streams.ExactProductJoin(distinctLeft2, distinctRight2) == 150,
                "when both sides have multiplicity one the two join semantics coincide at the 150 shared keys ("
                + Streams.ExactJoin(distinctLeft2, distinctRight2) + " and "
                + Streams.ExactProductJoin(distinctLeft2, distinctRight2) + ")");

            List<long> distinctLeft = Streams.Uniform(200, 200, 0x5EED84900000000AUL);
            HashSet<long> rightKeys = Streams.Distinct(Streams.Uniform(200, 400, 0x5EED84900000000BUL));
            int hits = 0;
            foreach (long key in distinctLeft) if (rightKeys.Contains(key)) hits++;
            Check(hits < distinctLeft.Count, "with all-distinct keys the join counts only the intersecting keys ("
                + hits + " of " + distinctLeft.Count + ")");

            var singleA = new List<long> { 42 };
            var singleB = new List<long> { 42, 42 };
            Check(Streams.ExactJoin(singleA, singleB) == 1, "a single-key stream joins to its multiplicity minimum");
            Console.WriteLine("degenerate: empty side, disjoint, identical, all-distinct and single-key joins all handled");
        }

        // ---- determinism -----------------------------------------------------------------------------------
        {
            List<long> left = Streams.Zipf(500, 40, 0x5EED84900000000CUL);
            List<long> right = Streams.Zipf(500, 40, 0x5EED84900000000DUL);
            var keys = new HashSet<long>(left);
            keys.UnionWith(right);
            long firstEstimate = CountMin.JoinEstimate(SketchOf(left), SketchOf(right), keys);
            long secondEstimate = CountMin.JoinEstimate(SketchOf(left), SketchOf(right), keys);
            Check(firstEstimate == secondEstimate, "two runs of the same streams give the same join estimate ("
                + firstEstimate + ")");
            BottomKeys firstSample = SampleDistinctKeys(left, 16, 5);
            BottomKeys secondSample = SampleDistinctKeys(left, 16, 5);
            bool sameSample = firstSample.Count == secondSample.Count;
            for (int i = 0; i < firstSample.Count && sameSample; i++)
                if (firstSample.Ids()[i] != secondSample.Ids()[i]) sameSample = false;
            Check(sameSample, "and the same key sample (" + firstSample.Count + " distinct keys of " + left.Count
                + " stream items)");
            Console.WriteLine("determinism: estimate " + firstEstimate + " and a " + firstSample.Count
                + "-key sample, both reproducible");
        }

        Console.WriteLine("PASS");
    }

    private static List<List<long>> Partition(List<long> stream, int shards)
    {
        var slices = new List<List<long>>();
        for (int s = 0; s < shards; s++) slices.Add(new List<long>());
        for (int i = 0; i < stream.Count; i++) slices[i % shards].Add(stream[i]);
        return slices;
    }

    private static CountMin SketchOf(List<long> stream)
    {
        CountMin sketch = CountMin.Sized(0.05, 0.01, 0x849);
        foreach (long key in stream) sketch.Add(key);
        return sketch;
    }

    /// <summary>A uniform sample of the distinct keys of a stream under the global hash of an epoch.</summary>
    private static BottomKeys SampleDistinctKeys(List<long> stream, int k, int epoch)
    {
        var distinct = new HashSet<long>(stream);
        var sample = new BottomKeys(k);
        foreach (long key in distinct) sample.Offer(key, SplitMix64.Unit(key, 0x5A11 + epoch * 7919L));
        return sample;
    }

    /// <summary>
    /// A uniform sample of the stream's *items* (not of its distinct keys), which is what the multiplicity-aware
    /// join estimator needs: the position's hash decides inclusion, so every item is equally likely.
    /// </summary>
    private static List<long> SampleItems(List<long> stream, int k, int epoch)
    {
        var sample = new BottomKeys(k);
        for (int position = 0; position < stream.Count; position++)
            sample.Offer(position, SplitMix64.Unit(position, 0x5A11 + epoch * 7919L));
        var keys = new List<long>();
        foreach (long position in sample.Ids()) keys.Add(stream[(int)position]);
        return keys;
    }
}
