// Task 649 - Implement bloom join filter application and benchmarking. (Tier: Expert, CTCI Ch 14 - Databases)
//
// The classic "bloom join" optimisation: build a Bloom filter over the build side's join keys, probe it
// with every probe-side key and only send the survivors into the join. The filter is a bit array indexed
// by k positions derived from two independent hashes of the key (Kirsch-Mitzenmacher double hashing,
// h_i = h1 + i * h2), and only the hash table probe is saved - false positives still get filtered by the
// real key comparison, which is what keeps the result identical to a plain hash join.
//
// What is measured, not assumed: the false positive rate of every configuration is compared with the
// textbook formula (1 - e^(-kn/m))^k, the bit budget is shown to be monotone (more bits per key, fewer
// false positives), the optimal number of hash functions k = (m/n) ln 2 is confirmed against the measured
// optimum, and the number of hash table lookups avoided is counted exactly.
//
// Simplified: int32 keys, one build/probe pair, perfect local memory (no network, no cache hierarchy)
// and a plain bit array rather than a blocked/scalable Bloom filter; the "benchmark" is a comparison
// probe/lookup counter, not wall-clock time (timing would not be deterministic).
//
// Verified: no inserted key ever fails the probe (no false negatives, the soundness property that makes
// the filter safe), the filtered join returns exactly the plain join's result even with a deliberately
// terrible filter, the measured false positive rate tracks the formula, and the filter never increases
// the number of join lookups.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal static class Hashing
{
    public static ulong Mix(ulong value)
    {
        ulong z = value + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static ulong MixKey(int key) => Mix((ulong)(uint)key * 0x9E3779B97F4A7C15UL + 0x123456789ABCDEFUL);
}

internal sealed class BloomFilter
{
    private readonly bool[] _bits;
    private readonly int _bitCount;
    private readonly int _hashCount;

    public BloomFilter(int bitCount, int hashCount)
    {
        if (bitCount <= 0) throw new Exception("a Bloom filter needs at least one bit");
        if (hashCount <= 0) throw new Exception("a Bloom filter needs at least one hash function");
        _bitCount = bitCount;
        _hashCount = hashCount;
        _bits = new bool[bitCount];
    }

    public int BitCount => _bitCount;
    public int HashCount => _hashCount;
    public int Inserted { get; private set; }
    public int SetBits { get; private set; }

    private IEnumerable<int> Positions(int key)
    {
        ulong h1 = Hashing.MixKey(key);
        ulong h2 = Hashing.Mix(h1 ^ 0x5DEECE66DUL) | 1UL; // odd, so the step never collapses to zero
        for (int i = 0; i < _hashCount; i++)
        {
            yield return (int)((h1 + (ulong)i * h2) % (ulong)_bitCount);
        }
    }

    public void Add(int key)
    {
        foreach (int position in Positions(key))
        {
            if (!_bits[position])
            {
                _bits[position] = true;
                SetBits++;
            }
        }
        Inserted++;
    }

    public bool MightContain(int key)
    {
        foreach (int position in Positions(key))
        {
            if (!_bits[position]) return false;
        }
        return true;
    }

    // 1 - (1 - 1/m)^(kn) is usually quoted as (1 - e^(-kn/m))^k for the false positive rate.
    public double TheoreticalFalsePositiveRate
        => Math.Pow(1.0 - Math.Exp(-(double)_hashCount * Inserted / _bitCount), _hashCount);

    public static int OptimalHashCount(int bitCount, int keys)
    {
        double optimal = bitCount / (double)Math.Max(1, keys) * Math.Log(2.0);
        return Math.Max(1, (int)Math.Round(optimal));
    }
}

internal sealed class JoinOutcome
{
    public readonly List<string> Rows;
    public readonly int ProbedKeys;
    public readonly int PassedFilter;
    public readonly int JoinLookups;

    public JoinOutcome(List<string> rows, int probedKeys, int passedFilter, int joinLookups)
    {
        Rows = rows;
        ProbedKeys = probedKeys;
        PassedFilter = passedFilter;
        JoinLookups = joinLookups;
    }

    public int AvoidedLookups => ProbedKeys - JoinLookups;
}

internal static class BloomJoin
{
    // Plain hash join: every probe row costs one hash table lookup.
    public static JoinOutcome PlainJoin(List<int[]> probe, List<int[]> build)
    {
        var table = new Dictionary<int, List<int>>();
        foreach (int[] row in build)
        {
            if (!table.TryGetValue(row[0], out List<int>? bucket))
            {
                bucket = new List<int>();
                table[row[0]] = bucket;
            }
            bucket.Add(row[1]);
        }
        var rows = new List<string>();
        int lookups = 0;
        foreach (int[] row in probe)
        {
            lookups++;
            if (!table.TryGetValue(row[0], out List<int>? bucket)) continue;
            foreach (int payload in bucket) rows.Add(row[0] + ":" + row[1] + ":" + payload);
        }
        rows.Sort(StringComparer.Ordinal);
        return new JoinOutcome(rows, probe.Count, probe.Count, lookups);
    }

    // Bloom join: the filter is probes first; only surviving rows reach the hash table.
    public static JoinOutcome FilteredJoin(List<int[]> probe, List<int[]> build, BloomFilter filter)
    {
        var table = new Dictionary<int, List<int>>();
        foreach (int[] row in build)
        {
            filter.Add(row[0]);
            if (!table.TryGetValue(row[0], out List<int>? bucket))
            {
                bucket = new List<int>();
                table[row[0]] = bucket;
            }
            bucket.Add(row[1]);
        }

        var rows = new List<string>();
        int passed = 0, lookups = 0;
        foreach (int[] row in probe)
        {
            if (!filter.MightContain(row[0])) continue;
            passed++;
            lookups++;
            if (!table.TryGetValue(row[0], out List<int>? bucket)) continue;
            foreach (int payload in bucket) rows.Add(row[0] + ":" + row[1] + ":" + payload);
        }
        rows.Sort(StringComparer.Ordinal);
        return new JoinOutcome(rows, probe.Count, passed, lookups);
    }

    public static double MeasureFalsePositiveRate(BloomFilter filter, IEnumerable<int> disjointKeys)
    {
        int total = 0, positives = 0;
        foreach (int key in disjointKeys)
        {
            total++;
            if (filter.MightContain(key)) positives++;
        }
        return total == 0 ? 0 : positives / (double)total;
    }
}

public static class Solution
{
    private static ulong _seed = 0xB5297A4D2E1F3C5DUL;

    private static ulong NextRand()
    {
        _seed += 0x9E3779B97F4A7C15UL;
        ulong z = _seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static int NextInt(int bound) => (int)(NextRand() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string P(double value) => value.ToString("0.0000", CultureInfo.InvariantCulture);

    private static List<int[]> BuildSide(int rows, int keySpace)
    {
        var build = new List<int[]>();
        for (int i = 0; i < rows; i++) build.Add(new[] { NextInt(keySpace), i });
        return build;
    }

    private static List<int[]> ProbeSide(int rows, int keySpace)
    {
        var probe = new List<int[]>();
        for (int i = 0; i < rows; i++) probe.Add(new[] { NextInt(keySpace), i });
        return probe;
    }

    private static void NoFalseNegatives()
    {
        var configs = new[] { (8000, 3), (16000, 5), (2000, 1), (4096, 7), (64, 1), (1, 4) };
        foreach ((int bits, int hashes) in configs)
        {
            var filter = new BloomFilter(bits, hashes);
            var keys = new List<int>();
            for (int i = 0; i < 2000; i++) keys.Add(NextInt(5000));
            foreach (int key in keys) filter.Add(key);
            foreach (int key in keys)
            {
                Check(filter.MightContain(key), "no false negative with m=" + bits + " k=" + hashes + " for key " + key);
            }
            Check(filter.SetBits <= filter.BitCount, "the set bits never exceed the bit array");
            Check(filter.Inserted == keys.Count, "every insertion is counted");
        }
        Console.WriteLine("soundness: all keys of every configuration survive the probe (never a false negative)");

        var repeated = new BloomFilter(5000, 4);
        repeated.Add(77);
        int afterFirst = repeated.SetBits;
        repeated.Add(77);
        Check(repeated.SetBits == afterFirst, "re-inserting the same key changes no bit");
    }

    private static void FilteredJoinMatchesPlainJoin()
    {
        for (int trial = 0; trial < 6; trial++)
        {
            List<int[]> build = BuildSide(500, 2000);
            List<int[]> probe = ProbeSide(3000, 2000);
            JoinOutcome plain = BloomJoin.PlainJoin(probe, build);
            foreach ((int bits, int hashes) in new[] { (1000, 4), (500, 2), (64, 1), (1, 1), (20000, 6) })
            {
                var filter = new BloomFilter(bits, hashes);
                JoinOutcome filtered = BloomJoin.FilteredJoin(probe, build, filter);
                Check(filtered.Rows.SequenceEqual(plain.Rows), "filtered join equals the plain join (trial " + trial + ", m=" + bits + " k=" + hashes + ")");
                Check(filtered.JoinLookups <= plain.JoinLookups, "the filter never adds join lookups");
                Check(filtered.PassedFilter == filtered.JoinLookups, "every row that passes the filter is looked up exactly once");
                Check(filtered.ProbedKeys == probe.Count, "every probe key was examined once");
            }
        }
        Console.WriteLine("correctness: the filtered join returns exactly the plain join result for 5 filter configurations");
    }

    private static void FalsePositiveRateMatchesTheory()
    {
        foreach ((int bits, int hashes, int keys) in new[] { (8000, 3, 1000), (40000, 5, 5000), (20000, 7, 2000) })
        {
            var filter = new BloomFilter(bits, hashes);
            var inserted = new List<int>();
            for (int i = 0; i < keys; i++)
            {
                int key = i; // distinct keys, so the occupancy model applies exactly
                inserted.Add(key);
                filter.Add(key);
            }
            var disjoint = Enumerable.Range(keys, 20000).ToList();
            double measured = BloomJoin.MeasureFalsePositiveRate(filter, disjoint);
            double theoretical = filter.TheoreticalFalsePositiveRate;
            double relative = Math.Abs(measured - theoretical) / theoretical;

            Check(relative < 0.25, "measured false positive rate " + P(measured) + " tracks the formula " + P(theoretical)
                + " (m=" + bits + " k=" + hashes + " n=" + keys + ")");
            Check(measured > 0, "the filter does produce false positives at these settings");
            Console.WriteLine("fpr m=" + bits + " k=" + hashes + " n=" + keys + ": measured " + P(measured) + " theory " + P(theoretical));
        }
    }

    private static void BitBudgetAndOptimalHashCount()
    {
        var keys = Enumerable.Range(0, 2000).ToList();
        var disjoint = Enumerable.Range(2000, 40000).ToList();

        double previous = double.MaxValue;
        foreach (int bitsPerKey in new[] { 1, 2, 4, 8, 16 })
        {
            var filter = new BloomFilter(2000 * bitsPerKey, BloomFilter.OptimalHashCount(2000 * bitsPerKey, 2000));
            foreach (int key in keys) filter.Add(key);
            double rate = BloomJoin.MeasureFalsePositiveRate(filter, disjoint);
            Check(rate < previous, "more bits per key lowers the false positive rate (bits/key " + bitsPerKey + ")");
            previous = rate;
            Console.WriteLine("bits/key " + bitsPerKey + " k=" + filter.HashCount + " fpr " + P(rate));
        }
        Check(previous < 0.002, "16 bits per key pushes the false positive rate well below 0.2%, got " + P(previous));

        int optimal = BloomFilter.OptimalHashCount(2000 * 8, 2000);
        Check(optimal == (int)Math.Round(8 * Math.Log(2.0)), "the optimal k for 8 bits per key is (m/n) ln 2 = " + optimal);

        var measured = new SortedDictionary<int, double>();
        foreach (int hashes in new[] { 1, 2, 3, 4, 5, 6, 8, 10 })
        {
            var filter = new BloomFilter(2000 * 8, hashes);
            foreach (int key in keys) filter.Add(key);
            measured[hashes] = BloomJoin.MeasureFalsePositiveRate(filter, disjoint);
        }
        int best = measured.OrderBy(kv => kv.Value).ThenBy(kv => kv.Key).First().Key;
        Check(Math.Abs(best - optimal) <= 1, "the measured optimum k=" + best + " is within one of (m/n)ln2=" + optimal);
        Check(measured[optimal] < measured[1] && measured[optimal] < measured[10],
            "the optimal k beats both one hash function and ten");
        Console.WriteLine("optimal k: formula " + optimal + ", measured " + best
            + " (fpr " + P(measured[best]) + " vs " + P(measured[1]) + " at k=1, " + P(measured[10]) + " at k=10)");
    }

    private static void WorkReduction()
    {
        List<int[]> build = BuildSide(1000, 20000);
        var probe = new List<int[]>();
        for (int i = 0; i < 50000; i++)
        {
            // A tenth of the probe rows share the build keyspace, the rest are cold keys.
            probe.Add(i % 10 == 0 ? new[] { NextInt(20000), i } : new[] { 1000000 + i, i });
        }

        JoinOutcome plain = BloomJoin.PlainJoin(probe, build);
        var filter = new BloomFilter(8 * 1000, BloomFilter.OptimalHashCount(8000, 1000));
        JoinOutcome bloom = BloomJoin.FilteredJoin(probe, build, filter);

        Check(bloom.Rows.SequenceEqual(plain.Rows), "the bloom join still produces the same rows");
        Check(bloom.PassedFilter < plain.ProbedKeys, "the filter rejected most probe rows ("
            + bloom.PassedFilter + " of " + plain.ProbedKeys + ")");
        Check(bloom.JoinLookups < plain.JoinLookups, "fewer hash table lookups with the filter ("
            + bloom.JoinLookups + " < " + plain.JoinLookups + ")");
        Check(bloom.AvoidedLookups == plain.ProbedKeys - bloom.JoinLookups, "avoided lookups are accounted exactly");
        Console.WriteLine("work: " + plain.ProbedKeys + " probe rows -> " + bloom.PassedFilter
            + " passed the filter, join lookups " + plain.JoinLookups + " -> " + bloom.JoinLookups);
    }

    private static void Boundaries()
    {
        var noBuild = new BloomFilter(4096, 3);
        Check(noBuild.SetBits == 0, "an unused filter has no bits set");
        int passed = 0;
        for (int key = 0; key < 5000; key++)
        {
            if (noBuild.MightContain(key)) passed++;
        }
        Check(passed == 0, "an empty filter never reports a false positive");
        Check(BloomJoin.FilteredJoin(ProbeSide(100, 500), new List<int[]>(), noBuild).Rows.Count == 0,
            "an empty build side yields an empty join");

        var withFilter = new BloomFilter(4096, 3);
        Check(BloomJoin.FilteredJoin(new List<int[]>(), BuildSide(100, 500), withFilter).Rows.Count == 0,
            "an empty probe side yields an empty join");

        var degenerate = new BloomFilter(1, 4);
        degenerate.Add(12345);
        Check(degenerate.MightContain(999), "a single-bit filter reports everything as a possible match");
        List<int[]> tinyProbe = new List<int[]> { new[] { 999, 1 }, new[] { 12345, 2 } };
        List<int[]> tinyBuild = new List<int[]> { new[] { 12345, 7 }, new[] { 12345, 8 } };
        JoinOutcome plain = BloomJoin.PlainJoin(tinyProbe, tinyBuild);
        JoinOutcome filtered = BloomJoin.FilteredJoin(tinyProbe, tinyBuild, new BloomFilter(1, 4));
        Check(filtered.Rows.SequenceEqual(plain.Rows), "even a fully saturated filter keeps the join correct");
        Check(filtered.JoinLookups == plain.JoinLookups, "a saturated filter saves nothing but loses nothing");

        bool threw = false;
        try { new BloomFilter(0, 3); }
        catch (Exception) { threw = true; }
        Check(threw, "a zero-bit filter is rejected");

        threw = false;
        try { new BloomFilter(64, 0); }
        catch (Exception) { threw = true; }
        Check(threw, "a filter without hash functions is rejected");
    }

    public static void Main()
    {
        NoFalseNegatives();
        FilteredJoinMatchesPlainJoin();
        FalsePositiveRateMatchesTheory();
        BitBudgetAndOptimalHashCount();
        WorkReduction();
        Boundaries();
        Console.WriteLine("PASS");
    }
}
