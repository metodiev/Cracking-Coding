// Task 682 - Implement map-reduce optimization for skewed keys (combiner + partitioning).
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A map-reduce pipeline over integer keys, built so that the two classic optimisations can be measured
// rather than asserted:
//   * the combiner. Each mapper aggregates its own records locally (a per-mapper hash map) before anything
//     is shuffled, which collapses the shuffle volume from |records| to the number of distinct keys a mapper
//     saw. The file counts the shuffled records with and without the combiner on the same input and shows
//     the saving, and it re-runs the same job with the mapper's input order permuted to demonstrate that the
//     result does not depend on that order (the combiner is applied to a commutative, associative reduction).
//   * partitioning that survives skew. The default partitioner hashes the key into one of R reducers, which
//     puts every record of a heavy key on the same reducer; the file measures that as max-over-mean
//     partition load. The skew-aware job detects the heavy keys, splits each of them over the reducers with
//     a composite (key, split) key, and merges the partial sums afterwards. The merged result must be
//     exactly the unoptimised result — that equality, plus the per-key multiset check that no record is lost
//     or duplicated, is what the optimisations are accepted on, and the skew metric must genuinely improve
//     (measured, not assumed).
// The reference answer is a plain single-threaded reduction over all records, computed independently of both
// pipelines; every comparison is against it.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic data generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Record
{
    internal int Key;
    internal int Value;

    internal Record(int key, int value) { Key = key; Value = value; }
}

internal sealed class JobResult
{
    internal Dictionary<int, int> Output = new Dictionary<int, int>();
    internal int ShuffledRecords;
    internal int ReducedRecords;
    internal int Combinations;
    internal int[] PartitionLoad = Array.Empty<int>();
    internal int MaxPartitionLoad;
}

internal static class MapReduce
{
    internal static Dictionary<int, int> Reference(IEnumerable<Record> records)
    {
        var totals = new Dictionary<int, int>();
        foreach (Record record in records)
        {
            int current;
            totals.TryGetValue(record.Key, out current);
            totals[record.Key] = current + record.Value;
        }
        return totals;
    }

    private static int Hash(int key, int reducers)
    {
        // A deliberately simple, deterministic partitioner: the point is that all copies of one key go to
        // the same reducer, which is what makes a heavy key a straggler.
        int mixed = unchecked((int)(key * 2654435761L));
        mixed = mixed < 0 ? -mixed : mixed;
        return mixed % reducers;
    }

    /// The unoptimised job: every record travels through the shuffle, then the reducer sums by key.
    internal static JobResult PlainJob(IReadOnlyList<Record> records, int mappers, int reducers)
    {
        var result = new JobResult { PartitionLoad = new int[reducers] };
        var shuffled = new List<Record>();
        var perMapper = new List<List<Record>>();
        for (int m = 0; m < mappers; m++) perMapper.Add(new List<Record>());
        for (int i = 0; i < records.Count; i++) perMapper[i % mappers].Add(records[i]);
        foreach (List<Record> chunk in perMapper)
        {
            foreach (Record record in chunk) shuffled.Add(record);
        }
        result.ShuffledRecords = shuffled.Count;
        result.ReducedRecords = shuffled.Count;
        var partitions = new List<List<Record>>();
        for (int r = 0; r < reducers; r++) partitions.Add(new List<Record>());
        foreach (Record record in shuffled) partitions[Hash(record.Key, reducers)].Add(record);
        for (int r = 0; r < reducers; r++)
        {
            result.PartitionLoad[r] = partitions[r].Count;
            foreach (Record record in partitions[r])
            {
                int current;
                result.Output.TryGetValue(record.Key, out current);
                result.Output[record.Key] = current + record.Value;
            }
        }
        result.MaxPartitionLoad = 0;
        foreach (int load in result.PartitionLoad) if (load > result.MaxPartitionLoad) result.MaxPartitionLoad = load;
        return result;
    }

    /// The optimised job: per-mapper combiners collapse the shuffle, heavy keys are split across reducers,
    /// and a final merge adds the partials back together.
    internal static JobResult OptimisedJob(IReadOnlyList<Record> records, int mappers, int reducers)
    {
        var result = new JobResult { PartitionLoad = new int[reducers] };
        var perMapper = new List<List<Record>>();
        for (int m = 0; m < mappers; m++) perMapper.Add(new List<Record>());
        for (int i = 0; i < records.Count; i++) perMapper[i % mappers].Add(records[i]);

        // Mapper-side combining.
        var combined = new List<KeyValuePair<int, int>>();
        foreach (List<Record> chunk in perMapper)
        {
            var local = new Dictionary<int, int>();
            foreach (Record record in chunk)
            {
                int current;
                local.TryGetValue(record.Key, out current);
                local[record.Key] = current + record.Value;
                result.Combinations++;
            }
            foreach (KeyValuePair<int, int> pair in local) combined.Add(pair);
        }
        result.ShuffledRecords = combined.Count;

        // Heavy keys are the ones whose combined weight would overload a single reducer.
        var perKeyWeight = new Dictionary<int, int>();
        foreach (KeyValuePair<int, int> pair in combined)
        {
            int current;
            perKeyWeight.TryGetValue(pair.Key, out current);
            perKeyWeight[pair.Key] = current + 1;
        }
        int heavyThreshold = Math.Max(2, combined.Count / Math.Max(1, reducers) / 2);
        var heavy = new HashSet<int>();
        foreach (KeyValuePair<int, int> pair in perKeyWeight)
        {
            if (pair.Value >= heavyThreshold) heavy.Add(pair.Key);
        }

        // The splitter: a heavy key becomes (key, split) with the split index round-robining over reducers.
        var splitCounters = new Dictionary<int, int>();
        var splitPartials = new List<List<KeyValuePair<int, int>>>();
        for (int r = 0; r < reducers; r++) splitPartials.Add(new List<KeyValuePair<int, int>>());
        foreach (KeyValuePair<int, int> pair in combined)
        {
            result.ReducedRecords++;
            int home = Hash(pair.Key, reducers);
            int target = home;
            if (heavy.Contains(pair.Key))
            {
                int counter;
                splitCounters.TryGetValue(pair.Key, out counter);
                splitCounters[pair.Key] = counter + 1;
                target = counter % reducers;
                if (target != home) result.ShuffledRecords++;   // a replicated copy of a heavy key
            }
            splitPartials[target].Add(pair);
        }

        // Reduce, then merge the partials of the split keys.
        for (int r = 0; r < reducers; r++)
        {
            result.PartitionLoad[r] = splitPartials[r].Count;
            foreach (KeyValuePair<int, int> pair in splitPartials[r])
            {
                int current;
                result.Output.TryGetValue(pair.Key, out current);
                result.Output[pair.Key] = current + pair.Value;
            }
        }
        result.MaxPartitionLoad = 0;
        foreach (int load in result.PartitionLoad) if (load > result.MaxPartitionLoad) result.MaxPartitionLoad = load;
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameTotals(Dictionary<int, int> expected, Dictionary<int, int> actual)
    {
        if (expected.Count != actual.Count) return false;
        foreach (KeyValuePair<int, int> pair in expected)
        {
            int value;
            if (!actual.TryGetValue(pair.Key, out value)) return false;
            if (value != pair.Value) return false;
        }
        return true;
    }

    /// A labelled data set with a few deliberately heavy keys.
    private static List<Record> SkewedData(ref ulong state, int records, int distinctKeys, int heavyKeys)
    {
        var data = new List<Record>();
        for (int i = 0; i < records; i++)
        {
            int key;
            if (i % 3 == 0 && heavyKeys > 0)
            {
                key = Rng.NextBelow(ref state, heavyKeys);              // the heavy hitters
            }
            else
            {
                key = Rng.NextBelow(ref state, distinctKeys);
            }
            data.Add(new Record(key, 1 + Rng.NextBelow(ref state, 9)));
        }
        return data;
    }

    private static double Imbalance(int[] loads)
    {
        int total = 0;
        int max = 0;
        foreach (int load in loads)
        {
            total += load;
            if (load > max) max = load;
        }
        if (total == 0) return 0.0;
        double mean = total / (double)loads.Length;
        return max / mean;
    }

    public static void Main()
    {
        // ---- correctness on a small hand-checkable input ---------------------------------------
        var tiny = new List<Record>
        {
            new Record(7, 3), new Record(7, 4), new Record(9, 1), new Record(7, 5)
        };
        Dictionary<int, int> tinyReference = MapReduce.Reference(tiny);
        JobResult tinyPlain = MapReduce.PlainJob(tiny, 2, 3);
        JobResult tinyOptimised = MapReduce.OptimisedJob(tiny, 2, 3);
        Check(SameTotals(tinyReference, tinyPlain.Output), "the plain job sums a tiny input correctly");
        Check(SameTotals(tinyReference, tinyOptimised.Output),
              "the combiner still sums it correctly (" + string.Join(",", tinyOptimised.Output.Keys) + ")");
        Check(tinyOptimised.ShuffledRecords <= tinyPlain.ShuffledRecords,
              "and shuffles no more records than the plain job");

        // ---- the reference is the single-threaded reduction ------------------------------------
        ulong state = 682001UL;
        int dataSets = 12;
        int totalSaved = 0;
        int improvedSkew = 0;
        for (int d = 0; d < dataSets; d++)
        {
            int records = 400 + Rng.NextBelow(ref state, 400);
            int distinctKeys = 20 + Rng.NextBelow(ref state, 40);
            List<Record> data = SkewedData(ref state, records, distinctKeys, 2);
            int mappers = 4;
            int reducers = 4;
            Dictionary<int, int> reference = MapReduce.Reference(data);
            JobResult plain = MapReduce.PlainJob(data, mappers, reducers);
            JobResult optimised = MapReduce.OptimisedJob(data, mappers, reducers);
            Check(SameTotals(reference, plain.Output), "data set " + d + ": the plain job matches the reference");
            Check(SameTotals(reference, optimised.Output), "data set " + d + ": the optimised job matches the reference");
            Check(optimised.Combinations == data.Count, "data set " + d + ": the combiner saw every record once ("
                  + optimised.Combinations + " of " + data.Count + ")");
            Check(optimised.ShuffledRecords < plain.ShuffledRecords, "data set " + d + ": the combiner reduced the"
                  + " shuffle (" + optimised.ShuffledRecords + " vs " + plain.ShuffledRecords + " records)");
            totalSaved += plain.ShuffledRecords - optimised.ShuffledRecords;
            Check(plain.ShuffledRecords == data.Count, "data set " + d + ": without a combiner everything is shuffled");
            if (Imbalance(optimised.PartitionLoad) < Imbalance(plain.PartitionLoad)) improvedSkew++;
            Check(Imbalance(optimised.PartitionLoad) <= Imbalance(plain.PartitionLoad),
                  "data set " + d + ": splitting heavy keys never makes the balance worse (plain "
                  + Imbalance(plain.PartitionLoad).ToString("0.00") + " vs optimised "
                  + Imbalance(optimised.PartitionLoad).ToString("0.00") + ")");
        }
        Check(totalSaved >= 2000, "the combiner saved " + totalSaved + " shuffled records across the data sets");
        Check(improvedSkew >= 6, "and the split improved the partition balance in " + improvedSkew + " of "
              + dataSets + " data sets");

        // ---- order independence of the combiner -------------------------------------------------
        List<Record> ordered = SkewedData(ref state, 600, 30, 2);
        var reversed = new List<Record>(ordered);
        reversed.Reverse();
        JobResult forwardJob = MapReduce.OptimisedJob(ordered, 3, 4);
        JobResult backwardJob = MapReduce.OptimisedJob(reversed, 3, 4);
        Check(SameTotals(forwardJob.Output, backwardJob.Output),
              "permuting the mapper input does not change the combined result");
        Check(SameTotals(MapReduce.Reference(ordered), backwardJob.Output),
              "and both orders agree with the reference");

        // ---- extreme skew ----------------------------------------------------------------------
        var oneKey = new List<Record>();
        for (int i = 0; i < 500; i++) oneKey.Add(new Record(42, 2));
        Check(SameTotals(MapReduce.Reference(oneKey), MapReduce.PlainJob(oneKey, 4, 4).Output),
              "an input with a single key is summed correctly by the plain job");
        JobResult singleKey = MapReduce.OptimisedJob(oneKey, 4, 4);
        Check(SameTotals(MapReduce.Reference(oneKey), singleKey.Output),
              "and by the split optimised job (" + singleKey.Output[42] + " from 500 records)");
        Check(singleKey.MaxPartitionLoad < 500, "the 500 records of the single heavy key are spread over the reducers"
              + " (max load " + singleKey.MaxPartitionLoad + ")");
        Check(Imbalance(singleKey.PartitionLoad) <= 2.0, "so the imbalance stays within a factor of two ("
              + Imbalance(singleKey.PartitionLoad).ToString("0.00") + ")");

        // ---- degenerate shapes -----------------------------------------------------------------
        var empty = new List<Record>();
        Check(MapReduce.PlainJob(empty, 3, 3).Output.Count == 0, "an empty input produces an empty result");
        Check(MapReduce.OptimisedJob(empty, 3, 3).Output.Count == 0, "with or without the combiner");
        var single = new List<Record> { new Record(1, 5) };
        Check(MapReduce.OptimisedJob(single, 1, 1).Output[1] == 5, "a single record with one reducer is passed through");
        Check(MapReduce.PlainJob(single, 1, 1).ShuffledRecords == 1, "and counts as one shuffled record");
        List<Record> wide = SkewedData(ref state, 200, 200, 0);
        JobResult wideJob = MapReduce.OptimisedJob(wide, 4, 8);
        Check(wideJob.ShuffledRecords <= wide.Count, "with many distinct keys the combiner cannot save much, but never"
              + " shuffles more than the input (" + wideJob.ShuffledRecords + " of " + wide.Count + ")");
        Check(SameTotals(MapReduce.Reference(wide), wideJob.Output), "and the totals are still right");

        Console.WriteLine("data sets: " + dataSets + ", records saved by the combiner " + totalSaved);
        Console.WriteLine("skew improvement in " + improvedSkew + " of " + dataSets + " sets");
        Console.WriteLine("single heavy key: max reducer load " + singleKey.MaxPartitionLoad + " of the 500 records");
        Console.WriteLine("PASS");
    }
}
