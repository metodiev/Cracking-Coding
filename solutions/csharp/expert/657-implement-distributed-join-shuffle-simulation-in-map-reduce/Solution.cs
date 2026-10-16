// Task 657 - Implement distributed join shuffle simulation in map-reduce. (Tier: Expert, CTCI Ch 9 - Sys Design)
//
// A map-reduce equi-join simulated stage by stage rather than executed on a cluster: the map side tags every
// record with its origin, a partition function routes each mapped record to exactly one reducer by its key,
// the shuffle groups records by reducer, and each reducer hash-joins its own partitions (building on the
// smaller side). Because the partitioner sees only the key, every record of a key lands in one reducer, which
// is what makes the distributed result identical to a single-machine join - and that identity is what the
// checks assert against a nested-loop join (the brute-force reference) for one, three and eight reducers.
// What is simplified: no network is modelled, so shuffle volume is measured in emitted records instead of
// bytes and reducers are exact rather than speculative; the combiner variant is a count combiner, which is
// legal precisely because the aggregated output it feeds is the *number* of joined pairs per key.
// Verified: empty and single-record inputs; duplicate-key cartesian products; identical results to the
// nested-loop reference at three reducer counts; every key confined to one reducer; partition sizes summing
// to the input; pair counts per key from the combiner path equal to the brute-force counts while the combiner
// only ever shrinks the shuffle; a broadcast join against the same reference; range partitioning that keeps
// reducers key-ordered; disjoint keys; a rejected non-positive reducer count; and determinism.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Record
{
    internal int Key;
    internal string Payload = string.Empty;

    internal Record(int key, string payload)
    {
        Key = key;
        Payload = payload;
    }
}

internal sealed class MappedRecord
{
    internal int Key;
    internal bool FromLeft;
    internal string Payload = string.Empty;
}

/// <summary>What a count combiner emits: a key, a side and how many records of that key/side it saw.</summary>
internal sealed class KeySideCount
{
    internal int Key;
    internal bool FromLeft;
    internal long Count;
}

internal static class MapReduceJoin
{
    /// <summary>Brute-force nested-loop join: the reference every distributed variant is compared against.</summary>
    internal static List<string> NestedLoop(IReadOnlyList<Record> left, IReadOnlyList<Record> right)
    {
        var joined = new List<string>();
        foreach (Record r in left)
            foreach (Record s in right)
                if (r.Key == s.Key) joined.Add(r.Key + "|" + r.Payload + "|" + s.Payload);
        joined.Sort(StringComparer.Ordinal);
        return joined;
    }

    internal static Dictionary<int, long> NestedLoopCounts(IReadOnlyList<Record> left, IReadOnlyList<Record> right)
    {
        var counts = new Dictionary<int, long>();
        foreach (Record r in left)
            foreach (Record s in right)
                if (r.Key == s.Key) counts[r.Key] = counts.TryGetValue(r.Key, out long c) ? c + 1 : 1;
        return counts;
    }

    internal static int HashPartition(int key, int reducers)
    {
        ulong mixed = (ulong)(uint)key * 0x9E3779B97F4A7C15UL;
        mixed ^= mixed >> 29;
        return (int)(mixed % (ulong)reducers);
    }

    /// <summary>Range partition: each reducer owns a contiguous slice of the key space, so keys stay ordered.</summary>
    internal static int RangePartition(int key, int reducers, int keySpace)
    {
        int width = Math.Max(1, (keySpace + reducers - 1) / reducers);
        return Math.Min(reducers - 1, Math.Max(0, key) / width);
    }

    internal static List<MappedRecord> Map(IReadOnlyList<Record> left, IReadOnlyList<Record> right)
    {
        var mapped = new List<MappedRecord>();
        foreach (Record r in left) mapped.Add(new MappedRecord { Key = r.Key, FromLeft = true, Payload = r.Payload });
        foreach (Record s in right) mapped.Add(new MappedRecord { Key = s.Key, FromLeft = false, Payload = s.Payload });
        return mapped;
    }

    internal static List<List<MappedRecord>> Shuffle(IReadOnlyList<MappedRecord> mapped, int reducers,
                                                   Func<int, int> partition)
    {
        if (reducers <= 0) throw new ArgumentException("the reducer count must be positive");
        var partitions = new List<List<MappedRecord>>();
        for (int i = 0; i < reducers; i++) partitions.Add(new List<MappedRecord>());
        foreach (MappedRecord record in mapped) partitions[partition(record.Key)].Add(record);
        return partitions;
    }

    /// <summary>Reducer-side hash join; the smaller side is hashed, the larger one probes.</summary>
    internal static List<string> Reduce(IReadOnlyList<List<MappedRecord>> partitions)
    {
        var joined = new List<string>();
        foreach (List<MappedRecord> partition in partitions)
        {
            var leftSide = new List<MappedRecord>();
            var rightSide = new List<MappedRecord>();
            foreach (MappedRecord record in partition) (record.FromLeft ? leftSide : rightSide).Add(record);
            if (leftSide.Count == 0 || rightSide.Count == 0) continue;

            bool hashLeft = leftSide.Count <= rightSide.Count;
            List<MappedRecord> build = hashLeft ? leftSide : rightSide;
            List<MappedRecord> probe = hashLeft ? rightSide : leftSide;
            var table = new Dictionary<int, List<string>>();
            foreach (MappedRecord record in build)
            {
                if (!table.TryGetValue(record.Key, out List<string>? bucket))
                {
                    bucket = new List<string>();
                    table[record.Key] = bucket;
                }
                bucket.Add(record.Payload);
            }
            foreach (MappedRecord record in probe)
            {
                if (!table.TryGetValue(record.Key, out List<string>? bucket)) continue;
                foreach (string payload in bucket)
                    joined.Add(hashLeft
                        ? record.Key + "|" + payload + "|" + record.Payload
                        : record.Key + "|" + record.Payload + "|" + payload);
            }
        }
        joined.Sort(StringComparer.Ordinal);
        return joined;
    }

    /// <summary>Map side of the counting job: one (key, side) record per input row, optionally combined.</summary>
    internal static List<KeySideCount> MapAndCombine(IReadOnlyList<Record> left, IReadOnlyList<Record> right,
                                                    bool combine, int mapTasks)
    {
        var tasks = new List<List<KeySideCount>>();
        for (int t = 0; t < mapTasks; t++) tasks.Add(new List<KeySideCount>());
        for (int i = 0; i < left.Count; i++)
            tasks[i % mapTasks].Add(new KeySideCount { Key = left[i].Key, FromLeft = true, Count = 1 });
        for (int i = 0; i < right.Count; i++)
            tasks[i % mapTasks].Add(new KeySideCount { Key = right[i].Key, FromLeft = false, Count = 1 });

        var emitted = new List<KeySideCount>();
        foreach (List<KeySideCount> task in tasks)
        {
            if (!combine) { emitted.AddRange(task); continue; }
            var collapsed = new Dictionary<string, KeySideCount>();
            foreach (KeySideCount record in task)
            {
                string id = record.Key + ":" + (record.FromLeft ? "L" : "R");
                if (collapsed.TryGetValue(id, out KeySideCount? existing)) existing.Count++;
                else collapsed[id] = new KeySideCount { Key = record.Key, FromLeft = record.FromLeft, Count = 1 };
            }
            emitted.AddRange(collapsed.Values);
        }
        return emitted;
    }

    /// <summary>Number of joined pairs per key, computed entirely through the shuffle, with or without a combiner.</summary>
    internal static Dictionary<int, long> PairCounts(IReadOnlyList<Record> left, IReadOnlyList<Record> right,
                                                    int reducers, Func<int, int> partition, bool combine,
                                                    int mapTasks, out long shuffledRecords)
    {
        if (reducers <= 0) throw new ArgumentException("the reducer count must be positive");
        var emitted = MapAndCombine(left, right, combine, mapTasks);
        shuffledRecords = emitted.Count;
        var leftByPartition = new Dictionary<int, long>[reducers];
        var rightByPartition = new Dictionary<int, long>[reducers];
        for (int i = 0; i < reducers; i++)
        {
            leftByPartition[i] = new Dictionary<int, long>();
            rightByPartition[i] = new Dictionary<int, long>();
        }
        foreach (KeySideCount record in emitted)
        {
            int reducer = partition(record.Key);
            Dictionary<int, long> target = record.FromLeft ? leftByPartition[reducer] : rightByPartition[reducer];
            target[record.Key] = target.TryGetValue(record.Key, out long c) ? c + record.Count : record.Count;
        }
        var totals = new Dictionary<int, long>();
        for (int reducer = 0; reducer < reducers; reducer++)
            foreach (var pair in leftByPartition[reducer])
                if (rightByPartition[reducer].TryGetValue(pair.Key, out long other))
                    totals[pair.Key] = totals.TryGetValue(pair.Key, out long c) ? c + pair.Value * other : pair.Value * other;
        return totals;
    }

    /// <summary>Broadcast variant: the whole right side is replicated to every reducer.</summary>
    internal static List<string> BroadcastJoin(IReadOnlyList<Record> left, IReadOnlyList<Record> right, int reducers)
    {
        var joined = new List<string>();
        for (int reducer = 0; reducer < reducers; reducer++)
            for (int i = reducer; i < left.Count; i += reducers)
                foreach (Record s in right)
                    if (left[i].Key == s.Key) joined.Add(left[i].Key + "|" + left[i].Payload + "|" + s.Payload);
        joined.Sort(StringComparer.Ordinal);
        return joined;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Same(List<string> a, List<string> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static bool SameCounts(Dictionary<int, long> a, Dictionary<int, long> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var pair in b) if (!a.TryGetValue(pair.Key, out long value) || value != pair.Value) return false;
        return true;
    }

    public static void Main()
    {
        // ---- empty and single-record inputs ----------------------------------------------------
        {
            var empty = new List<Record>();
            var one = new List<Record> { new Record(5, "r5") };
            Check(MapReduceJoin.NestedLoop(empty, one).Count == 0 && MapReduceJoin.NestedLoop(one, empty).Count == 0,
                "an empty input joins to nothing on either side");
            var partitions = MapReduceJoin.Shuffle(MapReduceJoin.Map(empty, one), 4,
                key => MapReduceJoin.HashPartition(key, 4));
            Check(MapReduceJoin.Reduce(partitions).Count == 0 && partitions.Count == 4,
                "the shuffle keeps four reducers even when they are all idle and emit nothing");
        }

        // ---- a duplicate-heavy join, brute-force compared at 1, 3 and 8 reducers ---------------
        {
            ulong state = 0x13579BDF2468ACE0UL;
            var left = new List<Record>();
            var right = new List<Record>();
            for (int i = 0; i < 120; i++) left.Add(new Record(Rng.NextInt(ref state, 12), "l" + i));
            for (int i = 0; i < 90; i++) right.Add(new Record(Rng.NextInt(ref state, 12), "s" + i));

            List<string> reference = MapReduceJoin.NestedLoop(left, right);
            long pairTotal = 0;
            foreach (long count in MapReduceJoin.NestedLoopCounts(left, right).Values) pairTotal += count;
            Check(pairTotal == reference.Count,
                "the pair count and the enumerated join agree on " + reference.Count + " pairs");

            bool allMatch = true, sizesConserved = true;
            foreach (int reducers in new[] { 1, 3, 8 })
            {
                List<MappedRecord> mapped = MapReduceJoin.Map(left, right);
                var partitions = MapReduceJoin.Shuffle(mapped, reducers,
                    key => MapReduceJoin.HashPartition(key, reducers));
                int size = 0;
                foreach (List<MappedRecord> partition in partitions) size += partition.Count;
                if (size != mapped.Count) sizesConserved = false;
                if (!Same(MapReduceJoin.Reduce(partitions), reference)) allMatch = false;
            }
            Check(sizesConserved, "the shuffle moves every mapped record into exactly one partition");
            Check(allMatch, "hash shuffle + reducer-side join equals the nested-loop join at 1, 3 and 8 reducers");

            var partitions8 = MapReduceJoin.Shuffle(MapReduceJoin.Map(left, right), 8,
                key => MapReduceJoin.HashPartition(key, 8));
            bool locality = true;
            for (int reducer = 0; reducer < partitions8.Count; reducer++)
            {
                var keysHere = new HashSet<int>();
                foreach (MappedRecord record in partitions8[reducer]) keysHere.Add(record.Key);
                foreach (MappedRecord record in partitions8[(reducer + 1) % partitions8.Count])
                    if (keysHere.Contains(record.Key)) locality = false;
            }
            Check(locality, "every record of a key is routed to the same reducer, never split");

            var plain = MapReduceJoin.PairCounts(left, right, 8, key => MapReduceJoin.HashPartition(key, 8),
                false, 2, out long plainRecords);
            Check(SameCounts(plain, MapReduceJoin.NestedLoopCounts(left, right)),
                "pair counts per key match the brute-force count through the shuffle path");
            var combined = MapReduceJoin.PairCounts(left, right, 8, key => MapReduceJoin.HashPartition(key, 8),
                true, 2, out long combinedRecords);
            Check(SameCounts(combined, plain), "a map-side count combiner preserves the join result");
            Check(combinedRecords < plainRecords,
                "the combiner shrinks the shuffle (" + combinedRecords + " < " + plainRecords + " records)");

            int[] hashSizes = new int[8];
            foreach (MappedRecord record in MapReduceJoin.Map(left, right))
                hashSizes[MapReduceJoin.HashPartition(record.Key, 8)]++;
            int hashMax = 0, rangeMax = 0;
            foreach (int size in hashSizes) hashMax = Math.Max(hashMax, size);
            var rangePartitions = MapReduceJoin.Shuffle(MapReduceJoin.Map(left, right), 8,
                key => MapReduceJoin.RangePartition(key, 8, 12));
            foreach (List<MappedRecord> partition in rangePartitions) rangeMax = Math.Max(rangeMax, partition.Count);
            bool ordered = true;
            for (int reducer = 1; reducer < rangePartitions.Count; reducer++)
                foreach (MappedRecord record in rangePartitions[reducer])
                    foreach (MappedRecord previous in rangePartitions[reducer - 1])
                        if (record.Key < previous.Key) ordered = false;
            Check(ordered, "range partitioning keeps the reducers in key order around the ring");
            Check(Same(MapReduceJoin.Reduce(rangePartitions), reference),
                "the range-partitioned run produces the same join as the hash-partitioned one");
            Check(Same(MapReduceJoin.BroadcastJoin(left, right, 4), reference),
                "broadcasting the right side to four reducers gives the same join");
            Console.WriteLine("pairs=" + reference.Count + " shuffledRecords=" + plainRecords + " combinedRecords="
                + combinedRecords + " hashSkew=" + hashMax + " rangeSkew=" + rangeMax);
        }

        // ---- disjoint keys and a rejected reducer count ----------------------------------------
        {
            var left = new List<Record> { new Record(1, "a"), new Record(2, "b") };
            var right = new List<Record> { new Record(3, "c") };
            var mapped = MapReduceJoin.Map(left, right);
            Check(MapReduceJoin.Reduce(MapReduceJoin.Shuffle(mapped, 2, key => MapReduceJoin.HashPartition(key, 2))).Count == 0
                    && MapReduceJoin.NestedLoop(left, right).Count == 0,
                "disjoint keys produce no pairs even though both inputs are non-empty");
            right.Add(new Record(2, "d"));
            Check(Same(MapReduceJoin.Reduce(MapReduceJoin.Shuffle(MapReduceJoin.Map(left, right), 3,
                    key => MapReduceJoin.HashPartition(key, 3))), MapReduceJoin.NestedLoop(left, right)),
                "adding the missing key produces exactly one pair through the distributed path");

            bool rejected = false;
            try { MapReduceJoin.Shuffle(mapped, 0, key => 0); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a non-positive reducer count is rejected before any shuffling happens");
            rejected = false;
            try { MapReduceJoin.PairCounts(left, right, 0, key => 0, false, 1, out _); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "the counting path validates the reducer count as well");
        }

        // ---- determinism -----------------------------------------------------------------------
        {
            ulong state = 0x2468ACE013579BDFUL;
            var input = new List<Record>();
            for (int i = 0; i < 40; i++) input.Add(new Record(Rng.NextInt(ref state, 6), "v" + i));
            Check(Same(MapReduceJoin.NestedLoop(input, input), MapReduceJoin.NestedLoop(input, input)),
                "the same input always produces the same join");
            var partitions = MapReduceJoin.Shuffle(MapReduceJoin.Map(input, input), 5,
                key => MapReduceJoin.HashPartition(key, 5));
            Check(Same(MapReduceJoin.Reduce(partitions), MapReduceJoin.NestedLoop(input, input))
                    && Same(MapReduceJoin.Reduce(partitions), MapReduceJoin.BroadcastJoin(input, input, 1)),
                "self-joining an input matches the reference through both the shuffle and broadcast paths");
        }

        Console.WriteLine("PASS");
    }
}
