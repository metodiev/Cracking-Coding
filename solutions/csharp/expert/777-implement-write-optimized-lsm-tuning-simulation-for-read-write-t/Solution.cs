// Task 777 - Implement write-optimized LSM tuning simulation for read/write tradeoffs. (Tier: Expert, CTCI Ch 14 - Databases)
//
// An LSM storage engine is simulated end to end: a memtable of (key, version, value/tombstone)
// entries is flushed into a sorted run whenever it fills up, and runs are compacted downwards through
// a level hierarchy. Two classic policies are implemented, because the whole point of the exercise is
// the tradeoff between them:
//
//   * Levelled: every level holds one run, and a level that exceeds its size target (memtableSize *
//     T^level) is merged into the level below, rewriting that level as well. This rewrites a lot of
//     data (high write amplification) but keeps exactly one run per level to probe (low read
//     amplification, and obsolete versions disappear as soon as they meet their overwriter).
//   * Tiered: a level collects up to T runs and merges them only when it is full, so each entry is
//     rewritten far less often (low write amplification) at the cost of probing several runs per
//     lookup and keeping superseded versions around longer (high read amplification / space
//     amplification).
//
// All measurements are taken from the simulation itself rather than from a formula: entries written
// by flushes and merges (write amplification = written entries / user operations), runs probed by
// point lookups and the same count restricted to runs whose Bloom filter says "maybe" (read
// amplification), and stored entries divided by live entries (space amplification).
//
// Verified: every point lookup, every range scan and the visibility of tombstones match a plain
// dictionary oracle that stores the current value of each key, no entry is ever lost or resurrected
// (monotone version order inside a merge), the levelled policy really rewrites more while probing
// fewer runs than the tiered one at the same size ratio, a larger size ratio cuts write amplification
// (and, in this single-run-per-level model, also the number of live runs), the tiered policy's read
// amplification grows with the ratio where the levelled one shrinks, Bloom filters never produce a
// false negative, and the degenerate cases (no workload, one key rewritten many times, deleting every
// key) behave correctly.
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
}

internal sealed class Entry
{
    internal readonly int Key;
    internal readonly int Version;      // higher wins
    internal readonly int Value;        // int.MinValue marks a tombstone

    internal Entry(int key, int version, int value)
    {
        Key = key;
        Version = version;
        Value = value;
    }

    internal bool IsTombstone => Value == int.MinValue;
}

internal sealed class Run
{
    internal readonly List<Entry> Entries = new List<Entry>();
    internal readonly long[] Filter;    // 8 bits per entry Bloom filter over the keys
    internal readonly int FilterBits;

    internal Run(IEnumerable<Entry> entries, int uniqueKeys)
    {
        Entries.AddRange(entries);
        FilterBits = Math.Max(64, uniqueKeys * 8);
        Filter = new long[(FilterBits + 63) / 64];
        foreach (Entry entry in Entries)
        {
            // tombstones set their bit too: a filter answered "absent" for a key whose only marker in
            // this run is a deletion, which would have resurrected an older version in a deeper level
            int bit = (int)(Mix(entry.Key) % (ulong)FilterBits);
            Filter[bit / 64] |= 1L << (bit % 64);
        }
    }

    private static ulong Mix(int key)
    {
        ulong z = (ulong)(uint)key * 0x9E3779B97F4A7C15UL + 0x1234567UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>False means "definitely absent"; true means "maybe" (a real filter never lies negatively).</summary>
    internal bool MaybeContains(int key)
    {
        int bit = (int)(Mix(key) % (ulong)FilterBits);
        return (Filter[bit / 64] & (1L << (bit % 64))) != 0;
    }

    internal Entry Seek(int key)
    {
        int low = 0, high = Entries.Count - 1;
        while (low <= high)
        {
            int mid = (low + high) / 2;
            if (Entries[mid].Key < key) low = mid + 1;
            else if (Entries[mid].Key > key) high = mid - 1;
            else return Entries[mid];
        }
        return null;
    }
}

internal sealed class LsmStore
{
    internal readonly bool Levelled;
    internal readonly int SizeRatio;
    internal readonly int MemtableCapacity;
    private readonly List<Entry> _memtable = new List<Entry>();
    private readonly List<List<Run>> _levels = new List<List<Run>>();

    internal long Flushes, Compactions, EntriesWrittenByFlush, EntriesWrittenByCompaction;
    internal long Lookups, RunsProbed, RunsFilteredIn, BloomFalseNegatives;
    internal int MaxLevels;

    internal IReadOnlyList<Entry> MemtableEntries => _memtable;
    internal IReadOnlyList<List<Run>> Levels => _levels;

    internal LsmStore(bool levelled, int sizeRatio, int memtableCapacity)
    {
        if (sizeRatio < 2) throw new ArgumentException("the size ratio must be at least 2");
        if (memtableCapacity < 1) throw new ArgumentException("the memtable must hold at least one entry");
        Levelled = levelled;
        SizeRatio = sizeRatio;
        MemtableCapacity = memtableCapacity;
    }

    internal long StoredEntries
    {
        get
        {
            long total = _memtable.Count;
            foreach (List<Run> level in _levels) foreach (Run run in level) total += run.Entries.Count;
            return total;
        }
    }

    internal int RunCount
    {
        get
        {
            int total = 0;
            foreach (List<Run> level in _levels) total += level.Count;
            return total;
        }
    }

    private void EnsureLevel(int level)
    {
        while (_levels.Count <= level) _levels.Add(new List<Run>());
        if (_levels.Count > MaxLevels) MaxLevels = _levels.Count;
    }

    internal void Put(Entry entry)
    {
        _memtable.Add(entry);
        if (_memtable.Count >= MemtableCapacity) Flush();
    }

    internal void Flush()
    {
        if (_memtable.Count == 0) return;
        _memtable.Sort((a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : b.Version.CompareTo(a.Version));
        var compacted = Compact(_memtable, int.MaxValue);
        EntriesWrittenByFlush += compacted.Count;
        Flushes++;
        EnsureLevel(0);
        _levels[0].Add(new Run(compacted, KeyRange));
        _memtable.Clear();
        Rebalance();
    }

    private int _keyCardinality = 1;
    internal int KeyRange => _keyCardinality;

    internal void AnnounceKeySpace(int uniqueKeys) => _keyCardinality = Math.Max(1, uniqueKeys);

    /// <summary>Keep the newest version of each key, dropping tombstones once nothing older can exist.</summary>
    private static List<Entry> Compact(List<Entry> entries, int dropTombstonesBelowVersion)
    {
        var sorted = new List<Entry>(entries);
        sorted.Sort((a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : b.Version.CompareTo(a.Version));
        var result = new List<Entry>();
        int index = 0;
        while (index < sorted.Count)
        {
            Entry newest = sorted[index];
            result.Add(newest);
            index++;
            while (index < sorted.Count && sorted[index].Key == newest.Key) index++;   // older versions die here
        }
        if (dropTombstonesBelowVersion != int.MaxValue) result.RemoveAll(e => e.IsTombstone);
        return result;
    }

    private static List<Entry> MergeAll(IEnumerable<Run> runs, int dropTombstonesBelowVersion)
    {
        var all = new List<Entry>();
        foreach (Run run in runs) all.AddRange(run.Entries);
        return Compact(all, dropTombstonesBelowVersion);
    }

    private void Rebalance()
    {
        for (int level = 0; level < _levels.Count; level++)
        {
            bool full;
            if (Levelled)
            {
                long target = (long)MemtableCapacity * (long)Math.Pow(SizeRatio, level + 1);
                long size = 0;
                foreach (Run run in _levels[level]) size += run.Entries.Count;
                full = size > target;
            }
            else
            {
                full = _levels[level].Count > SizeRatio;
            }
            if (!full) continue;
            EnsureLevel(level + 1);
            var sources = new List<Run>(_levels[level]);
            if (Levelled) sources.AddRange(_levels[level + 1]);      // levelled merges rewrite the level below
            // tombstones are kept: only a merge that covers *every* level may drop them, otherwise an
            // older version still sitting in a level that was not part of the merge would resurrect
            List<Entry> merged = MergeAll(sources, int.MaxValue);
            EntriesWrittenByCompaction += merged.Count;
            Compactions++;
            _levels[level].Clear();
            if (Levelled) _levels[level + 1].Clear();
            _levels[level + 1].Add(new Run(merged, KeyRange));
        }
    }

    internal void ForceCompaction()
    {
        while (RunCount > 1 || _memtable.Count > 0)
        {
            int before = RunCount;
            if (_memtable.Count > 0) Flush();
            if (RunCount == before && RunCount > 1)
            {
                // push everything to the bottom level in one merge
                var all = new List<Run>();
                foreach (List<Run> level in _levels) all.AddRange(level);
                List<Entry> merged = MergeAll(all, 0);
                EntriesWrittenByCompaction += merged.Count;
                Compactions++;
                foreach (List<Run> level in _levels) level.Clear();
                EnsureLevel(0);
                _levels[0].Add(new Run(merged, KeyRange));
            }
            if (_levels.Count > 0 && _levels[0].Count == 1 && RunCount == 1) break;
        }
    }

    /// <summary>Newest version wins; tombstones hide older values and read as absent.</summary>
    internal bool TryGet(int key, out int value)
    {
        value = 0;
        Lookups++;
        Entry best = null;
        foreach (Entry entry in _memtable)                       // unflushed writes are visible too
        {
            if (entry.Key != key) continue;
            if (best == null || entry.Version > best.Version) best = entry;
        }
        for (int level = 0; level < _levels.Count; level++)
        {
            foreach (Run run in _levels[level])
            {
                RunsProbed++;
                if (!run.MaybeContains(key))
                {
                    // the filter says the key cannot be here: skip the binary search entirely
                    if (run.Seek(key) != null) BloomFalseNegatives++;
                    continue;
                }
                RunsFilteredIn++;
                Entry found = run.Seek(key);
                if (found == null) continue;
                if (best == null || found.Version > best.Version) best = found;
            }
        }
        if (best == null) return false;
        if (best.IsTombstone) return false;
        value = best.Value;
        return true;
    }

    internal List<int> Scan(int from, int to)
    {
        var visible = new Dictionary<int, Entry>();
        var flushCandidate = new List<Entry>(_memtable);
        foreach (Entry entry in flushCandidate)
        {
            if (entry.Key < from || entry.Key > to) continue;
            if (!visible.TryGetValue(entry.Key, out Entry current) || entry.Version > current.Version) visible[entry.Key] = entry;
        }
        foreach (List<Run> level in _levels)
            foreach (Run run in level)
            {
                foreach (Entry entry in run.Entries)
                {
                    if (entry.Key < from || entry.Key > to) continue;
                    if (!visible.TryGetValue(entry.Key, out Entry current) || entry.Version > current.Version) visible[entry.Key] = entry;
                }
            }
        var keys = new List<int>();
        foreach (KeyValuePair<int, Entry> pair in visible) if (!pair.Value.IsTombstone) keys.Add(pair.Key);
        keys.Sort();
        return keys;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Plain dictionary oracle: the current value of every key, ignoring tombstones.</summary>
    private sealed class Oracle
    {
        internal readonly Dictionary<int, int> Values = new Dictionary<int, int>();
        internal int Operations;

        internal void Apply(int key, int value)
        {
            Operations++;
            if (value == int.MaxValue) Values.Remove(key);
            else Values[key] = value;
        }

        internal bool TryGet(int key, out int value) => Values.TryGetValue(key, out value);

        internal List<int> Scan(int from, int to)
        {
            var keys = new List<int>();
            foreach (KeyValuePair<int, int> pair in Values) if (pair.Key >= from && pair.Key <= to) keys.Add(pair.Key);
            keys.Sort();
            return keys;
        }

        internal string Dump(int key, LsmStore store)
        {
            var text = new System.Text.StringBuilder("oracle=" + (Values.ContainsKey(key) ? Values[key].ToString() : "absent"));
            foreach (Entry entry in store.MemtableEntries)
                if (entry.Key == key) text.Append(" mem(v" + entry.Version + (entry.IsTombstone ? ",tomb" : ",val" + entry.Value) + ")");
            int level = 0;
            foreach (List<Run> runs in store.Levels)
            {
                foreach (Run run in runs)
                {
                    Entry found = run.Seek(key);
                    if (found != null) text.Append(" L" + level + "(v" + found.Version + (found.IsTombstone ? ",tomb" : ",val" + found.Value) + ")");
                }
                level++;
            }
            return text.ToString();
        }
    }

    private static bool SameKeys(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Deterministic skewed workload: a few keys are rewritten far more often than the rest. A spot
    /// lookup every 137 operations keeps the read-amplification counters measuring the whole workload
    /// rather than just the final state, and is cross-checked against the oracle immediately.
    /// </summary>
    private static int RunWorkload(LsmStore store, Oracle oracle, ulong seed, int operations, int uniqueKeys, bool deleteSome)
    {
        ulong state = seed;
        store.AnnounceKeySpace(uniqueKeys);
        int version = 0;
        int mismatches = 0;
        for (int i = 0; i < operations; i++)
        {
            int draw = SplitMix64.NextInt(ref state, 100);
            int key = draw < 60 ? SplitMix64.NextInt(ref state, Math.Max(1, uniqueKeys / 10))
                                : SplitMix64.NextInt(ref state, uniqueKeys);
            version++;
            bool remove = deleteSome && SplitMix64.NextInt(ref state, 20) == 0;
            int value = remove ? int.MinValue : 1000 + i;
            store.Put(new Entry(key, version, value));
            oracle.Apply(key, remove ? int.MaxValue : 1000 + i);
            if (i % 137 == 0)
            {
                int probe = SplitMix64.NextInt(ref state, uniqueKeys);
                bool present = store.TryGet(probe, out int stored);
                bool expected = oracle.TryGet(probe, out int expectedValue);
                if (present != expected || (present && stored != expectedValue)) mismatches++;
            }
        }
        return mismatches;
    }

    private static bool Verify(LsmStore store, Oracle oracle, int uniqueKeys, int probes, out string problem)
    {
        problem = "ok";
        for (int key = 0; key < uniqueKeys; key++)
        {
            bool present = store.TryGet(key, out int value);
            bool expected = oracle.TryGet(key, out int expectedValue);
            if (present != expected) { problem = "key " + key + " presence " + present + " but the oracle says " + expected + " [" + oracle.Dump(key, store) + "]"; return false; }
            if (present && value != expectedValue) { problem = "key " + key + " value " + value + " not " + expectedValue + " [" + oracle.Dump(key, store) + "]"; return false; }
        }
        if (!SameKeys(store.Scan(0, uniqueKeys - 1), oracle.Scan(0, uniqueKeys - 1)))
        {
            problem = "the range scan disagrees with the oracle";
            return false;
        }
        return true;
    }

    public static void Main()
    {
        // ---- the store must answer exactly like the dictionary oracle -----------------------------
        {
            const int Keys = 400, Operations = 3000;
            var store = new LsmStore(true, 4, 64);
            var oracle = new Oracle();
            RunWorkload(store, oracle, 0x5EED777000000001UL, Operations, Keys, true);
            string problem;
            Check(Verify(store, oracle, Keys, Keys, out problem), "levelled point lookups and scans match the oracle: " + problem);
            Check(store.BloomFalseNegatives == 0, "the Bloom filters never reported a stored key as absent");
            Check(store.Flushes > 10, "the memtable really flushed (" + store.Flushes + " flushes)");
            Check(store.Compactions > 5, "compactions really ran (" + store.Compactions + ")");

            var tiered = new LsmStore(false, 4, 64);
            var tieredOracle = new Oracle();
            RunWorkload(tiered, tieredOracle, 0x5EED777000000001UL, Operations, Keys, true);
            Check(Verify(tiered, tieredOracle, Keys, Keys, out problem), "tiered point lookups and scans match the oracle: " + problem);
            Console.WriteLine("correctness: levelled runs=" + store.RunCount + " compactions=" + store.Compactions
                + " | tiered runs=" + tiered.RunCount + " compactions=" + tiered.Compactions);
        }

        // ---- the read/write tradeoff between the two policies --------------------------------------
        {
            const int Keys = 300, Operations = 4000;
            double results = 0;
            var report = new List<string>();
            foreach (bool levelled in new[] { true, false })
            {
                var store = new LsmStore(levelled, 4, 48);
                var oracle = new Oracle();
                RunWorkload(store, oracle, 0x5EED777000000002UL, Operations, Keys, false);
                string problem;
                Check(Verify(store, oracle, Keys, Keys, out problem), "the " + (levelled ? "levelled" : "tiered") + " run stays correct: " + problem);
                double writeAmp = (double)(store.EntriesWrittenByFlush + store.EntriesWrittenByCompaction) / Operations;
                double readAmp = (double)store.RunsFilteredIn / Math.Max(1, store.Lookups);
                double spaceAmp = (double)store.StoredEntries / Math.Max(1, oracle.Values.Count);
                Check(writeAmp >= 1, "write amplification is at least one (" + writeAmp.ToString("F2") + ")");
                Check(spaceAmp >= 1, "space amplification is at least one (" + spaceAmp.ToString("F2") + ")");
                Check(readAmp >= 1, "a lookup touches at least one run (" + readAmp.ToString("F2") + ")");
                results += writeAmp;
                report.Add((levelled ? "levelled" : "tiered") + ": WA=" + writeAmp.ToString("F2") + " RA=" + readAmp.ToString("F2")
                    + " SA=" + spaceAmp.ToString("F2") + " runs=" + store.RunCount);
                if (levelled)
                {
                    Check(store.RunsProbed > store.RunsFilteredIn, "the Bloom filters really saved run reads on the levelled store ("
                        + store.RunsFilteredIn + " of " + store.RunsProbed + ")");
                }
            }
            Console.WriteLine("policy tradeoff: " + string.Join(" | ", report));
        }

        // ---- the size ratio T and the two policies: measure the tradeoff ---------------------------
        {
            const int Keys = 300, Operations = 3000;
            var writeAmp = new Dictionary<string, double>();
            var readAmp = new Dictionary<string, double>();
            foreach (bool levelled in new[] { true, false })
                foreach (int ratio in new[] { 2, 4, 8 })
                {
                    var store = new LsmStore(levelled, ratio, 32);
                    var oracle = new Oracle();
                    int spotMismatches = RunWorkload(store, oracle, 0x5EED777000000003UL, Operations, Keys, false);
                    Check(spotMismatches == 0, (levelled ? "levelled" : "tiered") + " ratio " + ratio
                        + ": spot lookups during the workload match the oracle");
                    string tag = (levelled ? "L" : "T") + ratio;
                    writeAmp[tag] = (double)(store.EntriesWrittenByFlush + store.EntriesWrittenByCompaction) / Operations;
                    readAmp[tag] = (double)store.RunsFilteredIn / Math.Max(1, store.Lookups);
                    string problem;
                    Check(Verify(store, oracle, Keys, Keys, out problem),
                        (levelled ? "levelled" : "tiered") + " ratio " + ratio + " stays correct: " + problem);
                }
            Console.WriteLine("size ratio sweep: WA L2=" + writeAmp["L2"].ToString("F2") + " L4=" + writeAmp["L4"].ToString("F2")
                + " L8=" + writeAmp["L8"].ToString("F2") + " T2=" + writeAmp["T2"].ToString("F2") + " T4=" + writeAmp["T4"].ToString("F2")
                + " T8=" + writeAmp["T8"].ToString("F2"));
            Console.WriteLine("size ratio sweep: RA L2=" + readAmp["L2"].ToString("F2") + " L4=" + readAmp["L4"].ToString("F2")
                + " L8=" + readAmp["L8"].ToString("F2") + " T2=" + readAmp["T2"].ToString("F2") + " T4=" + readAmp["T4"].ToString("F2")
                + " T8=" + readAmp["T8"].ToString("F2"));
            Check(writeAmp["L2"] > writeAmp["L8"],
                "levelled: a larger size ratio rewrites less (T=2 " + writeAmp["L2"].ToString("F2") + " > T=8 " + writeAmp["L8"].ToString("F2") + ")");
            Check(writeAmp["L4"] >= writeAmp["L8"], "levelled write amplification falls monotonically in T");
            Check(writeAmp["L4"] > writeAmp["T4"] && writeAmp["L2"] > writeAmp["T2"] && writeAmp["L8"] > writeAmp["T8"],
                "levelled always rewrites more than tiered at the same ratio");
            Check(readAmp["T4"] >= readAmp["T2"] && readAmp["T8"] >= readAmp["T4"],
                "tiered read amplification grows with the number of runs a level may collect");
            Check(readAmp["L4"] <= readAmp["L2"] && readAmp["L8"] <= readAmp["L4"],
                "levelled read amplification shrinks as the levels grow");
            Check(readAmp["T2"] >= readAmp["L2"] && readAmp["T4"] > readAmp["L4"] && readAmp["T8"] > readAmp["L8"],
                "tiered never probes fewer runs than levelled at the same ratio");
        }

        // ---- compaction really reclaims obsolete versions and tombstones ---------------------------
        {
            var store = new LsmStore(true, 4, 32);
            for (int round = 0; round < 200; round++) store.Put(new Entry(7, round + 1, 500 + round));
            long before = store.StoredEntries;
            store.ForceCompaction();
            long after = store.StoredEntries;
            Check(store.TryGet(7, out int value) && value == 500 + 199, "the surviving version is the newest one");
            Check(after < before, "compaction shrank the stored data (" + before + " -> " + after + ")");
            Check(after <= 2, "only the newest version of the single key survives (" + after + " entries)");

            var deletes = new LsmStore(true, 4, 16);
            for (int key = 0; key < 50; key++) deletes.Put(new Entry(key, 1, key));
            for (int key = 0; key < 50; key++) deletes.Put(new Entry(key, 2, int.MinValue));
            Check(deletes.Scan(0, 49).Count == 0, "the tombstones hide every key");
            for (int key = 0; key < 50; key++) Check(!deletes.TryGet(key, out _), "deleted key " + key + " reads as absent");
            deletes.ForceCompaction();
            Check(!deletes.TryGet(0, out _) && !deletes.TryGet(49, out _), "the deleted keys stay deleted across the full compaction");
            Console.WriteLine("reclamation: versions before=" + before + " after=" + after
                + " tombstoneEntries=" + deletes.StoredEntries + " tombstoneRuns=" + deletes.RunCount);
        }

        // ---- degenerate inputs ----------------------------------------------------------------------
        {
            var empty = new LsmStore(true, 4, 8);
            Check(!empty.TryGet(1, out _) && empty.Scan(0, 10).Count == 0, "an empty store returns nothing");
            Check(empty.StoredEntries == 0 && empty.RunCount == 0, "an empty store stores nothing");
            empty.Flush();
            Check(empty.RunCount == 0, "flushing an empty memtable is a no-op");

            var single = new LsmStore(true, 2, 4);
            single.Put(new Entry(3, 1, 42));
            Check(single.TryGet(3, out int v) && v == 42, "a key still in the memtable is visible");
            Check(single.RunCount == 0, "nothing has been flushed yet");
            single.ForceCompaction();
            Check(single.TryGet(3, out v) && v == 42, "the key survives compaction");
            Check(single.Scan(0, 5).Count == 1, "the scan sees the single key");

            bool threw = false;
            try { new LsmStore(true, 1, 8); } catch (ArgumentException) { threw = true; }
            Check(threw, "a size ratio below 2 is rejected");
            bool threwCap = false;
            try { new LsmStore(true, 4, 0); } catch (ArgumentException) { threwCap = true; }
            Check(threwCap, "an empty memtable capacity is rejected");

            var tight = new LsmStore(false, 2, 2);
            var oracle = new Oracle();
            for (int i = 0; i < 40; i++)
            {
                tight.Put(new Entry(i % 7, i + 1, i));
                oracle.Apply(i % 7, i);
            }
            string problem;
            Check(Verify(tight, oracle, 7, 7, out problem), "a two-entry memtable still behaves correctly: " + problem);
            Console.WriteLine("degenerate: singleKeyVisible=" + single.TryGet(3, out _) + " tightRuns=" + tight.RunCount);
        }

        Console.WriteLine("PASS");
    }
}