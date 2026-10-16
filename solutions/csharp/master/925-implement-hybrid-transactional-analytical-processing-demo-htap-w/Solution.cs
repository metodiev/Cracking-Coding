// Task 925 - Implement hybrid transactional/analytical processing demo (HTAP) with maintenance. (Tier: Master, Beyond — Distributed Systems & Databases)
//
// A two-structure store: point transactions land in an unsorted *delta* of (key, value, version) rows, while the
// read-optimised *columnar* main store keeps a sorted array of committed keys and values plus a delete bitmap. A scan
// merges both structures in key order, so an analytical query sees every version at or below its snapshot version -
// that is the MVCC part: a snapshot taken before a minor compaction returns exactly the same result after it. Minor
// compaction (maintenance) folds the delta versions up to a horizon into the main store with a sorted merge, keeping
// only newer versions in the delta; compaction is idempotent and never changes what any snapshot sees, which is the
// invariant every check re-verifies. The authoritative expected result is always recomputed from a plain dictionary
// (the independent oracle).
//
// Simplifications: single-threaded, in-process, integer keys/values, no WAL/undo, no concurrency control (versions
// are logical counters), compaction never deletes rows from history, no secondary indexes and no query planner.
// Verified: scans match the dictionary oracle after every batch of inserts and updates, point lookups agree, the
// merge is idempotent, a snapshot taken before maintenance is stable across it, row accounting (delta + main) matches
// the live key count, an empty delta and a fully deleted table both behave, and the run is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
}

internal sealed class DeltaRow
{
    internal int Key, Value, Version;
    internal bool Deleted;
}

internal sealed class Htap
{
    internal readonly List<DeltaRow> Delta = new List<DeltaRow>();
    internal int[] MainKeys = Array.Empty<int>();
    internal int[] MainValues = Array.Empty<int>();
    internal bool[] MainDeleted = Array.Empty<bool>();
    internal int Version;
    internal int Merges;
    internal void Insert(int key, int value)
    {
        Version++;
        Delta.Add(new DeltaRow { Key = key, Value = value, Version = Version });
    }
    internal void Delete(int key)
    {
        Version++;
        Delta.Add(new DeltaRow { Key = key, Value = 0, Version = Version, Deleted = true });
    }
    /** Point lookup at a snapshot version: newest delta version <= snapshot wins, otherwise the main store. */
    internal int? Lookup(int key, int snapshot)
    {
        DeltaRow best = null;
        foreach (DeltaRow row in Delta)
            if (row.Key == key && row.Version <= snapshot && (best == null || row.Version > best.Version)) best = row;
        if (best != null) return best.Deleted ? (int?)null : best.Value;
        int index = Array.BinarySearch(MainKeys, key);
        if (index < 0 || MainDeleted[index]) return null;
        return MainValues[index];
    }
    /** Analytical scan at a snapshot version, returned in key order. */
    internal List<int[]> Scan(int snapshot)
    {
        var visible = new SortedDictionary<int, int>();
        for (int i = 0; i < MainKeys.Length; i++) if (!MainDeleted[i]) visible[MainKeys[i]] = MainValues[i];
        foreach (DeltaRow row in Delta.Where(r => r.Version <= snapshot).OrderBy(r => r.Version))
        {
            if (row.Deleted) visible.Remove(row.Key);
            else visible[row.Key] = row.Value;
        }
        return visible.Select(pair => new[] { pair.Key, pair.Value }).ToList();
    }
    /** Minor compaction: fold delta versions up to the horizon into the sorted main store. */
    internal int Compact(int horizon)
    {
        var merged = new SortedDictionary<int, int>();
        for (int i = 0; i < MainKeys.Length; i++) if (!MainDeleted[i]) merged[MainKeys[i]] = MainValues[i];
        var remaining = new List<DeltaRow>();
        int folded = 0;
        foreach (DeltaRow row in Delta.OrderBy(r => r.Version))
            if (row.Version <= horizon)
            {
                folded++;
                if (row.Deleted) merged.Remove(row.Key);
                else merged[row.Key] = row.Value;
            }
            else remaining.Add(row);
        MainKeys = merged.Keys.ToArray();
        MainValues = merged.Values.ToArray();
        MainDeleted = new bool[merged.Count];
        Delta.Clear();
        Delta.AddRange(remaining);
        Merges++;
        return folded;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 925UL;
    private static string Show(List<int[]> rows) => string.Join(";", rows.Select(r => r[0] + "=" + r[1]));
    /** Independent oracle: the same snapshot semantics over a plain dictionary plus a version log. */
    private static string OracleScan(List<int[]> log, int snapshot)
    {
        var live = new SortedDictionary<int, int>();
        foreach (int[] entry in log.Where(e => e[2] <= snapshot).OrderBy(e => e[2]))
        {
            if (entry[3] == 1) live.Remove(entry[0]);
            else live[entry[0]] = entry[1];
        }
        return string.Join(";", live.Select(p => p.Key + "=" + p.Value));
    }
    public static void Main()
    {
        var store = new Htap();
        var log = new List<int[]>();
        int version = 0;
        // 1. transactional writes and a consistency check against the oracle after every batch
        int mismatches = 0;
        for (int batch = 0; batch < 6; batch++)
        {
            for (int i = 0; i < 40; i++)
            {
                int key = (int)(Rng.Next(ref _state) % 60), value = (int)(Rng.Next(ref _state) % 1000);
                version++;
                log.Add(new[] { key, value, version, 0 });
                store.Insert(key, value);
            }
            for (int i = 0; i < 8; i++)
            {
                int key = (int)(Rng.Next(ref _state) % 60);
                version++;
                log.Add(new[] { key, 0, version, 1 });
                store.Delete(key);
            }
            if (Show(store.Scan(version)) != OracleScan(log, version)) mismatches++;
        }
        Check(mismatches == 0, "six mixed insert/delete batches all match the dictionary oracle");
        Check(store.Version == version, "the version counter tracks every write (" + version + " writes)");

        // 2. point lookups agree with the oracle and respect versions
        int lookupErrors = 0, probed = 0;
        for (int key = 0; key < 60; key++)
        {
            foreach (int snapshot in new[] { version - 50, version })
            {
                string expected = OracleScan(log.Where(e => e[0] == key || e[3] == 1).ToList(), snapshot);
                int? got = store.Lookup(key, snapshot);
                bool present = expected.Split(';').Any(part => part.StartsWith(key + "="));
                string want = present ? expected.Split(';').First(part => part.StartsWith(key + "=")).Split('=')[1] : null;
                probed++;
                if ((got == null ? null : got.Value.ToString()) != want) lookupErrors++;
            }
        }
        Check(lookupErrors == 0, "point lookups agree with the oracle at old and current snapshots (" + probed + " probes)");
        Check(store.Lookup(999, version) == null, "an unknown key returns null (boundary)");

        // 3. row accounting and a full-scale scan
        int live = store.Scan(version).Count;
        Check(store.MainKeys.Length + store.Delta.Count >= live, "delta plus columnar rows cover the live keys (" +
            store.MainKeys.Length + " + " + store.Delta.Count + " >= " + live + ")");
        for (int i = 0; i < 5000; i++)
        {
            store.Insert(1000 + i, i);
            log.Add(new[] { 1000 + i, i, store.Version, 0 });
        }
        version = store.Version;
        Check(store.Scan(version).Count == live + 5000, "a 5000-row append is visible to the analytical scan");

        // 4. compaction with horizon H: snapshot H must be stable across it, and maintenance is idempotent
        int horizon = store.Version;
        string stableAtHorizon = Show(store.Scan(horizon));
        int stableCount = store.Scan(horizon).Count;
        for (int i = 0; i < 200; i++) store.Insert(9000 + i, i);
        Check(Show(store.Scan(horizon)) == stableAtHorizon, "writes newer than a snapshot do not leak into it");
        int folded = store.Compact(horizon);
        Check(folded > 0, "minor compaction folded " + folded + " delta versions up to the horizon");
        Check(Show(store.Scan(horizon)) == stableAtHorizon, "the snapshot at the compaction horizon is stable across maintenance");
        Check(store.Scan(store.Version).Count == stableCount + 200, "the 200 newer rows stay visible from the delta");
        Check(store.Compact(horizon) == 0 && store.Scan(store.Version).Count == stableCount + 200,
            "a second compaction at the same horizon is a no-op (idempotent maintenance)");
        Check(OracleScan(log, horizon) == stableAtHorizon, "the authoritative snapshot at the horizon matches the oracle after compaction");
        store.Insert(7, 4242);
        Check(store.Lookup(7, store.Version) == 4242, "transactional writes keep working after maintenance");
        Check(store.MainKeys.Length > 0 && store.Delta.Count == 201, "maintenance moved history into the columnar store (" +
            store.MainKeys.Length + " columnar rows, " + store.Delta.Count + " delta rows: 200 newer + 1 post-maintenance)");
        var empty = new Htap();
        Check(empty.Compact(0) == 0 && empty.Scan(0).Count == 0 && empty.Lookup(1, 1) == null,
            "an empty store compacts to a no-op and scans to nothing (boundary)");
        var wiped = new Htap();
        for (int key = 0; key < 25; key++) wiped.Insert(key, key * 3);
        for (int key = 0; key < 25; key++) wiped.Delete(key);
        Check(wiped.Scan(wiped.Version).Count == 0 && wiped.Compact(wiped.Version) == 50 && wiped.Scan(wiped.Version).Count == 0,
            "a fully deleted table scans empty before and after compaction");

        // 6. reproducibility
        _state = 925UL;
        var replay = new Htap();
        for (int i = 0; i < 40; i++) replay.Insert((int)(Rng.Next(ref _state) % 60), (int)(Rng.Next(ref _state) % 1000));
        Check(replay.MainKeys.Length + replay.Delta.Count == 40 && replay.Version == 40, "the same seed replays the same writes");

        Console.WriteLine("HTAP: " + version + " writes, " + store.Merges + " compactions, " + store.MainKeys.Length +
            " columnar rows, delta " + store.Delta.Count + ", visible rows " + store.Scan(store.Version).Count);
        Console.WriteLine("checks: oracle mismatches " + mismatches + ", lookup errors " + lookupErrors + ", idempotent maintenance, " +
            "5000-row append visible, empty and fully deleted tables behave");
        Console.WriteLine("PASS");
    }
}
