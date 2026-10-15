// Task 484 - Implement simple key-value store with memtable + SSTable. (Tier: Advanced, CTCI Ch 14)
//
// LSM-style prototype with documented semantics:
// - Writes land in an in-memory memtable (key -> value; deletes insert a null TOMBSTONE marker).
// - When the memtable reaches `flush_threshold` entries it flushes to a new immutable SSTable:
//   the records are written sorted by key and appended to the sstable list (oldest first).
// - Get(key) probes the memtable, then the SSTables from newest to oldest; the first record for the
//   key wins. A tombstone means "deleted" (returns null) and must shadow older values in older
//   SSTables - that is what makes deletes correct without rewriting data.
// - Compact() flushes the memtable, merges all SSTables newest-wins into a single one, and drops
//   tombstones plus every record they shadow, so the result holds exactly the live keys.
//
// The test asserts Get == oracle after every single operation of an LCG-driven 240-op script
// (puts, gets, deletes) including tombstone shadowing across SSTables, then checks the
// post-compaction layout and idempotence of a second compaction.
using System;
using System.Collections.Generic;

internal class KVStore
{
    private readonly int flushThreshold;
    public readonly Dictionary<string, int?> Memtable = new Dictionary<string, int?>();
    public readonly List<List<KeyValuePair<string, int?>>> Sstables = new List<List<KeyValuePair<string, int?>>>();
    public int Flushes;
    public int Compactions;

    public KVStore(int flushThreshold)
    {
        this.flushThreshold = flushThreshold;
    }

    public void Put(string key, int? value)
    {
        Memtable[key] = value;
        MaybeFlush();
    }

    public void Delete(string key) => Put(key, null);

    private void MaybeFlush()
    {
        if (Memtable.Count >= flushThreshold) Flush();
    }

    public void Flush()
    {
        if (Memtable.Count == 0) return;
        var records = new List<KeyValuePair<string, int?>>(Memtable);
        records.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
        Sstables.Add(records);
        Memtable.Clear();
        Flushes++;
    }

    public int? Get(string key)
    {
        if (Memtable.TryGetValue(key, out int? value)) return value;
        for (int i = Sstables.Count - 1; i >= 0; i--)
        {
            foreach (var record in Sstables[i])
            {
                if (record.Key == key) return record.Value;
            }
        }
        return null;
    }

    public void Compact()
    {
        Flush();
        var newest = new Dictionary<string, int?>();
        for (int i = Sstables.Count - 1; i >= 0; i--)
        {
            foreach (var record in Sstables[i])
            {
                if (!newest.ContainsKey(record.Key)) newest[record.Key] = record.Value;
            }
        }
        var liveKeys = new List<string>(newest.Keys);
        liveKeys.Sort(StringComparer.Ordinal);
        var live = new List<KeyValuePair<string, int?>>();
        foreach (string key in liveKeys)
        {
            if (newest[key] != null) live.Add(new KeyValuePair<string, int?>(key, newest[key]));
        }
        Sstables.Clear();
        if (live.Count > 0) Sstables.Add(live);
        Compactions++;
    }
}

internal class Oracle
{
    public readonly Dictionary<string, int?> Live = new Dictionary<string, int?>();

    public Oracle(IEnumerable<string> keys)
    {
        foreach (string key in keys) Live[key] = null;
    }

    public void Put(string key, int? value) => Live[key] = value;

    public void Delete(string key) => Live.Remove(key);

    public int? Get(string key) => Live.TryGetValue(key, out int? value) ? value : (int?)null;
}

public static class Solution
{
    private static int seed = 484;

    private static int[][] LcgOps(int count, int keyCount)
    {
        var ops = new int[count][];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int key = seed % keyCount;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 1000;
            ops[i] = new[] { kind, key, value };
        }
        return ops;
    }

    private static string RecordValue(List<List<KeyValuePair<string, int?>>> sstables, string key)
    {
        foreach (var records in sstables)
        {
            foreach (var record in records)
            {
                if (record.Key == key) return record.Value == null ? "tombstone" : record.Value.ToString();
            }
        }
        return "missing";
    }

    public static void Main()
    {
        var store = new KVStore(2);
        store.Put("a", 1);
        store.Put("b", 2);
        Check(store.Flushes == 1 && store.Memtable.Count == 0, "first flush");
        Check(store.Get("a") == 1 && store.Get("b") == 2, "values after flush");
        store.Put("a", 5);
        store.Flush();
        store.Delete("b");
        Check(store.Get("b") == null, "tombstone visible immediately");
        store.Flush();
        Check(store.Get("a") == 5 && store.Get("b") == null, "read through tombstones");
        Check(store.Sstables.Count == 3, "three sstables");
        Check(RecordValue(new List<List<KeyValuePair<string, int?>>> { store.Sstables[2] }, "b") == "tombstone", "newest is tombstone");
        Check(RecordValue(new List<List<KeyValuePair<string, int?>>> { store.Sstables[0] }, "b") == "2", "oldest keeps value");
        store.Compact();
        Check(store.Compactions == 1 && store.Sstables.Count == 1, "compacted");
        Check(store.Sstables[0].Count == 1, "only live record");
        Check(store.Sstables[0][0].Key == "a" && store.Sstables[0][0].Value == 5, "live record");
        Check(store.Get("a") == 5 && store.Get("b") == null, "post compact reads");
        store.Compact();
        Check(store.Sstables.Count == 1 && store.Sstables[0][0].Value == 5, "idempotent compact");

        var keys = new List<string>();
        for (int i = 0; i < 12; i++) keys.Add("k" + i);
        store = new KVStore(4);
        var oracle = new Oracle(keys);
        int gets = 0, tombstones = 0;
        var ops = LcgOps(240, 12);
        for (int step = 0; step < ops.Length; step++)
        {
            int kind = ops[step][0];
            string key = "k" + ops[step][1];
            int value = ops[step][2];
            if (kind == 0)
            {
                store.Put(key, value);
                oracle.Put(key, value);
            }
            else if (kind == 1)
            {
                Check(store.Get(key) == oracle.Get(key), "direct get " + step);
                gets++;
            }
            else
            {
                store.Delete(key);
                oracle.Delete(key);
                tombstones++;
            }
            foreach (string probe in keys)
            {
                Check(store.Get(probe) == oracle.Get(probe), "probe " + step + " " + probe);
            }
            if (step % 60 == 59)
            {
                store.Compact();
                foreach (string probe in keys)
                {
                    Check(store.Get(probe) == oracle.Get(probe), "post compact " + step + " " + probe);
                }
            }
        }
        Check(gets == 84 && tombstones == 71, "op counts " + gets + " " + tombstones);
        Check(store.Flushes == 35, "flush count " + store.Flushes);
        Check(store.Compactions == 4, "compaction count");
        store.Compact();
        Check(store.Sstables.Count == 1, "final single sstable");
        var records = store.Sstables[0];
        var liveKeys = new List<string>();
        foreach (var entry in oracle.Live)
        {
            if (entry.Value != null) liveKeys.Add(entry.Key);
        }
        liveKeys.Sort(StringComparer.Ordinal);
        Check(records.Count == liveKeys.Count, "record count " + records.Count);
        for (int i = 0; i < records.Count; i++)
        {
            Check(records[i].Key == liveKeys[i], "key order " + i);
            Check(records[i].Value != null && records[i].Value == oracle.Live[liveKeys[i]], "value " + i);
        }
        foreach (string probe in keys)
        {
            Check(store.Get(probe) == oracle.Get(probe), "final probe " + probe);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
