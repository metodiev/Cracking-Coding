// Task 478 - Implement log-structured merge tree (LSM) basic flush/merge simulation. (Tier: Advanced, CTCI Ch 14 — Databases)
using System;
using System.Collections.Generic;

internal class LsmEntry
{
    public readonly int Seq;
    public readonly int? Value; // null is a tombstone

    public LsmEntry(int seq, int? value)
    {
        Seq = seq;
        Value = value;
    }
}

internal class Sstable
{
    public readonly int Id;
    public readonly Dictionary<int, LsmEntry> Entries;

    public Sstable(int id, Dictionary<int, LsmEntry> entries)
    {
        Id = id;
        Entries = entries;
    }
}

internal class LSMStore
{
    private const int MergeFanout = 4;

    public Dictionary<int, LsmEntry> Memtable = new Dictionary<int, LsmEntry>();
    public List<Sstable> Sstables = new List<Sstable>();
    public int Seq;
    private int opsSinceFlush;
    private readonly int flushAfter;
    public int FlushCount;
    public int MergeCount;
    private int nextTableId;

    public LSMStore(int flushAfter) => this.flushAfter = flushAfter;

    public LSMStore() : this(4) { }

    public void Write(int key, int? value)
    {
        Seq++;
        Memtable[key] = new LsmEntry(Seq, value);
        opsSinceFlush++;
        if (opsSinceFlush >= flushAfter) Flush();
    }

    public void Delete(int key) => Write(key, null);

    public static Sstable MergeTables(List<Sstable> tables)
    {
        var merged = new Dictionary<int, LsmEntry>();
        foreach (Sstable table in tables)
        {
            foreach (var pair in table.Entries)
            {
                if (!merged.TryGetValue(pair.Key, out LsmEntry current) || pair.Value.Seq > current.Seq)
                {
                    merged[pair.Key] = pair.Value;
                }
            }
        }
        return new Sstable(-1, merged);
    }

    public void Flush()
    {
        if (Memtable.Count == 0) return;
        nextTableId++;
        Sstables.Add(new Sstable(nextTableId, new Dictionary<int, LsmEntry>(Memtable)));
        Memtable = new Dictionary<int, LsmEntry>();
        opsSinceFlush = 0;
        FlushCount++;
        while (Sstables.Count >= MergeFanout)
        {
            Sstable merged = MergeTables(Sstables.GetRange(0, MergeFanout));
            var rest = Sstables.GetRange(MergeFanout, Sstables.Count - MergeFanout);
            Sstables = new List<Sstable> { merged };
            Sstables.AddRange(rest);
            MergeCount++;
        }
    }

    public int? Get(int key)
    {
        LsmEntry best = Memtable.TryGetValue(key, out LsmEntry own) ? own : null;
        foreach (Sstable table in Sstables)
        {
            if (table.Entries.TryGetValue(key, out LsmEntry entry) && (best == null || entry.Seq > best.Seq))
            {
                best = entry;
            }
        }
        return best == null ? (int?)null : best.Value;
    }

    public List<int[]> Scan()
    {
        var merged = new Dictionary<int, LsmEntry>();
        foreach (Sstable table in Sstables)
        {
            foreach (var pair in table.Entries)
            {
                if (!merged.TryGetValue(pair.Key, out LsmEntry current) || pair.Value.Seq > current.Seq)
                {
                    merged[pair.Key] = pair.Value;
                }
            }
        }
        foreach (var pair in Memtable)
        {
            if (!merged.TryGetValue(pair.Key, out LsmEntry current) || pair.Value.Seq > current.Seq)
            {
                merged[pair.Key] = pair.Value;
            }
        }
        var keys = new List<int>(merged.Keys);
        keys.Sort();
        var output = new List<int[]>();
        foreach (int key in keys)
        {
            if (merged[key].Value.HasValue) output.Add(new[] { key, merged[key].Value.Value });
        }
        return output;
    }
}

public static class Solution
{
    private const int Keys = 16;

    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static int[][] GenOps(int count, int seed)
    {
        var ops = new int[count][];
        for (int i = 0; i < count; i++)
        {
            seed = Lcg(seed);
            int kind = seed % 5;
            seed = Lcg(seed);
            int key = seed % Keys;
            seed = Lcg(seed);
            int value = seed % 100;
            ops[i] = new[] { kind, key, value };
        }
        return ops;
    }

    private static bool SameScan(List<int[]> a, List<int[]> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        }
        return true;
    }

    private static int? OracleValue(Dictionary<int, int?> oracle, int key) =>
        oracle.TryGetValue(key, out int? value) ? value : null;

    private static List<int[]> ExpectedScan(Dictionary<int, int?> oracle)
    {
        var keys = new List<int>(oracle.Keys);
        keys.Sort();
        var output = new List<int[]>();
        foreach (int key in keys)
        {
            if (oracle[key].HasValue) output.Add(new[] { key, oracle[key].Value });
        }
        return output;
    }

    private static void RunWorkload()
    {
        var store = new LSMStore();
        var oracle = new Dictionary<int, int?>();
        int[][] ops = GenOps(80, 31337);
        for (int i = 0; i < ops.Length; i++)
        {
            int kind = ops[i][0], key = ops[i][1], value = ops[i][2];
            if (kind == 4)
            {
                store.Delete(key);
                oracle[key] = null;
            }
            else
            {
                store.Write(key, value);
                oracle[key] = value;
            }
            for (int k = 0; k < Keys; k++)
            {
                Check(store.Get(k) == OracleValue(oracle, k), "get " + i + " key " + k);
            }
            Check(SameScan(store.Scan(), ExpectedScan(oracle)), "scan " + i);
        }
        Check(store.FlushCount == 20, "one flush per four ops");
        Check(store.MergeCount == 6, "merge at flushes 4,7,10,13,16,19");
        Check(store.Sstables.Count == 2, "one merged table plus the last flush");
        Check(store.Memtable.Count == 0, "memtable drained by the last flush");
    }

    public static void Main()
    {
        // Flush policy: the first table appears exactly on the 4th operation.
        var s = new LSMStore();
        s.Write(0, 10);
        s.Write(1, 11);
        s.Write(2, 12);
        Check(s.Sstables.Count == 0 && s.Get(0) == 10, "no flush yet");
        s.Write(3, 13);
        Check(s.Sstables.Count == 1 && s.Memtable.Count == 0 && s.FlushCount == 1, "flush on 4th op");
        Check(SameScan(s.Scan(), new List<int[]> { new[] { 0, 10 }, new[] { 1, 11 }, new[] { 2, 12 }, new[] { 3, 13 } }),
            "scan after flush");
        s.Delete(1);
        Check(s.Get(1) == null, "deleted");
        s.Write(1, 99);
        Check(s.Get(1) == 99, "rewritten");
        Check(SameScan(s.Scan(), new List<int[]> { new[] { 0, 10 }, new[] { 1, 99 }, new[] { 2, 12 }, new[] { 3, 13 } }),
            "scan after rewrite");

        // Tombstones survive flushing and merging: a deleted key stays deleted.
        var s2 = new LSMStore(2);
        s2.Write(5, 1);
        s2.Write(5, 2);
        s2.Delete(5);
        s2.Write(6, 3);
        Check(s2.Get(5) == null && s2.Get(6) == 3, "tombstone visible");
        s2.Sstables = new List<Sstable> { LSMStore.MergeTables(s2.Sstables) };
        Check(s2.Get(5) == null, "merge must not resurrect the tombstone");
        Check(SameScan(s2.Scan(), new List<int[]> { new[] { 6, 3 } }), "scan without the tombstone");
        s2.Write(5, 7);
        s2.Write(8, 8);
        Check(s2.Get(5) == 7 && s2.Get(8) == 8, "reinserted key");
        Check(SameScan(s2.Scan(), new List<int[]> { new[] { 5, 7 }, new[] { 6, 3 }, new[] { 8, 8 } }), "final scan");

        RunWorkload();
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
