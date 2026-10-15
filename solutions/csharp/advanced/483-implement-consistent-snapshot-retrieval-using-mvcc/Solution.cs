// Task 483 - Implement consistent snapshot retrieval using MVCC. (Tier: Advanced, CTCI Ch 14)
//
// Multi-version store: every key holds the full list of (commit_ts, value) versions produced by
// whole-transaction commits (all keys of one transaction share a single commit timestamp). A read
// issued at snapshot timestamp s returns, per key, the newest version with commit_ts <= s, i.e. the
// state as of a single point in time even if newer commits exist. Because one transaction stamps
// all of its writes with the same ts, a snapshot never observes a torn transaction.
//
// The scripted history is seeded with base versions at ts = 0, then 12 LCG-driven transactions at
// ts = 1..12. A slow reader opens a snapshot at ts = 3 after commits 1..8 exist, reads all keys,
// waits while 9..12 commit, and re-reads: both reads must equal the brute-force oracle (a linear
// scan of raw history), while fresh snapshots keep seeing the newest writers.
using System;
using System.Collections.Generic;

internal class VersionEntry
{
    public readonly int Ts;
    public readonly int Value;

    public VersionEntry(int ts, int value)
    {
        Ts = ts;
        Value = value;
    }
}

internal class HistoryRecord
{
    public readonly int Ts;
    public readonly string Key;
    public readonly int Value;

    public HistoryRecord(int ts, string key, int value)
    {
        Ts = ts;
        Key = key;
        Value = value;
    }
}

internal class VersionedStore
{
    public readonly Dictionary<string, List<VersionEntry>> Versions = new Dictionary<string, List<VersionEntry>>();
    public int CommitCount;

    public VersionedStore(IEnumerable<string> keys)
    {
        foreach (string key in keys) Versions[key] = new List<VersionEntry>();
    }

    public void Commit(int ts, List<KeyValuePair<string, int>> writes)
    {
        foreach (var write in writes) Versions[write.Key].Add(new VersionEntry(ts, write.Value));
        CommitCount++;
    }

    public int? Get(string key, int snapshotTs)
    {
        List<VersionEntry> entries = Versions[key];
        int lo = 0, hi = entries.Count - 1;
        int? best = null;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (entries[mid].Ts <= snapshotTs)
            {
                best = entries[mid].Value;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return best;
    }

    public Snapshot Snapshot(int ts) => new Snapshot(this, ts);

    public int HeadTs()
    {
        int best = 0;
        foreach (var entries in Versions.Values)
        {
            foreach (var entry in entries) best = Math.Max(best, entry.Ts);
        }
        return best;
    }
}

internal class Snapshot
{
    private readonly VersionedStore store;
    private readonly int ts;

    public Snapshot(VersionedStore store, int ts)
    {
        this.store = store;
        this.ts = ts;
    }

    public int? Get(string key) => store.Get(key, ts);
}

public static class Solution
{
    private static int seed = 483;

    private static int? OracleValue(List<HistoryRecord> history, string key, int snapshotTs)
    {
        int? value = null;
        foreach (HistoryRecord record in history)
        {
            if (record.Key == key && record.Ts <= snapshotTs) value = record.Value;
        }
        return value;
    }

    private static List<HistoryRecord> BuildHistory(Dictionary<string, int> baseVersions, int transactions)
    {
        var history = new List<HistoryRecord>();
        var sortedBase = new List<string>(baseVersions.Keys);
        sortedBase.Sort(StringComparer.Ordinal);
        foreach (string key in sortedBase) history.Add(new HistoryRecord(0, key, baseVersions[key]));
        for (int t = 1; t <= transactions; t++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + t % 2;
            for (int w = 0; w < count; w++)
            {
                seed = (seed * 75 + 74) % 65537;
                string key = "k" + (seed % 5);
                seed = (seed * 75 + 74) % 65537;
                int value = 100 + seed % 900;
                history.Add(new HistoryRecord(t, key, value));
            }
        }
        return history;
    }

    private static List<KeyValuePair<int, List<KeyValuePair<string, int>>>> GroupTransactions(List<HistoryRecord> history)
    {
        var grouped = new List<KeyValuePair<int, List<KeyValuePair<string, int>>>>();
        foreach (HistoryRecord record in history)
        {
            if (grouped.Count == 0 || grouped[grouped.Count - 1].Key != record.Ts)
            {
                grouped.Add(new KeyValuePair<int, List<KeyValuePair<string, int>>>(record.Ts, new List<KeyValuePair<string, int>>()));
            }
            grouped[grouped.Count - 1].Value.Add(new KeyValuePair<string, int>(record.Key, record.Value));
        }
        return grouped;
    }

    public static void Main()
    {
        var baseVersions = new Dictionary<string, int>();
        for (int i = 0; i < 5; i++) baseVersions["k" + i] = 10 * i + 1;
        seed = 483;
        var history = BuildHistory(baseVersions, 12);
        Check(history.Count == 5 + 18, "history size");
        var transactions = GroupTransactions(history);
        Check(transactions.Count == 13, "transaction count");
        Check(transactions[0].Key == 0, "base ts");

        var store = new VersionedStore(baseVersions.Keys);
        for (int i = 0; i < 9; i++) store.Commit(transactions[i].Key, transactions[i].Value);
        Check(store.CommitCount == 9, "commits so far");

        Snapshot reader = store.Snapshot(3);
        var first = new Dictionary<string, int?>();
        foreach (string key in baseVersions.Keys) first[key] = reader.Get(key);

        for (int i = 9; i < transactions.Count; i++) store.Commit(transactions[i].Key, transactions[i].Value);
        foreach (string key in baseVersions.Keys)
        {
            Check(reader.Get(key) == first[key], "repeatable read " + key);
            Check(first[key] == OracleValue(history, key, 3), "snapshot oracle " + key);
            Check(first[key] != null, "visible key " + key);
        }

        Check(store.HeadTs() == 12, "head ts");
        Snapshot fresh = store.Snapshot(store.HeadTs());
        bool moved = false;
        foreach (string key in baseVersions.Keys)
        {
            int? headValue = fresh.Get(key);
            Check(headValue == OracleValue(history, key, 12), "head oracle " + key);
            if (first[key] != headValue) moved = true;
        }
        Check(moved, "writers moved state");

        foreach (string key in baseVersions.Keys)
        {
            for (int ts = -1; ts <= 13; ts++)
            {
                Check(store.Get(key, ts) == OracleValue(history, key, ts), "isolation " + key + " " + ts);
            }
        }

        foreach (var transaction in transactions.GetRange(1, transactions.Count - 1))
        {
            int ts = transaction.Key;
            Snapshot current = store.Snapshot(ts);
            Snapshot previous = store.Snapshot(ts - 1);
            var distinct = new HashSet<string>();
            foreach (var write in transaction.Value)
            {
                distinct.Add(write.Key);
                Check(current.Get(write.Key) == write.Value, "read your writes " + ts + " " + write.Key);
                Check(previous.Get(write.Key) == OracleValue(history, write.Key, ts - 1), "prior state " + ts + " " + write.Key);
            }
            Check(distinct.Count <= 2, "atomic key count " + ts);
        }

        var versions = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in store.Versions) versions[entry.Key] = entry.Value.Count;
        Check(versions["k0"] == 3 && versions["k1"] == 6 && versions["k2"] == 4
              && versions["k3"] == 3 && versions["k4"] == 7, "version counts");
        int total = 0;
        foreach (int count in versions.Values) total += count;
        Check(total == 5 + 18, "version total");

        int countBefore = store.CommitCount;
        store.Commit(13, new List<KeyValuePair<string, int>> { new KeyValuePair<string, int>("k0", 777) });
        store.Commit(14, new List<KeyValuePair<string, int>> { new KeyValuePair<string, int>("k1", 888) });
        Check(store.CommitCount == countBefore + 2, "writer progress");
        Check(store.Get("k0", 13) == 777 && store.Get("k0", 12) != 777, "k0 commit");
        Check(store.Get("k1", 14) == 888 && store.Get("k1", 13) != 888, "k1 commit");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
