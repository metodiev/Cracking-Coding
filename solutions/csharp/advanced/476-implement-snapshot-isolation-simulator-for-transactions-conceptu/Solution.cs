// Task 476 - Implement snapshot isolation simulator for transactions (conceptual). (Tier: Advanced, CTCI Ch 14 — Databases)
using System;
using System.Collections.Generic;

internal class Txn
{
    public readonly string Id;
    public readonly int Snapshot;
    public readonly Dictionary<string, int> Writes = new Dictionary<string, int>();
    public string State = "active";

    public Txn(string id, int snapshot)
    {
        Id = id;
        Snapshot = snapshot;
    }
}

internal class MvccStore
{
    private readonly Dictionary<string, List<int[]>> versions = new Dictionary<string, List<int[]>>();
    public int CommitCounter;
    private int nextTid;

    public MvccStore(Dictionary<string, int> initial)
    {
        foreach (var pair in initial) AppendCommitted(pair.Key, pair.Value);
    }

    private void AppendCommitted(string key, int value)
    {
        CommitCounter++;
        if (!versions.ContainsKey(key)) versions[key] = new List<int[]>();
        versions[key].Add(new[] { CommitCounter, value });
    }

    public Txn Begin()
    {
        nextTid++;
        return new Txn("T" + nextTid, CommitCounter);
    }

    public int? Read(Txn txn, string key)
    {
        if (txn.Writes.TryGetValue(key, out int own)) return own; // read-your-writes
        int? best = null;
        if (versions.TryGetValue(key, out List<int[]> list))
        {
            foreach (int[] v in list)
            {
                if (v[0] <= txn.Snapshot) best = v[1];               // state as of BEGIN
            }
        }
        return best;
    }

    public void Write(Txn txn, string key, int value) => txn.Writes[key] = value;

    public bool Commit(Txn txn)
    {
        foreach (string key in txn.Writes.Keys)
        {
            if (versions.TryGetValue(key, out List<int[]> list))
            {
                foreach (int[] v in list)
                {
                    if (v[0] > txn.Snapshot)                          // committed after BEGIN
                    {
                        txn.State = "aborted";
                        return false;
                    }
                }
            }
        }
        CommitCounter++;
        int ts = CommitCounter;
        foreach (var pair in txn.Writes)
        {
            if (!versions.ContainsKey(pair.Key)) versions[pair.Key] = new List<int[]>();
            versions[pair.Key].Add(new[] { ts, pair.Value });
        }
        txn.State = "committed";
        return true;
    }

    public Dictionary<string, int> Materialize()
    {
        var output = new Dictionary<string, int>();
        foreach (var pair in versions)
        {
            int latest = 0;
            foreach (int[] v in pair.Value) latest = v[1];
            output[pair.Key] = latest;
        }
        return output;
    }
}

public static class Solution
{
    private static readonly string[] Keys = { "a", "b", "c", "d" };

    private static Dictionary<string, int> Initial() => new Dictionary<string, int>
    {
        { "a", 0 }, { "b", 1 }, { "c", 2 }, { "d", 3 },
    };

    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static int[][] RandomScript(int count, int seed)
    {
        var script = new int[count][];
        for (int i = 0; i < count; i++)
        {
            seed = Lcg(seed);
            int action = seed % 5;
            seed = Lcg(seed);
            int pick = seed;
            seed = Lcg(seed);
            int keyIndex = seed % Keys.Length;
            seed = Lcg(seed);
            int value = seed % 100;
            script[i] = new[] { action, pick, keyIndex, value };
        }
        return script;
    }

    private static bool SameMap(Dictionary<string, int> a, Dictionary<string, int> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var pair in a)
        {
            if (!b.TryGetValue(pair.Key, out int other) || other != pair.Value) return false;
        }
        return true;
    }

    private static void RunRandomPhase()
    {
        var store = new MvccStore(Initial());
        var committed = Initial();
        var versionTs = new Dictionary<string, int>();
        int ts = 1;
        foreach (string key in Initial().Keys) versionTs[key] = ts++;
        int counter = store.CommitCounter;
        Check(counter == 4, "initial commit counter");
        var liveTxns = new Dictionary<string, Txn>();
        var liveSnapshots = new Dictionary<string, Dictionary<string, int>>();
        var liveWrites = new Dictionary<string, Dictionary<string, int>>();
        int nextTid = 0;
        foreach (int[] op in RandomScript(60, 97531))
        {
            int action = op[0];
            int pick = op[1];
            string key = Keys[op[2]];
            int value = op[3];
            if (action == 0 || liveTxns.Count == 0)
            {
                nextTid++;
                Txn txn = store.Begin();
                Check(txn.Snapshot == counter, "snapshot pinned at BEGIN");
                liveTxns[txn.Id] = txn;
                liveSnapshots[txn.Id] = new Dictionary<string, int>(committed);
                liveWrites[txn.Id] = new Dictionary<string, int>();
            }
            else
            {
                var ids = new List<string>(liveTxns.Keys);
                ids.Sort();
                string tid = ids[pick % ids.Count];
                Txn txn = liveTxns[tid];
                if (action == 1 || action == 4)
                {
                    int? got = store.Read(txn, key);
                    int? expected = liveWrites[tid].TryGetValue(key, out int own)
                        ? own
                        : (liveSnapshots[tid].TryGetValue(key, out int snap) ? snap : (int?)null);
                    Check(got == expected, "random read " + tid + " " + key);
                }
                else if (action == 2)
                {
                    store.Write(txn, key, value);
                    liveWrites[tid][key] = value;
                }
                else
                {
                    bool success = true;
                    foreach (string k in txn.Writes.Keys)
                    {
                        int lastTs = versionTs.TryGetValue(k, out int v) ? v : -1;
                        if (lastTs > txn.Snapshot) success = false;
                    }
                    bool got = store.Commit(txn);
                    Check(got == success, "random commit " + tid);
                    if (got)
                    {
                        counter = store.CommitCounter;
                        foreach (var pair in txn.Writes)
                        {
                            versionTs[pair.Key] = counter;
                            committed[pair.Key] = pair.Value;
                        }
                    }
                    liveTxns.Remove(tid);
                    liveSnapshots.Remove(tid);
                    liveWrites.Remove(tid);
                }
            }
            Check(SameMap(store.Materialize(), committed), "materialized state matches oracle");
            Check(store.CommitCounter == counter, "commit counter matches oracle");
        }
        foreach (var pair in liveTxns)
        {
            Txn txn = pair.Value;
            foreach (string key in Keys)
            {
                int? got = store.Read(txn, key);
                int? expected = liveWrites[txn.Id].TryGetValue(key, out int own)
                    ? own
                    : (liveSnapshots[txn.Id].TryGetValue(key, out int snap) ? snap : (int?)null);
                Check(got == expected, "leftover txn read");
            }
        }
        Check(SameMap(store.Materialize(), committed), "final materialized state matches oracle");
    }

    public static void Main()
    {
        var store = new MvccStore(Initial()); // versions a@1, b@2, c@3, d@4
        Txn t1 = store.Begin();               // snapshot 4
        store.Write(t1, "a", 99);
        Txn t2 = store.Begin();               // snapshot 4
        Txn t3 = store.Begin();               // snapshot 4
        Check(store.Read(t2, "a") == 0, "no dirty read of t1's uncommitted 99");
        store.Write(t2, "a", 100);
        Check(store.Commit(t1), "first committer wins");
        Check(!store.Commit(t2), "write-write conflict must abort the loser");
        Check(store.Read(t3, "a") == 0, "snapshot pinned before t1's commit");
        Check(store.Read(t3, "a") == 0, "repeatable read");
        Check(store.Commit(t3), "read-only txn commits");
        Txn t4 = store.Begin();
        Check(store.Read(t4, "a") == 99, "fresh snapshot sees the commit");

        Txn t5 = store.Begin();
        store.Write(t5, "d", 6);
        Check(store.Read(t5, "d") == 6, "read-your-writes");
        Check(store.Commit(t5), "t5 commits");
        Txn t6 = store.Begin();
        Check(store.Read(t6, "d") == 6, "t6 sees committed d");

        Txn t7 = store.Begin();
        Txn t8 = store.Begin();
        Check(store.Read(t7, "d") == 6 && store.Read(t8, "d") == 6, "both read d=6");
        store.Write(t7, "d", 7);
        store.Write(t8, "d", 8);
        Check(store.Commit(t7), "t7 wins");
        Check(!store.Commit(t8), "lost update must be prevented");
        Check(store.Read(store.Begin(), "d") == 7, "d stays 7");
        Check(SameMap(store.Materialize(), new Dictionary<string, int>
        {
            { "a", 99 }, { "b", 1 }, { "c", 2 }, { "d", 7 },
        }), "materialized state");

        // Write skew (documented SI anomaly): disjoint write sets both commit (0 = "on", 1 = "off").
        var docs = new MvccStore(new Dictionary<string, int> { { "x", 0 }, { "y", 0 } });
        Txn t9 = docs.Begin();
        Txn t10 = docs.Begin();
        Check(docs.Read(t9, "x") == 0 && docs.Read(t9, "y") == 0, "t9 snapshot");
        Check(docs.Read(t10, "x") == 0 && docs.Read(t10, "y") == 0, "t10 snapshot");
        docs.Write(t9, "x", 1);
        docs.Write(t10, "y", 1);
        Check(docs.Commit(t9) && docs.Commit(t10), "write skew is allowed under SI");
        Check(SameMap(docs.Materialize(), new Dictionary<string, int> { { "x", 1 }, { "y", 1 } }),
            "the on-call invariant is broken");
        Txn fresh = docs.Begin();
        Check(docs.Read(fresh, "x") == 1 && docs.Read(fresh, "y") == 1, "fresh snapshot");

        RunRandomPhase();
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
