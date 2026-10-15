// Task 346 - Optimistic concurrency control sketch with version numbers. (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;

public class VersionedStore
{
    private readonly Dictionary<string, long[]> records = new Dictionary<string, long[]>();   // key -> {value, version}

    public void Create(string key, long value)
    {
        if (records.ContainsKey(key)) throw new ArgumentException("key already exists");
        records[key] = new long[] { value, 1 };        // new records start at version 1
    }

    public long[] Read(string key)
    {
        if (!records.TryGetValue(key, out long[]? record)) throw new ArgumentException("no such key: " + key);
        return (long[])record.Clone();
    }

    public bool Commit(string key, long newValue, long expectedVersion)
    {
        if (!records.TryGetValue(key, out long[]? record)) return false;
        if (record[1] != expectedVersion) return false;   // stale read - the caller retries
        record[0] = newValue;
        record[1] = expectedVersion + 1;
        return true;
    }
}

/// <summary>A client with a read snapshot; Attempt commits or reports a conflict.</summary>
public class OccClient
{
    private readonly VersionedStore store;
    private readonly string key;

    public OccClient(string name, VersionedStore store, string key)
    {
        Name = name;
        this.store = store;
        this.key = key;
    }

    public string Name { get; }
    public long[] Snapshot { get; private set; } = Array.Empty<long>();
    public int Retries { get; private set; }

    public void Read() => Snapshot = store.Read(key);

    public bool Attempt(long newValue)
    {
        bool ok = store.Commit(key, newValue, Snapshot[1]);
        if (!ok) Retries++;
        return ok;
    }
}

public static class Solution
{
    public static void Main()
    {
        var store = new VersionedStore();
        store.Create("counter", 10);

        // scripted interleaving: A and B both read version 1 before anyone writes.
        // Single-threaded simulation, deterministic in every language; production OCC
        // would compare-and-swap the version field atomically.
        var a = new OccClient("A", store, "counter");
        var b = new OccClient("B", store, "counter");
        a.Read();
        b.Read();
        Check(a.Snapshot[0] == 10 && a.Snapshot[1] == 1, "A read (10, 1)");
        Check(b.Snapshot[0] == 10 && b.Snapshot[1] == 1, "B read (10, 1)");

        Check(a.Attempt(11), "A commits first");
        Check(store.Read("counter")[0] == 11 && store.Read("counter")[1] == 2, "store now (11, 2)");
        Check(!b.Attempt(11), "B's stale commit is rejected");
        Check(b.Retries == 1, "B counted one retry");
        Check(store.Read("counter")[0] == 11 && store.Read("counter")[1] == 2, "rejected write changed nothing");

        b.Read();
        Check(b.Snapshot[0] == 11 && b.Snapshot[1] == 2, "B re-read (11, 2)");
        Check(b.Attempt(12), "B retries and wins");
        Check(a.Retries == 0 && b.Retries == 1, "retry counts A 0, B 1");
        Check(store.Read("counter")[0] == 12 && store.Read("counter")[1] == 3, "final state (12, 3): 10 + 1 + 1");

        Check(!store.Commit("counter", 99, 1), "stale commit long after is still rejected");
        Check(store.Read("counter")[0] == 12, "value unchanged");
        Check(!store.Commit("missing", 1, 1), "unknown key never commits");
        try
        {
            store.Create("counter", 0);
            throw new Exception("FAILED: expected ArgumentException on duplicate create");
        }
        catch (ArgumentException)
        {
            // documented behaviour
        }

        // three clients, deterministic round-robin scheduler, retry until success
        store.Create("total", 0);
        var clients = new List<OccClient>();
        foreach (string name in new[] { "X", "Y", "Z" }) clients.Add(new OccClient(name, store, "total"));
        foreach (OccClient client in clients) client.Read();   // all read version 1 before anyone writes
        var pending = new List<OccClient>(clients);
        int rounds = 0;
        while (pending.Count > 0)
        {
            rounds++;
            var stillPending = new List<OccClient>();
            foreach (OccClient client in pending)
            {
                if (!client.Attempt(client.Snapshot[0] + 1))
                {
                    client.Read();                         // conflict -> refresh, retry later
                    stillPending.Add(client);
                }
            }
            pending = stillPending;
        }
        Check(rounds == 3, "X, then Y, then Z");
        Check(clients[0].Retries == 0 && clients[1].Retries == 1 && clients[2].Retries == 2, "round-robin retry counts [0, 1, 2]");
        Check(store.Read("total")[0] == 3 && store.Read("total")[1] == 4, "no lost update: 0 + 3 increments");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
