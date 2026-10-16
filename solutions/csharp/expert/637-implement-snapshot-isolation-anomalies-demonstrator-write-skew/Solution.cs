// Task 637 - Implement SNAPSHOT isolation anomalies demonstrator (write skew). (Tier: Expert, CTCI Ch 14 - Databases)
//
// A miniature multiversion store: every key keeps a history of (commitTs, value) versions and a
// transaction reads the newest version whose commitTs is <= the commit clock at its BEGIN. Writes
// go to a private buffer and are installed atomically at COMMIT, which enforces first-committer-wins:
// the commit aborts when a key it wrote received a newer version after the transaction's snapshot.
// There are no threads - every schedule is an explicitly interleaved statement list - so the whole
// demonstrator is deterministic.
//
// Simplified relative to a real engine: values are ints, there are no deletes / version chain
// collections, no predicate (range) reads and no garbage collection; conflicts are detected per key
// only. The invariants verified are the interesting ones:
//   * write skew - the classic "at least one doctor on call" schedule commits BOTH transactions
//     under snapshot isolation and breaks the invariant, while the serial schedule of the very same
//     transactions keeps it (each transaction only reads state the other transaction writes);
//   * lost updates are prevented - the second committer hits a write-write conflict and aborts;
//   * an aborted transaction leaves no trace, uncommitted data is invisible (no dirty reads) and a
//     read-only transaction always commits against its own stable snapshot.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class Version
{
    public readonly long CommitTs;
    public readonly int Value;

    public Version(long commitTs, int value)
    {
        CommitTs = commitTs;
        Value = value;
    }
}

internal sealed class SnapshotStore
{
    private readonly Dictionary<string, List<Version>> _history = new Dictionary<string, List<Version>>();

    // Commit clock: incremented once per successful commit, so "snapshot ts" == number of commits
    // visible at BEGIN.
    public long Clock { get; private set; }

    public void Seed(string key, int value)
    {
        List<Version> versions = History(key);
        versions.Add(new Version(0, value));
    }

    private List<Version> History(string key)
    {
        if (!_history.TryGetValue(key, out List<Version> versions))
        {
            versions = new List<Version>();
            _history[key] = versions;
        }
        return versions;
    }

    // Newest version visible at snapshotTs; a key with no visible version reads as 0 (missing row).
    public int ReadAt(string key, long snapshotTs)
    {
        if (!_history.TryGetValue(key, out List<Version> versions)) return 0;
        for (int i = versions.Count - 1; i >= 0; i--)
        {
            if (versions[i].CommitTs <= snapshotTs) return versions[i].Value;
        }
        return 0;
    }

    public long LatestCommitTs(string key)
    {
        if (!_history.TryGetValue(key, out List<Version> versions) || versions.Count == 0) return -1;
        return versions[versions.Count - 1].CommitTs;
    }

    public void Install(string key, int value, long commitTs)
    {
        History(key).Add(new Version(commitTs, value));
    }

    public long NextCommitTs()
    {
        Clock++;
        return Clock;
    }

    public int Sum(params string[] keys)
    {
        int total = 0;
        foreach (string key in keys) total += ReadAt(key, Clock);
        return total;
    }
}

internal sealed class Txn
{
    private readonly SnapshotStore _store;
    private readonly Dictionary<string, int> _readValues = new Dictionary<string, int>();
    private readonly Dictionary<string, int> _writeBuffer = new Dictionary<string, int>();

    public readonly string Name;
    public readonly long StartTs;
    public readonly List<string> Log = new List<string>();

    public bool Committed { get; private set; }
    public bool Aborted { get; private set; }

    public Txn(SnapshotStore store, string name, long startTs)
    {
        _store = store;
        Name = name;
        StartTs = startTs;
    }

    public IEnumerable<string> ReadSet => _readValues.Keys.OrderBy(k => k, StringComparer.Ordinal);
    public IEnumerable<string> WriteSet => _writeBuffer.Keys.OrderBy(k => k, StringComparer.Ordinal);

    public int Read(string key)
    {
        RequireActive("read");
        int value = _store.ReadAt(key, StartTs);
        _readValues[key] = value;
        Log.Add(Name + " read " + key + "=" + value + " (snapshot " + StartTs + ")");
        return value;
    }

    public void Write(string key, int value)
    {
        RequireActive("write");
        _writeBuffer[key] = value;
        Log.Add(Name + " write " + key + "=" + value + " (buffered)");
    }

    private void RequireActive(string op)
    {
        if (Committed) throw new Exception(Name + ": " + op + " after commit");
        if (Aborted) throw new Exception(Name + ": " + op + " after abort");
    }

    // First-committer-wins: any key written here that changed after our snapshot means somebody else
    // committed concurrently, so this transaction is rolled back (its buffered writes are dropped).
    public bool Commit()
    {
        if (Aborted) throw new Exception(Name + ": commit after abort");
        if (Committed) throw new Exception(Name + ": double commit");

        foreach (string key in _writeBuffer.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            long latest = _store.LatestCommitTs(key);
            if (latest > StartTs)
            {
                Aborted = true;
                Log.Add(Name + " ABORT write-write conflict on " + key + " (version " + latest + " > snapshot " + StartTs + ")");
                return false;
            }
        }

        long commitTs = _store.NextCommitTs();
        foreach (KeyValuePair<string, int> write in _writeBuffer) _store.Install(write.Key, write.Value, commitTs);
        Committed = true;
        Log.Add(Name + " commit ts=" + commitTs + " writes=[" + string.Join(",", WriteSet) + "]");
        return true;
    }
}

internal sealed class Engine
{
    private int _serial;

    public readonly SnapshotStore Store = new SnapshotStore();
    public readonly List<Txn> Transactions = new List<Txn>();

    public int Aborts => Transactions.Count(t => t.Aborted);

    public Txn Begin(string name)
    {
        if (name.Length == 0)
        {
            _serial++;
            name = "T" + _serial;
        }
        var txn = new Txn(Store, name, Store.Clock);
        Transactions.Add(txn);
        return txn;
    }

    public void Report()
    {
        foreach (Txn txn in Transactions)
        {
            foreach (string line in txn.Log) Console.WriteLine("  " + line);
        }
    }
}

public static class Solution
{
    // "At least one doctor must stay on call": rows on_call(x) and on_call(y) start at 1 and each
    // transaction steps down only if it sees two doctors on call.
    private static void WriteSkewIsAllowed()
    {
        var engine = new Engine();
        engine.Store.Seed("x", 1);
        engine.Store.Seed("y", 1);

        Txn t1 = engine.Begin("T1");
        Txn t2 = engine.Begin("T2");
        int t1x = t1.Read("x"), t2x = t2.Read("x");
        int t1y = t1.Read("y"), t2y = t2.Read("y");
        if (t1x + t1y >= 2) t1.Write("x", 0);
        if (t2x + t2y >= 2) t2.Write("y", 0);

        bool c1 = t1.Commit();
        bool c2 = t2.Commit();

        Check(c1 && c2, "write skew: both transactions commit under snapshot isolation");
        Check(t1.WriteSet.Count() == 1 && t2.WriteSet.Count() == 1, "write skew: each transaction wrote one row");
        Check(engine.Store.Sum("x", "y") == 0, "write skew: invariant x+y>=1 is VIOLATED (x+y=0)");
        Check(engine.Store.ReadAt("z", engine.Store.Clock) == 0, "write skew: unknown row reads as 0");
        Console.WriteLine("write skew under SI: rows on_call x,y 1,1 -> 0,0, invariant x+y>=1 broken");
        engine.Report();
    }

    // The very same transactions run without overlap: the second one reads the first one's write and
    // therefore declines to step down, so the invariant survives. This is the "serial schedule"
    // reference the write-skew run is compared against.
    private static void SerialExecutionKeepsInvariant()
    {
        var engine = new Engine();
        engine.Store.Seed("x", 1);
        engine.Store.Seed("y", 1);

        Txn a = engine.Begin("A");
        int ax = a.Read("x"), ay = a.Read("y");
        if (ax + ay >= 2) a.Write("x", 0);
        Check(a.Commit(), "serial: first transaction commits");

        Txn b = engine.Begin("B");
        int bx = b.Read("x"), by = b.Read("y");
        Check(bx == 0 && by == 1, "serial: second transaction sees the first one's write (0,1)");
        if (bx + by >= 2) b.Write("y", 0);
        Check(b.Commit(), "serial: second transaction commits");
        Check(b.WriteSet.Count() == 0, "serial: second transaction writes nothing because x+y=1");
        Check(engine.Store.Sum("x", "y") == 1, "serial: invariant x+y>=1 holds");
        Check(engine.Aborts == 0, "serial: no aborts needed");
        Console.WriteLine("serial order: rows on_call x,y 1,1 -> 0,1, invariant x+y>=1 preserved");
    }

    private static void LostUpdateIsPrevented()
    {
        var engine = new Engine();
        engine.Store.Seed("c", 0);

        Txn t1 = engine.Begin("T1");
        Txn t2 = engine.Begin("T2");
        t1.Write("c", t1.Read("c") + 1);
        t2.Write("c", t2.Read("c") + 1);

        Check(t1.Commit(), "lost update: first committer wins");
        Check(!t2.Commit(), "lost update: second committer aborts (write-write conflict)");
        Check(t2.Aborted && !t2.Committed, "lost update: loser is marked aborted");
        Check(engine.Store.ReadAt("c", engine.Store.Clock) == 1, "lost update: counter still 1 after the abort");

        Txn retry = engine.Begin("T2-retry");
        retry.Write("c", retry.Read("c") + 1);
        Check(retry.Commit(), "lost update: retried transaction commits");
        Check(engine.Store.ReadAt("c", engine.Store.Clock) == 2, "lost update: two increments survive, counter 2");
        Check(engine.Aborts == 1, "lost update: exactly one abort");
        Console.WriteLine("lost update: 2 concurrent c=c+1 -> 1 abort, retry -> c=2");
    }

    private static void DisjointWritesCommute()
    {
        var engine = new Engine();
        engine.Store.Seed("a", 0);
        engine.Store.Seed("b", 0);

        Txn t1 = engine.Begin("T1");
        Txn t2 = engine.Begin("T2");
        t1.Write("a", 10);
        t2.Write("b", 20);
        Check(t1.Commit() && t2.Commit(), "disjoint writes: both commit");
        Check(engine.Store.ReadAt("a", engine.Store.Clock) == 10, "disjoint writes: a=10");
        Check(engine.Store.ReadAt("b", engine.Store.Clock) == 20, "disjoint writes: b=20");
        Check(t1.ReadSet.Count() == 0 && t2.ReadSet.Count() == 0, "disjoint writes: neither transaction read anything");
    }

    private static void NoDirtyReads()
    {
        var engine = new Engine();
        engine.Store.Seed("k", 5);

        Txn writer = engine.Begin("W");
        Txn reader = engine.Begin("R");
        writer.Write("k", 99);
        Check(reader.Read("k") == 5, "dirty read: uncommitted write is invisible");
        Check(!writer.Committed, "dirty read: writer still uncommitted");
        Check(reader.Commit(), "dirty read: read-only transaction commits");
        Check(engine.Store.ReadAt("k", engine.Store.Clock) == 5, "dirty read: committed state unchanged before writer commits");
        Check(writer.Commit(), "dirty read: writer commits afterwards");
        Check(engine.Store.ReadAt("k", engine.Store.Clock) == 99, "dirty read: writer's value visible after commit");
    }

    private static void ReadOnlySnapshotIsStable()
    {
        var engine = new Engine();
        engine.Store.Seed("s", 1);

        Txn reader = engine.Begin("R");
        Check(reader.Read("s") == 1, "read-only: sees the value at BEGIN");

        Txn writer = engine.Begin("W");
        writer.Write("s", 2);
        Check(writer.Commit(), "read-only: writer commits");
        Check(reader.Read("s") == 1, "read-only: snapshot is stable despite the newer commit");
        Check(reader.Commit(), "read-only: never aborts on a write-write conflict");
        Check(engine.Store.ReadAt("s", engine.Store.Clock) == 2, "read-only: newest committed value is 2");
    }

    private static void AbortedWritesAreDiscarded()
    {
        var engine = new Engine();
        engine.Store.Seed("c", 0);

        Txn winner = engine.Begin("W");
        Txn loser = engine.Begin("L");
        winner.Write("c", 7);
        loser.Write("c", 100);
        Check(winner.Commit(), "discard: winner commits");
        Check(!loser.Commit(), "discard: loser aborts");
        Check(engine.Store.ReadAt("c", engine.Store.Clock) == 7, "discard: loser's buffered write 100 never landed");
        Check(engine.Store.ReadAt("c", loser.StartTs) == 0, "discard: loser's snapshot still shows the pre-conflict value");

        bool threw = false;
        try { loser.Commit(); }
        catch (Exception) { threw = true; }
        Check(threw, "discard: an aborted transaction refuses to commit again");

        bool readAfterAbort = false;
        try { loser.Read("c"); }
        catch (Exception) { readAfterAbort = true; }
        Check(readAfterAbort, "discard: an aborted transaction refuses further reads");
    }

    public static void Main()
    {
        WriteSkewIsAllowed();
        SerialExecutionKeepsInvariant();
        LostUpdateIsPrevented();
        DisjointWritesCommute();
        NoDirtyReads();
        ReadOnlySnapshotIsStable();
        AbortedWritesAreDiscarded();
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
