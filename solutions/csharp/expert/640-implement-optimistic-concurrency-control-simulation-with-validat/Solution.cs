// Task 640 - Implement optimistic concurrency control simulation with validation phase. (Tier: Expert, CTCI Ch 14 - Databases)
//
// The three-phase OCC protocol, simulated deterministically. In the READ phase a transaction reads
// the committed database (each read records the key's version number) and buffers its NEW values in
// a private workspace. The round-robin operator loop explicitly interleaves the read phases. In the
// VALIDATION phase - which a single global latch makes atomic together with the WRITE phase - a
// transaction is accepted only if
//   * read set: every key it read still carries the version it saw (nobody changed it since the read),
//   * write set: every key it writes *blind* (never read) still carries a version no newer than the
//     transaction start version, which is what stops two blind writers from clobbering each other.
// On success the buffered writes are installed with fresh version numbers; on failure the workspace is
// discarded (nothing partial ever reaches the database) and the transaction must be retried.
//
// Simplified: one operation per transaction per round, a counter instead of real timestamps, no
// parallel validation window and no retry scheduling (the test drives retries explicitly).
// Verified: the read-set check catches a reader that overlaps a writer, blind write-write conflicts
// abort the loser, a static read-only workload never aborts, a single validation order reproduces
// the whole run as a serial replay, and - as the negative check that motivates the latch - running
// validation and writing non-atomically lets both of two conflicting transactions commit and lose an
// update. Randomised workloads are checked for the OCC guarantee that the committed set replays
// serially in validation order.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal sealed class VersionedStore
{
    private readonly Dictionary<string, int> _values = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _versions = new Dictionary<string, int>(StringComparer.Ordinal);
    private int _global;

    public VersionedStore(SortedDictionary<string, int> initial)
    {
        foreach (KeyValuePair<string, int> seed in initial)
        {
            _global++;
            _values[seed.Key] = seed.Value;
            _versions[seed.Key] = _global;
        }
    }

    public int GlobalVersion => _global;
    public int Value(string key) => _values.TryGetValue(key, out int value) ? value : 0;
    public int Version(string key) => _versions.TryGetValue(key, out int version) ? version : 0;

    public void Install(string key, int value)
    {
        _global++;
        _values[key] = value;
        _versions[key] = _global;
    }

    public string Fingerprint()
    {
        var sb = new StringBuilder();
        foreach (string key in _values.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            sb.Append(key).Append('=').Append(_values[key]).Append(';');
        }
        return sb.ToString();
    }
}

internal sealed class Op
{
    public readonly bool IsRead;
    public readonly bool IsIncrement;
    public readonly string Key;
    public readonly int Value;

    private Op(bool isRead, bool isIncrement, string key, int value)
    {
        IsRead = isRead;
        IsIncrement = isIncrement;
        Key = key;
        Value = value;
    }

    public static Op Read(string key) => new Op(true, false, key, 0);
    public static Op Write(string key, int value) => new Op(false, false, key, value);
    public static Op Add(string key, int delta) => new Op(false, true, key, delta);
}

internal sealed class Body
{
    public readonly string Name;
    public readonly List<Op> Ops;
    public readonly int StartRound;

    public Body(string name, int startRound, params Op[] ops)
    {
        Name = name;
        StartRound = startRound;
        Ops = new List<Op>(ops);
    }
}

internal sealed class OccWorkload
{
    public readonly string Name;
    public readonly List<Body> Bodies;
    public readonly List<int> ValidationOrder;
    public readonly SortedDictionary<string, int> Initial;

    public OccWorkload(string name, List<Body> bodies, List<int> validationOrder, SortedDictionary<string, int> initial)
    {
        Name = name;
        Bodies = bodies;
        ValidationOrder = validationOrder;
        Initial = initial;
    }
}

internal sealed class OccTxn
{
    public readonly Body Body;
    public readonly int StartVersion;
    public readonly SortedDictionary<string, int> ReadValues = new SortedDictionary<string, int>(StringComparer.Ordinal);
    public readonly SortedDictionary<string, int> ReadVersions = new SortedDictionary<string, int>(StringComparer.Ordinal);
    public readonly SortedDictionary<string, int> Workspace = new SortedDictionary<string, int>(StringComparer.Ordinal);
    public readonly List<string> Log = new List<string>();

    public bool Committed;
    public bool Aborted;
    public int ValidatedAt = -1;
    public string Failure = "";

    public OccTxn(Body body, int startVersion)
    {
        Body = body;
        StartVersion = startVersion;
    }
}

internal sealed class OccExecution
{
    public readonly OccWorkload Workload;
    public readonly List<OccTxn> Txns;
    public readonly string Final;
    public readonly List<string> Trace;

    public OccExecution(OccWorkload workload, List<OccTxn> txns, string final, List<string> trace)
    {
        Workload = workload;
        Txns = txns;
        Final = final;
        Trace = trace;
    }

    public int CommitCount => Txns.Count(t => t.Committed);
    public int AbortCount => Txns.Count(t => t.Aborted);
}

internal static class OccEngine
{
    // Round-robin read phase (one operation per transaction per round) followed by the validation and
    // write phases. `atomicLatch` is the correct protocol: validate-and-install under one latch. With
    // it disabled every transaction validates against the same pre-write state and then all of them
    // write, which is precisely the race OCC needs the latch to avoid.
    public static OccExecution Run(OccWorkload workload, bool atomicLatch)
    {
        var store = new VersionedStore(workload.Initial);
        var txns = new List<OccTxn>();
        foreach (Body body in workload.Bodies) txns.Add(new OccTxn(body, store.GlobalVersion));
        var trace = new List<string>();

        int rounds = 0;
        if (txns.Count > 0) rounds = Math.Max(1, txns.Max(t => t.Body.StartRound + t.Body.Ops.Count));
        for (int round = 0; round < rounds; round++)
        {
            for (int i = 0; i < txns.Count; i++)
            {
                OccTxn txn = txns[i];
                int index = round - txn.Body.StartRound;
                if (index < 0 || index >= txn.Body.Ops.Count) continue;
                Execute(txn, txn.Body.Ops[index], store);
            }
        }

        int validations = 0;
        if (atomicLatch)
        {
            foreach (int index in workload.ValidationOrder)
            {
                OccTxn txn = txns[index];
                string failure = Validate(txn, store);
                validations++;
                if (failure.Length != 0)
                {
                    txn.Aborted = true;
                    txn.Failure = failure;
                    trace.Add(txn.Body.Name + " ABORT " + failure);
                    continue;
                }
                Install(txn, store, validations);
            }
        }
        else
        {
            var accepted = new List<OccTxn>();
            foreach (int index in workload.ValidationOrder)
            {
                OccTxn txn = txns[index];
                string failure = Validate(txn, store);
                if (failure.Length != 0)
                {
                    txn.Aborted = true;
                    txn.Failure = failure;
                    trace.Add(txn.Body.Name + " ABORT " + failure);
                    continue;
                }
                accepted.Add(txn);
            }
            foreach (OccTxn txn in accepted) Install(txn, store, ++validations);
        }

        return new OccExecution(workload, txns, store.Fingerprint(), trace);
    }

    private static void Install(OccTxn txn, VersionedStore store, int position)
    {
        foreach (KeyValuePair<string, int> write in txn.Workspace) store.Install(write.Key, write.Value);
        txn.Committed = true;
        txn.ValidatedAt = position;
    }

    private static void Execute(OccTxn txn, Op op, VersionedStore store)
    {
        if (op.IsRead)
        {
            Read(txn, op.Key, store);
            return;
        }
        if (op.IsIncrement)
        {
            if (!txn.ReadVersions.ContainsKey(op.Key)) Read(txn, op.Key, store);
            txn.Workspace[op.Key] = txn.ReadValues[op.Key] + op.Value;
            return;
        }
        txn.Workspace[op.Key] = op.Value;
    }

    private static void Read(OccTxn txn, string key, VersionedStore store)
    {
        txn.ReadValues[key] = store.Value(key);
        txn.ReadVersions[key] = store.Version(key);
    }

    private static string Validate(OccTxn txn, VersionedStore store)
    {
        foreach (KeyValuePair<string, int> read in txn.ReadVersions)
        {
            if (store.Version(read.Key) != read.Value) return "read set changed: " + read.Key;
        }
        foreach (string key in txn.Workspace.Keys)
        {
            if (txn.ReadVersions.ContainsKey(key)) continue;
            if (store.Version(key) > txn.StartVersion) return "blind write conflict: " + key;
        }
        return "";
    }
}

public static class Solution
{
    private static int _seed = 1234567;

    private static int NextInt(int bound)
    {
        _seed = (int)((_seed * 1103515245L + 12345L) & 0x7FFFFFFF);
        return (_seed >> 8) % bound;
    }

    private static SortedDictionary<string, int> Seed(params (string Key, int Value)[] pairs)
    {
        var initial = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach ((string key, int value) in pairs) initial[key] = value;
        return initial;
    }

    private static List<int> Order(params int[] indices) => new List<int>(indices);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    // T1 reads and rewrites k, T2 does the same; T2's read of k is invalidated by T1's commit.
    private static OccWorkload ReaderWriterConflict()
    {
        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("k"), Op.Write("k", 10)),
            new Body("T2", 0, Op.Read("k"), Op.Write("k", 20)),
        };
        return new OccWorkload("rw-conflict", bodies, Order(0, 1), Seed(("k", 1)));
    }

    private static void ReadSetValidation()
    {
        OccExecution execution = OccEngine.Run(ReaderWriterConflict(), true);
        Check(execution.CommitCount == 1 && execution.AbortCount == 1, "read-set check: exactly one transaction commits");
        Check(execution.Txns[1].Aborted && execution.Txns[1].Failure.StartsWith("read set changed", StringComparison.Ordinal),
            "read-set check: the loser reports a changed read set");
        Check(execution.Final == "k=10;", "read-set check: the loser's workspace never reached the database");
        Check(execution.Txns[1].Workspace["k"] == 20, "read-set check: the loser's workspace is kept in memory only");
        Check(ValidationOrderReplayMatches(execution), "read-set check: committed set replays serially in validation order");

        // The retry starts after T1 committed, so it observes 10 and validates.
        var retry = new OccWorkload("retry", new List<Body> { new Body("T2", 0, Op.Read("k"), Op.Write("k", 20)) },
            Order(0), Seed(("k", 10)));
        OccExecution second = OccEngine.Run(retry, true);
        Check(second.CommitCount == 1 && second.Final == "k=20;", "read-set check: the retried transaction commits");
        Console.WriteLine("read-set validation: overlapping writer wins, reader aborts (workspace discarded), retry commits");
    }

    private static void BlindWriteConflictAndLatency()
    {
        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("c"), Op.Write("c", 1)),
            new Body("T2", 0, Op.Read("c"), Op.Write("c", 2)),
            new Body("T3", 0, Op.Write("c", 3)),
        };
        var workload = new OccWorkload("mixed", bodies, Order(0, 1, 2), Seed(("c", 0)));

        OccExecution atomic = OccEngine.Run(workload, true);
        Check(atomic.CommitCount == 1 && atomic.AbortCount == 2, "atomic latch: only the first validator survives");
        Check(atomic.Final == "c=1;", "atomic latch: winner's value is the one installed");
        Check(ValidationOrderReplayMatches(atomic), "atomic latch: validation-order replay matches");

        OccExecution racy = OccEngine.Run(workload, false);
        Check(racy.CommitCount == 3, "non-atomic latch: all three transactions pass validation");
        Check(racy.Final == "c=3;", "non-atomic latch: the last writer silently overwrites the others");
        Console.WriteLine("latch necessity: atomic run commits 1 of 3, non-atomic run commits 3 (lost updates)");
    }

    private static void LostUpdateWithIncrements()
    {
        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("counter"), Op.Add("counter", 1)),
            new Body("T2", 0, Op.Read("counter"), Op.Add("counter", 1)),
        };
        var workload = new OccWorkload("lost-update", bodies, Order(0, 1), Seed(("counter", 0)));

        OccExecution atomic = OccEngine.Run(workload, true);
        Check(atomic.CommitCount == 1 && atomic.AbortCount == 1, "lost update (atomic): one increment is rejected");
        Check(atomic.Final == "counter=1;", "lost update (atomic): counter is 1 before the retry");

        OccExecution racy = OccEngine.Run(workload, false);
        Check(racy.Final == "counter=1;", "lost update (non-atomic): two increments still land as one");
        Console.WriteLine("lost update: T1,T2 both increment the same counter; non-atomic wording loses one increment");
    }

    private static void ReadOnlyWorkloadNeverAborts()
    {
        var bodies = new List<Body>
        {
            new Body("R1", 0, Op.Read("a"), Op.Read("b")),
            new Body("R2", 0, Op.Read("b"), Op.Read("c")),
            new Body("R3", 0, Op.Read("c"), Op.Read("a"), Op.Read("missing")),
        };
        var workload = new OccWorkload("read-only", bodies, Order(2, 0, 1), Seed(("a", 1), ("b", 2), ("c", 3)));
        OccExecution execution = OccEngine.Run(workload, true);
        Check(execution.CommitCount == 3 && execution.AbortCount == 0, "read-only workload: every reader validates");
        Check(execution.Txns[2].ReadValues["missing"] == 0, "read-only workload: a missing key reads as 0");
        Check(execution.Final == "a=1;b=2;c=3;", "read-only workload: database untouched");
        Check(execution.Txns[0].ValidatedAt < execution.Txns[1].ValidatedAt, "read-only workload: validation order respected");
    }

    private static void Boundaries()
    {
        OccExecution empty = OccEngine.Run(new OccWorkload("empty", new List<Body>(), Order(), Seed(("a", 1))), true);
        Check(empty.CommitCount == 0 && empty.Final == "a=1;", "empty workload: no transactions, no writes");

        var overwrite = new OccWorkload("overwrite",
            new List<Body> { new Body("T", 0, Op.Write("k", 1), Op.Write("k", 2), Op.Write("k", 3)) },
            Order(0), Seed(("k", 0)));
        OccExecution repeated = OccEngine.Run(overwrite, true);
        Check(repeated.Final == "k=3;", "workspace: the last write to a key wins");

        var noOps = new OccWorkload("no-ops", new List<Body> { new Body("T", 0) }, Order(0), Seed(("k", 0)));
        Check(OccEngine.Run(noOps, true).CommitCount == 1, "a transaction without operations validates");

        var reread = new OccWorkload("re-read",
            new List<Body> { new Body("T", 0, Op.Read("k"), Op.Read("k"), Op.Write("k", 9)) },
            Order(0), Seed(("k", 5)));
        OccExecution twice = OccEngine.Run(reread, true);
        Check(twice.Txns[0].ReadVersions.Count == 1, "re-reading a key keeps a single read version entry");
        Check(twice.Final == "k=9;", "re-reading then writing works");
    }

    private static OccWorkload RandomWorkload(int id, string[] keys)
    {
        var initial = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (string key in keys) initial[key] = NextInt(3);

        int count = 2 + NextInt(3);
        var bodies = new List<Body>();
        for (int i = 0; i < count; i++)
        {
            var ops = new List<Op>();
            int steps = 1 + NextInt(3);
            for (int s = 0; s < steps; s++)
            {
                string key = keys[NextInt(keys.Length)];
                int kind = NextInt(3);
                if (kind == 0) ops.Add(Op.Read(key));
                else if (kind == 1) ops.Add(Op.Write(key, NextInt(5)));
                else ops.Add(Op.Add(key, 1 + NextInt(2)));
            }
            bodies.Add(new Body("T" + (i + 1), 0, ops.ToArray()));
        }

        var order = new List<int>();
        for (int i = 0; i < count; i++) order.Add(i);
        for (int i = count - 1; i > 0; i--)
        {
            int j = NextInt(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return new OccWorkload("random-" + id, bodies, order, initial);
    }

    private static void RandomisedValidationSweep()
    {
        string[] keys = { "a", "b", "c" };
        int aborts = 0, commits = 0, latchMatters = 0;
        for (int id = 0; id < 150; id++)
        {
            OccWorkload workload = RandomWorkload(id, keys);
            OccExecution execution = OccEngine.Run(workload, true);
            commits += execution.CommitCount;
            aborts += execution.AbortCount;

            // The OCC theorem under test: committed transactions are exactly the ones that replay
            // serially in validation order, with identical reads, writes and final state.
            Check(ValidationOrderReplayMatches(execution), "validation-order replay (" + workload.Name + ")");

            // Negative property: without the atomic validate+write latch even the randomised
            // workloads stop being serialisable, and the latch run can only be more conservative.
            OccExecution racy = OccEngine.Run(workload, false);
            Check(racy.CommitCount >= execution.CommitCount, "latch run commits no more than the racy run (" + workload.Name + ")");
            if (!ValidationOrderReplayMatches(racy)) latchMatters++;

            OccExecution again = OccEngine.Run(workload, true);
            Check(again.Final == execution.Final && again.AbortCount == execution.AbortCount,
                "deterministic rerun (" + workload.Name + ")");
        }
        Check(aborts > 0, "the sweep produced real validation failures");
        Check(latchMatters > 0, "the sweep found workloads where a non-atomic latch loses serialisability");
        Console.WriteLine("random sweep: 150 workloads, " + commits + " commits, " + aborts + " aborts, "
            + latchMatters + " became non-serialisable without the latch");
    }

    // Replays the committed transactions one at a time in validation order on a fresh database and
    // demands identical reads, identical workspace values and the identical final state.
    private static bool ValidationOrderReplayMatches(OccExecution execution)
    {
        var store = new VersionedStore(execution.Workload.Initial);
        foreach (OccTxn txn in execution.Txns.Where(t => t.Committed).OrderBy(t => t.ValidatedAt))
        {
            var reads = new SortedDictionary<string, int>(StringComparer.Ordinal);
            var writes = new SortedDictionary<string, int>(StringComparer.Ordinal);
            foreach (Op op in txn.Body.Ops)
            {
                if (op.IsRead)
                {
                    reads[op.Key] = store.Value(op.Key);
                    continue;
                }
                if (op.IsIncrement)
                {
                    if (!reads.ContainsKey(op.Key)) reads[op.Key] = store.Value(op.Key);
                    writes[op.Key] = reads[op.Key] + op.Value;
                    continue;
                }
                writes[op.Key] = op.Value;
            }
            if (reads.Count != txn.ReadValues.Count) return false;
            foreach (KeyValuePair<string, int> read in reads)
            {
                if (!txn.ReadValues.TryGetValue(read.Key, out int seen) || seen != read.Value) return false;
            }
            if (writes.Count != txn.Workspace.Count) return false;
            foreach (KeyValuePair<string, int> write in writes)
            {
                if (!txn.Workspace.TryGetValue(write.Key, out int seen) || seen != write.Value) return false;
                store.Install(write.Key, write.Value);
            }
        }
        return store.Fingerprint() == execution.Final;
    }

    public static void Main()
    {
        ReadSetValidation();
        BlindWriteConflictAndLatency();
        LostUpdateWithIncrements();
        ReadOnlyWorkloadNeverAborts();
        Boundaries();
        RandomisedValidationSweep();
        Console.WriteLine("PASS");
    }
}
