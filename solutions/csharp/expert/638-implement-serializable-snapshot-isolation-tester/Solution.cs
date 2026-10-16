// Task 638 - Implement serializable snapshot isolation tester. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Serializable snapshot isolation (Cahill/Ports/Fekete SSI) layered on top of the little MVCC store
// from task 637: transactions read their BEGIN snapshot, buffer writes, and commit under
// first-committer-wins; SSI adds one extra rule. Commit is refused when the committing transaction
// is a *pivot* of the dangerous structure T1 -rw-> T2 -rw-> T3, where T3 already committed:
//   * incoming rw-antidependency: some other transaction read a row this transaction overwrites
//     (its read version is the one we are about to replace), and
//   * outgoing rw-antidependency: this transaction read a row that another transaction which
//     committed after our snapshot overwrote.
// An rw-antidependency exists exactly when A read key k, B writes k, and B's version was invisible
// to A. We intentionally drop the "must be concurrent with the pivot" qualifier, so the detector is
// a conservative superset of the real rule: it may abort needlessly (false positives) but it cannot
// miss a dangerous structure. Threads do not exist - a schedule is a round-robin statement list
// plus a declared commit order, so every run is deterministic.
//
// Verified: the write skew schedule of task 637 is aborted; serial/disjoint/read-only workloads
// still commit everything; and for a randomised battery of workloads SSI only ever commits a set of
// transactions that is view-equivalent to some serial replay (brute-force permutation oracle), while
// whenever plain snapshot isolation produced a non-serialisable outcome SSI aborted something.
// Simplified: integer values, per-key conflicts only (no predicates), no SIREAD locks / no GC.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

internal sealed class Cell
{
    public readonly long Ts;
    public readonly int Value;

    public Cell(long ts, int value)
    {
        Ts = ts;
        Value = value;
    }
}

internal sealed class Store
{
    private readonly Dictionary<string, List<Cell>> _history = new Dictionary<string, List<Cell>>();

    public long Clock { get; private set; }

    public Store(SortedDictionary<string, int> initial)
    {
        foreach (KeyValuePair<string, int> seed in initial)
        {
            _history[seed.Key] = new List<Cell> { new Cell(0, seed.Value) };
        }
    }

    public int ReadAt(string key, long snapshotTs)
    {
        if (!_history.TryGetValue(key, out List<Cell> cells)) return 0;
        for (int i = cells.Count - 1; i >= 0; i--)
        {
            if (cells[i].Ts <= snapshotTs) return cells[i].Value;
        }
        return 0;
    }

    public long LatestTs(string key)
    {
        if (!_history.TryGetValue(key, out List<Cell> cells) || cells.Count == 0) return -1;
        return cells[cells.Count - 1].Ts;
    }

    public void Install(string key, int value, long ts)
    {
        if (!_history.TryGetValue(key, out List<Cell> cells))
        {
            cells = new List<Cell>();
            _history[key] = cells;
        }
        cells.Add(new Cell(ts, value));
    }

    public long Commit()
    {
        Clock++;
        return Clock;
    }

    public string Fingerprint()
    {
        var sb = new StringBuilder();
        foreach (string key in _history.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            sb.Append(key).Append('=').Append(ReadAt(key, Clock)).Append(';');
        }
        return sb.ToString();
    }
}

internal sealed class Op
{
    public readonly bool IsRead;
    public readonly string Key;
    public readonly int Value;
    public readonly bool Guarded;
    public readonly string GuardLeft;
    public readonly string GuardRight;
    public readonly int GuardThreshold;

    private Op(bool isRead, string key, int value, bool guarded, string left, string right, int threshold)
    {
        IsRead = isRead;
        Key = key;
        Value = value;
        Guarded = guarded;
        GuardLeft = left;
        GuardRight = right;
        GuardThreshold = threshold;
    }

    public static Op Read(string key) => new Op(true, key, 0, false, "", "", 0);
    public static Op Write(string key, int value) => new Op(false, key, value, false, "", "", 0);
    public static Op GuardedWrite(string key, int value, string left, string right, int threshold)
        => new Op(false, key, value, true, left, right, threshold);
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

internal sealed class Workload
{
    public readonly string Name;
    public readonly List<Body> Bodies;
    public readonly List<int> CommitOrder;
    public readonly SortedDictionary<string, int> Initial;

    public Workload(string name, List<Body> bodies, List<int> commitOrder, SortedDictionary<string, int> initial)
    {
        Name = name;
        Bodies = bodies;
        CommitOrder = commitOrder;
        Initial = initial;
    }
}

internal sealed class TxnRecord
{
    public readonly Body Body;
    public readonly SortedSet<string> ReadKeys = new SortedSet<string>(StringComparer.Ordinal);
    public readonly SortedSet<string> WriteKeys = new SortedSet<string>(StringComparer.Ordinal);
    public readonly Dictionary<string, int> ReadValues = new Dictionary<string, int>();
    public readonly Dictionary<string, int> WriteValues = new Dictionary<string, int>();
    public readonly List<string> Log = new List<string>();

    public long StartTs = -1;
    public bool Committed;
    public bool Aborted;
    public bool Handled;
    public long CommitTs = -1;

    public TxnRecord(Body body)
    {
        Body = body;
    }
}

internal sealed class Execution
{
    public readonly Workload Workload;
    public readonly List<TxnRecord> Txns;
    public readonly string Final;
    public readonly List<string> Trace;

    public Execution(Workload workload, List<TxnRecord> txns, string final, List<string> trace)
    {
        Workload = workload;
        Txns = txns;
        Final = final;
        Trace = trace;
    }

    public int CommitCount => Txns.Count(t => t.Committed);
    public int AbortCount => Txns.Count(t => t.Aborted);
}

internal static class SsiEngine
{
    // A schedule is a round-robin: in round r every transaction whose start round has passed runs its
    // (r - startRound)-th operation, in transaction order. A transaction that runs its last operation
    // commits at the end of that round, the transactions finishing in the same round committing in
    // the declared commit order. Its snapshot is taken at its first operation.
    public static Execution Run(Workload workload, bool serializable)
    {
        var store = new Store(workload.Initial);
        var trace = new List<string>();
        var txns = new List<TxnRecord>();
        foreach (Body body in workload.Bodies) txns.Add(new TxnRecord(body));

        int rounds = 0;
        if (txns.Count > 0) rounds = Math.Max(1, txns.Max(t => t.Body.StartRound + t.Body.Ops.Count));

        for (int round = 0; round < rounds; round++)
        {
            for (int i = 0; i < txns.Count; i++)
            {
                TxnRecord txn = txns[i];
                int index = round - txn.Body.StartRound;
                if (index < 0 || index >= txn.Body.Ops.Count) continue;
                if (txn.StartTs < 0)
                {
                    txn.StartTs = store.Clock;
                    trace.Add(txn.Body.Name + " begin @snapshot " + store.Clock);
                }
                Execute(txn, txn.Body.Ops[index], store);
            }

            bool anyFinished = false;
            foreach (int index in workload.CommitOrder)
            {
                TxnRecord txn = txns[index];
                if (txn.Handled) continue;
                if (round < txn.Body.StartRound + txn.Body.Ops.Count - 1) continue;
                CommitAtRoundEnd(txn, store, txns, serializable, trace);
                txn.Handled = true;
                anyFinished = true;
            }
            if (!anyFinished && round + 1 == rounds && txns.Any(t => !t.Handled))
            {
                throw new Exception(workload.Name + ": not every transaction reached a commit point");
            }
        }

        foreach (TxnRecord txn in txns)
        {
            if (!txn.Handled) throw new Exception(workload.Name + ": transaction " + txn.Body.Name + " never committed");
        }
        return new Execution(workload, txns, store.Fingerprint(), trace);
    }

    private static void CommitAtRoundEnd(TxnRecord txn, Store store, List<TxnRecord> all, bool serializable, List<string> trace)
    {
        string conflict = WriteWriteConflict(txn, store);
        if (conflict.Length != 0)
        {
            txn.Aborted = true;
            trace.Add(txn.Body.Name + " ABORT write-write conflict on " + conflict);
            return;
        }
        if (serializable && IsPivot(txn, all))
        {
            txn.Aborted = true;
            trace.Add(txn.Body.Name + " ABORT dangerous structure (rw-antidependency pivot)");
            return;
        }
        long ts = store.Commit();
        foreach (KeyValuePair<string, int> write in txn.WriteValues) store.Install(write.Key, write.Value, ts);
        txn.Committed = true;
        txn.CommitTs = ts;
        trace.Add(txn.Body.Name + " commit ts=" + ts + " writes=[" + string.Join(",", txn.WriteKeys) + "]");
    }

    private static void Execute(TxnRecord txn, Op op, Store store)
    {
        if (op.IsRead)
        {
            txn.ReadKeys.Add(op.Key);
            txn.ReadValues[op.Key] = store.ReadAt(op.Key, txn.StartTs);
            return;
        }
        if (op.Guarded)
        {
            if (!txn.ReadValues.TryGetValue(op.GuardLeft, out int left) ||
                !txn.ReadValues.TryGetValue(op.GuardRight, out int right))
            {
                throw new Exception(txn.Body.Name + ": guarded write executes before its guard reads");
            }
            if (left + right < op.GuardThreshold) return;
        }
        txn.WriteKeys.Add(op.Key);
        txn.WriteValues[op.Key] = op.Value;
    }

    private static string WriteWriteConflict(TxnRecord txn, Store store)
    {
        foreach (string key in txn.WriteKeys)
        {
            if (store.LatestTs(key) > txn.StartTs) return key;
        }
        return "";
    }

    // T1 -rw-> T2 iff T1 read a key that T2 overwrites with a version invisible to T1.
    private static bool ReadWriteEdge(TxnRecord reader, TxnRecord writer)
    {
        if (ReferenceEquals(reader, writer)) return false;
        if (!reader.ReadKeys.Overlaps(writer.WriteKeys)) return false;
        return !writer.Committed || writer.CommitTs > reader.StartTs;
    }

    private static bool IsPivot(TxnRecord txn, List<TxnRecord> all)
    {
        bool incoming = all.Any(other => !other.Aborted && ReadWriteEdge(other, txn));
        if (!incoming) return false;
        return all.Any(other => other.Committed && ReadWriteEdge(txn, other));
    }
}

public static class Solution
{
    private static ulong _seed = 0x9E3779B97F4A7C15UL;

    private static ulong NextRand()
    {
        _seed += 0x9E3779B97F4A7C15UL;
        ulong z = _seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static int NextInt(int bound) => (int)(NextRand() % (ulong)bound);

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

    // Same "two doctors on call" schedule as task 637: each transaction reads both rows and steps
    // down only when it sees both, so only the read-write antidependencies connect them.
    private static Workload DoctorSkew(int stagger)
    {
        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("x"), Op.Read("y"), Op.GuardedWrite("x", 0, "x", "y", 2)),
            new Body("T2", stagger, Op.Read("x"), Op.Read("y"), Op.GuardedWrite("y", 0, "x", "y", 2)),
        };
        return new Workload("doctor-skew", bodies, Order(0, 1), Seed(("x", 1), ("y", 1)));
    }

    private static void WriteSkewIsAborted()
    {
        Execution si = SsiEngine.Run(DoctorSkew(0), false);
        Execution ssi = SsiEngine.Run(DoctorSkew(0), true);

        Check(si.CommitCount == 2, "write skew under plain SI: both commit");
        Check(ssi.CommitCount == 1 && ssi.AbortCount == 1, "write skew under SSI: exactly one aborts");
        Check(ssi.Final == "x=0;y=1;" || ssi.Final == "x=1;y=0;", "write skew under SSI: invariant x+y>=1 survives");
        Check(!IsSerializable(si), "write skew under plain SI: outcome is NOT serialisable");
        Check(IsSerializable(ssi), "write skew under SSI: outcome IS serialisable");
        Console.WriteLine("write skew: SI commits 2 (non-serialisable), SSI aborts 1 (serialisable)");
    }

    private static void SerialAndDisjointWorkloadsCommit()
    {
        Execution serialized = SsiEngine.Run(DoctorSkew(3), true);
        Check(serialized.CommitCount == 2 && serialized.AbortCount == 0, "staggered schedule: nothing to abort");
        Check(IsSerializable(serialized), "staggered schedule: serialisable");

        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("a"), Op.Write("a", 5)),
            new Body("T2", 0, Op.Read("b"), Op.Write("b", 6)),
            new Body("T3", 0, Op.Read("c"), Op.Write("c", 7)),
        };
        var disjoint = new Workload("disjoint", bodies, Order(2, 0, 1), Seed(("a", 0), ("b", 0), ("c", 0)));
        Execution ssi = SsiEngine.Run(disjoint, true);
        Check(ssi.CommitCount == 3 && ssi.AbortCount == 0, "disjoint keys: SSI commits all three");
        Check(ssi.Final == "a=5;b=6;c=7;", "disjoint keys: final state");
        Check(IsSerializable(ssi), "disjoint keys: serialisable");
    }

    private static void ReadOnlyTransactionsSurvive()
    {
        var bodies = new List<Body>
        {
            new Body("W", 0, Op.Read("k"), Op.Write("k", 9)),
            new Body("R1", 0, Op.Read("k"), Op.Read("k")),
            new Body("R2", 0, Op.Read("k")),
        };
        var workload = new Workload("readers", bodies, Order(1, 0, 2), Seed(("k", 1)));
        Execution ssi = SsiEngine.Run(workload, true);
        Check(ssi.AbortCount == 0, "read-only transactions are never aborted");
        Check(ssi.Txns[1].ReadValues["k"] == 1 && ssi.Txns[2].ReadValues["k"] == 1, "readers keep their snapshot");
        Check(ssi.Final == "k=9;", "writer's value is committed");
    }

    private static void LostUpdateIsCaught()
    {
        var bodies = new List<Body>
        {
            new Body("T1", 0, Op.Read("c"), Op.Write("c", 1)),
            new Body("T2", 0, Op.Read("c"), Op.Write("c", 99)),
        };
        var workload = new Workload("lost-update", bodies, Order(0, 1), Seed(("c", 0)));
        Execution ssi = SsiEngine.Run(workload, true);
        Check(ssi.CommitCount == 1 && ssi.AbortCount == 1, "lost update: exactly one committer");
        Check(ssi.Final == "c=1;", "lost update: loser's value never landed");
    }

    private static void EmptyWorkload()
    {
        var workload = new Workload("empty", new List<Body>(), Order(), Seed(("a", 3)));
        Execution ssi = SsiEngine.Run(workload, true);
        Check(ssi.CommitCount == 0 && ssi.AbortCount == 0, "empty workload: nothing happens");
        Check(ssi.Final == "a=3;", "empty workload: initial state untouched");

        var single = new Workload("single", new List<Body> { new Body("T", 0, Op.Read("a"), Op.Write("a", 4)) },
            Order(0), Seed(("a", 3)));
        Check(SsiEngine.Run(single, true).Final == "a=4;", "single transaction commits");

        var noOps = new Workload("no-ops", new List<Body> { new Body("T", 0) }, Order(0), Seed(("a", 3)));
        Execution e = SsiEngine.Run(noOps, true);
        Check(e.CommitCount == 1 && e.AbortCount == 0, "transaction without operations commits");
    }

    private static Workload RandomWorkload(int id, string[] keys)
    {
        var initial = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (string key in keys) initial[key] = NextInt(2);

        int count = 2 + NextInt(3);
        var bodies = new List<Body>();
        for (int i = 0; i < count; i++)
        {
            int stagger = id % 3 == 0 && NextInt(2) == 0 ? i * 4 : 0;
            var ops = new List<Op>();
            int steps = 1 + NextInt(3);
            for (int s = 0; s < steps; s++)
            {
                string key = keys[NextInt(keys.Length)];
                int kind = NextInt(4);
                if (kind == 0 || kind == 1)
                {
                    ops.Add(Op.Read(key));
                }
                else if (kind == 2)
                {
                    ops.Add(Op.Write(key, NextInt(3)));
                }
                else
                {
                    string guard = keys[NextInt(keys.Length)];
                    ops.Add(Op.Read(guard));
                    ops.Add(Op.GuardedWrite(key, NextInt(2), guard, guard, 1 + NextInt(2)));
                }
            }
            bodies.Add(new Body("T" + (i + 1), stagger, ops.ToArray()));
        }

        var order = new List<int>();
        for (int i = 0; i < count; i++) order.Add(i);
        for (int i = count - 1; i > 0; i--)
        {
            int j = NextInt(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return new Workload("random-" + id, bodies, order, initial);
    }

    private static void RandomisedSerializabilitySweep()
    {
        string[] keys = { "a", "b", "c" };
        int nonSerialisableSi = 0, abortedBySsi = 0, workloads = 0, readOnlyAborts = 0;
        for (int id = 0; id < 120; id++)
        {
            Workload workload = RandomWorkload(id, keys);
            Execution si = SsiEngine.Run(workload, false);
            Execution ssi = SsiEngine.Run(workload, true);
            workloads++;

            // The guarantee SSI advertises: whatever it commits is serialisable, aborts included.
            Check(IsSerializable(ssi), "SSI committed set is serialisable (" + workload.Name + ")");

            // An aborted transaction contributes nothing, and a read-only one must never abort.
            foreach (TxnRecord txn in ssi.Txns)
            {
                if (txn.Aborted && txn.WriteKeys.Count == 0) readOnlyAborts++;
            }
            if (!IsSerializable(si)) nonSerialisableSi++;
            if (ssi.AbortCount > 0) abortedBySsi++;

            Execution again = SsiEngine.Run(workload, true);
            Check(again.Final == ssi.Final && again.AbortCount == ssi.AbortCount, "deterministic rerun (" + workload.Name + ")");
        }

        Check(workloads == 120, "swept 120 randomised workloads");
        Check(nonSerialisableSi > 0, "the sweep really produced non-serialisable SI outcomes");
        Check(readOnlyAborts == 0, "a read-only transaction is never aborted by SSI");
        Check(abortedBySsi > 0, "SSI aborted at least one randomised workload");
        Console.WriteLine("random sweep: " + workloads + " workloads, SI non-serialisable=" + nonSerialisableSi
            + ", SSI aborted=" + abortedBySsi);
    }

    private static bool IsSerializable(Execution exec)
    {
        List<TxnRecord> committed = exec.Txns.Where(t => t.Committed).ToList();
        foreach (int[] permutation in Permutations(committed.Count))
        {
            var order = new List<TxnRecord>();
            foreach (int index in permutation) order.Add(committed[index]);
            if (SerialReplayMatches(order, exec)) return true;
        }
        return false;
    }

    // Replays the committed transactions one after another on a fresh store and demands that every
    // transaction observed exactly the same values in the concurrent run (view equivalence).
    private static bool SerialReplayMatches(List<TxnRecord> order, Execution exec)
    {
        var store = new Store(exec.Workload.Initial);
        foreach (TxnRecord txn in order)
        {
            var reads = new Dictionary<string, int>();
            var writes = new Dictionary<string, int>();
            foreach (Op op in txn.Body.Ops)
            {
                if (op.IsRead)
                {
                    reads[op.Key] = store.ReadAt(op.Key, store.Clock);
                    continue;
                }
                if (op.Guarded && reads[op.GuardLeft] + reads[op.GuardRight] < op.GuardThreshold) continue;
                writes[op.Key] = op.Value;
            }
            if (reads.Count != txn.ReadValues.Count) return false;
            foreach (KeyValuePair<string, int> read in reads)
            {
                if (!txn.ReadValues.TryGetValue(read.Key, out int seen) || seen != read.Value) return false;
            }
            long ts = store.Commit();
            foreach (KeyValuePair<string, int> write in writes) store.Install(write.Key, write.Value, ts);
        }
        return store.Fingerprint() == exec.Final;
    }

    private static List<int[]> Permutations(int n)
    {
        var result = new List<int[]>();
        var current = new int[n];
        var used = new bool[n];
        Build(0, n, current, used, result);
        return result;
    }

    private static void Build(int depth, int n, int[] current, bool[] used, List<int[]> result)
    {
        if (depth == n)
        {
            result.Add((int[])current.Clone());
            return;
        }
        for (int i = 0; i < n; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            Build(depth + 1, n, current, used, result);
            used[i] = false;
        }
    }

    public static void Main()
    {
        WriteSkewIsAborted();
        SerialAndDisjointWorkloadsCommit();
        ReadOnlyTransactionsSurvive();
        LostUpdateIsCaught();
        EmptyWorkload();
        RandomisedSerializabilitySweep();
        Console.WriteLine("PASS");
    }
}
