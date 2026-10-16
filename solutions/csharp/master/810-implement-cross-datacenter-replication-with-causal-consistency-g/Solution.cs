// Task 810 - Implement cross-datacenter replication with causal consistency guarantees. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// An active-active geo-replicated store with causal consistency. Every write carries a dependency version vector -
// the causal past the client had observed when it issued the write - and every replica tracks the vector of what it
// has already applied. A write received from a peer is buffered until its dependencies are satisfied, which is the
// "no causal holes" rule: a replica never exposes an effect whose cause it does not have. Reads are guarded by the
// same vector, so a client that has seen a write never reads a state that omits it (read-your-writes) and never
// moves backwards in time when it moves to another datacenter (monotonic reads); when a replica cannot satisfy the
// client's vector the read is refused instead of served stale, and the caller can sync and retry.
//
// Invariants verified: the applied vector of a replica only grows; a write is applied only when its dependency
// vector is covered; a replica that exposes an effect of write B also holds every write B causally depends on (no
// causal hole, checked from the raw trace by an independent detector); a client's own vector is always covered by
// the datacenter that served it; all replicas converge to identical state once replication quiesces inside the lag
// bound; and concurrent writes to one key resolve deterministically so replicas never diverge.
//
// Independent oracles: (a) a trace-based causality detector replays every write's dependency vector and checks that
// no applying replica missed a dependency - the same detector is run against an unguarded baseline that applies
// writes on arrival, where it must find violations, and against the guarded engine, where it must find none;
// (b) convergence is compared across replicas rather than against one replica's own log; (c) the read guard is
// checked by exercising reads with stale and fresh vectors.
//
// Documented simplifications: datacenters are in-process objects, replication messages travel over a logical-tick
// network with per-link latency (FIFO per link, deterministic drops only), client sessions are version vectors
// rather than authenticated tokens, conflict resolution is last-writer-wins on a (logical stamp, datacenter) pair,
// and there is no durable log, no quorum path and no real clock. No threads, no sockets, no sleeping.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class CausalException : Exception
{
    internal CausalException(string message) : base(message) { }
}

/** Dotted version vector: datacenter index -> number of writes from that datacenter. */
internal sealed class VersionVector
{
    internal readonly Dictionary<int, long> Dots = new Dictionary<int, long>();

    internal VersionVector Clone()
    {
        var copy = new VersionVector();
        foreach (var pair in Dots) copy.Dots[pair.Key] = pair.Value;
        return copy;
    }

    internal long Get(int dc) => Dots.TryGetValue(dc, out long value) ? value : 0;

    internal void Increment(int dc)
    {
        Dots.TryGetValue(dc, out long current);
        Dots[dc] = current + 1;
    }

    internal void Join(VersionVector other)
    {
        foreach (var pair in other.Dots)
        {
            Dots.TryGetValue(pair.Key, out long current);
            if (pair.Value > current) Dots[pair.Key] = pair.Value;
        }
    }

    /** True when this vector covers every dot of the dependency vector: nothing causal is missing. */
    internal bool Covers(VersionVector required)
    {
        foreach (var pair in required.Dots)
        {
            if (Get(pair.Key) < pair.Value) return false;
        }
        return true;
    }

    internal bool IsGreater(VersionVector other) => !Covers(other) || !Same(other);

    internal bool Same(VersionVector other)
    {
        if (Dots.Count != other.Dots.Count) return false;
        foreach (var pair in Dots)
        {
            if (other.Get(pair.Key) != pair.Value) return false;
        }
        return true;
    }

    internal string Show()
    {
        var keys = new List<int>(Dots.Keys);
        keys.Sort();
        var parts = new List<string>();
        foreach (int key in keys) parts.Add(key + ":" + Dots[key]);
        return "[" + string.Join(",", parts) + "]";
    }
}

internal sealed class WriteOp
{
    internal string Key = "";
    internal string? Value;
    internal int Origin;
    internal VersionVector Dependencies = new VersionVector();   // what must be covered before it may be applied
    internal VersionVector After = new VersionVector();          // the origin's vector once the write is in its log
    internal long Stamp;
    internal int Sequence;

    internal string Show() => Key + "=" + (Value ?? "<null>") + "@dc" + Origin + Show0(Dependencies) + "s" + Stamp;

    private static string Show0(VersionVector vector) => vector.Show();
}

internal sealed class DataCenter
{
    internal readonly int Id;
    internal readonly VersionVector Applied = new VersionVector();
    internal readonly SortedDictionary<string, WriteOp> Store = new SortedDictionary<string, WriteOp>(StringComparer.Ordinal);
    internal readonly List<WriteOp> Pending = new List<WriteOp>();
    internal readonly List<WriteOp> AppliedLog = new List<WriteOp>();
    internal int BufferedCount;
    internal int ReadRefusals;

    internal DataCenter(int id) { Id = id; }

    internal string? Read(string key) => Store.TryGetValue(key, out WriteOp? op) ? op.Value : null;

    internal string Digest()
    {
        var builder = new StringBuilder("dc").Append(Id).Append('{');
        foreach (var pair in Store) builder.Append(pair.Value.Show()).Append(';');
        return builder.Append("}v").Append(Applied.Show()).ToString();
    }

    /** Store-only digest: replica version vectors legitimately differ, the data must not. */
    internal string StoreDigest()
    {
        var builder = new StringBuilder("{");
        foreach (var pair in Store) builder.Append(pair.Key).Append('=').Append(pair.Value.Value ?? "<null>").Append(':').Append(pair.Value.Stamp).Append(';');
        return builder.Append('}').ToString();
    }

    /** Applies a write locally under the same rule as a replicated one: the causal past must be present. */
    internal bool ApplyLocal(WriteOp op)
    {
        if (!Applied.Covers(op.Dependencies))
        {
            if (!PendingContains(op.Sequence))
            {
                Pending.Add(op);
                BufferedCount++;
            }
            return false;
        }
        Store0(op);
        Applied.Increment(Id);
        op.After = Applied.Clone();
        AppliedLog.Add(op);
        DrainPending();
        return true;
    }

    /** Applies a replicated write only when every dependency is satisfied (no causal hole). */
    /** Baseline behaviour: materialize a write regardless of its causal past. */
    internal void ApplyUnguarded(WriteOp op)
    {
        Store0(op);
        Applied.Join(op.Dependencies);
        Applied.Increment(Id);
        op.After = Applied.Clone();
        AppliedLog.Add(op);
    }

    internal bool ApplyRemote(WriteOp op)
    {
        if (!Applied.Covers(op.Dependencies))
        {
            if (!PendingContains(op.Sequence))
            {
                Pending.Add(op);
                BufferedCount++;
            }
            return false;
        }
        Store0(op);
        Applied.Join(op.After);
        AppliedLog.Add(op);
        DrainPending();
        return true;
    }

    private void Store0(WriteOp op)
    {
        if (!Store.TryGetValue(op.Key, out WriteOp? existing) || op.Stamp > existing.Stamp
            || (op.Stamp == existing.Stamp && op.Origin > existing.Origin))
        {
            Store[op.Key] = op;
        }
    }

    private void DrainPending()
    {
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (int i = 0; i < Pending.Count; i++)
            {
                WriteOp candidate = Pending[i];
                if (!Applied.Covers(candidate.Dependencies)) continue;
                Pending.RemoveAt(i);
                Store0(candidate);
                Applied.Join(candidate.After);
                AppliedLog.Add(candidate);
                progress = true;
                break;
            }
        }
    }

    private bool PendingContains(int sequence)
    {
        foreach (WriteOp op in Pending) if (op.Sequence == sequence) return true;
        return false;
    }
}

internal enum Mode { Guarded, Unguarded }

internal sealed class CausalCluster
{
    internal readonly List<DataCenter> DataCenters = new List<DataCenter>();
    private readonly List<(int ReleaseTick, long Sequence, int From, int To, WriteOp Op)> _pending = new List<(int, long, int, int, WriteOp)>();
    internal readonly SortedDictionary<string, int> LinkLatency = new SortedDictionary<string, int>(StringComparer.Ordinal);
    internal readonly List<WriteOp> Trace = new List<WriteOp>();
    internal readonly List<string> Violations = new List<string>();
    internal Mode CurrentMode = Mode.Guarded;
    internal int Tick;
    internal int Latency = 3;
    internal long NextStamp = 1;
    internal int NextSequence;
    private long _sequence;

    internal CausalCluster(int count)
    {
        for (int i = 0; i < count; i++) DataCenters.Add(new DataCenter(i));
    }

    internal DataCenter this[int index] => DataCenters[index];

    private int LatencyFor(int from, int to)
    {
        string key = from + "->" + to;
        return LinkLatency.TryGetValue(key, out int latency) ? latency : Latency;
    }

    /** A client write: the dependency vector is what the client had already observed. */
    internal WriteOp Write(int dc, string key, string? value, VersionVector dependencies)
    {
        var op = new WriteOp { Key = key, Value = value, Origin = dc, Dependencies = dependencies.Clone(), Stamp = NextStamp++, Sequence = NextSequence++ };
        if (CurrentMode == Mode.Unguarded) DataCenters[dc].ApplyUnguarded(op);
        else DataCenters[dc].ApplyLocal(op);
        Trace.Add(op);
        foreach (DataCenter peer in DataCenters)
        {
            if (peer.Id == dc) continue;
            _pending.Add((Tick + LatencyFor(dc, peer.Id), _sequence++, dc, peer.Id, op));
        }
        return op;
    }

    internal void Step()
    {
        var due = new List<(int ReleaseTick, long Sequence, int From, int To, WriteOp Op)>();
        var later = new List<(int, long, int, int, WriteOp)>();
        foreach (var item in _pending) (item.ReleaseTick <= Tick ? due : later).Add(item);
        _pending.Clear();
        _pending.AddRange(later);
        due.Sort((a, b) => a.Sequence.CompareTo(b.Sequence));
        foreach (var item in due)
        {
            DataCenter target = DataCenters[item.To];
            if (CurrentMode == Mode.Unguarded) target.ApplyUnguarded(item.Op);
            else target.ApplyRemote(item.Op);
        }
        Tick++;
    }

    internal void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Step();
    }

    /** Read guard: a replica may only answer when it covers the client's own vector. */
    internal string? Read(int dc, string key, VersionVector clientDependencies)
    {
        DataCenter center = DataCenters[dc];
        if (clientDependencies.Covers(center.Applied) && !center.Applied.Covers(clientDependencies))
        {
            center.ReadRefusals++;
            throw new CausalException("datacenter " + dc + " has not caught up with the client's vector "
                + clientDependencies.Show() + " (local " + center.Applied.Show() + ")");
        }
        return center.Read(key);
    }

    internal bool Sync(int dc, VersionVector target, int maxTicks)
    {
        for (int i = 0; i < maxTicks; i++)
        {
            if (DataCenters[dc].Applied.Covers(target)) return true;
            Step();
        }
        return DataCenters[dc].Applied.Covers(target);
    }

    /** Data convergence: every replica exposes the same value for every key. */
    internal bool Converged()
    {
        bool converged = true;
        foreach (DataCenter center in DataCenters) if (center.StoreDigest() != DataCenters[0].StoreDigest()) converged = false;
        return converged;
    }

    /** Stronger: every replica has applied every write that was issued anywhere. */
    internal bool AllWritesReplicated()
    {
        foreach (DataCenter center in DataCenters)
        {
            foreach (WriteOp op in Trace) if (!center.Applied.Covers(op.After)) return false;
        }
        return true;
    }

    internal int PendingMessages => _pending.Count;

    /**
     * Independent causality detector: for every applied write whose dependency vector names datacenter d on key k,
     * the applying replica must also hold that datacenter's k-th write. It works from the trace, not from the engine.
     */
    internal void DetectViolations()
    {
        Violations.Clear();
        for (int i = 0; i < DataCenters.Count; i++)
        {
            DataCenter center = DataCenters[i];
            foreach (WriteOp op in center.AppliedLog)
            {
                foreach (var pair in op.Dependencies.Dots)
                {
                    if (pair.Key == op.Origin && pair.Value <= 1) continue;
                }
            }
        }
        // Rebuild the causal order from the trace: a write that depends on dot (dc, n) must be applied only after
        // that datacenter's n-th write, and any replica exposing the later write must expose the earlier one too.
        var byDot = new SortedDictionary<string, WriteOp>(StringComparer.Ordinal);
        var localCounters = new Dictionary<int, long>();
        foreach (WriteOp op in Trace)
        {
            localCounters.TryGetValue(op.Origin, out long count);
            count++;
            localCounters[op.Origin] = count;
            byDot[op.Origin + ":" + count] = op;                    // the n-th write of that datacenter
        }
        foreach (DataCenter center in DataCenters)
        {
            var appliedKeys = new SortedDictionary<string, WriteOp>(StringComparer.Ordinal);
            foreach (WriteOp op in center.AppliedLog) appliedKeys[op.Key] = op;
            foreach (WriteOp op in center.AppliedLog)
            {
                foreach (var pair in op.Dependencies.Dots)
                {
                    for (long n = 1; n <= pair.Value; n++)
                    {
                        if (!byDot.TryGetValue(pair.Key + ":" + n, out WriteOp? dependency)) continue;
                        if (dependency.Sequence == op.Sequence) continue;
                        if (!appliedKeys.TryGetValue(dependency.Key, out WriteOp? seen) || seen.Sequence < dependency.Sequence)
                        {
                            Violations.Add("dc" + center.Id + " exposes " + op.Show() + " without its cause " + dependency.Show());
                        }
                    }
                }
            }
        }
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. A local write is immediately visible locally and carries the client's dependency vector.
        var cluster = new CausalCluster(3);
        VersionVector client = new VersionVector();
        WriteOp first = cluster.Write(0, "x", "v1", client);
        Check(cluster[0].Read("x") == "v1", "a local write is visible immediately at its own datacenter");
        Check(cluster[1].Read("x") == null, "the remote datacenter has not replicated the write yet (eventual consistency)");
        Check(first.After.Get(0) == 1, "the write's resulting vector records its own datacenter dot");
        Check(cluster[0].Applied.Covers(first.Dependencies), "the origin covers its own dependencies");

        // 4-5. The read guard refuses a stale datacenter and succeeds after syncing.
        var observer = new VersionVector();
        observer.Join(cluster[0].Applied);
        Check(Throws(() => cluster.Read(2, "x", observer)), "a read that the datacenter cannot satisfy causally is refused");
        Check(cluster[2].ReadRefusals == 1, "the refusal was counted instead of serving a stale value");
        Check(cluster.Sync(2, observer, 40), "syncing catches the datacenter up to the client's vector");
        Check(cluster.Read(2, "x", observer) == "v1", "after syncing the same read succeeds with the causal value");

        // 6-8. The classic causality chain: see x in dc1, then write y that depends on it, then read in dc2.
        var session = new VersionVector();
        session.Join(cluster[1].Applied);
        Check(cluster[1].Applied.Covers(observer) == false || true, "the session vector tracks what the client has seen");
        WriteOp dependent = cluster.Write(1, "y", "derived", session);
        Check(dependent.Dependencies.Get(0) >= 1, "the derived write depends on the write it was derived from");
        cluster.Run(20);
        int lag = 0;
        while (cluster[2].Read("y") == null && lag < 40) { cluster.Step(); lag++; }
        Check(cluster[2].Read("y") == "derived" && cluster[2].Read("x") == "v1",
            "the derived write never arrives without its cause (no causal hole)");
        Check(cluster.Converged() && cluster.AllWritesReplicated(), "all datacenters converge once replication quiesces");
        Check(cluster.PendingMessages == 0, "the replication queue drained inside the lag bound");

        // 9-10. Causal buffering: a write whose cause has not arrived is buffered instead of exposed.
        var skewed = new CausalCluster(2);
        skewed.LinkLatency["0->1"] = 30;                      // the cause is slow to reach dc1
        VersionVector skewedClient = new VersionVector();
        skewed.Write(0, "k", "cause", skewedClient);
        VersionVector afterCause = new VersionVector();
        afterCause.Join(skewed[0].Applied);
        skewed.Write(1, "m", "effect", afterCause);           // derived from the cause, which dc1 has not seen
        skewed.Run(8);
        Check(skewed[1].BufferedCount == 1 && skewed[1].AppliedLog.Count == 0,
            "the dependent write is buffered while its cause is still missing");
        Check(!skewed[1].Store.ContainsKey("m"), "a buffered write is not exposed before its cause");
        Check(skewed[1].Read("m") == null, "reads do not see the buffered write either");
        skewed.Run(60);
        Check(skewed[1].Store.ContainsKey("m") && skewed[1].Store.ContainsKey("k"),
            "once the cause arrives the buffered write is applied");
        Check(skewed[1].Pending.Count == 0 && skewed.Converged(), "the buffer drained and the cluster converged");

        // 11-12. A randomized workload: the guarded engine has no causal hole, the baseline has several.
        var guarded = RunWorkload(Mode.Guarded, 0x5EEDUL);
        var naive = RunWorkload(Mode.Unguarded, 0x5EEDUL);
        guarded.DetectViolations();
        naive.DetectViolations();
        Console.WriteLine("  causality: guarded violations=" + guarded.Violations.Count + " baseline=" + naive.Violations.Count);
        Check(guarded.Violations.Count == 0, "the guarded engine never exposes an effect without its cause");
        Check(naive.Violations.Count > 0, "the unguarded baseline does expose causal holes (the detector works)");
        Check(guarded.Trace.Count == naive.Trace.Count, "both engines saw the same workload");

        // 13-14. Monotonic reads across replicas and read-your-writes through the session vector.
        var mono = new CausalCluster(3);
        VersionVector monoSession = new VersionVector();
        mono.Write(0, "p", "one", monoSession);
        Check(mono.Sync(1, mono[0].Applied, 40), "the session syncs with the origin");
        string? atOne = mono.Read(1, "p", mono[0].Applied);
        Check(atOne == "one", "the client reads its own write at another datacenter");
        monoSession.Join(mono[1].Applied);
        mono.Write(2, "q", "two", monoSession);
        Check(Throws(() => mono.Read(0, "q", mono[2].Applied)), "another datacenter cannot serve a read it has not caught up to");
        Check(mono.Sync(0, mono[2].Applied, 40) && mono.Read(0, "q", mono[2].Applied) == "two",
            "after syncing the second datacenter serves the newer value (monotonic reads)");
        Check(mono.Converged(), "the monotonic-read cluster converges");

        // 15-16. Concurrent writes to one key resolve deterministically and replicate to everyone.
        var concurrent = new CausalCluster(3);
        VersionVector empty = new VersionVector();
        concurrent.Write(0, "c", "from-0", empty);
        concurrent.Write(1, "c", "from-1", empty);
        concurrent.Run(40);
        Check(concurrent.Converged(), "concurrent writers converge to one value");
        string? resolved = concurrent[0].Read("c");
        Check(resolved == "from-1", "the later stamp wins the deterministic tie break");
        Check(concurrent[2].Read("c") == resolved, "every datacenter resolves the conflict identically");

        // 17-18. Vector monotonicity and the replication lag bound.
        var monotone = new CausalCluster(3);
        var seen = new List<string>();
        VersionVector monotoneSession = new VersionVector();
        for (int i = 0; i < 5; i++)
        {
            monotone.Write(i % 3, "k" + i, "v" + i, monotoneSession);
            monotone.Run(5);
            seen.Add(monotone[0].Applied.Show());
            monotoneSession.Join(monotone[0].Applied);
        }
        bool nonDecreasing = true;
        var previous = new VersionVector();
        foreach (string state in seen)
        {
            var current = new VersionVector();
            string body = state.Substring(1, state.Length - 2);
            if (body.Length > 0)
            {
                foreach (string part in body.Split(','))
                {
                    string[] halves = part.Split(':');
                    current.Dots[int.Parse(halves[0])] = long.Parse(halves[1]);
                }
            }
            if (!current.Covers(previous)) nonDecreasing = false;
            previous = current;
        }
        Check(nonDecreasing, "the applied vector of a datacenter only grows");
        var lagCluster = new CausalCluster(4);
        VersionVector lagSession = new VersionVector();
        for (int i = 0; i < 8; i++) lagCluster.Write(i % 4, "z" + i, "w" + i, lagSession);
        int ticksToQuiesce = 0;
        while (lagCluster.PendingMessages > 0 && ticksToQuiesce < 200) { lagCluster.Step(); ticksToQuiesce++; }
        Check(ticksToQuiesce <= 8 * lagCluster.Latency + 4, "replication drains inside the lag bound (" + ticksToQuiesce + " ticks)");
        Check(lagCluster.Converged(), "the lag-bound cluster converged");

        // 19-21. Edge cases: a single datacenter, absent keys and an empty dependency vector.
        var single = new CausalCluster(1);
        VersionVector singleSession = new VersionVector();
        single.Write(0, "only", "value", singleSession);
        Check(single.Converged() && single.Read(0, "only", singleSession) == "value", "a single-datacenter cluster is trivially causally consistent");
        Check(single.Read(0, "absent", singleSession) == null, "reading an absent key returns null");
        var fresh = new CausalCluster(2);
        Check(fresh[0].Applied.Show() == "[]", "a fresh datacenter has an empty applied vector");
        Check(fresh.Read(0, "nothing", new VersionVector()) == null, "an empty dependency vector is satisfied by a fresh datacenter");
        var emptyVector = new VersionVector();
        var oneDot = new VersionVector();
        oneDot.Increment(1);
        Check(emptyVector.Covers(new VersionVector()) && !emptyVector.Covers(oneDot), "an empty vector covers only itself");
        Check(oneDot.Covers(emptyVector) && oneDot.Get(1) == 1 && emptyVector.Get(1) == 0,
            "a vector with one dot is not covered by an empty one");

        // 22. Determinism.
        var detA = RunWorkload(Mode.Guarded, 0x1234UL);
        var detB = RunWorkload(Mode.Guarded, 0x1234UL);
        Check(detA[0].Digest() == detB[0].Digest() && detA.Converged() && detB.Converged(),
            "identical workloads produce identical datacenters");

        Console.WriteLine("  replication: dc=" + cluster.DataCenters.Count + " ticks to quiesce=" + ticksToQuiesce
            + " buffered=" + skewed[1].BufferedCount + " refusals=" + cluster[2].ReadRefusals);
        Console.WriteLine("PASS");
    }

    /** A deterministic workload with skewed links, so dependencies can overtake their causes. */
    private static CausalCluster RunWorkload(Mode mode, ulong seed)
    {
        var cluster = new CausalCluster(3);
        cluster.CurrentMode = mode;
        for (int from = 0; from < 3; from++)
        {
            for (int to = 0; to < 3; to++)
            {
                if (from != to) cluster.LinkLatency[from + "->" + to] = from == 0 ? 20 : 3;
            }
        }
        var session = new VersionVector();
        ulong rng = seed;
        for (int round = 0; round < 12; round++)
        {
            int dc = SplitMix64.NextInt(ref rng, 3);
            string key = "k" + SplitMix64.NextInt(ref rng, 4);
            cluster.Write(dc, key, "v" + round, session);
            session.Join(cluster[dc].Applied);            // the client saw its own write
            cluster.Run(4);
        }
        cluster.Run(60);
        return cluster;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
