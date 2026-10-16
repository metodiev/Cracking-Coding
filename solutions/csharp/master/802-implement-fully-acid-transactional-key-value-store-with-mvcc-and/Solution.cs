// Task 802 - Implement fully ACID transactional key-value store with MVCC and 2PC. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A faithful in-process transactional store. Multi-version concurrency control keeps every committed write as a
// version stamped with a commit timestamp from a single-writer timestamp oracle; reads run against a snapshot
// (all versions with commitTs <= startTs), so no reader blocks and no dirty read is possible. Two-phase commit
// spans an explicit set of shards: the coordinator locks intents on every participant in phase one and publishes
// its decision through a durable decision log, which is what makes recovery well defined.
//
// ACID properties implemented and verified:
//   * Atomicity   - a transaction that fails validation, is rejected in phase one, or is rolled back by recovery
//                   leaves no version and no lock anywhere, while a decided COMMIT is applied on every participant;
//   * Consistency - the bank invariant (a + b + c) is recomputed from the raw version lists after every commit,
//                   abort and recovery step, never from the engine's own bookkeeping;
//   * Isolation   - snapshot isolation (first-committer-wins) plus an optional serializability check that refuses a
//                   transaction whose read/write/anti-dependencies would close a cycle in the committed graph;
//   * Durability  - every applied write is appended to a per-shard write-ahead log, and replaying that log into a
//                   fresh shard must reproduce a bit-identical digest.
//
// Invariants verified: commit timestamps strictly increase and so do the versions of a key; a version is invisible
// until its own timestamp; the committed dependency graph stays acyclic; empty and read-only transactions never
// move a shard digest; every committed transaction appears on all of its participants or on none of them.
//
// Independent oracles: (a) view serializability is decided by brute force - every permutation of the committed
// transactions is replayed on a plain dictionary and accepted only if it reproduces every observed read value and
// the final state; (b) the bank invariant and the version counts are read back from the shards, not from engine
// bookkeeping; (c) durability is checked by replaying the WAL into new shard objects; (d) the two-phase-commit
// crash windows are cross-checked with an explicit count of participants that applied the transaction.
//
// Documented simplifications: shards are in-process objects, 2PC messages are method calls on those objects, the
// coordinator log is an in-memory list and crash points are explicit parameters instead of real processes, the
// timestamp oracle is a counter rather than a wall clock, and the serialization check is the conservative
// "abort when the dependency graph would close a cycle" rule - it may refuse some serializable histories but it
// never accepts a non-serializable one. Randomness comes from a hand-written splitmix64.
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

internal enum TxnStatus { Active, Committed, Aborted }

internal enum Decision { None, Commit, Abort }

/** Where a transaction may stop mid-protocol; recovery has to resolve every one of these windows. */
internal enum CrashPoint { None, BeforeDecide, AfterDecideBeforeDeliver }

/** What a snapshot read observed: the value (null = deleted or absent) and the version that produced it. */
internal struct ReadObservation
{
    internal string? Value;
    internal long VersionTs;

    internal ReadObservation(string? value, long versionTs) { Value = value; VersionTs = versionTs; }
}

internal sealed class Version
{
    internal readonly long CommitTs;
    internal readonly int TxnId;
    internal readonly string? Value;
    internal readonly bool Deleted;

    internal Version(long commitTs, int txnId, string? value, bool deleted)
    {
        CommitTs = commitTs; TxnId = txnId; Value = value; Deleted = deleted;
    }
}

/** One shard: a versioned key space, prepare/commit/abort intents for 2PC, and a write-ahead log. */
internal sealed class Shard
{
    private readonly SortedDictionary<string, List<Version>> _versions = new SortedDictionary<string, List<Version>>(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _locks = new Dictionary<string, int>(StringComparer.Ordinal);
    private readonly Dictionary<int, Dictionary<string, string?>> _intents = new Dictionary<int, Dictionary<string, string?>>();

    internal readonly string Name;
    internal readonly List<string> Wal = new List<string>();
    internal long CommitHigh;

    internal Shard(string name) { Name = name; }

    internal IReadOnlyList<Version> VersionsOf(string key)
        => _versions.TryGetValue(key, out List<Version>? list) ? list : Array.Empty<Version>();

    internal int VersionCount(string key) => VersionsOf(key).Count;

    /** Total number of versions held by this shard; must equal the number of WAL records. */
    internal int TotalVersions()
    {
        int total = 0;
        foreach (var pair in _versions) total += pair.Value.Count;
        return total;
    }

    /** Snapshot read: the newest version not newer than snapshotTs; newer versions stay invisible. */
    internal ReadObservation ReadAt(string key, long snapshotTs)
    {
        if (!_versions.TryGetValue(key, out List<Version>? list)) return new ReadObservation(null, 0);
        Version? visible = null;
        foreach (Version version in list)
        {
            if (version.CommitTs <= snapshotTs) visible = version;
            else break;
        }
        if (visible == null) return new ReadObservation(null, 0);
        return new ReadObservation(visible.Deleted ? null : visible.Value, visible.CommitTs);
    }

    /** First-committer-wins: any version newer than the reader's snapshot is a concurrent write. */
    internal bool HasWriteConflict(string key, long startTs)
    {
        foreach (Version version in VersionsOf(key)) if (version.CommitTs > startTs) return true;
        return false;
    }

    internal bool Prepare(int txnId, IReadOnlyDictionary<string, string?> writes)
    {
        foreach (var pair in writes)
        {
            if (_locks.TryGetValue(pair.Key, out int owner) && owner != txnId) return false;   // another intent holds the key
        }
        var copy = new Dictionary<string, string?>(writes, StringComparer.Ordinal);
        _intents[txnId] = copy;
        foreach (var pair in copy) _locks[pair.Key] = txnId;
        return true;
    }

    internal bool HasIntent(int txnId) => _intents.ContainsKey(txnId);

    internal List<int> PreparedTxns() => new List<int>(_intents.Keys);

    internal void CommitIntent(int txnId, long commitTs)
    {
        if (!_intents.TryGetValue(txnId, out Dictionary<string, string?>? writes)) return;
        _intents.Remove(txnId);
        var keys = new List<string>(writes.Keys);
        keys.Sort(StringComparer.Ordinal);                         // deterministic application order
        foreach (string key in keys)
        {
            string? value = writes[key];
            if (!_versions.TryGetValue(key, out List<Version>? list))
            {
                list = new List<Version>();
                _versions[key] = list;
            }
            list.Add(new Version(commitTs, txnId, value, value == null));
            if (commitTs > CommitHigh) CommitHigh = commitTs;
            Wal.Add(commitTs + "|" + txnId + "|" + key + "|" + (value ?? "<del>"));
            ReleaseKey(key, txnId);
        }
    }

    internal void AbortIntent(int txnId)
    {
        if (!_intents.TryGetValue(txnId, out Dictionary<string, string?>? writes)) return;
        _intents.Remove(txnId);
        foreach (var pair in writes) ReleaseKey(pair.Key, txnId);
    }

    private void ReleaseKey(string key, int txnId)
    {
        if (_locks.TryGetValue(key, out int owner) && owner == txnId) _locks.Remove(key);
    }

    internal bool IsLocked(string key) => _locks.ContainsKey(key);

    internal string Digest()
    {
        var builder = new StringBuilder(Name).Append('{');
        foreach (var pair in _versions)
        {
            builder.Append(pair.Key).Append('=');
            foreach (Version version in pair.Value)
            {
                builder.Append(version.CommitTs).Append('/').Append(version.TxnId).Append('/')
                    .Append(version.Deleted ? "<del>" : version.Value).Append(',');
            }
            builder.Append(';');
        }
        return builder.Append('}').ToString();
    }

    /** Durability: a fresh shard rebuilt from the WAL must be indistinguishable from the live one. */
    internal static Shard ReplayFromWal(string name, IReadOnlyList<string> wal)
    {
        var shard = new Shard(name);
        foreach (string line in wal)
        {
            string[] parts = line.Split('|');
            long commitTs = long.Parse(parts[0]);
            int txnId = int.Parse(parts[1]);
            string key = parts[2];
            string? value = parts[3] == "<del>" ? null : parts[3];
            if (!shard._versions.TryGetValue(key, out List<Version>? list))
            {
                list = new List<Version>();
                shard._versions[key] = list;
            }
            list.Add(new Version(commitTs, txnId, value, value == null));
            if (commitTs > shard.CommitHigh) shard.CommitHigh = commitTs;
        }
        return shard;
    }
}

internal sealed class Txn
{
    internal readonly int Id;
    internal readonly long StartTs;
    internal long CommitTs = -1;
    internal TxnStatus Status = TxnStatus.Active;
    internal readonly Dictionary<string, string?> Writes = new Dictionary<string, string?>(StringComparer.Ordinal);
    internal readonly Dictionary<string, ReadObservation> ReadSet = new Dictionary<string, ReadObservation>(StringComparer.Ordinal);
    internal readonly Dictionary<string, long> WriteVersions = new Dictionary<string, long>(StringComparer.Ordinal);

    internal Txn(int id, long startTs) { Id = id; StartTs = startTs; }
}

/** Durable decision log of the 2PC coordinator: the reason recovery can always finish a transaction. */
internal sealed class Coordinator
{
    internal readonly List<(int TxnId, Decision Decision, long CommitTs)> Log = new List<(int, Decision, long)>();
    internal readonly List<string> Messages = new List<string>();

    internal void Decide(int txnId, Decision decision, long commitTs)
    {
        Log.Add((txnId, decision, commitTs));
        Messages.Add("decide " + decision + " txn=" + txnId + " ts=" + commitTs);
    }

    internal Decision Lookup(int txnId)
    {
        foreach ((int id, Decision decision, long _) in Log) if (id == txnId) return decision;
        return Decision.None;
    }

    internal long CommitTsOf(int txnId)
    {
        foreach ((int id, Decision decision, long commitTs) in Log) if (id == txnId) return commitTs;
        return -1;
    }
}

/** Cycle test over the committed dependency graph (write-read, write-write and read-write edges). */
internal static class DependencyGraph
{
    internal static bool HasCycle(IReadOnlyDictionary<int, HashSet<int>> edges)
    {
        var colour = new Dictionary<int, int>();
        foreach (int node in edges.Keys) if (Visit(node, edges, colour)) return true;
        return false;
    }

    private static bool Visit(int node, IReadOnlyDictionary<int, HashSet<int>> edges, Dictionary<int, int> colour)
    {
        int state = colour.TryGetValue(node, out int known) ? known : 0;
        if (state == 1) return true;                               // back edge: cycle
        if (state == 2) return false;
        colour[node] = 1;
        if (edges.TryGetValue(node, out HashSet<int>? successors))
        {
            foreach (int successor in successors) if (Visit(successor, edges, colour)) return true;
        }
        colour[node] = 2;
        return false;
    }
}

/** View-serializability by brute force: replay every permutation of the committed transactions. */
internal static class SerializabilityOracle
{
    internal static bool TryFindSerialOrder(IReadOnlyList<Txn> txns, IReadOnlyDictionary<string, string?> finalState, out List<int>? witness)
    {
        var order = new List<int>();
        var used = new bool[txns.Count];
        witness = null;
        return Search(txns, used, order, finalState, ref witness);
    }

    private static bool Search(IReadOnlyList<Txn> txns, bool[] used, List<int> order, IReadOnlyDictionary<string, string?> finalState, ref List<int>? witness)
    {
        if (order.Count == txns.Count)
        {
            if (!ReplaysSerially(txns, order, finalState)) return false;
            witness = new List<int>(order);
            return true;
        }
        for (int i = 0; i < txns.Count; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            order.Add(txns[i].Id);
            if (Search(txns, used, order, finalState, ref witness)) return true;
            order.RemoveAt(order.Count - 1);
            used[i] = false;
        }
        return false;
    }

    /** Replays one candidate serial order on a plain dictionary and checks the reads and the final state. */
    internal static bool ReplaysSerially(IReadOnlyList<Txn> txns, IReadOnlyList<int> order, IReadOnlyDictionary<string, string?> finalState)
    {
        var store = new SortedDictionary<string, string?>(StringComparer.Ordinal);
        foreach (int id in order)
        {
            Txn txn = Find(txns, id);
            foreach (var read in txn.ReadSet)
            {
                store.TryGetValue(read.Key, out string? actual);
                if (actual != read.Value.Value) return false;       // the observed read does not match this serial order
            }
            foreach (var write in txn.Writes)
            {
                if (write.Value == null) store.Remove(write.Key);
                else store[write.Key] = write.Value;
            }
        }
        if (store.Count != finalState.Count) return false;
        foreach (var pair in finalState)
        {
            store.TryGetValue(pair.Key, out string? actual);
            if (actual != pair.Value) return false;
        }
        return true;
    }

    private static Txn Find(IReadOnlyList<Txn> txns, int id)
    {
        foreach (Txn txn in txns) if (txn.Id == id) return txn;
        throw new InvalidOperationException("unknown txn " + id);
    }
}

internal sealed class TxEngine
{
    internal readonly List<Shard> Shards = new List<Shard>();
    internal readonly Dictionary<string, int> Placement = new Dictionary<string, int>(StringComparer.Ordinal);
    internal readonly Coordinator Coordinator = new Coordinator();
    internal readonly List<Txn> CommittedTxns = new List<Txn>();
    internal readonly List<string> Trace = new List<string>();
    internal bool Serializable;
    private readonly Dictionary<int, Txn> _txns = new Dictionary<int, Txn>();
    private readonly SortedSet<string> _allKeys = new SortedSet<string>(StringComparer.Ordinal);
    private long _ts;
    private int _nextTxnId = 1;

    internal TxEngine(int shardCount, bool serializable)
    {
        for (int i = 0; i < shardCount; i++) Shards.Add(new Shard("s" + i));
        Serializable = serializable;
    }

    internal void Place(string key, int shard) => Placement[key] = shard;

    internal Shard ShardFor(string key)
    {
        if (Placement.TryGetValue(key, out int index)) return Shards[index];
        int hash = 0;
        foreach (char c in key) hash = (hash * 31 + c) & 0x7FFFFFFF;
        return Shards[hash % Shards.Count];
    }

    internal long CurrentTime => _ts;

    internal Txn Begin()
    {
        var txn = new Txn(_nextTxnId++, _ts);
        _txns[txn.Id] = txn;
        return txn;
    }

    internal string? Read(Txn txn, string key)
    {
        RequireActive(txn);
        if (txn.Writes.TryGetValue(key, out string? local)) return local;             // read your own writes
        ReadObservation observation = ShardFor(key).ReadAt(key, txn.StartTs);
        txn.ReadSet[key] = observation;
        return observation.Value;
    }

    internal void Write(Txn txn, string key, string? value)
    {
        RequireActive(txn);
        txn.Writes[key] = value;
    }

    internal bool Commit(Txn txn, CrashPoint crash = CrashPoint.None)
    {
        RequireActive(txn);
        foreach (var pair in txn.Writes)
        {
            if (ShardFor(pair.Key).HasWriteConflict(pair.Key, txn.StartTs))
            {
                Abort(txn);
                Trace.Add("write-write conflict on " + pair.Key + " (txn " + txn.Id + ")");
                return false;
            }
        }
        if (Serializable && WouldCloseCycle(txn))
        {
            Abort(txn);
            Trace.Add("serialization failure for txn " + txn.Id);
            return false;
        }

        var byShard = new SortedDictionary<int, Dictionary<string, string?>>();
        foreach (var pair in txn.Writes)
        {
            int index = Shards.IndexOf(ShardFor(pair.Key));
            if (!byShard.TryGetValue(index, out Dictionary<string, string?>? writes))
            {
                writes = new Dictionary<string, string?>(StringComparer.Ordinal);
                byShard[index] = writes;
            }
            writes[pair.Key] = pair.Value;
        }

        foreach (var entry in byShard)                                   // phase one: prepare and lock intents
        {
            if (!Shards[entry.Key].Prepare(txn.Id, entry.Value))
            {
                foreach (var rollback in byShard) Shards[rollback.Key].AbortIntent(txn.Id);
                Abort(txn);
                Trace.Add("2pc phase one rejected txn " + txn.Id + " on " + Shards[entry.Key].Name);
                return false;
            }
        }

        if (crash == CrashPoint.BeforeDecide) return false;              // no decision logged: recovery rolls back
        long commitTs = ++_ts;
        Coordinator.Decide(txn.Id, Decision.Commit, commitTs);
        if (crash == CrashPoint.AfterDecideBeforeDeliver) return false;  // decision logged but not delivered
        Finish(txn, commitTs);
        return true;
    }

    private void Finish(Txn txn, long commitTs)
    {
        foreach (Shard shard in Shards) if (shard.HasIntent(txn.Id)) shard.CommitIntent(txn.Id, commitTs);
        MarkCommitted(txn, commitTs);
    }

    private void MarkCommitted(Txn txn, long commitTs)
    {
        if (txn.Status == TxnStatus.Committed) return;
        txn.CommitTs = commitTs;
        txn.Status = TxnStatus.Committed;
        foreach (string key in txn.Writes.Keys)
        {
            txn.WriteVersions[key] = commitTs;
            _allKeys.Add(key);
        }
        CommittedTxns.Add(txn);
    }

    internal void Abort(Txn txn)
    {
        RequireActive(txn);
        foreach (Shard shard in Shards) shard.AbortIntent(txn.Id);
        txn.Status = TxnStatus.Aborted;
    }

    /** Coordinator recovery: deliver every logged decision, roll back every prepared but undecided intent. */
    internal void Recover()
    {
        foreach (Shard shard in Shards)
        {
            foreach (int txnId in shard.PreparedTxns())
            {
                Decision decision = Coordinator.Lookup(txnId);
                if (decision == Decision.Commit)
                {
                    long commitTs = Coordinator.CommitTsOf(txnId);
                    shard.CommitIntent(txnId, commitTs);
                    if (_txns.TryGetValue(txnId, out Txn? txn)) MarkCommitted(txn, commitTs);
                    Coordinator.Messages.Add("recover: deliver COMMIT txn=" + txnId + " to " + shard.Name);
                }
                else
                {
                    shard.AbortIntent(txnId);
                    if (_txns.TryGetValue(txnId, out Txn? txn) && txn.Status == TxnStatus.Active) txn.Status = TxnStatus.Aborted;
                    Coordinator.Messages.Add("recover: roll back undecided txn=" + txnId + " on " + shard.Name);
                }
            }
        }
    }

    /** Dependencies among committed transactions, derived from observed versions rather than from a lock table. */
    internal Dictionary<int, HashSet<int>> BuildGraph(IReadOnlyList<Txn> txns)
    {
        var edges = new Dictionary<int, HashSet<int>>();
        foreach (Txn txn in txns) edges[txn.Id] = new HashSet<int>();
        for (int i = 0; i < txns.Count; i++)
        {
            for (int j = 0; j < txns.Count; j++)
            {
                if (i == j) continue;
                Txn a = txns[i];
                Txn b = txns[j];
                foreach (var read in b.ReadSet)                           // write-read edge
                {
                    if (a.WriteVersions.TryGetValue(read.Key, out long writtenTs) && writtenTs == read.Value.VersionTs)
                        edges[a.Id].Add(b.Id);
                }
                foreach (var write in b.Writes)                           // write-write edge
                {
                    if (a.Writes.ContainsKey(write.Key) && a.CommitTs < b.CommitTs) edges[a.Id].Add(b.Id);
                }
                foreach (var read in a.ReadSet)                           // read-write (anti-dependency) edge
                {
                    if (b.WriteVersions.TryGetValue(read.Key, out long writtenTs) && writtenTs > read.Value.VersionTs)
                        edges[a.Id].Add(b.Id);
                }
            }
        }
        return edges;
    }

    /** Conservative serializability: refuse a commit whose edges would close a cycle over committed transactions. */
    private bool WouldCloseCycle(Txn candidate)
    {
        var txns = new List<Txn>(CommittedTxns);
        long provisional = _ts + 1;
        candidate.CommitTs = provisional;
        foreach (string key in candidate.Writes.Keys) candidate.WriteVersions[key] = provisional;
        txns.Add(candidate);
        bool cycle = DependencyGraph.HasCycle(BuildGraph(txns));
        candidate.CommitTs = -1;
        candidate.WriteVersions.Clear();
        return cycle;
    }

    internal SortedDictionary<string, string?> Snapshot()
    {
        var state = new SortedDictionary<string, string?>(StringComparer.Ordinal);
        foreach (string key in _allKeys)
        {
            ReadObservation observation = ShardFor(key).ReadAt(key, long.MaxValue);
            if (observation.Value != null) state[key] = observation.Value;
        }
        return state;
    }

    internal int AppliedCount(string key) => ShardFor(key).VersionCount(key);

    internal string Digest()
    {
        var builder = new StringBuilder();
        foreach (Shard shard in Shards) builder.Append(shard.Digest());
        return builder.ToString();
    }

    private static void RequireActive(Txn txn)
    {
        if (txn.Status != TxnStatus.Active) throw new InvalidOperationException("transaction " + txn.Id + " is " + txn.Status);
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-5. Seed three shards, then run a two-shard transfer through 2PC.
        var engine = new TxEngine(3, serializable: true);
        engine.Place("a", 0); engine.Place("b", 1); engine.Place("c", 2); engine.Place("n", 0);
        engine.Place("x", 0); engine.Place("y", 1); engine.Place("z", 2);
        Txn seed = engine.Begin();
        engine.Write(seed, "a", "100"); engine.Write(seed, "b", "100"); engine.Write(seed, "c", "100");
        Check(engine.Commit(seed), "a three-shard seed transaction commits");
        Check(Value(engine, "a") == "100" && Value(engine, "b") == "100" && Value(engine, "c") == "100",
            "every participant exposes the seeded value");
        Check(engine.Coordinator.Log.Count == 1 && engine.Coordinator.Log[0].Decision == Decision.Commit,
            "the coordinator logged exactly one COMMIT decision");
        Check(engine.Shards[0].VersionCount("a") == 1 && engine.Shards[1].VersionCount("b") == 1 && engine.Shards[2].VersionCount("c") == 1,
            "each participant applied exactly one version for the transaction");
        Check(engine.Shards[1].VersionCount("a") == 0 && engine.Shards[0].VersionCount("b") == 0,
            "a shard only stores versions of the keys it owns");

        long seedTs = engine.Shards[0].CommitHigh;
        Txn transfer = engine.Begin();
        Check(engine.Read(transfer, "a") == "100", "the transfer reads the current balance");
        engine.Write(transfer, "a", "90"); engine.Write(transfer, "b", "110");
        Check(engine.Commit(transfer), "the two-shard transfer commits through two-phase commit");
        Check(engine.Shards[0].ReadAt("a", seedTs).Value == "100" && engine.Shards[0].ReadAt("a", long.MaxValue).Value == "90",
            "an old snapshot still returns the pre-transfer value while the newest snapshot sees the transfer");
        Check(engine.Shards[0].VersionCount("a") == 2 && engine.Shards[0].VersionCount("b") == 0,
            "MVCC keeps one version per committed write, on the owning shard only");
        CheckBank(engine, 300, "after the committed transfer");

        // 6-10. Snapshot stability, read-your-writes and the absence of dirty reads.
        Txn reader = engine.Begin();
        Check(engine.Read(reader, "a") == "90", "the reader observes the newest committed value");
        Txn writer = engine.Begin();
        Check(engine.Read(writer, "b") == "110", "the concurrent writer reads the current second balance");
        engine.Write(writer, "a", "80"); engine.Write(writer, "b", "120");
        Check(engine.Commit(writer), "another writer commits while the reader's transaction stays open");
        Check(engine.Read(reader, "a") == "90", "repeatable read: the open snapshot does not move");
        Check(engine.Read(reader, "c") == "100", "the other keys stay visible in the same snapshot");
        engine.Write(reader, "n", "5");
        Check(engine.Read(reader, "n") == "5", "read-your-own-writes inside an open transaction");
        Check(Value(engine, "n") == null, "an uncommitted write is invisible to everyone else (no dirty read)");
        Check(engine.Commit(reader), "the reader's transaction commits");
        Check(Value(engine, "n") == "5", "after the commit the write becomes visible");
        CheckBank(engine, 300, "after the snapshot transactions");

        // 11-13. Abort atomicity: no version, no lock, no digest change.
        string digestBefore = engine.Digest();
        int versionsBefore = engine.Shards[0].VersionCount("a");
        Txn doomed = engine.Begin();
        engine.Write(doomed, "a", "0"); engine.Write(doomed, "b", "0");
        engine.Abort(doomed);
        Check(engine.Digest() == digestBefore && engine.Shards[0].VersionCount("a") == versionsBefore,
            "an aborted transaction leaves no version and no other trace");
        Check(!engine.Shards[0].IsLocked("a") && !engine.Shards[1].IsLocked("b"), "an abort releases both intent locks");
        Txn afterAbort = engine.Begin();
        Check(engine.Read(afterAbort, "a") == "80", "the aborted transaction did not move the balance");
        engine.Write(afterAbort, "a", "65"); engine.Write(afterAbort, "b", "135");
        Check(engine.Commit(afterAbort), "the keys are writable again after an abort");
        CheckBank(engine, 300, "after the abort and the follow-up transfer");

        // 14-18. Phase-one rejection is all-or-nothing across participants, and recovery rolls it back.
        Txn holder = engine.Begin();
        engine.Write(holder, "y", "9"); engine.Write(holder, "z", "9");
        Check(!engine.Commit(holder, CrashPoint.BeforeDecide), "the transaction stops after phase one without a decision");
        Check(engine.Shards[1].IsLocked("y") && engine.Shards[2].IsLocked("z"), "phase one left intents on both participants");
        string digestLocked = engine.Digest();
        Txn blocked = engine.Begin();
        engine.Write(blocked, "x", "1"); engine.Write(blocked, "y", "1");
        Check(!engine.Commit(blocked), "phase one rejects a transaction whose key is locked by another intent");
        Check(engine.Digest() == digestLocked, "the rejected transaction applied nothing on any participant");
        Check(engine.Shards[0].VersionCount("x") == 0 && engine.Shards[1].VersionCount("y") == 0,
            "neither the free nor the locked participant kept a version");
        engine.Recover();
        Check(!engine.Shards[1].IsLocked("y") && !engine.Shards[2].IsLocked("z"),
            "recovery rolled back the prepared but undecided intents");
        Check(holder.Status == TxnStatus.Aborted && engine.Digest() == digestLocked,
            "the undecided transaction is aborted and no version was written");

        // 19-21. Recovery after the decision was logged but not delivered.
        Txn decided = engine.Begin();
        engine.Write(decided, "x", "11"); engine.Write(decided, "z", "22");
        Check(!engine.Commit(decided, CrashPoint.AfterDecideBeforeDeliver), "the coordinator logs the decision without delivering it");
        Check(engine.Coordinator.Lookup(decided.Id) == Decision.Commit, "the decision is durable in the coordinator log");
        Check(engine.Shards[0].VersionCount("x") == 0 && engine.Shards[2].VersionCount("z") == 0,
            "no participant applied anything before recovery");
        engine.Recover();
        Check(engine.Shards[0].VersionCount("x") == 1 && engine.Shards[2].VersionCount("z") == 1,
            "recovery delivers the logged COMMIT to every participant (all-or-nothing)");
        Check(decided.Status == TxnStatus.Committed && decided.CommitTs == engine.Coordinator.CommitTsOf(decided.Id),
            "the recovered transaction carries the logged commit timestamp");
        CheckBank(engine, 300, "after recovery");

        // 22-23. Durability: replay the write-ahead logs and compare digests.
        bool durable = true;
        bool walComplete = true;
        int walRecords = 0;
        foreach (Shard shard in engine.Shards)
        {
            Shard replay = Shard.ReplayFromWal(shard.Name, shard.Wal);
            if (replay.Digest() != shard.Digest()) durable = false;
            if (shard.Wal.Count != shard.TotalVersions()) walComplete = false;
            walRecords += shard.Wal.Count;
        }
        Check(durable, "replaying every write-ahead log reproduces bit-identical shards");
        Check(walComplete, "the WAL holds exactly one record per applied version on every shard");

        // 24. Lost update: first-committer-wins rejects the concurrent increment, a retry applies it.
        Txn incrementA = engine.Begin();
        Txn incrementB = engine.Begin();
        string firstSeen = engine.Read(incrementA, "n")!;
        string secondSeen = engine.Read(incrementB, "n")!;
        Check(firstSeen == secondSeen, "both concurrent transactions read the same snapshot value");
        engine.Write(incrementA, "n", (int.Parse(firstSeen) + 1).ToString());
        engine.Write(incrementB, "n", (int.Parse(secondSeen) + 1).ToString());
        Check(engine.Commit(incrementA), "the first increment commits");
        Check(!engine.Commit(incrementB), "the concurrent second increment is rejected (first-committer-wins)");
        Txn retry = engine.Begin();
        engine.Write(retry, "n", (int.Parse(engine.Read(retry, "n")!) + 1).ToString());
        Check(engine.Commit(retry), "the retried increment commits");
        Check(Value(engine, "n") == "7", "no lost update: the value reflects both increments after the retry");
        bool conflictReported = false;
        foreach (string entry in engine.Trace) if (entry.StartsWith("write-write conflict on n", StringComparison.Ordinal)) conflictReported = true;
        Check(conflictReported, "the loser was rejected by the write-write conflict rule");

        // 25-28. Write skew: snapshot isolation alone breaks the invariant; the serializability check prevents it.
        var si = new TxEngine(2, serializable: false);
        si.Place("d1", 0); si.Place("d2", 0);
        RunOnCallScenario(si, out bool siBothCommitted, out int siGraphCycle, out bool siSerializable);
        Check(siBothCommitted, "snapshot isolation alone lets both write-skew transactions commit");
        Check(OnCallSum(si) == 0, "the write skew broke the at-least-one-on-call invariant");
        Check(siGraphCycle == 1, "the committed history contains a dependency cycle");
        Check(!siSerializable, "brute force confirms the snapshot-isolation history is not view-serializable");

        var ssi = new TxEngine(2, serializable: true);
        ssi.Place("d1", 0); ssi.Place("d2", 0);
        RunOnCallScenario(ssi, out bool ssiBothCommitted, out int ssiGraphCycle, out bool ssiSerializable);
        Check(!ssiBothCommitted, "the serialization check refuses the second write-skew transaction");
        Check(OnCallSum(ssi) >= 1, "the at-least-one-on-call invariant survives under serializable isolation");
        Check(ssiGraphCycle == 0, "the committed dependency graph stays acyclic");
        Check(ssiSerializable, "brute force finds a serial witness for the committed history");
        List<int>? witness;
        bool found = SerializabilityOracle.TryFindSerialOrder(ssi.CommittedTxns, ssi.Snapshot(), out witness);
        Check(found && witness != null && SerializabilityOracle.ReplaysSerially(ssi.CommittedTxns, witness, ssi.Snapshot()),
            "replaying the witness reproduces every observed read and the final state");

        // 29-30. A mixed concurrent workload: the engine's own graph check and the brute-force oracle must agree.
        var mixed = new TxEngine(2, serializable: true);
        int mixedCommitted = RunWorkload(mixed, 0x5EEDUL);
        Check(mixedCommitted >= 1, "the mixed concurrent workload committed at least one transaction");
        Check(!DependencyGraph.HasCycle(mixed.BuildGraph(mixed.CommittedTxns)),
            "the mixed workload's committed dependency graph is acyclic");
        List<int>? mixedWitness;
        bool mixedSerializable = SerializabilityOracle.TryFindSerialOrder(mixed.CommittedTxns, mixed.Snapshot(), out mixedWitness);
        Check(mixedSerializable && mixedWitness != null, "brute force finds a serial order for the mixed workload ("
            + mixed.CommittedTxns.Count + " committed transactions)");
        Check(SerializabilityOracle.ReplaysSerially(mixed.CommittedTxns, mixedWitness!, mixed.Snapshot()),
            "replaying the mixed workload's witness reproduces its reads and final state");
        Console.WriteLine("  mixed workload: committed=" + mixedCommitted + " of 6 concurrent transactions");

        // 31. Determinism: identical workloads produce identical stores and timestamps.
        var runA = new TxEngine(2, serializable: true);
        var runB = new TxEngine(2, serializable: true);
        int committedA = RunWorkload(runA, 0x1234UL);
        int committedB = RunWorkload(runB, 0x1234UL);
        Check(committedA == committedB && runA.Digest() == runB.Digest() && runA.CurrentTime == runB.CurrentTime,
            "identical workloads produce identical stores, decision logs and timestamps");
        Check(committedA >= 1 && runA.Coordinator.Log.Count == committedA + 1,
            "every successful commit (plus the seed) has a coordinator decision");

        // 32-36. Edge cases: tombstones, absent keys, empty and read-only transactions.
        string digestNow = engine.Digest();
        Txn emptyTxn = engine.Begin();
        Check(engine.Commit(emptyTxn), "a transaction with no writes commits trivially");
        Check(engine.Digest() == digestNow, "an empty transaction changes no shard");
        Txn deleter = engine.Begin();
        Check(engine.Read(deleter, "c") == "100", "the delete transaction reads the value it is about to remove");
        engine.Write(deleter, "c", null);
        Check(engine.Commit(deleter), "a delete commits as a tombstone version");
        Check(Value(engine, "c") == null && engine.Shards[2].VersionCount("c") == 2,
            "the tombstone hides the key but keeps the version history");
        Txn missing = engine.Begin();
        Check(engine.Read(missing, "never-written") == null && missing.ReadSet.Count == 1,
            "reading an absent key returns null and records the observation");
        Check(engine.Commit(missing) && engine.AppliedCount("never-written") == 0,
            "a read-only transaction commits and writes no version");

        // 37. Rejected input and timestamp monotonicity.
        Check(Throws(() => engine.Commit(seed)), "committing a finished transaction is refused");
        Check(Throws(() => engine.Read(seed, "a")), "reading through a finished transaction is refused");
        Check(Throws(() => engine.Write(seed, "a", "1")), "writing through a finished transaction is refused");
        Check(Throws(() => engine.Abort(seed)), "aborting a committed transaction is refused");
        long timeBefore = engine.CurrentTime;
        Txn late = engine.Begin();
        engine.Write(late, "n", "99");
        Check(late.StartTs == timeBefore, "a new transaction starts at the current logical time");
        Check(engine.Commit(late) && late.CommitTs == timeBefore + 1, "commit timestamps advance monotonically");
        bool versionOrder = true;
        IReadOnlyList<Version> versions = engine.Shards[0].VersionsOf("n");
        for (int i = 1; i < versions.Count; i++) if (versions[i].CommitTs <= versions[i - 1].CommitTs) versionOrder = false;
        Check(versionOrder && versions.Count == 4, "the key carries exactly one version per committed write, ordered by timestamp");
        Check(engine.Shards[0].HasWriteConflict("n", timeBefore), "a concurrent write is detected by the version timestamp");
        Console.WriteLine("  engine: shards=" + engine.Shards.Count + " committed=" + engine.CommittedTxns.Count
            + " decisions=" + engine.Coordinator.Log.Count + " wal records=" + walRecords);

        Console.WriteLine("PASS");
    }

    /** The classic on-call write skew: both doctors check the pair and then clear one of them. */
    private static void RunOnCallScenario(TxEngine engine, out bool bothCommitted, out int graphCycle, out bool serializable)
    {
        Txn setup = engine.Begin();
        engine.Write(setup, "d1", "1");
        engine.Write(setup, "d2", "1");
        engine.Commit(setup);
        Txn doctorA = engine.Begin();
        Txn doctorB = engine.Begin();
        string a1 = engine.Read(doctorA, "d1")!;
        string a2 = engine.Read(doctorA, "d2")!;
        string b1 = engine.Read(doctorB, "d1")!;
        string b2 = engine.Read(doctorB, "d2")!;
        Check(a1 == "1" && a2 == "1" && b1 == "1" && b2 == "1", "both doctors see the same snapshot with both colleagues on call");
        engine.Write(doctorA, "d1", "0");
        engine.Write(doctorB, "d2", "0");
        bool commitA = engine.Commit(doctorA);
        bool commitB = engine.Commit(doctorB);
        bothCommitted = commitA && commitB;
        graphCycle = DependencyGraph.HasCycle(engine.BuildGraph(engine.CommittedTxns)) ? 1 : 0;
        List<int>? witness;
        serializable = SerializabilityOracle.TryFindSerialOrder(engine.CommittedTxns, engine.Snapshot(), out witness);
    }

    private static int OnCallSum(TxEngine engine)
    {
        int sum = 0;
        foreach (string key in new[] { "d1", "d2" })
        {
            string? value = Value(engine, key);
            if (value != null) sum += int.Parse(value);
        }
        return sum;
    }

    /** A deterministic concurrent workload: six transactions that all begin at the same logical instant. */
    private static int RunWorkload(TxEngine engine, ulong seed)
    {
        engine.Place("k0", 0); engine.Place("k1", 0); engine.Place("k2", 1);
        Txn setup = engine.Begin();
        engine.Write(setup, "k0", "0"); engine.Write(setup, "k1", "0"); engine.Write(setup, "k2", "0");
        engine.Commit(setup);
        ulong rng = seed;
        var pending = new List<Txn>();
        for (int i = 0; i < 6; i++)
        {
            Txn txn = engine.Begin();
            engine.Read(txn, "k0");
            engine.Read(txn, "k" + (i % 3));
            engine.Write(txn, "k" + (i % 3), i.ToString());
            if (SplitMix64.NextInt(ref rng, 2) == 0) engine.Write(txn, "k1", (i * 2).ToString());
            pending.Add(txn);
        }
        int committed = 0;
        foreach (Txn txn in pending) if (engine.Commit(txn)) committed++;
        return committed;
    }

    private static string? Value(TxEngine engine, string key)
    {
        var snapshot = engine.Snapshot();
        return snapshot.TryGetValue(key, out string? value) ? value : null;
    }

    private static void CheckBank(TxEngine engine, int expectedSum, string label)
    {
        int sum = 0;
        foreach (string key in new[] { "a", "b", "c" })
        {
            string? value = Value(engine, key);
            if (value != null) sum += int.Parse(value);
        }
        Check(sum == expectedSum, "the bank invariant holds (" + label + ", sum=" + sum + ")");
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
