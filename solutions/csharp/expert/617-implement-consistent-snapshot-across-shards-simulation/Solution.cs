// Task 617 - Implement consistent snapshot across shards simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// Snapshot reads over a sharded, multi-version store. Every shard keeps one version chain per key -
// (commit timestamp, value) pairs appended in commit order - and a transaction is *atomic across
// shards*: it writes several keys under a single commit timestamp, but each shard applies only its
// own slice, so at any instant a shard can still be behind for a transaction another shard already
// applied. The snapshot protocol is the standard "safe point" one: the coordinator picks a read
// timestamp S, asks each shard whether it has applied every transaction with timestamp < S, and if a
// lagging shard refuses, it drops S below that shard's oldest pending commit and retries. Once every
// shard agrees, each shard answers with the newest version strictly below S for each of its keys.
//
// The simulation is deterministic and single-threaded: applications happen in an explicit schedule of
// (transaction, shard) pairs, and that schedule is what creates the lag - no clock, network or locks
// are involved. What is verified is the property the protocol exists for: the snapshot contains
// exactly the transactions committed before S, i.e. it equals an independent replay of all those
// transactions in timestamp order, so no multi-shard transaction is ever half-visible; per-key
// versions are non-decreasing as S grows; reads at the same S are repeatable; and a deliberately
// unsafe read that ignores readiness tears a transaction - the atomicity and oracle checks must both
// report it, so the verdict is not vacuous.

using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;          // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class Transaction
{
    internal int Ts;
    internal readonly List<int[]> Writes = new List<int[]>();     // (key, value)
}

internal sealed class Snapshot
{
    internal int Timestamp;
    internal int Attempts;
    internal readonly Dictionary<int, int[]> Visible = new Dictionary<int, int[]>();   // key -> (ts, value)

    internal int VisibleTs(int key) => Visible.ContainsKey(key) ? Visible[key][0] : -1;
    internal int VisibleValue(int key) => Visible.ContainsKey(key) ? Visible[key][1] : 0;
}

internal sealed class ShardedStore
{
    private readonly int shardCount;
    private readonly int keyCount;
    private readonly List<Transaction> transactions = new List<Transaction>();
    private readonly Dictionary<int, List<int[]>>[] chains;
    private readonly HashSet<int>[] pending;             // transaction indexes not yet applied per shard

    internal ShardedStore(int shardCount, int keyCount)
    {
        this.shardCount = shardCount;
        this.keyCount = keyCount;
        chains = new Dictionary<int, List<int[]>>[shardCount];
        pending = new HashSet<int>[shardCount];
        for (int shard = 0; shard < shardCount; shard++)
        {
            chains[shard] = new Dictionary<int, List<int[]>>();
            pending[shard] = new HashSet<int>();
            for (int key = shard; key < keyCount; key += shardCount) chains[shard][key] = new List<int[]>();
        }
    }

    internal int ShardCount => shardCount;
    internal int KeyCount => keyCount;
    internal int TransactionCount => transactions.Count;

    internal void AddTransaction(int ts, List<int[]> writes)
    {
        var transaction = new Transaction { Ts = ts };
        transaction.Writes.AddRange(writes);
        transactions.Add(transaction);
        var touched = new HashSet<int>();
        foreach (int[] write in writes) touched.Add(write[0] % shardCount);
        foreach (int shard in touched) pending[shard].Add(transactions.Count - 1);
    }

    /** Apply one transaction's slice on one shard; the other shards may still be behind. */
    internal void Apply(int transactionIndex, int shard)
    {
        Transaction transaction = transactions[transactionIndex];
        foreach (int[] write in transaction.Writes)
        {
            if (write[0] % shardCount != shard) continue;
            chains[shard][write[0]].Add(new[] { transaction.Ts, write[1] });
        }
        if (pending[shard].Remove(transactionIndex))
        {
            // The slice for this shard is now durable.
        }
    }

    internal bool ShardReady(int shard, int snapshotTs)
    {
        foreach (int index in pending[shard])
        {
            if (transactions[index].Ts < snapshotTs) return false;
        }
        return true;
    }

    internal bool AllShardsReady(int snapshotTs)
    {
        for (int shard = 0; shard < shardCount; shard++)
        {
            if (!ShardReady(shard, snapshotTs)) return false;
        }
        return true;
    }

    internal int OldestPendingTs()
    {
        int oldest = int.MaxValue;
        for (int shard = 0; shard < shardCount; shard++)
        {
            foreach (int index in pending[shard]) oldest = Math.Min(oldest, transactions[index].Ts);
        }
        return oldest;
    }

    internal int PendingCount()
    {
        var all = new HashSet<int>();
        for (int shard = 0; shard < shardCount; shard++)
        {
            foreach (int index in pending[shard]) all.Add(index);
        }
        return all.Count;
    }

    /** Coordinator protocol: retry with a lower timestamp until every shard is caught up. */
    internal Snapshot TakeSnapshot(int wanted)
    {
        int ts = wanted;
        int attempts = 0;
        for (int guard = 0; guard < 16; guard++)
        {
            attempts++;
            if (AllShardsReady(ts)) break;
            ts = OldestPendingTs();                       // step below the oldest pending commit
        }
        return Read(ts, attempts);
    }

    /** Read every key at its newest version strictly below the timestamp. */
    internal Snapshot Read(int snapshotTs, int attempts)
    {
        var snapshot = new Snapshot { Timestamp = snapshotTs, Attempts = attempts };
        for (int shard = 0; shard < shardCount; shard++)
        {
            foreach (var entry in chains[shard])
            {
                int bestTs = -1;
                int bestValue = 0;
                foreach (int[] version in entry.Value)
                {
                    if (version[0] < snapshotTs && version[0] > bestTs)
                    {
                        bestTs = version[0];
                        bestValue = version[1];
                    }
                }
                if (bestTs >= 0) snapshot.Visible[entry.Key] = new[] { bestTs, bestValue };
            }
        }
        return snapshot;
    }

    /** Oracle: replay every transaction with ts < snapshotTs in timestamp order. */
    internal Dictionary<int, int> OracleState(int snapshotTs)
    {
        var state = new Dictionary<int, int>();
        var ordered = new List<int>();
        for (int index = 0; index < transactions.Count; index++)
        {
            if (transactions[index].Ts < snapshotTs) ordered.Add(index);
        }
        ordered.Sort((a, b) => transactions[a].Ts.CompareTo(transactions[b].Ts));
        foreach (int index in ordered)
        {
            foreach (int[] write in transactions[index].Writes) state[write[0]] = write[1];
        }
        return state;
    }

    /** True when the snapshot equals the replay of the transactions committed before its timestamp. */
    internal bool MatchesOracle(Snapshot snapshot, out string report)
    {
        report = "";
        Dictionary<int, int> oracle = OracleState(snapshot.Timestamp);
        for (int key = 0; key < keyCount; key++)
        {
            int expected = oracle.ContainsKey(key) ? oracle[key] : 0;
            if (snapshot.VisibleValue(key) != expected)
            {
                report = "key " + key + " snapshot=" + snapshot.VisibleValue(key) + " oracle=" + expected;
                return false;
            }
        }
        return true;
    }

    /** True when no transaction below the snapshot timestamp is half-visible: every key a committed
     *  transaction wrote must show a version at least as new as that transaction. */
    internal bool Atomic(Snapshot snapshot, out string report)
    {
        report = "";
        foreach (Transaction transaction in transactions)
        {
            if (transaction.Ts >= snapshot.Timestamp) continue;
            foreach (int[] write in transaction.Writes)
            {
                if (snapshot.VisibleTs(write[0]) < transaction.Ts)
                {
                    report = "key " + write[0] + " shows ts " + snapshot.VisibleTs(write[0])
                        + " but transaction " + transaction.Ts + " had already written it";
                    return false;
                }
            }
        }
        return true;
    }
}

public static class Solution
{
    private static ulong state = 0x5A9D5A9DUL;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        const int shards = 3;
        const int keys = 12;
        var store = new ShardedStore(shards, keys);
        for (int i = 0; i < 12; i++)
        {
            ulong roll = SplitMix64.Next(ref state);
            int first = (int)(roll % keys);
            int second = (int)((roll >> 8) % keys);
            int third = (int)((roll >> 16) % keys);
            int ts = i + 1;
            store.AddTransaction(ts, new List<int[]>
            {
                new[] { first, ts * 100 + first },
                new[] { second, ts * 100 + second },
                new[] { third, ts * 100 + third },
            });
        }
        Check(store.TransactionCount == 12 && store.PendingCount() == 12, "all transactions pending");
        Check(!store.AllShardsReady(13) && store.ShardReady(0, 1), "readiness before any apply");

        // Explicit lag schedule: shard 2 never receives the last three transactions in this round.
        for (int index = 0; index < 12; index++)
        {
            for (int shard = 0; shard < shards; shard++)
            {
                if (shard == 2 && index >= 9) continue;
                store.Apply(index, shard);
            }
        }
        int oldest = store.OldestPendingTs();
        Check(!store.AllShardsReady(13), "the lagging shard blocks an eager snapshot");
        Check(oldest >= 10 && oldest <= 12, "oldest pending commit is in the lagging tail: " + oldest);

        Snapshot snapshot = store.TakeSnapshot(13);
        Check(snapshot.Attempts == 2, "one retry was needed, attempts=" + snapshot.Attempts);
        Check(snapshot.Timestamp == oldest, "safe point dropped to the oldest pending commit");
        Check(store.MatchesOracle(snapshot, out string oracleReport), "snapshot is the oracle state "
            + oracleReport);
        Check(store.Atomic(snapshot, out string atomicReport), "no torn transaction " + atomicReport);
        for (int key = 0; key < keys; key++)
        {
            Check(snapshot.VisibleTs(key) < snapshot.Timestamp, "versions strictly below the timestamp");
        }

        Snapshot repeat = store.Read(snapshot.Timestamp, 1);
        bool identical = repeat.Timestamp == snapshot.Timestamp;
        for (int key = 0; key < keys && identical; key++)
        {
            identical = repeat.VisibleTs(key) == snapshot.VisibleTs(key)
                && repeat.VisibleValue(key) == snapshot.VisibleValue(key);
        }
        Check(identical, "reads are repeatable at the same timestamp");

        // Negative control: a snapshot that ignores readiness must be caught by both checks.
        var tornStore = new ShardedStore(2, 2);
        tornStore.AddTransaction(1, new List<int[]> { new[] { 0, 70 }, new[] { 1, 90 } });
        tornStore.Apply(0, 0);                              // shard 1 still behind
        Check(!tornStore.AllShardsReady(2), "lagging shard blocks the snapshot");
        Snapshot torn = tornStore.Read(2, 1);
        Check(!tornStore.MatchesOracle(torn, out string tornOracle), "torn read differs from the oracle");
        Check(!tornStore.Atomic(torn, out string tornAtomic), "torn read detected: " + tornAtomic);
        Check(torn.VisibleTs(0) == 1 && torn.VisibleTs(1) == -1, "exactly one half of the pair is visible");
        tornStore.Apply(0, 1);
        Snapshot repaired = tornStore.TakeSnapshot(2);
        Check(repaired.Attempts == 1 && tornStore.MatchesOracle(repaired, out _)
            && tornStore.Atomic(repaired, out _), "the same read becomes valid once the shard catches up");
        Check(repaired.Timestamp == 2 && repaired.VisibleValue(0) == 70 && repaired.VisibleValue(1) == 90,
            "repaired snapshot content");

        // Catch the lagging shard up, then check monotonicity of every key as the timestamp grows.
        for (int index = 9; index < 12; index++) store.Apply(index, 2);
        Check(store.PendingCount() == 0 && store.AllShardsReady(13), "all shards caught up");
        var previous = new int[keys];
        for (int key = 0; key < keys; key++) previous[key] = -1;
        for (int wanted = 1; wanted <= 14; wanted++)
        {
            Snapshot later = store.TakeSnapshot(wanted);
            Check(later.Timestamp == wanted, "no retry needed at " + wanted);
            Check(store.MatchesOracle(later, out string laterReport), "oracle at " + wanted + " " + laterReport);
            Check(store.Atomic(later, out string laterAtomic), "atomic at " + wanted + " " + laterAtomic);
            for (int key = 0; key < keys; key++)
            {
                Check(later.VisibleTs(key) >= previous[key], "monotone versions for key " + key);
                previous[key] = later.VisibleTs(key);
            }
        }
        Snapshot full = store.TakeSnapshot(14);
        Check(store.MatchesOracle(full, out string fullReport), "complete snapshot is the full state "
            + fullReport);
        Check(full.Visible.Count == keys, "every key has a version once all commits are visible");

        // Boundaries: before the first commit, and right after it.
        Snapshot empty = store.TakeSnapshot(1);
        Check(empty.Timestamp == 1 && empty.Visible.Count == 0, "snapshot before the first commit is empty");
        Check(store.MatchesOracle(empty, out _) && store.Atomic(empty, out _), "empty snapshot is valid");
        Snapshot single = store.Read(2, 1);
        int firstCommitKeys = 0;
        foreach (var entry in single.Visible)
        {
            if (entry.Value[0] == 1) firstCommitKeys++;
        }
        Check(firstCommitKeys > 0, "one commit visible");
        Check(store.MatchesOracle(single, out string singleReport), "single commit oracle " + singleReport);

        // A store whose transactions each touch a single shard still needs the safe point.
        var local = new ShardedStore(2, 4);
        local.AddTransaction(1, new List<int[]> { new[] { 0, 10 } });
        local.AddTransaction(2, new List<int[]> { new[] { 2, 20 } });
        local.Apply(0, 0);
        Snapshot localSnapshot = local.TakeSnapshot(3);
        Check(localSnapshot.Attempts == 2 && localSnapshot.Timestamp == 2,
            "an unapplied single-shard transaction still delays the safe point");
        local.Apply(1, 0);
        Snapshot localReady = local.TakeSnapshot(3);
        Check(localReady.Attempts == 1 && localReady.VisibleValue(0) == 10
            && localReady.VisibleValue(2) == 20, "single-shard snapshot after catching up");

        Console.WriteLine("  shards=" + shards + " keys=" + keys + " transactions=" + store.TransactionCount
            + " safePoint=" + snapshot.Timestamp + " attempts=" + snapshot.Attempts
            + " visible=" + snapshot.Visible.Count);
        Console.WriteLine("  monotoneTimestamps=1..14 oracleMatched=true atomicityVerified=true");
        Console.WriteLine("PASS");
    }
}
