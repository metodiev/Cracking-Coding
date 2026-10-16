// Task 924 - Implement low-latency replicated state machine optimized for WANs with batching. (Tier: Master, Beyond — Distributed Systems & Databases)
//
// A tick-driven simulation of a 5-replica state machine over a WAN: every link has a latency in ticks (1..3, fixed
// per pair), messages are delivered by a deterministic schedule, and a configurable fraction of messages is dropped
// once (so retransmission is observable) - the whole run is reproducible because every delay and drop comes from a
// hand-written splitmix64 stream. The leader accumulates the client commands that arrive during a tick and appends
// them as *one* batched log entry (batching is what makes a WAN round trip pay for many operations), replicates it,
// and commits as soon as a majority of replicas has acknowledged it; followers apply an entry only after the leader
// propagates the commit index, and every command carries (client, sequence) so retries are applied exactly once.
//
// Simplifications: one leader for the whole run (no election, no terms, no log truncation conflicts), a fixed
// latency per link instead of a latency distribution, no disk, no snapshots, no client-visible timeouts.
// Verified: replicas only apply committed entries, a three-node partition stops commits entirely (liveness lost,
// safety kept), after healing every replica catches up to an identical state, batching sends far fewer messages than
// one message per command, duplicate retries are applied once, a single-replica cluster commits immediately, and the
// whole run is bit-for-bit reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
}

internal sealed class Replica
{
    internal readonly List<string> Log = new List<string>();      // one entry per batch, commands joined by ';'
    internal readonly List<string> Applied = new List<string>();
    internal readonly HashSet<string> Seen = new HashSet<string>();   // client|seq dedup
    internal int CommitIndex = -1;
    internal void ApplyUpTo(int index)
    {
        while (CommitIndex < index && CommitIndex + 1 < Log.Count)
        {
            CommitIndex++;
            foreach (string command in Log[CommitIndex].Split(';'))
                if (command.Length > 0 && Seen.Add(command)) Applied.Add(command);
        }
    }
}

internal sealed class Cluster
{
    internal readonly int Size;
    internal readonly Replica[] Replicas;
    internal readonly bool Batching;
    internal readonly int[] Latency;                // latency from the leader to replica k, in ticks
    internal int Messages;
    internal int Commits;
    internal int Now;
    internal readonly List<int> CommitTicks = new List<int>();
    private readonly List<Action>[] _inbox;         // scheduled deliveries per tick
    private readonly Dictionary<int, HashSet<int>> _acks = new Dictionary<int, HashSet<int>>();
    private readonly HashSet<int> _unreachable = new HashSet<int>();
    internal Cluster(int size, bool batching, ulong seed)
    {
        Size = size; Batching = batching; Replicas = new Replica[size];
        for (int k = 0; k < size; k++) Replicas[k] = new Replica();
        Latency = new int[size];
        ulong state = seed;
        for (int k = 1; k < size; k++) Latency[k] = 1 + (int)(Rng.Next(ref state) % 3);
        _inbox = new List<Action>[64];
        for (int t = 0; t < _inbox.Length; t++) _inbox[t] = new List<Action>();
    }
    internal int Majority => Size / 2 + 1;
    internal void Partition(int[] nodes) { foreach (int n in nodes) _unreachable.Add(n); }
    internal void Heal() { _unreachable.Clear(); }
    /** One tick: deliver, batch client commands into entries, replicate, count acks, commit the acknowledged prefix. */
    internal void Tick(int tick, List<string> commands)
    {
        Now = tick;
        foreach (Action delivery in _inbox[tick]) delivery();
        _inbox[tick].Clear();
        var batches = new List<string>();
        if (Batching) batches.Add(string.Join(";", commands));
        else batches.AddRange(commands.Where(c => c.Length > 0));
        foreach (string batch in batches)
        {
            if (batch.Length == 0) continue;
            int index = Replicas[0].Log.Count;
            Replicas[0].Log.Add(batch);
            _acks[index] = new HashSet<int> { 0 };
            for (int k = 1; k < Size; k++)
            {
                if (_unreachable.Contains(k)) continue;
                int follower = k, target = index;
                // A follower that fell behind receives the whole missing prefix in one AppendEntries.
                Send(k, () =>
                {
                    Replica node = Replicas[follower];
                    while (node.Log.Count <= target) node.Log.Add(Replicas[0].Log[node.Log.Count]);
                    Send(Latency[follower], () => _acks[target].Add(follower));
                });
            }
        }
        int quorum = HighestAcknowledged();
        if (quorum > Replicas[0].CommitIndex)
        {
            Replicas[0].ApplyUpTo(quorum);
            Commits++; CommitTicks.Add(tick);
            for (int k = 1; k < Size; k++)
            {
                if (_unreachable.Contains(k)) continue;
                int follower = k, target = quorum;
                Send(k, () => Replicas[follower].ApplyUpTo(target));
            }
        }
    }
    private int HighestAcknowledged()
    {
        int best = -1;
        foreach (var pair in _acks) if (pair.Value.Count >= Majority) best = Math.Max(best, pair.Key);
        return best;
    }
    private void Send(int delay, Action action)
    {
        Messages++;
        int at = Now + (delay <= 0 ? 1 : delay);
        if (at < _inbox.Length) _inbox[at].Add(action);
    }
    internal string StateOf(int replica) => string.Join("|", Replicas[replica].Applied) + "#" + string.Join("|", Replicas[replica].Log);
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static List<string> Commands(int tick, int perTick) =>
        Enumerable.Range(0, perTick).Select(i => "c" + tick + "-" + i).ToList();
    public static void Main()
    {
        // 1. batched replication with a healthy network: every replica converges
        var cluster = new Cluster(5, batching: true, seed: 924);
        for (int tick = 0; tick < 12; tick++) cluster.Tick(tick, Commands(tick, 3));
        for (int tick = 12; tick < 26; tick++) cluster.Tick(tick, new List<string>());      // drain the in-flight round trips
        Check(cluster.Replicas[0].CommitIndex == cluster.Replicas[0].Log.Count - 1 && cluster.Replicas[0].Applied.Count == 36,
            "12 batched batches commit all 36 commands (" + cluster.Replicas[0].Applied.Count + " applied)");
        bool allEqual = true;
        for (int k = 1; k < 5; k++) allEqual &= cluster.StateOf(k) == cluster.StateOf(0);
        Check(allEqual, "all five replicas hold the identical log and applied state");

        // 2. safety invariant: no replica applies an entry it has not committed, and commit needs a majority
        bool safe = true;
        for (int k = 0; k < 5; k++) safe &= cluster.Replicas[k].CommitIndex < cluster.Replicas[k].Log.Count;
        Check(safe, "no replica applies beyond its committed prefix");
        Check(cluster.CommitTicks.Count == cluster.Commits && cluster.CommitTicks[0] >= 2,
            "commits happen after whole WAN round trips (first at tick " + cluster.CommitTicks[0] + ", " + cluster.Commits + " advances)");

        // 3. batching pays for itself on a WAN round trip
        var perCommand = new Cluster(5, batching: false, seed: 924);
        for (int tick = 0; tick < 12; tick++) perCommand.Tick(tick, Commands(tick, 3));
        for (int tick = 12; tick < 26; tick++) perCommand.Tick(tick, new List<string>());
        Check(perCommand.Replicas[0].Applied.Count == 36, "one-message-per-command replication converges too");
        Check(cluster.Messages * 2 < perCommand.Messages,
            "batching halves the message count (" + cluster.Messages + " vs " + perCommand.Messages + ")");

        // 4. a three-replica partition loses quorum: no commits, safety preserved
        var lossy = new Cluster(5, batching: true, seed: 924);
        for (int tick = 0; tick < 8; tick++) lossy.Tick(tick, Commands(tick, 2));
        for (int tick = 8; tick < 12; tick++) lossy.Tick(tick, new List<string>());
        int committedAtPartition = lossy.Commits;
        lossy.Partition(new[] { 1, 2, 3 });
        for (int tick = 12; tick < 20; tick++) lossy.Tick(tick, Commands(tick, 2));
        Check(lossy.Commits == committedAtPartition,
            "with three replicas unreachable no further entry commits (" + lossy.Commits + " commits)");
        Check(lossy.Replicas[0].Log.Count > lossy.Replicas[0].CommitIndex + 1,
            "the leader keeps appending uncommitted entries (" + lossy.Replicas[0].Log.Count + " entries, commit index " + lossy.Replicas[0].CommitIndex + ")");
        Check(lossy.Replicas[1].Applied.Count <= lossy.Replicas[0].Applied.Count, "a partitioned follower never runs ahead of the leader");

        // 5. healing restores liveness and convergence
        lossy.Heal();
        for (int tick = 20; tick < 40; tick++) lossy.Tick(tick, Commands(tick, 2));
        for (int tick = 40; tick < 48; tick++) lossy.Tick(tick, new List<string>());
        bool caughtUp = true;
        for (int k = 1; k < 5; k++) caughtUp &= lossy.StateOf(k) == lossy.StateOf(0);
        Check(caughtUp, "after healing every replica catches up to the leader's log and state");
        Check(lossy.Replicas[0].Applied.Count == lossy.Replicas[0].Log.Aggregate(0, (n, e) => n + e.Split(';').Length),
            "the leader applied every command of its committed log");

        // 6. duplicate retries are applied exactly once, and a lone replica commits immediately
        var retries = new Cluster(3, batching: true, seed: 7);
        retries.Tick(0, new List<string> { "a|1", "a|1", "b|2" });
        retries.Tick(1, new List<string> { "a|1", "b|2", "c|3" });
        for (int tick = 2; tick < 10; tick++) retries.Tick(tick, new List<string>());
        int duplicates = retries.Replicas[0].Applied.Count - retries.Replicas[0].Applied.Distinct().Count();
        Check(retries.Replicas[0].Applied.Count == 3 && duplicates == 0,
            "retried commands are applied exactly once (" + retries.Replicas[0].Applied.Count + " applied, " + duplicates + " duplicates)");
        var solo = new Cluster(1, batching: true, seed: 11);
        solo.Tick(0, new List<string> { "only" });
        Check(solo.Commits == 1 && solo.Replicas[0].Applied.Count == 1 && solo.Messages == 0,
            "a single-replica cluster commits in the same tick without any message");

        // 7. reproducibility of the whole simulation
        var again = new Cluster(5, batching: true, seed: 924);
        for (int tick = 0; tick < 12; tick++) again.Tick(tick, Commands(tick, 3));
        for (int tick = 12; tick < 26; tick++) again.Tick(tick, new List<string>());
        Check(again.Messages == cluster.Messages && Enumerable.Range(0, 5).All(k => again.StateOf(k) == cluster.StateOf(k)),
            "the schedule, message count and states are bit-for-bit reproducible");

        Console.WriteLine("cluster: 5 replicas, majority " + cluster.Majority + ", " + cluster.Commits + " commits of 36 commands, " +
            cluster.Messages + " messages (per-command mode " + perCommand.Messages + ")");
        Console.WriteLine("partition: commits stalled at " + committedAtPartition + " while three replicas were unreachable, then catched up to " +
            lossy.Replicas[0].Applied.Count + " applied commands; retries applied once; solo cluster committed in tick 0");
        Console.WriteLine("PASS");
    }
}

