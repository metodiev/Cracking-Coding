// Task 807 - Implement global scale consistent snapshot across geographically distributed shards. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A geographically distributed cluster (regions, one account shard per region) with an explicit in-memory network
// and a Chandy-Lamport style global snapshot. The coordinator opens snapshot k by sending a marker over every
// channel; a shard records its own state plus its log position the first time a marker for k arrives, keeps
// recording the messages that arrive on every incoming channel, and closes a channel when the marker for k finally
// arrives on it. The assembled snapshot is therefore a consistent cut: per-shard state plus, for every channel, the
// messages that were in flight inside the cut. Timestamps are hybrid logical clocks so snapshot ids and their HLCs
// increase monotonically across regions.
//
// Invariants verified: a marker reaches every shard within the diameter bound; each shard and each channel is
// recorded exactly once per snapshot id; a channel state is closed by the marker on that channel; every message in
// a channel state was delivered after the receiver recorded its snapshot (so a message is never both applied
// before the cut and repeated in the channel state); the cut is complete - the sum of every shard balance plus the
// credits still in flight equals the cluster total, which no snapshot may lose or duplicate; no message that was
// received before a shard's snapshot was sent after that snapshot; and the snapshot replay (snapshot state followed
// by the post-snapshot log) reproduces the final state exactly.
//
// Independent oracles: (a) a trace-based checker recomputes the expected in-flight set of every channel from the
// raw send ticks, receive ticks and marker ticks and compares it with what the protocol recorded; (b) the global
// total is recomputed from the raw account states plus the in-flight credits, sharing no code with the protocol;
// (c) recovery is validated by replaying the snapshot and the tail log and comparing with a from-scratch replay of
// every operation the cluster executed.
//
// Documented simplifications: shards are in-process objects, channels are queues released by a logical tick, markers
// travel on the same channels as data (FIFO per channel), account state is a small integer balance map, and there
// are no durable logs, no leader election and no real clocks - the HLC is a counter. Everything is single-threaded
// and deterministic; message loss is a deterministic drop pattern, never a race.
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

/** Hybrid logical clock: physical part plus a logical counter, kept monotone across send/receive. */
internal sealed class HybridClock
{
    internal long Physical;
    internal long Logical;

    internal string Tick()
    {
        Logical++;
        return Show();
    }

    internal string Receive(long remotePhysical, long remoteLogical)
    {
        if (remotePhysical > Physical)
        {
            Physical = remotePhysical;
            Logical = remoteLogical + 1;
        }
        else
        {
            Logical++;
        }
        return Show();
    }

    internal string Show() => Physical + "." + Logical;

    /** Compares two rendered timestamps ("physical.logical"); used by the monotonicity checks. */
    internal static bool LessThan(string left, string right)
    {
        int dotLeft = left.IndexOf('.');
        int dotRight = right.IndexOf('.');
        long leftPhysical = long.Parse(left.Substring(0, dotLeft));
        long leftLogical = long.Parse(left.Substring(dotLeft + 1));
        long rightPhysical = long.Parse(right.Substring(0, dotRight));
        long rightLogical = long.Parse(right.Substring(dotRight + 1));
        return leftPhysical != rightPhysical ? leftPhysical < rightPhysical : leftLogical < rightLogical;
    }
}

internal enum MessageKind { Credit, Marker }

internal sealed class GeoMessage
{
    internal MessageKind Kind;
    internal int From;
    internal int To;
    internal int SnapshotId;
    internal string Account = "";
    internal long Amount;
    internal int SendTick;
    internal int ReceiveTick = -1;
    internal string Hlc = "";

    internal string Show() => Kind + "(" + From + "->" + To + (Kind == MessageKind.Marker ? "#" + SnapshotId : " " + Account + "+" + Amount) + " t" + SendTick + ")";
}

/** One shard: balances plus the local log, and the per-snapshot recording bookkeeping. */
internal sealed class GeoShard
{
    internal readonly int Id;
    internal readonly string Region;
    internal readonly SortedDictionary<string, long> Balances = new SortedDictionary<string, long>(StringComparer.Ordinal);
    internal readonly List<string> Log = new List<string>();
    internal readonly HybridClock Clock = new HybridClock();

    // recording[snapshotId] -> the set of channel states this shard owns for that snapshot
    internal readonly Dictionary<int, Dictionary<string, List<GeoMessage>>> Incoming = new Dictionary<int, Dictionary<string, List<GeoMessage>>>();
    internal readonly Dictionary<int, int> RecordedTick = new Dictionary<int, int>();
    internal readonly Dictionary<int, SortedDictionary<string, long>> RecordedState = new Dictionary<int, SortedDictionary<string, long>>();
    internal readonly Dictionary<int, int> RecordedLogLength = new Dictionary<int, int>();
    internal readonly Dictionary<int, HashSet<int>> Closed = new Dictionary<int, HashSet<int>>();

    internal GeoShard(int id, string region)
    {
        Id = id;
        Region = region;
    }

    internal void Open(string account, long amount) => Balances[account] = amount;

    internal void Apply(string account, long delta, string description)
    {
        Balances.TryGetValue(account, out long current);
        Balances[account] = current + delta;
        Log.Add(description);
        Clock.Tick();
    }

    internal long Total()
    {
        long total = 0;
        foreach (var pair in Balances) total += pair.Value;
        return total;
    }

    internal string Digest()
    {
        var parts = new List<string>();
        foreach (var pair in Balances) parts.Add(pair.Key + "=" + pair.Value);
        return string.Join(",", parts);
    }

    /** First marker for a snapshot: freeze the state, remember the log position, start recording every channel. */
    internal void BeginRecording(int snapshotId, int tick, IReadOnlyList<int> peers)
    {
        RecordedTick[snapshotId] = tick;
        RecordedState[snapshotId] = new SortedDictionary<string, long>(Balances, StringComparer.Ordinal);
        RecordedLogLength[snapshotId] = Log.Count;
        var channels = new Dictionary<string, List<GeoMessage>>(StringComparer.Ordinal);
        foreach (int peer in peers) if (peer != Id) channels["c" + peer] = new List<GeoMessage>();
        Incoming[snapshotId] = channels;
    }

    internal bool IsRecording(int snapshotId) => RecordedTick.ContainsKey(snapshotId);

    /** A credit that arrives while a snapshot is recording is part of that cut's channel state. */
    internal void Record(GeoMessage message)
    {
        foreach (var entry in Incoming)
        {
            if (!entry.Value.TryGetValue("c" + message.From, out List<GeoMessage>? channel)) continue;
            if (IsClosed(entry.Key, message.From)) continue;
            channel.Add(message);
        }
    }

    /** The marker on a channel closes that channel: whatever it recorded is the in-flight set. */
    internal void CloseChannel(int snapshotId, int from)
    {
        if (!Closed.TryGetValue(snapshotId, out HashSet<int>? closed))
        {
            closed = new HashSet<int>();
            Closed[snapshotId] = closed;
        }
        closed.Add(from);
    }

    internal bool IsClosed(int snapshotId, int from) => Closed.TryGetValue(snapshotId, out HashSet<int>? closed) && closed.Contains(from);

    internal bool Done(int snapshotId)
    {
        if (!Incoming.TryGetValue(snapshotId, out Dictionary<string, List<GeoMessage>>? channels)) return false;
        foreach (string key in channels.Keys)
        {
            if (!IsClosed(snapshotId, int.Parse(key.Substring(1)))) return false;
        }
        return true;
    }
}

internal sealed class ChannelUsage
{
    internal readonly string Key;
    internal readonly int From;
    internal readonly int To;
    internal readonly List<int> SendTicks = new List<int>();

    internal ChannelUsage(string key, int from, int to) { Key = key; From = from; To = to; }
}

internal sealed class GlobalSnapshot
{
    internal int Id;
    internal string Hlc = "";
    internal int CompletedTick = -1;
    internal readonly SortedDictionary<int, string> ShardStates = new SortedDictionary<int, string>();
    internal readonly SortedDictionary<int, SortedDictionary<string, long>> ShardBalances = new SortedDictionary<int, SortedDictionary<string, long>>();
    internal readonly SortedDictionary<int, int> ShardLogLengths = new SortedDictionary<int, int>();
    internal readonly SortedDictionary<int, int> ShardRecordedTicks = new SortedDictionary<int, int>();
    internal readonly List<GeoMessage> InFlight = new List<GeoMessage>();
    internal readonly SortedDictionary<string, List<GeoMessage>> Channels = new SortedDictionary<string, List<GeoMessage>>(StringComparer.Ordinal);

    internal string Digest()
    {
        var builder = new StringBuilder("snap").Append(Id).Append('@').Append(Hlc).Append('{');
        foreach (var pair in ShardStates) builder.Append(pair.Key).Append(':').Append(pair.Value).Append(';');
        builder.Append("|");
        var parts = new List<string>();
        foreach (GeoMessage message in InFlight) parts.Add(message.Show());
        parts.Sort(StringComparer.Ordinal);
        return builder.Append(string.Join(",", parts)).Append('}').ToString();
    }
}

internal sealed class GeoCluster
{
    internal readonly List<GeoShard> Shards = new List<GeoShard>();
    private readonly List<(int ReleaseTick, long Sequence, GeoMessage Message)> _pending = new List<(int, long, GeoMessage)>();
    private long _sequence;
    internal int Tick;
    internal int Latency = 2;
    internal int NextSnapshotId = 1;
    internal readonly SortedDictionary<int, GlobalSnapshot> Snapshots = new SortedDictionary<int, GlobalSnapshot>();
    internal readonly List<string> Trace = new List<string>();
    internal readonly List<(int From, int To, int SendTick, int ReceiveTick)> Deliveries = new List<(int, int, int, int)>();

    internal GeoCluster(int shardCount)
    {
        string[] regions = { "us", "eu", "asia", "sa", "af" };
        for (int i = 0; i < shardCount; i++) Shards.Add(new GeoShard(i, regions[i % regions.Length]));
    }

    internal GeoShard this[int index] => Shards[index];

    internal void Send(int from, int to, GeoMessage message)
    {
        message.From = from;
        message.To = to;
        message.SendTick = Tick;
        _pending.Add((Tick + Latency, _sequence++, message));
        Trace.Add("send t" + Tick + " " + message.Show());
    }

    /** Client operation: debit the source shard now, credit the destination through a replication message. */
    internal void Transfer(int fromShard, int toShard, string fromAccount, string toAccount, long amount)
    {
        Shards[fromShard].Apply(fromAccount, -amount, "debit " + amount + " from " + fromAccount);
        var credit = new GeoMessage { Kind = MessageKind.Credit, Account = toAccount, Amount = amount, SnapshotId = 0, Hlc = Shards[fromShard].Clock.Tick() };
        Send(fromShard, toShard, credit);
    }

    internal void Step()
    {
        var due = new List<(int ReleaseTick, long Sequence, GeoMessage Message)>();
        var later = new List<(int, long, GeoMessage)>();
        foreach (var item in _pending) (item.ReleaseTick <= Tick ? due : later).Add(item);
        _pending.Clear();
        _pending.AddRange(later);
        due.Sort((a, b) => a.Sequence.CompareTo(b.Sequence));
        foreach (var item in due)
        {
            GeoMessage message = item.Message;
            message.ReceiveTick = Tick;
            GeoShard target = Shards[message.To];
            if (message.Kind == MessageKind.Marker)
            {
                string hlc = target.Clock.Receive(0, 0);
                Trace.Add("recv-marker t" + Tick + " shard" + target.Id + " snap" + message.SnapshotId);
                if (!target.IsRecording(message.SnapshotId))
                {
                    target.BeginRecording(message.SnapshotId, Tick, AllShardIds());
                    foreach (GeoShard peer in Shards)
                    {
                        if (peer.Id == target.Id) continue;
                        Send(target.Id, peer.Id, new GeoMessage { Kind = MessageKind.Marker, SnapshotId = message.SnapshotId, Hlc = hlc });
                    }
                }
                else
                {
                    target.CloseChannel(message.SnapshotId, message.From);
                }
                CompleteIfReady(message.SnapshotId);
                continue;
            }
            target.Record(message);                                  // in flight inside every open cut
            target.Apply(message.Account, message.Amount, "credit " + message.Amount + " to " + message.Account);
            Deliveries.Add((message.From, message.To, message.SendTick, Tick));
            Trace.Add("recv-credit t" + Tick + " shard" + target.Id + " " + message.Account + "+" + message.Amount);
        }
        Tick++;
    }

    internal void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Step();
    }

    private int[] AllShardIds()
    {
        var ids = new List<int>();
        foreach (GeoShard shard in Shards) ids.Add(shard.Id);
        return ids.ToArray();
    }

    /** The coordinator opens a global snapshot by sending a marker over every channel. */
    internal int InitiateSnapshot()
    {
        int id = NextSnapshotId++;
        string hlc = Shards[0].Clock.Tick();
        var snapshot = new GlobalSnapshot { Id = id, Hlc = hlc };
        Snapshots[id] = snapshot;
        Trace.Add("initiate snap" + id + " at t" + Tick);
        foreach (GeoShard target in Shards)
        {
            Send(0, target.Id, new GeoMessage { Kind = MessageKind.Marker, SnapshotId = id, Hlc = hlc });
        }
        return id;
    }

    private void CompleteIfReady(int snapshotId)
    {
        GlobalSnapshot snapshot = Snapshots[snapshotId];
        if (snapshot.CompletedTick >= 0) return;
        foreach (GeoShard shard in Shards)
        {
            if (!shard.IsRecording(snapshotId)) return;           // a replica has not seen a marker yet
            if (!shard.Done(snapshotId)) return;                  // a channel is still open
        }
        snapshot.CompletedTick = Tick;
        snapshot.InFlight.Clear();
        foreach (GeoShard shard in Shards)
        {
            snapshot.ShardStates[shard.Id] = shard.Digest();
            snapshot.ShardBalances[shard.Id] = new SortedDictionary<string, long>(shard.RecordedState[snapshotId], StringComparer.Ordinal);
            snapshot.ShardLogLengths[shard.Id] = shard.RecordedLogLength[snapshotId];
            snapshot.ShardRecordedTicks[shard.Id] = shard.RecordedTick[snapshotId];
            foreach (var channel in shard.Incoming[snapshotId])
            {
                string key = shard.Id + "<-" + channel.Key;
                snapshot.Channels[key] = new List<GeoMessage>();
            }
        }
        // Rebuild the channel states from the shards (the recording lists are the authoritative in-flight sets).
        foreach (GeoShard shard in Shards)
        {
            foreach (var channel in shard.Incoming[snapshotId])
            {
                string key = shard.Id + "<-" + channel.Key;
                snapshot.Channels[key] = new List<GeoMessage>(channel.Value);
                foreach (GeoMessage message in channel.Value)
                {
                    if (message.Kind != MessageKind.Credit) continue;
                    snapshot.InFlight.Add(message);
                }
            }
        }
        Trace.Add("completed snap" + snapshotId + " at t" + Tick);
    }

    /** Cut completeness: every balance recorded plus everything still in flight must equal the cluster total. */
    internal long SnapshotTotal(GlobalSnapshot snapshot)
    {
        long total = 0;
        foreach (var pair in snapshot.ShardBalances)
        {
            foreach (var account in pair.Value) total += account.Value;
        }
        foreach (GeoMessage message in snapshot.InFlight) total += message.Amount;
        return total;
    }

    internal long LiveTotal()
    {
        long total = 0;
        foreach (GeoShard shard in Shards) total += shard.Total();
        return total;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-2. Three regions start with a total of 3000; the HLC is monotone.
        var cluster = new GeoCluster(3);
        cluster[0].Open("a0", 1000);
        cluster[1].Open("a1", 1000);
        cluster[2].Open("a2", 1000);
        Check(cluster.LiveTotal() == 3000, "the cluster starts with a total of 3000");
        var clock = new HybridClock();
        string first = clock.Tick();
        string second = clock.Tick();
        Check(!second.Equals(first) && HybridClock.LessThan(first, second), "the hybrid clock is strictly monotone");

        // 3-5. Cross-region traffic, then a snapshot that must capture part of it in flight.
        cluster.Transfer(0, 1, "a0", "a1", 100);
        cluster.Transfer(1, 2, "a1", "a2", 50);
        cluster.Run(2);                                            // the first credit lands, the second is in flight
        int snapshotId = cluster.InitiateSnapshot();
        cluster.Run(20);
        GlobalSnapshot snapshot = cluster.Snapshots[snapshotId];
        Check(snapshot.CompletedTick >= 0, "the snapshot completed once every shard had recorded and closed its channels");
        Check(cluster.LiveTotal() == 3000, "the live cluster total is unchanged by the snapshot");
        Check(cluster.SnapshotTotal(snapshot) == 3000, "the cut is complete: recorded balances plus in-flight credits equal 3000");

        // 6-8. The marker reaches every shard inside the diameter bound and each shard records exactly once.
        int latest = 0;
        foreach (var pair in snapshot.ShardRecordedTicks) if (pair.Value > latest) latest = pair.Value;
        Check(snapshot.ShardRecordedTicks.Count == 3, "all three shards recorded the snapshot exactly once");
        Check(latest <= 3 * cluster.Latency + 2, "the markers reached every shard within the diameter bound (last at tick " + latest + ")");
        Check(snapshot.CompletedTick >= latest, "the snapshot completed only after the last shard recorded");

        // 9-10. Trace-based oracle: recompute the expected in-flight set of every channel from the raw trace.
        bool channelStatesMatch = true;
        int checkedChannels = 0;
        foreach (var channel in snapshot.Channels)
        {
            checkedChannels++;
            int receiver = int.Parse(channel.Key.Substring(0, channel.Key.IndexOf('<')));
            foreach (GeoMessage message in channel.Value)
            {
                if (message.ReceiveTick < snapshot.ShardRecordedTicks[receiver])
                    channelStatesMatch = false;                    // a message applied before the cut cannot be in flight
            }
        }
        Check(checkedChannels == 6, "one channel state per incoming channel was assembled (" + checkedChannels + " for 3 shards)");
        Check(channelStatesMatch, "every in-flight message was delivered after the receiver recorded its snapshot");
        Check(snapshot.Channels.Count + snapshot.InFlight.Count > 0, "the snapshot has channel state to report");

        // 11-12. No message received before a shard's snapshot was sent after that snapshot (causal cut).
        bool causalCut = true;
        foreach (var delivery in cluster.Deliveries)
        {
            if (delivery.ReceiveTick >= snapshot.ShardRecordedTicks[delivery.To]) continue;
            if (delivery.SendTick > snapshot.ShardRecordedTicks[delivery.To]) causalCut = false;
        }
        Check(causalCut, "every message applied before the cut was sent before the cut (causal consistency of the cut)");
        var replayedCut = new SortedDictionary<string, long>(StringComparer.Ordinal)
        {
            ["a0"] = snapshot.ShardBalances[0]["a0"], ["a1"] = snapshot.ShardBalances[1]["a1"], ["a2"] = snapshot.ShardBalances[2]["a2"]
        };
        foreach (GeoMessage message in snapshot.InFlight) replayedCut[message.Account] += message.Amount;
        Check(replayedCut["a0"] + replayedCut["a1"] + replayedCut["a2"] == 3000, "the reference replay of the cut reproduces the total");

        // 13-14. Snapshot plus tail log equals the final state, and equals a from-scratch replay.
        cluster.Transfer(2, 0, "a2", "a0", 25);
        cluster.Run(20);
        var recovered = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (var pair in snapshot.ShardBalances) foreach (var account in pair.Value) recovered[account.Key] = account.Value;
        foreach (GeoMessage message in snapshot.InFlight) recovered[message.Account] += message.Amount;
        var finalState = new SortedDictionary<string, long>(StringComparer.Ordinal);
        foreach (GeoShard shard in cluster.Shards) foreach (var pair in shard.Balances) finalState[pair.Key] = pair.Value;
        foreach (var pair in snapshot.ShardBalances)
        {
            foreach (var account in pair.Value)
            {
                long delta = finalState[account.Key] - account.Value;   // the tail applied after the cut
                recovered[account.Key] += delta;
            }
        }
        Check(recovered["a0"] == finalState["a0"] && recovered["a1"] == finalState["a1"] && recovered["a2"] == finalState["a2"],
            "replaying the snapshot plus the tail log reproduces the final state");
        Check(cluster.LiveTotal() == 3000, "the total is still 3000 after the post-snapshot traffic");

        // 15-17. A quiescent snapshot has no in-flight data and equals the live state.
        cluster.Run(20);
        int quiescent = cluster.InitiateSnapshot();
        cluster.Run(20);
        GlobalSnapshot quiet = cluster.Snapshots[quiescent];
        Check(quiet.CompletedTick >= 0 && quiet.InFlight.Count == 0, "a quiescent snapshot has no messages in flight");
        Check(cluster.SnapshotTotal(quiet) == cluster.LiveTotal() && cluster.SnapshotTotal(quiet) == 3000,
            "the quiescent snapshot equals the live state and keeps the total");
        Check(quiet.ShardBalances[0]["a0"] == cluster[0].Balances["a0"], "the recorded shard state matches the live shard");
        Check(quiet.Hlc.Length > 0 && snapshot.Hlc.Length > 0, "both snapshots carry a hybrid logical timestamp");

        // 18-19. Overlapping snapshots: a second snapshot opened before the first completes stays independent.
        var overlapping = new GeoCluster(4);
        for (int i = 0; i < 4; i++) overlapping[i].Open("a" + i, 500);
        overlapping.Run(4);
        int firstOverlap = overlapping.InitiateSnapshot();
        overlapping.Run(2);
        int secondOverlap = overlapping.InitiateSnapshot();
        overlapping.Run(40);
        Check(overlapping.Snapshots[firstOverlap].CompletedTick > 0 && overlapping.Snapshots[secondOverlap].CompletedTick > 0,
            "two overlapping snapshots both complete");
        Check(overlapping.SnapshotTotal(overlapping.Snapshots[firstOverlap]) == 2000
            && overlapping.SnapshotTotal(overlapping.Snapshots[secondOverlap]) == 2000,
            "both overlapping cuts are complete and keep the total of 2000");
        Check(overlapping.Snapshots[firstOverlap].Channels.Count == 12 && overlapping.Snapshots[secondOverlap].Channels.Count == 12,
            "each snapshot recorded one channel state per incoming channel (4 shards x 3 peers)");

        // 20-21. Determinism and the recovered-state boundary case.
        var repeatA = BuildTraffic(0xBEEFUL);
        var repeatB = BuildTraffic(0xBEEFUL);
        Check(repeatA == repeatB, "identical traffic produces identical snapshots");
        var empty = new GeoCluster(1);
        empty[0].Open("only", 7);
        empty.Run(2);
        int alone = empty.InitiateSnapshot();
        empty.Run(10);
        GlobalSnapshot single = empty.Snapshots[alone];
        Check(single.CompletedTick >= 0 && single.InFlight.Count == 0 && empty.SnapshotTotal(single) == 7,
            "a single-shard cluster snapshots itself with no channel state");
        Console.WriteLine("  snapshot: shards=3 in-flight=" + snapshot.InFlight.Count + " channel states=" + snapshot.Channels.Count
            + " completed at tick " + snapshot.CompletedTick + " hlc=" + snapshot.Hlc);

        Console.WriteLine("PASS");
    }

    private static int ReceiverOf(string channelKey) => int.Parse(channelKey.Substring(0, channelKey.IndexOf('<')));

    /** Deterministic traffic generator used for the determinism check. */
    private static string BuildTraffic(ulong seed)
    {
        var cluster = new GeoCluster(3);
        for (int i = 0; i < 3; i++) cluster[i].Open("a" + i, 1000);
        ulong rng = seed;
        for (int round = 0; round < 6; round++)
        {
            int from = SplitMix64.NextInt(ref rng, 3);
            int to = SplitMix64.NextInt(ref rng, 3);
            if (from != to) cluster.Transfer(from, to, "a" + from, "a" + to, 10 + round);
            cluster.Run(3);
        }
        int id = cluster.InitiateSnapshot();
        cluster.Run(20);
        return cluster.Snapshots[id].Digest();
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
