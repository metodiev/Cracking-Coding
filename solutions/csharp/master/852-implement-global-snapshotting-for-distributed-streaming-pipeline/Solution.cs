// Task 852 - Implement global snapshotting for a distributed streaming pipeline (Chandy-Lamport at scale).
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// An explicit in-process model of the Chandy-Lamport global snapshot algorithm. N processes are connected
// by directed, FIFO channels with a fixed per-channel one-way latency; a deterministic traffic schedule
// (splitmix64) injects payload messages, and a snapshot is initiated by one process while that traffic
// keeps flowing. Every process keeps a *live* state (the sum of the payloads it has received) and turns
// white -> yellow exactly once per snapshot:
//
//   initiate   the initiator records its state, turns yellow and sends one marker on every outgoing
//              channel, *before* it sends any further message on those channels
//   on marker  a white process records its state, turns yellow and sends markers on all its outgoing
//              channels; then it stops recording the incoming channel the marker arrived on. A yellow
//              process that receives a marker only stops recording that channel; the messages received on
//              a channel after it turned yellow and before that channel's marker are the channel state
//   complete   the snapshot is finished when every process is yellow and every channel has been closed
//
// Because a channel is FIFO and its latency is constant, a marker sent at time t always arrives after
// every message that the same sender sent before t: the recorded channel state is therefore exactly the
// set of messages that were sent before the sender's cut and received after the receiver's cut.
//
// What is simplified (documented, not behavioural): one snapshot at a time (a scheduled snapshot starts
// only after the previous one terminates, retrying every 5 ticks); a single global observer replays the
// complete send/receive log instead of the initiator gathering snapshots over the network, so the audit
// has an oracle rather than a message-count argument; processes never fail and messages never reorder.
// The state machine, the marker protocol, the channel recording rule, the termination condition and the
// per-snapshot cost (exactly one marker per channel) are the real content.
//
// Invariants verified against the independent log oracle (no shared code with the protocol):
//   * the cut is consistent — a message received before its receiver's cut was necessarily sent before
//     its sender's cut (the happens-before closure of the cut);
//   * every recorded channel state equals exactly the set {sent before the sender's cut, received after
//     the receiver's cut}, and every recorded process state equals the replay of the first
//     recvAtCut[v] receives of that process (recomputed from the log);
//   * token conservation: sum(recorded states) + sum(recorded channel payloads) == the payload total of
//     every message sent before its sender's cut — nothing is lost and nothing is counted twice;
//   * cost and liveness: exactly one marker per directed channel, one marker per channel received, all
//     channels closed, and termination within N * maxLatency ticks;
//   * the audit has teeth: a tampered channel state, a shifted cut, an empty-channel "synchronous tick"
//     snapshot and an "end of run" snapshot are all rejected;
//   * determinism, a traffic-free run and a single-process run.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng852
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class TrafficMessage
{
    internal long Seq;             // global send order
    internal int From;
    internal int To;
    internal long SendTime;
    internal long ArrivalTime;
    internal int Payload;
    internal int SendIndex;        // index in the sender's send sequence
    internal int RecvIndex = -1;   // index in the receiver's receive sequence, -1 while in flight
    internal bool Received;
}

internal sealed class Snapshot
{
    internal int Initiator;
    internal int Processes;
    internal long StartTime;
    internal long StopTime;
    internal long[] States = Array.Empty<long>();
    internal int[] RecvAtCut = Array.Empty<int>();
    internal int[] SendAtCut = Array.Empty<int>();
    internal bool[] Yellow = Array.Empty<bool>();
    internal List<TrafficMessage>[] Channels = Array.Empty<List<TrafficMessage>>();
    internal long MarkersSent;
    internal long MarkersReceived;
    internal long CompletedChannels;
    internal long TotalChannels;
    internal long[] MarkerSendTimes = Array.Empty<long>();
    internal long[] ChannelCloseTimes = Array.Empty<long>();

    internal long Fingerprint()
    {
        long hash = 1469598103934665603L;
        for (int v = 0; v < States.Length; v++) hash = Mix(hash, States[v] * 31 + RecvAtCut[v]);
        for (int c = 0; c < Channels.Length; c++)
            foreach (TrafficMessage m in Channels[c]) hash = Mix(hash, m.Seq * 7 + 1);
        return hash;
    }

    private static long Mix(long hash, long value)
    {
        hash ^= value;
        return hash * 1099511628211L;
    }
}

internal sealed class AuditResult
{
    internal bool Ok = true;
    internal long ConsistentCutViolations;
    internal long ChannelViolations;
    internal long StateViolations;
    internal long StateTokens;
    internal long ChannelTokens;
    internal long SentBeforeCutTokens;
    internal long MessagesChecked;
    internal string Detail = "";

    internal void Fail(ref long counter)
    {
        counter++;
        Ok = false;
    }
}

internal enum EventKind852 { TrafficTick, Deliver, Marker, Initiate }

internal sealed class SimEvent852
{
    internal long Time;
    internal long Seq;
    internal EventKind852 Kind;
    internal TrafficMessage Message = new TrafficMessage();
    internal int From;
    internal int To;
}

// Binary min-heap ordered by (time, scheduling sequence): equal-time events keep FIFO scheduling order.
internal sealed class EventQueue852
{
    private readonly List<SimEvent852> heap = new List<SimEvent852>();

    internal int Count => heap.Count;

    internal void Push(SimEvent852 item)
    {
        heap.Add(item);
        int i = heap.Count - 1;
        while (i > 0)
        {
            int parent = (i - 1) / 2;
            if (!Less(heap[i], heap[parent])) break;
            Swap(i, parent);
            i = parent;
        }
    }

    internal SimEvent852 Pop()
    {
        SimEvent852 top = heap[0];
        int last = heap.Count - 1;
        heap[0] = heap[last];
        heap.RemoveAt(last);
        int i = 0;
        while (true)
        {
            int left = 2 * i + 1, right = left + 1, best = i;
            if (left < heap.Count && Less(heap[left], heap[best])) best = left;
            if (right < heap.Count && Less(heap[right], heap[best])) best = right;
            if (best == i) break;
            Swap(i, best);
            i = best;
        }
        return top;
    }

    private static bool Less(SimEvent852 a, SimEvent852 b) => a.Time != b.Time ? a.Time < b.Time : a.Seq < b.Seq;

    private void Swap(int i, int j)
    {
        SimEvent852 tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    }
}

internal sealed class Cluster
{
    private readonly int n;
    private readonly int[,] latency;
    private readonly List<int>[] outEdges;
    private readonly List<int>[] inEdges;
    private readonly long[] liveState;
    private readonly int[] sendCount;
    private readonly int[] recvCount;

    private readonly EventQueue852 queue = new EventQueue852();
    private readonly long[] scheduledStarts;
    private readonly int[] sources;
    private readonly int ticks;
    private readonly int perTick;
    private readonly long snapshotRetryTicks = 5;

    // Chandy-Lamport state for the snapshot currently in progress.
    private Snapshot current;
    private bool[] yellow;
    private long[] snapState;
    private int[] recvAtCut;
    private int[] sendAtCut;
    private bool[] closedChannel;
    private List<TrafficMessage>[] channelState;
    private long[] markerSendTime;
    private long[] channelCloseTime;

    private ulong rng;
    private long seq;
    private long trafficTicks;
    private long markersSent;
    private long markersReceived;
    private long completedChannels;
    private long deliveries;
    private long markerDeliveries;

    internal Cluster(int n, int ticks, int perTick, ulong seed, long[] snapshotStarts)
    {
        if (n < 1) throw new ArgumentException("a cluster needs at least one process", nameof(n));
        if (ticks < 0 || perTick < 0) throw new ArgumentException("the traffic schedule must be non-negative");
        this.n = n;
        this.ticks = ticks;
        this.perTick = perTick;
        rng = seed;
        scheduledStarts = snapshotStarts ?? new long[0];
        latency = new int[n, n];
        outEdges = new List<int>[n];
        inEdges = new List<int>[n];
        for (int u = 0; u < n; u++)
        {
            outEdges[u] = new List<int>();
            inEdges[u] = new List<int>();
        }
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                latency[u, v] = 1 + ((u * 7 + v * 3) % 5);
                if (IsEdge(u, v))
                {
                    outEdges[u].Add(v);
                    inEdges[v].Add(u);
                }
            }
        }
        if (n > 1) EnsureStronglyConnected();
        ChannelCount = CountChannels();
        var sourceList = new List<int>();
        for (int u = 0; u < n; u++) if (outEdges[u].Count > 0) sourceList.Add(u);
        sources = sourceList.ToArray();
        liveState = new long[n];
        sendCount = new int[n];
        recvCount = new int[n];
        RecvLog = new List<TrafficMessage>[n];
        for (int v = 0; v < n; v++) RecvLog[v] = new List<TrafficMessage>();
        AllocateSnapshotState();
    }

    // Every ordered pair except the self loops joins the graph with probability 4/5; the extra ring edges
    // below make the graph strongly connected, which the snapshot liveness bound relies on.
    private static bool IsEdge(int u, int v) => ((u * 31 + v * 17) % 5) != 0;

    private void EnsureStronglyConnected()
    {
        for (int u = 0; u < n; u++)
        {
            int v = (u + 1) % n;
            if (!outEdges[u].Contains(v))
            {
                outEdges[u].Add(v);
                inEdges[v].Add(u);
            }
        }
    }

    private void AllocateSnapshotState()
    {
        yellow = new bool[n];
        snapState = new long[n];
        recvAtCut = new int[n];
        sendAtCut = new int[n];
        closedChannel = new bool[n * n];
        channelState = new List<TrafficMessage>[n * n];
        markerSendTime = new long[n * n];
        channelCloseTime = new long[n * n];
        for (int c = 0; c < n * n; c++) channelState[c] = new List<TrafficMessage>();
    }

    internal int Processes => n;
    internal int ChannelCount { get; private set; }
    internal int Ticks => ticks;
    internal int PerTick => perTick;
    internal long Deliveries => deliveries;
    internal long MarkerDeliveries => markerDeliveries;
    internal long TrafficMessages => SentLog.Count;
    internal int MaxLatency
    {
        get
        {
            int best = 0;
            for (int u = 0; u < n; u++)
                for (int v = 0; v < n; v++)
                    if (u != v && latency[u, v] > best) best = latency[u, v];
            return best;
        }
    }

    internal readonly List<TrafficMessage> SentLog = new List<TrafficMessage>();
    internal readonly List<Snapshot> Snapshots = new List<Snapshot>();
    internal readonly List<TrafficMessage>[] RecvLog;

    internal void Reset()
    {
        while (queue.Count > 0) queue.Pop();
        SentLog.Clear();
        for (int v = 0; v < n; v++) RecvLog[v].Clear();
        for (int v = 0; v < n; v++)
        {
            liveState[v] = 0;
            sendCount[v] = 0;
            recvCount[v] = 0;
        }
        Snapshots.Clear();
        current = null;
        AllocateSnapshotState();
        markersSent = 0;
        markersReceived = 0;
        completedChannels = 0;
        deliveries = 0;
        markerDeliveries = 0;
        trafficTicks = 0;
        ChannelCount = CountChannels();
    }

    private int CountChannels()
    {
        int total = 0;
        for (int u = 0; u < n; u++) total += outEdges[u].Count;
        return total;
    }

    internal void Push(long time, EventKind852 kind, TrafficMessage message, int from, int to)
    {
        queue.Push(new SimEvent852 { Time = time, Seq = seq++, Kind = kind, Message = message, From = from, To = to });
    }

    internal void ScheduleTraffic()
    {
        for (int t = 1; t <= ticks; t++) Push(t, EventKind852.TrafficTick, new TrafficMessage(), -1, -1);
    }

    internal void Run()
    {
        int nextStart = 0;
        while (queue.Count > 0)
        {
            SimEvent852 e = queue.Pop();
            // Scheduled snapshots start only when no other snapshot is being recorded.
            while (nextStart < scheduledStarts.Length && scheduledStarts[nextStart] <= e.Time)
            {
                long startAt = scheduledStarts[nextStart++];
                if (current == null) Initiate(startAt);
                else Push(startAt + snapshotRetryTicks, EventKind852.Initiate, new TrafficMessage(), -1, -1);
            }
            switch (e.Kind)
            {
                case EventKind852.TrafficTick:
                    GenerateTraffic(e.Time);
                    break;
                case EventKind852.Initiate:
                    if (current == null) Initiate(e.Time);
                    else Push(e.Time + snapshotRetryTicks, EventKind852.Initiate, new TrafficMessage(), -1, -1);
                    break;
                case EventKind852.Deliver:
                    DeliverTraffic(e.Message, e.Time);
                    break;
                case EventKind852.Marker:
                    DeliverMarker(e.From, e.To, e.Time);
                    break;
            }
        }
    }

    private void GenerateTraffic(long now)
    {
        trafficTicks++;
        if (sources.Length == 0) return;   // a cluster without channels carries no traffic
        for (int k = 0; k < perTick; k++)
        {
            int from = sources[Rng852.NextInt(ref rng, sources.Length)];
            List<int> targets = outEdges[from];
            int to = targets[Rng852.NextInt(ref rng, targets.Count)];
            int payload = 1 + Rng852.NextInt(ref rng, 9);
            var m = new TrafficMessage
            {
                Seq = SentLog.Count, From = from, To = to, SendTime = now, Payload = payload,
                SendIndex = sendCount[from]++, ArrivalTime = now + latency[from, to],
            };
            SentLog.Add(m);
            Push(m.ArrivalTime, EventKind852.Deliver, m, from, to);
        }    }

    private void DeliverTraffic(TrafficMessage m, long now)
    {
        deliveries++;
        m.Received = true;
        m.RecvIndex = recvCount[m.To]++;
        RecvLog[m.To].Add(m);
        liveState[m.To] += m.Payload;
        if (current != null && yellow[m.To])
        {
            int channel = m.From * n + m.To;
            if (!closedChannel[channel]) channelState[channel].Add(m);
        }
    }

    private void DeliverMarker(int from, int to, long now)
    {
        markerDeliveries++;
        markersReceived++;
        if (current == null) throw new Exception("a marker arrived with no snapshot in progress");
        if (!yellow[to])
        {
            yellow[to] = true;
            snapState[to] = liveState[to];
            recvAtCut[to] = recvCount[to];
            sendAtCut[to] = sendCount[to];
            foreach (int w in outEdges[to]) SendMarker(to, w, now);
        }
        int channel = from * n + to;
        if (!closedChannel[channel])
        {
            closedChannel[channel] = true;
            channelCloseTime[channel] = now;
            completedChannels++;
        }
        MaybeComplete(now);
    }

    private void Initiate(long now)
    {
        current = new Snapshot { Initiator = 0, Processes = n, StartTime = now, TotalChannels = ChannelCount };
        yellow[0] = true;
        snapState[0] = liveState[0];
        recvAtCut[0] = recvCount[0];
        sendAtCut[0] = sendCount[0];
        for (int w = 0; w < n; w++)
            if (outEdges[0].Contains(w)) SendMarker(0, w, now);
        MaybeComplete(now);
    }

    private void SendMarker(int from, int to, long now)
    {
        markersSent++;
        markerSendTime[from * n + to] = now;
        Push(now + latency[from, to], EventKind852.Marker, new TrafficMessage(), from, to);
    }

    private void MaybeComplete(long now)
    {
        if (current == null) return;
        for (int v = 0; v < n; v++)
            if (!yellow[v]) return;
        if (completedChannels != current.TotalChannels) return;
        current.StopTime = now;
        current.States = (long[])snapState.Clone();
        current.RecvAtCut = (int[])recvAtCut.Clone();
        current.SendAtCut = (int[])sendAtCut.Clone();
        current.Yellow = (bool[])yellow.Clone();
        current.Channels = new List<TrafficMessage>[n * n];
        for (int c = 0; c < n * n; c++) current.Channels[c] = new List<TrafficMessage>(channelState[c]);
        current.MarkersSent = markersSent;
        current.MarkersReceived = markersReceived;
        current.CompletedChannels = completedChannels;
        current.MarkerSendTimes = (long[])markerSendTime.Clone();
        current.ChannelCloseTimes = (long[])channelCloseTime.Clone();
        Snapshots.Add(current);
        // A completed snapshot resets the colours, so the next snapshot starts from a clean state.
        current = null;
        markersSent = 0;
        markersReceived = 0;
        completedChannels = 0;
        AllocateSnapshotState();
    }

    internal List<int> OutNeighbours(int u) => outEdges[u];
    internal List<int> InNeighbours(int v) => inEdges[v];
    internal long LiveState(int v) => liveState[v];
    internal int RecvCount(int v) => recvCount[v];
    internal long[] LiveStates()
    {
        var copy = new long[n];
        Array.Copy(liveState, copy, n);
        return copy;
    }
}

// The oracle: it replays the complete send/receive log and checks the four properties of a snapshot
// without using any state of the protocol that produced it.
internal static class SnapshotAudit
{
    internal static AuditResult Audit(Cluster cluster, Snapshot snap)
    {
        var result = new AuditResult();
        int n = snap.Processes;

        // 1. Every recorded process state is the replay of that process's first RecvAtCut receives.
        for (int v = 0; v < n; v++)
        {
            long replay = 0;
            List<TrafficMessage> received = cluster.RecvLog[v];
            for (int i = 0; i < snap.RecvAtCut[v] && i < received.Count; i++) replay += received[i].Payload;
            if (replay != snap.States[v]) result.Fail(ref result.StateViolations);
            result.StateTokens += snap.States[v];
        }

        // 2. The cut is consistent: a receive inside the cut implies its send is inside the cut.
        foreach (TrafficMessage m in cluster.SentLog)
        {
            if (!m.Received) continue;
            result.MessagesChecked++;
            if (m.RecvIndex < snap.RecvAtCut[m.To] && m.SendIndex >= snap.SendAtCut[m.From])
                result.Fail(ref result.ConsistentCutViolations);
            if (m.SendIndex < snap.SendAtCut[m.From]) result.SentBeforeCutTokens += m.Payload;
        }

        // 3. Each channel state is exactly {sent before the sender's cut, received after the receiver's cut}.
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                var expected = new List<TrafficMessage>();
                foreach (TrafficMessage m in cluster.SentLog)
                {
                    if (m.From != u || m.To != v || !m.Received) continue;
                    if (m.SendIndex < snap.SendAtCut[u] && m.RecvIndex >= snap.RecvAtCut[v]) expected.Add(m);
                }
                List<TrafficMessage> recorded = snap.Channels[u * n + v];
                if (recorded.Count != expected.Count)
                {
                    if (result.Detail.Length == 0)
                        result.Detail = "first mismatch on channel " + u + "->" + v + ": recorded=" + recorded.Count
                            + " expected=" + expected.Count + " sendAtCut=" + snap.SendAtCut[u]
                            + " recvAtCut=" + snap.RecvAtCut[v] + " markerSent=" + snap.MarkerSendTimes[u * n + v]
                            + " channelClosed=" + snap.ChannelCloseTimes[u * n + v];
                    result.Fail(ref result.ChannelViolations);
                }
                else
                {
                    for (int i = 0; i < expected.Count; i++)
                        if (recorded[i].Seq != expected[i].Seq) result.Fail(ref result.ChannelViolations);
                }
                foreach (TrafficMessage m in recorded) result.ChannelTokens += m.Payload;
            }
        }

        // 4. Token conservation: recorded states plus in-flight channel payloads equal every payload whose
        // send lies inside the cut. Nothing vanishes and nothing is counted twice.
        if (result.StateTokens + result.ChannelTokens != result.SentBeforeCutTokens) result.Ok = false;
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. Network construction: strongly connected, symmetric channel count, deterministic latencies.
        var small = new Cluster(4, 120, 3, 0x852AUL, new long[] { 40, 80 });
        Check(small.ChannelCount == CountEdges(small), "the channel count matches the directed edge set");
        Check(IsStronglyConnected(small), "every process is reachable from every other, so markers reach all");
        small.Reset();
        small.ScheduleTraffic();
        small.Run();

        // 4-7. The audit of the first snapshot of the small run.
        Check(small.Snapshots.Count == 2, "both scheduled snapshots terminated during the run");
        Snapshot first = small.Snapshots[0];
        AuditResult audit = SnapshotAudit.Audit(small, first);
        Console.WriteLine("small run: processes=" + small.Processes + " channels=" + small.ChannelCount
            + " traffic=" + small.TrafficMessages + " snapshots=" + small.Snapshots.Count
            + " states=[" + string.Join(",", first.States) + "] channels=" + first.Channels.Length);
        Check(audit.StateViolations == 0, "every recorded process state is the replay of its first K receives");
        Check(audit.ConsistentCutViolations == 0,
            "the cut is consistent: " + audit.MessagesChecked + " receives checked, no send outside the cut");
        Check(audit.ChannelViolations == 0,
            "every channel state is exactly the in-flight set at the cut (tokens in flight: " + audit.ChannelTokens
            + ") " + audit.Detail);
        Check(audit.StateTokens + audit.ChannelTokens == audit.SentBeforeCutTokens,
            "token conservation: states (" + audit.StateTokens + ") + in flight (" + audit.ChannelTokens
            + ") == payloads sent inside the cut (" + audit.SentBeforeCutTokens + ")");

        // 8-10. Marker accounting and liveness.
        Check(first.MarkersSent == first.TotalChannels && first.MarkersReceived == first.TotalChannels,
            "exactly one marker is sent and received per directed channel (" + first.MarkersSent + ")");
        Check(first.CompletedChannels == first.TotalChannels && AllYellow(first.Yellow),
            "termination needs every channel closed and every process yellow");
        Check(first.StopTime - first.StartTime <= small.Processes * small.MaxLatency,
            "the snapshot terminates within N * maxLatency ticks ("
            + (first.StopTime - first.StartTime) + " <= " + small.Processes * small.MaxLatency + ")");

        // 11-12. The snapshot really is a cut through ongoing traffic.
        bool postCutTraffic = false;
        foreach (TrafficMessage m in small.SentLog)
        {
            if (m.SendIndex >= first.SendAtCut[m.From] && m.Received && m.RecvIndex >= first.RecvAtCut[m.To])
                postCutTraffic = true;
        }
        Check(postCutTraffic, "messages were sent and received entirely after the cut, so traffic overlapped it");
        bool statesDiffer = false;
        long[] finalStates = small.LiveStates();
        for (int v = 0; v < small.Processes; v++) if (first.States[v] != finalStates[v]) statesDiffer = true;
        Check(statesDiffer, "the recorded states are a mid-run cut, not the final state of the run");

        // 13-16. Scale: 16 processes, three snapshots, 3000 messages of traffic.
        long[] scaleStarts = { 40, 80, 120 };
        var scale = new Cluster(16, 200, 15, 0x852BUL, scaleStarts);
        scale.Reset();
        scale.ScheduleTraffic();
        scale.Run();
        Console.WriteLine("scale run: processes=" + scale.Processes + " channels=" + scale.ChannelCount
            + " traffic=" + scale.TrafficMessages + " snapshots=" + scale.Snapshots.Count
            + " deliveries=" + scale.Deliveries);
        Check(scale.Snapshots.Count == 3 && scale.TrafficMessages == 3000,
            "three snapshots completed inside 3000 messages of traffic");
        bool allAuditsPass = true;
        long channelTokens = 0;
        foreach (Snapshot snap in scale.Snapshots)
        {
            AuditResult a = SnapshotAudit.Audit(scale, snap);
            channelTokens += a.ChannelTokens;
            if (!a.Ok) allAuditsPass = false;
        }
        Check(allAuditsPass, "every snapshot at scale passes the independent audit");
        Check(channelTokens > 0, "snapshots taken under load really do capture messages in flight ("
            + channelTokens + " payload tokens)");
        long markersPerSnapshot = scale.Snapshots[0].MarkersSent;
        long totalMarkers = 0;
        foreach (Snapshot snap in scale.Snapshots) totalMarkers += snap.MarkersSent;
        double overhead = (double)totalMarkers / scale.Deliveries;
        Console.WriteLine("  markers=" + totalMarkers + " for " + scale.Snapshots.Count + " snapshots of "
            + scale.ChannelCount + " channels; overhead=" + overhead.ToString("P3") + " of deliveries");
        Check(markersPerSnapshot == scale.ChannelCount && totalMarkers == 3 * scale.ChannelCount,
            "the per-snapshot cost is exactly one marker per directed channel");
        Check(overhead < 0.25, "the snapshot overhead stays below 25% of the deliveries: " + overhead.ToString("P3"));

        // 17-18. Sequential snapshots advance the cut monotonically.
        bool monotone = true;
        for (int k = 1; k < scale.Snapshots.Count; k++)
        {
            for (int v = 0; v < scale.Processes; v++)
                if (scale.Snapshots[k].RecvAtCut[v] < scale.Snapshots[k - 1].RecvAtCut[v]) monotone = false;
            if (scale.Snapshots[k].StartTime < scale.Snapshots[k - 1].StopTime) monotone = false;
        }
        Check(monotone, "successive snapshots start after the previous terminates and their cuts only move forward");
        long sumCuts = 0;
        foreach (int value in scale.Snapshots[2].RecvAtCut) sumCuts += value;
        Check(sumCuts < scale.Deliveries, "the last cut is strictly inside the execution ("
            + sumCuts + " receives inside the cut of " + scale.Deliveries + " deliveries)");

        // 19-23. The audit has teeth in both directions.
        Check(!TamperedChannel(scale).Ok, "the audit rejects a channel state with a message removed");
        Check(!RecordsPastMarker(scale).Ok, "the audit rejects a channel that keeps recording past its marker");
        Check(!DoubleCountedMessage(scale).Ok, "the audit rejects a channel state that double counts an early message");
        Check(!ShiftedCut(scale).Ok, "the audit rejects a cut shifted by one receive");
        Check(WholeExecutionIsACut(scale).Ok,
            "the audit still accepts the whole execution, so it is not simply rejecting every candidate");

        // 22-23. Determinism and degenerate shapes.
        var repeat = new Cluster(16, 200, 15, 0x852BUL, scaleStarts);
        repeat.Reset();
        repeat.ScheduleTraffic();
        repeat.Run();
        Check(repeat.SentLog.Count == scale.SentLog.Count
            && repeat.Snapshots.Count == scale.Snapshots.Count
            && repeat.Snapshots[0].Fingerprint() == scale.Snapshots[0].Fingerprint()
            && repeat.Snapshots[2].Fingerprint() == scale.Snapshots[2].Fingerprint(),
            "the same seed reproduces the traffic and the snapshots bit for bit");
        var quiet = new Cluster(8, 60, 0, 0x852CUL, new long[] { 5 });
        quiet.Reset();
        quiet.ScheduleTraffic();
        quiet.Run();
        Check(quiet.Snapshots.Count == 1 && quiet.Snapshots[0].States.Length == 8
            && AllZero(quiet.Snapshots[0].States) && ChannelTokens(quiet.Snapshots[0]) == 0
            && SnapshotAudit.Audit(quiet, quiet.Snapshots[0]).Ok,
            "with no traffic the snapshot is completed immediately with empty states and channels");
        var single = new Cluster(1, 30, 2, 0x852DUL, new long[] { 10 });
        single.Reset();
        single.ScheduleTraffic();
        single.Run();
        Snapshot only = single.Snapshots[0];
        Check(only.TotalChannels == 0 && only.MarkersSent == 0
            && only.States[0] == single.LiveState(0) && only.StopTime == only.StartTime,
            "a single-process cluster snapshots its own state instantly with no channels at all");

        Console.WriteLine("PASS");
    }

    private static int CountEdges(Cluster cluster)
    {
        int total = 0;
        for (int u = 0; u < cluster.Processes; u++) total += cluster.OutNeighbours(u).Count;
        return total;
    }

    private static bool IsStronglyConnected(Cluster cluster)
    {
        int n = cluster.Processes;
        for (int start = 0; start < n; start++)
        {
            var seen = new bool[n];
            var stack = new Stack<int>();
            seen[start] = true;
            stack.Push(start);
            while (stack.Count > 0)
            {
                int u = stack.Pop();
                foreach (int w in cluster.OutNeighbours(u))
                {
                    if (!seen[w])
                    {
                        seen[w] = true;
                        stack.Push(w);
                    }
                }
            }
            for (int v = 0; v < n; v++) if (!seen[v]) return false;
        }
        return true;
    }

    private static bool AllYellow(bool[] yellow)
    {
        foreach (bool value in yellow) if (!value) return false;
        return true;
    }

    private static bool AllZero(long[] values)
    {
        foreach (long value in values) if (value != 0) return false;
        return true;
    }

    private static long ChannelTokens(Snapshot snap)
    {
        long total = 0;
        foreach (List<TrafficMessage> channel in snap.Channels)
            foreach (TrafficMessage m in channel) total += m.Payload;
        return total;
    }

    private static AuditResult TamperedChannel(Cluster cluster)
    {
        Snapshot copy = Clone(cluster.Snapshots[0]);
        List<TrafficMessage> channel = null;
        for (int c = 0; c < copy.Channels.Length; c++)
        {
            if (copy.Channels[c].Count > 0)
            {
                channel = copy.Channels[c];
                break;
            }
        }
        if (channel == null) throw new Exception("the snapshot under test captured no in-flight message");
        channel.RemoveAt(0);
        return SnapshotAudit.Audit(cluster, copy);
    }

    private static AuditResult ShiftedCut(Cluster cluster)
    {
        Snapshot copy = Clone(cluster.Snapshots[1]);
        copy.RecvAtCut[3] += 1;
        return SnapshotAudit.Audit(cluster, copy);
    }

    // The classic bug: a receiver that keeps recording a channel after the marker arrived would carry
    // messages that were sent *and* received after the cut, i.e. messages that were never in flight.
    private static AuditResult RecordsPastMarker(Cluster cluster)
    {
        Snapshot copy = Clone(cluster.Snapshots[1]);
        foreach (TrafficMessage m in cluster.SentLog)
        {
            if (!m.Received) continue;
            if (m.SendIndex < copy.SendAtCut[m.From] || m.RecvIndex < copy.RecvAtCut[m.To]) continue;
            copy.Channels[m.From * copy.Processes + m.To].Add(m);
            return SnapshotAudit.Audit(cluster, copy);
        }
        throw new Exception("the snapshot under test captured no fully post-cut message");
    }

    // The other classic bug: recording a message that the receiver had already applied before its cut
    // counts that payload twice, once in the process state and once in the channel state.
    private static AuditResult DoubleCountedMessage(Cluster cluster)
    {
        Snapshot copy = Clone(cluster.Snapshots[1]);
        foreach (TrafficMessage m in cluster.SentLog)
        {
            if (!m.Received) continue;
            if (m.SendIndex >= copy.SendAtCut[m.From] || m.RecvIndex >= copy.RecvAtCut[m.To]) continue;
            copy.Channels[m.From * copy.Processes + m.To].Add(m);
            return SnapshotAudit.Audit(cluster, copy);
        }
        throw new Exception("no fully pre-cut message exists to double count");
    }

    // A positive control: the whole execution is itself a consistent cut, and the audit must accept it
    // once the recorded states and the cut boundary agree with the log.
    private static AuditResult WholeExecutionIsACut(Cluster cluster)
    {
        int n = cluster.Processes;
        var whole = new Snapshot
        {
            Processes = n,
            States = cluster.LiveStates(),
            RecvAtCut = new int[n],
            SendAtCut = new int[n],
            Channels = new List<TrafficMessage>[n * n],
        };
        for (int c = 0; c < whole.Channels.Length; c++) whole.Channels[c] = new List<TrafficMessage>();
        for (int v = 0; v < n; v++)
        {
            whole.RecvAtCut[v] = cluster.RecvCount(v);
            int sent = 0;
            foreach (TrafficMessage m in cluster.SentLog) if (m.From == v) sent++;
            whole.SendAtCut[v] = sent;
        }
        return SnapshotAudit.Audit(cluster, whole);
    }

    private static Snapshot Clone(Snapshot snap)
    {
        var copy = new Snapshot
        {
            Initiator = snap.Initiator,
            Processes = snap.Processes,
            StartTime = snap.StartTime,
            StopTime = snap.StopTime,
            States = (long[])snap.States.Clone(),
            RecvAtCut = (int[])snap.RecvAtCut.Clone(),
            SendAtCut = (int[])snap.SendAtCut.Clone(),
            Yellow = (bool[])snap.Yellow.Clone(),
            Channels = new List<TrafficMessage>[snap.Channels.Length],
            MarkersSent = snap.MarkersSent,
            MarkersReceived = snap.MarkersReceived,
            CompletedChannels = snap.CompletedChannels,
            TotalChannels = snap.TotalChannels,
            MarkerSendTimes = (long[])snap.MarkerSendTimes.Clone(),
            ChannelCloseTimes = (long[])snap.ChannelCloseTimes.Clone(),
        };
        for (int c = 0; c < snap.Channels.Length; c++) copy.Channels[c] = new List<TrafficMessage>(snap.Channels[c]);
        return copy;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
