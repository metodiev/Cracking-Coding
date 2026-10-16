// Task 760 - Implement distributed snapshot-based auditing for integrity. (Tier: Expert, Ch 9 - System Design and Scalability)
//
// Chandy-Lamport snapshots over four token-transferring nodes, plus the audit that decides whether the recorded
// cut is *consistent* and *complete*. A node that is asked to snapshot records its own state, starts recording
// every incoming channel, and passes a marker on every outgoing channel. A node that receives its first marker
// records its state, marks that channel empty (FIFO channels guarantee nothing else from that channel can be in
// flight), starts recording its other incoming channels and forwards markers on its other outgoing channels.
// Data messages that arrive on a channel *while* it is being recorded are logged as that channel's state, which
// is exactly the set of transfers that were sent before the sender's snapshot and received after this node's.
//
// The audit verifies the classic invariants with global knowledge of the run: (a) token conservation - the
// recorded balances plus the recorded in-transit tokens equal the total number of tokens in the system; (b)
// causal closure - a message that was received *and* included in a recorded state must have been sent before the
// sender recorded its own state, and a message recorded in transit must have been sent before the sender's state
// and not yet received when this node recorded its own; (c) completeness - every node and every channel has a
// recorded state; (d) an independent recomputation of every channel state straight from the message log, which
// must agree with what the algorithm recorded; (e) authenticity - each part is digested and the digest chain
// detects any tampering, and the audit also runs on a deliberately inconsistent cut to prove it has teeth.
//
// Documented simplifications: four nodes in one process, a deterministic clock instead of real time, no
// retransmission or partitioning, no durable storage and a hand written FNV digest rather than a collision
// resistant hash. Invariant verified in addition: after every simulation step the live total (balances plus
// in-flight messages) is unchanged, so the conservation check cannot be satisfied by coincidence.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Digest64
{
    private const ulong Offset = 14695981039346656037UL;
    private const ulong Prime = 1099511628211UL;

    internal static string Of(string text)
    {
        ulong hash = Offset;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= Prime;
        }
        return hash.ToString("X16");
    }
}

internal sealed class Node
{
    internal readonly int Id;
    internal long Balance;
    internal long LocalIndex;                       // number of events this node has processed
    internal bool Recorded;
    internal long RecordedAtLocalIndex = -1;
    internal long RecordedBalance = -1;

    internal readonly HashSet<int> RecordingChannels = new HashSet<int>();
    internal readonly Dictionary<int, long> ChannelTotals = new Dictionary<int, long>();
    internal readonly Dictionary<int, long> MarkerAtLocalIndex = new Dictionary<int, long>();

    internal Node(int id, long balance)
    {
        Id = id;
        Balance = balance;
    }

    /** Records this node's state and starts recording every incoming channel. */
    internal void RecordState(int nodeCount)
    {
        Recorded = true;
        RecordedAtLocalIndex = LocalIndex;
        RecordedBalance = Balance;
        RecordingChannels.Clear();
        for (int i = 0; i < nodeCount; i++)
        {
            if (i != Id) RecordingChannels.Add(i);
        }
    }
}

internal sealed class Message
{
    internal int From;
    internal int To;
    internal long Amount;
    internal bool IsMarker;
    internal long DeliverAtClock;
    internal long Order;
    internal long SentAtLocal;
    internal long ReceivedAtLocal = -1;
}

internal sealed class SnapshotEngine
{
    internal readonly Node[] Nodes;
    private readonly List<Message> pending = new List<Message>();
    internal readonly List<Message> Log = new List<Message>();
    private long order;
    private long clock;

    internal long MarkersSent;
    internal long MarkersDelivered;
    internal long Transfers;

    internal SnapshotEngine(int nodeCount, long balanceEach)
    {
        Nodes = new Node[nodeCount];
        for (int i = 0; i < nodeCount; i++) Nodes[i] = new Node(i, balanceEach);
    }

    internal long TotalTokens()
    {
        long total = 0;
        foreach (Node node in Nodes) total += node.Balance;
        return total;
    }

    internal long InFlight()
    {
        long total = 0;
        foreach (Message message in pending)
        {
            if (!message.IsMarker) total += message.Amount;
        }
        return total;
    }

    internal void Transfer(int from, int to, long amount)
    {
        Transfers++;
        Node sender = Nodes[from];
        var message = new Message
        {
            From = from,
            To = to,
            Amount = amount,
            DeliverAtClock = clock + ChannelDelay(from, to),      // constant per channel, so FIFO holds
            Order = ++order,
            SentAtLocal = sender.LocalIndex,
        };
        sender.LocalIndex++;                                   // sending is a local event as well
        pending.Add(message);
        Log.Add(message);
        sender.Balance -= amount;
    }

    internal void Initiate(int initiator)
    {
        Nodes[initiator].RecordState(Nodes.Length);
        foreach (int target in Others(initiator)) SendMarker(initiator, target);
    }

    private void SendMarker(int from, int to)
    {
        MarkersSent++;
        pending.Add(new Message
        {
            From = from,
            To = to,
            IsMarker = true,
            DeliverAtClock = clock + ChannelDelay(from, to),      // markers share the channel's FIFO ordering
            Order = ++order,
            SentAtLocal = Nodes[from].LocalIndex,
        });
        Nodes[from].LocalIndex++;                              // forwarding a marker is a local event as well
    }

    /** A constant delay per channel keeps the delivery order equal to the send order (FIFO), as the algorithm needs. */
    private static long ChannelDelay(int from, int to) => 2 + (from + to) % 3;

    /** Delivers every message whose delivery time already arrived, in (time, insertion) order. */
    internal int DeliverDue()
    {
        int delivered = 0;
        while (true)
        {
            Message? next = Earliest();
            if (next == null) break;
            pending.Remove(next);
            clock = next.DeliverAtClock;
            Node node = Nodes[next.To];
            if (next.IsMarker) DeliverMarker(node, next);
            else DeliverData(node, next);
            delivered++;
        }
        return delivered;
    }

    private void DeliverData(Node node, Message message)
    {
        message.ReceivedAtLocal = node.LocalIndex;
        node.LocalIndex++;
        if (node.RecordingChannels.Contains(message.From))
        {
            node.ChannelTotals.TryGetValue(message.From, out long recorded);
            node.ChannelTotals[message.From] = recorded + message.Amount;
        }
        node.Balance += message.Amount;
    }

    private void DeliverMarker(Node node, Message marker)
    {
        MarkersDelivered++;
        long markerIndex = node.LocalIndex;
        node.LocalIndex++;
        if (!node.Recorded)
        {
            node.RecordState(Nodes.Length);
            node.RecordingChannels.Remove(marker.From);      // FIFO: nothing in flight on this channel can be missed
            node.ChannelTotals[marker.From] = 0;
            node.MarkerAtLocalIndex[marker.From] = node.RecordedAtLocalIndex;
            foreach (int target in Others(node.Id))
            {
                SendMarker(node.Id, target);           // every outgoing channel, including the one we were told from
            }
            return;
        }
        if (node.RecordingChannels.Remove(marker.From))
        {
            node.MarkerAtLocalIndex[marker.From] = markerIndex;      // the channel's window ends at this event
        }
    }

    private Message? Earliest()
    {
        if (pending.Count == 0) return null;
        Message best = pending[0];
        foreach (Message candidate in pending)
        {
            if (candidate.DeliverAtClock < best.DeliverAtClock
                || (candidate.DeliverAtClock == best.DeliverAtClock && candidate.Order < best.Order))
            {
                best = candidate;
            }
        }
        return best;
    }

    internal bool SnapshotComplete()
    {
        foreach (Node node in Nodes)
        {
            if (!node.Recorded) return false;
            if (node.MarkerAtLocalIndex.Count != Nodes.Length - 1) return false;
        }
        return true;
    }

    internal IEnumerable<int> Others(int self)
    {
        for (int i = 0; i < Nodes.Length; i++)
        {
            if (i != self) yield return i;
        }
    }
}

internal sealed class SnapshotPart
{
    internal int NodeId;
    internal long Balance;
    internal SortedDictionary<int, long> Incoming = new SortedDictionary<int, long>();
    internal string Digest = "";

    internal string Text()
    {
        var builder = new StringBuilder();
        builder.Append("node=").Append(NodeId).Append(";balance=").Append(Balance).Append(";intransit=");
        foreach (KeyValuePair<int, long> entry in Incoming)
        {
            builder.Append(entry.Key).Append(':').Append(entry.Value).Append(',');
        }
        return builder.ToString();
    }
}

internal sealed class DistributedSnapshot
{
    internal readonly List<SnapshotPart> Parts = new List<SnapshotPart>();
    internal string Root = "";

    internal SnapshotPart Part(int nodeId) => Parts[nodeId];

    internal void Seal()
    {
        var builder = new StringBuilder();
        foreach (SnapshotPart part in Parts)
        {
            part.Digest = Digest64.Of(part.Text());
            builder.Append(part.Digest);
        }
        Root = Digest64.Of(builder.ToString());
    }
}

internal static class Snapshot
{
    internal static DistributedSnapshot Capture(SnapshotEngine engine)
    {
        var snapshot = new DistributedSnapshot();
        foreach (Node node in engine.Nodes)
        {
            var part = new SnapshotPart { NodeId = node.Id, Balance = node.RecordedBalance };
            for (int source = 0; source < engine.Nodes.Length; source++)
            {
                if (source == node.Id) continue;
                part.Incoming[source] = node.ChannelTotals.TryGetValue(source, out long value) ? value : 0;
            }
            snapshot.Parts.Add(part);
        }
        snapshot.Seal();
        return snapshot;
    }
}

internal sealed class AuditReport
{
    internal int DigestFailures;
    internal int ConsistencyViolations;
    internal int ChannelMismatches;
    internal int InTransitViolations;
    internal long RecordedTokens;
    internal long ExpectedTokens;
    internal string FirstProblem = "";
}

internal static class Auditor
{
    internal static AuditReport Audit(DistributedSnapshot snapshot, SnapshotEngine engine, long expectedTokens)
    {
        var report = new AuditReport { ExpectedTokens = expectedTokens };
        var recalculated = new DistributedSnapshot();
        recalculated.Parts.AddRange(snapshot.Parts);
        foreach (SnapshotPart part in recalculated.Parts)
        {
            if (Digest64.Of(part.Text()) != part.Digest) report.DigestFailures++;
        }

        long total = 0;
        foreach (SnapshotPart part in snapshot.Parts)
        {
            total += part.Balance;
            foreach (KeyValuePair<int, long> entry in part.Incoming) total += entry.Value;
        }
        report.RecordedTokens = total;

        // Causal closure over the message log, using each node's recorded event index.
        foreach (Message message in engine.Log)
        {
            if (message.IsMarker || message.ReceivedAtLocal < 0) continue;
            long recordedReceiver = engine.Nodes[message.To].RecordedAtLocalIndex;
            long recordedSender = engine.Nodes[message.From].RecordedAtLocalIndex;
            bool inReceiverState = message.ReceivedAtLocal < recordedReceiver;
            bool inSenderState = message.SentAtLocal < recordedSender;
            if (inReceiverState && !inSenderState)
            {
                report.ConsistencyViolations++;
                if (report.FirstProblem.Length == 0)
                {
                    report.FirstProblem = "message " + message.From + "->" + message.To + " was received at local index "
                        + message.ReceivedAtLocal + " (recorded " + recordedReceiver + ") but only sent at local index "
                        + message.SentAtLocal + " (recorded " + recordedSender + ")";
                }
            }
        }

        // Independent recomputation of every channel state from the log.
        foreach (Node node in engine.Nodes)
        {
            foreach (int source in engine.Others(node.Id))
            {
                if (!node.MarkerAtLocalIndex.TryGetValue(source, out long end)) continue;
                long window = 0;
                foreach (Message message in engine.Log)
                {
                    if (message.IsMarker || message.From != source || message.To != node.Id) continue;
                    if (message.ReceivedAtLocal >= node.RecordedAtLocalIndex && message.ReceivedAtLocal < end)
                    {
                        window += message.Amount;
                        if (message.SentAtLocal >= engine.Nodes[source].RecordedAtLocalIndex) report.InTransitViolations++;
                    }
                }
                snapshot.Part(node.Id).Incoming.TryGetValue(source, out long claimed);
                if (window != claimed)
                {
                    report.ChannelMismatches++;
                    if (report.FirstProblem.Length == 0)
                    {
                        report.FirstProblem = "channel " + source + "->" + node.Id + " recorded " + claimed
                            + " but the log yields " + window;
                    }
                }
            }
        }
        return report;
    }
}

public static class Solution
{
    private const int Nodes = 4;
    private const long StartBalance = 1000;

    public static void Main()
    {
        SnapshotEngine engine = Simulate(5, 60, 0, out long total, out int liveViolations);
        Check(liveViolations == 0, "tokens are conserved after every simulation step");
        DistributedSnapshot snapshot = Snapshot.Capture(engine);
        AuditReport report = Auditor.Audit(snapshot, engine, total);
        Check(report.DigestFailures == 0, "every recorded part matches its digest");
        long inTransitRecorded = 0;
        foreach (SnapshotPart part in snapshot.Parts)
        {
            foreach (KeyValuePair<int, long> entry in part.Incoming) inTransitRecorded += entry.Value;
        }
        Check(inTransitRecorded > 0,
            "the cut really caught tokens mid-flight: " + inTransitRecorded + " of " + total);
        Check(report.RecordedTokens == total,
            "the snapshot holds all " + total + " tokens (balances plus in-transit)");
        Check(report.ConsistencyViolations == 0, "the cut is causally closed: " + report.FirstProblem);
        Check(report.ChannelMismatches == 0, "every channel state equals the independent recomputation");
        Check(report.InTransitViolations == 0, "every recorded in-transit message was sent before the sender snapped");
        Check(engine.SnapshotComplete(), "all " + Nodes + " nodes and all " + Nodes * (Nodes - 1) + " channels recorded");
        Check(engine.MarkersDelivered == Nodes * (Nodes - 1),
            "one marker per channel carried the snapshot: " + engine.MarkersDelivered);
        Check(engine.Transfers >= 10, "the workload moved tokens " + engine.Transfers + " times");

        // The audit must reject a tampered snapshot: any edited field breaks the digest chain.
        DistributedSnapshot tampered = Snapshot.Capture(engine);
        tampered.Part(1).Balance += 7;
        Check(Auditor.Audit(tampered, engine, total).DigestFailures == 1, "an edited balance is caught by the digest");
        DistributedSnapshot tamperedChannel = Snapshot.Capture(engine);
        tamperedChannel.Part(2).Incoming[3] += 5;
        Check(Auditor.Audit(tamperedChannel, engine, total).DigestFailures == 1, "an edited channel state is caught too");
        tamperedChannel.Seal();                                        // what a verifier recomputes
        Check(tamperedChannel.Root != snapshot.Root,
            "re-sealing the edited snapshot yields a different root, so the tamper cannot be re-attested");

        // And it must have teeth: a hand built, causally inconsistent cut is reported as inconsistent.
        AuditReport inconsistent = InconsistentCut(engine, total);
        Check(inconsistent.ConsistencyViolations > 0,
            "an artificially inconsistent cut is rejected: " + inconsistent.FirstProblem);

        // Boundary: with the snapshot initiated before any transfer and no transfers in the window at all,
        // every channel is recorded as empty while conservation still holds.
        SnapshotEngine fresh = Simulate(0, 12, 20, out long freshTotal, out int freshViolations);
        DistributedSnapshot emptyCut = Snapshot.Capture(fresh);
        AuditReport freshReport = Auditor.Audit(emptyCut, fresh, freshTotal);
        long inTransit = 0;
        foreach (SnapshotPart part in emptyCut.Parts)
        {
            foreach (KeyValuePair<int, long> entry in part.Incoming) inTransit += entry.Value;
        }
        Check(freshViolations == 0 && freshReport.ConsistencyViolations == 0 && inTransit == 0,
            "a snapshot taken before any transfer records zero tokens in transit and stays consistent");
        Check(freshReport.RecordedTokens == total && freshReport.DigestFailures == 0,
            "that early snapshot still accounts for every token and carries valid digests");

        Console.WriteLine("  snapshot: transfers=" + engine.Transfers + " markers=" + engine.MarkersDelivered
            + " recorded tokens=" + report.RecordedTokens + "/" + total);
        Console.WriteLine("  channels recorded: " + Nodes * (Nodes - 1) + " (independently recomputed, all equal)");
        Console.WriteLine("PASS");
    }

    /** Deterministic workload: planned transfers (shifted in time) plus a snapshot initiated at `initiateAt`. */
    private static SnapshotEngine Simulate(int initiateAt, int stopAt, int transferShift, out long total, out int liveViolations)
    {
        var engine = new SnapshotEngine(Nodes, StartBalance);
        total = engine.TotalTokens();
        liveViolations = 0;
        var schedule = new (int Step, int From, int To, long Amount)[]
        {
            (1, 0, 1, 40), (2, 1, 2, 15), (4, 2, 3, 25), (5, 3, 0, 10), (7, 0, 2, 60),
            (9, 1, 3, 5), (11, 2, 0, 30), (13, 3, 1, 45), (15, 0, 3, 20), (17, 1, 0, 35),
        };

        for (int current = 0; current <= stopAt; current++)
        {
            if (current == initiateAt) engine.Initiate(0);
            foreach ((int plannedStep, int from, int to, long amount) in schedule)
            {
                if (plannedStep + transferShift == current) engine.Transfer(from, to, amount);
            }
            engine.DeliverDue();
            if (engine.TotalTokens() + engine.InFlight() != total) liveViolations++;
        }
        return engine;
    }

    /** A cut that takes node 0's state at the very beginning and everybody else's much later: not causally closed. */
    private static AuditReport InconsistentCut(SnapshotEngine engine, long total)
    {
        var cut = new SnapshotEngine(Nodes, StartBalance);
        foreach (Node node in engine.Nodes)
        {
            Node copy = cut.Nodes[node.Id];
            copy.Recorded = true;
            copy.RecordedAtLocalIndex = node.Id == 0 ? 0 : node.RecordedAtLocalIndex;
            copy.RecordedBalance = node.RecordedBalance;
            foreach (KeyValuePair<int, long> entry in node.ChannelTotals) copy.ChannelTotals[entry.Key] = entry.Value;
            foreach (KeyValuePair<int, long> entry in node.MarkerAtLocalIndex) copy.MarkerAtLocalIndex[entry.Key] = entry.Value;
        }
        foreach (Message message in engine.Log) cut.Log.Add(message);
        DistributedSnapshot broken = Snapshot.Capture(cut);
        return Auditor.Audit(broken, cut, total);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
