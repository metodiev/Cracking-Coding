// Task 616 - Implement distributed snapshot algorithm (Chandy-Lamport). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// The Chandy-Lamport global snapshot over a *simulated* distributed system. There are no threads or
// sockets: a process ring of four nodes with FIFO channels (each node sends to the next two nodes)
// is advanced by an explicit schedule of events - a local computation step (which appends one
// message to a channel), the delivery of the head of a channel, and the snapshot trigger at the
// initiator. Markers travel through the same queues as ordinary messages.
//
// The protocol is implemented exactly: the initiator records its own state and sends a marker on
// every outgoing channel; a node that has not recorded yet records its state on receiving the first
// marker, forwards markers on all its outgoing channels, and starts recording every incoming channel
// except the one the marker came in on; a node that has already recorded treats a marker on a
// channel as the close of that channel, whose recorded state is exactly the messages it delivered
// from that channel since it recorded. The channel the first marker arrived on is recorded as empty,
// which is the standard convention (FIFO guarantees the messages sent before the cut were already
// delivered, hence already inside the receiving node's state).
//
// Verified: the snapshot completes (every node recorded, every channel closed); each recorded channel
// state equals *exactly* the set of messages sent before the sender's cut and delivered after the
// receiver's cut, recomputed independently from the message log; the cut is consistent in the classic
// sense (no message is received inside the cut unless it was sent inside it); at least one message
// really is in flight at the cut, so the channel recording is exercised; two initiators snapshotting
// concurrently still yield a consistent cut; the whole run is reproducible; and when the FIFO
// assumption is deliberately broken (a marker is delivered ahead of an earlier message on the same
// channel) the check must report the inconsistency - so the verifier is not vacuous.

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

internal sealed class Message
{
    internal int Id;
    internal bool Marker;
    internal int Sender;
    internal int Receiver;
    internal int SentGlobal;
    internal int DeliveredGlobal = -1;
}

internal sealed class SnapshotNode
{
    internal readonly int Id;
    internal readonly List<int> Outgoing = new List<int>();
    internal int LocalSteps;
    internal bool Recorded;
    internal int CutGlobal = -1;
    internal int RecordedState;
    internal int ProgramCursor;
    internal readonly Dictionary<int, List<int>> ChannelState = new Dictionary<int, List<int>>();
    internal readonly HashSet<int> Closed = new HashSet<int>();

    internal SnapshotNode(int id) { Id = id; }
}

internal sealed class ChandyLamportSimulation
{
    private readonly SnapshotNode[] nodes;
    private readonly Dictionary<int, Queue<Message>> channels = new Dictionary<int, Queue<Message>>();
    private readonly List<Message> log = new List<Message>();
    private int globalClock;
    private int nextId = 1;

    internal readonly bool FifoHeld;
    internal readonly List<string> Events = new List<string>();

    internal ChandyLamportSimulation(int nodeCount, bool fifoHeld)
    {
        FifoHeld = fifoHeld;
        nodes = new SnapshotNode[nodeCount];
        for (int i = 0; i < nodeCount; i++)
        {
            nodes[i] = new SnapshotNode(i);
            nodes[i].Outgoing.Add((i + 1) % nodeCount);
            nodes[i].Outgoing.Add((i + 2) % nodeCount);
        }
        for (int from = 0; from < nodeCount; from++)
        {
            foreach (int to in nodes[from].Outgoing) channels[Key(from, to)] = new Queue<Message>();
        }
    }

    internal int NodeCount => nodes.Length;
    internal SnapshotNode Node(int id) => nodes[id];
    private static int Key(int from, int to) => from * 100 + to;

    internal bool HasChannel(int from, int to) => channels.ContainsKey(Key(from, to));

    internal bool Queued(int from, int to)
    {
        Queue<Message> queue = channels[Key(from, to)];
        return queue.Count > 0;
    }

    internal bool AnyQueued()
    {
        foreach (var queue in channels.Values)
        {
            if (queue.Count > 0) return true;
        }
        return false;
    }

    /** A local computation step: the node's state advances and its program sends one message. */
    internal void LocalStep(int node)
    {
        SnapshotNode self = nodes[node];
        self.LocalSteps++;
        int receiver = self.Outgoing[self.ProgramCursor % self.Outgoing.Count];
        self.ProgramCursor++;
        Enqueue(new Message
        {
            Id = nextId++,
            Sender = node,
            Receiver = receiver,
            Marker = false,
            SentGlobal = globalClock++,
        });
        Events.Add("node" + node + " local step -> " + receiver);
    }

    /** Deliver the head of a channel; with FIFO broken, a marker may overtake earlier messages. */
    internal void Deliver(int from, int to)
    {
        Queue<Message> queue = channels[Key(from, to)];
        if (queue.Count == 0) return;
        Message message = queue.Peek();
        if (!FifoHeld)
        {
            foreach (Message candidate in queue)
            {
                if (candidate.Marker)
                {
                    message = candidate;
                    break;
                }
            }
            if (!ReferenceEquals(message, queue.Peek()))
            {
                var reordered = new Queue<Message>();
                foreach (Message candidate in queue)
                {
                    if (!ReferenceEquals(candidate, message)) reordered.Enqueue(candidate);
                }
                channels[Key(from, to)] = reordered;
                Events.Add("FIFO VIOLATION on " + from + "->" + to);
            }
        }
        channels[Key(from, to)].Dequeue();
        message.DeliveredGlobal = globalClock++;
        Receive(nodes[to], message);
    }

    private void Enqueue(Message message)
    {
        log.Add(message);
        channels[Key(message.Sender, message.Receiver)].Enqueue(message);
    }

    private void SendMarker(SnapshotNode from, int to)
    {
        Enqueue(new Message
        {
            Id = nextId++,
            Sender = from.Id,
            Receiver = to,
            Marker = true,
            SentGlobal = globalClock++,
        });
    }

    /** The initiator: record the local state, then send a marker on every outgoing channel. */
    internal void TakeSnapshot(int initiator)
    {
        SnapshotNode self = nodes[initiator];
        if (!self.Recorded) Record(self, -1);
        foreach (int to in self.Outgoing) SendMarker(self, to);
        Events.Add("node" + initiator + " snapshot trigger");
    }

    private void Receive(SnapshotNode node, Message message)
    {
        if (!message.Marker)
        {
            if (node.Recorded && node.ChannelState.ContainsKey(message.Sender)
                && !node.Closed.Contains(message.Sender))
            {
                node.ChannelState[message.Sender].Add(message.Id);
            }
            Events.Add("node" + node.Id + " delivered message " + message.Id);
            return;
        }
        if (!node.Recorded)
        {
            Record(node, message.Sender);
            foreach (int to in node.Outgoing) SendMarker(node, to);
            Events.Add("node" + node.Id + " recorded on marker from " + message.Sender);
            return;
        }
        if (node.ChannelState.ContainsKey(message.Sender) && !node.Closed.Contains(message.Sender))
        {
            node.Closed.Add(message.Sender);
            Events.Add("node" + node.Id + " closed channel from " + message.Sender);
        }
    }

    /** Record the state and open a channel recording for every incoming channel but the trigger. */
    private void Record(SnapshotNode node, int triggerSender)
    {
        node.Recorded = true;
        node.CutGlobal = globalClock++;
        node.RecordedState = node.LocalSteps;
        for (int from = 0; from < nodes.Length; from++)
        {
            if (from == node.Id || !HasChannel(from, node.Id)) continue;
            node.ChannelState[from] = new List<int>();
            if (from == triggerSender) node.Closed.Add(from);
        }
    }

    /** Deliver everything that is still queued so every channel gets closed. */
    internal void Drain(int limit)
    {
        int guard = 0;
        while (AnyQueued() && guard++ < limit)
        {
            foreach (var entry in channels)
            {
                if (entry.Value.Count == 0) continue;
                Deliver(entry.Key / 100, entry.Key % 100);
                break;
            }
        }
    }

    internal bool Complete()
    {
        foreach (SnapshotNode node in nodes)
        {
            if (!node.Recorded) return false;
            for (int from = 0; from < nodes.Length; from++)
            {
                if (from == node.Id || !HasChannel(from, node.Id)) continue;
                if (!node.ChannelState.ContainsKey(from) || !node.Closed.Contains(from)) return false;
            }
        }
        return !AnyQueued();
    }

    /** Messages sent before the sender's cut and delivered after the receiver's cut. */
    internal List<int> ExpectedInFlight(int from, int to)
    {
        var expected = new List<int>();
        foreach (Message message in log)
        {
            if (message.Sender != from || message.Receiver != to || message.Marker) continue;
            if (message.SentGlobal >= nodes[from].CutGlobal) continue;
            if (message.DeliveredGlobal >= 0 && message.DeliveredGlobal <= nodes[to].CutGlobal) continue;
            expected.Add(message.Id);
        }
        expected.Sort();
        return expected;
    }

    internal List<int> RecordedChannel(int from, int to)
    {
        if (!nodes[to].ChannelState.ContainsKey(from)) return new List<int>();
        var recorded = new List<int>(nodes[to].ChannelState[from]);
        recorded.Sort();
        return recorded;
    }

    /** The recorded cut must equal the independently recomputed in-flight sets. */
    internal bool ChannelsConsistent(out string report)
    {
        report = "";
        foreach (var entry in channels)
        {
            int from = entry.Key / 100;
            int to = entry.Key % 100;
            List<int> expected = ExpectedInFlight(from, to);
            List<int> recorded = RecordedChannel(from, to);
            if (expected.Count != recorded.Count)
            {
                report = "channel " + from + "->" + to + " expected " + expected.Count + " recorded "
                    + recorded.Count;
                return false;
            }
            for (int i = 0; i < expected.Count; i++)
            {
                if (expected[i] != recorded[i])
                {
                    report = "channel " + from + "->" + to + " differs at " + i;
                    return false;
                }
            }
        }
        return true;
    }

    /** Classic consistency: no message may be received inside the cut unless it was sent inside it. */
    internal bool CutConsistent(out string report)
    {
        report = "";
        foreach (Message message in log)
        {
            if (message.Marker || message.DeliveredGlobal < 0) continue;
            bool receivedInCut = message.DeliveredGlobal < nodes[message.Receiver].CutGlobal;
            bool sentInCut = message.SentGlobal < nodes[message.Sender].CutGlobal;
            if (receivedInCut && !sentInCut)
            {
                report = "message " + message.Id + " received inside the cut but sent outside it";
                return false;
            }
        }
        return true;
    }

    internal int TotalRecordedChannelMessages()
    {
        int total = 0;
        foreach (SnapshotNode node in nodes)
        {
            foreach (List<int> state in node.ChannelState.Values) total += state.Count;
        }
        return total;
    }

    internal string Cut()
    {
        var text = new List<string>();
        foreach (SnapshotNode node in nodes) text.Add("S" + node.Id + "=" + node.RecordedState);
        foreach (var entry in channels)
        {
            text.Add("c" + entry.Key / 100 + "->" + entry.Key % 100 + ":"
                + string.Join("-", RecordedChannel(entry.Key / 100, entry.Key % 100)));
        }
        return string.Join(" ", text);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Advance the simulation with a deterministic event schedule and then drain all channels. */
    private static void RunSchedule(ChandyLamportSimulation sim, int seed, int steps, bool secondInitiator,
        int initiator)
    {
        ulong state = (ulong)seed * 0x2545F4914F6CDD1DUL + 7;
        bool triggered = false;
        bool secondTriggered = false;
        for (int step = 0; step < steps; step++)
        {
            ulong roll = SplitMix64.Next(ref state);
            if (!triggered && step >= 15)
            {
                sim.TakeSnapshot(initiator);
                triggered = true;
                continue;
            }
            if (secondInitiator && !secondTriggered && step >= 30)
            {
                sim.TakeSnapshot((initiator + 2) % sim.NodeCount);
                secondTriggered = true;
                continue;
            }
            if (roll % 3 == 0 || !sim.AnyQueued())
            {
                sim.LocalStep((int)(roll % (ulong)sim.NodeCount));
                continue;
            }
            var pending = new List<int>();
            for (int from = 0; from < sim.NodeCount; from++)
            {
                for (int to = 0; to < sim.NodeCount; to++)
                {
                    if (sim.HasChannel(from, to) && sim.Queued(from, to)) pending.Add(from * 100 + to);
                }
            }
            int chosen = pending[(int)(roll % (ulong)pending.Count)];
            sim.Deliver(chosen / 100, chosen % 100);
        }
        sim.Drain(500);
    }

    private static ChandyLamportSimulation Simulate(int seed, int steps, bool secondInitiator, int initiator)
    {
        var sim = new ChandyLamportSimulation(4, true);
        RunSchedule(sim, seed, steps, secondInitiator, initiator);
        return sim;
    }

    public static void Main()
    {
        ChandyLamportSimulation sim = Simulate(11, 160, false, 0);
        Check(sim.Complete(), "snapshot completed: every node recorded, every channel closed");
        Check(sim.ChannelsConsistent(out string channelReport), "channel states " + channelReport);
        Check(sim.CutConsistent(out string cutReport), "consistent cut " + cutReport);
        Check(sim.TotalRecordedChannelMessages() > 0, "messages really were in flight at the cut");
        foreach (SnapshotNode node in new[] { sim.Node(0), sim.Node(1), sim.Node(2), sim.Node(3) })
        {
            Check(node.Recorded && node.RecordedState >= 0, "node " + node.Id + " recorded a state");
            Check(node.ChannelState.Count == 2 && node.Closed.Count == 2,
                "node " + node.Id + " closed its two incoming channels");
        }
        Check(sim.Node(0).RecordedState + sim.Node(1).RecordedState + sim.Node(2).RecordedState
            + sim.Node(3).RecordedState > 0, "recorded states are not all empty");

        // Reproducible: the same seed produces the same cut.
        ChandyLamportSimulation again = Simulate(11, 160, false, 0);
        Check(again.Cut() == sim.Cut(), "the run is reproducible");
        Check(again.TotalRecordedChannelMessages() == sim.TotalRecordedChannelMessages(),
            "reproducible in-flight count");

        // Other seeds and a different initiator still produce consistent cuts.
        foreach (int seed in new[] { 3, 29, 101, 7777 })
        {
            ChandyLamportSimulation other = Simulate(seed, 140, false, seed % 4);
            bool complete = other.Complete();
            bool consistent = other.ChannelsConsistent(out string report);
            Check(complete && consistent, "seed " + seed + " " + report);
            Check(other.CutConsistent(out string cut), "seed " + seed + " cut " + cut);
        }

        // Two concurrent initiators: the second marker set must not disturb the recording.
        ChandyLamportSimulation concurrent = Simulate(57, 180, true, 0);
        Check(concurrent.Complete(), "concurrent snapshot completed");
        Check(concurrent.ChannelsConsistent(out string concurrentReport), "concurrent cut " + concurrentReport);
        Check(concurrent.CutConsistent(out string concurrentCut), "concurrent consistency " + concurrentCut);
        Check(concurrent.TotalRecordedChannelMessages() > 0, "concurrent snapshot has in-flight messages");

        // Negative control: break FIFO and the very same verifier must report the inconsistency.
        var broken = new ChandyLamportSimulation(4, false);
        RunSchedule(broken, 11, 160, false, 0);
        bool brokenChannels = broken.ChannelsConsistent(out string brokenReport);
        bool brokenCut = broken.CutConsistent(out string brokenCutReport);
        Check(!brokenChannels || !brokenCut,
            "the verifier catches the FIFO violation: channels=" + brokenChannels + " (" + brokenReport
            + ") cut=" + brokenCut + " (" + brokenCutReport + ")");
        Check(broken.Events.FindAll(line => line.StartsWith("FIFO VIOLATION")).Count > 0,
            "the schedule really did violate FIFO: " + broken.Events.Count + " events");

        Console.WriteLine("  nodes=4 events=" + sim.Events.Count + " inFlightMessages="
            + sim.TotalRecordedChannelMessages() + " recordedStates=" + sim.Node(0).RecordedState + "/"
            + sim.Node(1).RecordedState + "/" + sim.Node(2).RecordedState + "/" + sim.Node(3).RecordedState);
        Console.WriteLine("  cut=" + sim.Cut());
        Console.WriteLine("PASS");
    }
}
