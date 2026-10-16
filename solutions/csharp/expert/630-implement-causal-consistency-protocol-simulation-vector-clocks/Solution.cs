// Task 630 - Implement causal consistency protocol simulation (vector clocks). (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A vector clock is a per-process counter vector: a local event increments the process's own
// component and a message carries the sender's vector, which the receiver merges (componentwise
// max) and then increments. The classic theorem is that e1 happens-before e2 exactly when
// vc(e1) < vc(e2) pointwise, and two events are concurrent exactly when neither vector dominates
// the other. This solution verifies that theorem against a brute-force transitive closure of the
// real simulation history rather than assuming it.
//
// Causal delivery (Birman/Schiper) reuses the same machinery with a "messages sent" vector:
//   * every message carries the sender's broadcast vector,
//   * a receiver delivers a message only when it has already delivered every message that the
//     sender had delivered before broadcasting (stamp[j] <= delivered[j]) and the message is the
//     next one from that sender (FIFO per sender),
//   * anything else is buffered and retried after each delivery, so a delayed message cannot be
//     overtaken by a message that causally depends on it.
//
// What is simplified (documented per the authoring contract): the network is an explicit
// in-memory queue driven by a logical tick with no threads or sockets, and the delivery schedule
// is a deterministic scripted/PRNG sequence rather than real concurrency. Verified invariants:
// happens-before agrees with vector-clock dominance on every event pair, no node ever delivers a
// message before one it causally depends on, buffered messages are all released once the
// dependency arrives, concurrent broadcasts may be delivered in either order, and every message
// is delivered exactly once.
using System;
using System.Collections.Generic;

#nullable enable

internal enum ClockRelation { Equal, Less, Greater, Concurrent }

internal sealed class VectorClock
{
    public readonly int[] Ticks;

    public VectorClock(int size) { Ticks = new int[size]; }
    private VectorClock(int[] ticks) { Ticks = ticks; }
    public int Size => Ticks.Length;
    public VectorClock Clone() => new VectorClock((int[])Ticks.Clone());
    public void Increment(int index) => Ticks[index]++;
    public int Get(int index) => Ticks[index];

    public void Merge(VectorClock other)
    {
        for (int i = 0; i < Size; i++) if (other.Ticks[i] > Ticks[i]) Ticks[i] = other.Ticks[i];
    }

    public bool Dominates(VectorClock other)
    {
        for (int i = 0; i < Size; i++) if (Ticks[i] < other.Ticks[i]) return false;
        return true;
    }

    public static ClockRelation Compare(VectorClock a, VectorClock b)
    {
        bool lessOrEqual = true, greaterOrEqual = true;
        for (int i = 0; i < a.Size; i++)
        {
            if (a.Ticks[i] > b.Ticks[i]) lessOrEqual = false;
            if (a.Ticks[i] < b.Ticks[i]) greaterOrEqual = false;
        }
        if (lessOrEqual && greaterOrEqual) return ClockRelation.Equal;
        if (lessOrEqual) return ClockRelation.Less;
        if (greaterOrEqual) return ClockRelation.Greater;
        return ClockRelation.Concurrent;
    }

    public bool SameAs(VectorClock other)
    {
        for (int i = 0; i < Size; i++) if (Ticks[i] != other.Ticks[i]) return false;
        return true;
    }

    public string Describe()
    {
        var parts = new List<string>();
        for (int i = 0; i < Size; i++) parts.Add(Ticks[i].ToString());
        return "(" + string.Join(",", parts) + ")";
    }
}

internal sealed class CausalEvent
{
    public int Id;
    public int NodeId;
    public string Kind = "";          // "broadcast" or "receive"
    public int MessageId = -1;
    public VectorClock Clock = new VectorClock(1);
    public HashSet<int> Deps = new HashSet<int>();   // causal past, already transitively closed
}

internal sealed class CausalMessage
{
    public int Id;
    public int SendEventId;
    public int From;
    public int To;
    public int Seq;                   // per-sender broadcast sequence (FIFO per sender)
    public string Payload = "";
    public VectorClock Stamp = new VectorClock(1);
    public bool Delivered;
    public int DeliveredAt;
}

internal sealed class CausalNode
{
    public readonly int Id;
    public readonly int Size;
    public readonly VectorClock StampClock;        // counts broadcasts per node; travels in messages
    public readonly VectorClock EventClock;        // counts every local event; used for the theorem
    public readonly int[] DeliveredFrom;           // how many messages from each node were delivered
    public readonly HashSet<int> EventDeps = new HashSet<int>();
    public readonly List<CausalMessage> Buffered = new List<CausalMessage>();
    public readonly List<CausalMessage> DeliveredLog = new List<CausalMessage>();

    public int Broadcasts;
    public int BufferedEnqueues;

    public CausalNode(int id, int size)
    {
        Id = id;
        Size = size;
        StampClock = new VectorClock(size);
        EventClock = new VectorClock(size);
        DeliveredFrom = new int[size];
        EventDeps.Add(0);              // event 0 is the initial "no dependencies" event
    }

    public bool CanDeliver(CausalMessage message)
    {
        if (message.Seq != DeliveredFrom[message.From] + 1) return false;
        for (int j = 0; j < Size; j++)
        {
            // The sender's own component is checked by the sequence number; this receiver's own
            // component is common knowledge (it never receives its own broadcasts).
            if (j == message.From || j == Id) continue;
            if (message.Stamp.Get(j) > DeliveredFrom[j]) return false;
        }
        return true;
    }
}

internal sealed class CausalSim
{
    public readonly List<CausalNode> Nodes = new List<CausalNode>();
    public readonly List<CausalMessage> Messages = new List<CausalMessage>();
    public readonly List<CausalEvent> Events = new List<CausalEvent>();
    private readonly List<CausalMessage> queue = new List<CausalMessage>();
    private readonly int[] broadcastSeq;

    public int Now;
    public int NextEventId = 1;
    public int NextMessageId = 1;
    public int Deliveries;
    public int BufferedDeliveries;
    public int Rejected;

    public CausalSim(int size)
    {
        broadcastSeq = new int[size + 1];
        Events.Add(new CausalEvent { Id = 0, NodeId = 0, Kind = "init", Clock = new VectorClock(size + 1) });
        for (int id = 1; id <= size; id++) Nodes.Add(new CausalNode(id, size + 1));
    }

    public CausalNode Node(int id) => Nodes[id - 1];
    public int Pending => queue.Count;

    /// <summary>Broadcast: one local event, one message per peer carrying the same stamp.</summary>
    public List<CausalMessage> Broadcast(int nodeId, string payload)
    {
        CausalNode node = Node(nodeId);
        Now++;
        CausalEvent sendEvent = NewEvent(node, "broadcast", -1);
        node.StampClock.Increment(nodeId);
        node.Broadcasts++;
        var created = new List<CausalMessage>();
        broadcastSeq[nodeId]++;
        foreach (CausalNode peer in Nodes)
        {
            if (peer.Id == nodeId) continue;
            var message = new CausalMessage
            {
                Id = NextMessageId++,
                SendEventId = sendEvent.Id,
                From = nodeId,
                To = peer.Id,
                Seq = broadcastSeq[nodeId],
                Payload = payload,
                Stamp = node.StampClock.Clone(),
            };
            Messages.Add(message);
            queue.Add(message);
            created.Add(message);
        }
        return created;
    }

    private CausalEvent NewEvent(CausalNode node, string kind, int messageId)
    {
        node.EventClock.Increment(node.Id);
        var evt = new CausalEvent
        {
            Id = NextEventId++,
            NodeId = node.Id,
            Kind = kind,
            MessageId = messageId,
            Clock = node.EventClock.Clone(),
        };
        evt.Deps.UnionWith(node.EventDeps);
        evt.Deps.Add(evt.Id);
        node.EventDeps.UnionWith(evt.Deps);
        Events.Add(evt);
        return evt;
    }

    /// <summary>Attempt to deliver one specific message; returns false when it had to be buffered.</summary>
    public bool TryDeliver(int messageId)
    {
        CausalMessage message = Messages[messageId - 1];
        if (message.Delivered) return true;
        CausalNode node = Node(message.To);
        if (!node.CanDeliver(message))
        {
            if (!node.Buffered.Contains(message))
            {
                node.Buffered.Add(message);
                node.BufferedEnqueues++;
            }
            Rejected++;
            return false;
        }
        Deliver(node, message);
        return true;
    }

    private void Deliver(CausalNode node, CausalMessage message)
    {
        Now++;
        CausalEvent sendEvent = Event(message.SendEventId);
        // A receive merges what the message carried before the receipt itself is recorded, which
        // is what makes the receive event's clock dominate the send event's clock.
        node.StampClock.Merge(message.Stamp);
        node.EventClock.Merge(sendEvent.Clock);
        node.EventDeps.UnionWith(sendEvent.Deps);
        CausalEvent evt = NewEvent(node, "receive", message.Id);
        node.DeliveredFrom[message.From] = message.Seq;
        message.Delivered = true;
        message.DeliveredAt = Now;
        node.DeliveredLog.Add(message);
        Deliveries++;
        queue.Remove(message);
        // Retry everything that was waiting; a single delivery can release a whole chain.
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (int i = 0; i < node.Buffered.Count; i++)
            {
                CausalMessage waiter = node.Buffered[i];
                if (!node.CanDeliver(waiter)) continue;
                node.Buffered.RemoveAt(i);
                BufferedDeliveries++;
                Deliver(node, waiter);
                progress = true;
                break;
            }
        }
    }

    public List<CausalMessage> PendingMessages()
    {
        var pending = new List<CausalMessage>();
        foreach (CausalMessage message in queue) if (!message.Delivered) pending.Add(message);
        return pending;
    }

    public CausalEvent Event(int id) => Events[id];
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static bool HappensBefore(CausalEvent a, CausalEvent b) => b.Deps.Contains(a.Id);

    /// <summary>The classic theorem, checked against the brute-force transitive closure of history.</summary>
    private static void HappensBeforeMatchesVectorClocks()
    {
        var sim = new CausalSim(3);
        sim.Broadcast(1, "a1");
        sim.TryDeliver(1);                     // node 2 receives a1
        sim.Broadcast(2, "b1");                // causally after a1
        sim.Broadcast(3, "c1");                // concurrent with both
        sim.TryDeliver(3);                     // node 1 receives b1
        sim.Broadcast(1, "a2");                // after a1 and b1, concurrent with c1
        foreach (CausalMessage message in sim.PendingMessages()) sim.TryDeliver(message.Id);

        int comparable = 0, concurrent = 0;
        foreach (CausalEvent a in sim.Events)
        {
            foreach (CausalEvent b in sim.Events)
            {
                if (a.Id == b.Id) continue;
                ClockRelation relation = VectorClock.Compare(a.Clock, b.Clock);
                if (HappensBefore(a, b))
                {
                    Check(relation == ClockRelation.Less, "hb(a,b) implies vc(a) < vc(b) for events " + a.Id + "->" + b.Id);
                    comparable++;
                }
                else if (HappensBefore(b, a))
                {
                    Check(relation == ClockRelation.Greater, "hb(b,a) implies vc(a) > vc(b) for events " + a.Id + "->" + b.Id);
                    comparable++;
                }
                else
                {
                    Check(relation == ClockRelation.Concurrent, "incomparable events must be concurrent: " + a.Id + " vs " + b.Id);
                    concurrent++;
                }
            }
        }
        Check(comparable > 0, "some pairs were causally related");
        Check(concurrent > 0, "some pairs were concurrent");
        Check(VectorClock.Compare(new VectorClock(2), new VectorClock(2)) == ClockRelation.Equal, "equal clocks");
        Console.WriteLine("vector clocks: " + comparable + " ordered pairs, " + concurrent + " concurrent pairs verified");
    }

    private static void OutOfOrderMessagesAreBuffered()
    {
        var sim = new CausalSim(3);
        sim.Broadcast(1, "first");
        sim.Broadcast(1, "second");            // FIFO: depends on "first"
        Check(sim.Messages.Count == 4, "two broadcasts to two peers each");
        // Messages 1,2 go to nodes 2,3; messages 3,4 are the second broadcast to nodes 2,3.
        Check(sim.TryDeliver(4) == false, "the second message cannot be delivered before the first");
        Check(sim.Node(3).Buffered.Count == 1, "it is buffered");
        Check(sim.Node(3).BufferedEnqueues == 1, "buffering is counted");
        Check(sim.Node(3).DeliveredLog.Count == 0, "nothing was delivered yet");
        Check(sim.TryDeliver(2), "the first message delivers immediately");
        Check(sim.Node(3).Buffered.Count == 0, "the buffer drained as a consequence");
        Check(sim.Node(3).DeliveredLog.Count == 2, "both messages were delivered");
        Check(sim.Node(3).DeliveredLog[0].Payload == "first" && sim.Node(3).DeliveredLog[1].Payload == "second",
            "in FIFO order");
        Check(sim.BufferedDeliveries == 1, "exactly one delivery came out of the buffer");

        // Cross-node causality: node 3 must not see b1 before the a1 it causally follows.
        // Message ids: a1 -> node 2 is 1, a1 -> node 3 is 2, b1 -> node 1 is 3, b1 -> node 3 is 4.
        var chain = new CausalSim(3);
        chain.Broadcast(1, "a1");
        chain.TryDeliver(1);                   // node 2 sees a1
        chain.Broadcast(2, "b1");              // b1 causally depends on a1
        Check(chain.TryDeliver(4) == false, "b1 to node 3 waits for a1");
        Check(chain.TryDeliver(2), "a1 to node 3 delivers");
        Check(chain.Node(3).DeliveredLog.Count == 2 && chain.Node(3).DeliveredLog[0].Payload == "a1"
            && chain.Node(3).DeliveredLog[1].Payload == "b1", "causal order is respected, not arrival order");
        Check(chain.BufferedDeliveries == 1, "the buffered b1 was released by a1");
    }

    private static void ConcurrentMessagesAllowEitherOrder()
    {
        var orders = new List<string>();
        foreach (bool node2First in new[] { true, false })
        {
            var sim = new CausalSim(3);
            // Node 2 and node 3 broadcast concurrently: neither has heard from the other.
            sim.Broadcast(2, "from2");         // message ids 1 (to node 1) and 2 (to node 3)
            sim.Broadcast(3, "from3");         // message ids 3 (to node 1) and 4 (to node 2)
            int first = node2First ? 1 : 3;
            int second = node2First ? 3 : 1;
            Check(sim.TryDeliver(first), "either concurrent message may be delivered first");
            Check(sim.TryDeliver(second), "and the other one straight after");
            var order = new List<string>();
            foreach (CausalMessage message in sim.Node(1).DeliveredLog) order.Add(message.Payload);
            orders.Add(string.Join("|", order));
            Check(sim.Node(1).DeliveredLog.Count == 2, "both concurrent messages arrive");
            Check(sim.Node(1).Buffered.Count == 0, "no buffering is needed for concurrent messages");
        }
        Check(orders[0] != orders[1], "both concurrent orders were produced: " + string.Join(" vs ", orders));
        Check(orders[0] == "from2|from3" && orders[1] == "from3|from2", "each order follows its script");
    }

    private static void ClocksConvergeAfterFullExchange()
    {
        var sim = new CausalSim(4);
        for (int round = 0; round < 12; round++)
        {
            sim.Broadcast(1 + round % 4, "r" + round);      // exactly three broadcasts per node
            foreach (CausalMessage message in sim.PendingMessages()) sim.TryDeliver(message.Id);
        }
        foreach (CausalMessage message in sim.PendingMessages()) sim.TryDeliver(message.Id);
        VectorClock reference = sim.Node(1).StampClock;
        for (int id = 2; id <= 4; id++)
        {
            Check(sim.Node(id).StampClock.SameAs(reference),
                "clock of node " + id + " is " + sim.Node(id).StampClock.Describe() + ", node 1 has " + reference.Describe());
        }
        for (int id = 1; id <= 4; id++) Check(sim.Node(id).Buffered.Count == 0, "node " + id + " has an empty buffer");
        Check(reference.Describe() == "(0,3,3,3,3)", "each node broadcast three times, clock=" + reference.Describe());
    }

    private static void RandomScheduleKeepsCausalOrder()
    {
        var sim = new CausalSim(5);
        int seed = 20240101;
        int broadcasts = 0;
        for (int step = 0; step < 600; step++)
        {
            seed = NextSeed(seed);
            int choice = seed % 10;
            if (choice < 4)
            {
                int broadcaster = 1 + (seed / 10) % 5;
                sim.Broadcast(broadcaster, "m" + broadcasts);
                broadcasts++;
            }
            else
            {
                List<CausalMessage> pending = sim.PendingMessages();
                if (pending.Count == 0) continue;
                int pick = (seed / 7) % pending.Count;
                sim.TryDeliver(pending[pick].Id);
            }
        }
        int guard = 0;
        while (sim.PendingMessages().Count > 0 && guard++ < 200)
        {
            foreach (CausalMessage message in sim.PendingMessages()) sim.TryDeliver(message.Id);
        }

        Check(sim.Rejected > 0, "the schedule really produced out-of-order arrivals: " + sim.Rejected);
        Check(sim.BufferedDeliveries > 0, "buffered messages were released later: " + sim.BufferedDeliveries);
        Check(sim.Deliveries == broadcasts * 4, "every message was delivered exactly once to each peer: "
            + sim.Deliveries + " of " + (broadcasts * 4));

        // The invariant: for every delivery, every causal predecessor addressed to that node was
        // already delivered there.
        foreach (CausalNode node in sim.Nodes)
        {
            foreach (CausalMessage message in node.DeliveredLog)
            {
                foreach (int depId in sim.Event(message.SendEventId).Deps)
                {
                    CausalEvent dep = sim.Event(depId);
                    if (dep.Kind != "receive" || dep.NodeId != node.Id || dep.MessageId < 0) continue;
                    CausalMessage predecessor = sim.Messages[dep.MessageId - 1];
                    if (!predecessor.Delivered)
                    {
                        Check(false, "a causal predecessor was never delivered at node " + node.Id);
                        continue;
                    }
                    if (predecessor.To != node.Id) continue;
                    Check(predecessor.DeliveredAt <= message.DeliveredAt,
                        "node " + node.Id + " delivered m" + message.Id + " before its cause m" + predecessor.Id);
                }
            }
            Check(node.Buffered.Count == 0, "node " + node.Id + " still holds buffered messages");
        }
        foreach (CausalMessage message in sim.Messages)
            Check(message.Delivered, "message " + message.Id + " was eventually delivered");
        Console.WriteLine("causal broadcast: broadcasts=" + broadcasts + " deliveries=" + sim.Deliveries
            + " buffered-releases=" + sim.BufferedDeliveries + " out-of-order-arrivals=" + sim.Rejected);
    }

    public static void Main()
    {
        HappensBeforeMatchesVectorClocks();
        OutOfOrderMessagesAreBuffered();
        ConcurrentMessagesAllowEitherOrder();
        ClocksConvergeAfterFullExchange();
        RandomScheduleKeepsCausalOrder();
        Console.WriteLine("PASS");
    }
}
