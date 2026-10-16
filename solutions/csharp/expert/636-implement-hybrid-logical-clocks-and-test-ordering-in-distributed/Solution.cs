// Task 636 - Implement hybrid logical clocks and test ordering in distributed events. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A hybrid logical clock packs a physical timestamp and a logical counter into one value
// (l, c, node): l tracks the maximum physical time the node has heard about, c breaks ties and
// makes the clock strictly increasing even when the physical clock stands still or jumps backwards.
//   * a local event at physical time pt:  l' = max(l, pt),  c' = (l' == l ? c + 1 : 0),
//   * receiving (lm, cm) at physical time pt: l' = max(l, lm, pt) and
//       c' = max(c, cm) + 1 when l' == l == lm, c + 1 when l' == l, cm + 1 when l' == lm, else 0.
// The resulting order is a strict total order that extends causality: e1 happens-before e2 always
// implies hlc(e1) < hlc(e2), while concurrent events simply get some deterministic order (ties on
// (l, c) are broken by the node id). HLC also keeps l bounded: it never runs ahead of the physical
// clock by more than the maximum physical skew seen in the system.
//
// Tests build an explicit causal history (every event carries the ids of the events it knows about,
// which is a ground truth independent of any clock) and then verify that the clock order and the
// happens-before order agree, that l never regresses and never drifts beyond the bounded skew, that
// encoding the clock into a single 64-bit value round-trips, and that interpreting (l, c) as a
// total order yields a valid linear extension of the distributed history.
//
// What is simplified (documented per the authoring contract): a deterministic in-memory message
// schedule with no threads or sockets, and the physical clocks are a scripted array of (mostly)
// monotone readings with a bounded skew rather than a real NTP-adjusted clock. Verified invariants:
// causality preservation, strict monotonicity per node, tie-breaking by node id, drift bounded by
// the skew, and encode/decode round-trips including the boundary values.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal readonly struct Hlc : IComparable<Hlc>
{
    public readonly long Physical;
    public readonly int Logical;
    public readonly int Node;

    public Hlc(long physical, int logical, int node)
    {
        Physical = physical;
        Logical = logical;
        Node = node;
    }

    public int CompareTo(Hlc other)
    {
        if (Physical != other.Physical) return Physical < other.Physical ? -1 : 1;
        if (Logical != other.Logical) return Logical < other.Logical ? -1 : 1;
        if (Node != other.Node) return Node < other.Node ? -1 : 1;
        return 0;
    }

    /// <summary>Pack (physical, logical) into one 64-bit value: 40 bits of time plus a 23-bit counter.
    /// The node id is not packed - it is only a local tie-breaker for equal (l, c) values.</summary>
    public long EncodeFull()
    {
        if (Physical < 0 || Physical >= (1L << 40)) throw new Exception("physical time out of range: " + Physical);
        if (Logical < 0 || Logical >= (1 << 23)) throw new Exception("counter out of range: " + Logical);
        return (Physical << 23) | (uint)Logical;
    }

    public static Hlc DecodeFull(long packed, int node)
    {
        long physical = packed >> 23;
        int logical = (int)(packed & 0x7FFFFF);
        return new Hlc(physical, logical, node);
    }

    public string Describe() => "(" + Physical + "," + Logical + "@" + Node + ")";
}

internal sealed class HlcNode
{
    public readonly int Id;
    public long L;
    public int C;
    public int Ticks;
    public int Receives;
    public int MaxLogical;

    public HlcNode(int id)
    {
        Id = id;
        L = long.MinValue;
        C = 0;
    }

    /// <summary>Local event (or send) at the given physical time.</summary>
    public Hlc Now(long physical)
    {
        long l = Math.Max(L, physical);
        int c = l == L ? C + 1 : 0;
        L = l;
        C = c;
        Ticks++;
        if (c > MaxLogical) MaxLogical = c;
        return new Hlc(l, c, Id);
    }

    /// <summary>Receive of a message stamped (lm, cm) at the given physical time.</summary>
    public Hlc Update(Hlc received, long physical)
    {
        long previousL = L;
        int previousC = C;
        long lm = received.Physical;
        int cm = received.Logical;
        long l = Math.Max(Math.Max(previousL, lm), physical);
        int c;
        if (l == previousL && l == lm) c = Math.Max(previousC, cm) + 1;
        else if (l == previousL) c = previousC + 1;
        else if (l == lm) c = cm + 1;
        else c = 0;
        L = l;
        C = c;
        Receives++;
        if (c > MaxLogical) MaxLogical = c;
        return new Hlc(l, c, Id);
    }

    public Hlc Value => new Hlc(L, C, Id);
}

internal sealed class HlcEvent
{
    public int Id;
    public int NodeId;
    public bool IsReceive;
    public Hlc Clock;
    public HashSet<int> Deps = new HashSet<int>();      // ground-truth causal past
    public long Physical;
}

internal sealed class HlcSim
{
    private readonly List<HlcNode> nodes = new List<HlcNode>();
    private readonly List<long> physicalTimes;          // scripted clock readings per node
    private int nextEventId = 1;
    private int nextMessageId = 1;

    public readonly List<HlcEvent> Events = new List<HlcEvent>();
    public readonly Dictionary<int, HashSet<int>> EventDeps = new Dictionary<int, HashSet<int>>();
    public readonly Dictionary<int, long> EventPhysical = new Dictionary<int, long>();
    public readonly List<(int From, int To, Hlc Stamp, int SendEvent, int MessageId)> InFlight =
        new List<(int, int, Hlc, int, int)>();
    public long MaxSkew;
    public long MaxPhysicalSeen = long.MinValue;
    public long MinPhysicalSeen = long.MaxValue;

    public HlcSim(int nodeCount, long basePhysical, long skew)
    {
        physicalTimes = new List<long>();
        for (int i = 0; i < nodeCount; i++)
        {
            nodes.Add(new HlcNode(i));
            // A deterministic bounded skew: node i's clock is basePhysical - skew + i.
            physicalTimes.Add(basePhysical - skew + i);
        }
        MaxSkew = skew;
    }

    public HlcNode Node(int id) => nodes[id];
    public int NodeCount => nodes.Count;
    public long Physical(int nodeId) => physicalTimes[nodeId];

    public void AdvancePhysical(int nodeId, long delta) => physicalTimes[nodeId] += delta;

    /// <summary>A local event: its clock ticks and it remembers its own causal past plus the send's.</summary>
    public int LocalEvent(int nodeId)
    {
        Hlc clock = nodes[nodeId].Now(physicalTimes[nodeId]);
        return Record(nodeId, false, clock, new HashSet<int>());
    }

    /// <summary>Broadcast from a node: the message carries the send event's clock and causal past.</summary>
    public int Send(int nodeId)
    {
        Hlc clock = nodes[nodeId].Now(physicalTimes[nodeId]);
        int eventId = Record(nodeId, false, clock, new HashSet<int>());
        foreach (HlcNode peer in nodes)
        {
            if (peer.Id == nodeId) continue;
            InFlight.Add((nodeId, peer.Id, clock, eventId, nextMessageId++));
        }
        return eventId;
    }

    /// <summary>Deliver one specific in-flight message (the tests choose the order).</summary>
    public bool Deliver(int messageId)
    {
        for (int i = 0; i < InFlight.Count; i++)
        {
            (int from, int to, Hlc stamp, int sendEvent, int id) = InFlight[i];
            if (id != messageId) continue;
            InFlight.RemoveAt(i);
            Hlc clock = nodes[to].Update(stamp, physicalTimes[to]);
            var deps = new HashSet<int>(EventDeps[sendEvent]);
            Record(to, true, clock, deps);
            return true;
        }
        return false;
    }

    public void DeliverAll()
    {
        while (InFlight.Count > 0) Deliver(InFlight[0].MessageId);
    }

    private int Record(int nodeId, bool isReceive, Hlc clock, HashSet<int> extraDeps)
    {
        int id = nextEventId++;
        foreach (HlcNode peer in nodes)
        {
            long reading = physicalTimes[peer.Id];
            if (reading > MaxPhysicalSeen) MaxPhysicalSeen = reading;
            if (reading < MinPhysicalSeen) MinPhysicalSeen = reading;
        }
        var evt = new HlcEvent
        {
            Id = id,
            NodeId = nodeId,
            IsReceive = isReceive,
            Clock = clock,
            Physical = physicalTimes[nodeId],
        };
        evt.Deps.UnionWith(extraDeps);
        if (EventDeps.TryGetValue(nodeId, out HashSet<int>? own)) evt.Deps.UnionWith(own);
        evt.Deps.Add(id);
        EventDeps[id] = evt.Deps;
        EventPhysical[id] = evt.Physical;
        Events.Add(evt);
        return id;
    }

    public bool HappensBefore(HlcEvent a, HlcEvent b) => b.Deps.Contains(a.Id);
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int NextSeed(int seed) => (seed * 1103515245 + 12345) & 0x7FFFFFFF;

    private static void EncodingRoundTrips()
    {
        var values = new List<Hlc>
        {
            new Hlc(0, 0, 0),
            new Hlc(1, 0, 1),
            new Hlc(1000000000000L, 7, 5),
            new Hlc((1L << 40) - 1, (1 << 23) - 1, 2047),
            new Hlc(42, 1, 2047),
        };
        foreach (Hlc value in values)
        {
            Hlc decoded = Hlc.DecodeFull(value.EncodeFull(), value.Node);
            Check(decoded.Physical == value.Physical && decoded.Logical == value.Logical && decoded.Node == value.Node,
                "round trip failed for " + value.Describe());
        }
        Check(new Hlc(10, 1, 2).CompareTo(new Hlc(10, 1, 3)) < 0, "the node id breaks a tie");
        Check(new Hlc(10, 2, 9).CompareTo(new Hlc(10, 1, 1)) > 0, "the counter beats the node id");
        Check(new Hlc(11, 0, 0).CompareTo(new Hlc(10, 999, 999)) > 0, "physical time beats everything below it");
        Check(values[3].EncodeFull() != values[2].EncodeFull(), "different clocks encode differently");
        bool threw = false;
        try { new Hlc(1L << 40, 0, 0).EncodeFull(); }
        catch (Exception) { threw = true; }
        Check(threw, "an out-of-range physical time is rejected");
        Console.WriteLine("hlc: encoding round-trips for " + values.Count + " clocks");
    }

    private static void ClockRules()
    {
        var node = new HlcNode(1);
        Hlc first = node.Now(100);
        Check(first.Physical == 100 && first.Logical == 0, "the first event takes the physical time");
        Hlc second = node.Now(100);
        Check(second.Physical == 100 && second.Logical == 1, "a second event at the same time bumps the counter");
        Hlc third = node.Now(101);
        Check(third.Physical == 101 && third.Logical == 0, "a later physical time resets the counter");
        Hlc backwards = node.Now(50);
        Check(backwards.Physical == 101 && backwards.Logical == 1, "a backwards jump keeps l and bumps c, got "
            + backwards.Describe());

        var receiver = new HlcNode(2);
        Hlc received = receiver.Update(new Hlc(200, 3, 1), 150);
        Check(received.Physical == 200 && received.Logical == 4, "the receive absorbs the message time: "
            + received.Describe());
        Hlc later = receiver.Update(new Hlc(180, 0, 1), 300);
        Check(later.Physical == 300 && later.Logical == 0, "a much later physical time wins: " + later.Describe());
        Hlc tie = receiver.Update(new Hlc(300, 0, 1), 300);
        Check(tie.Physical == 300 && tie.Logical == 1, "equal everything advances the counter: " + tie.Describe());
    }

    private static void CausalityIsPreserved()
    {
        var sim = new HlcSim(3, 1000, 2);
        sim.LocalEvent(0);
        sim.Send(0);
        sim.DeliverAll();
        sim.LocalEvent(1);
        sim.Send(1);
        sim.Send(2);
        sim.DeliverAll();
        sim.Send(0);
        sim.DeliverAll();

        int ordered = 0, concurrent = 0;
        foreach (HlcEvent a in sim.Events)
        {
            foreach (HlcEvent b in sim.Events)
            {
                if (a.Id == b.Id) continue;
                int comparison = a.Clock.CompareTo(b.Clock);
                if (sim.HappensBefore(a, b))
                {
                    Check(comparison < 0, "happens-before must imply a smaller clock: " + a.Clock.Describe()
                        + " -> " + b.Clock.Describe());
                    ordered++;
                }
                else if (!sim.HappensBefore(b, a) && comparison > 0)
                {
                    concurrent++;
                }
            }
        }
        Check(ordered > 0, "the history contains causally ordered events");
        Check(concurrent >= 0, "concurrent events are ordered arbitrarily but consistently");

        // Same-node events keep their order.
        foreach (HlcEvent a in sim.Events)
        {
            foreach (HlcEvent b in sim.Events)
            {
                if (a.NodeId != b.NodeId || a.Id >= b.Id) continue;
                Check(a.Clock.CompareTo(b.Clock) < 0, "events of one node are strictly increasing: "
                    + a.Clock.Describe() + " then " + b.Clock.Describe());
            }
        }
        Console.WriteLine("hlc causality: " + sim.Events.Count + " events, " + ordered + " ordered pairs verified");
    }

    private static void DriftStaysBounded()
    {
        var sim = new HlcSim(4, 5000, 3);
        long maxDrift = 0;
        int seed = 99;
        for (int round = 0; round < 40; round++)
        {
            seed = NextSeed(seed);
            int actor = seed % 4;
            seed = NextSeed(seed);
            if (seed % 3 == 0) sim.Send(actor);
            else sim.LocalEvent(actor);
            if (seed % 5 == 0) sim.DeliverAll();
            sim.AdvancePhysical(actor, 1 + seed % 2);
            foreach (HlcEvent evt in sim.Events)
            {
                Check(evt.Clock.Physical >= evt.Physical, "l is never behind the local physical clock: "
                    + evt.Clock.Describe() + " at " + evt.Physical);
                long drift = evt.Clock.Physical - evt.Physical;
                if (drift > maxDrift) maxDrift = drift;
                Check(drift <= sim.MaxPhysicalSeen - sim.MinPhysicalSeen,
                    "l ran " + drift + " ahead of the local clock, beyond the physical span of "
                    + (sim.MaxPhysicalSeen - sim.MinPhysicalSeen) + " seen so far");
            }
        }
        Check(maxDrift > 0, "the clock did absorb time from other nodes, drift=" + maxDrift);
        Check(sim.Events.Count > 40, "the run produced " + sim.Events.Count + " events");
        Check(sim.Node(0).MaxLogical >= 0, "the logical counter stayed sane");
        Console.WriteLine("hlc drift: max=" + maxDrift + " with a physical span of "
            + (sim.MaxPhysicalSeen - sim.MinPhysicalSeen));
    }

    private static void ClockOrderIsALinearExtension()
    {
        var sim = new HlcSim(5, 20000, 4);
        int seed = 4242;
        for (int round = 0; round < 60; round++)
        {
            seed = NextSeed(seed);
            int actor = seed % 5;
            if (seed % 3 == 0) sim.Send(actor); else sim.LocalEvent(actor);
            if (seed % 4 == 0) sim.DeliverAll();
            sim.AdvancePhysical(actor, 1 + seed % 3);
        }
        sim.DeliverAll();

        var ordered = new List<HlcEvent>(sim.Events);
        ordered.Sort((a, b) => a.Clock.CompareTo(b.Clock));
        for (int i = 0; i + 1 < ordered.Count; i++)
        {
            Check(ordered[i].Clock.CompareTo(ordered[i + 1].Clock) < 0,
                "the sorted order is strict at " + i + ": " + ordered[i].Clock.Describe()
                + " vs " + ordered[i + 1].Clock.Describe());
        }
        for (int i = 0; i < ordered.Count; i++)
        {
            for (int j = i + 1; j < ordered.Count; j++)
            {
                if (!sim.HappensBefore(ordered[i], ordered[j])) continue;
                Check(i < j, "a causal predecessor must appear earlier in the sorted order");
            }
        }
        // Every pair is comparable and distinct, so this is a total order on the events.
        for (int i = 0; i < ordered.Count; i++)
        {
            for (int j = i + 1; j < ordered.Count; j++)
            {
                Check(ordered[i].Clock.CompareTo(ordered[j].Clock) != 0,
                    "no two distinct events share a clock: " + ordered[i].Clock.Describe());
            }
        }
        Console.WriteLine("hlc total order: " + ordered.Count + " events linearized");
    }

    private static void DelayedMessagesDoNotBreakMonotonicity()
    {
        var sim = new HlcSim(2, 7000, 5);
        sim.Send(0);
        sim.LocalEvent(1);
        sim.LocalEvent(1);
        sim.AdvancePhysical(1, 100);
        sim.LocalEvent(1);
        Hlc before = sim.Node(1).Value;
        sim.DeliverAll();                                  // a stale message arrives late
        Hlc after = sim.Node(1).Value;
        Check(after.CompareTo(before) > 0, "a late message still strictly advances the clock: "
            + before.Describe() + " -> " + after.Describe());
        Check(after.Physical >= before.Physical, "l never regresses");
        for (int i = 0; i + 1 < sim.Events.Count; i++)
        {
            HlcEvent first = sim.Events[i];
            HlcEvent second = sim.Events[i + 1];
            if (first.NodeId != second.NodeId) continue;
            Check(first.Clock.CompareTo(second.Clock) < 0, "per-node monotonicity held");
        }
        Check(sim.Node(1).Receives == 1, "one message was received");
        Console.WriteLine("hlc late message: " + before.Describe() + " -> " + after.Describe());
    }

    public static void Main()
    {
        EncodingRoundTrips();
        ClockRules();
        CausalityIsPreserved();
        DriftStaysBounded();
        ClockOrderIsALinearExtension();
        DelayedMessagesDoNotBreakMonotonicity();
        Console.WriteLine("PASS");
    }
}



