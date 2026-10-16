// Task 631 - Implement gossip protocol for membership and failure detection. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A SWIM-style membership service: every node keeps a member list of (id, incarnation, status)
// and every gossip interval it sends a digest of that list to a few peers, who merge it with the
// rule "higher incarnation wins, and on equal incarnation the more severe status wins"
// (Alive < Suspect < Dead). Failure detection is direct: a node pings members round-robin, and
// when the ack does not arrive within the ping timeout it marks the member Suspect in its own
// view; if no refutation arrives before the suspicion timeout, the member is declared Dead. A
// node that hears a Suspect/Dead claim about ITSELF refutes it by incrementing its own
// incarnation, because alive@i+1 beats dead@i - that refutation is what makes false suspicions
// self-healing once connectivity returns, while a node that is really down can never refute and
// stays excluded.
//
// What is simplified (documented per the authoring contract): the network is an explicit in-memory
// queue driven by a logical tick (no threads, no sockets), peer selection uses a per-node
// deterministic PRNG instead of randomness, membership starts from a full bootstrap list, and
// pings are point-to-point rather than the full indirect-ping protocol. Verified invariants:
// gossip reaches every live node and their views converge, a crashed node is declared dead and
// purged everywhere, suspicions of an alive node are refuted and lifted, partitioned nodes are
// excluded while isolated, and the merge operation is idempotent and order-independent.
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable

internal enum MemberStatus { Alive, Suspect, Dead }

internal sealed class MemberEntry
{
    public int Id;
    public int Incarnation;
    public MemberStatus Status;

    public MemberEntry(int id, int incarnation, MemberStatus status)
    {
        Id = id;
        Incarnation = incarnation;
        Status = status;
    }
}

internal sealed class MemberList
{
    private readonly Dictionary<int, MemberEntry> entries = new Dictionary<int, MemberEntry>();

    public IEnumerable<MemberEntry> Entries => entries.Values;
    public int Count => entries.Count;
    public MemberEntry? Get(int id) => entries.TryGetValue(id, out MemberEntry? entry) ? entry : null;
    public bool IsAlive(int id) => entries.TryGetValue(id, out MemberEntry? entry) && entry.Status == MemberStatus.Alive;

    public void Set(int id, int incarnation, MemberStatus status)
        => entries[id] = new MemberEntry(id, incarnation, status);

    /// <summary>Merge one observation; returns true when the view actually changed.</summary>
    public bool Merge(int id, int incarnation, MemberStatus status)
    {
        if (!entries.TryGetValue(id, out MemberEntry? entry))
        {
            entries[id] = new MemberEntry(id, incarnation, status);
            return true;
        }
        if (incarnation > entry.Incarnation
            || (incarnation == entry.Incarnation && (int)status > (int)entry.Status))
        {
            entry.Incarnation = incarnation;
            entry.Status = status;
            return true;
        }
        return false;
    }

    public List<int> SortedIds()
    {
        var ids = new List<int>(entries.Keys);
        ids.Sort();
        return ids;
    }

    public List<int> AliveIds()
    {
        var ids = new List<int>();
        foreach (int id in SortedIds()) if (IsAlive(id)) ids.Add(id);
        return ids;
    }

    public List<MemberEntry> Digest()
    {
        var digest = new List<MemberEntry>();
        foreach (int id in SortedIds()) digest.Add(entries[id]);
        return digest;
    }

    public int CountWith(MemberStatus status)
    {
        int count = 0;
        foreach (MemberEntry entry in entries.Values) if (entry.Status == status) count++;
        return count;
    }

    public string Describe()
    {
        var sb = new StringBuilder();
        foreach (int id in SortedIds())
        {
            MemberEntry entry = entries[id];
            sb.Append(id).Append(':').Append(entry.Incarnation).Append(entry.Status.ToString()[0]).Append(' ');
        }
        return sb.ToString().TrimEnd();
    }
}

internal sealed class GossipMessage
{
    public int Seq;
    public int From;
    public int To;
    public string Kind = "";
    public int Incarnation;
    public List<MemberEntry> Digest = new List<MemberEntry>();
    public int DeliverTick;
}

internal sealed class GossipNode
{
    public readonly int Id;
    public readonly MemberList View = new MemberList();
    public int Incarnation = 1;
    public bool Down;
    public int Cursor;
    public int PingTarget = -1;
    public int PingSentTick = -1;
    public int SuspectTarget = -1;
    public int SuspectDeadline = -1;
    public int Seed;
    public int PingTimeouts;
    public int DeadDeclarations;
    public int Refutations;
    public int GossipSends;
    public int ReceivedGossip;

    public GossipNode(int id) { Id = id; }
    public bool IsSuspecting(int id) => SuspectTarget == id;
}

internal sealed class GossipSim
{
    public readonly List<GossipNode> Nodes = new List<GossipNode>();
    private readonly List<GossipMessage> queue = new List<GossipMessage>();
    private readonly HashSet<string> blocked = new HashSet<string>();
    private int seq;

    public int Now;
    public int PingInterval = 3;
    public int PingTimeout = 4;
    public int SuspicionTimeout = 5;
    public int GossipInterval = 2;
    public int Fanout = 2;
    public int MessagesSent;
    public int MessagesDropped;
    public int ViewChanges;
    public int NoopMerges;

    public GossipSim(int size)
    {
        for (int id = 1; id <= size; id++)
        {
            var node = new GossipNode(id) { Seed = id * 7919 + 13 };
            Nodes.Add(node);
        }
        foreach (GossipNode node in Nodes)
        {
            node.View.Set(node.Id, 1, MemberStatus.Alive);
            if (node.Id == 1)
            {
                // The seed node starts with the whole configuration; everybody else is told about
                // the seed only and discovers the rest through gossip.
                foreach (GossipNode other in Nodes) node.View.Set(other.Id, 1, MemberStatus.Alive);
            }
            else
            {
                node.View.Set(1, 1, MemberStatus.Alive);
            }
        }
    }

    public GossipNode Node(int id) => Nodes[id - 1];

    public void Block(int a, int b) { blocked.Add(a + "->" + b); blocked.Add(b + "->" + a); }
    public void Heal(int a, int b) { blocked.Remove(a + "->" + b); blocked.Remove(b + "->" + a); }

    public void Partition(int[] side, int[] other)
    {
        foreach (int a in side) foreach (int b in other) Block(a, b);
    }

    public void HealPartition(int[] side, int[] other)
    {
        foreach (int a in side) foreach (int b in other) Heal(a, b);
    }

    public void Send(int from, int to, string kind, GossipNode sender, List<MemberEntry>? digest)
    {
        MessagesSent++;
        if (blocked.Contains(from + "->" + to)) { MessagesDropped++; return; }
        queue.Add(new GossipMessage
        {
            Seq = seq++,
            From = from,
            To = to,
            Kind = kind,
            Incarnation = sender.Incarnation,
            Digest = digest ?? new List<MemberEntry>(),
            DeliverTick = Now + 1,
        });
    }

    public void Tick()
    {
        Now++;
        foreach (GossipNode node in Nodes)
        {
            if (node.Down) continue;
            DeclareDeathIfDue(node);
            SuspectIfPingTimedOut(node);
            if (Now % GossipInterval == 0) GossipRound(node);
            if (Now % PingInterval == 0) StartPing(node);
        }
        DeliverDue();
    }

    public void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Tick();
    }

    private void StartPing(GossipNode node)
    {
        if (node.PingTarget != -1) return;                       // one outstanding probe per node
        List<int> alive = node.View.AliveIds();
        alive.Remove(node.Id);
        if (alive.Count == 0) return;
        node.Cursor = (node.Cursor + 1) % alive.Count;
        node.PingTarget = alive[node.Cursor];
        node.PingSentTick = Now;
        Send(node.Id, node.PingTarget, "ping", node, null);
    }

    private void SuspectIfPingTimedOut(GossipNode node)
    {
        if (node.PingTarget == -1 || Now - node.PingSentTick < PingTimeout) return;
        int target = node.PingTarget;
        node.PingTarget = -1;
        node.PingTimeouts++;
        MemberEntry? entry = node.View.Get(target);
        if (entry == null || entry.Status != MemberStatus.Alive) return;
        ChangeView(node, target, entry.Incarnation, MemberStatus.Suspect);
        node.SuspectTarget = target;
        node.SuspectDeadline = Now + SuspicionTimeout;
    }

    private void DeclareDeathIfDue(GossipNode node)
    {
        if (node.SuspectTarget == -1 || Now < node.SuspectDeadline) return;
        int target = node.SuspectTarget;
        node.SuspectTarget = -1;
        MemberEntry? entry = node.View.Get(target);
        if (entry == null || entry.Status != MemberStatus.Suspect) return;
        ChangeView(node, target, entry.Incarnation, MemberStatus.Dead);
        node.DeadDeclarations++;
    }

    private void ChangeView(GossipNode node, int id, int incarnation, MemberStatus status)
    {
        if (node.View.Merge(id, incarnation, status)) ViewChanges++;
        else NoopMerges++;
    }

    private void GossipRound(GossipNode node)
    {
        // Gossip goes to known members even if they are currently suspect or dead: that is how a
        // node learns it has been declared dead and gets the chance to refute.
        List<int> targets = node.View.SortedIds();
        targets.Remove(node.Id);
        if (targets.Count == 0) return;
        List<MemberEntry> digest = node.View.Digest();
        for (int i = 0; i < Fanout && i < targets.Count; i++)
        {
            node.Seed = (node.Seed * 1103515245 + 12345) & 0x7FFFFFFF;
            int pick = node.Seed % targets.Count;
            Send(node.Id, targets[pick], "gossip", node, digest);
            node.GossipSends++;
        }
    }

    private void DeliverDue()
    {
        while (true)
        {
            int best = -1;
            for (int i = 0; i < queue.Count; i++)
            {
                if (queue[i].DeliverTick > Now) continue;
                if (best < 0 || queue[i].DeliverTick < queue[best].DeliverTick
                    || (queue[i].DeliverTick == queue[best].DeliverTick && queue[i].Seq < queue[best].Seq)) best = i;
            }
            if (best < 0) return;
            GossipMessage message = queue[best];
            queue.RemoveAt(best);
            GossipNode target = Node(message.To);
            if (target.Down) { MessagesDropped++; continue; }
            if (message.Kind == "ping")
            {
                // A ping also teaches the receiver that the sender exists.
                target.View.Merge(message.From, message.Incarnation, MemberStatus.Alive);
                Send(target.Id, message.From, "ack", target, null);
                continue;
            }
            if (message.Kind == "ack")
            {
                // The node that sent the ping is the addressee; its outstanding probe succeeded.
                if (target.PingTarget == message.From) target.PingTarget = -1;
                continue;
            }
            MergeDigest(target, message);
        }
    }

    private void MergeDigest(GossipNode node, GossipMessage message)
    {
        node.ReceivedGossip++;
        foreach (MemberEntry incoming in message.Digest)
        {
            if (incoming.Id == node.Id)
            {
                // A claim about myself: if it says I am suspect or dead, refute it loudly.
                if (incoming.Status != MemberStatus.Alive && incoming.Incarnation >= node.Incarnation)
                {
                    node.Incarnation++;
                    node.Refutations++;
                    ChangeView(node, node.Id, node.Incarnation, MemberStatus.Alive);
                }
                else
                {
                    ChangeView(node, node.Id, node.Incarnation, MemberStatus.Alive);
                }
                continue;
            }
            ChangeView(node, incoming.Id, incoming.Incarnation, incoming.Status);
        }
    }

    public bool ViewsAgree()
    {
        string reference = Nodes[0].View.Describe();
        foreach (GossipNode node in Nodes)
        {
            if (node.Down) continue;
            if (node.View.Describe() != reference) return false;
        }
        return true;
    }

    public bool AllAliveAndComplete()
    {
        foreach (GossipNode node in Nodes)
        {
            if (node.Down) continue;
            foreach (GossipNode other in Nodes)
            {
                if (other.Down) continue;
                if (!node.View.IsAlive(other.Id)) return false;
            }
            if (node.View.CountWith(MemberStatus.Dead) != 0 || node.View.CountWith(MemberStatus.Suspect) != 0) return false;
        }
        return true;
    }

    public int FirstConvergedTick(int limit)
    {
        for (int i = 0; i < limit; i++)
        {
            Tick();
            if (ViewsAgree()) return Now;
        }
        return -1;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void MergeIsIdempotentAndOrderIndependent()
    {
        var a = new MemberList();
        var b = new MemberList();
        a.Set(1, 1, MemberStatus.Alive);
        b.Set(1, 1, MemberStatus.Alive);
        var observations = new List<(int Id, int Inc, MemberStatus Status)>
        {
            (1, 1, MemberStatus.Suspect), (1, 2, MemberStatus.Alive), (2, 1, MemberStatus.Alive),
            (1, 1, MemberStatus.Dead), (2, 3, MemberStatus.Dead), (1, 2, MemberStatus.Suspect),
        };
        foreach ((int id, int inc, MemberStatus status) in observations) b.Merge(id, inc, status);
        for (int i = observations.Count - 1; i >= 0; i--) a.Merge(observations[i].Id, observations[i].Inc, observations[i].Status);
        Check(a.Describe() == b.Describe(), "merge order does not matter: " + a.Describe() + " vs " + b.Describe());
        int changesBefore = 0;
        foreach ((int id, int inc, MemberStatus status) in observations)
        {
            bool changed = a.Merge(id, inc, status);
            if (changed) changesBefore++;
        }
        Check(changesBefore == 0, "re-merging the same observations changes nothing (idempotence)");
        Check(a.Get(1)!.Status == MemberStatus.Suspect && a.Get(1)!.Incarnation == 2,
            "alive@2 beats dead@1 and suspect@2 wins the tie at incarnation 2");
        Check(!a.IsAlive(2), "node 2 was declared dead at incarnation 3");
        Console.WriteLine("merge: " + a.Describe());
    }

    private static void ViewsConvergeQuickly()
    {
        var sim = new GossipSim(6);
        int converged = sim.FirstConvergedTick(60);
        Check(converged > 0, "the views converged within 60 ticks");
        Check(converged <= 10, "convergence took " + converged + " ticks, expected the fan-out to be fast");
        Check(sim.AllAliveAndComplete(), "every node knows every other node as alive");
        Check(sim.Node(1).View.Describe() == sim.Node(5).View.Describe(), "view 1 and view 5 are identical");
        Check(sim.MessagesSent > 0 && sim.ViewChanges > 0, "gossip traffic flowed");

        // A larger cluster still converges in a logarithmic number of gossip rounds.
        var big = new GossipSim(24);
        int bigConverged = big.FirstConvergedTick(200);
        Check(bigConverged > 0 && bigConverged <= 24, "24 nodes converged in " + bigConverged + " ticks");
        Check(big.AllAliveAndComplete(), "all 24 nodes agree");
        Console.WriteLine("convergence: 6 nodes in " + converged + " ticks, 24 nodes in " + bigConverged + " ticks");
    }

    private static void CrashedNodeIsDeclaredDeadEverywhere()
    {
        var sim = new GossipSim(6);
        sim.Run(6);
        GossipNode crashed = sim.Node(6);
        crashed.Down = true;
        sim.Run(80);

        Check(sim.Node(1).View.Get(6)!.Status == MemberStatus.Dead, "node 1 declared node 6 dead");
        Check(sim.Node(4).View.Get(6)!.Status == MemberStatus.Dead, "node 4 declared node 6 dead too");
        for (int id = 1; id <= 5; id++) Check(sim.Node(id).View.CountWith(MemberStatus.Dead) == 1, "exactly one death known to node " + id);
        Check(sim.Node(1).PingTimeouts > 0, "the ping timeout fired at least once");
        Check(crashed.DeadDeclarations == 0, "the crashed node never declared anyone dead");
        Check(crashed.Refutations == 0, "a crashed node cannot refute");
        Check(sim.ViewsAgree(), "all live nodes agree on the membership");

        // The crashed node's membership is not part of the live set any more.
        Check(!sim.Node(1).View.IsAlive(6), "node 6 is excluded from the live set");
        Check(sim.Node(1).View.AliveIds().Count == 5, "five live members remain");

        // If it restarts with a fresh incarnation it can rejoin.
        crashed.Down = false;
        crashed.Incarnation = 5;
        crashed.View.Set(6, 5, MemberStatus.Alive);
        sim.Run(40);
        Check(sim.Node(1).View.IsAlive(6) && sim.Node(1).View.Get(6)!.Incarnation == 5,
            "the restarted node rejoined with a newer incarnation: " + sim.Node(1).View.Describe());
        Check(sim.AllAliveAndComplete(), "the whole cluster is healthy again");
        Console.WriteLine("crash detection: dropped=" + sim.MessagesDropped + " timeouts=" + sim.Node(1).PingTimeouts
            + " refutations=" + crashed.Refutations);
    }

    private static void SuspicionOfAnAliveNodeIsRefuted()
    {
        var sim = new GossipSim(5);
        sim.Run(6);
        sim.Partition(new[] { 3 }, new[] { 1, 2, 4, 5 });
        sim.Run(60);
        Check(sim.Node(1).View.Get(3)!.Status != MemberStatus.Alive, "the isolated node is suspected or dead");
        Check(sim.Node(3).Refutations == 0, "node 3 has not heard about it while isolated, so it cannot refute yet");

        sim.HealPartition(new[] { 3 }, new[] { 1, 2, 4, 5 });
        sim.Run(80);
        Check(sim.Node(3).Refutations > 0, "node 3 refuted the suspicion, refutations=" + sim.Node(3).Refutations);
        Check(sim.Node(1).View.IsAlive(3), "node 1 sees node 3 alive again: " + sim.Node(1).View.Describe());
        Check(sim.AllAliveAndComplete(), "every node is alive again in every view");
        Check(sim.ViewsAgree(), "views converged after the heal");
        Console.WriteLine("refutation: incarnations=" + sim.Node(1).View.Describe());
    }

    private static void PartitionedNodesAreExcludedWhileIsolated()
    {
        var sim = new GossipSim(6);
        sim.Run(6);
        sim.Partition(new[] { 1, 2, 3 }, new[] { 4, 5, 6 });
        sim.Run(40);
        Check(sim.Node(1).View.CountWith(MemberStatus.Dead) >= 1, "the far side is declared dead in this partition");
        Check(sim.Node(4).View.CountWith(MemberStatus.Dead) >= 1, "and vice versa");
        Check(sim.Node(1).View.Get(1)!.Status == MemberStatus.Alive, "a node never declares itself dead");
        Check(!sim.Node(1).View.IsAlive(4), "the other side is not in this side's live set");
        Check(sim.Node(1).View.IsAlive(2) && sim.Node(1).View.IsAlive(3), "the local side stays healthy");

        sim.HealPartition(new[] { 1, 2, 3 }, new[] { 4, 5, 6 });
        sim.Run(120);
        Check(sim.AllAliveAndComplete(), "after healing every node is alive again: " + sim.Node(1).View.Describe());
        Check(sim.ViewsAgree(), "and all six views are identical");
        int refutations = 0;
        foreach (GossipNode node in sim.Nodes) refutations += node.Refutations;
        Check(refutations > 0, "the false suspicions were refuted, count=" + refutations);
        Console.WriteLine("partition: refutations=" + refutations + " view=" + sim.Node(1).View.Describe());
    }

    public static void Main()
    {
        MergeIsIdempotentAndOrderIndependent();
        ViewsConvergeQuickly();
        CrashedNodeIsDeclaredDeadEverywhere();
        SuspicionOfAnAliveNodeIsRefuted();
        PartitionedNodesAreExcludedWhileIsolated();
        Console.WriteLine("PASS");
    }
}
