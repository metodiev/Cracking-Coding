// Task 686 - Implement BGP path-vector simplified policy simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A scaled-down but faithful BGP path-vector model. Every AS is a node holding a RIB
// (prefix -> best route) and a candidate table (prefix -> neighbour -> route advertised by that
// neighbour). Route propagation is event driven: a node recomputes its best route, compares the
// route it would advertise per neighbour with what it advertised last time, and only then
// enqueues an UPDATE (or a WITHDRAW) - exactly the incremental behaviour of a real speaker, so
// the simulation converges to the BGP fixed point instead of flooding.
//
// Semantics that are modelled:
//   * loop prevention: a route whose AS_PATH already contains the receiving AS is dropped, and
//     any route the same peer advertised for that prefix before is implicitly withdrawn;
//   * export prepend: the advertising AS prepends itself, so AS_PATH length == number of AS hops;
//   * import/export policy: per (neighbour, prefix) allow/deny lists, plus per-neighbour
//     LOCAL_PREF assigned by the receiving AS; both are re-evaluated at decision time, so a
//     policy change can withdraw a route that was already installed;
//   * best-path selection: self-originated first, then highest LOCAL_PREF, shortest AS_PATH,
//     lowest MED, then the lowest neighbour AS (standing in for the router-id tie-break).
//
// Documented simplifications: one prefix per simulated object (no aggregation), no iBGP/route
// reflection, no timers or MRAI, MED compared across neighbours (real BGP restricts that to the
// same neighbour AS), and no AS_SETs. What is verified: the converged RIB equals an
// independent fold over the candidate set, reachability matches a BFS hop count under
// permissive policy, every installed AS_PATH is duplicate-free and never contains its own AS,
// a policy change converges to exactly the same RIB as a fresh simulation with that policy from
// the start, and two runs produce byte-identical output.
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class BgpRoute
{
    public readonly List<int> AsPath;   // leftmost = AS that advertised it to us, rightmost = origin
    public readonly int LocalPref;
    public readonly int Med;
    public readonly int Neighbor;       // 0 for a self-originated route
    public readonly bool SelfOriginated;

    public BgpRoute(List<int> asPath, int localPref, int med, int neighbor, bool selfOriginated)
    {
        AsPath = new List<int>(asPath);
        LocalPref = localPref;
        Med = med;
        Neighbor = neighbor;
        SelfOriginated = selfOriginated;
    }

    public static BgpRoute Originated() => new BgpRoute(new List<int>(), 0, 0, 0, true);

    /// <summary>eBGP export: the sender prepends its own AS to the path it advertises.</summary>
    public BgpRoute Prepended(int asNumber)
    {
        var path = new List<int> { asNumber };
        path.AddRange(AsPath);
        return new BgpRoute(path, LocalPref, Med, Neighbor, false);
    }

    public bool SameAs(BgpRoute? other)
    {
        if (other == null) return false;
        if (AsPath.Count != other.AsPath.Count || Med != other.Med || SelfOriginated != other.SelfOriginated) return false;
        for (int i = 0; i < AsPath.Count; i++) if (AsPath[i] != other.AsPath[i]) return false;
        return true;
    }

    /// <summary>-1 when <paramref name="a"/> wins, +1 when <paramref name="b"/> wins, 0 on a tie.</summary>
    public static int Compare(BgpRoute a, BgpRoute b)
    {
        if (a.SelfOriginated != b.SelfOriginated) return a.SelfOriginated ? -1 : 1;
        if (a.LocalPref != b.LocalPref) return b.LocalPref.CompareTo(a.LocalPref);
        if (a.AsPath.Count != b.AsPath.Count) return a.AsPath.Count.CompareTo(b.AsPath.Count);
        if (a.Med != b.Med) return a.Med.CompareTo(b.Med);
        if (a.Neighbor != b.Neighbor) return a.Neighbor.CompareTo(b.Neighbor);
        return 0;
    }

    /// <summary>Independent oracle: the same decision expressed as one lexicographic key.</summary>
    public static ValueTuple<int, int, int, int, int> Key(BgpRoute r) =>
        ValueTuple.Create(r.SelfOriginated ? 0 : 1, -r.LocalPref, r.AsPath.Count, r.Med, r.Neighbor);

    public static int CompareViaKey(BgpRoute a, BgpRoute b)
    {
        var ka = Key(a);
        var kb = Key(b);
        return ka.CompareTo(kb);
    }

    public string Describe() =>
        $"{LocalPref}/{string.Join(",", AsPath)}/{Med}/nh{Neighbor}{(SelfOriginated ? "/self" : "")}";
}

internal sealed class BgpNode
{
    public readonly int Id;
    public readonly List<int> Peers = new List<int>();
    public readonly SortedDictionary<int, BgpRoute> Rib = new SortedDictionary<int, BgpRoute>();
    public readonly SortedDictionary<int, SortedDictionary<int, BgpRoute>> Candidates =
        new SortedDictionary<int, SortedDictionary<int, BgpRoute>>();
    public readonly SortedDictionary<int, int> LocalPref = new SortedDictionary<int, int>();
    public readonly List<int> Originated = new List<int>();
    public readonly HashSet<(int, int)> ImportDenied = new HashSet<(int, int)>();
    public readonly HashSet<(int, int)> ExportDenied = new HashSet<(int, int)>();

    public BgpNode(int id) => Id = id;

    public void Peer(int other)
    {
        if (!Peers.Contains(other)) Peers.Add(other);
        Peers.Sort();
    }

    public int LocalPrefFrom(int peer) => LocalPref.TryGetValue(peer, out int value) ? value : 100;

    public bool ImportAllowed(int peer, int prefix) => !ImportDenied.Contains((peer, prefix));

    public bool ExportAllowed(int peer, int prefix) => !ExportDenied.Contains((peer, prefix));

    public SortedDictionary<int, BgpRoute> CandidatesFor(int prefix)
    {
        if (!Candidates.TryGetValue(prefix, out var table))
        {
            table = new SortedDictionary<int, BgpRoute>();
            Candidates[prefix] = table;
        }
        return table;
    }

    /// <summary>Prefixes this node can possibly hold: its own, everything offered, everything installed.</summary>
    public List<int> RelevantPrefixes()
    {
        var set = new SortedSet<int>(Originated);
        foreach (int prefix in Candidates.Keys) set.Add(prefix);
        foreach (int prefix in Rib.Keys) set.Add(prefix);
        return new List<int>(set);
    }
}

internal sealed class BgpSimulation
{
    public const int StepCap = 200000;

    private sealed class Message
    {
        public readonly int To;
        public readonly int From;
        public readonly int Prefix;
        public readonly BgpRoute? Route;   // null == withdraw

        public Message(int to, int from, int prefix, BgpRoute? route)
        {
            To = to; From = from; Prefix = prefix; Route = route;
        }
    }

    private readonly SortedDictionary<int, BgpNode> _nodes = new SortedDictionary<int, BgpNode>();
    private readonly Queue<Message> _deliver = new Queue<Message>();
    private readonly Queue<int> _recompute = new Queue<int>();
    private readonly HashSet<int> _pending = new HashSet<int>();
    private readonly Dictionary<(int, int, int), BgpRoute?> _lastSent = new Dictionary<(int, int, int), BgpRoute?>();

    public long Steps;
    public long Recomputes;
    public long LoopDrops;
    public long PolicyDrops;
    public long UpdatesSent;
    public long WithdrawsSent;
    public bool HitStepCap;

    public BgpNode Node(int id)
    {
        if (!_nodes.TryGetValue(id, out var node))
        {
            node = new BgpNode(id);
            _nodes[id] = node;
        }
        return node;
    }

    public IEnumerable<BgpNode> AllNodes() => _nodes.Values;

    public void Link(int a, int b)
    {
        Node(a).Peer(b);
        Node(b).Peer(a);
    }

    public void Originate(int asNumber, int prefix) => Node(asNumber).Originated.Add(prefix);

    public void RequestRecompute(int node)
    {
        if (_pending.Add(node)) _recompute.Enqueue(node);
    }

    public void Converge()
    {
        Steps = 0;
        Recomputes = 0;
        HitStepCap = false;
        foreach (int id in new List<int>(_nodes.Keys)) RequestRecompute(id);
        while (_deliver.Count > 0 || _recompute.Count > 0)
        {
            if (Steps > StepCap) { HitStepCap = true; return; }
            if (_deliver.Count > 0) { Steps++; Deliver(_deliver.Dequeue()); }
            else { Recomputes++; ProcessRecompute(); }
        }
    }

    private void ProcessRecompute()
    {
        int id = _recompute.Dequeue();
        _pending.Remove(id);
        Recompute(_nodes[id]);
    }

    private void Deliver(Message message)
    {
        BgpNode receiver = _nodes[message.To];
        var table = receiver.CandidatesFor(message.Prefix);
        if (message.Route == null)
        {
            if (table.Remove(message.From)) Recompute(receiver);
            return;
        }
        if (message.Route.AsPath.Contains(receiver.Id))
        {
            // Loop prevention. An UPDATE also implicitly withdraws whatever this peer advertised
            // before for the same prefix, so the stale candidate has to go.
            LoopDrops++;
            if (table.Remove(message.From)) Recompute(receiver);
            return;
        }
        if (!receiver.ImportAllowed(message.From, message.Prefix))
        {
            PolicyDrops++;
            if (table.Remove(message.From)) Recompute(receiver);
            return;
        }
        table[message.From] = message.Route;   // LOCAL_PREF is applied at decision time, not at reception
        Recompute(receiver);
    }

    /// <summary>Recomputes every relevant prefix, then pushes only the changes to the neighbours.</summary>
    public void Recompute(BgpNode node)
    {
        foreach (int prefix in node.RelevantPrefixes())
        {
            BgpRoute? best = BestRoute(node, prefix);
            if (best == null) node.Rib.Remove(prefix);
            else node.Rib[prefix] = best;

            foreach (int peer in node.Peers)
            {
                BgpRoute? advertised = null;
                if (best != null && node.ExportAllowed(peer, prefix)) advertised = best.Prepended(node.Id);
                var key = (node.Id, peer, prefix);
                BgpRoute? previous = _lastSent.TryGetValue(key, out var sent) ? sent : null;
                if (SameAdvert(previous, advertised)) continue;
                _lastSent[key] = advertised;
                if (advertised == null) WithdrawsSent++; else UpdatesSent++;
                _deliver.Enqueue(new Message(peer, node.Id, prefix, advertised));
            }
        }
    }

    private static bool SameAdvert(BgpRoute? a, BgpRoute? b)
    {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;
        return a.SameAs(b);
    }

    /// <summary>Applies the receiver's current LOCAL_PREF policy to a route held in the table.</summary>
    public static BgpRoute Effective(BgpNode node, int neighbor, BgpRoute received) =>
        new BgpRoute(received.AsPath, node.LocalPrefFrom(neighbor), received.Med, neighbor, false);

    /// <summary>The node's decision process, driven by <see cref="BgpRoute.Compare"/>.</summary>
    public BgpRoute? BestRoute(BgpNode node, int prefix)
    {
        if (node.Originated.Contains(prefix)) return BgpRoute.Originated();
        BgpRoute? best = null;
        if (node.Candidates.TryGetValue(prefix, out var table))
        {
            foreach (var pair in table)
            {
                if (!node.ImportAllowed(pair.Key, prefix)) continue;   // policy is re-evaluated on every decision
                BgpRoute candidate = Effective(node, pair.Key, pair.Value);
                if (best == null || BgpRoute.Compare(candidate, best) < 0) best = candidate;
            }
        }
        return best;
    }

    /// <summary>Oracle: full fold over the candidate set using the independent tuple key.</summary>
    public static BgpRoute? BestRouteViaKey(BgpNode node, int prefix)
    {
        if (node.Originated.Contains(prefix)) return BgpRoute.Originated();
        BgpRoute? best = null;
        if (!node.Candidates.TryGetValue(prefix, out var table)) return null;
        foreach (var pair in table)
        {
            if (!node.ImportAllowed(pair.Key, prefix)) continue;
            BgpRoute candidate = Effective(node, pair.Key, pair.Value);
            if (best == null || BgpRoute.CompareViaKey(candidate, best) < 0) best = candidate;
        }
        return best;
    }

    public string Dump()
    {
        var sb = new StringBuilder();
        foreach (var node in _nodes)
        {
            sb.Append("AS").Append(node.Key).Append(':');
            foreach (var entry in node.Value.Rib)
                sb.Append(' ').Append(entry.Key.ToString("X8")).Append('=').Append(entry.Value.Describe());
            sb.AppendLine();
        }
        return sb.ToString();
    }
}

internal sealed class BgpTopology
{
    public static readonly int[] Nodes = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    public static readonly (int, int)[] Links =
    {
        (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (4, 8), (2, 6), (8, 9),
    };

    public const int PrefixA = 0x0A000000;              // 10.0.0.0/8, originated by AS 1
    public const int PrefixB = unchecked((int)0xAC100000);  // 172.16.0.0/12, originated by AS 8

    public static BgpSimulation BuildRaw()
    {
        var sim = new BgpSimulation();
        foreach (int id in Nodes) sim.Node(id);
        foreach ((int a, int b) in Links) sim.Link(a, b);
        sim.Originate(1, PrefixA);
        sim.Originate(8, PrefixB);
        return sim;
    }

    public static BgpSimulation Build()
    {
        var sim = BuildRaw();
        sim.Converge();
        return sim;
    }
}

public static class Solution
{
    public static void Main()
    {
        var sim = BgpTopology.Build();
        Check(!sim.HitStepCap, "simulation reached the fixed point well inside the step cap");
        Check(sim.Steps < 5000, "incremental updates stayed bounded, steps=" + sim.Steps);
        Check(sim.LoopDrops > 0, "loop prevention actually fired during convergence");

        // The origin keeps its own prefix as a self-originated route.
        BgpNode origin = sim.Node(1);
        Check(origin.Rib[BgpTopology.PrefixA].SelfOriginated, "AS1 holds its own prefix locally");
        Check(origin.Rib[BgpTopology.PrefixA].AsPath.Count == 0, "self-originated path is empty");
        Check(origin.Rib.ContainsKey(BgpTopology.PrefixB) && !origin.Rib[BgpTopology.PrefixB].SelfOriginated,
            "AS1 learns AS8's prefix instead of claiming it locally");

        // Reachability: with the permissive default policy the hop count must equal BFS distance.
        int[] distances = Bfs(1);
        for (int id = 2; id <= 9; id++)
        {
            if (!sim.Node(id).Rib.TryGetValue(BgpTopology.PrefixA, out var route))
                throw new Exception($"FAILED: AS{id} lost reachability");
            Check(route.AsPath.Count == distances[id], $"AS{id} path length matches BFS distance {distances[id]}");
            Check(route.AsPath[0] == NearestHop(1, id), $"AS{id} learned from its BFS parent");
        }
        Check(sim.Node(9).Rib.ContainsKey(BgpTopology.PrefixB), "AS9 also holds the second prefix");

        // Structural invariants on every installed path.
        foreach (BgpNode node in sim.AllNodes())
        {
            foreach (var entry in node.Rib)
            {
                if (entry.Value.SelfOriginated) continue;
                var seen = new HashSet<int>();
                foreach (int asNumber in entry.Value.AsPath)
                    Check(seen.Add(asNumber), $"AS{node.Id} path {entry.Value.Describe()} visits an AS twice");
                Check(!entry.Value.AsPath.Contains(node.Id), $"AS{node.Id} installed a path containing itself");
            }
        }

        // The incremental decision must equal an independent full recomputation.
        foreach (BgpNode node in sim.AllNodes())
        {
            foreach (int prefix in node.RelevantPrefixes())
            {
                BgpRoute? mine = node.Rib.TryGetValue(prefix, out var installed) ? installed : null;
                BgpRoute? oracle = BgpSimulation.BestRouteViaKey(node, prefix);
                Check((mine == null) == (oracle == null), $"AS{node.Id} presence mismatch for prefix {prefix:X8}");
                if (mine != null) Check(mine.SameAs(oracle!), $"AS{node.Id} choice mismatch for prefix {prefix:X8}");
            }
        }

        // Fixed point: a second convergence pass must not process or produce a single message.
        string before = sim.Dump();
        long updatesBefore = sim.UpdatesSent, withdrawsBefore = sim.WithdrawsSent;
        sim.Converge();
        Check(sim.Steps == 0, "second pass processes no UPDATE messages");
        Check(sim.UpdatesSent == updatesBefore && sim.WithdrawsSent == withdrawsBefore,
            "second pass sends no UPDATE/WITHDRAW messages");
        Check(sim.Dump() == before, "the fixed point RIB is unchanged by a second pass");
        Check(updatesBefore > 0 && sim.Recomputes > 0, "the first pass did send updates and recomputed");

        // Determinism: a fresh simulation produces byte-identical RIBs and message counts.
        var again = BgpTopology.Build();
        Check(again.Dump() == before, "two runs agree on the converged RIBs");
        Check(again.UpdatesSent == sim.UpdatesSent && again.WithdrawsSent == sim.WithdrawsSent,
            "two runs send the same number of UPDATE/WITHDRAW messages");

        // LOCAL_PREF beats AS_PATH length: persuade AS8 to reach prefix A through its longer path.
        var policy = BgpTopology.Build();
        policy.Node(8).LocalPref[7] = 300;
        policy.RequestRecompute(8);
        policy.Converge();
        var viaSeven = policy.Node(8).Rib[BgpTopology.PrefixA];
        Check(viaSeven.Neighbor == 7, "higher LOCAL_PREF selects the neighbour AS7");
        Check(viaSeven.AsPath.Count == 4, "the selected path is longer than the BFS distance 3");
        Check(viaSeven.LocalPref == 300, "the winning route carries the configured LOCAL_PREF");
        Check(policy.Node(9).Rib[BgpTopology.PrefixA].Neighbor == 8, "AS9 still prefers its only neighbour");

        // MED and neighbour-id tie-breaks.
        var cheap = new BgpRoute(new List<int> { 7, 1 }, 100, 5, 7, false);
        var dear = new BgpRoute(new List<int> { 6, 1 }, 100, 50, 6, false);
        Check(BgpRoute.Compare(cheap, dear) < 0, "lower MED wins on equal LOCAL_PREF and path length");
        var lowId = new BgpRoute(new List<int> { 6, 1 }, 100, 5, 6, false);
        Check(BgpRoute.Compare(lowId, cheap) < 0, "lower neighbour AS breaks a full tie");
        Check(BgpRoute.Compare(cheap, cheap) == 0, "identical routes tie");
        Check(BgpRoute.CompareViaKey(cheap, dear) < 0 && BgpRoute.CompareViaKey(dear, cheap) > 0,
            "the tuple-key oracle agrees on the MED comparison");
        Check(BgpRoute.CompareViaKey(lowId, cheap) < 0, "the tuple-key oracle agrees on the tie-break");
        Check(BgpRoute.Compare(BgpRoute.Originated(), cheap) < 0, "a self-originated route outranks any learned route");

        // Folding a whole candidate set must produce the same winner as the per-pair comparator.
        var chooser = new BgpNode(42);
        var chooserTable = chooser.CandidatesFor(BgpTopology.PrefixA);
        chooserTable[6] = dear;
        chooserTable[7] = cheap;
        var chooserSim = new BgpSimulation();
        chooserSim.Node(42);
        Check(chooserSim.BestRoute(chooser, BgpTopology.PrefixA)!.Neighbor == 7, "candidate fold picks the low-MED route");
        chooserTable[6] = lowId;
        Check(chooserSim.BestRoute(chooser, BgpTopology.PrefixA)!.Neighbor == 6, "candidate fold then applies the AS tie-break");
        Check(BgpSimulation.BestRouteViaKey(chooser, BgpTopology.PrefixA)!.Neighbor == 6, "the oracle fold agrees");
        Check(chooserSim.BestRoute(chooser, BgpTopology.PrefixB) == null, "no candidates means no best route");

        // Import policy: AS9 denies the prefix from its only neighbour and therefore loses the
        // route entirely, while the rest of the network keeps the alternative paths.
        var filtered = BgpTopology.Build();
        filtered.Node(9).ImportDenied.Add((8, BgpTopology.PrefixA));
        filtered.RequestRecompute(9);
        filtered.Converge();
        Check(!filtered.Node(9).Rib.ContainsKey(BgpTopology.PrefixA), "denied import leaves AS9 without the route");
        Check(filtered.Node(5).Rib.ContainsKey(BgpTopology.PrefixA), "AS5 keeps its route via AS6");
        Check(filtered.Node(8).Rib.ContainsKey(BgpTopology.PrefixA), "the neighbour that was denied is unaffected");

        // The same policy applied before the first UPDATE drops the message at reception instead.
        var deniedFromStart = BgpTopology.BuildRaw();
        deniedFromStart.Node(9).ImportDenied.Add((8, BgpTopology.PrefixA));
        deniedFromStart.Converge();
        Check(deniedFromStart.PolicyDrops > 0, "the import denial was counted when the UPDATE arrived");
        Check(deniedFromStart.Dump() == filtered.Dump(), "an up-front denial converges like a post-hoc one");

        // Export policy + withdrawal: denying AS1's export to AS2 must converge to exactly the same
        // RIB as a simulation built with that denial in place from the start.
        var denied = BgpTopology.Build();
        denied.Node(1).ExportDenied.Add((2, BgpTopology.PrefixA));
        denied.RequestRecompute(1);
        denied.Converge();
        var fromScratch = BgpTopology.BuildRaw();
        fromScratch.Node(1).ExportDenied.Add((2, BgpTopology.PrefixA));
        fromScratch.Converge();
        Check(denied.Dump() == fromScratch.Dump(), "revoked export converges to the fresh-simulation RIB");
        Check(denied.WithdrawsSent > 0, "the revocation produced withdraws");
        Check(denied.Node(5).Rib.ContainsKey(BgpTopology.PrefixA), "AS5 is still reachable through AS3");

        // Withdrawing the last remaining source removes the prefix from everyone.
        var starved = BgpTopology.Build();
        starved.Node(1).ExportDenied.Add((2, BgpTopology.PrefixA));
        starved.Node(1).ExportDenied.Add((3, BgpTopology.PrefixA));
        starved.RequestRecompute(1);
        starved.Converge();
        for (int id = 2; id <= 9; id++)
            Check(!starved.Node(id).Rib.ContainsKey(BgpTopology.PrefixA), $"AS{id} lost the unadvertised prefix");
        Check(starved.Node(1).Rib.ContainsKey(BgpTopology.PrefixA), "the origin still owns its prefix");

        Console.WriteLine($"steps={sim.Steps} updates={sim.UpdatesSent} withdraws={sim.WithdrawsSent} " +
                          $"loopdrops={sim.LoopDrops} policydrops={sim.PolicyDrops}");
        Console.WriteLine("PASS");
    }

    private static int[] Bfs(int origin)
    {
        var distance = new int[BgpTopology.Nodes.Length + 1];
        for (int i = 0; i < distance.Length; i++) distance[i] = -1;
        distance[origin] = 0;
        var queue = new Queue<int>();
        queue.Enqueue(origin);
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            foreach ((int a, int b) in BgpTopology.Links)
            {
                int next = a == current ? b : b == current ? a : -1;
                if (next < 0 || distance[next] >= 0) continue;
                distance[next] = distance[current] + 1;
                queue.Enqueue(next);
            }
        }
        return distance;
    }

    /// <summary>The smallest neighbour of the BFS predecessor chain, matching the tie-break rule.</summary>
    private static int NearestHop(int origin, int target)
    {
        if (target == origin) return 0;
        var distance = Bfs(origin);
        int best = -1;
        foreach (int id in BgpTopology.Nodes)
        {
            if (distance[id] != distance[target] - 1) continue;
            bool adjacent = false;
            foreach ((int a, int b) in BgpTopology.Links)
                if ((a == id && b == target) || (b == id && a == target)) adjacent = true;
            if (!adjacent) continue;
            if (best < 0 || id < best) best = id;
        }
        return best;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
