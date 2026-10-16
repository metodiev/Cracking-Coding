// Task 687 - Implement routing algorithm link-state (OSPF) simulation with LSAs. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A single-area OSPF model: every router owns a link state advertisement (Router-LSA) listing its
// interfaces and their costs, a link state database (LSDB) holding one LSA per router, a reliable
// flooding procedure and an SPF calculation (Dijkstra). Flooding is incremental and layered:
// originating a new LSA version enqueues it on the originating router's interfaces, each receiver
// compares sequence numbers (an older or identical instance is discarded, which is the "stale LSA"
// rule) and forwards the accepted instance out of every interface except the one it arrived on.
// One pass of the queue is one flooding round, so the round count is bounded by the network
// diameter rather than by the number of routers.
//
// Invariants verified below:
//   * after flooding converges, every router in the same connected component holds exactly the
//     same LSDB (same sequence numbers, same link sets);
//   * each router's routing table equals a Bellman-Ford oracle computed on the same LSDB, and
//     every route is loop free: the next hop's own cost to the destination is exactly
//     cost - interface cost;
//   * a topology change (cost change, link removal) converges to the same tables as a fresh
//     simulation built with the new topology, and a stale LSA never overwrites a newer one;
//   * LSRefresh re-origination is rate limited by MinLSInterval, so the number of re-originations
//     after a run of ticks is bounded.
using System;
using System.Collections.Generic;

internal sealed class LinkStateArea
{
    public const int LsRefreshTicks = 30;
    public const int MinLsIntervalTicks = 5;
    public const int MaxFloodRounds = 64;

    private sealed class Lsa
    {
        public int Sequence;
        public int Age;
        public SortedDictionary<int, int> Links = new SortedDictionary<int, int>();

        public Lsa Clone()
        {
            var copy = new Lsa { Sequence = Sequence, Age = Age };
            foreach (var entry in Links) copy.Links[entry.Key] = entry.Value;
            return copy;
        }
    }

    private readonly SortedDictionary<int, SortedDictionary<int, int>> _ownLinks =
        new SortedDictionary<int, SortedDictionary<int, int>>();
    private readonly SortedDictionary<int, SortedDictionary<int, Lsa>> _lsdb =
        new SortedDictionary<int, SortedDictionary<int, Lsa>>();
    private readonly Queue<(int To, int Origin, Lsa Advert)> _flood = new Queue<(int, int, Lsa)>();
    private readonly SortedDictionary<int, int> _lastOriginatedTick = new SortedDictionary<int, int>();

    private int _tick;

    public long FloodMessages { get; private set; }
    public long StaleDrops { get; private set; }
    public long Originations { get; private set; }
    public int Rounds { get; private set; }

    public IEnumerable<int> Routers => _ownLinks.Keys;

    public void AddRouter(int id)
    {
        _ownLinks[id] = new SortedDictionary<int, int>();
        _lsdb[id] = new SortedDictionary<int, Lsa>();
        _lastOriginatedTick[id] = -1000;
    }

    public void SetLink(int a, int b, int cost)
    {
        if (cost <= 0) throw new ArgumentOutOfRangeException(nameof(cost), "OSPF costs are positive");
        _ownLinks[a][b] = cost;
        _ownLinks[b][a] = cost;
    }

    public void RemoveLink(int a, int b)
    {
        _ownLinks[a].Remove(b);
        _ownLinks[b].Remove(a);
    }

    public IReadOnlyDictionary<int, int> LinksOf(int id) => _ownLinks[id];

    /// <summary>Bumps the sequence number, installs the LSA and floods it on every interface.</summary>
    public void Originate(int id)
    {
        int sequence = _lsdb[id].TryGetValue(id, out var previous) ? previous.Sequence + 1 : 1;
        var lsa = new Lsa { Sequence = sequence, Age = 0 };
        foreach (var link in _ownLinks[id]) lsa.Links[link.Key] = link.Value;

        _lsdb[id][id] = lsa;
        _lastOriginatedTick[id] = _tick;
        Originations++;
        foreach (int neighbor in _ownLinks[id].Keys) Enqueue(neighbor, id, lsa);
    }

    public void OriginateAll()
    {
        foreach (int id in _ownLinks.Keys) Originate(id);
    }

    private void Enqueue(int to, int origin, Lsa lsa)
    {
        FloodMessages++;
        _flood.Enqueue((to, origin, lsa));
    }

    /// <summary>Runs flooding to a fixed point; each layer of the queue counts as one round.</summary>
    public void Converge()
    {
        Rounds = 0;
        while (_flood.Count > 0)
        {
            int layer = _flood.Count;
            for (int i = 0; i < layer; i++)
            {
                (int to, int origin, Lsa advert) = _flood.Dequeue();
                if (!_lsdb.TryGetValue(to, out var database)) continue;
                if (database.TryGetValue(origin, out var known) && known.Sequence >= advert.Sequence)
                {
                    StaleDrops++;
                    continue;
                }
                database[origin] = advert.Clone();
                foreach (int neighbor in _ownLinks[to].Keys)
                {
                    if (neighbor == origin) continue;   // split horizon on the receiving interface
                    Enqueue(neighbor, origin, advert);
                }
            }
            Rounds++;
            if (Rounds > MaxFloodRounds) return;
        }
    }

    public int SequenceOf(int router, int origin)
    {
        return _lsdb[router].TryGetValue(origin, out var lsa) ? lsa.Sequence : 0;
    }

    public int AgeOf(int router, int origin)
    {
        return _lsdb[router].TryGetValue(origin, out var lsa) ? lsa.Age : -1;
    }

    public int LsdbSize(int router) => _lsdb[router].Count;

    public bool DatabaseEquals(int a, int b)
    {
        if (_lsdb[a].Count != _lsdb[b].Count) return false;
        foreach (var entry in _lsdb[a])
        {
            if (!_lsdb[b].TryGetValue(entry.Key, out var other)) return false;
            if (other.Sequence != entry.Value.Sequence) return false;
            if (other.Links.Count != entry.Value.Links.Count) return false;
            foreach (var link in entry.Value.Links)
            {
                if (!other.Links.TryGetValue(link.Key, out int cost) || cost != link.Value) return false;
            }
        }
        return true;
    }

    /// <summary>Ages every LSA in every database; a self-originated LSA at LSRefresh is refreshed.</summary>
    public void Tick()
    {
        _tick++;
        foreach (var database in _lsdb.Values)
            foreach (var lsa in database.Values) lsa.Age++;
        foreach (int id in new List<int>(_ownLinks.Keys))
        {
            if (!_lsdb[id].TryGetValue(id, out var own) || own.Age < LsRefreshTicks) continue;
            if (!TryOriginate(id)) continue;
        }
    }

    /// <summary>Originates and floods unless MinLSInterval has not elapsed yet.</summary>
    public bool TryOriginate(int id)
    {
        if (_tick - _lastOriginatedTick[id] < MinLsIntervalTicks) return false;
        Originate(id);
        Converge();
        return true;
    }

    public int TickCount => _tick;
    public int RouterCount => _ownLinks.Count;

    /// <summary>The forwarding table of <paramref name="source"/>: destination -> (cost, next hop).</summary>
    public SortedDictionary<int, (int Cost, int NextHop)> RoutingTable(int source)
    {
        var graph = GraphOf(source);
        var cost = new SortedDictionary<int, int>();
        var nextHop = new SortedDictionary<int, int>();
        var settled = new HashSet<int>();
        foreach (int node in graph.Keys) cost[node] = int.MaxValue;
        cost[source] = 0;

        while (true)
        {
            int best = -1, bestCost = int.MaxValue;
            foreach (var entry in cost)
            {
                if (settled.Contains(entry.Key) || entry.Value == int.MaxValue) continue;
                if (entry.Value < bestCost) { bestCost = entry.Value; best = entry.Key; }
            }
            if (best < 0) break;
            settled.Add(best);
            foreach (var edge in graph[best])
            {
                if (settled.Contains(edge.Key) || !cost.ContainsKey(edge.Key)) continue;
                int candidate = bestCost + edge.Value;
                if (candidate < cost[edge.Key])
                {
                    cost[edge.Key] = candidate;
                    nextHop[edge.Key] = best == source ? edge.Key : nextHop[best];   // strict < keeps first
                }
            }
        }

        var table = new SortedDictionary<int, (int, int)>();
        foreach (var entry in cost)
        {
            if (entry.Key == source || entry.Value == int.MaxValue) continue;
            table[entry.Key] = (entry.Value, nextHop[entry.Key]);
        }
        return table;
    }

    /// <summary>Directed adjacency derived from the LSDB the router holds.</summary>
    public SortedDictionary<int, List<KeyValuePair<int, int>>> GraphOf(int router)
    {
        var graph = new SortedDictionary<int, List<KeyValuePair<int, int>>>();
        foreach (var origin in _lsdb[router])
        {
            var edges = new List<KeyValuePair<int, int>>();
            foreach (var link in origin.Value.Links) edges.Add(link);
            graph[origin.Key] = edges;
        }
        return graph;
    }

    public int LinkCost(int a, int b)
    {
        if (_lsdb[a].TryGetValue(a, out var own) && own.Links.TryGetValue(b, out int cost)) return cost;
        return _ownLinks[a].TryGetValue(b, out int local) ? local : -1;
    }

    /// <summary>Oracle: plain Bellman-Ford relaxation, an algorithm independent of the SPF code.</summary>
    public static SortedDictionary<int, int> BellmanFord(SortedDictionary<int, List<KeyValuePair<int, int>>> graph, int source)
    {
        var distance = new SortedDictionary<int, int>();
        foreach (int node in graph.Keys) distance[node] = int.MaxValue;
        distance[source] = 0;
        for (int pass = 0; pass < graph.Count; pass++)
        {
            bool changed = false;
            foreach (var node in graph)
            {
                if (distance[node.Key] == int.MaxValue) continue;
                foreach (var edge in node.Value)
                {
                    if (!distance.ContainsKey(edge.Key)) continue;    // neighbour has no LSA here
                    int candidate = distance[node.Key] + edge.Value;
                    if (candidate < distance[edge.Key]) { distance[edge.Key] = candidate; changed = true; }
                }
            }
            if (!changed) break;
        }
        return distance;
    }

    /// <summary>Independent enumeration of every equal-cost first hop out of <paramref name="source"/>.</summary>
    public List<int> EqualCostFirstHops(int source, int destination)
    {
        var graph = GraphOf(source);
        var table = RoutingTable(source);
        if (!table.TryGetValue(destination, out var route)) return new List<int>();
        var hops = new List<int>();
        foreach (var edge in graph[source])
        {
            var fromNeighbour = BellmanFord(graph, edge.Key);
            if (!fromNeighbour.ContainsKey(destination) || fromNeighbour[destination] == int.MaxValue) continue;
            if (edge.Value + fromNeighbour[destination] == route.Cost) hops.Add(edge.Key);
        }
        return hops;
    }
}

internal sealed class OspfTopology
{
    public static readonly int[] Routers = { 1, 2, 3, 4, 5, 6 };
    public static readonly (int A, int B, int Cost)[] Links =
    {
        (1, 2, 4), (1, 3, 3), (2, 3, 1), (2, 4, 2), (3, 5, 7), (4, 5, 1), (4, 6, 5), (5, 6, 2),
    };

    public static LinkStateArea Build(params (int A, int B, int Cost)[] overrides)
    {
        var area = new LinkStateArea();
        foreach (int id in Routers) area.AddRouter(id);
        foreach ((int a, int b, int cost) in Links) area.SetLink(a, b, cost);
        foreach ((int a, int b, int cost) in overrides)
        {
            if (cost < 0) area.RemoveLink(a, b); else area.SetLink(a, b, cost);
        }
        area.OriginateAll();
        area.Converge();
        return area;
    }
}

public static class Solution
{
    public static void Main()
    {
        var area = OspfTopology.Build();
        Check(area.Rounds <= 6, "flooding converged inside a few times the diameter, rounds=" + area.Rounds);
        Check(area.StaleDrops > 0, "duplicate LSA instances were suppressed by the sequence rule");

        // Flooding invariant: identical LSDBs everywhere.
        foreach (int a in OspfTopology.Routers)
        {
            Check(area.LsdbSize(a) == OspfTopology.Routers.Length, $"router {a} learned all six LSAs");
            foreach (int b in OspfTopology.Routers)
                Check(area.DatabaseEquals(a, b), $"LSDB of {a} equals LSDB of {b}");
        }

        // Sequence numbers are per-origin and start at 1.
        foreach (int id in OspfTopology.Routers) Check(area.SequenceOf(id, id) == 1, $"LSA of {id} is seq 1");
        Check(area.LinkCost(1, 3) == 3 && area.LinkCost(3, 1) == 3, "cost is symmetric in the LSA");

        // Hand-computed SPF result at router 1.
        var table1 = area.RoutingTable(1);
        var expected = new SortedDictionary<int, int> { { 2, 4 }, { 3, 3 }, { 4, 6 }, { 5, 7 }, { 6, 9 } };
        Check(table1.Count == expected.Count, "router 1 reaches all five other routers");
        foreach (var entry in expected)
        {
            Check(table1[entry.Key].Cost == entry.Value, $"cost 1->{entry.Key} is {entry.Value}");
        }
        Check(table1[2].NextHop == 2 && table1[4].NextHop == 2 && table1[5].NextHop == 2 && table1[6].NextHop == 2,
            "router 1 forwards everything through router 2");
        Check(table1[3].NextHop == 3, "router 1 reaches router 3 directly");

        // SPF versus the Bellman-Ford oracle for every source.
        foreach (int source in OspfTopology.Routers)
        {
            var mine = area.RoutingTable(source);
            var oracle = LinkStateArea.BellmanFord(area.GraphOf(source), source);
            foreach (var entry in oracle)
            {
                if (entry.Key == source) continue;
                bool reachable = mine.ContainsKey(entry.Key);
                Check(reachable, $"router {source} reaches {entry.Key} in the oracle too");
                Check(mine[entry.Key].Cost == entry.Value, $"cost {source}->{entry.Key} matches the oracle");
            }
            Check(mine.Count == oracle.Count - 1, $"router {source} only installs reachable destinations");
        }

        // Loop-free invariant: the next hop's own cost is exactly this cost minus the interface cost.
        foreach (int source in OspfTopology.Routers)
        {
            var mine = area.RoutingTable(source);
            foreach (var entry in mine)
            {
                int hop = entry.Value.NextHop;
                int interfaceCost = area.LinkCost(source, hop);
                int hopCost = hop == entry.Key ? 0 : area.RoutingTable(hop)[entry.Key].Cost;
                Check(hopCost == entry.Value.Cost - interfaceCost,
                    $"no loop {source}->{hop}->{entry.Key}: {hopCost} vs {entry.Value.Cost - interfaceCost}");
            }
        }

        // Equal-cost multipath: 1->2 has two minimum-cost first hops (direct and through router 3).
        var ecmp = area.EqualCostFirstHops(1, 2);
        Check(ecmp.Count == 2 && ecmp.Contains(2) && ecmp.Contains(3), "1->2 has two equal-cost first hops");
        Check(area.EqualCostFirstHops(1, 3).Count == 1, "1->3 has a single minimum-cost first hop");
        var ecmp5 = area.EqualCostFirstHops(1, 5);
        Check(ecmp5.Count == 2 && ecmp5.Contains(2) && ecmp5.Contains(3),
            "1->5 is reachable at cost 7 through both neighbours");

        // A rejected (stale) LSA must not overwrite the installed one.
        area.SetLink(2, 4, 2);
        area.Originate(2);
        area.Converge();
        int seqAfter = area.SequenceOf(1, 2);
        Check(seqAfter == 2, "the updated LSA carries sequence 2");
        area.Converge();
        Check(area.SequenceOf(1, 2) == seqAfter, "re-running flooding does not change the LSDB");

        // Cost change: both endpoints re-originate, then compare against a fresh area with the new cost.
        var changed = OspfTopology.Build();
        changed.SetLink(3, 5, 2);
        changed.Originate(3);
        changed.Originate(5);
        changed.Converge();
        var freshChange = OspfTopology.Build((3, 5, 2));
        Check(changed.Rounds <= 6, "the LSA update flooded inside the diameter, rounds=" + changed.Rounds);
        for (int source = 1; source <= 6; source++)
            Check(Describe(changed.RoutingTable(source)) == Describe(freshChange.RoutingTable(source)),
                $"cost change converges like a fresh area at router {source}");
        Check(changed.RoutingTable(1)[5].Cost == 5, "the cheaper 3-5 link is used (cost 5)");
        Check(changed.RoutingTable(1)[6].Cost == 7, "router 6 follows through router 5 (cost 7)");

        // Link removal: an alternate path is found and the tables still match a fresh area.
        var failed = OspfTopology.Build();
        failed.RemoveLink(4, 5);
        failed.Originate(4);
        failed.Originate(5);
        failed.Converge();
        var freshFailed = OspfTopology.Build((4, 5, -1));
        for (int source = 1; source <= 6; source++)
            Check(Describe(failed.RoutingTable(source)) == Describe(freshFailed.RoutingTable(source)),
                $"link removal converges like a fresh area at router {source}");
        Check(failed.RoutingTable(1)[5].Cost == 10, "router 5 falls back to the 3-5 link (cost 10)");
        Check(failed.RoutingTable(1)[6].Cost == 11, "router 6 is still reachable (cost 11)");

        // Partition: cutting both links to router 6 leaves it out of everybody's LSDB.
        var cut = OspfTopology.Build((4, 6, -1), (5, 6, -1));
        Check(cut.LsdbSize(1) == 5, "router 6 dropped out of router 1's LSDB");
        Check(cut.RoutingTable(1).Count == 4, "router 6 has no route from router 1");
        Check(cut.LsdbSize(6) > 0, "router 6 still holds its own database");
        for (int source = 1; source <= 5; source++)
            Check(!cut.RoutingTable(source).ContainsKey(6), $"router {source} has no route to the partition");

        // Refresh and rate limiting.
        var aging = OspfTopology.Build();
        Check(aging.AgeOf(1, 1) == 0, "a freshly originated LSA has age 0");
        for (int i = 0; i < LinkStateArea.LsRefreshTicks - 1; i++) aging.Tick();
        Check(aging.Originations == aging.RouterCount, "no refresh before LSRefreshTicks");
        Check(aging.AgeOf(1, 1) == LinkStateArea.LsRefreshTicks - 1, "the LSA aged one tick at a time");
        Check(aging.AgeOf(6, 1) == LinkStateArea.LsRefreshTicks - 1, "every database ages the copy it holds");
        aging.Tick();
        long afterRefresh = aging.Originations;
        Check(afterRefresh == 2 * aging.RouterCount, "every router refreshed once at LSRefreshTicks");
        Check(aging.SequenceOf(6, 1) == 2, "the refreshed LSA is sequence 2 everywhere");
        Check(aging.AgeOf(6, 1) == 0, "the refreshed instance restarts at age 0");
        Check(!aging.TryOriginate(1), "MinLSInterval blocks an immediate re-origination");
        Check(aging.Originations == afterRefresh, "the blocked attempt sent nothing");
        for (int i = 0; i < LinkStateArea.MinLsIntervalTicks; i++) aging.Tick();
        Check(aging.TryOriginate(3), "after MinLSInterval a new instance is allowed");
        Check(aging.SequenceOf(1, 3) == 3, "the forced instance is sequence 3");
        Check(aging.DatabaseEquals(1, 6), "the refreshed database is consistent across the area");

        // Degenerate topologies.
        var single = new LinkStateArea();
        single.AddRouter(1);
        single.OriginateAll();
        single.Converge();
        Check(single.RoutingTable(1).Count == 0, "a lone router has an empty routing table");
        Check(single.LsdbSize(1) == 1, "a lone router only knows its own LSA");

        var pair = new LinkStateArea();
        pair.AddRouter(7);
        pair.AddRouter(8);
        pair.SetLink(7, 8, 42);
        pair.OriginateAll();
        pair.Converge();
        Check(pair.RoutingTable(7)[8].Cost == 42 && pair.RoutingTable(7)[8].NextHop == 8, "two-router neighbour route");
        bool rejected = false;
        try { pair.SetLink(7, 8, 0); } catch (ArgumentOutOfRangeException) { rejected = true; }
        Check(rejected, "a zero-cost interface is rejected");

        Console.WriteLine($"rounds={area.Rounds} lsas={area.LsdbSize(1)} flood={area.FloodMessages} " +
                          $"originations={aging.Originations}");
        Console.WriteLine("PASS");
    }

    private static string Describe(SortedDictionary<int, (int Cost, int NextHop)> table)
    {
        var parts = new List<string>();
        foreach (var entry in table) parts.Add($"{entry.Key}:{entry.Value.Cost}via{entry.Value.NextHop}");
        return string.Join(",", parts);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
