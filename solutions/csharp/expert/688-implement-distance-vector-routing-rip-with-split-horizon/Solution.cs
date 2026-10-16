// Task 688 - Implement distance-vector routing (RIP) with split horizon. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A round-based RIP model. Every router keeps a distance vector (destination -> cost, next hop),
// directly attached networks sit in the vector at cost 0 and hop counts cap at 16, which is RIP's
// definition of infinity. A round is one synchronous exchange: each router builds the
// advertisement it would send on an interface, all advertisements are collected, and only then
// applied, so a round corresponds to one RIP update interval. Behaviour that is modelled:
//
//   * Bellman-Ford (here: Bellman-Ford-Moore, vector by vector) with the "route through this
//     neighbour changes -> take it even if it is worse" rule, which is what makes withdrawals
//     work;
//   * split horizon (never advertise a destination back to the neighbour it was learned from),
//     poisoned reverse (advertise it there with metric 16) and no suppression, all three
//     selectable, so the two fixes can be compared with the naive protocol;
//   * triggered updates plus a periodic timer, and a route timeout that drops a route nobody
//     confirmed for RipTimeout rounds - the mechanism that lets a silent router be forgotten;
//   * metric 16 is infinity: a destination 16 hops away is unreachable (checked on an 18 router
//     line).
//
// The count-to-infinity demonstration uses a silent interface failure: the stub network is
// detached on the router that owned it *without* notifying anybody, which is exactly the failure
// split horizon defends against. Without split horizon the two routers then bounce the route
// between each other, the strict "cost decreases along the next hop" invariant is violated, and
// the cost climbs to 16 one step per round; with split horizon or poisoned reverse the loop can
// never form.
using System;
using System.Collections.Generic;

internal sealed class RipRouter
{
    public readonly int Id;
    public readonly SortedDictionary<int, int> Interfaces = new SortedDictionary<int, int>();
    public readonly SortedDictionary<int, (int Cost, int NextHop)> Vector = new SortedDictionary<int, (int, int)>();
    public readonly SortedDictionary<int, int> LastSeen = new SortedDictionary<int, int>();

    public bool Dirty;
    public bool Active = true;

    public RipRouter(int id) => Id = id;

    public void AttachNetwork(int networkId)
    {
        Vector[networkId] = (0, Id);
        Dirty = true;
    }

    /// <summary>Removes an interface route without telling anybody - the classic silent failure.</summary>
    public void SilentDetachNetwork(int networkId)
    {
        Vector.Remove(networkId);
        LastSeen.Remove(networkId);
    }

    public void DetachNetwork(int networkId)
    {
        if (Vector.Remove(networkId)) Dirty = true;
        LastSeen.Remove(networkId);
    }
}

internal sealed class RipNetwork
{
    public const int Infinity = 16;
    public const int RipPeriod = 3;
    public const int RipTimeout = 8;

    private readonly SortedDictionary<int, RipRouter> _routers = new SortedDictionary<int, RipRouter>();

    public bool UseSplitHorizon;
    public bool UsePoisonedReverse;

    public long Rounds;
    public long UpdatesSent;
    public long AppliedChanges;
    public bool LoopInvariantViolated;

    public IEnumerable<RipRouter> Routers => _routers.Values;

    public RipRouter Router(int id)
    {
        if (!_routers.TryGetValue(id, out var router))
        {
            router = new RipRouter(id);
            router.Dirty = true;
            // Every RIP router advertises its loopback, so a router id is a routable destination too.
            router.AttachNetwork(id);
            _routers[id] = router;
        }
        return router;
    }

    public void AddLink(int a, int b)
    {
        if (a == b) throw new ArgumentException("a router cannot be its own neighbour");
        Router(a).Interfaces[b] = 1;
        Router(b).Interfaces[a] = 1;
        Router(a).Dirty = true;
        Router(b).Dirty = true;
    }

    public void FailLink(int a, int b)
    {
        Router(a).Interfaces.Remove(b);
        Router(b).Interfaces.Remove(a);
        Router(a).Dirty = true;
        Router(b).Dirty = true;
    }

    private List<(int Dest, int Cost)> Advertisement(RipRouter from, int neighbour)
    {
        var advert = new List<(int, int)>();
        foreach (var route in from.Vector)
        {
            if (route.Value.NextHop == neighbour && UseSplitHorizon) continue;
            int cost = route.Value.NextHop == neighbour && UsePoisonedReverse ? Infinity : route.Value.Cost;
            advert.Add((route.Key, cost));
        }
        return advert;
    }

    private int Apply(RipRouter receiver, int from, List<(int Dest, int Cost)> advert)
    {
        int linkCost = receiver.Interfaces[from];
        int changes = 0;
        foreach ((int dest, int cost) in advert)
        {
            if (dest == receiver.Id) continue;
            int candidate = Math.Min(cost + linkCost, Infinity);
            bool hasRoute = receiver.Vector.TryGetValue(dest, out var route);
            if (hasRoute && route.NextHop == from)
            {
                // The neighbour owns this route: accept the change, even a worse one (withdrawal path).
                if (candidate >= Infinity)
                {
                    receiver.Vector.Remove(dest);
                    receiver.LastSeen.Remove(dest);
                    receiver.Dirty = true;
                    changes++;
                }
                else
                {
                    receiver.LastSeen[dest] = (int)Rounds;
                    if (route.Cost != candidate)
                    {
                        receiver.Vector[dest] = (candidate, from);
                        receiver.Dirty = true;
                        changes++;
                    }
                }
            }
            else if (candidate < (hasRoute ? route.Cost : Infinity))
            {
                receiver.Vector[dest] = (candidate, from);
                receiver.LastSeen[dest] = (int)Rounds;
                receiver.Dirty = true;
                changes++;
            }
            // A worse route from a different neighbour neither replaces the entry nor resets its
            // timeout: only the router that owns the route keeps it alive.
        }
        return changes;
    }

    /// <summary>One RIP exchange: collect every advertisement, then apply them all.</summary>
    public int RunRound()
    {
        Rounds++;
        bool periodic = Rounds % RipPeriod == 0;
        var deliveries = new List<(int To, int From, List<(int Dest, int Cost)> Advert)>();

        foreach (var entry in _routers)
        {
            RipRouter router = entry.Value;
            if (!router.Active) continue;
            if (!periodic && !router.Dirty) continue;
            foreach (int neighbour in router.Interfaces.Keys)
            {
                deliveries.Add((neighbour, router.Id, Advertisement(router, neighbour)));
                UpdatesSent++;
            }
            router.Dirty = false;
        }

        int changes = 0;
        foreach ((int to, int from, var advert) in deliveries) changes += Apply(_routers[to], from, advert);

        foreach (var entry in _routers)
        {
            var router = entry.Value;
            var expired = new List<int>();
            foreach (var route in new List<KeyValuePair<int, (int Cost, int NextHop)>>(router.Vector))
            {
                if (route.Value.NextHop == router.Id) continue;               // directly attached
                if (!router.LastSeen.TryGetValue(route.Key, out int seen)) continue;
                if (Rounds - seen <= RipTimeout) continue;
                expired.Add(route.Key);
            }
            foreach (int dest in expired)
            {
                router.Vector.Remove(dest);
                router.LastSeen.Remove(dest);
                router.Dirty = true;
                changes++;
            }
        }

        AppliedChanges += changes;
        if (!DecreasingAlongNextHop()) LoopInvariantViolated = true;
        return changes;
    }

    /// <summary>True when no router has anything to advertise.</summary>
    public bool Stable()
    {
        foreach (var entry in _routers) if (entry.Value.Dirty) return false;
        return true;
    }

    public void Run(int maxRounds)
    {
        int quiet = 0;
        for (int i = 0; i < maxRounds; i++)
        {
            int changes = RunRound();
            // Keep going through a full timeout horizon after the last change so that routes that
            // were never refreshed have a chance to expire.
            quiet = changes == 0 ? quiet + 1 : 0;
            if (quiet >= RipTimeout + RipPeriod) return;
        }
    }

    /// <summary>
    /// The split-horizon invariant: whenever a route's next hop also knows the destination, that
    /// neighbour must be strictly closer. A violation means the two routers are feeding each other
    /// a bogus route (the counting loop). A neighbour with no route at all is a blackhole rather
    /// than a loop; those are cleaned up by the route timeout and checked separately.
    /// </summary>
    public bool DecreasingAlongNextHop()
    {
        foreach (var entry in _routers)
        {
            RipRouter router = entry.Value;
            foreach (var route in router.Vector)
            {
                if (route.Value.NextHop == router.Id) continue;
                if (!_routers.TryGetValue(route.Value.NextHop, out var next)) return false;
                if (!next.Vector.TryGetValue(route.Key, out var downstream)) continue;
                if (downstream.Cost >= route.Value.Cost) return false;
            }
        }
        return true;
    }

    public bool HasLoop()
    {
        foreach (var entry in _routers)
        {
            foreach (var route in entry.Value.Vector)
            {
                int current = entry.Key;
                var visited = new HashSet<int>();
                int steps = 0;
                while (true)
                {
                    if (!_routers.TryGetValue(current, out var router)) break;
                    if (!router.Vector.TryGetValue(route.Key, out var hop)) break;
                    if (hop.NextHop == current) break;
                    if (!visited.Add(current)) return true;
                    current = hop.NextHop;
                    if (++steps > _routers.Count + 2) return true;
                }
            }
        }
        return false;
    }

    public int CostOf(int routerId, int destination)
    {
        return _routers[routerId].Vector.TryGetValue(destination, out var route) ? route.Cost : Infinity;
    }

    public string Dump()
    {
        var parts = new List<string>();
        foreach (var entry in _routers)
        {
            var routes = new List<string>();
            foreach (var route in entry.Value.Vector)
                routes.Add($"{route.Key}:{route.Value.Cost}via{route.Value.NextHop}");
            parts.Add($"{entry.Key}[{string.Join(",", routes)}]");
        }
        return string.Join(" ", parts);
    }

    /// <summary>Reference hop count from BFS over the topology, ignoring the protocol entirely.</summary>
    public static int BfsCost(SortedDictionary<int, List<int>> adjacency, int source, int destination)
    {
        if (source == destination) return 0;
        var distance = new SortedDictionary<int, int> { { source, 0 } };
        var queue = new Queue<int>();
        queue.Enqueue(source);
        while (queue.Count > 0)
        {
            int current = queue.Dequeue();
            if (!adjacency.TryGetValue(current, out var neighbours)) continue;
            foreach (int next in neighbours)
            {
                if (distance.ContainsKey(next)) continue;
                distance[next] = distance[current] + 1;
                queue.Enqueue(next);
            }
        }
        return distance.TryGetValue(destination, out int hops) ? hops : Infinity;
    }
}

public static class Solution
{
    private const int StubX = 100;

    public static void Main()
    {
        // --- Converged hop counts must equal BFS distances. ---------------------------------
        var network = new RipNetwork { UseSplitHorizon = true };
        for (int id = 1; id <= 5; id++) network.Router(id);
        network.AddLink(1, 2);
        network.AddLink(1, 3);
        network.AddLink(2, 4);
        network.AddLink(3, 4);
        network.AddLink(4, 5);
        network.Router(1).AttachNetwork(StubX);
        network.Run(60);

        var adjacency = new SortedDictionary<int, List<int>>
        {
            { 1, new List<int> { 2, 3, StubX } },
            { 2, new List<int> { 1, 4 } },
            { 3, new List<int> { 1, 4 } },
            { 4, new List<int> { 2, 3, 5 } },
            { 5, new List<int> { 4 } },
            { StubX, new List<int> { 1 } },
        };
        for (int id = 1; id <= 5; id++)
        {
            // The owner holds the directly attached network at cost 0, so the metric of any other
            // router is its hop count to the owner, not to the network itself.
            int expected = id == 1 ? 0 : RipNetwork.BfsCost(adjacency, id, 1);
            Check(network.CostOf(id, StubX) == expected, $"router {id} cost to the stub is {expected}");
        }
        Check(network.CostOf(1, StubX) == 0, "the owning router holds the stub at cost 0");
        Check(network.CostOf(2, 5) == 2 && network.CostOf(3, 5) == 2, "router 2 and 3 count two hops to router 5");
        Check(network.CostOf(1, 5) == 3, "router 1 is three hops from router 5");
        Check(!network.HasLoop(), "the converged vectors are loop free");

        // Fixed point: the next round applies nothing new.
        int before = (int)network.AppliedChanges;
        network.RunRound();
        Check(network.AppliedChanges == before, "a converged network applies no further changes");

        // --- Split horizon keeps the loop-free invariant while converging. ------------------
        var guarded = new RipNetwork { UseSplitHorizon = true };
        guarded.Router(1);
        guarded.Router(2);
        guarded.AddLink(1, 2);
        guarded.Router(2).AttachNetwork(StubX);
        guarded.Run(20);
        Check(guarded.CostOf(1, StubX) == 1, "router 1 reaches the stub through router 2");
        Check(!guarded.LoopInvariantViolated, "split horizon never breaks the decreasing-cost invariant");

        // --- Classic count-to-infinity: silent failure of the stub interface. ---------------
        var naive = new RipNetwork { UseSplitHorizon = false };
        naive.Router(1);
        naive.Router(2);
        naive.AddLink(1, 2);
        naive.Router(2).AttachNetwork(StubX);
        naive.Run(20);
        Check(naive.CostOf(1, StubX) == 1, "naive RIP agrees before the failure");
        naive.Router(2).SilentDetachNetwork(StubX);      // nobody is told about the failure
        long roundsUntilGone = -1;
        for (int i = 0; i < 40; i++)
        {
            naive.RunRound();
            if (naive.CostOf(1, StubX) >= RipNetwork.Infinity && roundsUntilGone < 0) roundsUntilGone = i + 1;
        }
        Check(roundsUntilGone > 8, "the naive protocol counts to infinity slowly, rounds=" + roundsUntilGone);
        Check(roundsUntilGone < 30, "but it does reach infinity, rounds=" + roundsUntilGone);
        Check(naive.LoopInvariantViolated, "the counting phase violates the decreasing-cost invariant");
        Check(naive.CostOf(2, StubX) == RipNetwork.Infinity, "the silent router ends up without the stub");

        var guardedFailure = new RipNetwork { UseSplitHorizon = true };
        guardedFailure.Router(1);
        guardedFailure.Router(2);
        guardedFailure.AddLink(1, 2);
        guardedFailure.Router(2).AttachNetwork(StubX);
        guardedFailure.Run(20);
        guardedFailure.Router(2).SilentDetachNetwork(StubX);
        for (int i = 0; i < 5; i++) guardedFailure.RunRound();
        Check(guardedFailure.CostOf(1, StubX) == 1, "split horizon keeps the stale route until it times out");
        for (int i = 0; i < RipNetwork.RipTimeout + 4; i++) guardedFailure.RunRound();
        Check(guardedFailure.CostOf(1, StubX) == RipNetwork.Infinity, "the unreachable stub finally times out");
        Check(guardedFailure.CostOf(1, StubX) != 2, "router 2 never lent router 1 a bogus path back");

        var poisoned = new RipNetwork { UsePoisonedReverse = true };
        poisoned.Router(1);
        poisoned.Router(2);
        poisoned.AddLink(1, 2);
        poisoned.Router(2).AttachNetwork(StubX);
        poisoned.Run(20);
        poisoned.Router(2).SilentDetachNetwork(StubX);
        for (int i = 0; i < 5; i++) poisoned.RunRound();
        Check(poisoned.CostOf(1, StubX) == 1, "poisoned reverse behaves like split horizon for router 1");
        Check(poisoned.CostOf(2, StubX) == RipNetwork.Infinity, "the poisoned advertisement is never installed");
        Check(!poisoned.LoopInvariantViolated, "poisoned reverse never breaks the invariant");
        for (int i = 0; i < RipNetwork.RipTimeout + 4; i++) poisoned.RunRound();
        Check(poisoned.CostOf(1, StubX) == RipNetwork.Infinity, "poisoned reverse also times the route out");

        // --- Split horizon must not hide a legitimate alternative path. --------------------
        var triangle = new RipNetwork { UseSplitHorizon = true };
        for (int id = 1; id <= 3; id++) triangle.Router(id);
        triangle.AddLink(1, 2);
        triangle.AddLink(2, 3);
        triangle.AddLink(1, 3);
        triangle.Router(3).AttachNetwork(StubX);
        triangle.Run(20);
        Check(triangle.CostOf(1, StubX) == 1, "router 1 uses the direct link to router 3");
        triangle.FailLink(1, 3);
        triangle.Run(30);
        Check(triangle.CostOf(1, StubX) == 2, "after the failure router 1 goes through router 2: rounds=" + triangle.Rounds + " " + triangle.Dump());
        Check(triangle.CostOf(1, 3) == 2, "router 3 itself is also reached through router 2");
        Check(!triangle.HasLoop(), "the reconverged vectors are loop free");
        Check(triangle.DecreasingAlongNextHop(), "the reconverged triangle satisfies the decreasing-cost invariant");

        // --- A silent router times out and comes back. ------------------------------------
        var timeout = new RipNetwork { UseSplitHorizon = true };
        timeout.Router(1);
        timeout.Router(2);
        timeout.Router(3);
        timeout.AddLink(1, 2);
        timeout.AddLink(2, 3);
        timeout.Router(3).AttachNetwork(StubX);
        timeout.Run(20);
        Check(timeout.CostOf(1, StubX) == 2, "router 1 reaches the stub two hops away");
        timeout.Router(2).Active = false;                 // the middle router goes quiet
        timeout.Run(30);
        Check(timeout.CostOf(1, StubX) == RipNetwork.Infinity, "the silent router's routes expire");
        timeout.Router(2).Active = true;
        timeout.Router(2).Dirty = true;
        timeout.Run(30);
        Check(timeout.CostOf(1, StubX) == 2, "the recovering router restores the route");

        // --- Metric 16 is infinity: the 18th router of a line is unreachable. --------------
        var line = new RipNetwork { UseSplitHorizon = true };
        for (int id = 1; id <= 18; id++) line.Router(id);
        for (int id = 1; id < 18; id++) line.AddLink(id, id + 1);
        line.Router(1).AttachNetwork(StubX);
        line.Run(80);
        Check(line.CostOf(2, StubX) == 1, "the neighbour counts one hop");
        Check(line.CostOf(16, StubX) == 15, "router 16 is still inside the metric range");
        Check(line.CostOf(17, StubX) == RipNetwork.Infinity, "router 17 is exactly at infinity");
        Check(line.CostOf(18, StubX) == RipNetwork.Infinity, "router 18 stays unreachable");
        Check(!line.HasLoop(), "the long line is loop free");

        // --- Determinism and input validation. -------------------------------------------
        var first = new RipNetwork { UseSplitHorizon = true };
        var second = new RipNetwork { UseSplitHorizon = true };
        foreach (var target in new[] { first, second })
        {
            for (int id = 1; id <= 6; id++) target.Router(id);
            target.AddLink(1, 2);
            target.AddLink(2, 3);
            target.AddLink(3, 4);
            target.AddLink(4, 5);
            target.AddLink(5, 6);
            target.AddLink(6, 1);
            target.Router(4).AttachNetwork(StubX);
            target.Run(50);
        }
        Check(first.Dump() == second.Dump(), "two identical runs converge to the same vectors");
        Check(first.UpdatesSent == second.UpdatesSent, "two identical runs send the same number of updates");
        Check(first.CostOf(1, StubX) == RipNetwork.BfsCost(
            new SortedDictionary<int, List<int>>
            {
                { 1, new List<int> { 2, 6 } }, { 2, new List<int> { 1, 3 } }, { 3, new List<int> { 2, 4 } },
                { 4, new List<int> { 3, 5, StubX } }, { 5, new List<int> { 4, 6 } }, { 6, new List<int> { 5, 1 } },
                { StubX, new List<int> { 4 } },
            }, 1, 4), "the ring matches the BFS oracle");

        bool rejected = false;
        try { first.AddLink(3, 3); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a self link is rejected");

        Console.WriteLine($"rounds={network.Rounds} updates={network.UpdatesSent} naiveRoundsToInfinity={roundsUntilGone} " +
                          $"lineRounds={line.Rounds}");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
