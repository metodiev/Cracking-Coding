"""Task 688 - Implement distance-vector routing (RIP) with split horizon.  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

RIP is Bellman-Ford run by routers instead of by one process: every router periodically hands its
whole distance vector to each neighbour, and a router accepts an advertised destination when the
advertised metric plus the link cost beats what it already has.  Two rules define the interesting
behaviour and both are implemented:

  * **split horizon** — a router never advertises a destination back to the neighbour it learned it
    from, because that neighbour's metric cannot possibly be better than ours;
  * **poisoned reverse** — instead of staying silent it sends metric 16 ("unreachable") for those
    destinations, which propagates a *withdrawal* immediately and kills two-router loops outright.

A route that is already owned through the neighbour sending the update is refreshed even when the
new metric is *worse* — that is exactly how a distance-vector protocol counts to infinity, and the
three strategies (none, split horizon, poisoned reverse) are compared on that same scenario.  A
scaled-down invalid timer (4 rounds in place of RIP's 180 s) drops a route no neighbour refreshed,
which is what makes plain split horizon eventually stop believing a stale route.

Simplifications: no triggered/periodic timing subtleties beyond a configurable refresh, no route
hold-down or flush timers, symmetric link costs, one destination per router (its own network), and
metric arithmetic capped at 16 exactly as RIP defines it.  Verified: every converged table satisfies
the Bellman-Ford fixed-point equation *and* matches an independent Dijkstra oracle; the hop count is
limited to 15 (a 17-router chain leaves the far end unreachable); metrics never exceed 15 in a table
and 16 on the wire; split horizon never leaks a route back to its next hop; poisoned reverse sends
16 exactly to that neighbour; and triggered updates reach the same table carrying fewer route
entries than a schedule of whole-table updates.
"""

from collections import namedtuple

INFINITY = 16                                   # RIP: 16 means unreachable, so 15 is the longest path
NONE, SPLIT_HORIZON, POISONED = "none", "split_horizon", "poisoned_reverse"
TIMEOUT_ROUNDS = 4                              # stands in for RIP's 180 s invalid timer
WITHDRAW_ROUNDS = 2                             # how often a poisoned withdrawal is repeated

Stats = namedtuple("Stats", "rounds messages entries stabilised")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class RipRouter:
    def __init__(self, rid, timeout=TIMEOUT_ROUNDS):
        self.rid = rid
        self.links = {}                          # neighbour rid -> link cost
        self.table = {rid: (0, rid)}             # dst -> (metric, next hop); our own network is 0
        self.age = {}                            # dst -> rounds since some neighbour refreshed it
        self.refreshed = set()
        self.dirty = set()                       # destinations to send in a triggered update
        self.withdrawn = {}                      # dst -> rounds left to keep advertising "gone"
        self.timeout = timeout

    def advertise(self, neighbour, strategy, dirty=None):
        """The route list sent to `neighbour`.  Whole table when `dirty` is None, else only those
        destinations (a triggered update, which also carries withdrawals for vanished routes).

        Only poisoned reverse reports a local withdrawal as metric 16; plain split horizon stays
        silent and "no horizon" simply omits the route, which is the textbook configuration in which
        a distance vector counts to infinity."""
        out = {}
        keys = set(self.table) if dirty is None else set(dirty)
        keys |= set(self.withdrawn)
        for dst in sorted(keys):
            entry = self.table.get(dst)
            if entry is None:
                if strategy == POISONED:
                    out[dst] = INFINITY          # withdrawal: tell the neighbour the route is gone
                continue
            metric, hop = entry
            if hop == neighbour and strategy != NONE:
                if strategy == POISONED:
                    out[dst] = INFINITY          # the poison that prevents a two-router loop
                continue                         # split horizon: say nothing about it at all
            out[dst] = metric
        return out

    def update(self, neighbour, advertised, link_cost):
        """One Bellman-Ford step.  A route already owned through this neighbour is refreshed even
        upwards, which is how count-to-infinity arises; otherwise only strictly better metrics win."""
        changed = False
        for dst, metric in sorted(advertised.items()):
            if dst == self.rid:
                continue
            candidate = min(INFINITY, link_cost + metric)
            current = self.table.get(dst)
            if current is not None and current[1] == neighbour:
                self.refreshed.add(dst)
                if candidate == current[0]:
                    continue
                if candidate >= INFINITY:
                    del self.table[dst]
                    self.age.pop(dst, None)
                else:
                    self.table[dst] = (candidate, neighbour)
                self.dirty.add(dst)
                changed = True
                continue
            if not (current is None or candidate < current[0]) or candidate >= INFINITY:
                continue
            self.table[dst] = (candidate, neighbour)
            self.age[dst] = 0
            self.refreshed.add(dst)
            self.dirty.add(dst)
            changed = True
        return changed

    def age_round(self):
        """The invalid timer: drop what no neighbour refreshed for `timeout` rounds."""
        expired = []
        for dst in sorted(self.table):
            if dst == self.rid:
                continue
            if dst in self.refreshed:
                self.age[dst] = 0
            else:
                self.age[dst] = self.age.get(dst, 0) + 1
                if self.age[dst] > self.timeout:
                    expired.append(dst)
        for dst in expired:
            del self.table[dst]
            self.age.pop(dst, None)
            self.withdrawn[dst] = WITHDRAW_ROUNDS
            self.dirty.add(dst)
        for dst in list(self.withdrawn):
            self.withdrawn[dst] -= 1
            if self.withdrawn[dst] <= 0:
                del self.withdrawn[dst]
        self.refreshed.clear()
        return bool(expired)

    def pending_timeouts(self):
        """True while some entry is ageing towards the invalid timer: the run is not stable yet."""
        return any(dst != self.rid and self.age.get(dst, 0) > 0 for dst in self.table)


def build(routers, links):
    """Create the routers (an iterable of ids) and the symmetric links ((a, b, cost) triples)."""
    table = {rid: RipRouter(rid) for rid in routers}
    for a, b, cost in links:
        table[a].links[b] = cost
        table[b].links[a] = cost
    return table


def converge(routers, strategy=NONE, triggered=False, refresh=3, max_rounds=200, observe=None):
    """Synchronous rounds until every table stops moving and no entry is still ageing.

    `messages` counts transmissions (one per link per round), `entries` counts the route entries
    actually carried — the number that triggered updates are supposed to shrink."""
    rounds, messages, entries = 0, 0, 0
    stabilised = False
    while rounds < max_rounds:
        rounds += 1
        changed = False
        full = not triggered or (rounds - 1) % refresh == 0
        pending = []
        for sender in sorted(routers):
            router = routers[sender]
            dirty = None if full else set(router.dirty)
            router.dirty.clear()              # this round's changes start a fresh set
            if not full and not dirty:
                continue
            for receiver in sorted(router.links):
                advertised = router.advertise(receiver, strategy, dirty)
                pending.append((receiver, sender, advertised))
                messages += 1
                entries += len(advertised)
        for receiver, sender, advertised in pending:
            if routers[receiver].update(sender, advertised, routers[receiver].links[sender]):
                changed = True
        for router in routers.values():
            if router.age_round():
                changed = True
        if observe is not None:
            observe(rounds, routers)
        ageing = any(router.pending_timeouts() for router in routers.values())
        if not changed and not ageing:
            stabilised = True
            break
    return Stats(rounds, messages, entries, stabilised)


def fail_router(routers, rid):
    """The router (and the network it announces) disappears: its neighbours lose the interface and
    immediately invalidate every route they had through it, the destination itself included."""
    for neighbour in sorted(routers[rid].links):
        router = routers[neighbour]
        del router.links[rid]
        for dst in [d for d, (_, hop) in router.table.items() if hop == rid]:
            del router.table[dst]
            router.age.pop(dst, None)
            router.withdrawn[dst] = WITHDRAW_ROUNDS
            router.dirty.add(dst)
    del routers[rid]


def fail_link(routers, a, b):
    del routers[a].links[b]
    del routers[b].links[a]
    for rid, other in ((a, b), (b, a)):
        router = routers[rid]
        for dst in [d for d, (_, hop) in router.table.items() if hop == other]:
            del router.table[dst]
            router.age.pop(dst, None)
            router.withdrawn[dst] = WITHDRAW_ROUNDS
            router.dirty.add(dst)


def brute_force(links, nodes, source):
    """Independent Dijkstra oracle; distances of 16 or more are unreachable in RIP terms."""
    adjacency = {node: {} for node in nodes}
    for a, b, cost in links:
        adjacency[a][b] = cost
        adjacency[b][a] = cost
    distance = {node: None for node in nodes}
    distance[source] = 0
    settled = set()
    while True:
        choice = None
        for node in sorted(nodes):
            if node in settled or distance[node] is None:
                continue
            if choice is None or distance[node] < distance[choice]:
                choice = node
        if choice is None:
            break
        settled.add(choice)
        for node, cost in sorted(adjacency[choice].items()):
            candidate = distance[choice] + cost
            if distance[node] is None or candidate < distance[node]:
                distance[node] = candidate
    return {node: (None if value is None or value >= INFINITY else value)
            for node, value in distance.items()}


def expected_table(links, nodes, source):
    """RIP semantics: the oracle distance, or None when it does not fit in 15 hops."""
    return brute_force(links, nodes, source)


def fixed_point_defect(routers):
    """First place where the Bellman-Ford equation fails, or None when every table is a fixed point."""
    for rid in sorted(routers):
        router = routers[rid]
        for dst in sorted(routers):
            if dst == rid:
                want = 0
            else:
                want = INFINITY
                for neighbour, cost in sorted(router.links.items()):
                    metric = routers[neighbour].table.get(dst, (INFINITY, None))[0]
                    want = min(want, min(INFINITY, cost + metric))
            have = router.table.get(dst, (INFINITY, None))[0]
            if have != want:
                return (rid, dst, have, want)
    return None


def main():
    nodes = [1, 2, 3, 4, 5]
    links = [(1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 4), (1, 5, 7)]
    for strategy in (NONE, SPLIT_HORIZON, POISONED):
        routers = build(nodes, links)
        stats = converge(routers, strategy)
        check(stats.stabilised, f"strategy {strategy} converges")
        check(stats.rounds <= len(nodes) + TIMEOUT_ROUNDS,
              f"strategy {strategy} converged in {stats.rounds} rounds")
        check(fixed_point_defect(routers) is None, f"strategy {strategy} reaches a Bellman-Ford fixpoint")
        for rid in nodes:
            oracle = expected_table(links, nodes, rid)
            for dst in nodes:
                want = oracle[dst]
                have = routers[rid].table.get(dst)
                if want is None:
                    check(have is None, f"strategy {strategy}: {rid} -> {dst} should be unreachable")
                else:
                    check(have is not None and have[0] == want,
                          f"strategy {strategy}: {rid} -> {dst} is {have}, RIP says {want}")
            check(all(metric <= 15 for metric, _ in routers[rid].table.values()),
                  f"strategy {strategy}: router {rid} stored an unreachable metric")
            for neighbour, _ in sorted(routers[rid].links.items()):
                for dst, metric in routers[rid].advertise(neighbour, strategy).items():
                    check(metric <= INFINITY, "no advertisement may exceed 16")
                    if strategy == SPLIT_HORIZON:
                        hop = routers[rid].table[dst][1]
                        check(hop != neighbour, "split horizon leaked a route back to its next hop")
                    if strategy == POISONED and routers[rid].table[dst][1] == neighbour:
                        check(metric == INFINITY, "poisoned reverse must send 16")

    for strategy in (SPLIT_HORIZON, POISONED):
        routers = build(nodes, links)
        converge(routers, strategy)
        for rid in nodes:
            for dst in nodes:
                entry = routers[rid].table.get(dst)
                if entry is not None and dst != rid:
                    other = routers[entry[1]].table.get(dst)
                    check(other is None or other[1] != rid,
                          f"strategy {strategy}: {rid} and {entry[1]} form a two-router loop")

    chain = build(range(1, 18), [(i, i + 1, 1) for i in range(1, 17)])
    converge(chain, SPLIT_HORIZON, max_rounds=400)
    chain_links = [(i, i + 1, 1) for i in range(1, 17)]
    for rid in range(2, 13):
        check(chain[rid].table[17][0] == 17 - rid, f"chain: router {rid} is {17 - rid} hops from 17")
    check(17 not in chain[1].table, "a 16-hop destination is unreachable in RIP")
    check(chain[2].table.get(17, (None,))[0] == 15, "the 15-hop limit is the last usable one")
    check(brute_force(chain_links, range(1, 18), 1)[17] is None,
          "the oracle agrees that 17 is beyond the metric limit")

    history = []
    none_history = []

    def record(round_number, routers):
        history.append((round_number, routers[1].table.get(3, (None, None))[0],
                        routers[2].table.get(3, (None, None))[0]))

    for strategy in (NONE, SPLIT_HORIZON, POISONED):
        routers = build([1, 2, 3], [(1, 2, 1), (2, 3, 1)])
        converge(routers, strategy)
        check(routers[1].table[3] == (2, 2), "before the failure the route is two hops through 2")
        history.clear()
        fail_router(routers, 3)
        stats = converge(routers, strategy, refresh=1, observe=record)
        check(stats.stabilised, f"{strategy}: the failure converges")
        check(3 not in routers[1].table and 3 not in routers[2].table,
              f"{strategy}: nobody believes the dead route any more")
        seen = [metric for _, metric, _ in history if metric is not None]
        if strategy == NONE:
            none_history = list(history)
            check(stats.rounds >= 6, "without split horizon the news spreads one hop at a time")
            check(seen == sorted(seen) and max(seen) >= 12,
                  "the metric at 1 climbs towards 16 - counting to infinity")
            check(all(15 >= metric >= 2 for metric in seen), "and it never exceeds the maximum metric")
        elif strategy == SPLIT_HORIZON:
            check(max(seen) == 2, "split horizon stops the metric from growing at all")
            check(stats.rounds <= TIMEOUT_ROUNDS + 3, "the stale route dies on the invalid timer")
        else:
            check(all(metric <= 2 for metric in seen), "poisoned reverse never inflates the metric")
            check(stats.rounds <= 4, "poisoned reverse withdraws the route immediately")

    periodic = build(nodes, links)
    periodic_stats = converge(periodic, POISONED)
    triggered = build(nodes, links)
    triggered_stats = converge(triggered, POISONED, triggered=True)
    check({rid: periodic[rid].table for rid in nodes} == {rid: triggered[rid].table for rid in nodes},
          "triggered updates reach the same table as periodic updates")
    check(triggered_stats.entries < periodic_stats.entries,
          "triggered updates carry fewer route entries than whole-table periodic updates")

    changed = build(nodes, links)
    converge(changed, POISONED)
    check(changed[1].table[5][0] == 7, "the expensive direct link to 5 is used before the change")
    changed[1].links[5] = 2
    changed[5].links[1] = 2
    changed[1].dirty.add(1)
    changed[5].dirty.add(5)
    converge(changed, POISONED, triggered=True, max_rounds=400)
    cheaper = [(1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 4), (1, 5, 2)]
    check(changed[1].table[5][0] == expected_table(cheaper, nodes, 1)[5] == 2,
          "a cheaper link is adopted after re-convergence")
    check(changed[4].table[1][0] == expected_table(cheaper, nodes, 4)[1] == 4,
          "and the deeper routers follow the shorter path")

    broken = build(nodes, links)
    converge(broken, POISONED)
    fail_link(broken, 3, 4)
    converge(broken, POISONED, max_rounds=400)
    remaining = [(1, 2, 1), (2, 3, 2), (4, 5, 4), (1, 5, 7)]
    for rid in nodes:
        oracle = expected_table(remaining, nodes, rid)
        for dst in nodes:
            want = oracle[dst]
            have = broken[rid].table.get(dst)
            check((None if have is None else have[0]) == want,
                  f"after the 3-4 link failure {rid} -> {dst} is {have}, RIP says {want}")

    twin_a = build(nodes, links)
    twin_b = build(nodes, links)
    converge(twin_a, POISONED)
    converge(twin_b, POISONED)
    check({rid: twin_a[rid].table for rid in nodes} == {rid: twin_b[rid].table for rid in nodes},
          "the simulation is deterministic")

    print(f"5-router area: periodic {periodic_stats.rounds} rounds / {periodic_stats.messages} adverts "
          f"/ {periodic_stats.entries} entries, triggered {triggered_stats.rounds} rounds / "
          f"{triggered_stats.messages} adverts / {triggered_stats.entries} entries")
    print("17-router chain: 17 is beyond the 15-hop limit, 15 is the longest usable distance")
    print(f"count-to-infinity at router 1 (no horizon): {[m for _, m, _ in none_history]}")
    print("PASS")


if __name__ == "__main__":
    main()
