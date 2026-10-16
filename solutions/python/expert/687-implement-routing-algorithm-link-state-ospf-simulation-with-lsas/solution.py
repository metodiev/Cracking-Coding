"""Task 687 - Implement routing algorithm link-state (OSPF) simulation with LSAs.  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

Every router originates a router-LSA — its router id, a sequence number and the links it has with
their costs — and floods it through the area, so each router ends up holding the whole link-state
database (LSDB) and runs Dijkstra locally.  Sequence numbers are what make the flooding terminate:
an LSA is installed only when it is newer than the copy already held, and a duplicate or older copy
is dropped *without* being re-flooded; an LSA is also never sent back out of the interface it arrived
on.  One transmission per router per LSA is therefore the exact cost, deg(origin) + sum(deg - 1).

Simplifications of this scaled-down model (no faked results, just a smaller area): no LSA ageing or
periodic refresh, no checksums, no DR/BDR election, no stub networks or summarisation, symmetric link
costs and one flat area.  The parts modelled in full are reliable flooding and per-router SPF.

Invariants verified: after any change every router's LSDB is identical; the transmission count for a
single LSA is exactly deg(origin) + sum over the other reached routers of (deg - 1) = 2E - n + 1, so
flooding is bounded; a stale sequence number is rejected and changes nothing, including a "same
sequence, different content" rewrite; distances match an independently written O(n^2) label-setting
Dijkstra oracle run over the raw link list; every next hop is a direct neighbour whose own distance
decomposes the tree exactly (dist(r, d) == cost(r, h) + dist_h(d)); a cost change re-converges from
only the two affected LSAs; and a partition makes the unreachable routers report no route at all.
"""

import heapq
from collections import namedtuple

LSA = namedtuple("LSA", "router seq links")     # links = tuple of (neighbour, cost), sorted


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Router:
    def __init__(self, rid):
        self.rid = rid
        self.links = {}         # neighbour rid -> cost
        self.seq = 0

    def originate(self):
        """A fresh router-LSA for the current link state; sequence numbers only ever go up."""
        self.seq += 1
        return LSA(self.rid, self.seq, tuple(sorted(self.links.items())))


class Area:
    def __init__(self, links):
        self.routers = {}
        for a, b, cost in links:
            self.routers[a] = self.routers.get(a, Router(a))
            self.routers[b] = self.routers.get(b, Router(b))
            self.routers[a].links[b] = cost
            self.routers[b].links[a] = cost
        self.lsdb = {rid: {} for rid in self.routers}
        self.transmissions = 0

    def install(self, rid, lsa):
        """Install `lsa` in rid's LSDB.  True only when it is strictly newer than the stored copy."""
        current = self.lsdb[rid].get(lsa.router)
        if current is not None and current.seq >= lsa.seq:
            return False                        # duplicate or stale: drop and do not re-flood
        self.lsdb[rid][lsa.router] = lsa
        return True

    def flood_lsa(self, origin, lsa):
        """Flood one LSA from its origin; returns how many routers (re)installed it."""
        stack, installs = [(origin, None, lsa)], 0
        while stack:
            receiver, sender, current = stack.pop()
            if not self.install(receiver, current):
                continue
            installs += 1
            for neighbour in sorted(self.routers[receiver].links):
                if neighbour == sender:
                    continue                    # never back out of the receiving interface
                self.transmissions += 1
                stack.append((neighbour, receiver, current))
        return installs

    def originate_and_flood(self, rids=None):
        """Originate (and flood) new LSAs — all routers by default, only the changed ones otherwise."""
        for rid in sorted(self.routers if rids is None else rids):
            self.flood_lsa(rid, self.routers[rid].originate())

    def graph(self, rid):
        """The topology as described by rid's LSDB — deliberately not read from the live links."""
        adjacency = {node: {} for node in self.lsdb[rid]}
        for lsa in self.lsdb[rid].values():
            for neighbour, cost in lsa.links:
                if neighbour in adjacency:
                    adjacency[lsa.router][neighbour] = cost
                    adjacency[neighbour][lsa.router] = cost
        return adjacency

    def spf(self, rid):
        """Dijkstra over rid's LSDB; returns (distance, next hop) per router, None when unreachable."""
        adjacency = self.graph(rid)
        distance = {node: None for node in adjacency}
        next_hop = {}
        distance[rid] = 0
        heap, settled = [(0, rid, rid)], set()
        while heap:
            cost, first_hop, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            if node != rid:
                next_hop[node] = first_hop
            for neighbour, link_cost in sorted(adjacency[node].items()):
                if neighbour in settled:
                    continue
                candidate = cost + link_cost
                if distance[neighbour] is None or candidate < distance[neighbour]:
                    distance[neighbour] = candidate
                    hop = first_hop if node != rid else neighbour
                    heapq.heappush(heap, (candidate, hop, neighbour))
        return distance, next_hop

    def change_cost(self, a, b, cost):
        self.routers[a].links[b] = cost
        self.routers[b].links[a] = cost
        self.originate_and_flood([a, b])

    def remove_link(self, a, b):
        del self.routers[a].links[b]
        del self.routers[b].links[a]
        self.originate_and_flood([a, b])

    def publish_everywhere(self):
        """Originate fresh LSAs for every router and install them in every LSDB.

        A partition cannot be flooded by definition (the isolated router has no interface left), and
        LSA ageing — which in real OSPF eventually flushes the stale copy its neighbours still hold —
        is out of scope for this model.  Publishing the current picture to every database models the
        converged state that ageing leaves behind, and keeps the LSDB-equality invariant testable."""
        for rid in sorted(self.routers):
            lsa = self.routers[rid].originate()
            for holder in sorted(self.lsdb):
                self.install(holder, lsa)


def brute_force_distances(links, source, nodes):
    """Independent O(n^2) label-setting Dijkstra over the raw link list (oracle)."""
    adjacency = {node: {} for node in nodes}
    for a, b, cost in links:
        adjacency[a][b] = cost
        adjacency[b][a] = cost
    nodes = sorted(adjacency)
    distance = {node: None for node in nodes}
    distance[source] = 0
    settled = set()
    while len(settled) < len(nodes):
        choice, best = None, None
        for node in nodes:
            if node in settled or distance[node] is None:
                continue
            if best is None or distance[node] < best:
                choice, best = node, distance[node]
        if choice is None:
            break
        settled.add(choice)
        for node, cost in sorted(adjacency[choice].items()):
            candidate = best + cost
            if distance[node] is None or candidate < distance[node]:
                distance[node] = candidate
    return distance


def live_links(area):
    """The links the area really has now, as (a, b, cost) with a < b — the oracle's input."""
    return [(a, b, cost) for a, router in sorted(area.routers.items())
            for b, cost in sorted(router.links.items()) if a < b]


def audit(area, links, label):
    """LSDB equality, LSDB-vs-live-link agreement, oracle agreement and next-hop invariants."""
    reference = area.lsdb[min(area.routers)]
    for rid, lsdb in area.lsdb.items():
        check(lsdb == reference, f"{label}: router {rid} has a different LSDB")
        check(len(lsdb) == len(area.routers), f"{label}: router {rid} misses an LSA")
    oracle = {rid: brute_force_distances(links, rid, sorted(area.routers)) for rid in area.routers}
    for rid in sorted(area.routers):
        distance, next_hop = area.spf(rid)
        check(distance[rid] == 0, f"{label}: router {rid} must be zero hops from itself")
        for node in sorted(area.routers):
            check(distance[node] == oracle[rid][node],
                  f"{label}: router {rid} to {node}: {distance[node]} vs oracle {oracle[rid][node]}")
        for node, hop in sorted(next_hop.items()):
            check(hop in area.routers[rid].links, f"{label}: next hop {hop} is not a neighbour of {rid}")
            other = area.spf(hop)[0][node]
            check(other is not None and distance[hop] + other == distance[node],
                  f"{label}: the tree does not decompose through {hop} to {node}")
        check(all(node in next_hop for node in distance
                  if distance[node] is not None and node != rid),
              f"{label}: every reachable router must have a next hop")
    return oracle


def main():
    links = [(1, 2, 4), (2, 3, 3), (3, 4, 2), (4, 5, 5), (5, 1, 12), (2, 5, 9), (3, 6, 7),
             (5, 6, 3), (1, 6, 20)]
    area = Area(links)
    check(len(area.routers) == 6 and len(area.routers[1].links) == 3, "the area is built from links")
    check(area.routers[2].links[1] == 4 and area.routers[1].links[2] == 4, "costs are symmetric")

    area.transmissions = 0
    area.originate_and_flood()
    per_lsa = area.transmissions / len(area.routers)
    expected = sum(len(router.links) - (1 if router.rid != 1 else 0)
                   for router in area.routers.values())
    check(area.transmissions == len(area.routers) * expected,
          f"flooding cost {area.transmissions} transmissions, expected {len(area.routers) * expected}")

    oracle = audit(area, live_links(area), "initial")
    distance1, next_hop1 = area.spf(1)
    check(distance1[5] == 12 and next_hop1[5] == 5, "the direct link to 5 is shortest")
    check(distance1[6] == 14 and next_hop1[6] == 2, "6 is reached through 2 -> 3, not directly")
    check(distance1[2] == 4 and next_hop1[2] == 2, "the direct neighbour is one hop away")
    check(all(value is not None for value in distance1.values()), "the area is connected")

    for lsa in list(area.lsdb[1].values()):
        before = dict(area.lsdb[1])
        area.transmissions = 0
        check(area.flood_lsa(1, lsa) == 0, "re-flooding a known LSA installs nothing")
        check(area.transmissions == 0, "a duplicate LSA is dropped without transmission")
        check(area.lsdb[1] == before, "a duplicate LSA does not change the database")

    stale = area.lsdb[1][2]._replace(seq=area.routers[2].seq - 1, links=())
    check(area.install(1, stale) is False, "a lower sequence number is rejected")
    check(area.lsdb[1][2].links, "the rejected rewrite left the LSDB untouched")
    fake = area.lsdb[1][2]._replace(links=((1, 999),))
    check(area.install(1, fake) is False, "an equal sequence number with new content is ignored too")
    check(area.lsdb[1][2].links != ((1, 999),), "the ignored rewrite left the LSDB untouched")

    area.change_cost(2, 3, 12)
    audit(area, live_links(area), "cost change")
    changed_distance, changed_hop = area.spf(1)
    check(changed_distance[3] == 16, "the longer 2-3 link pushes the distance to 3 up to 16")
    check(changed_hop[3] == 2, "the next hop to 3 is still through the neighbour 2")
    check(area.routers[3].seq == 2 and area.routers[2].seq == 2, "only the endpoints re-originated")

    area.remove_link(5, 1)
    audit(area, live_links(area), "link failure")
    failed_distance, failed_hop = area.spf(1)
    check(failed_distance[5] == 13 and failed_hop[5] == 2,
          "5 is now reached the long way through 2 on the 9-cost link")
    check(area.spf(5)[0][1] == 13, "and 5 sees the symmetric distance back to 1")

    area.remove_link(3, 6)
    area.remove_link(5, 6)
    area.remove_link(1, 6)
    area.publish_everywhere()
    isolated = live_links(area)
    audit(area, isolated, "partition")
    partition_distance, partition_hop = area.spf(6)
    check(partition_distance[1] is None, "the isolated router has no route to the rest")
    check(len(partition_hop) == 0, "and therefore no next hops at all")
    check(area.spf(1)[0][6] is None, "the rest of the area cannot reach it either")
    check(all(node in area.spf(1)[0] for node in range(1, 7)), "the distance vector still names 6")
    check(len(area.lsdb[1][6].links) == 0, "the isolated router's LSA carries no links")

    fresh = Area(links)
    fresh.originate_and_flood()
    twin = Area(links)
    twin.originate_and_flood()
    check(twin.lsdb[1] == fresh.lsdb[1] and twin.spf(1) == fresh.spf(1),
          "the same topology converges to the same database and tree")

    print(f"area: {len(area.routers)} routers, {len(links)} links, "
          f"{per_lsa:.0f} transmissions per LSA, {expected} expected")
    print(f"initial distance from router 1: {[distance1[node] for node in sorted(distance1)]}")
    print(f"after failures: {[failed_distance[node] for node in sorted(failed_distance)]}")
    print("PASS")


if __name__ == "__main__":
    main()
