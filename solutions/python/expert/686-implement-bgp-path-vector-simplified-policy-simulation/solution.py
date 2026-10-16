"""Task 686 - Implement BGP path-vector simplified policy simulation.  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

ASes exchange *path vectors*: a route carries the full AS_PATH, so a receiver can reject any update
that already contains its own ASN — the one mechanism that makes a path-vector protocol loop free
without global knowledge.  The simulation is faithful but deliberately scaled down:

  * one prefix per run, no iBGP, no route reflectors, no route dampening, no communities and no
    AS_PATH prepending; synchronised rounds stand in for timers, and the receiver's tie-break uses
    the neighbour ASN in place of the BGP router id.
  * the decision process has the real order: higher LOCAL_PREF, then shorter AS_PATH, then lower
    ORIGIN, then lower MULTI_EXIT_DISC — MED is compared **only between routes from the same
    neighbouring AS**, which is why the candidates are first reduced per sender.
  * export policy is Gao-Rexford: customers receive everything, while providers and peers receive
    only the AS's own routes and routes learned from its customers.  A route is never sent back to
    the neighbour it was learned from.

Verified invariants: convergence is a genuine fixed point (one extra round changes nothing); no
converged path repeats an ASN; every advertisement respects the export policy, so no AS transits
between two of its providers/peers (the "valley-free" property); LOCAL_PREF provably beats AS_PATH
length; the result is independent of the order in which neighbours are visited; and with policy
switched off the converged paths equal an exhaustive simple-path oracle (all simple paths from the
receiver to the origin, minimised by length then hop sequence) — an oracle that shares no code with
the distributed iteration.
"""

from collections import namedtuple

LOCAL_PREF_DEFAULT = 100
CUSTOMER, PROVIDER, PEER = "customer", "provider", "peer"

Route = namedtuple("Route", "prefix path next_hop origin local_pref med learned_from")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Router:
    def __init__(self, asn, gao_rexford=True):
        self.asn = asn
        self.router_id = asn              # stands in for the BGP router identifier
        self.peers = {}                   # neighbour asn -> relationship as seen from here
        self.import_local_pref = {}       # neighbour asn -> LOCAL_PREF for its routes
        self.denied = set()               # prefixes filtered on import
        self.export_denied = set()        # neighbours we never advertise to
        self.gao_rexford = gao_rexford
        self.originated = set()           # prefixes this AS originates itself

    def peer_with(self, other, relation):
        self.peers[other] = relation
        return self

    def local_routes(self, prefix):
        if prefix not in self.originated:
            return []
        return [Route(prefix, (self.asn,), self.asn, 0, LOCAL_PREF_DEFAULT, 0, None)]

    def import_route(self, neighbour, route):
        """Receiver-side processing: loop check, filters, then the local preference is set here."""
        if self.asn in route.path or route.prefix in self.denied:
            return None
        return Route(route.prefix, (self.asn,) + route.path, neighbour, route.origin,
                     self.import_local_pref.get(neighbour, LOCAL_PREF_DEFAULT), route.med,
                     neighbour)

    def export_route(self, neighbour, route):
        """Gao-Rexford: a provider or peer only hears our own routes and our customers' routes."""
        if neighbour in self.export_denied:
            return None
        if route.learned_from == neighbour:                  # never echo a route back
            return None
        if not self.gao_rexford or self.peers[neighbour] == CUSTOMER:
            return route
        if route.learned_from is None:
            return route
        return route if self.peers[route.learned_from] == CUSTOMER else None


def select_best(candidates):
    """BGP decision process.  MED is only compared between routes from the same neighbour AS, so the
    candidates are reduced per sender first; the final key is deterministic (last resort: next hop)."""
    if not candidates:
        return None
    per_sender = {}
    for route in candidates:
        sender = route.learned_from
        if sender not in per_sender or route.med < per_sender[sender].med:
            per_sender[sender] = route
    return min(per_sender.values(),
               key=lambda r: (-r.local_pref, len(r.path), r.origin, r.next_hop))


def round_step(routers, rib, prefix, order):
    """One synchronous round: everybody advertises the current RIB, then everybody re-selects."""
    advertisements = {}
    for sender in sorted(routers):
        route = rib[sender]
        if route is None:
            continue
        for receiver in order[sender]:
            sent = routers[sender].export_route(receiver, route)
            if sent is not None:
                advertisements[(sender, receiver)] = sent
    new_rib = {}
    for receiver in sorted(routers):
        candidates = list(routers[receiver].local_routes(prefix))
        for sender in order[receiver]:
            sent = advertisements.get((sender, receiver))
            if sent is None:
                continue
            imported = routers[receiver].import_route(sender, sent)
            if imported is not None:
                candidates.append(imported)
        new_rib[receiver] = select_best(candidates)
    return new_rib, advertisements


def simulate(routers, prefix, max_rounds=80, order=None):
    """Round-synchronous BGP.  Returns (rib, last advertisements, rounds used, converged flag)."""
    asns = sorted(routers)
    order = order or {asn: sorted(routers[asn].peers) for asn in asns}
    rib = {asn: None for asn in asns}
    advertisements, rounds, converged = {}, 0, False
    while rounds < max_rounds:
        rounds += 1
        new_rib, advertisements = round_step(routers, rib, prefix, order)
        if new_rib == rib:
            converged = True
            break
        rib = new_rib
    return rib, advertisements, rounds, converged


def enumerate_simple_paths(edges, source, target):
    """Every simple AS path from source to target; the oracle for the policy-free case."""
    paths, stack = [], [(source, (source,))]
    while stack:
        node, path = stack.pop()
        if node == target:
            paths.append(path)
            continue
        for nxt in sorted(edges.get(node, ())):
            if nxt not in path:
                stack.append((nxt, path + (nxt,)))
    return paths


def oracle_rib(routers, edges, prefix, origin):
    """Best simple path per AS, minimised by (length, hop sequence) — no shared code with simulate."""
    result = {}
    for asn in sorted(routers):
        if prefix in routers[asn].originated:
            result[asn] = (asn,)
            continue
        paths = enumerate_simple_paths(edges, asn, origin)
        result[asn] = min(paths, key=lambda p: (len(p), p[1:])) if paths else None
    return result


def valley_free(routers, path):
    """For each interior AS Y with neighbours X (told) and Z (told by): Y may pass Z's route to X
    only when X is Y's customer, or when the route came from a customer of Y."""
    for index in range(1, len(path) - 1):
        y = routers[path[index]]
        if y.peers[path[index - 1]] == CUSTOMER:
            continue
        upstream = path[index + 1]
        if y.peers.get(upstream) == CUSTOMER:
            continue
        return False
    return True


def check_advertisements(routers, advertisements):
    for (sender, receiver), route in sorted(advertisements.items()):
        y = routers[sender]
        if not y.gao_rexford or y.peers[receiver] == CUSTOMER:
            continue
        upstream = route.learned_from
        check(upstream is None or y.peers[upstream] == CUSTOMER,
              f"AS {sender} leaked a provider/peer route to {receiver}")


def two_tier_topology():
    """1 and 2 are settlement-free tier-1 peers; 3, 4, 5 buy transit; 6 buys from 3; 7 peers 3; 8 peers 1."""
    routers = {asn: Router(asn) for asn in (1, 2, 3, 4, 5, 6, 7, 8)}
    relation = [(3, 1, PROVIDER), (5, 1, PROVIDER), (4, 2, PROVIDER), (6, 3, PROVIDER),
                (7, 3, PEER), (8, 1, PEER), (1, 2, PEER)]
    for near, far, how in relation:
        routers[near].peer_with(far, how)
        routers[far].peer_with(near, {PROVIDER: CUSTOMER, CUSTOMER: PROVIDER, PEER: PEER}[how])
    routers[6].originated.add("10.0.0.0/8")
    routers[2].originated.add("20.0.0.0/8")
    return routers


def make_prng(seed):
    """splitmix64, so the shuffled round order is reproducible."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def main():
    routers = two_tier_topology()
    rib, advertisements, rounds, converged = simulate(routers, "10.0.0.0/8")
    check(converged, "the customer route converges")
    check(rib[6].path == (6,), "the origin keeps its own path")
    check(rib[3].path == (3, 6), "a provider learns its customer's prefix directly")
    check(rib[1].path == (1, 3, 6), "the tier-1 learns the prefix through its customer")
    check(rib[5].path == (5, 1, 3, 6), "a customer learns it from its provider")
    check(rib[2].path == (2, 1, 3, 6), "the peer tier-1 takes the customer route")
    check(rib[4].path == (4, 2, 1, 3, 6), "the far customer follows the ladder")
    check(rib[7].path == (7, 3, 6), "a peer of 3 hears a customer route")
    check(rib[8].path == (8, 1, 3, 6), "a peer of 1 hears a customer route")
    check_advertisements(routers, advertisements)
    for asn in sorted(routers):
        check(valley_free(routers, rib[asn].path), f"path of AS {asn} is valley-free")
        check(len(set(rib[asn].path)) == len(rib[asn].path), f"AS {asn} path has no AS loop")

    second, _, second_rounds, second_converged = simulate(routers, "10.0.0.0/8")
    check(second == rib and second_converged, "a fresh run reproduces the same converged RIB")
    fixed_rib, _ = round_step(routers, rib, "10.0.0.0/8",
                              {asn: sorted(routers[asn].peers) for asn in routers})
    check(fixed_rib == rib, "one more synchronous round changes nothing (genuine fixed point)")

    rib20, _, _, converged20 = simulate(routers, "20.0.0.0/8")
    check(converged20, "the peer-originated route converges")
    check(rib20[1].path == (1, 2), "AS 1 learns the prefix from its peer")
    check(rib20[3].path == (3, 1, 2), "a customer may receive a route learned from a peer")
    check(rib20[5].path == (5, 1, 2), "so may another customer")
    check(rib20[7] is None, "a provider/peer route is never passed to a peer (no transit)")
    check(rib20[8] is None, "the other peer of AS 1 is locked out the same way")
    check(rib20[4].path == (4, 2), "AS 2's own customer still gets the route directly")

    prng = make_prng(0x5EED)
    for _ in range(4):
        shuffled = {}
        for asn, router in routers.items():
            order = sorted(router.peers)
            for i in range(len(order) - 1, 0, -1):          # deterministically shuffled
                j = prng() % (i + 1)
                order[i], order[j] = order[j], order[i]
            shuffled[asn] = order
        other, _, _, ok = simulate(routers, "10.0.0.0/8", order=shuffled)
        check(ok and other == rib, "the converged RIB does not depend on the neighbour order")

    policy_free = {asn: Router(asn, gao_rexford=False) for asn in sorted(routers)}
    for asn, router in routers.items():
        for neighbour, relation in router.peers.items():
            policy_free[asn].peer_with(neighbour, relation)
    policy_free[6].originated.add("10.0.0.0/8")
    edges = {asn: sorted(router.peers) for asn, router in routers.items()}
    free_rib, _, _, free_ok = simulate(policy_free, "10.0.0.0/8")
    check(free_ok, "the policy-free run converges")
    expected = oracle_rib(policy_free, edges, "10.0.0.0/8", 6)
    for asn in sorted(routers):
        check(free_rib[asn].path == expected[asn],
              f"AS {asn}: converged path {free_rib[asn].path} but the oracle says {expected[asn]}")
    check(free_rib[7].path == (7, 3, 6), "without policy the ladder is the same on this topology")

    long_local_pref = {asn: Router(asn) for asn in (10, 20, 21, 25, 30)}
    links = [(10, 20, PROVIDER), (10, 21, PROVIDER), (20, 25, PROVIDER),
             (25, 30, PROVIDER), (21, 30, PROVIDER)]
    for near, far, how in links:
        long_local_pref[near].peer_with(far, how)
        long_local_pref[far].peer_with(near, {PROVIDER: CUSTOMER, CUSTOMER: PROVIDER,
                                              PEER: PEER}[how])
    long_local_pref[30].originated.add("30.0.0.0/8")
    plain_rib, _, _, _ = simulate(long_local_pref, "30.0.0.0/8")
    check(plain_rib[10].path == (10, 21, 30), "with equal preference the shorter path wins")
    long_local_pref[10].import_local_pref = {20: 200, 21: 100}
    prefer_rib, _, _, _ = simulate(long_local_pref, "30.0.0.0/8")
    check(prefer_rib[10].path == (10, 20, 25, 30),
          "LOCAL_PREF outranks AS_PATH length in the decision process")
    check(len(prefer_rib[10].path) > len(plain_rib[10].path), "the preferred path really is longer")

    same_sender = [Route("p", (1, 2, 3), 2, 0, 100, 50, 2), Route("p", (1, 2, 3), 2, 0, 100, 5, 2)]
    check(select_best(same_sender).med == 5,
          "MED decides between two routes from the same neighbouring AS")
    med_across = [Route("p", (1, 2, 3), 2, 0, 100, 1, 2), Route("p", (1, 3), 3, 0, 100, 9, 3)]
    check(select_best(med_across).path == (1, 3),
          "MED from another neighbour is ignored once the AS_PATH is shorter")
    origin_tie = [Route("p", (1, 2, 3), 2, 2, 100, 0, 2), Route("p", (1, 3), 3, 0, 100, 0, 3)]
    check(select_best(origin_tie).path == (1, 3), "a lower ORIGIN code wins at equal length")
    check(select_best([]) is None, "no candidate means no route")

    filtered = {asn: Router(asn) for asn in (1, 2)}
    filtered[1].peer_with(2, CUSTOMER)
    filtered[2].peer_with(1, PROVIDER)
    filtered[1].originated.add("1.0.0.0/8")
    filtered[2].denied.add("1.0.0.0/8")
    deny_rib, _, _, _ = simulate(filtered, "1.0.0.0/8")
    check(deny_rib[1].path == (1,) and deny_rib[2] is None, "an import filter removes the route")
    check(filtered[2].import_route(1, Route("1.0.0.0/8", (2, 1), 1, 0, 100, 0, 1)) is None,
          "an AS_PATH that already contains the receiver is rejected")
    check(filtered[1].export_route(2, Route("x", (2, 1), 1, 0, 100, 0, 2)) is None,
          "a route is not echoed back to the neighbour it came from")

    empty_rib, _, empty_rounds, empty_ok = simulate(routers, "99.0.0.0/8")
    check(empty_ok and empty_rounds == 1 and all(route is None for route in empty_rib.values()),
          "nobody originates 99.0.0.0/8, so the run converges empty in one round")

    print(f"two-tier run: {rounds} rounds, {len(advertisements)} advertisements in the last round")
    print(f"policy-free run: {free_ok and max(len(free_rib[a].path) for a in routers)} hop longest path")
    print(f"LOCAL_PREF preferring a longer path: {prefer_rib[10].path} vs {plain_rib[10].path}")
    print("PASS")


if __name__ == "__main__":
    main()
