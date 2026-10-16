"""Task 903 - Implement adaptive replication factoring hot keys and node load.
(Tier: Master, CTCI Ch 9 - System Design and Scalability)

What is implemented (in-process, deterministic; sha256-derived tokens are the only "randomness"):

  * `AdaptiveCluster.primary` - consistent hashing with virtual nodes: a key is owned by the
    first ring point clockwise of its token, so joining or leaving a node moves only ~1/N of
    the keys (the classic minimal-disruption property, checked below against keys/N).
  * `plan` - adaptive replication factoring hot keys and node load.  The base placement gives
    every key one replica at its primary owner; every key whose access count clears
    `hot_threshold` may be replicated up to `replica_cap` distinct nodes, and replicas are only
    granted while the *peak* read load strictly decreases, within a global replica budget.
    Read load is modelled as an even split of a key's traffic across its replicas:
        load(n) = sum over keys holding n of weight(key) / |replicas(key)|
    which is what makes a replica pay off for a hot key while a cold key's replica only adds
    write amplification.  All arithmetic is integer (weights are scaled by SCALE = lcm(1..4)),
    so the report is exact and reproducible.

Simplified versus production: nodes are modelled purely by their read load - no network
topology, no failure detection, no concurrent membership churn; the access weights are a fixed
deterministic Zipf-like trace rather than a live measurement.  The invariants verified are the
real ones: replica sets stay distinct and inside the cap/budget, cold keys are never replicated,
the plan is a fixed point of the search, and the achieved peak load is within a small factor of
the exhaustive optimum on a tiny instance.

Verified below: ring determinism and the ~1/N movement bound on join/leave, the hot/cold split,
budget and cap accounting, peak-load improvement with a lower-bound sanity check, the
brute-force optimality oracle, a negative control without replication, idempotence, and the
single-node / empty-key / all-cold boundaries.
"""

import bisect
import hashlib
import itertools

SCALE = 2520                     # divisible by 1, 2, 3 and 4: exact share arithmetic
HOT_RATIO = 4                    # a key is hot when its accesses clear 4x the mean


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def ring_points(nodes, vnodes, salt="ring"):
    points = []
    for node in sorted(nodes):
        for v in range(vnodes):
            digest = hashlib.sha256(("%s|%s|%d" % (salt, node, v)).encode()).digest()
            points.append((int.from_bytes(digest[:8], "big"), node))
    points.sort()
    return points


class AdaptiveCluster:
    """Consistent-hash placement plus a greedy hot-key replication search."""

    def __init__(self, nodes, vnodes=32, salt="ring"):
        self.nodes = sorted(nodes)
        self.points = ring_points(self.nodes, vnodes, salt)
        self.tokens = [point[0] for point in self.points]

    def primary(self, key):
        token = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        return self.points[bisect.bisect_left(self.tokens, token) % len(self.points)][1]

    def base_load(self, weights):
        load = {node: 0 for node in self.nodes}
        for key, weight in weights.items():
            load[self.primary(key)] += weight * SCALE
        return load

    def plan(self, weights, hot_threshold, replica_cap=3, extra_budget=10 ** 9, placement=None):
        """Returns (placement, read_load, replicas_added); replicas are granted only while the
        peak load strictly decreases, which also makes the result a fixed point of the search."""
        cap = min(replica_cap, len(self.nodes))
        if placement is None:
            placement = {key: {self.primary(key)} for key in sorted(weights)}
        placement = {key: set(holders) for key, holders in placement.items()}
        load = {node: 0 for node in self.nodes}
        for key, holders in placement.items():
            share = weights[key] * SCALE // len(holders)
            for node in holders:
                load[node] += share
        added = 0
        while added < extra_budget:
            best = None
            for key in sorted(weights):
                held = placement[key]
                if weights[key] < hot_threshold or len(held) >= cap:
                    continue
                share = weights[key] * SCALE // len(held)
                grown = weights[key] * SCALE // (len(held) + 1)
                shifted = [load[node] - share + grown for node in held]
                for node in self.nodes:
                    if node in held:
                        continue
                    untouched = [load[other] for other in self.nodes
                                 if other not in held and other != node]
                    peak = max(untouched + shifted + [load[node] + grown])
                    candidate = (peak, key, node)
                    if best is None or candidate < best:
                        best = candidate
            if best is None or best[0] >= max(load.values()):
                break
            _, key, node = best
            held = placement[key]
            share = weights[key] * SCALE // len(held)
            grown = weights[key] * SCALE // (len(held) + 1)
            for holder in held:
                load[holder] = load[holder] - share + grown
            load[node] += grown
            placement[key].add(node)
            added += 1
        return placement, load, added


def brute_force_optimum(nodes, weights, replica_cap):
    """Exhaustive oracle: the minimum achievable peak load over every legal replica assignment."""
    cap = min(replica_cap, len(nodes))
    keys = sorted(weights)
    options = [[frozenset(combo)
                for size in range(1, cap + 1)
                for combo in itertools.combinations(nodes, size)] for _ in keys]
    best = None
    for combination in itertools.product(*options):
        load = {node: 0 for node in nodes}
        for key, holders in zip(keys, combination):
            share = weights[key] * SCALE // len(holders)
            for node in holders:
                load[node] += share
        peak = max(load.values())
        if best is None or peak < best:
            best = peak
    return best


def workload(keys, spread):
    """Deterministic Zipf-like trace: rank comes from the key hash, weight ~ spread / rank."""
    ranked = sorted(keys, key=lambda k: hashlib.sha256(k.encode()).digest())
    return {key: max(1, spread // (rank + 1)) for rank, key in enumerate(ranked)}


def main():
    nodes = ["node-%d" % i for i in range(8)]
    keys = ["key-%04d" % i for i in range(240)]
    weights = workload(keys, 4000)
    mean = sum(weights.values()) / len(weights)
    threshold = int(HOT_RATIO * mean)
    hot = [k for k in keys if weights[k] >= threshold]
    check(hot and len(hot) < len(keys), "the trace must have both hot and cold keys")

    cluster = AdaptiveCluster(nodes)
    # ---- 1. ring placement: total, deterministic and stable
    primaries = {k: cluster.primary(k) for k in keys}
    check(all(p in nodes for p in primaries.values()), "every key must land on a ring node")
    check(primaries == {k: AdaptiveCluster(nodes).primary(k) for k in keys},
          "placement must be reproducible from the ring alone")
    check(len(cluster.points) == len(nodes) * 32, "each node must own its virtual nodes")

    # ---- 2. minimal disruption: joining/leaving a node moves about 1/N of the keys
    joined = AdaptiveCluster(nodes + ["node-new"])
    moved = [k for k in keys if joined.primary(k) != primaries[k]]
    expected = len(keys) / len(joined.nodes)
    check(expected * 0.5 <= len(moved) <= expected * 1.5,
          "a join must move ~1/N of the keys, moved %d vs %.1f" % (len(moved), expected))
    left = AdaptiveCluster([n for n in nodes if n != "node-3"])
    moved_out = [k for k in keys if left.primary(k) != primaries[k]]
    check(len(moved_out) <= len(keys) / len(nodes) * 1.5,
          "a leave must move at most ~1/N of the keys")
    check(all(primaries[k] == "node-3" for k in moved_out),
          "only keys owned by the departing node may move")

    # ---- 3. base placement without replication keeps the hot-key skew
    base_load = cluster.base_load(weights)
    base_peak = max(base_load.values())
    check(sum(base_load.values()) == sum(weights.values()) * SCALE,
          "the base placement must account for all traffic exactly once")
    placement, load, added = cluster.plan(weights, threshold, replica_cap=3, extra_budget=60)
    peak = max(load.values())

    # ---- 4. the adaptive plan: only hot keys replicated, cold keys untouched, invariants hold
    replicated = [key for key in keys if len(placement[key]) > 1]
    for key in keys:
        holders = placement[key]
        check(1 <= len(holders) <= 3, "replica count out of range for %s" % key)
        check(holders <= set(nodes), "a replica must live on a real node")
        if weights[key] < threshold:
            check(holders == {primaries[key]}, "cold key %s must not be replicated" % key)
    check(len(replicated) >= 3, "the skew must make several hot keys worth replicating")
    check(all(weights[key] >= threshold for key in replicated),
          "no budget may be spent on cold keys")
    check(sum(len(placement[k]) - 1 for k in keys) == added,
          "one budget unit per extra replica: %d" % added)
    check(len(replicated) <= added, "each replicated key must consume at least one budget unit")
    check(added >= 3, "the skew must make replication worth several budget units")
    check(added <= 60, "the replica budget must be respected, added %d" % added)
    check(peak < base_peak, "replication must reduce the peak load")
    check(peak <= base_peak * 0.85,
          "the peak must drop by at least 15%% (was %d, now %d)" % (base_peak, peak))
    spread_before = sum(v * v for v in base_load.values()) // len(nodes)
    spread_after = sum(v * v for v in load.values()) // len(nodes)
    check(spread_after < spread_before,
          "replication must even the load out (spread %d -> %d)" % (spread_before, spread_after))

    # ---- 5. the metric cannot beat the fundamental lower bound
    lower = max(sum(weights.values()) * SCALE // len(nodes),
                max(w * SCALE for w in weights.values()) // len(nodes))
    check(lower <= peak, "the reported peak cannot be below the theoretical optimum")

    # ---- 6. exhaustive oracle on a tiny instance
    small_nodes = ["n0", "n1", "n2"]
    small_weights = {"a": 900, "b": 300, "c": 120, "d": 40}
    small = AdaptiveCluster(small_nodes, vnodes=8)
    small_plan, small_load, _ = small.plan(small_weights, hot_threshold=1, replica_cap=3,
                                           extra_budget=10 ** 9)
    optimum = brute_force_optimum(small_nodes, small_weights, 3)
    greedy_peak = max(small_load.values())
    check(greedy_peak >= optimum, "no assignment may beat the exhaustive optimum")
    check(greedy_peak <= optimum * 5 // 4,
          "greedy peak %d must be within 25%% of the optimum %d" % (greedy_peak, optimum))

    # ---- 7. negative control: without replication the skew stays
    no_rep, no_rep_load, none_added = cluster.plan(weights, threshold, replica_cap=1)
    check(none_added == 0 and max(no_rep_load.values()) == base_peak,
          "a cap of 1 must leave the placement untouched")
    check(base_peak > peak * 1.15, "the negative control needs a genuinely skewed trace")

    # ---- 8. fixed point: resuming the search from the converged placement must add nothing
    again, reload, added_again = cluster.plan(weights, threshold, 3, 60, placement=placement)
    check(added_again == 0, "the converged plan must be a fixed point of the search")
    check(reload == load and again == placement, "resuming must reproduce the same load map")
    reserved, reserved_load, _ = cluster.plan(weights, threshold, 3, 0)
    check(reserved == no_rep and reserved_load == base_load,
          "a zero budget must reproduce the base placement")
    check(max(reserved_load.values()) == base_peak, "a zero budget leaves the peak untouched")

    # ---- 9. boundaries: one node, all-cold keys, empty key set
    solo = AdaptiveCluster(["only"], vnodes=4)
    solo_plan, solo_load, solo_added = solo.plan(weights, threshold, replica_cap=3, extra_budget=10)
    check(solo_added == 0 and all(solo_plan[k] == {"only"} for k in keys),
          "a single-node cluster cannot replicate: the cap must clamp to 1")
    check(solo_load["only"] == sum(weights.values()) * SCALE, "the solo node carries everything")
    cold, cold_load, cold_added = cluster.plan(weights, 10 ** 9, replica_cap=3, extra_budget=10)
    check(cold_added == 0 and max(cold_load.values()) == base_peak,
          "with no hot keys nothing may be replicated")
    empty_plan, empty_load, empty_added = cluster.plan({}, threshold)
    check(empty_plan == {} and empty_added == 0 and set(empty_load.values()) == {0},
          "an empty key set must produce an empty plan")

    # ---- 10. determinism across identical cluster descriptions
    twin = AdaptiveCluster(list(reversed(nodes)), vnodes=32)
    check(twin.plan(weights, threshold, 3, 60)[0] == placement, "node order must not matter")
    print("base peak %d -> adaptive peak %d (%d replicas added, %d hot keys)"
          % (base_peak, peak, added, len(hot)))
    print("tiny instance: greedy %d vs exhaustive optimum %d" % (greedy_peak, optimum))
    print("PASS")


if __name__ == "__main__":
    main()
