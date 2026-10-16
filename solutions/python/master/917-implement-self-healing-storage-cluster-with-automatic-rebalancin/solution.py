"""Task 917 - Implement self-healing storage cluster with automatic rebalancing and throttled repairs.
(Tier: Master, CTCI Ch 9 - System Design and Scalability)

What is implemented (in-process, deterministic; failures, sizes and rounds are explicit, so a
whole incident is replayable):

  * `placement` - objects mapped to replica sets on distinct nodes.  Every mutation is followed by
    `verify_placement`, an independent re-derivation of the safety invariants from the raw mapping
    (distinct replicas, known nodes, at least one live replica, never more than the replication
    factor), so a broken state can never pass unnoticed.
  * `repair_round` - self-healing under a throttle: replicas lost to down nodes are re-created on
    the least loaded healthy node that does not already hold the object, at most `budget`
    migrations per round, and an object whose every replica is unreachable is reported rather than
    silently dropped.
  * `rebalance` - the same budget also funds rebalancing: a replica is moved off an overloaded node
    onto the lightest healthy node, keeping the replication factor intact at every step.

Simplified versus production: byte-weighted load only, no network partitions, no rack awareness,
instantaneous transfers, and node failures arrive on a fixed schedule instead of from monitoring.

Verified below: the safety invariants after every mutation, throttled repair progress and
convergence to the target replication factor, zero data loss throughout an incident, load balance
improvement with the 1+alpha bound, the zero-budget stall (reported, not hidden), the
single-node / excessive-factor / empty-cluster clamping, and full determinism.
"""

ALPHA = 0.25


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def verify_placement(placement, nodes, down, factor, objects):
    """Independent re-derivation of every safety invariant; returns the list of violations."""
    problems = []
    for oid in objects:
        replicas = placement.get(oid)
        if not replicas:
            problems.append("%s: no replica at all" % oid)
            continue
        if len(set(replicas)) != len(replicas):
            problems.append("%s: duplicate replica" % oid)
        if any(node not in nodes for node in replicas):
            problems.append("%s: replica on an unknown node" % oid)
        if len(replicas) > factor:
            problems.append("%s: %d replicas exceed the factor %d" % (oid, len(replicas), factor))
        if not [node for node in replicas if node not in down]:
            problems.append("%s: every replica is unreachable" % oid)
    return problems


def live_replicas(placement, oid, down):
    return [node for node in placement[oid] if node not in down]


def loads_of(placement, objects, down, nodes):
    loads = {node: 0 for node in nodes}
    for oid, replicas in placement.items():
        for node in replicas:
            if node not in down:
                loads[node] += objects[oid]
    return loads


def initial_placement(nodes, objects, factor, sizes):
    """Round-robin over the healthy nodes, one replica per node per object."""
    factor = min(factor, len(nodes))
    placement, cursor = {}, 0
    for oid in sorted(objects, key=lambda item: (-sizes[item], item)):
        chosen = [nodes[(cursor + step) % len(nodes)] for step in range(factor)]
        cursor += 1
        placement[oid] = set(chosen)
    return placement


def repair_round(placement, nodes, down, objects, factor, budget, sizes):
    """One throttled round: repairs first, then rebalancing, at most `budget` moves in total."""
    healthy = [node for node in nodes if node not in down]
    loads = loads_of(placement, objects, down, nodes)
    moves, unreachable = [], []
    budget_left = budget

    for oid in sorted(objects):
        live = live_replicas(placement, oid, down)
        if not live:
            unreachable.append(oid)
            continue
        missing = factor - len(live)
        for _ in range(max(0, missing)):
            if budget_left <= 0:
                break
            candidates = [node for node in healthy if node not in placement[oid]]
            if not candidates:
                break
            target = min(candidates, key=lambda node: (loads[node], node))
            placement[oid].add(target)
            loads[target] += sizes[oid]
            moves.append(("repair", oid, target))
            budget_left -= 1

    while budget_left > 0:
        average = sum(loads.values()) / max(1, len(healthy))
        busiest = max(healthy, key=lambda node: (loads[node], node))
        if loads[busiest] <= average * (1.0 + ALPHA) or loads[busiest] == 0:
            break
        moved = False
        for oid in sorted(objects):
            if busiest not in placement[oid] or len(live_replicas(placement, oid, down)) <= 1:
                continue
            candidates = [node for node in healthy
                          if node not in placement[oid] and loads[node] < loads[busiest]]
            if not candidates:
                continue
            target = min(candidates, key=lambda node: (loads[node], node))
            placement[oid].discard(busiest)
            placement[oid].add(target)
            loads[busiest] -= sizes[oid]
            loads[target] += sizes[oid]
            moves.append(("rebalance", oid, target))
            budget_left -= 1
            moved = True
            break
        if not moved:
            break
    return moves, unreachable, loads


def simulate(nodes, objects, factor, sizes, failures, rounds, budget):
    """Run the incident: apply failures, then heal; return the per-round telemetry."""
    down = set()
    placement = initial_placement(nodes, objects, factor, sizes)
    check(verify_placement(placement, nodes, down, factor, objects) == [],
          "the initial placement must satisfy every invariant")
    telemetry = []
    for round_index in range(rounds):
        for node in failures.get(round_index, ()):
            down.add(node)
            for oid in objects:
                placement[oid].discard(node)
            check(verify_placement(placement, nodes, down, factor, objects) == [],
                  "a failure may not break a safety invariant")
        moves, unreachable, loads = repair_round(placement, nodes, down, objects, factor,
                                                 budget, sizes)
        check(len(moves) <= budget,
              "round %d issued %d moves for a budget of %d" % (round_index, len(moves), budget))
        check(verify_placement(placement, nodes, down, factor, objects) == [],
              "a repair round broke an invariant: %r"
              % verify_placement(placement, nodes, down, factor, objects))
        deficits = sum(max(0, factor - len(live_replicas(placement, oid, down))) for oid in objects)
        peak = max(loads.values())
        average = sum(loads.values()) / max(1, len([n for n in nodes if n not in down]))
        telemetry.append({"round": round_index, "moves": len(moves), "deficit": deficits,
                          "ratio": peak / average if average else 0.0, "unreachable": unreachable})
    return placement, down, telemetry, loads


def main():
    objects = {"obj-%02d" % index: 32 + 8 * index for index in range(24)}
    sizes = dict(objects)
    factor, budget = 3, 5
    nodes = ["n%d" % index for index in range(8)]
    per_node = sum(sizes.values()) * factor / len(nodes)

    # ---- 1. a clean cluster, then one node lost
    placement, down, telemetry, loads = simulate(nodes, objects, factor, sizes, {}, 1, budget)
    check(len(placement) == len(objects) and all(len(r) == factor for r in placement.values()),
          "every object must start with three replicas")
    check(max(loads.values()) <= per_node * (1.0 + ALPHA) + max(sizes.values()),
          "the initial round-robin placement must be reasonably balanced")
    incidents = {3: ("n3",), 4: ("n6",), 9: ("n1", "n7"), 12: ("n3",)}
    placement, down, telemetry, loads = simulate(nodes, objects, factor, sizes, incidents, 14, budget)
    check(all(row["moves"] <= budget for row in telemetry), "the throttle must hold every round")
    check(telemetry[3]["deficit"] > 0, "losing a node must create a replication deficit")
    check(telemetry[6]["deficit"] == 0 and telemetry[13]["deficit"] == 0,
          "the deficit must be repaired within a couple of rounds: %r"
          % [row["deficit"] for row in telemetry])
    drops = [telemetry[i]["deficit"] - telemetry[i + 1]["deficit"]
             for i in range(len(telemetry) - 1)]
    check(all(drop <= budget for drop in drops),
          "one round may never repair more than the budget: %r" % drops)
    check(all(not row["unreachable"] for row in telemetry),
          "with at most two simultaneous failures no object may become unreachable")

    # ---- 2. the repair is bounded, and progress is guaranteed while the budget lasts
    check(sum(row["moves"] for row in telemetry) > 10, "the incident must actually need repairs")
    check(all(len(live_replicas(placement, oid, down)) == factor for oid in objects),
          "the cluster must end fully replicated")
    check(all(len(replicas) == factor for replicas in placement.values()),
          "the invariant checker already proved every object has a live replica")
    check(telemetry[-1]["ratio"] <= 1.0 + ALPHA + 1e-9,
          "the converged cluster must respect the balance bound (%.3f)" % telemetry[-1]["ratio"])
    repaired_round = [row for row in telemetry if row["moves"] > 0]
    check(repaired_round[0]["deficit"] > repaired_round[-1]["deficit"],
          "the deficit must shrink as the budget is spent")

    # ---- 3. balance: after the incident the load ratio must beat the pre-repair one
    skewed = initial_placement(nodes, objects, factor, sizes)
    for oid in ("obj-00", "obj-01", "obj-02", "obj-03"):
        skewed[oid] = {"n0", "n1", "n2"}
    before = loads_of(skewed, objects, set(), nodes)
    ratio_before = max(before.values()) / (sum(before.values()) / len(nodes))
    moves, _, after = repair_round(skewed, nodes, set(), objects, factor, budget * 40, sizes)
    ratio_after = max(after.values()) / (sum(after.values()) / len(nodes))
    check(ratio_after < ratio_before,
          "rebalancing must reduce the peak ratio (%.3f -> %.3f)" % (ratio_before, ratio_after))
    check(ratio_after <= 1.0 + ALPHA + 1e-9,
          "the converged ratio must respect 1+alpha (%.3f)" % ratio_after)
    check(any(kind == "rebalance" for kind, _, _ in moves), "the rebalancer must have moved data")
    check(verify_placement(skewed, nodes, set(), factor, objects) == [],
          "rebalancing must preserve the invariants")
    check(sum(len(r) for r in skewed.values()) == factor * len(objects),
          "rebalancing must not change the replica count")

    # ---- 4. a node that comes back empty must receive replicas again
    rejoin = initial_placement(nodes, objects, factor, sizes)
    for replicas in rejoin.values():
        replicas.discard("n7")               # n7 has just rejoined and holds nothing
    before_join = loads_of(rejoin, objects, set(), nodes)
    moves, _, after_join = repair_round(rejoin, nodes, set(), objects, factor, budget * 40, sizes)
    check(before_join["n7"] == 0 and after_join["n7"] > 0,
          "a rejoining node must be used again (%.0f bytes)" % after_join["n7"])
    check(all(len(replicas) == factor for replicas in rejoin.values()),
          "the rejoin must restore the replication factor")
    check(telemetry[-1]["ratio"] <= 1.0 + ALPHA + 1e-9,
          "the converged cluster must respect the balance bound (%.3f)" % telemetry[-1]["ratio"])

    # ---- 5. the zero-budget stall is reported, not hidden
    stalled = initial_placement(nodes, objects, factor, sizes)
    frozen_down = {"n0"}
    moves, unreachable, _ = repair_round(stalled, nodes, frozen_down, objects, factor, 0, sizes)
    check(moves == [] and unreachable == [],
          "a zero budget must stall repairs rather than break invariants")
    check(verify_placement(stalled, nodes, frozen_down, factor, objects) == [],
          "a stalled cluster is still safe")
    deficit = sum(max(0, factor - len(live_replicas(stalled, oid, frozen_down))) for oid in objects)
    check(deficit > 0, "the stall must be visible as a replication deficit (%d)" % deficit)
    throttled, _, _ = repair_round(stalled, nodes, frozen_down, objects, factor, 2, sizes)
    check(len(throttled) == 2, "a budget of two must issue exactly two repairs when work exists")

    # ---- 6. boundaries: clamping, empty cluster, unreachable object
    solo = initial_placement(["only"], objects, 3, sizes)
    check(all(replicas == {"only"} for replicas in solo.values()),
          "a single-node cluster must clamp the factor to one")
    check(verify_placement(solo, ["only"], set(), 1, objects) == [],
          "the clamped placement must satisfy the factor-one invariants")
    empty = initial_placement(nodes, {}, 3, {})
    check(empty == {} and verify_placement(empty, nodes, set(), 3, {}) == [],
          "an empty object set is trivially safe")
    lonely = {"obj-00": set()}
    check(verify_placement(lonely, nodes, {"n0"}, 3, {"obj-00": 1})
          == ["obj-00: no replica at all"],
          "an object whose every replica is gone must be reported")
    moves, unreachable, _ = repair_round(lonely, nodes, {"n0"}, {"obj-00": 1}, 3, 4,
                                         {"obj-00": 1})
    check(moves == [] and unreachable == ["obj-00"],
          "an unrecoverable object must be reported, never invented")

    # ---- 7. determinism
    twin_placement, _, twin_telemetry, _ = simulate(nodes, objects, factor, sizes, incidents, 14, budget)
    check(twin_telemetry == telemetry and twin_placement == placement,
          "the whole incident must be reproducible")

    print("rounds %d, moves %d, final deficit %d"
          % (len(telemetry), sum(row["moves"] for row in telemetry), telemetry[-1]["deficit"]))
    print("ratio before %.3f -> after %.3f (bound %.2f)" % (ratio_before, ratio_after, 1 + ALPHA))
    print("moves by round %r" % [row["moves"] for row in telemetry])
    print("PASS")


if __name__ == "__main__":
    main()
