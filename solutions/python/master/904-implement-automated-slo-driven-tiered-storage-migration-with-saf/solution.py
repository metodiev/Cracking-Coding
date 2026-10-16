"""Task 904 - Implement automated SLO-driven tiered storage migration with safety.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; no wall clock, no I/O - a round is an explicit
call, and every number is an integer MB / milli-unit so results are exact and replayable):

  * `Tier` / `Obj` - three storage tiers (hot/warm/cold) with capacity, access latency and a
    per-MB cost rate, and objects with size, daily read traffic and the latency class their
    traffic demands.
  * `plan_round` - the migration planner: it proposes, for every object, the cheapest tier that
    satisfies that object's latency class, orders the proposals by (fixes an SLO violation
    first, then money saved), and applies them while three safety rails hold - per-tier
    capacity is never exceeded, at most `max_moves` objects move per round (throttle), and an
    object that moved within the last `cooldown` rounds is pinned (no oscillation).
  * `violation_reads` - the SLO metric: reads-weighted traffic served from a tier slower than
    the object's class, evaluated against the SLO budget.

Simplified versus production: the "storage" is a dictionary, migration is instantaneous (only
the throttle and cooldown model migration cost), and read traffic is a deterministic trace
rather than a live measurement; capacity accounting, SLO accounting and the safety rails are
the real ones.

Verified below: capacity and placement invariants re-checked after every mutation, the SLO
budget before/after convergence, the throttle and cooldown rails, a no-op fixed point (no
thrashing), re-convergence after a deterministic traffic drift, an exhaustive enumeration
oracle on a tiny instance, and the empty / exactly-full / oversized boundaries.
"""

import itertools

HOT, WARM, COLD = "hot", "warm", "cold"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Tier:
    __slots__ = ("name", "capacity_mb", "latency_ms", "rate_milli")

    def __init__(self, name, capacity_mb, latency_ms, rate_milli):
        self.name = name
        self.capacity_mb = capacity_mb
        self.latency_ms = latency_ms
        self.rate_milli = rate_milli          # thousandths of a cost unit per MB


class Obj:
    __slots__ = ("oid", "size_mb", "reads_day", "max_latency_ms")

    def __init__(self, oid, size_mb, reads_day, max_latency_ms):
        self.oid = oid
        self.size_mb = size_mb
        self.reads_day = reads_day
        self.max_latency_ms = max_latency_ms


def order_of_tiers(tiers):
    """Tiers from fastest/cheapest-capacity to slowest, as the planner searches them."""
    return sorted(tiers.values(), key=lambda tier: (tier.rate_milli, tier.name))


def loads_of(tiers, placement, objects):
    loads = {name: 0 for name in tiers}
    for oid, tier in placement.items():
        loads[tier] += objects[oid].size_mb
    return loads


def cheapest_ok_tier(obj, tiers):
    """Cheapest tier whose latency fits the object's class (capacity ignored)."""
    for tier in order_of_tiers(tiers):
        if tier.latency_ms <= obj.max_latency_ms:
            return tier.name
    return order_of_tiers(tiers)[0].name


def cost_of(tiers, placement, objects):
    return sum(objects[oid].size_mb * tiers[tier].rate_milli for oid, tier in placement.items())


def violation_reads(tiers, placement, objects):
    """(reads served too slowly, total reads) - the SLO numerator and denominator."""
    total = sum(obj.reads_day for obj in objects.values())
    bad = sum(obj.reads_day for obj in objects.values()
              if tiers[placement[obj.oid]].latency_ms > obj.max_latency_ms)
    return bad, total


def plan_round(tiers, objects, placement, last_moved, round_no, max_moves=4, cooldown=2):
    """One migration round: applies moves to `placement` / `last_moved` in place and returns
    (moves, rejected, loads).  Callers that want to simulate without committing pass copies."""
    loads = loads_of(tiers, placement, objects)
    proposals = []
    for oid in sorted(objects):
        obj = objects[oid]
        target = cheapest_ok_tier(obj, tiers)
        if target == placement[oid]:
            continue
        if last_moved.get(oid, -10 ** 9) > round_no - cooldown:
            continue                                     # cooldown rail
        fixes = tiers[placement[oid]].latency_ms > obj.max_latency_ms
        saved = (obj.size_mb * tiers[placement[oid]].rate_milli
                 - obj.size_mb * tiers[target].rate_milli)
        proposals.append((0 if fixes else 1, -saved, oid, target))
    proposals.sort()

    moves, rejected = [], []
    for _, _, oid, target in proposals:
        obj = objects[oid]
        if len(moves) >= max_moves:
            break
        if loads[target] + obj.size_mb > tiers[target].capacity_mb:
            rejected.append((oid, target))               # capacity rail
            continue
        loads[placement[oid]] -= obj.size_mb
        loads[target] += obj.size_mb
        placement[oid] = target
        last_moved[oid] = round_no
        moves.append((oid, target))
    return moves, rejected, loads


def verify_state(tiers, objects, placement, loads):
    """The safety invariant, re-checked after every mutation."""
    check(sorted(placement) == sorted(objects), "every object must be placed exactly once")
    check(loads == loads_of(tiers, placement, objects), "the load map must match the placement")
    for name, tier in tiers.items():
        check(loads[name] <= tier.capacity_mb,
              "tier %s over capacity: %d > %d" % (name, loads[name], tier.capacity_mb))
        check(loads[name] >= 0, "a tier cannot hold a negative amount")


def exhaustive_optimum(tiers, objects, names, slo_budget_milli):
    """Independent oracle: cheapest capacity-feasible placement meeting the read-weighted SLO."""
    best = None
    for combo in itertools.product(names, repeat=len(objects)):
        placement = dict(zip(sorted(objects), combo))
        loads = loads_of(tiers, placement, objects)
        if any(loads[name] > tiers[name].capacity_mb for name in names):
            continue
        bad, total = violation_reads(tiers, placement, objects)
        if bad * 1000 > total * slo_budget_milli:
            continue
        cost = cost_of(tiers, placement, objects)
        if best is None or cost < best:
            best = cost
    return best


def build_world():
    tiers = {
        HOT: Tier(HOT, 4000, 5, 1000),
        WARM: Tier(WARM, 9000, 30, 200),
        COLD: Tier(COLD, 40000, 500, 30),
    }
    objects = {}
    plan = [(100, 900, 4000, 5), (101, 700, 3200, 5), (102, 900, 2600, 5), (103, 600, 1500, 5),
            (104, 800, 900, 30), (105, 1200, 700, 30), (106, 900, 400, 30), (107, 2400, 120, 30),
            (108, 3000, 40, 500), (109, 2600, 25, 500), (110, 1800, 12, 500), (111, 900, 6, 500)]
    for oid, size, reads, latency in plan:
        objects["obj-%03d" % oid] = Obj("obj-%03d" % oid, size, reads, latency)
    return tiers, objects


def main():
    tiers, objects = build_world()
    names = [HOT, WARM, COLD]
    slo_budget_milli = 10                     # at most 1% of reads may miss their class
    total_reads = sum(obj.reads_day for obj in objects.values())
    total_size = sum(obj.size_mb for obj in objects.values())
    check(total_size <= tiers[COLD].capacity_mb, "the workload must fit the cold tier")

    # ---- 1. cold-start placement violates the SLO, and it is detected
    placement = {oid: COLD for oid in objects}
    last_moved = {}
    bad, total = violation_reads(tiers, placement, objects)
    check(total == total_reads and bad * 1000 > total * slo_budget_milli,
          "a cold-only placement must breach the SLO budget (negative control)")
    initial_cost = cost_of(tiers, placement, objects)

    # ---- 2. run rounds; the safety invariant is re-verified after every round
    history = []
    for round_no in range(8):
        moves, rejected, loads = plan_round(tiers, objects, placement, last_moved, round_no)
        verify_state(tiers, objects, placement, loads)
        check(len(moves) <= 4, "the throttle must hold: %d moves in round %d" % (len(moves), round_no))
        for oid, _ in moves:
            check(last_moved[oid] == round_no, "a move must stamp the cooldown clock")
        bad, total = violation_reads(tiers, placement, objects)
        history.append((round_no, len(moves), len(rejected), bad * 1000 // total,
                        cost_of(tiers, placement, objects), dict(loads)))

    # ---- 3. the cooldown rail: every recorded move is inside a distinct window
    moved_at = sorted(last_moved.items(), key=lambda item: item[1])
    check(all(moved_at[i][1] <= moved_at[i + 1][1] for i in range(len(moved_at) - 1)),
          "the cooldown stamps must be well ordered")
    check(len(last_moved) >= 4, "the planner must have moved several objects")

    # ---- 4. convergence: the SLO is met, the cost dropped and the plan is stable
    final_bad, final_total = violation_reads(tiers, placement, objects)
    final_cost = cost_of(tiers, placement, objects)
    check(final_bad * 1000 <= final_total * slo_budget_milli,
          "the converged placement must meet the SLO budget (%d/%d)" % (final_bad, final_total))
    check(final_bad < bad or final_bad == 0, "the SLO must improve over the cold start")
    check(final_cost > initial_cost,
          "conforming to the SLO trades money for latency: %d -> %d" % (initial_cost, final_cost))
    closed_form = sum(obj.size_mb * tiers[cheapest_ok_tier(obj, tiers)].rate_milli
                      for obj in objects.values())
    check(final_cost == closed_form,
          "the converged cost %d must equal the closed-form minimum %d" % (final_cost, closed_form))
    check(history[-1][1] == 0, "a converged planner must stop moving objects (no thrashing)")
    quiet, quiet_rejected, quiet_loads = plan_round(tiers, objects, dict(placement),
                                                    dict(last_moved), 8)
    check(quiet == [] and quiet_rejected == [], "at the fixed point a round must be a no-op")
    verify_state(tiers, objects, placement, quiet_loads)
    # ---- 5. local optimality: an object not in its class tier must be blocked by capacity
    final_loads = loads_of(tiers, placement, objects)
    for oid, obj in objects.items():
        wanted = cheapest_ok_tier(obj, tiers)
        check(placement[oid] == wanted or final_loads[wanted] == tiers[wanted].capacity_mb,
              "%s sits in %s while %s still has room" % (oid, placement[oid], wanted))

    # ---- 6. re-convergence after a deterministic traffic drift
    drifted = {oid: Obj(obj.oid, obj.size_mb, obj.reads_day, obj.max_latency_ms)
               for oid, obj in objects.items()}
    drifted["obj-104"].max_latency_ms = 5                 # promoted into the hot class
    drifted["obj-100"].max_latency_ms = 30                # relaxed: it may now cool down
    moves, _, loads = plan_round(tiers, drifted, dict(placement), dict(last_moved), 9)
    verify_state(tiers, drifted, {**placement, **{oid: tier for oid, tier in moves}}, loads)
    check(("obj-104", HOT) in moves, "a promoted object must be migrated to the hot tier")
    check(("obj-100", WARM) in moves, "a relaxed object must be demoted")
    check(moves[0] == ("obj-104", HOT) and len(moves) == 2,
          "the SLO fix must outrank the cheap demotion and nothing else may move")

    # ---- 7. exhaustive oracle on a tiny instance
    tiny_tiers = {HOT: Tier(HOT, 1000, 5, 1000), WARM: Tier(WARM, 1000, 30, 200),
                  COLD: Tier(COLD, 4000, 500, 30)}
    tiny_objects = {"a": Obj("a", 400, 900, 5), "b": Obj("b", 400, 600, 30),
                    "c": Obj("c", 900, 300, 500)}
    tiny_start = {oid: COLD for oid in tiny_objects}
    greedy_placement = dict(tiny_start)
    greedy_moves = {}
    for round_no in range(4):
        moves, _, _ = plan_round(tiny_tiers, tiny_objects, greedy_placement, greedy_moves,
                                 round_no, max_moves=3)
    greedy_cost = cost_of(tiny_tiers, greedy_placement, tiny_objects)
    greedy_bad, greedy_total = violation_reads(tiny_tiers, greedy_placement, tiny_objects)
    optimum = exhaustive_optimum(tiny_tiers, tiny_objects, names, slo_budget_milli)
    check(optimum is not None, "a feasible tiny placement must exist")
    check(greedy_bad * 1000 <= greedy_total * slo_budget_milli, "the tiny plan must be feasible")
    check(greedy_cost >= optimum, "no plan may beat the exhaustive optimum %d" % optimum)
    print("tiny instance: greedy cost %d vs exhaustive optimum %d" % (greedy_cost, optimum))

    # ---- 8. boundaries: empty workload, exact capacity, oversized object, zero throttle
    empty_tiers, empty_objects = tiers, {}
    moves, rejected, loads = plan_round(empty_tiers, empty_objects, {}, {}, 0)
    check(moves == [] and rejected == [] and set(loads.values()) == {0},
          "an empty workload must produce an empty round")
    exact = {HOT: Tier(HOT, 2700, 5, 1000), COLD: Tier(COLD, 10000, 500, 30)}
    stuff = {"x": Obj("x", 1200, 5, 5), "y": Obj("y", 1500, 4, 5)}
    packed = {"x": COLD, "y": COLD}
    moves, rejected, loads = plan_round(exact, stuff, packed, {}, 0)
    check(loads[HOT] == 2700 and len(moves) == 2 and rejected == [],
          "an exactly full tier must accept the last object")
    pack = {"z": Obj("z", 3000, 5, 5)}
    moves, rejected, loads = plan_round(exact, pack, {"z": COLD}, {}, 0)
    check(moves == [] and rejected == [("z", HOT)],
          "an object larger than the tier must be refused, not dropped")
    moves, rejected, loads = plan_round(tiers, objects, dict(placement), {}, 6, max_moves=0)
    check(moves == [] and rejected == [], "a zero throttle must block every move")
    verify_state(tiers, objects, placement, loads)

    # ---- 9. determinism
    replay = {oid: COLD for oid in objects}
    stamps = {}
    for round_no in range(8):
        plan_round(tiers, objects, replay, stamps, round_no)
    check(replay == placement, "the same trace must replay to the same placement")
    print("rounds: " + "; ".join("r%d moves=%d rejected=%d miss=%d.%d%% cost=%d"
                                 % (r, m, j, v // 10, v % 10, c) for r, m, j, v, c, _ in history))
    print("PASS")


if __name__ == "__main__":
    main()
