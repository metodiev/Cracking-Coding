"""Task 894 - Implement automated root cause analysis heuristics using dependency graphs.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

A microservice dependency graph is a DAG: `calls[u]` lists what `u` depends on, and the
latency a caller observes for `u` is `u`'s own work plus everything it waits for.  A fault is
one service whose own work is multiplied by a factor.  Two views of the same fault are produced,
because they behave completely differently and the difference is the whole lesson:

* the **relative** increase `(observed - baseline) / baseline` is what a latency dashboard
  shows.  A service whose own work is small compared to the work it waits for barely moves
  proportionally while its callers light up, so this view is dominated by the loudest symptom;
* the **added** latency `observed - baseline` is additive along the graph: every caller inherits
  exactly the sum of its dependencies' increases, so the residual `added[u] - sum(added[deps])`
  is non-zero exactly at the faulty service and zero everywhere above it.  That residual is the
  novelty score and it isolates the cause instead of the symptom.  (Attributing everything above
  the strongest dependency instead of the whole sum leaves phantom residual wherever two branches
  of the anomaly meet again, which the demo also checks.)

Four heuristics are implemented and scored against the injected fault: propagation (a
personalised PageRank over the caller direction seeded by the relative view - the standard
"diffuse the symptom" baseline, which lands on a caller of the fault rather than on the fault),
novelty (the residual above), onset (the fault becomes visible at its own service first and only
later upstream, so the earliest onset among the suspects is a candidate), and the combined
ranking (novelty first, onset as the tie-breaker) which is the reported answer.

The PageRank vector is validated as a genuine fixed point with unit probability mass, the
additivity of the added-latency view is checked edge by edge, onset ordering is validated against
the propagation delay in the dependency graph, and every heuristic is scored on hundreds of
randomised scenarios - clean, noisy and two-fault - with the exact ground truth being whichever
service's own work was multiplied.

Simplified versus a production RCA system: no real telemetry (observations come from the same
latency model the fault is injected into), one dominant fault per scenario plus a two-fault case,
no correlated or cardinality faults, hand-set heuristic weights rather than learned ones, and
graphs of up to a few hundred services.  Verified: the graph is acyclic and the latency model is
monotone along it, the relative view dilutes a deep fault while the additive view localises it
with zero residual above the fault, PageRank is a unit-mass fixed point whose top pick is a
caller of the fault rather than the fault, onset follows the hop delay, the combined and novelty
heuristics recover the fault in every clean scenario and almost every noisy one, healthy
scenarios yield no suspect, a fault at a leaf or at the entry point is found, two simultaneous
faults both land in the top three, accuracy holds on a 500 service graph, and every run
reproduces from its seed.
"""
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound


class GraphError(Exception):
    """Raised for a malformed dependency graph."""


class ServiceGraph:
    """A call DAG built in topological order: node i only depends on lower nodes."""

    def __init__(self, node_count, seed, max_fanout=3):
        if node_count < 3:
            raise GraphError("a dependency graph needs at least three services")
        if max_fanout < 1:
            raise GraphError("a fan-out of at least one is required")
        rng = Rng(seed)
        self.node_count = node_count
        self.calls = {}
        for node in range(node_count):
            if node < 2:
                self.calls[node] = []
                continue
            count = 1 + rng.below(max_fanout)
            chosen = sorted({rng.below(node) for _ in range(2 * count)})
            self.calls[node] = chosen[:max_fanout] if chosen else [node - 1]
        self.callers = {node: [] for node in range(node_count)}
        for node, dependencies in self.calls.items():
            for dependency in dependencies:
                self.callers[dependency].append(node)
        self.base_latency = [1 + rng.below(5) for _ in range(node_count)]
        self.front_end = node_count - 1

    def effective_latency(self, base):
        """A service waits for its own work plus everything it calls."""
        effective = [0] * self.node_count
        for node in range(self.node_count):
            total = base[node]
            for dependency in self.calls[node]:
                total += effective[dependency]
            effective[node] = total
        return effective

    def depths_from(self, source):
        """Hop distance in the caller direction - the direction a fault propagates."""
        distance = {source: 0}
        frontier = [source]
        while frontier:
            following = []
            for node in frontier:
                for caller in self.callers[node]:
                    if caller not in distance:
                        distance[caller] = distance[node] + 1
                        following.append(caller)
            frontier = following
        return distance

    def downstream(self, node):
        seen = set()
        frontier = list(self.calls[node])
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.calls[current])
        return seen


def inject(graph, faults, factor):
    """Multiply the own work of one or more services and produce both anomaly views."""
    if isinstance(faults, int):
        faults = [faults]
    baseline = graph.effective_latency(graph.base_latency)
    faulty = list(graph.base_latency)
    for fault in faults:
        faulty[fault] = faulty[fault] * factor
    observed = graph.effective_latency(faulty)
    return {
        "baseline": baseline,
        "observed": observed,
        "relative": [(observed[node] - baseline[node]) / baseline[node]
                     for node in range(graph.node_count)],
        "added": [observed[node] - baseline[node] for node in range(graph.node_count)],
    }


def make_onsets(graph, faults, rng, added, delay=5, jitter=2, threshold=1):
    """The first time each affected service shows the anomaly; the fault shows it first."""
    if isinstance(faults, int):
        faults = [faults]
    onsets = {}
    for fault in faults:
        for node, hops in graph.depths_from(fault).items():
            if added[node] < threshold:
                continue
            time = 100 + hops * delay + rng.below(jitter + 1)
            onsets[node] = min(onsets.get(node, time), time)
    return onsets


# ------------------------------------------------------------------- heuristics
def _seed_vector(graph, scores):
    total = sum(scores)
    if total <= 0:
        return [0.0] * graph.node_count
    return [value / total for value in scores]


def _page_rank_step(graph, mass, seed, alpha):
    """One iteration; services without callers release their mass back to the seed vector."""
    incoming = [0.0] * graph.node_count
    dangling = 0.0
    for node in range(graph.node_count):
        callers = graph.callers[node]
        if callers:
            share = mass[node] / len(callers)
            for caller in callers:
                incoming[caller] += share
        else:
            dangling += mass[node]
    return [(1.0 - alpha) * (incoming[node] + dangling * seed[node]) + alpha * seed[node]
            for node in range(graph.node_count)]


def propagation_scores(graph, scores, alpha=0.15, iterations=200, tolerance=1e-12):
    """Personalised PageRank over the caller direction, seeded by the observed anomalies."""
    mass = [0.0] * graph.node_count
    if sum(scores) <= 0:
        return mass
    seed = _seed_vector(graph, scores)
    for _ in range(iterations):
        updated = _page_rank_step(graph, mass, seed, alpha)
        shift = sum(abs(updated[node] - mass[node]) for node in range(graph.node_count))
        mass = updated
        if shift < tolerance:
            break
    return mass


def fixed_point_residual(graph, mass, scores, alpha=0.15):
    updated = _page_rank_step(graph, mass, _seed_vector(graph, scores), alpha)
    return sum(abs(mass[node] - updated[node]) for node in range(graph.node_count))


def novelty_scores(graph, added):
    """Added latency a service cannot explain by the services it calls.

    The model is additive along the call chain, so `added[u] - sum(added[deps])` is exactly the
    service's own added work: zero for every healthy service, and the fault's own inflation at
    the culprit.  Using the maximum instead of the sum would leave phantom residual wherever two
    branches of the fault's anomaly meet again, which the demo checks against.
    """
    return {node: added[node] - sum(added[d] for d in graph.calls[node])
            for node in range(graph.node_count)}


def branch_max_residual(graph, added):
    """The naive variant: attribute everything above the strongest dependency to this service."""
    return {node: added[node] - max((added[d] for d in graph.calls[node]), default=0)
            for node in range(graph.node_count)}


def onset_scores(onsets):
    return {node: -time for node, time in onsets.items()}


def rank(heuristic):
    """Descending score, then ascending node id, so the ranking is deterministic."""
    lookup = heuristic if isinstance(heuristic, dict) else dict(enumerate(heuristic))
    return sorted(lookup, key=lambda node: (-lookup[node], node))


def suspects_from(relative, level=0.25, floor=1e-9):
    peak = max(relative, default=0.0)
    if peak <= floor:
        return []
    return [node for node in range(len(relative)) if relative[node] >= level * peak]


def diagnose(graph, view, onsets):
    """The reported heuristic: suspects from the dashboard view, ordered by novelty."""
    suspects = suspects_from(view["relative"])
    if not suspects:
        return []
    if len(suspects) == 1:
        return suspects
    novelty = novelty_scores(graph, view["added"])
    return sorted(suspects, key=lambda node: (-novelty[node], onsets.get(node, 10 ** 9), node))


# ----------------------------------------------------------------------- main
def build_scenario(seed, node_count=40, factor=4, jitter=2, noise=0):
    graph = ServiceGraph(node_count, seed)
    rng = Rng(seed + 7)
    fault = 2 + rng.below(node_count - 2)
    view = inject(graph, fault, factor)
    if noise:
        view = dict(view)
        view["relative"] = [value + rng.below(noise + 1) / 100.0 for value in view["relative"]]
        view["added"] = [value + rng.below(noise + 1) for value in view["added"]]
    onsets = make_onsets(graph, [fault], rng, view["added"], jitter=jitter)
    return graph, fault, view, onsets


def main():
    # --- 1. the graph is a valid DAG ----------------------------------------
    graph = ServiceGraph(24, 894001)
    check(all(dependency < node for node in range(graph.node_count)
              for dependency in graph.calls[node]),
          "every dependency points to a lower node, so the graph is acyclic")
    for node in range(graph.node_count):
        for dependency in graph.calls[node]:
            check(node in graph.callers[dependency], "the reverse index must be consistent")
            check(node not in graph.downstream(node), "no service may call itself")
    check(graph.calls[0] == [] and graph.calls[1] == [],
          "the two lowest nodes must be leaves with no dependencies")
    for bad in (lambda: ServiceGraph(2, 894002), lambda: ServiceGraph(10, 894003, max_fanout=0)):
        raised = False
        try:
            bad()
        except GraphError:
            raised = True
        check(raised, "an invalid graph request must be rejected")

    # --- 2. latency is monotone along the graph -----------------------------
    effective = graph.effective_latency(graph.base_latency)
    for node in range(graph.node_count):
        expected = graph.base_latency[node] + sum(effective[d] for d in graph.calls[node])
        check(effective[node] == expected,
              "the topological pass must match the definition at node %d" % node)
        for dependency in graph.calls[node]:
            check(effective[node] > effective[dependency],
                  "a caller must always wait longer than what it calls")
    heaviest = max(range(graph.node_count), key=lambda node: effective[node])
    check(graph.callers[heaviest] == [], "the most expensive service must be an entry point")

    # --- 3. the two views of one fault --------------------------------------
    fault = 5
    view = inject(graph, fault, 4)
    check(view["relative"][fault] > 0 and view["added"][fault] > 0,
          "the faulty service must show up in both views")
    for node in range(graph.node_count):
        affected = node == fault or fault in graph.downstream(node)
        check((view["added"][node] > 0) == affected,
              "node %d must accumulate added latency exactly when the fault is downstream" % node)
        check((view["relative"][node] > 0) == affected,
              "node %d must show a relative increase exactly when the fault is downstream" % node)
        expected = sum(view["added"][dependency] for dependency in graph.calls[node])
        if node == fault:
            check(view["added"][node] == expected + 3 * graph.base_latency[fault],
                  "the fault must add exactly (factor - 1) times its own work")
        else:
            check(view["added"][node] == expected,
                  "the added latency must be additive above the fault at node %d" % node)
    check(all(view["added"][caller] >= view["added"][dependency]
              for caller in range(graph.node_count) for dependency in graph.calls[caller]),
          "added latency must never shrink as it propagates towards the callers")
    worst = max(range(graph.node_count), key=lambda node: view["added"][node])
    check(worst == fault or fault in graph.downstream(worst),
          "the worst affected service must be the fault or sit above it")

    # --- 4. the novelty residual isolates the fault -------------------------
    novelty = novelty_scores(graph, view["added"])
    check(novelty[fault] == max(novelty.values()),
          "the fault must have the highest novelty score")
    check(novelty[fault] == 3 * graph.base_latency[fault],
          "the novelty of the fault must be exactly its own added work")
    check(all(novelty[node] == 0 for node in range(graph.node_count) if node != fault),
          "every healthy service must have exactly zero novelty")
    branch = branch_max_residual(graph, view["added"])
    phantom = [node for node in range(graph.node_count) if branch[node] > 0 and node != fault]
    check(phantom != [],
          "the branch variant must show phantom residual where the anomaly reconverges")
    check(all(branch[node] >= novelty[node] for node in range(graph.node_count)),
          "the branch variant can only over-attribute, never under-attribute")
    for node in range(graph.node_count):
        check(novelty[node] >= 0, "novelty can never be negative")

    # --- 5. the relative view dilutes a deep fault --------------------------
    deep = ServiceGraph(30, 894010)
    deep_effective = deep.effective_latency(deep.base_latency)
    with_dependencies = [node for node in range(deep.node_count) if deep.calls[node]]
    deep_fault = min(with_dependencies,
                     key=lambda node: deep.base_latency[node] / deep_effective[node])
    deep_view = inject(deep, deep_fault, 4)
    expected_ratio = 3 * deep.base_latency[deep_fault] / deep_effective[deep_fault]
    check(abs(deep_view["relative"][deep_fault] - expected_ratio) < 1e-9,
          "the relative increase must be the fault's own added work divided by its total latency")
    check(deep_view["relative"][deep_fault] == max(deep_view["relative"]),
          "the diluted service still shows the largest relative increase")
    check(deep_view["relative"][deep_fault] < 0.05,
          "a small service under heavy dependencies must stay under a 5 percent dashboard"
          " threshold, got %.4f" % deep_view["relative"][deep_fault])
    check(deep_fault not in [node for node in range(deep.node_count)
                             if deep_view["relative"][node] >= 0.05],
          "an absolute dashboard threshold would miss the fault entirely")
    deep_novelty = novelty_scores(deep, deep_view["added"])
    check(deep_novelty[deep_fault] == max(deep_novelty.values()),
          "the additive view must still locate the deep fault")
    check(deep_novelty[deep_fault] == 3 * deep.base_latency[deep_fault],
          "the additive residual must still be the fault's own added work")

    # --- 6. the propagation baseline is a fixed point that blames a caller --
    mass = propagation_scores(graph, view["relative"])
    check(abs(sum(mass) - 1.0) < 1e-9, "the PageRank mass must sum to one")
    check(fixed_point_residual(graph, mass, view["relative"]) < 1e-9,
          "the PageRank must be a fixed point")
    check(all(value >= 0 for value in mass), "no service may get negative mass")
    propagated_top = rank(mass)[0]
    check(propagated_top != fault,
          "pure propagation must not blame the fault, only the symptom it caused")
    check(fault in graph.downstream(propagated_top),
          "the blamed service must be one that depends on the fault")
    check(mass[propagated_top] > mass[fault],
          "the blamed caller must outrank the faulty service itself")
    check(all(value == 0 for value in propagation_scores(graph, [0.0] * graph.node_count)),
          "a healthy graph must produce no propagation mass")

    # --- 7. onset ordering follows the hop delay ----------------------------
    rng = Rng(894100)
    second_view = inject(graph, 9, 4)
    onsets = make_onsets(graph, [9], rng, second_view["added"], delay=5, jitter=0)
    distances = graph.depths_from(9)
    check(onsets[9] == 100, "the fault must be visible at its own service first")
    for node, time in onsets.items():
        check(time == 100 + distances[node] * 5,
              "the onset at node %d must follow the hop delay" % node)
    check(all(distances[inner] <= distances[outer]
              for inner in distances for outer in graph.callers[inner]),
          "hop distance in the caller direction must be non-decreasing")
    second_novelty = novelty_scores(graph, second_view["added"])
    check(second_novelty[9] == max(second_novelty.values()),
          "the novelty residual must agree on a second fault")

    # --- 8. the heuristics compared on clean scenarios ----------------------
    combined_hits = 0
    novelty_hits = 0
    onset_hits = 0
    propagation_hits = 0
    propagation_blames_upstream = 0
    suspects_seen = 0
    scenarios = 0
    for seed in range(894200, 894300):
        local, local_fault, local_view, local_onsets = build_scenario(seed)
        answer = diagnose(local, local_view, local_onsets)
        check(answer, "a faulted scenario must always produce a suspect")
        if answer[0] == local_fault:
            combined_hits += 1
        if rank(novelty_scores(local, local_view["added"]))[0] == local_fault:
            novelty_hits += 1
        if rank(onset_scores(local_onsets))[0] == local_fault:
            onset_hits += 1
        leading = rank(propagation_scores(local, local_view["relative"]))[0]
        if leading == local_fault:
            propagation_hits += 1
        elif local_fault in local.downstream(leading):
            propagation_blames_upstream += 1
        suspects_seen += len(answer)
        scenarios += 1
    check(scenarios > 90, "the sweep must be substantial, got %d" % scenarios)
    check(combined_hits == scenarios,
          "the reported heuristic must find every fault, got %d of %d" % (combined_hits, scenarios))
    check(novelty_hits == scenarios, "the novelty residual must find every fault")
    check(suspects_seen <= scenarios * (40 // 4),
          "the suspect set must stay much smaller than the graph, averaged %d"
          % (suspects_seen // scenarios))
    check(propagation_hits + propagation_blames_upstream >= scenarios - 2,
          "propagation must always blame the fault or a caller of it, never something else")
    check(propagation_blames_upstream > scenarios // 5,
          "propagation must blame a caller often enough to show the failure mode: %d of %d"
          % (propagation_blames_upstream, scenarios))
    check(onset_hits >= scenarios // 3,
          "onset alone should still be useful, got %d of %d" % (onset_hits, scenarios))

    # --- 9. noisy observations ---------------------------------------------
    noisy_hits = 0
    noisy_top3 = 0
    noisy_scenarios = 60
    for seed in range(894400, 894400 + noisy_scenarios):
        local, local_fault, local_view, local_onsets = build_scenario(seed, node_count=60, noise=1)
        answer = diagnose(local, local_view, local_onsets)
        if answer and answer[0] == local_fault:
            noisy_hits += 1
        if local_fault in answer[:3]:
            noisy_top3 += 1
    check(noisy_hits >= noisy_scenarios - 6,
          "the heuristic must survive observation noise: %d of %d" % (noisy_hits, noisy_scenarios))
    check(noisy_top3 >= noisy_scenarios - 2,
          "the fault must stay in the top three under noise: %d of %d"
          % (noisy_top3, noisy_scenarios))
    check(noisy_top3 >= noisy_hits, "the top three can never be worse than the top one")

    # --- 10. degenerate scenarios ------------------------------------------
    healthy = ServiceGraph(30, 894500)
    flat_latency = healthy.effective_latency(healthy.base_latency)
    flat = {"baseline": flat_latency, "observed": flat_latency,
            "relative": [0.0] * healthy.node_count, "added": [0] * healthy.node_count}
    check(diagnose(healthy, flat, {}) == [], "a healthy graph must yield no suspect")
    check(suspects_from(flat["relative"]) == [], "no node may cross the anomaly threshold")
    check(all(value == 0 for value in novelty_scores(healthy, flat["added"]).values()),
          "a healthy graph has no novelty anywhere")

    leaf_graph = ServiceGraph(30, 894501)
    leaf_view = inject(leaf_graph, 0, 6)
    leaf_onsets = make_onsets(leaf_graph, [0], Rng(894502), leaf_view["added"])
    check(diagnose(leaf_graph, leaf_view, leaf_onsets)[0] == 0, "a fault at a leaf must be found")

    front_graph = ServiceGraph(30, 894503)
    front = front_graph.front_end
    front_view = inject(front_graph, front, 6)
    check(front_view["added"][front] > 0 and front_view["added"][front - 1] == 0,
          "a fault at the entry point must not affect anything else")
    front_onsets = make_onsets(front_graph, [front], Rng(894504), front_view["added"])
    check(diagnose(front_graph, front_view, front_onsets) == [front],
          "a fault at the entry point must be the only suspect")

    pair_graph = ServiceGraph(45, 894505)
    pair_view = inject(pair_graph, [7, 19], 4)
    pair_onsets = make_onsets(pair_graph, [7, 19], Rng(894506), pair_view["added"])
    pair_answer = diagnose(pair_graph, pair_view, pair_onsets)
    check(7 in pair_answer[:3] and 19 in pair_answer[:3],
          "both faults must be in the top three: %r" % (pair_answer[:5],))

    # --- 11. scale ----------------------------------------------------------
    scale_hits = 0
    scale_top3 = 0
    scale_scenarios = 150
    scale_nodes = 500
    for seed in range(894700, 894700 + scale_scenarios):
        local, local_fault, local_view, local_onsets = build_scenario(seed, node_count=scale_nodes)
        answer = diagnose(local, local_view, local_onsets)
        if answer and answer[0] == local_fault:
            scale_hits += 1
        if local_fault in answer[:3]:
            scale_top3 += 1
    check(scale_hits == scale_scenarios,
          "accuracy must hold at scale: %d of %d" % (scale_hits, scale_scenarios))
    check(scale_top3 == scale_scenarios,
          "the top three must always contain the fault at scale: %d of %d"
          % (scale_top3, scale_scenarios))
    scale_graph = ServiceGraph(scale_nodes, 894600)
    edges = sum(len(dependencies) for dependencies in scale_graph.calls.values())
    check(edges > scale_nodes, "the scaled graph must be well connected, got %d edges" % edges)
    check(scale_graph.callers[scale_graph.front_end] == [],
          "nothing may call the entry point of the scaled graph")
    biggest = max(range(scale_nodes), key=lambda node: scale_graph.base_latency[node])
    scaled = inject(scale_graph, biggest, 5)
    scaled_novelty = novelty_scores(scale_graph, scaled["added"])
    check(scaled_novelty[biggest] == max(scaled_novelty.values()),
          "the novelty residual must isolate the fault on the large graph too")

    # --- 12. determinism ----------------------------------------------------
    def replay(seed):
        local, local_fault, local_view, local_onsets = build_scenario(seed, node_count=80)
        answer = diagnose(local, local_view, local_onsets)
        return (local_fault, answer[:6],
                round(propagation_scores(local, local_view["relative"])[0], 12))

    check(replay(894800) == replay(894800), "an identical seed must replay the diagnosis")

    print("graph: %d services, %d call edges"
          % (graph.node_count, sum(len(value) for value in graph.calls.values())))
    print("scenarios: %d clean, top-1 combined %d, novelty %d, onset %d, propagation %d"
          % (scenarios, combined_hits, novelty_hits, onset_hits, propagation_hits))
    print("propagation blamed a caller of the fault in %d of %d scenarios"
          % (propagation_blames_upstream, scenarios))
    print("noise: %d of %d top-1 and %d of %d top-3 on 60 services"
          % (noisy_hits, noisy_scenarios, noisy_top3, noisy_scenarios))
    print("scale: %d services, %d scenarios, top-1 %d, top-3 %d"
          % (scale_nodes, scale_scenarios, scale_hits, scale_top3))
    print("PASS")


if __name__ == "__main__":
    main()
