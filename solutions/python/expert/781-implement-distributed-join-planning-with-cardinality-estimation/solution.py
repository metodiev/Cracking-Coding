"""Task 781 - Implement distributed join planning with cardinality estimation errors simulation.  (Tier: Expert, Ch 9 - System Design and Scalability)

A join planner picks an order in which to join the relations of a query; the cost is the sum of the
intermediate result sizes, so the order that keeps intermediates small wins.  The simplified
optimizer here is exact dynamic programming over relation subsets (Selinger style): dp[S] is the
cheapest way to join the relations of S, extended one relation at a time, and the intermediate size
of a subset uses the textbook estimator

    |R1 x ... x Rn|  =  (prod |Ri|) * (prod selectivity of every join edge *inside* the subset),

which assumes independence and containment.  Two baselines are compared with it: a greedy planner
(intersect the cheapest relation that joins the current set) and the exact optimum found by
enumerating every permutation (feasible for the six-relation queries used here).

The point of the task is the *estimator error*: the planner is fed perturbed statistics (every base
cardinality and every selectivity multiplied by 2^(+-e)) while the plan's real cost is always
measured with the true statistics, so the ratio "true cost of the plan chosen from wrong numbers /
true cost of the optimal plan" shows how badly estimation errors hurt.  Verified: with exact
statistics the DP plan's cost equals the brute-force optimum for many random query graphs, plans are
valid (every subset grows along an available join edge, or the cross product is reported as such),
all 2^n subsets are enumerated, greedy never beats the optimum, the e = 0 ratio is exactly 1, the
error ratio grows with e, and the degenerate one- and two-relation graphs plus a disconnected
(cross-product) graph are handled.  Deterministic: the perturbations come from one splitmix64 stream.

Simplified: cost is a pure cardinality sum (no I/O, network or CPU terms), the estimator is the
independence/containment formula, perturbations are multiplicative and symmetric, and queries have
up to six relations so the permutation oracle stays cheap.
"""
import itertools

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


class QueryGraph:
    """Relations with base cardinalities plus join edges with selectivities."""

    def __init__(self, names, sizes, edges):
        self.names = list(names)
        self.count = len(names)
        self.sizes = list(sizes)
        self.edges = {}                      # (i, j) with i < j -> selectivity
        for left, right, selectivity in edges:
            self.edges[(min(left, right), max(left, right))] = selectivity

    def subset_size(self, subset, sizes=None, edges=None):
        """Estimated cardinality of joining the relations in `subset` (a frozenset/iterable)."""
        sizes = self.sizes if sizes is None else sizes
        edges = self.edges if edges is None else edges
        members = sorted(subset)
        product = 1.0
        for index in members:
            product *= sizes[index]
        for left, right in itertools.combinations(members, 2):
            selectivity = edges.get((left, right), 1.0)
            product *= selectivity
        return product

    def joins(self, subset, index):
        """True when `index` can be joined to the accumulated subset along a real predicate."""
        return any((min(index, other), max(index, other)) in self.edges for other in subset)

    def perturbed(self, exponent, seed):
        """A copy of the statistics with every size and selectivity scaled by 2^(+-exponent)."""
        state = seed
        sizes, edges = [], {}
        for size in self.sizes:
            state, r = splitmix64(state)
            factor = 2.0 ** (((r >> 13) % (2 * exponent + 1)) - exponent)
            sizes.append(max(size * factor, 1.0))
        for edge, selectivity in self.edges.items():
            state, r = splitmix64(state)
            factor = 2.0 ** (((r >> 13) % (2 * exponent + 1)) - exponent)
            edges[edge] = min(1.0, max(1e-6, selectivity * factor))
        return sizes, edges


def cost_of_order(graph, order, sizes=None, edges=None):
    """Sum of intermediate cardinalities along an order of relation indices."""
    cumulative = []
    total = 0.0
    for index in order:
        cumulative.append(index)
        total += graph.subset_size(cumulative, sizes, edges)
    return total


def order_valid(graph, order, allow_cross=True):
    """Every step must join along a real predicate unless a cross product is explicitly allowed."""
    if not order:
        return False
    accumulated = [order[0]]
    for index in order[1:]:
        if not graph.joins(accumulated, index):
            if not allow_cross:
                return False
        accumulated.append(index)
    return sorted(order) == list(range(graph.count))


def dp_plan(graph, sizes=None, edges=None):
    """Exact DP over subsets; returns (order, cost, subsets visited)."""
    n = graph.count
    if n == 0:
        return [], 0.0, 0
    best_cost = {frozenset(): 0.0}
    best_order = {frozenset(): ()}
    visited = 0
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            key = frozenset(subset)
            visited += 1
            best = (float("inf"), ())
            for index in subset:
                rest = key - {index}
                previous = best_cost[rest]
                candidate = previous + graph.subset_size(key, sizes, edges)
                if candidate < best[0]:
                    best = (candidate, best_order[rest] + (index,))
            best_cost[key] = best[0]
            best_order[key] = best[1]
    full = frozenset(range(n))
    return list(best_order[full]), best_cost[full], visited


def greedy_plan(graph, sizes=None, edges=None):
    """Greedy baseline: always extend with the relation that keeps the next intermediate smallest."""
    remaining = set(range(graph.count))
    accumulated = []
    total = 0.0
    while remaining:
        best = None
        for index in sorted(remaining):
            candidate = graph.subset_size(accumulated + [index], sizes, edges)
            if best is None or candidate < best[0]:
                best = (candidate, index)
        accumulated.append(best[1])
        total += best[0]
        remaining.discard(best[1])
    return accumulated, total


def brute_force_plan(graph, sizes=None, edges=None):
    """Independent oracle: the cheapest permutation of the relations."""
    best_cost, best_order = float("inf"), None
    for order in itertools.permutations(range(graph.count)):
        cost = cost_of_order(graph, list(order), sizes, edges)
        if cost < best_cost:
            best_cost, best_order = cost, list(order)
    return best_order, best_cost


def random_graph(n, seed, density=0.7, size_high=1000):
    """Deterministic random query graph: chain edges always present, others with the given density."""
    state = seed
    sizes = []
    for _ in range(n):
        state, r = splitmix64(state)
        sizes.append(float(1 + r % size_high))
    edges = []
    for index in range(n - 1):
        state, r = splitmix64(state)
        edges.append((index, index + 1, 0.02 + (r % 100) / 1000.0))
    for left, right in itertools.combinations(range(n), 2):
        if right == left + 1:
            continue
        state, r = splitmix64(state)
        if (r % 100) / 100.0 < density:
            edges.append((left, right, 0.01 + (r >> 7) % 90 / 1000.0))
    return QueryGraph(list("ABCDEF")[:n], sizes, edges)


def main():
    # 1. boundaries: a single relation and a pair
    single = QueryGraph(["A"], [500.0], [])
    order, cost, visited = dp_plan(single)
    check(order == [0] and cost == 500.0 and visited == 1, "one relation costs its own size")
    pair = QueryGraph(["A", "B"], [1000.0, 400.0], [(0, 1, 0.1)])
    order, cost, _ = dp_plan(pair)
    check(cost == min(1000.0, 400.0) + 1000.0 * 400.0 * 0.1,
          "the pair cost must be min(|A|, |B|) plus the join size")
    check(order[0] == 1, "the planner must start from the smaller relation")
    check(sorted(order) == [0, 1], "both relations must appear exactly once")
    check(order_valid(pair, order), "the plan must use the available join edge")

    # 2. with exact statistics the DP plan matches the permutation oracle, for random graphs
    comparisons = 0
    for trial in range(6):
        graph = random_graph(5, 0x5EED + trial)
        plan, plan_cost, visited = dp_plan(graph)
        oracle_order, oracle_cost = brute_force_plan(graph)
        check(abs(plan_cost - oracle_cost) < 1e-6,
              f"DP cost {plan_cost} != oracle {oracle_cost} on trial {trial}")
        check(visited == 2 ** graph.count - 1,
              f"the DP must visit every non-empty subset, visited {visited}")
        check(order_valid(graph, plan), "the DP plan must be executable")
        check(order_valid(graph, oracle_order), "the oracle plan must be executable")
        check(abs(cost_of_order(graph, plan) - plan_cost) < 1e-6,
              "the reported cost must equal the recomputed cost")
        comparisons += 1
    check(comparisons == 6, "all six graphs must be checked")

    # 3. greedy is a heuristic: never cheaper than the optimum, but usually worse
    worse = 0
    for trial in range(6):
        graph = random_graph(5, 0xBEEF + trial)
        _, optimum, _ = dp_plan(graph)
        greedy_order, greedy_cost = greedy_plan(graph)
        check(greedy_cost >= optimum - 1e-6, "greedy must not beat the optimum")
        check(order_valid(graph, greedy_order), "the greedy plan must also be executable")
        worse += greedy_cost > optimum + 1e-6
    check(worse >= 1, "greedy should be strictly worse on at least one of these graphs")

    # 4. subset sizes follow the independence/containment estimator
    graph = random_graph(4, 0x1234)
    check(abs(graph.subset_size([0]) - graph.sizes[0]) < 1e-9, "a singleton is just its base size")
    combined = graph.subset_size([0, 1])
    check(abs(combined - graph.sizes[0] * graph.sizes[1] * graph.edges[(0, 1)]) < 1e-9,
          "a pair size is the product times the edge selectivity")
    three = graph.subset_size([0, 1, 2])
    expected = graph.sizes[0] * graph.sizes[1] * graph.sizes[2]
    for left, right in itertools.combinations([0, 1, 2], 2):
        expected *= graph.edges.get((left, right), 1.0)
    check(abs(three - expected) < 1e-6, "a triple multiplies every inner edge")

    # 5. estimation errors: e = 0 is exact, and the true cost ratio grows with e
    ratios = {}
    worst = 0.0
    samples = 0
    for exponent in (0, 1, 2, 3):
        total_ratio = 0.0
        for trial in range(8):
            graph = random_graph(5, 0xABCD + trial * 7)
            true_order, true_cost = brute_force_plan(graph)
            sizes, edges = graph.perturbed(exponent, 0x99 + trial)
            estimated_order, _, _ = dp_plan(graph, sizes, edges)
            actual = cost_of_order(graph, estimated_order)
            ratio = actual / true_cost
            total_ratio += ratio
            worst = max(worst, ratio)
            samples += 1
        ratios[exponent] = total_ratio / 8
    check(abs(ratios[0] - 1.0) < 1e-9, f"with exact statistics the ratio must be 1, got {ratios[0]}")
    check(ratios[3] >= ratios[1], "bigger errors must not plan better")
    check(ratios[3] > 1.0, "some error must actually be paid for")
    check(worst <= 25.0, f"the worst observed ratio {worst:.1f} must stay bounded")
    check(samples == 32, "all error scenarios must be sampled")

    # 6. a disconnected query forces a cross product, and the plan says so
    lonely = QueryGraph(["A", "B", "C"], [100.0, 200.0, 300.0], [(0, 1, 0.5)])
    check(not order_valid(lonely, [0, 2, 1], allow_cross=False),
          "a cross-product step must be detectable")
    cross_order, cross_cost, _ = dp_plan(lonely)
    check(sorted(cross_order) == [0, 1, 2], "a cross product still plans all relations")
    cross_cost_reported = cost_of_order(lonely, cross_order)
    check(abs(cross_cost_reported - cross_cost) < 1e-6, "the cross-product cost must be consistent")
    check(cross_cost_reported >= sum(lonely.sizes), "a cross product cannot be cheaper than a base size")
    check(order_valid(lonely, cross_order), "the cross-product plan is still a valid order")

    # 7. more relations: the DP still matches the oracle and stays a polynomial sweep
    graph6 = random_graph(6, 0x777)
    plan6, cost6, visited6 = dp_plan(graph6)
    check(visited6 == 2 ** 6 - 1, "six relations mean 63 subsets")
    check(abs(cost6 - brute_force_plan(graph6)[1]) < 1e-6,
          "the DP must match the oracle on six relations")
    check(order_valid(graph6, plan6, allow_cross=False),
          "these chain-connected queries need no cross product")

    # 8. determinism
    again = dp_plan(random_graph(5, 0x5EED))
    check(again == dp_plan(random_graph(5, 0x5EED)), "planning must be deterministic")
    sizes, edges = random_graph(5, 0x5EED).perturbed(2, 0x42)
    first = dp_plan(random_graph(5, 0x5EED), sizes, edges)
    second = dp_plan(random_graph(5, 0x5EED), sizes, edges)
    check(first == second, "perturbed planning must be deterministic too")

    print(f"781 ratios e0={ratios[0]:.2f} e1={ratios[1]:.2f} e2={ratios[2]:.2f} e3={ratios[3]:.2f} "
          f"worst={worst:.2f}")
    print(f"781 subsets n=6: {visited6} greedy_worse={worse}/6 cross_cost={cross_cost:.1f} "
          f"dp_cost6={cost6:.1f}")
    print("PASS")


if __name__ == "__main__":
    main()
