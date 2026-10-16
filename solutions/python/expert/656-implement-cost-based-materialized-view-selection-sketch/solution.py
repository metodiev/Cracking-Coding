"""Task 656 - Implement cost-based materialized view selection sketch.  (Tier: Expert, Ch 14 - Databases)

The classic data-cube formulation (Harinarayan/Rajaraman/Ullman): the candidate views form the lattice
of dimension subsets of a fact table. A view `v` groups by the dimensions in `v` and holds
size(v) = product of the cardinalities of those dimensions rows; the parent of `v` is `v` plus one
dimension, so sizes grow monotonically upwards and the raw fact table is the top node. Answering a
query `w` from a set of materialized views `S` costs the size of the *smallest ancestor of w* in
S union {top}: an ancestor group-by can always be rolled up to a finer grouping, never the other way
round - that is the invariant that makes the model sound, and it is checked on every edge of the
lattice.

Choosing the best k views is NP-hard, so the shipped algorithm is the textbook greedy: repeatedly add
the view with the largest marginal benefit. Because benefit(S) = sum over queries of weight * max over
views of (size(top) - size(ancestor)) is a coverage function, it is monotone and submodular, giving the
guarantee optimum >= greedy >= (1 - 1/e) * optimum; both directions are cross-checked against an
exhaustive search over all subsets, dimension by dimension. A budget-aware variant additionally refuses
a view whose storage does not fit and prefers the best benefit per stored row.

Simplified: sizes are given analytically from dimension cardinalities instead of being estimated from
statistics; every lattice node is also a query with a weight; update/maintenance cost of a
materialized view is ignored (the model is query-cost only). Verified: lattice invariants, cost-model
invariants (baseline, monotonicity, self-materialization), submodularity, greedy versus exhaustive
optimum, the (1-1/e) bound, budget feasibility and determinism.
"""
from itertools import combinations

EULER_INV = 1 - 1 / 2.718281828459045
EPS = 1e-9


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class CubeLattice:
    """All group-by views of a fact table: nodes are frozensets of dimension indices."""

    def __init__(self, cardinalities, weights=None):
        self.cardinalities = list(cardinalities)
        self.dimension_count = len(self.cardinalities)
        self.nodes = [frozenset(subset)
                      for size in range(self.dimension_count + 1)
                      for subset in combinations(range(self.dimension_count), size)]
        self.top = frozenset(range(self.dimension_count))
        self.sizes = {node: self._size(node) for node in self.nodes}
        self.parents = {node: tuple(node | {dimension} for dimension in range(self.dimension_count)
                                    if dimension not in node)
                        for node in self.nodes}
        self.children = {node: tuple(other for other in self.nodes if node in self.parents[other])
                         for node in self.nodes}
        self.weights = {node: float((weights or {}).get(node, 1.0)) for node in self.nodes}

    def _size(self, node):
        size = 1
        for dimension in sorted(node):
            size *= self.cardinalities[dimension]
        return size

    def ancestors(self, node):
        """All supersets of `node`, closest first (the node itself, then one dimension up, ...)."""
        return [other for other in self.nodes if node <= other]

    def candidates(self):
        """Every view except the raw fact table, which is considered already materialized."""
        return [node for node in self.nodes if node != self.top]


class ViewSelection:
    """Cost model plus greedy/exhaustive selection over one lattice."""

    def __init__(self, lattice, precomputed=()):
        self.lattice = lattice
        self.precomputed = [frozenset(node) for node in precomputed] + [lattice.top]

    def cost(self, query, chosen):
        best = None
        for view in list(chosen) + self.precomputed:
            if query <= view:
                size = self.lattice.sizes[view]
                best = size if best is None else min(best, size)
        check(best is not None, "the top view always answers every query")
        return best

    def total_cost(self, chosen):
        return sum(self.lattice.weights[query] * self.cost(query, chosen) for query in self.lattice.nodes)

    def baseline(self):
        """Cost with nothing but the raw fact table materialized."""
        return self.total_cost([])

    def benefit(self, chosen):
        return self.baseline() - self.total_cost(chosen)

    def marginal(self, chosen, candidate):
        return self.benefit(list(chosen) + [candidate]) - self.benefit(chosen)

    def storage(self, chosen):
        return sum(self.lattice.sizes[view] for view in chosen)

    def greedy(self, max_views, pool=None):
        pool = self.lattice.candidates() if pool is None else list(pool)
        chosen = []
        while len(chosen) < max_views:
            best, best_gain = None, 0.0
            for candidate in pool:
                if candidate in chosen:
                    continue
                gain = self.marginal(chosen, candidate)
                if gain > best_gain + EPS:
                    best, best_gain = candidate, gain
            if best is None:
                break
            chosen.append(best)
        return chosen

    def greedy_with_budget(self, budget, pool=None):
        """Same greedy idea, but ranked by benefit per stored row and bounded by `budget` bytes."""
        pool = self.lattice.candidates() if pool is None else list(pool)
        chosen = []
        used = 0
        while True:
            best, best_ratio = None, 0.0
            for candidate in pool:
                if candidate in chosen:
                    continue
                size = self.lattice.sizes[candidate]
                if used + size > budget:
                    continue
                gain = self.marginal(chosen, candidate)
                if gain <= EPS:
                    continue
                ratio = gain / size
                if ratio > best_ratio + EPS:
                    best, best_ratio = candidate, ratio
            if best is None:
                break
            chosen.append(best)
            used += self.lattice.sizes[best]
        check(used <= budget, "the budget must be respected")
        return chosen

    def exhaustive_best(self, max_views, pool=None):
        pool = self.lattice.candidates() if pool is None else list(pool)
        best = 0.0
        for size in range(1, max_views + 1):
            for combo in combinations(pool, size):
                best = max(best, self.benefit(list(combo)))
        return best


def random_lattice(seed, dimensions=3, max_cardinality=4):
    cardinalities = []
    for _ in range(dimensions):
        seed, r = splitmix64(seed)
        cardinalities.append(2 + r % max_cardinality)
    return CubeLattice(cardinalities), seed


def main():
    # 1. lattice invariants
    lattice = CubeLattice([2, 3, 4])
    check(len(lattice.nodes) == 2 ** 3, "a cube over d dimensions has exactly 2^d views")
    check(lattice.sizes[lattice.top] == 24, "the raw cube holds the product of all cardinalities")
    check(lattice.sizes[frozenset()] == 1, "the grand total holds a single row")
    check(len(set(lattice.nodes)) == len(lattice.nodes), "lattice nodes must be unique")
    for node in lattice.nodes:
        for parent in lattice.parents[node]:
            check(node < parent, "a parent must be a strict superset of its child")
            check(lattice.sizes[parent] >= lattice.sizes[node], "parent views are never smaller")
            check(node in lattice.children[parent], "the child relation must be symmetric")
        check(len(lattice.parents[node]) == 3 - len(node), "one parent per missing dimension")
    check(set(lattice.children[lattice.top]) == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})},
          "the raw cube has one child per dropped dimension")
    check(lattice.children[frozenset()] == (),
          "the grand total has no view below it")

    # 2. cost model invariants
    selection = ViewSelection(lattice)
    for query in lattice.nodes:
        check(selection.cost(query, []) == lattice.sizes[lattice.top],
              "without materialized views every query scans the raw cube")
        check(selection.benefit([]) == 0, "no views means no benefit")
        check(selection.cost(query, []) >= lattice.sizes[query],
              "a larger group-by can never be cheaper than the query itself")
    for query in lattice.nodes:
        own = [query] if query != lattice.top else []
        if own:
            check(selection.cost(query, own) == lattice.sizes[query],
                  "a materialized view answers its own query at its own size")
    growing = []
    previous = selection.total_cost(growing)
    for view in sorted(lattice.candidates(), key=lambda node: -lattice.sizes[node]):
        growing.append(view)
        current = selection.total_cost(growing)
        check(current <= previous + EPS, "materializing more views must never cost more")
        previous = current
    check(selection.total_cost(lattice.candidates()) == sum(lattice.sizes[node] for node in lattice.nodes),
          "with every view materialized each query is answered by itself")

    # 3. monotonicity and submodularity of the benefit function
    seed = 0x5EED
    work, seed = random_lattice(seed)
    engine = ViewSelection(work)
    pool = work.candidates()
    check(engine.benefit(pool) >= engine.benefit(pool[:1]), "benefit must be monotone")
    for _ in range(24):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        seed, r3 = splitmix64(seed)
        small = [pool[r1 % len(pool)]]
        big = list(dict.fromkeys(small + [pool[r2 % len(pool)]]))
        candidate = pool[r3 % len(pool)]
        if candidate in big:
            continue
        check(engine.marginal(small, candidate) >= engine.marginal(big, candidate) - EPS,
              "diminishing returns: adding to a superset must not gain more")

    # 4. greedy versus the exhaustive optimum, and the (1 - 1/e) guarantee
    for trial in range(8):
        for dimensions in (2, 3):
            work, seed = random_lattice(seed, dimensions=dimensions)
            engine = ViewSelection(work)
            pool = work.candidates()
            for cap in (1, 2, 3):
                if cap > len(pool):
                    continue
                picked = engine.greedy(cap)
                value = engine.benefit(picked)
                optimum = engine.exhaustive_best(cap)
                check(len(picked) <= cap, "the greedy must respect the view cap")
                check(value <= optimum + EPS, "greedy cannot beat the exhaustive optimum")
                check(value + EPS >= EULER_INV * optimum,
                      f"greedy must reach the (1-1/e) factor (trial {trial}, cap {cap})")

    # 5. the uniform cube: greedy is known to be optimal, so it must match the exhaustive search
    uniform = CubeLattice([2, 3, 4])
    uniform_engine = ViewSelection(uniform)
    for cap in (1, 2, 3):
        greedy_value = uniform_engine.benefit(uniform_engine.greedy(cap))
        optimum = uniform_engine.exhaustive_best(cap)
        check(abs(greedy_value - optimum) < EPS,
              f"on the uniform cube greedy must be optimal for cap {cap}")
    print(f"656 uniform cube greedy cap1={uniform_engine.greedy(1)} "
          f"benefit={uniform_engine.benefit(uniform_engine.greedy(1)):.0f}")

    # 6. the first greedy pick is the view with the best ratio of covered queries to size
    best_single = max(uniform.candidates(), key=lambda node: uniform_engine.benefit([node]))
    check(uniform_engine.greedy(1) == [best_single],
          "the first greedy pick must be the best single view")

    # 7. weights steer the choice towards the expensive queries
    heavy = CubeLattice([2, 2, 2], weights={frozenset({0, 1}): 100.0})
    heavy_engine = ViewSelection(heavy)
    first = heavy_engine.greedy(1)[0]
    check(first <= frozenset({0, 1}), "the heavily weighted query must be covered by the first pick")
    check(heavy_engine.benefit([frozenset({0, 1})]) >= heavy_engine.benefit([frozenset({2})]),
          "covering the heavy query must pay more than covering a light one")

    # 8. budget-aware selection
    budget_lattice = CubeLattice([4, 4, 4])
    budget_engine = ViewSelection(budget_lattice)
    tight = budget_engine.greedy_with_budget(20)
    check(budget_engine.storage(tight) <= 20, "the storage budget must hold")
    check(all(budget_lattice.sizes[node] <= 20 for node in tight),
          "no single view may exceed the budget")
    check(tight == budget_engine.greedy_with_budget(20), "budget selection must be deterministic")
    single_best = max(budget_lattice.candidates(), key=lambda node: budget_engine.benefit([node]))
    check(budget_engine.benefit(tight) >= budget_engine.benefit([single_best]) - EPS,
          "a budgeted selection must do at least as well as the best single view")
    check(budget_engine.greedy_with_budget(0) == [], "a zero budget materializes nothing")

    # 9. degenerate latices
    flat = CubeLattice([1, 1])
    flat_engine = ViewSelection(flat)
    check(all(flat.sizes[node] == 1 for node in flat.nodes),
          "one-valued dimensions collapse every view to a single row")
    check(flat_engine.benefit(flat.candidates()) == 0,
          "when every view costs one row there is nothing to gain")
    check(flat_engine.greedy(2) == [], "greedy must stop when no view gains anything")
    one = CubeLattice([5])
    one_engine = ViewSelection(one)
    check(one_engine.greedy(1) == [frozenset()], "the only useful view of a one-dimension cube is the total")
    check(one_engine.cost(frozenset(), [frozenset()]) == 1,
          "the materialized grand total answers its own query with a single row")
    check(one_engine.benefit([frozenset()]) > 0, "the grand total must be worth materializing")

    # 10. determinism across repeated runs
    work, _ = random_lattice(0xABCD, dimensions=3)
    engine_a, engine_b = ViewSelection(work), ViewSelection(work)
    check(engine_a.greedy(3) == engine_b.greedy(3), "greedy must be deterministic")
    check(engine_a.greedy_with_budget(30) == engine_b.greedy_with_budget(30),
          "budgeted greedy must be deterministic")
    check(engine_a.total_cost(engine_a.greedy(3)) <= engine_a.baseline(),
          "selected views must never make the workload more expensive")

    print(f"656 lattice nodes={len(work.nodes)} baseline={engine_a.baseline():.0f} "
          f"greedy3={engine_a.total_cost(engine_a.greedy(3)):.0f}")
    print("PASS")


if __name__ == "__main__":
    main()
