"""Task 654 - Implement workload-driven index selection greedy heuristic.  (Tier: Expert, Ch 14 - Databases)

Given a table, a workload of equality-filtered queries with weights and a cost model, the advisor
chooses which indexes to build. An index is the ordered tuple of its key columns and is usable by a
query only under the prefix rule: an index can be probed on whole leading key columns, so the query's
filter list and the index key must start with each other (an index on (a) serves a query filtering on
a and b, applying b as a residual check; an index on (a, b) serves a query filtering only a, wasting
the b component). A usable index costs INDEX_BASE plus one probe per row matching the shared prefix;
if the index key is *exactly* the filter columns it is covering, so the row lookups disappear
(COVER_FACTOR). A full scan costs SCAN_ROW per row. Storage and build cost grow with the number of
key columns, and the advisor respects a storage budget expressed in row-entry units.

The search space is exponential, so the advisor runs the standard greedy heuristic: repeatedly add
the candidate with the largest net gain (marginal benefit minus one-off build cost), breaking ties
deterministically by key order, and stop as soon as no candidate pays off. The benefit function
benefit(S) = sum over queries of weight * max(0, scan - best usable index cost) - scan is a coverage
function (a maximum over non-negative per-index utilities), hence monotone and submodular; the
submodularity is checked explicitly (diminishing returns for every candidate), and under a cardinality
constraint the greedy result is cross-checked against the exhaustive optimum: optimum >= greedy and
greedy >= (1 - 1/e) * optimum, the classic guarantee.

Simplified: statistics are given (fixed selectivities per predicate, no histograms), one query shape
per workload entry, costs are unitless constants instead of I/O calibrations, and only equality
filters are modelled. Verified: exhaustive-optimum comparison on random small instances, diminishing
returns on random candidate sets, budget feasibility, greedy local optimality, determinism and the
hand-checked single-query case.
"""
from itertools import combinations

SCAN_ROW = 10.0              # cost of examining one row of the heap table
PROBE_ROW = 2.0              # cost of fetching one row through an index
COVER_FACTOR = 0.5           # an index-only (covering) scan skips the row fetches
INDEX_BASE = 50.0            # fixed per-index overhead charged on every query
BUILD_PER_ROW_PER_COL = 1.0  # one-off build/write cost per row and key column
STORE_PER_ROW_PER_COL = 1.0  # storage units per row and key column
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


class Query:
    """One workload entry: equality filters in evaluation order plus a frequency weight."""

    __slots__ = ("name", "filters", "weight")

    def __init__(self, name, filters, weight=1.0):
        self.name = name
        self.filters = tuple(filters)
        self.weight = float(weight)

    @property
    def columns(self):
        return tuple(column for column, _ in self.filters)

    @property
    def selectivity(self):
        selectivity = 1.0
        for _, partial in self.filters:
            selectivity *= partial
        return selectivity


def matched_prefix(query, index):
    """Length of the common prefix of the query's filter list and the index key (0 = unusable).

    The index can only be probed on whole leading key columns, so it is usable when one list starts
    with the other: an index on (a) serves a query filtering a and b (b is a residual check) and an
    index on (a, b) serves a query filtering only a (the b component stays unused).
    """
    length = 0
    for filter_column, key_column in zip(query.columns, index):
        if filter_column != key_column:
            break
        length += 1
    return length


def index_query_cost(query, index, rows):
    """Cost of answering `query` with `index` available, or None when the prefix rule rejects it."""
    length = matched_prefix(query, index)
    if length == 0:
        return None
    selectivity = 1.0
    for _, partial in query.filters[:length]:
        selectivity *= partial
    probed = rows * selectivity
    covering = index == query.columns
    return INDEX_BASE + probed * PROBE_ROW * (COVER_FACTOR if covering else 1.0)


def index_storage(index, rows):
    return len(index) * rows * STORE_PER_ROW_PER_COL


def index_build_cost(index, rows):
    return len(index) * rows * BUILD_PER_ROW_PER_COL


class Workload:
    """Cost/benefit bookkeeping for a table plus its query workload."""

    def __init__(self, rows, queries):
        self.rows = rows
        self.queries = list(queries)

    def scan_total(self):
        """Weighted cost of answering the whole workload without any index."""
        return sum(query.weight * SCAN_ROW * self.rows for query in self.queries)

    def total_cost(self, indices):
        """Sum of the cheapest cost per query; a query with no usable index falls back to a scan."""
        total = 0.0
        for query in self.queries:
            best = SCAN_ROW * self.rows
            for index in indices:
                cost = index_query_cost(query, index, self.rows)
                if cost is not None:
                    best = min(best, cost)
            total += query.weight * best
        return total

    def benefit(self, indices, scan_total=None):
        base = self.scan_total() if scan_total is None else scan_total
        return base - self.total_cost(indices)

    def marginal(self, indices, candidate):
        return self.benefit(indices + [candidate]) - self.benefit(indices)

    def net_gain(self, indices, candidate):
        """Marginal benefit minus the one-off build cost of the candidate index."""
        return self.marginal(indices, candidate) - index_build_cost(candidate, self.rows)


def candidate_indices(workload):
    """Every prefix of every query's filter list - the only keys the prefix rule can use."""
    candidates = set()
    for query in workload.queries:
        for length in range(1, len(query.columns) + 1):
            candidates.add(query.columns[:length])
    return sorted(candidates)


def greedy_by_benefit(workload, candidates, max_indices):
    """Cardinality-constrained greedy maximisation of the submodular benefit."""
    chosen = []
    while len(chosen) < max_indices:
        best, best_gain = None, 0.0
        for candidate in candidates:
            if candidate in chosen:
                continue
            gain = workload.marginal(chosen, candidate)
            if gain > best_gain + EPS:
                best, best_gain = candidate, gain
        if best is None:
            break
        chosen.append(best)
    return chosen


def exhaustive_best_benefit(workload, candidates, max_indices):
    best = 0.0
    for size in range(1, max_indices + 1):
        for combo in combinations(candidates, size):
            best = max(best, workload.benefit(list(combo)))
    return best


def select_indices(workload, budget, candidates=None):
    """Budget-aware greedy advisor: best net gain per storage unit, then best net gain."""
    if candidates is None:
        candidates = candidate_indices(workload)
    chosen = []
    used = 0.0
    scan_total = workload.scan_total()
    while True:
        best, best_ratio, best_gain = None, 0.0, 0.0
        for candidate in candidates:
            if candidate in chosen:
                continue
            storage = index_storage(candidate, workload.rows)
            if used + storage > budget + EPS:
                continue
            gain = workload.net_gain(chosen, candidate)
            if gain <= EPS:
                continue
            ratio = gain / storage
            if ratio > best_ratio + EPS:
                best, best_ratio, best_gain = candidate, ratio, gain
        if best is None:
            break
        chosen.append(best)
        used += index_storage(best, workload.rows)
    check(used <= budget + EPS, "the advisor must respect the storage budget")
    check(workload.benefit(chosen, scan_total) >= -EPS, "benefit can never be negative")
    return chosen


def random_workload(seed, rows=100000, column_count=4, query_count=5):
    columns = ["c%d" % i for i in range(column_count)]
    queries = []
    for index in range(query_count):
        seed, r1 = splitmix64(seed)
        width = 1 + r1 % 2
        seed, r2 = splitmix64(seed)
        start = r2 % (column_count - width + 1)
        filters = []
        for column in columns[start:start + width]:
            seed, r3 = splitmix64(seed)
            filters.append((column, (r3 % 40 + 1) / 100.0))
        seed, r4 = splitmix64(seed)
        queries.append(Query("q%d" % index, filters, weight=1 + r4 % 3))
    return Workload(rows, queries), seed


def main():
    # 1. single-query hand case: one index pays off, its net gain is exactly benefit - build cost
    workload = Workload(1000, [Query("q", [("a", 0.05)])])
    check(index_query_cost(workload.queries[0], ("a",), 1000) == INDEX_BASE + 50 * PROBE_ROW * COVER_FACTOR,
          "a covering index must charge only the matching rows")
    check(index_query_cost(workload.queries[0], ("a", "b"), 1000) == INDEX_BASE + 50 * PROBE_ROW,
          "a wider index is not covering and pays full row probes")
    check(index_storage(("a", "b"), 1000) == 2000, "storage must scale with keys and rows")
    chosen = select_indices(workload, budget=1e12)
    check(chosen == [("a",)], "the advisor must build the only useful index")
    check(abs(workload.benefit(chosen)
              - (workload.scan_total() - index_query_cost(workload.queries[0], ("a",), 1000))) < 1e-6,
          "benefit must be scan cost minus the index scan cost")
    check(workload.total_cost(chosen) < workload.total_cost([]), "an index must lower the cost")

    # 2. no workload / no usable filter: nothing to build
    check(select_indices(Workload(1000, []), budget=1e12) == [], "an empty workload needs no index")
    bare = Workload(1000, [Query("scan", [])])
    check(candidate_indices(bare) == [], "a query without filters offers no candidate index")
    check(select_indices(bare, budget=1e12) == [], "a scan-only query must not trigger an index")

    # 3. prefix rule rejects an index that does not lead with the first filter column
    two = Workload(1000, [Query("q", [("a", 0.1), ("b", 0.5)])])
    check(index_query_cost(two.queries[0], ("b",), 1000) is None,
          "an index leading with the wrong column cannot be used")
    check(index_query_cost(two.queries[0], ("a",), 1000) is not None,
          "the first filter column is a usable prefix")
    check(index_query_cost(two.queries[0], ("a", "b"), 1000) is not None, "the full key is usable")
    check(index_query_cost(two.queries[0], ("a", "b"), 1000)
          < index_query_cost(two.queries[0], ("a",), 1000),
          "the covering key must be cheaper than the partial one")

    # 4. budget handling: zero budget builds nothing, a tight budget still stays feasible
    rich = select_indices(workload, budget=1e12)
    check(select_indices(workload, budget=0.0) == [], "a zero budget must build nothing")
    tight = select_indices(workload, budget=index_storage(("a",), 1000))
    check(tight == [("a",)], "the cheapest useful index must fit an exact budget")
    check(sum(index_storage(i, 1000) for i in rich) <= 1e12, "storage must stay inside the budget")

    # 5. greedy local optimality and monotonicity
    workload, seed = random_workload(0xABCD)
    candidates = candidate_indices(workload)
    picked = select_indices(workload, budget=1e12)
    used = sum(index_storage(i, workload.rows) for i in picked)
    for candidate in candidates:
        if candidate in picked:
            continue
        if used + index_storage(candidate, workload.rows) <= 1e12:
            check(workload.net_gain(picked, candidate) <= EPS,
                  "greedy must stop only when no affordable candidate gains anything")
    for candidate in candidates:
        check(workload.marginal(picked, candidate) >= -EPS, "benefit must be monotone in the index set")

    # 6. submodularity (diminishing returns) on random index sets
    for _ in range(20):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        seed, r3 = splitmix64(seed)
        small = [candidates[r1 % len(candidates)]]
        big = list(dict.fromkeys(small + [candidates[r2 % len(candidates)]]))
        candidate = candidates[r3 % len(candidates)]
        if candidate in big:
            continue
        check(workload.marginal(small, candidate) >= workload.marginal(big, candidate) - EPS,
              "adding an index to a superset must not gain more (submodularity)")

    # 7. greedy versus the exhaustive optimum, and the (1 - 1/e) guarantee
    for trial in range(6):
        instance, seed = random_workload(seed, rows=2000, column_count=3, query_count=4)
        pool = candidate_indices(instance)
        if len(pool) > 8 or not pool:
            continue
        for cap in (1, 2, 3):
            greedy_set = greedy_by_benefit(instance, pool, cap)
            greedy_value = instance.benefit(greedy_set)
            optimum = exhaustive_best_benefit(instance, pool, cap)
            check(greedy_value <= optimum + EPS,
                  "no heuristic may beat the exhaustive optimum")
            check(greedy_value + EPS >= (1 - 1 / 2.718281828459045) * optimum,
                  f"greedy must reach the (1-1/e) factor (trial {trial}, cap {cap})")
            check(len(greedy_set) <= cap, "the cardinality constraint must hold")

    # 8. determinism of the advisor and of its candidate generation
    first = select_indices(workload, budget=1e12)
    second = select_indices(workload, budget=1e12)
    check(first == second, "the advisor must be deterministic")
    check(candidate_indices(workload) == candidates, "candidate generation must be deterministic")
    check(len(candidates) == len(set(candidates)), "candidates must be unique keys")

    # 9. the selected set really is a set of useful indexes
    check(all(candidate in candidates for candidate in first),
          "selected indexes must come from the candidate pool")
    check(len(set(first)) == len(first), "an index may not be selected twice")
    before = workload.scan_total()
    after = workload.total_cost(first)
    check(after < before, "the workload cost must drop when indexes are built")

    print(f"654 queries={len(workload.queries)} candidates={len(candidates)} selected={first}")
    print(f"654 cost scan={before:.0f} indexed={after:.0f} benefit={workload.benefit(first):.0f}")
    print("PASS")


if __name__ == "__main__":
    main()
