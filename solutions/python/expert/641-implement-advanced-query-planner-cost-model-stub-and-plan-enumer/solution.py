"""Task 641 - Implement advanced query planner cost model stub and plan enumeration.  (Tier: Expert, Ch 14 — Databases)

What is simplified (documented stub): the cost model is *unitless* - one CPU constant per
tuple, per index descent, per hash probe and per sort element - with no buffer pool, no
distinction between random and sequential I/O, no parallelism, and statistics that are a
single (rows, distinct values, selectivity) triple per table, i.e. uniform distributions
only.  The planner enumerates connected left-deep plans (every permutation x every access
path x every join algorithm) and keeps the cheapest.

The cost formulas are the interesting part and the checks pin their consequences:

    seq_scan(t)   = rows * C_TUPLE                                   (reads every row)
    index_scan(t) = (log2(rows) + 1) * C_INDEX + rows * sel * C_TUPLE
    nested_loop   = left * C_TUPLE + left * (index descent, or a full rescan of right)
    hash_join     = (left + right) * C_TUPLE + out * C_HASH_PROBE
    sort_merge    = (left + right) * C_TUPLE + sort(left) + sort(right)
    cardinality   = left * right / max(ndv(left), ndv(right))        (equality join)

Asserted consequences: the chosen plan is the minimum-cost plan of the enumerated set, all
enumerated plans are connected (no accidental cross product), index access wins exactly
below the closed-form selectivity threshold, a filter can only lower the best cost, an
index nested loop beats a hash join for a tiny outer side with an indexed inner while the
hash join wins when the inner has no index - and on uniformly distributed data the
cardinality estimate is *exactly* the number of rows the plan really produces.
"""

import itertools
import math

C_TUPLE = 1.0            # per tuple flowing through an operator
C_INDEX = 4.0            # per index descent
C_HASH_PROBE = 0.5       # per hash-table probe
C_SORT_TUPLE = 2.0       # per element of a sort
ALGORITHMS = ("nested_loop", "hash_join", "sort_merge")


def stat(rows, ndv, has_index=False, selectivity=1.0):
    return {"rows": rows, "ndv": ndv, "has_index": has_index, "selectivity": selectivity}


def estimated_rows(s):
    """Cardinality of the table *after* its pushed-down filter."""
    return max(1, int(round(s["rows"] * s["selectivity"])))


def seq_scan_cost(s):
    return s["rows"] * C_TUPLE


def index_scan_cost(s):
    return (math.log2(s["rows"]) + 1) * C_INDEX + estimated_rows(s) * C_TUPLE


def index_selectivity_threshold(rows):
    """Index access beats a sequential scan exactly below this selectivity."""
    return 1.0 - (math.log2(rows) + 1) * C_INDEX / rows


def scan_choice(s):
    """(access, cost) for the cheaper of the two access paths."""
    if not s["has_index"]:
        return "seq", seq_scan_cost(s)
    if index_scan_cost(s) < seq_scan_cost(s):
        return "index", index_scan_cost(s)
    return "seq", seq_scan_cost(s)


def join_cost(alg, left_rows, right_rows, out_rows, right_indexed):
    if alg == "nested_loop":
        per_probe = ((math.log2(right_rows) + 1) * C_INDEX if right_indexed
                     else right_rows * C_TUPLE)
        return left_rows * C_TUPLE + left_rows * per_probe
    if alg == "hash_join":
        return (left_rows + right_rows) * C_TUPLE + out_rows * C_HASH_PROBE
    if alg == "sort_merge":
        return ((left_rows + right_rows) * C_TUPLE
                + (left_rows * math.log2(max(left_rows, 2))
                   + right_rows * math.log2(max(right_rows, 2))) * C_SORT_TUPLE)
    raise AssertionError("unknown algorithm %r" % (alg,))


def evaluate(plan, catalog, memo=None):
    """(cost, estimated rows, ndv) of a plan; memoised because plans share subtrees."""
    if memo is None:
        memo = {}
    if plan in memo:
        return memo[plan]
    if plan[0] == "scan":
        _, name, access = plan
        s = catalog[name]
        cost = index_scan_cost(s) if access == "index" else seq_scan_cost(s)
        result = (cost, estimated_rows(s), s["ndv"])
    else:
        _, alg, left, right = plan
        lcost, lrows, lndv = evaluate(left, catalog, memo)
        rcost, rrows, rndv = evaluate(right, catalog, memo)
        out = lrows * rrows / max(lndv, rndv)
        right_indexed = right[0] == "scan" and right[2] == "index"
        jcost = join_cost(alg, lrows, rrows, out, right_indexed)
        if alg == "nested_loop" and right_indexed:
            total = lcost + jcost              # every probe pays for its own index descent
        else:
            total = lcost + rcost + jcost
        result = (total, out, max(lndv, rndv))
    memo[plan] = result
    return result


def plan_cost(plan, catalog):
    return evaluate(plan, catalog)[0]


def enumerate_plans(tables, catalog, edges):
    """All connected left-deep plans over a table set; `edges` is the join-predicate graph."""
    plans = []
    accesses = {t: (["seq", "index"] if catalog[t]["has_index"] else ["seq"]) for t in tables}
    for perm in itertools.permutations(tables):
        for access in itertools.product(*(accesses[t] for t in perm)):
            for algs in itertools.product(ALGORITHMS, repeat=max(0, len(tables) - 1)):
                plan = ("scan", perm[0], access[0])
                joined = {perm[0]}
                for i in range(1, len(tables)):
                    name = perm[i]
                    if not any(frozenset((name, other)) in edges for other in joined):
                        break                      # would be a cross product: discard
                    plan = ("join", algs[i - 1], plan, ("scan", name, access[i]))
                    joined.add(name)
                else:
                    plans.append(plan)
    return plans


def choose_best(plans, catalog):
    memo = {}
    best, best_cost = None, None
    for plan in plans:
        cost = evaluate(plan, catalog, memo)[0]
        if best_cost is None or cost < best_cost:
            best, best_cost = plan, cost
    return best, best_cost


def collect_scans(plan, out=None):
    if out is None:
        out = {}
    if plan[0] == "scan":
        out[plan[1]] = plan[2]
        return out
    collect_scans(plan[2], out)
    collect_scans(plan[3], out)
    return out


def collect_algs(plan, out=None):
    if out is None:
        out = []
    if plan[0] == "join":
        out.append(plan[1])
        collect_algs(plan[2], out)
        collect_algs(plan[3], out)
    return out


def table_set(plan):
    if plan[0] == "scan":
        return {plan[1]}
    return table_set(plan[2]) | table_set(plan[3])


def is_connected(plan, edges):
    """True when every join in the plan joins two sides linked by a join predicate."""
    if plan[0] == "scan":
        return True
    left_tables, right_tables = table_set(plan[2]), table_set(plan[3])
    if not any(frozenset((a, b)) in edges for a in left_tables for b in right_tables):
        return False
    return is_connected(plan[2], edges) and is_connected(plan[3], edges)


def build_relations(specs, key_domain):
    """Uniform test data: keys cycle over 1..key_domain, so ndv == key_domain."""
    relations = {}
    for seed, (name, rows) in enumerate(specs):
        relations[name] = [((seed + 1) * 1000 + i, (i % key_domain) + 1) for i in range(rows)]
    return relations


def execute(plan, relations):
    """Relational execution of a plan; an equality join makes every algorithm agree."""
    if plan[0] == "scan":
        return list(relations[plan[1]])
    left = execute(plan[2], relations)
    right = execute(plan[3], relations)
    buckets = {}
    for row in right:
        buckets.setdefault(row[-1], []).append(row)
    return [l + r for l in left for r in buckets.get(l[-1], ())]


def chain_edges(names):
    return {frozenset((names[i], names[i + 1])) for i in range(len(names) - 1)}


def main():
    # --- enumeration sizes follow permutations x access paths x algorithms ---------------
    star = {"C": stat(100, 50, True), "A": stat(80, 40, True), "B": stat(60, 30, True)}
    star_plans = enumerate_plans(["C", "A", "B"], star,
                                 {frozenset(("C", "A")), frozenset(("C", "B"))})
    check(len(star_plans) == 4 * 2 ** 3 * 3 ** 2,
          "star enumeration size (4 connected orders): %d" % len(star_plans))
    check(len(set(star_plans)) == len(star_plans), "no duplicate plans")

    two = {"S": stat(10, 10), "B": stat(100000, 100000, True)}
    check(len(enumerate_plans(["S", "B"], two, {frozenset(("S", "B"))})) == 2 * 2 * 3,
          "two-table enumeration size")

    # --- the chosen plan is the cheapest enumerated plan, and every plan is connected ---
    chain = {"T1": stat(1000, 1000, True), "T2": stat(500, 1000, True),
             "T3": stat(2000, 1000, True), "T4": stat(50, 1000, True)}
    edges = chain_edges(["T1", "T2", "T3", "T4"])
    plans = enumerate_plans(["T1", "T2", "T3", "T4"], chain, edges)
    best, best_cost = choose_best(plans, chain)
    check(len(plans) == len(set(plans)), "the chain enumeration has no duplicates")
    check(len(plans) == 8 * 2 ** 4 * 3 ** 3,
          "path enumeration size (8 connected orders, 2 accesses, 3 algorithms): %d"
          % len(plans))
    check(best_cost == min(plan_cost(p, chain) for p in plans),
          "choose_best returns the minimum-cost plan of the enumeration")
    check(all(is_connected(p, edges) for p in plans), "every enumerated plan is connected")
    check(choose_best(plans, chain)[0] == best, "the chosen plan is reproducible")
    check(len(collect_scans(best)) == 4, "the best plan scans every table exactly once")

    manual = ("join", "hash_join", ("scan", "T1", "seq"), ("scan", "T2", "seq"))
    manual_cost = (scan_choice(chain["T1"])[1] + scan_choice(chain["T2"])[1]
                   + join_cost("hash_join", 1000, 500, 1000 * 500 / 1000, False))
    check(abs(plan_cost(manual, chain) - manual_cost) < 1e-9,
          "a plan costs the sum of its scan, join and sort terms")

    # --- access path threshold: the closed form of the cost formula ---------------------
    threshold = index_selectivity_threshold(1000)
    check(0.0 < threshold < 1.0, "the crossover threshold is a real selectivity")
    for sel in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0):
        s = stat(1000, 1000, True, sel)
        access, cost = scan_choice(s)
        check((access == "index") == (sel < threshold),
              "crossover formula disagrees at selectivity %.2f (access %s)" % (sel, access))
        check(access != "index" or cost == index_scan_cost(s), "index cost is the index formula")
        check(access != "seq" or cost == seq_scan_cost(s), "seq cost is the sequential formula")
    check(scan_choice(stat(1000, 1000, True, 0.05))[0] == "index", "5%% selectivity uses the index")
    check(scan_choice(stat(1000, 1000, True, 0.99))[0] == "seq", "99%% selectivity scans the table")
    check(scan_choice(stat(1000, 1000, False, 0.001))[0] == "seq",
          "without an index only a sequential scan exists")

    # --- a filter can only lower the best feasible cost ---------------------------------
    filtered = dict(chain)
    filtered["T3"] = stat(2000, 1000, True, selectivity=0.25)
    filtered_best, filtered_cost = choose_best(
        enumerate_plans(["T1", "T2", "T3", "T4"], filtered, edges), filtered)
    check(filtered_cost < best_cost,
          "a 25%% filter lowers the best cost: %.1f < %.1f" % (filtered_cost, best_cost))
    check(collect_scans(filtered_best)["T3"] == "index",
          "the selective table is reached through its index: %r" % (collect_scans(filtered_best),))

    # --- algorithm consequences of the cost model ---------------------------------------
    tiny_indexed = join_cost("nested_loop", 10, 100000, 10, True)
    hash_tiny = join_cost("hash_join", 10, 100000, 10, True)
    check(tiny_indexed < hash_tiny,
          "index nested loop wins for a tiny outer: %.1f < %.1f" % (tiny_indexed, hash_tiny))
    check(hash_tiny < join_cost("nested_loop", 10, 100000, 10, False),
          "hash join wins when the inner cannot be indexed")
    check(join_cost("sort_merge", 1000, 1000, 1000, False)
          > join_cost("hash_join", 1000, 1000, 1000, False),
          "sort-merge pays for sorting and loses to a hash join at equal sizes")
    check(join_cost("sort_merge", 5, 5, 5, False) > join_cost("hash_join", 5, 5, 5, False)
          and join_cost("sort_merge", 5, 5, 5, False) > join_cost("nested_loop", 5, 5, 5, False),
          "sort-merge is the worst choice for tiny inputs")

    big = {"S": stat(10, 10), "B": stat(100000, 100000, True)}
    best_indexed, _ = choose_best(enumerate_plans(["S", "B"], big, {frozenset(("S", "B"))}), big)
    check(collect_algs(best_indexed) == ["nested_loop"],
          "planner picks the index nested loop: %r" % (collect_algs(best_indexed),))
    check(collect_scans(best_indexed) == {"S": "seq", "B": "index"},
          "with the indexed table on the inner side: %r" % (collect_scans(best_indexed),))
    noindex = {"S": stat(10, 10), "B": stat(100000, 100000, False)}
    best_plain, _ = choose_best(enumerate_plans(["S", "B"], noindex, {frozenset(("S", "B"))}),
                                noindex)
    check(collect_algs(best_plain) == ["hash_join"],
          "planner picks the hash join without an index: %r" % (collect_algs(best_plain),))

    # --- the cardinality estimate is exact on uniform data ------------------------------
    relations = build_relations([("T1", 8), ("T2", 12)], 4)
    uniform = {"T1": stat(8, 4), "T2": stat(12, 4)}
    plan2 = ("join", "hash_join", ("scan", "T1", "seq"), ("scan", "T2", "seq"))
    estimate = evaluate(plan2, uniform)[1]
    actual = len(execute(plan2, relations))
    check(estimate == actual == 24,
          "two-way estimate %s vs actual %s" % (estimate, actual))

    tri_relations = build_relations([("T1", 4), ("T2", 8), ("T3", 12)], 4)
    tri = {"T1": stat(4, 4), "T2": stat(8, 4), "T3": stat(12, 4)}
    tri_edges = chain_edges(["T1", "T2", "T3"])
    tri_plans = enumerate_plans(["T1", "T2", "T3"], tri, tri_edges)
    check(len(tri_plans) == 4 * 3 ** 2, "path enumeration keeps 4 connected orders: %d"
          % len(tri_plans))
    tri_best, _ = choose_best(tri_plans, tri)
    tri_actual = len(execute(tri_best, tri_relations))
    check(tri_actual == 24, "three-way join really produces 24 rows")
    check(evaluate(tri_best, tri)[1] == tri_actual,
          "three-way cardinality estimate equals the real row count")
    print("[641] planner: %d plans enumerated for the 4-table chain, best cost %.1f"
          % (len(plans), best_cost))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
