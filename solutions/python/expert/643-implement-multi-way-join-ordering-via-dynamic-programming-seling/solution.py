"""Task 643 - Implement multi-way join ordering via dynamic programming (Selinger-style).  (Tier: Expert, Ch 14 — Databases)

Selinger-style cost-based join enumeration.  `best[S]` is the cheapest plan for the join of
the table set S; for every proper subset A of S the DP combines `best[A]` with `best[S\\A]`,
prices the result and keeps the winner.  The DP relies on the *principle of optimality* -
the cheapest plan for a sub-expression is part of the cheapest plan for the whole query -
and the checks verify exactly that by comparing the DP against a brute-force enumeration of
every bushy join tree with every join-algorithm assignment on small queries: the two must
return the same plan cost, while the DP examines far fewer candidates as the query grows.

Documented simplifications: cross products are pruned - a split is only considered when at
least one join predicate connects its two sides (what real planners do).  The checks confirm
that the pruned optimum equals the best connected plan and never beats the unrestricted
optimum, so nothing cheaper is lost.

Cost model (unitless): a scan is free, a join pays `C_TUPLE` per input row if hashed and
`|L| * |R|` if nested-looped, plus one per output row; the output cardinality of a join is
`|L| * |R| * selectivity(L, R)`, the selectivity being the product of the selectivities of
all predicates crossing the two sides.  The chosen plan is additionally executed over small
relations and must reproduce the true join computed by a Cartesian-product oracle.
"""

import itertools

C_TUPLE = 1.0
ALGORITHMS = ("hash_join", "nested_loop")


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def sorted_edges(sels):
    return [tuple(sorted(edge)) for edge in sorted(sels, key=lambda e: tuple(sorted(e)))]


def crossing_edges(left_tables, right_tables, sels):
    return [(a, b) for a, b in sorted_edges(sels)
            if (a in left_tables and b in right_tables)
            or (b in left_tables and a in right_tables)]


def cross_selectivity(left_tables, right_tables, sels):
    product = 1.0
    for a, b in crossing_edges(left_tables, right_tables, sels):
        product *= sels[frozenset((a, b))]
    return product


def plan_tables(plan):
    if plan[0] == "scan":
        return frozenset([plan[1]])
    return plan_tables(plan[2]) | plan_tables(plan[3])


def is_connected(plan, sels):
    if plan[0] == "scan":
        return True
    if not crossing_edges(plan_tables(plan[2]), plan_tables(plan[3]), sels):
        return False
    return is_connected(plan[2], sels) and is_connected(plan[3], sels)


def join_variant_cost(alg, left_rows, right_rows, out_rows):
    if alg == "hash_join":
        return (left_rows + right_rows) * C_TUPLE + out_rows * C_TUPLE
    if alg == "nested_loop":
        return left_rows * right_rows * C_TUPLE + out_rows * C_TUPLE
    raise AssertionError("unknown algorithm %r" % (alg,))


def evaluate(plan, sizes, sels):
    """(cost, output rows) of a physical plan tree; cross products are priced too."""
    if plan[0] == "scan":
        return 0.0, float(sizes[plan[1]])
    left_cost, left_rows = evaluate(plan[2], sizes, sels)
    right_cost, right_rows = evaluate(plan[3], sizes, sels)
    out_rows = left_rows * right_rows * cross_selectivity(plan_tables(plan[2]),
                                                          plan_tables(plan[3]), sels)
    return (left_cost + right_cost
            + join_variant_cost(plan[1], left_rows, right_rows, out_rows), out_rows)


def proper_subsets(table_set):
    items = sorted(table_set)
    for size in range(1, len(items)):
        for combo in itertools.combinations(items, size):
            yield frozenset(combo)


def selinger_dp(tables, sizes, sels):
    """Returns (plan, cost, rows, splits_considered, splits_pruned)."""
    names = sorted(tables)
    info = {frozenset([name]): (("scan", name), 0.0, float(sizes[name])) for name in names}
    splits, pruned = 0, 0
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(names, size):
            table_set = frozenset(combo)
            best = None
            for left_set in proper_subsets(table_set):
                right_set = table_set - left_set
                if info[left_set] is None or info[right_set] is None:
                    pruned += 1                       # a disconnected side has no plan
                    continue
                if not crossing_edges(left_set, right_set, sels):
                    pruned += 1                       # would have been a cross product
                    continue
                splits += 1
                left_plan, left_cost, left_rows = info[left_set]
                right_plan, right_cost, right_rows = info[right_set]
                out_rows = left_rows * right_rows * cross_selectivity(left_set, right_set, sels)
                for alg in ALGORITHMS:
                    cost = (left_cost + right_cost
                            + join_variant_cost(alg, left_rows, right_rows, out_rows))
                    if best is None or cost < best[1]:
                        best = (("join", alg, left_plan, right_plan), cost, out_rows)
            info[table_set] = best
    plan, cost, rows = info[frozenset(names)]
    return plan, cost, rows, splits, pruned


def enumerate_trees(items):
    """Every unordered bushy join tree over `items` (each tree generated exactly once)."""
    if len(items) == 1:
        return [("scan", items[0])]
    first, rest = items[0], items[1:]
    trees = []
    for size in range(len(rest) + 1):
        for combo in itertools.combinations(rest, size):
            left_items = (first,) + combo
            right_items = tuple(name for name in rest if name not in combo)
            if not right_items:
                continue
            for left_tree in enumerate_trees(left_items):
                for right_tree in enumerate_trees(right_items):
                    trees.append(("join", left_tree, right_tree))
    return trees


def assign_algorithms(tree, algs):
    """Turn a raw tree into a physical plan, consuming algorithms in pre-order."""
    if tree[0] == "scan":
        return tree
    algorithm = next(algs)
    left = assign_algorithms(tree[1], algs)
    right = assign_algorithms(tree[2], algs)
    return ("join", algorithm, left, right)


def brute_force_best(tables, sizes, sels):
    """Unrestricted exhaustive enumeration: every tree x every algorithm assignment."""
    names = tuple(sorted(tables))
    best, count = None, 0
    for tree in enumerate_trees(names):
        for algs in itertools.product(ALGORITHMS, repeat=len(names) - 1):
            plan = assign_algorithms(tree, iter(algs))
            cost, _ = evaluate(plan, sizes, sels)
            count += 1
            if best is None or cost < best[1]:
                best = (plan, cost)
    return best[0], best[1], count


def connected_brute_force_best(tables, sizes, sels):
    """Exhaustive enumeration restricted to connected trees - the DP's own plan space."""
    names = tuple(sorted(tables))
    best = None
    for tree in enumerate_trees(names):
        if not is_connected(assign_algorithms(tree, iter(ALGORITHMS * len(names))), sels):
            continue
        for algs in itertools.product(ALGORITHMS, repeat=len(names) - 1):
            plan = assign_algorithms(tree, iter(algs))
            cost, _ = evaluate(plan, sizes, sels)
            if best is None or cost < best[1]:
                best = (plan, cost)
    return best


def random_query(names, seed, extra_edge=False):
    state, sizes, sels = seed, {}, {}
    for name in names:
        state = lcg(state)
        sizes[name] = 2 + state % 200
    for i in range(len(names) - 1):
        state = lcg(state)
        sels[frozenset((names[i], names[i + 1]))] = 0.02 + (state % 900) / 1000.0 * 0.45
    if extra_edge and len(names) > 2:
        state = lcg(state)
        sels[frozenset((names[0], names[2]))] = 0.05 + (state % 500) / 1000.0 * 0.3
    return sizes, sels


def build_relations(names, key_domain, rows_per_table):
    relations = {}
    for i, name in enumerate(names):
        relations[name] = [{"payload": i * 10 + j, "key": (i + j) % key_domain}
                           for j in range(rows_per_table)]
    return relations


def execute(plan, relations, sels):
    """Apply every predicate crossing each join, in tree order."""
    if plan[0] == "scan":
        return [{plan[1]: row} for row in relations[plan[1]]]
    left_rows = execute(plan[2], relations, sels)
    right_rows = execute(plan[3], relations, sels)
    applied = crossing_edges(plan_tables(plan[2]), plan_tables(plan[3]), sels)
    out = []
    for l in left_rows:
        for r in right_rows:
            merged = dict(l)
            merged.update(r)
            if all(merged[a]["key"] == merged[b]["key"] for a, b in applied):
                out.append(merged)
    return out


def canonical(rows):
    """Order-insensitive, comparable view of a join result."""
    return sorted(tuple(sorted((table, row["payload"]) for table, row in result.items()))
                  for result in rows)


def true_join(relations, sels):
    """Oracle: the Cartesian product of every table, filtered by every predicate."""
    names = sorted(relations)
    edges = sorted_edges(sels)
    out = []
    for combo in itertools.product(*(relations[name] for name in names)):
        row = dict(zip(names, combo))
        if all(row[a]["key"] == row[b]["key"] for a, b in edges):
            out.append(row)
    return out


def main():
    check(len(enumerate_trees(("a", "b", "c", "d"))) == 15
          and len(enumerate_trees(("a", "b", "c", "d", "e"))) == 105,
          "15 unordered trees for 4 tables, 105 for 5 tables")

    # asymmetric sizes: the two-row table must be joined first, never the big-big pair
    sizes = {"A": 1000, "B": 2, "C": 1000}
    sels = {frozenset(("A", "B")): 0.1, frozenset(("B", "C")): 0.1}
    plan, cost, rows, splits, pruned = selinger_dp(("A", "B", "C"), sizes, sels)
    brute_plan, brute_cost, brute_count = brute_force_best(("A", "B", "C"), sizes, sels)
    check(abs(cost - brute_cost) < 1e-9,
          "DP cost %.3f equals unrestricted exhaustive cost %.3f" % (cost, brute_cost))
    check(is_connected(plan, sels), "the chosen plan is connected: %r" % (plan,))
    check(abs(rows - 20000.0) < 1e-6,
          "estimated cardinality follows the formula: %r" % (rows,))
    worst_plan = ("join", "hash_join",
                  ("join", "hash_join", ("scan", "A"), ("scan", "C")), ("scan", "B"))
    check(cost < evaluate(worst_plan, sizes, sels)[0],
          "the big-big-first plan is more expensive than the chosen one")
    check(splits <= brute_count and pruned > 0,
          "the DP prunes cross products (%d pruned) and examines %d splits" % (pruned, splits))

    # headline invariant: pruned DP == exhaustive enumeration over the connected plan space
    for tables, seed, extra in ((["T1", "T2", "T3", "T4"], 11, False),
                                (["T1", "T2", "T3", "T4"], 12, True),
                                (["T1", "T2", "T3", "T4", "T5"], 13, False)):
        for step in range(8):
            sizes, sels = random_query(tables, seed * 1000 + step * 7919, extra)
            plan, cost, rows, splits, pruned = selinger_dp(tables, sizes, sels)
            _, brute_cost, brute_count = brute_force_best(tables, sizes, sels)
            connected = connected_brute_force_best(tables, sizes, sels)
            check(abs(cost - connected[1]) < 1e-9,
                  "DP %.6f != connected exhaustive %.6f (%r seed %d)"
                  % (cost, connected[1], tables, step))
            check(brute_cost <= cost + 1e-9,
                  "pruning never loses money: unrestricted %.6f <= DP %.6f"
                  % (brute_cost, cost))
            check(splits < brute_count and pruned > 0,
                  "the DP examines %d of %d candidate plans" % (splits, brute_count))
            check(is_connected(plan, sels),
                  "a connected query yields a connected cheapest plan: %r" % (plan,))
            check(selinger_dp(tables, sizes, sels)[0] == plan, "the DP is deterministic")

    # at six tables exhaustive enumeration explodes while the DP stays tiny
    six = ["T%d" % (i + 1) for i in range(6)]
    sizes6, sels6 = random_query(six, 98765)
    plan6, cost6, _, splits6, pruned6 = selinger_dp(six, sizes6, sels6)
    connected6 = connected_brute_force_best(six, sizes6, sels6)
    _, brute_cost6, brute_count6 = brute_force_best(six, sizes6, sels6)
    check(abs(cost6 - connected6[1]) < 1e-9,
          "6-table DP %.3f equals connected exhaustive %.3f" % (cost6, connected6[1]))
    check(brute_cost6 <= cost6 + 1e-9, "6-table pruning is safe")
    check(splits6 * 20 < brute_count6,
          "at six tables the DP examines %d splits against %d enumerated plans"
          % (splits6, brute_count6))
    check(pruned6 > 0 and is_connected(plan6, sels6), "the 6-table plan is connected")

    # executing the chosen plan reproduces the true join
    names = ["T1", "T2", "T3", "T4"]
    sizes, sels = random_query(names, 4242)
    relations = build_relations(names, key_domain=2, rows_per_table=2)
    plan, cost, _, _, _ = selinger_dp(names, sizes, sels)
    oracle = true_join(relations, sels)
    check(len(oracle) > 0, "the oracle produces a non-empty join")
    planned = execute(plan, relations, sels)
    alternative = execute(connected_brute_force_best(names, sizes, sels)[0], relations, sels)
    check(len(planned) == len(oracle) and len(alternative) == len(oracle),
          "both plans produce the true join: %d / %d / %d"
          % (len(planned), len(alternative), len(oracle)))
    check(canonical(planned) == canonical(oracle),
          "the executed plan matches the Cartesian-product oracle row by row")

    # nested_loop is chosen only when it is genuinely cheaper
    tiny = {"X": 1, "Y": 2, "Z": 100000}
    tiny_sels = {frozenset(("X", "Y")): 0.5, frozenset(("Y", "Z")): 0.5}
    tiny_plan, tiny_cost, _, _, _ = selinger_dp(("X", "Y", "Z"), tiny, tiny_sels)
    check(abs(tiny_cost - connected_brute_force_best(("X", "Y", "Z"), tiny, tiny_sels)[1]) < 1e-9,
          "the tiny-input query matches the connected exhaustive optimum")
    check("nested_loop" in walk_algs(tiny_plan),
          "a 1x2 intermediate is cheap enough for a nested loop: %r" % (tiny_plan,))
    biggish = {"X": 10, "Y": 10, "Z": 1000}
    biggish_sels = {frozenset(("X", "Y")): 0.5, frozenset(("Y", "Z")): 0.5}
    big_plan, _, _, _, _ = selinger_dp(("X", "Y", "Z"), biggish, biggish_sels)
    check("hash_join" in walk_algs(big_plan),
          "a 10x10 intermediate is hashed instead: %r" % (walk_algs(big_plan),))
    check(walk_algs(tiny_plan)[0] == "nested_loop",
          "the outermost 1-row join is nested-looped: %r" % (tiny_plan,))
    print("[643] Selinger DP == exhaustive enumeration; %d splits (%d pruned) vs %d plans at "
          "6 tables" % (splits6, pruned6, brute_count6))


def walk_algs(plan, out=None):
    if out is None:
        out = []
    if plan[0] == "join":
        out.append(plan[1])
        walk_algs(plan[2], out)
        walk_algs(plan[3], out)
    return out


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
