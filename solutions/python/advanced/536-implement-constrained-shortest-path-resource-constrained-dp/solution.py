"""Task 536 - Implement constrained shortest path (resource constrained) DP. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)

The resource-constrained shortest path problem: every arc carries a non-negative cost and a
non-negative integer resource consumption, and we want the cheapest source-to-target route whose
total consumption stays inside a budget B. Because consumption never decreases, the state space is
the layered product vertex x {0..B} and the optimum satisfies a Bellman equation
dp[r][v] = min cost reaching v with consumption exactly r; arcs with resource > 0 move to a higher
layer, arcs with resource == 0 stay inside the layer and are closed with n Bellman-Ford rounds per
layer (so zero-resource cycles and chains are handled exactly). The reported answer is the minimum
over the layers 0..B, which is what makes it "at most the budget" rather than "exactly".

Four independent methods are cross-checked and instrumented: the layered DP above (counting
relaxations), a Dijkstra over the same expanded state graph (an independent shortest-path engine;
state tuples (cost, vertex, used) break ties deterministically), a brute force that enumerates all
simple paths with pruning - valid because dropping a cycle never increases consumption (resources
are non-negative) and never increases cost (costs are non-negative) - and a plain Dijkstra on the
cost-only graph, which must coincide with the constrained solver once the budget is at least the
total consumption of every arc.

Checks: the hand-built example where the cheapest route (cost 0, resource 3) is infeasible at
budget 2 while the cost-2 route fits, then becomes optimal at budget 3; the zero-resource cycle
0->1->0 (both resource 0) that the layer closure must absorb without looping; source == target
returning 0 for every budget including 0; an unreachable target reported as INF by all three
solvers; agreement of DP, expanded Dijkstra and brute force on forty LCG arc-sets (n = 5, costs 0-9,
resources 0-3, budgets 0-6), monitored with the number of explicit paths enumerated; monotonicity
of the optimum in the budget; and the budget-independence check against cost-only Dijkstra.
"""
import heapq

INF = 10 ** 15


def lcg(seed):
    return (seed * 75 + 74) % 65537


class ResourceGraph:
    """Directed graph whose arcs are (target, cost, resource consumption)."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.arc_count = 0

    def add_edge(self, u, v, cost, resource):
        assert 0 <= u < self.n and 0 <= v < self.n, "endpoints in range"
        assert cost >= 0 and resource >= 0, "non-negative cost and consumption"
        self.adj[u].append((v, cost, resource))
        self.arc_count += 1

    def arcs(self):
        return [(u, v, c, r) for u in range(self.n) for (v, c, r) in self.adj[u]]


def rcsp_dp(graph, source, target, budget):
    """Layered DP. Returns (best cost or INF, relaxation count, dp table)."""
    n = graph.n
    arcs = graph.arcs()
    free = [(u, v, c) for (u, v, c, r) in arcs if r == 0]
    costly = [(u, v, c, r) for (u, v, c, r) in arcs if r > 0]
    dp = [[INF] * n for _ in range(budget + 1)]
    dp[0][source] = 0
    relaxations = 0
    for level in range(budget + 1):
        for _ in range(n):                       # close the layer over zero-resource arcs
            changed = False
            for (u, v, c) in free:
                if dp[level][u] + c < dp[level][v]:
                    dp[level][v] = dp[level][u] + c
                    changed = True
                    relaxations += 1
            if not changed:
                break
        for (u, v, c, r) in costly:              # move up to a higher layer
            if dp[level][u] == INF:
                continue
            upper = level + r
            if upper <= budget and dp[level][u] + c < dp[upper][v]:
                dp[upper][v] = dp[level][u] + c
                relaxations += 1
    best = min(dp[level][target] for level in range(budget + 1))
    return (best if best < INF else INF), relaxations, dp


def rcsp_dijkstra(graph, source, target, budget):
    """Dijkstra over the expanded state graph. Returns (best cost or INF, pop count)."""
    n = graph.n
    dist = [[INF] * n for _ in range(budget + 1)]
    dist[0][source] = 0
    heap = [(0, source, 0)]
    pops = 0
    while heap:
        d, u, used = heapq.heappop(heap)
        if d > dist[used][u]:
            continue
        pops += 1
        for (v, c, r) in graph.adj[u]:
            nxt_used = used + r
            if nxt_used > budget:
                continue
            if d + c < dist[nxt_used][v]:
                dist[nxt_used][v] = d + c
                heapq.heappush(heap, (d + c, v, nxt_used))
    best = min(dist[used][target] for used in range(budget + 1))
    return (best if best < INF else INF), pops


def rcsp_brute_force(graph, source, target, budget):
    """Enumerate every simple path; returns (best cost or INF, partial paths enumerated)."""
    best = [INF]
    enumerated = [0]

    def walk(u, cost, used, seen):
        if u == target and cost < best[0]:
            best[0] = cost
        for (v, c, r) in graph.adj[u]:
            if v in seen or used + r > budget:
                continue
            enumerated[0] += 1
            walk(v, cost + c, used + r, seen | {v})

    walk(source, 0, 0, frozenset([source]))
    return best[0], enumerated[0]


def dijkstra_cost_only(graph, source, target):
    """Plain Dijkstra ignoring the resource dimension - the budget-free optimum."""
    dist = [INF] * graph.n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for (v, c, _) in graph.adj[u]:
            if d + c < dist[v]:
                dist[v] = d + c
                heapq.heappush(heap, (d + c, v))
    return dist[target]


def random_graph(n, seed, arcs=9):
    graph = ResourceGraph(n)
    for _ in range(arcs):
        seed = lcg(seed)
        u = seed % n
        seed = lcg(seed)
        v = seed % n
        if u == v:
            continue
        seed = lcg(seed)
        cost = seed % 10
        seed = lcg(seed)
        resource = seed % 4
        graph.add_edge(u, v, cost, resource)
    return graph, seed


def main():
    # --- Hand-built example: the cheapest route is the infeasible one -------------------
    graph = ResourceGraph(4)
    graph.add_edge(0, 1, 1, 1)
    graph.add_edge(1, 3, 1, 1)
    graph.add_edge(0, 2, 0, 3)
    graph.add_edge(2, 3, 0, 0)
    best, relaxations, dp = rcsp_dp(graph, 0, 3, 2)
    assert best == 2, "0-1-3 costs 2 with resource 2; the free 0-2-3 route needs 3"
    assert rcsp_dijkstra(graph, 0, 3, 2)[0] == 2
    assert rcsp_brute_force(graph, 0, 3, 2)[0] == 2
    assert rcsp_dp(graph, 0, 3, 3)[0] == 0, "at budget 3 the zero-cost route is feasible"
    _, _, wide_dp = rcsp_dp(graph, 0, 3, 3)
    assert wide_dp[3][3] == 0 and wide_dp[2][3] == 2, \
        "layer 2 holds the cost-2 route, layer 3 the free one"
    assert wide_dp[1][3] == INF, "neither route fits in a single unit of resource"
    assert relaxations > 0

    # --- Zero-resource cycle: the layer closure must absorb it --------------------------
    cyc = ResourceGraph(3)
    cyc.add_edge(0, 1, 1, 0)
    cyc.add_edge(1, 0, 1, 0)                     # 0 -> 1 -> 0, both free, cost 1 each
    cyc.add_edge(0, 2, 5, 0)
    cyc.add_edge(1, 2, 1, 0)
    assert rcsp_dp(cyc, 0, 2, 0)[0] == 2, "0-1-2 = 2, the free cycle back to 0 never helps"
    assert rcsp_dijkstra(cyc, 0, 2, 0)[0] == 2
    assert rcsp_brute_force(cyc, 0, 2, 0)[0] == 2
    assert rcsp_dp(cyc, 0, 2, 0)[1] < 50, "the closure converges quickly on the tiny graph"

    # --- Boundary cases -----------------------------------------------------------------
    for budget in (0, 1, 5):
        assert rcsp_dp(graph, 2, 2, budget)[0] == 0, "source == target costs nothing"
        assert rcsp_dijkstra(graph, 2, 2, budget)[0] == 0
        assert rcsp_brute_force(graph, 2, 2, budget)[0] == 0
    assert rcsp_dp(graph, 3, 0, 9)[0] == INF, "no route back from the sink"
    assert rcsp_dijkstra(graph, 3, 0, 9)[0] == INF
    assert rcsp_brute_force(graph, 3, 0, 9)[0] == INF
    assert rcsp_dp(graph, 0, 3, 0)[0] == INF, "budget 0 cannot pay for either route"
    assert rcsp_dp(graph, 0, 3, 1)[0] == INF, "no route fits in budget 1"
    assert rcsp_brute_force(graph, 0, 3, 1)[0] == INF

    # --- LCG instances: three methods must agree ---------------------------------------
    seed = 987654321
    checked = 0
    enumerated_total = 0
    relaxations_total = 0
    for _ in range(40):
        gen, seed = random_graph(5, seed, arcs=10)
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                previous = None
                for budget in range(7):
                    dp_best, dp_relax, _ = rcsp_dp(gen, source, target, budget)
                    dij_best, _ = rcsp_dijkstra(gen, source, target, budget)
                    bf_best, enumerated = rcsp_brute_force(gen, source, target, budget)
                    assert dp_best == dij_best == bf_best, \
                        "DP, expanded Dijkstra and brute force must agree"
                    checked += 1
                    enumerated_total += enumerated
                    relaxations_total += dp_relax
                    if previous is not None:
                        assert dp_best <= previous, "a larger budget cannot hurt"
                    previous = dp_best
        # A generous budget equals the resource-free optimum.
        total_resource = sum(r for (_, _, _, r) in gen.arcs())
        for source in range(5):
            for target in range(5):
                if source == target:
                    continue
                assert rcsp_dp(gen, source, target, total_resource)[0] == \
                    dijkstra_cost_only(gen, source, target), \
                    "with an unlimited budget the resource constraint is vacuous"
    assert checked > 1000, "thousands of (source, target, budget) triples agreed"
    assert enumerated_total > 0 and relaxations_total > 0

    # An instance where the layered DP is dramatically cheaper than enumeration: the complete
    # digraph on six vertices with pseudo-random costs and consumptions.
    dense = ResourceGraph(6)
    for u in range(6):
        for v in range(6):
            if u != v:
                dense.add_edge(u, v, (u + v) % 7, (u * v) % 3)
    dp_best, dp_relax, _ = rcsp_dp(dense, 0, 5, 6)
    bf_best, enumerated = rcsp_brute_force(dense, 0, 5, 6)
    dij_best, pops = rcsp_dijkstra(dense, 0, 5, 6)
    assert dp_best == bf_best == dij_best, "all three solvers agree on the dense instance"
    assert enumerated > dp_relax, "the DP relaxes fewer states than enumeration visits"
    assert dp_relax <= (6 + 1) * (dense.arc_count * (6 + 1)), "documented DP complexity bound"
    assert pops <= 7 * 6, "Dijkstra pops at most one entry per expanded state"

    # --- Determinism and repeatability --------------------------------------------------
    gen, _ = random_graph(5, 555, arcs=10)
    first = [rcsp_dp(gen, 0, 4, b)[0] for b in range(7)]
    second = [rcsp_dp(gen, 0, 4, b)[0] for b in range(7)]
    assert first == second, "layer DP is deterministic"
    assert first == [rcsp_dijkstra(gen, 0, 4, b)[0] for b in range(7)]


if __name__ == "__main__":
    main()
    print("PASS")
