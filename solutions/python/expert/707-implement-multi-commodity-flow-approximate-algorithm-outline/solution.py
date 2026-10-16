"""Task 707 - Implement multi-commodity flow approximate algorithm outline.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

Multi-commodity flow asks for one flow per commodity, all sharing the same edge capacities: route
lambda * d_i units of commodity i simultaneously, and maximise the concurrency lambda.  Unlike the
single-commodity case there is no integrality and no min-cut duality, so this file implements the
outline of the Garg-Ko"nemann / Young multiplicative-weights scheme, which is the standard way to
get a (1 - eps)-approximation in polynomial time:

  * give every edge a price ell_e, initially inversely proportional to its capacity;
  * per phase, route the *full* demand of every commodity along the cheapest path under the
    current prices (one Dijkstra per commodity) and measure the congestion
    max_e (load_e / c_e);
  * the phase's flows scaled by 1/congestion are feasible, which certifies lambda = 1/congestion;
  * then raise the price of every congested edge multiplicatively,
    ell_e <- ell_e * (1 + eps * (load_e / c_e) / congestion), so the next phase routes around the
    bottleneck -- this is the multiplicative update that makes the scheme converge to a
    (1 - eps)-approximation after O(m log m / eps^2) phases (here capped by max_phases);
  * the best (maximum) certified lambda over all phases is returned, together with the
    commodity paths that realise it.

This is an *outline*: the phase count is a fixed cap rather than the theoretical stopping rule,
prices are plain floats, and no rounding of the fractional solution to an integral one is done
(that last step is what the rounding-based algorithms add).  What is verified is everything the
outline promises: the returned lambda is certified by an independent load computation (every edge
respects its capacity and every commodity carries exactly lambda * d_i), it never exceeds the
single-commodity upper bounds, and on tree instances -- where each commodity has exactly one
simple path and the exact optimum is therefore computable in closed form -- it stays within a
measured factor of that optimum.
"""
import heapq
import sys

INF = float("inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------- graph helpers
def build_adjacency(n, edges):
    adjacency = [[] for _ in range(n)]
    for index, (u, v, capacity) in enumerate(edges):
        check(0 <= u < n and 0 <= v < n, f"edge {index} has an endpoint outside 0..{n - 1}")
        check(capacity > 0, f"edge {index} must have a positive capacity")
        adjacency[u].append((v, index))
    return adjacency


def cheapest_path(n, adjacency, prices, source, target):
    """Dijkstra under the current edge prices; returns the list of edge indices or None."""
    if source == target:
        return []
    dist = [INF] * n
    previous = [None] * n
    dist[source] = 0.0
    queue = [(0.0, source)]
    done = [False] * n
    while queue:
        d, node = heapq.heappop(queue)
        if done[node]:
            continue
        done[node] = True
        for neighbour, edge_index in adjacency[node]:
            candidate = d + prices[edge_index]
            if candidate < dist[neighbour]:
                dist[neighbour] = candidate
                previous[neighbour] = (node, edge_index)
                heapq.heappush(queue, (candidate, neighbour))
    if dist[target] == INF:
        return None
    path = []
    node = target
    while node != source:
        parent, edge_index = previous[node]
        path.append(edge_index)
        node = parent
    path.reverse()
    return path


# --------------------------------------------------------------------------- the algorithm
def multicommodity_flow(n, edges, commodities, epsilon=0.05, max_phases=300):
    """Greedy multiplicative-weights outline.

    Returns (lambda, per-commodity edge paths, stats).  lambda is 0.0 when some commodity is
    disconnected (its demand can never be routed) and infinite when nothing has to be routed.
    """
    check(0 < epsilon <= 1, "epsilon must lie in (0, 1]")
    for index, (source, target, demand) in enumerate(commodities):
        check(0 <= source < n and 0 <= target < n, f"commodity {index} is outside the graph")
        check(demand >= 0, f"commodity {index} has a negative demand")
    adjacency = build_adjacency(n, edges)
    m = len(edges)

    active = [(s, t, d) for s, t, d in commodities if d > 0 and s != t]
    if not active:
        return INF, [[] for _ in commodities], {"phases": 0, "reason": "nothing to route"}
    for source, target, _ in active:
        if cheapest_path(n, adjacency, [1.0 / edges[i][2] for i in range(m)], source, target) is None:
            return 0.0, [None for _ in commodities], {"phases": 0, "reason": "disconnected"}

    # delta keeps the initial prices small compared with the total capacity, as in the paper
    delta = 1.0 / (m * (1.0 + epsilon) ** (1.0 / epsilon)) if m else 1.0
    prices = [delta / edges[index][2] for index in range(m)]

    best_lambda = 0.0
    best_paths = None
    best_congestion = INF
    phases_used = 0
    for phase in range(max_phases):
        phases_used = phase + 1
        load = [0.0] * m
        paths = {}
        for commodity_index, (source, target, demand) in enumerate(commodities):
            if demand <= 0 or source == target:
                paths[commodity_index] = []
                continue
            path = cheapest_path(n, adjacency, prices, source, target)
            paths[commodity_index] = path
            for edge_index in path:
                load[edge_index] += demand
        congestion = 0.0
        for edge_index in range(m):
            ratio = load[edge_index] / edges[edge_index][2]
            if ratio > congestion:
                congestion = ratio
        if congestion <= 0.0:
            return INF, [[] for _ in commodities], {"phases": phases_used, "reason": "no load"}
        lam = 1.0 / congestion
        if lam > best_lambda + 1e-15:
            best_lambda = lam
            best_paths = [paths[i] for i in range(len(commodities))]
            best_congestion = congestion
        if congestion >= 1.0:
            break
        for edge_index in range(m):
            if load[edge_index] > 0:
                prices[edge_index] *= (1.0 + epsilon * (load[edge_index]
                                                        / edges[edge_index][2]) / congestion)
    stats = {"phases": phases_used, "congestion_at_best": best_congestion,
             "total_price": sum(prices)}
    return best_lambda, best_paths, stats


def realised_loads(edges, commodities, paths, lam):
    """Independent accounting: per-edge load and per-commodity delivered rate."""
    load = [0.0] * len(edges)
    delivered = []
    for (source, target, demand), path in zip(commodities, paths):
        delivered.append(lam * demand)
        if path is None:
            return None, None
        for edge_index in path:
            load[edge_index] += lam * demand
    return load, delivered


# --------------------------------------------------------------------------- oracles
def single_commodity_upper_bound(n, edges, source, target):
    """Exact max flow of one commodity, by a small independent Ford-Fulkerson with BFS."""
    adjacency = [[] for _ in range(n)]
    capacity = {}
    for index, (u, v, c) in enumerate(edges):
        adjacency[u].append((v, index))
        adjacency[v].append((u, index))
        capacity[(u, v, index)] = c
        capacity[(v, u, index)] = 0
    flow = 0
    while True:
        previous = {source: None}
        queue = [source]
        while queue and target not in previous:
            node = queue.pop(0)
            for neighbour, index in adjacency[node]:
                if neighbour not in previous and capacity[(node, neighbour, index)] > 0:
                    previous[neighbour] = (node, index)
                    queue.append(neighbour)
        if target not in previous:
            return flow
        bottleneck = INF
        node = target
        while previous[node] is not None:
            parent, index = previous[node]
            bottleneck = min(bottleneck, capacity[(parent, node, index)])
            node = parent
        node = target
        while previous[node] is not None:
            parent, index = previous[node]
            capacity[(parent, node, index)] -= bottleneck
            capacity[(node, parent, index)] += bottleneck
            node = parent
        flow += bottleneck


def tree_exact_lambda(n, tree_edges, commodities):
    """On a tree every commodity has a unique path, so the optimum is min_e c_e / (demand on e)."""
    adjacency = [[] for _ in range(n)]
    for index, (u, v, capacity) in enumerate(tree_edges):
        adjacency[u].append((v, index))
        adjacency[v].append((u, index))
    used = [0] * len(tree_edges)
    for source, target, demand in commodities:
        if demand == 0:
            continue
        previous = {source: None}
        queue = [source]
        while queue and target not in previous:
            node = queue.pop(0)
            for neighbour, index in adjacency[node]:
                if neighbour not in previous:
                    previous[neighbour] = (node, index)
                    queue.append(neighbour)
        check(target in previous, "the tree must connect every commodity")
        node = target
        while previous[node] is not None:
            parent, index = previous[node]
            used[index] += demand
            node = parent
    best = INF
    for index, (_, _, capacity) in enumerate(tree_edges):
        if used[index] > 0:
            best = min(best, capacity / used[index])
    return best


def one_shot_lower_bound(n, edges, commodities):
    """Route everything once on the cheapest path: a feasible concurrency, hence a lower bound."""
    adjacency = build_adjacency(n, edges)
    prices = [1.0 / capacity for _, _, capacity in edges]
    load = [0.0] * len(edges)
    for source, target, demand in commodities:
        if demand <= 0 or source == target:
            continue
        path = cheapest_path(n, adjacency, prices, source, target)
        if path is None:
            return 0.0
        for edge_index in path:
            load[edge_index] += demand
    best = INF
    for index, (_, _, capacity) in enumerate(edges):
        if load[index] > 0:
            best = min(best, capacity / load[index])
    return best if best < INF else INF


# --------------------------------------------------------------------------- checks
def test_shared_bottleneck():
    # both commodities must cross edge 1->3 of capacity 2, with unit demands
    edges = [(0, 1, 5), (1, 3, 2), (2, 1, 5)]
    commodities = [(0, 3, 1), (2, 3, 1)]
    lam, paths, stats = multicommodity_flow(4, edges, commodities)
    check(abs(lam - 1.0) < 1e-9, f"the shared bottleneck allows concurrency 1, got {lam}")
    loads, delivered = realised_loads(edges, commodities, paths, lam)
    check(all(loads[i] <= edges[i][2] + 1e-9 for i in range(len(edges))),
          f"every edge must respect its capacity: {loads}")
    check(abs(loads[1] - 2.0) < 1e-9, f"the bottleneck edge is exactly saturated, got {loads[1]}")
    check(all(abs(value - lam) < 1e-9 for value in delivered),
          "each commodity carries exactly lambda * demand")
    check(paths[0] == [0, 1] and paths[1] == [2, 1], f"both routes cross the bottleneck: {paths}")
    check(stats["phases"] >= 1, "at least one phase runs")
    return lam, paths, stats


def test_tree_instances_are_near_optimal():
    # a path graph where the two commodities overlap on the middle edge
    tree = [(0, 1, 4), (1, 2, 3), (2, 3, 6), (3, 4, 5)]
    commodities = [(0, 4, 1), (1, 3, 1)]
    exact = tree_exact_lambda(5, tree, commodities)
    check(abs(exact - 1.5) < 1e-9, f"the shared edge 1-2 gives 3 / 2 = 1.5, got {exact}")
    lam, paths, _ = multicommodity_flow(5, tree, commodities)
    check(lam <= exact + 1e-9, f"lambda {lam} cannot beat the exact optimum {exact}")
    check(lam >= 0.9 * exact, f"lambda {lam} should be close to the optimum {exact}")

    # a star: three leaves talking to the hub, each edge loaded by its own commodity
    star = [(1, 0, 6), (2, 0, 3), (3, 0, 9)]
    commodities = [(1, 0, 2), (2, 0, 1), (3, 0, 3)]
    exact = tree_exact_lambda(4, star, commodities)
    check(abs(exact - 3.0) < 1e-9, f"the tight leaf needs 3/1, got {exact}")
    lam, paths, _ = multicommodity_flow(4, star, commodities)
    check(lam <= exact + 1e-9, "never better than the tree optimum")
    check(lam >= 0.9 * exact, f"star instance: {lam} vs optimum {exact}")

    # a bigger random tree, checked against the closed-form optimum
    mask = (1 << 64) - 1
    seed = 0x0F1E2D3C4B5A6978
    worst = 1.0
    for trial in range(12):
        n = 10
        tree_edges = []
        ancestors = [[0]]
        for node in range(1, n):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            parent = seed % node
            tree_edges.append((parent, node, 5 + (seed >> 17) % 20))
            ancestors.append(ancestors[parent] + [node])
        commodities = []
        for _ in range(4):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            target = 1 + (seed >> 13) % (n - 1)
            source = ancestors[target][(seed >> 29) % len(ancestors[target])]
            commodities.append((source, target, 1 + (seed >> 37) % 3))
        exact = tree_exact_lambda(n, tree_edges, commodities)
        if exact == INF or exact == 0:
            continue
        lam, paths, _ = multicommodity_flow(n, tree_edges, commodities)
        check(lam <= exact + 1e-9, f"trial {trial}: {lam} above the optimum {exact}")
        ratio = lam / exact
        check(ratio >= 0.85, f"trial {trial}: ratio {ratio} far below the tree optimum")
        loads, _ = realised_loads(tree_edges, commodities, paths, lam)
        check(all(loads[i] <= tree_edges[i][2] + 1e-9 for i in range(len(tree_edges))),
              f"trial {trial}: capacity violated")
        worst = min(worst, ratio)
    return worst


def test_upper_and_lower_bounds():
    mask = (1 << 64) - 1
    seed = 0xABCDEF0102030405
    comparisons = 0
    for trial in range(20):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        n = 5
        edges = []
        for _ in range(8):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            u = seed % n
            v = (seed >> 17) % n
            if u != v:
                edges.append((u, v, 1 + (seed >> 33) % 6))
        if not edges:
            continue
        commodities = []
        for _ in range(2):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            commodities.append((seed % n, (seed >> 11) % n, 1 + (seed >> 23) % 3))
        lam, paths, _ = multicommodity_flow(n, edges, commodities)
        if lam == 0.0:
            check(all(path is None for path in paths), "an unroutable instance reports no routes")
            continue
        check(lam < INF, f"trial {trial}: a routable instance must have a finite lambda")
        upper = INF
        for source, target, demand in commodities:
            if demand > 0 and source != target:
                upper = min(upper, single_commodity_upper_bound(n, edges, source, target) / demand)
        check(lam <= upper + 1e-9,
              f"trial {trial}: lambda {lam} exceeds the single-commodity bound {upper}")
        lower = one_shot_lower_bound(n, edges, commodities)
        if lower < INF and lower > 0:
            check(lam >= lower - 1e-9,
                  f"trial {trial}: lambda {lam} must not lose to the one-shot routing {lower}")
            comparisons += 1
        loads, delivered = realised_loads(edges, commodities, paths, lam)
        check(all(loads[i] <= edges[i][2] + 1e-9 for i in range(len(edges))),
              f"trial {trial}: an edge load exceeds its capacity: {loads}")
        for value, (_, _, demand) in zip(delivered, commodities):
            check(abs(value - lam * demand) < 1e-9,
                  f"trial {trial}: delivered rates must be lambda * demand")
    check(comparisons >= 5, f"only {comparisons} lower-bound comparisons happened")
    return comparisons


def test_boundaries():
    edges = [(0, 1, 4), (1, 2, 4)]
    lam, paths, stats = multicommodity_flow(3, edges, [])
    check(lam == INF, "with no commodities there is nothing to congest")
    check(paths == [] and stats["reason"] == "nothing to route", "and nothing is routed")

    lam, paths, _ = multicommodity_flow(3, edges, [(0, 2, 0)])
    check(lam == INF, "a zero demand never congests anything")

    lam, paths, _ = multicommodity_flow(3, edges, [(1, 1, 5)])
    check(lam == INF, "a commodity from a node to itself is already satisfied")

    lam, paths, _ = multicommodity_flow(4, [(0, 1, 2), (2, 3, 2)], [(0, 3, 1)])
    check(lam == 0.0, "a disconnected commodity makes the instance unroutable")

    lam, paths, _ = multicommodity_flow(3, edges, [(0, 2, 3)])
    check(abs(lam - 4.0 / 3.0) < 1e-9, f"a single commodity scales to max flow / demand, got {lam}")

    lam, paths, _ = multicommodity_flow(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)], [(0, 4, 1)])
    check(abs(lam - 1.0) < 1e-9, f"a chain of unit edges allows exactly 1, got {lam}")

    lam, paths, _ = multicommodity_flow(3, [(0, 1, 2), (1, 2, 2)], [(0, 2, 1), (0, 2, 1)])
    check(abs(lam - 1.0) < 1e-9, f"two unit commodities share a pipe of capacity 2: got {lam}")
    loads, _ = realised_loads([(0, 1, 2), (1, 2, 2)], [(0, 2, 1), (0, 2, 1)], paths, lam)
    check(all(abs(value - 2.0) < 1e-9 for value in loads), f"both edges carry 2, got {loads}")

    try:
        multicommodity_flow(3, edges, [(0, 2, 1)], epsilon=0.0)
        raise AssertionError("epsilon = 0 must be rejected")
    except AssertionError as error:
        check("epsilon" in str(error), f"unexpected message: {error}")
    try:
        multicommodity_flow(3, [(0, 1, 0)], [(0, 1, 1)])
        raise AssertionError("a zero-capacity edge must be rejected")
    except AssertionError as error:
        check("positive capacity" in str(error), f"unexpected message: {error}")

    first = multicommodity_flow(4, [(0, 1, 3), (1, 3, 3), (0, 2, 3), (2, 3, 3)],
                                [(0, 3, 1), (0, 3, 2)])
    second = multicommodity_flow(4, [(0, 1, 3), (1, 3, 3), (0, 2, 3), (2, 3, 3)],
                                 [(0, 3, 1), (0, 3, 2)])
    check(first[0] == second[0] and first[1] == second[1], "the solver must be deterministic")


def test_bigger_instance():
    n = 40
    mask = (1 << 64) - 1
    seed = 0x5EEDF00D12345678
    edges = []
    for node in range(1, n):                       # a chain guarantees connectivity
        edges.append((node - 1, node, 6 + (node % 5)))
    for _ in range(120):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 17) % n
        if u != v:
            edges.append((u, v, 2 + (seed >> 35) % 9))
    commodities = [(0, n - 1, 3), (1, n - 2, 2), (5, n - 5, 1)]
    lam, paths, stats = multicommodity_flow(n, edges, commodities)
    check(lam > 0.0 and lam < INF, f"the chain instance is routable, got {lam}")
    loads, _ = realised_loads(edges, commodities, paths, lam)
    check(all(loads[i] <= edges[i][2] + 1e-9 for i in range(len(edges))),
          "the large instance must respect every capacity")
    upper = min(single_commodity_upper_bound(n, edges, s, t) / d for s, t, d in commodities)
    check(lam <= upper + 1e-9, f"lambda {lam} above the single-commodity bound {upper}")
    check(stats["phases"] <= 300, "the phase cap is respected")
    return n, len(edges), lam, stats["phases"]


def main():
    lam, paths, stats = test_shared_bottleneck()
    worst_tree = test_tree_instances_are_near_optimal()
    comparisons = test_upper_and_lower_bounds()
    test_boundaries()
    n, edge_count, large_lambda, phases = test_bigger_instance()

    print(f"bottleneck_lambda={lam:.4f} phases={stats['phases']} "
          f"worst_tree_ratio={worst_tree:.4f} bound_comparisons={comparisons}")
    print(f"large: nodes={n} edges={edge_count} lambda={large_lambda:.4f} phases={phases}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
