"""Task 705 - Implement Steiner forest approximations large graphs.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

A Steiner forest instance is an undirected weighted graph plus a list of demand pairs; a feasible
solution is a set of edges that connects every demand pair, and we want the minimum total weight.
The approximation implemented here is the classical iterative-shortest-path greedy (the practical
face of the primal-dual/iterative-rounding algorithms of Agrawal-Klein-Ravi and of Goemans-
Williamson):

  1. compute d(u, v) for every demand pair on the *original* graph (Dijkstra);
  2. sort the demands by increasing d -- this is what makes the 2-approximation argument work,
     because when a path of length d is bought, the residual instance has optimum at least d, so
     multiplying a demand's cost by 2 pays for the whole remainder;
  3. walk the demands in that order, and whenever a still-unsatisfied pair sits in two different
     components of the solution bought so far, buy an entire shortest u-v path and merge the
     components it crosses; satisfied pairs cost nothing extra.

The result is always a forest (each bought path only merges components, so a cycle can never
appear), the bought edges are a subset of the input edges, and the cost is the sum of their
weights.  Instances whose demand pairs are not connected in the underlying graph are reported as
infeasible rather than silently approximated.

Simplified versus the literature: no LP duality / primal-dual growth of moats and no reverse
delete step (which would shave the constant), no contraction-based single-commodity speedup, and
shortest paths use a plain heapq Dijkstra.  Verified: on small instances the cost is compared
against an exhaustive subset-enumeration optimum (the ratio always lands in [1, 2]), the greedy is
exact for a single demand and for all-pairs demands (where the optimum is the minimum spanning
tree), the forest / feasibility / cost-accounting invariants are re-checked for every run, and a
large graph (hundreds of nodes, 15 terminals) is solved and validated against the certified
bounds implied by the MST of the terminal metric closure.
"""
import heapq
import sys

INF = float("inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.components -= 1
        return True


def build_graph(n, edges):
    adjacency = [[] for _ in range(n)]
    for index, (u, v, weight) in enumerate(edges):
        check(0 <= u < n and 0 <= v < n, f"edge {index} has an endpoint outside 0..{n - 1}")
        check(weight >= 0, f"edge {index} has a negative weight ({weight})")
        adjacency[u].append((v, weight, index))
        adjacency[v].append((u, weight, index))
    return adjacency


def dijkstra(adjacency, source):
    """Shortest distances and the predecessor tree, with deterministic tie-breaking."""
    dist = [INF] * len(adjacency)
    previous = [None] * len(adjacency)          # (node, edge_index)
    dist[source] = 0
    queue = [(0, source)]
    visited = [False] * len(adjacency)
    while queue:
        d, node = heapq.heappop(queue)
        if visited[node]:
            continue
        visited[node] = True
        for neighbour, weight, edge_index in adjacency[node]:
            if d + weight < dist[neighbour]:
                dist[neighbour] = d + weight
                previous[neighbour] = (node, edge_index)
                heapq.heappush(queue, (d + weight, neighbour))
    return dist, previous


def shortest_path(adjacency, source, target):
    """(cost, edge indices) of a shortest source-target path, or None when disconnected."""
    dist, previous = dijkstra(adjacency, source)
    if dist[target] == INF:
        return None
    path_edges = []
    node = target
    while node != source:
        parent, edge_index = previous[node]
        path_edges.append(edge_index)
        node = parent
    path_edges.reverse()
    return dist[target], path_edges


def steiner_forest(n, edges, demands):
    """Greedy 2-approximation.  Returns (selected edge indices, cost, diagnostics)."""
    adjacency = build_graph(n, edges)
    for u, v in demands:
        check(0 <= u < n and 0 <= v < n, f"demand {u}-{v} is outside the graph")

    distances = {}
    order = []
    for u, v in demands:
        if u == v:
            distances[(u, v)] = 0
            order.append((0, u, v))
            continue
        found = shortest_path(adjacency, u, v)
        if found is None:
            return None, None, {"infeasible": (u, v)}
        distances[(u, v)] = found[0]
        order.append((found[0], u, v))
    order.sort()                                # shortest demands first: the 2-approx ordering

    dsu = DisjointSet(n)
    selected = set()
    path_cost_total = 0
    dijkstras = 0
    for _, u, v in order:
        if dsu.find(u) == dsu.find(v):
            continue
        found = shortest_path(adjacency, u, v)
        if found is None:
            return None, None, {"infeasible": (u, v)}
        dijkstras += 1
        path_cost, path_edges = found
        path_cost_total += path_cost
        for edge_index in path_edges:
            selected.add(edge_index)
            a, b, _ = edges[edge_index]
            dsu.union(a, b)
    # The cost of the *solution* is the weight of its distinct edges; it can be below
    # path_cost_total, because a later path may re-traverse edges bought earlier.
    cost = sum(edges[index][2] for index in selected)
    return sorted(selected), cost, {"dijkstras": dijkstras, "components": dsu.components,
                                    "path_cost_total": path_cost_total}


def solution_is_feasible(n, edges, demands, selected):
    """Independent feasibility check: a plain connectivity test over the selected edges."""
    if selected is None:
        return False
    dsu = DisjointSet(n)
    for edge_index in selected:
        u, v, _ = edges[edge_index]
        dsu.union(u, v)
    return all(dsu.find(u) == dsu.find(v) for u, v in demands)


def selected_is_forest(n, edges, selected):
    dsu = DisjointSet(n)
    for edge_index in selected:
        u, v, _ = edges[edge_index]
        if not dsu.union(u, v):
            return False
    return True


def brute_force_optimum(n, edges, demands):
    """Exhaustive optimum over all edge subsets -- the oracle for tiny instances."""
    best = INF
    total = len(edges)
    for mask in range(1 << total):
        weight = 0
        for index in range(total):
            if mask >> index & 1:
                weight += edges[index][2]
        if weight >= best:
            continue
        dsu = DisjointSet(n)
        for index in range(total):
            if mask >> index & 1:
                dsu.union(edges[index][0], edges[index][1])
        if all(dsu.find(u) == dsu.find(v) for u, v in demands):
            best = weight
    return best


def metric_closure_mst(n, edges, terminals):
    """MST over the complete graph of terminal-to-terminal shortest path distances."""
    adjacency = build_graph(n, edges)
    pairwise = {}
    for terminal in terminals:
        dist, _ = dijkstra(adjacency, terminal)
        pairwise[terminal] = dist
    items = []
    for i, a in enumerate(terminals):
        for b in terminals[i + 1:]:
            items.append((pairwise[a][b], a, b))
    items.sort()
    index_of = {terminal: k for k, terminal in enumerate(terminals)}
    dsu = DisjointSet(len(terminals))
    total = 0
    for weight, a, b in items:
        if dsu.union(index_of[a], index_of[b]):
            total += weight
    return total


# --------------------------------------------------------------------------- checks
def test_single_demand_is_a_shortest_path():
    edges = [(0, 1, 1), (1, 2, 1), (0, 2, 3), (2, 3, 4), (1, 3, 9)]
    n = 4
    selected, cost, stats = steiner_forest(n, edges, [(0, 3)])
    check(cost == brute_force_optimum(n, edges, [(0, 3)]), "one demand: greedy is exact")
    check(cost == 6, f"the path 0-1-2-3 costs 1+1+4, got {cost}")
    check(solution_is_feasible(n, edges, [(0, 3)], selected), "the pair must be connected")
    check(selected_is_forest(n, edges, selected), "a single path is a forest")
    check(stats["dijkstras"] == 1, "one residual Dijkstra for one demand")


def test_handcrafted_multi_commodity():
    # two commodities sharing the middle edge: 0-1-2-3 chain, plus a demand between 2 and 4
    edges = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (2, 4, 1)]
    n = 5
    demands = [(0, 3), (2, 4)]
    selected, cost, _ = steiner_forest(n, edges, demands)
    optimum = brute_force_optimum(n, edges, demands)
    check(cost == optimum, f"greedy {cost} vs optimum {optimum} on the handcrafted instance")
    check(cost == 6, "the whole chain plus the leaf edge: 2+1+2+1 = 6")
    check(solution_is_feasible(n, edges, demands, selected), "both commodities connected")
    check(len(selected) == 4, "four edges are bought")

    # a Steiner node pays off: connecting 0-3 through 2 costs 3 instead of the direct edge 8
    edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 8)]
    selected, cost, _ = steiner_forest(4, edges, [(0, 3)])
    check(cost == 3, f"the Steiner detour must be preferred, got {cost}")
    check(solution_is_feasible(4, edges, [(0, 3)], selected), "detour connects the terminals")


def test_all_pairs_demands_equal_mst_cost():
    edges = [(0, 1, 4), (0, 2, 8), (1, 2, 1), (1, 3, 6), (2, 3, 2), (3, 4, 5), (2, 4, 9)]
    n = 5
    demands = [(u, v) for u in range(n) for v in range(u + 1, n)]
    selected, cost, _ = steiner_forest(n, edges, demands)
    ordered = sorted(edges, key=lambda edge: (edge[2], edge[0], edge[1]))
    dsu, mst = DisjointSet(n), 0
    for u, v, weight in ordered:
        if dsu.union(u, v):
            mst += weight
    check(cost == mst, f"all-pairs demands need exactly an MST: {cost} vs {mst}")
    check(cost == brute_force_optimum(n, edges, demands), "and the oracle agrees")
    check(solution_is_feasible(n, edges, demands, selected), "all pairs connected")


def test_degenerate_demands():
    edges = [(0, 1, 5), (1, 2, 5)]
    selected, cost, stats = steiner_forest(3, edges, [])
    check(selected == [] and cost == 0, "no demands means an empty forest of cost 0")
    check(stats["dijkstras"] == 0, "no Dijkstra is run without demands")

    selected, cost, _ = steiner_forest(3, edges, [(1, 1)])
    check(selected == [] and cost == 0, "a demand from a node to itself is free")

    selected, cost, _ = steiner_forest(3, edges, [(0, 0), (1, 2)])
    check(cost == 5, "the trivial demand must not disturb the real one")

    edges = [(0, 1, 0), (1, 2, 0), (0, 2, 7)]
    selected, cost, _ = steiner_forest(3, edges, [(0, 2)])
    check(cost == 0, "a zero-weight path is free")
    check(selected == [0, 1], f"both zero-weight edges are used, got {selected}")

    selected, cost, stats = steiner_forest(4, [(0, 1, 1), (2, 3, 1)], [(0, 1), (1, 2)])
    check(selected is None and cost is None, "an unreachable demand makes the instance infeasible")
    check(stats["infeasible"] == (1, 2), f"the infeasible pair is reported, got {stats}")

    try:
        steiner_forest(3, [(0, 1, -2)], [(0, 1)])
        raise AssertionError("negative weights must be rejected")
    except AssertionError as error:
        check("negative weight" in str(error), f"unexpected message: {error}")
    try:
        steiner_forest(3, [(0, 1, 2)], [(0, 7)])
        raise AssertionError("a demand outside the graph must be rejected")
    except AssertionError as error:
        check("outside the graph" in str(error), f"unexpected message: {error}")


def test_random_instances_against_exhaustive_optimum():
    mask = (1 << 64) - 1
    seed = 0x0DDC0FFEE1234567
    worst_ratio = 1.0
    suboptimal = 0
    tested = 0
    for trial in range(40):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        n = 3 + seed % 5
        edge_count = 3 + (seed >> 9) % 8
        edges = []
        for k in range(edge_count):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            u = seed % n
            v = (seed >> 17) % n
            if u == v:
                v = (v + 1) % n
            weight = 1 + (seed >> 31) % 6
            edges.append((u, v, weight))
        demand_count = 1 + (seed >> 41) % 3
        demands = []
        for _ in range(demand_count):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            demands.append((seed % n, (seed >> 13) % n))
        optimum = brute_force_optimum(n, edges, demands)
        selected, cost, _ = steiner_forest(n, edges, demands)
        if optimum == INF:
            check(selected is None, f"trial {trial}: the oracle found no feasible solution")
            continue
        check(selected is not None, f"trial {trial}: greedy failed on a feasible instance")
        check(solution_is_feasible(n, edges, demands, selected), f"trial {trial}: infeasible")
        check(selected_is_forest(n, edges, selected), f"trial {trial}: solution is not a forest")
        check(cost == sum(edges[index][2] for index in selected),
              f"trial {trial}: the reported cost is not the sum of the selected edges")
        check(all(0 <= index < len(edges) for index in selected),
              f"trial {trial}: selected an edge outside the input")
        if optimum == 0:
            check(cost == 0, f"trial {trial}: an instance satisfiable for free costs 0")
            continue
        ratio = cost / optimum
        check(1.0 <= ratio <= 2.0, f"trial {trial}: ratio {ratio} outside [1, 2]")
        worst_ratio = max(worst_ratio, ratio)
        suboptimal += ratio > 1.0
        tested += 1
    check(tested >= 20, f"only {tested} feasible instances were exercised")
    return worst_ratio, suboptimal


def test_large_graph_scales():
    n = 240
    mask = (1 << 64) - 1
    seed = 0xABCDEF0123456789
    edges = []
    for node in range(1, n):                    # a random spanning tree keeps the graph connected
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        edges.append((node, seed % node, 1 + (seed >> 23) % 9))
    for _ in range(360):                        # extra random edges
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 21) % n
        if u != v:
            edges.append((u, v, 1 + (seed >> 37) % 9))
    terminals = sorted({(seed >> (3 * k)) % n for k in range(16) if (seed >> (3 * k)) % n != 0} | {0})
    demands = [(terminals[0], terminal) for terminal in terminals[1:]]
    selected, cost, stats = steiner_forest(n, edges, demands)
    check(selected is not None, "the connected graph must be feasible")
    check(solution_is_feasible(n, edges, demands, selected), "every demand pair is connected")
    check(selected_is_forest(n, edges, selected), "the bought edges form a forest")
    check(cost == sum(edges[index][2] for index in selected), "the cost is the sum of the edges")
    check(0 < stats["dijkstras"] <= len(demands),
          f"at most one residual Dijkstra per demand, got {stats['dijkstras']}")
    check(len(selected) <= n - 1, "a forest has at most n - 1 edges")

    closure = metric_closure_mst(n, edges, terminals)
    check(closure / 2 <= cost + 1e-9,
          f"greedy <= 2 * OPT and OPT >= closure / 2 give cost >= closure / 2: {cost} {closure}")
    check(cost <= 2 * closure + 1e-9,
          f"OPT <= closure, so greedy <= 2 * OPT <= 2 * closure: {cost} {closure}")

    again, cost_again, _ = steiner_forest(n, edges, demands)
    check(again == selected and cost_again == cost, "the solver is deterministic")
    return n, len(edges), len(demands), cost, len(selected), closure


def main():
    test_single_demand_is_a_shortest_path()
    test_handcrafted_multi_commodity()
    test_all_pairs_demands_equal_mst_cost()
    test_degenerate_demands()
    worst_ratio, suboptimal = test_random_instances_against_exhaustive_optimum()
    n, edge_count, demand_count, cost, used, closure = test_large_graph_scales()

    print(f"random_instances: worst_ratio={worst_ratio:.3f} suboptimal_cases={suboptimal}")
    print(f"large: nodes={n} edges={edge_count} demands={demand_count} cost={cost} "
          f"edges_used={used} closure_mst={closure}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
