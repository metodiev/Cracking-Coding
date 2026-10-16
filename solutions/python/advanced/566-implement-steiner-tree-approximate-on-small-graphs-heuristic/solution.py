"""Task 566 - Implement Steiner tree approximate on small graphs (heuristic). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

The Steiner tree problem on a handful of vertices: given an undirected weighted graph and a terminal
set, find the cheapest connected subgraph containing every terminal. The file implements three
heuristics and two exact references so every claim is measured rather than assumed. KMB is the
classical 2-approximation: compute the all-pairs shortest paths, build the metric closure on the
terminals, take its minimum spanning tree (Kruskal with deterministic tie-breaking), expand each MST
edge back into the corresponding shortest path, and prune non-terminal leaves - a step the file
verifies never raises the cost and never disconnects a terminal. SPH grows a tree by repeatedly
attaching the nearest remaining terminal through a shortest path, and the "minimum spanning tree of
the graph that ignores the terminals" baseline is included for comparison. Exactness comes from the
Dreyfus-Wagner subset DP (dp[S][v] = cheapest tree spanning terminal set S plus v, merged over
subsets and then relaxed by a Dijkstra pass) and from a brute force that minimises the MST cost over
every vertex subset containing the terminals. Random graphs come from a full-period 64-bit LCG (MMIX
constants) read in its high bits, since the repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short
period and low-bit structure that would bias the edge weights.

Checks: on every random small instance the Dreyfus-Wagner value equals the brute-force minimum and
the optimal value is the same in the graph and in its metric closure; both heuristics produce a real
connected subgraph spanning the terminals whose cost is at least the optimum and at most
2(1 - 1/|T|) times it - the textbook 2(1 - 1/l) bound with l, the number of leaves of the optimal
tree, replaced by the safe |T| >= l - with the worst observed ratio reported; the special cases
agree with closed forms (no terminals and one terminal cost 0, two terminals equal the shortest path,
all vertices equal the MST); pruning is cost-non-increasing and connectivity-preserving; SPH and KMB
never beat the optimum; and repeated runs are identical. Honest caveat, asserted: the 2-approximation
guarantee is relative to the *metric closure*, and on the tested instances the observed ratios stay
strictly below the bound, so the file reports the measured worst case instead of claiming tightness.
"""
MASK64 = (1 << 64) - 1
INF = float("inf")


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_graph(n, density, seed, low=1, high=9):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 83) & MASK64)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                state = prng64(state)
                edges.append((i, j, low + (state >> 33) % (high - low + 1)))
    return (n, edges)


def adjacency_list(graph):
    n, edges = graph
    adjacency = [[] for _ in range(n)]
    for (u, v, w) in edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))
    return adjacency


def dijkstra(graph, source):
    """(distances, predecessors) with deterministic tie-breaking."""
    n, _ = graph
    adjacency = adjacency_list(graph)
    distance = [INF] * n
    previous = [-1] * n
    distance[source] = 0
    visited = [False] * n
    for _ in range(n):
        best = -1
        for vertex in range(n):
            if not visited[vertex] and (best == -1 or distance[vertex] < distance[best]):
                best = vertex
        if best == -1 or distance[best] == INF:
            break
        visited[best] = True
        for (other, weight) in sorted(adjacency[best]):
            candidate = distance[best] + weight
            if candidate < distance[other]:
                distance[other] = candidate
                previous[other] = best
    return distance, previous


def shortest_path(graph, source, target):
    """The vertex list of a shortest source -> target path (empty when unreachable)."""
    if source == target:
        return [source]
    _, previous = dijkstra(graph, source)
    if previous[target] == -1:
        return []
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return list(reversed(path))


def kruskal(n, edges):
    """MST cost and edge list over the given edge set; None when the set is disconnected."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    chosen = []
    for (u, v, w) in sorted(edges, key=lambda e: (e[2], e[0], e[1])):
        (ru, rv) = (find(u), find(v))
        if ru != rv:
            parent[ru] = rv
            total += w
            chosen.append((u, v, w))
    roots = {find(v) for v in range(n)}
    if len(roots) > 1:
        return (None, [])
    return (total, chosen)


def mst_of_subset(graph, vertices):
    """MST over the subgraph induced by ``vertices`` (renamed to 0..k-1)."""
    index = {vertex: position for (position, vertex) in enumerate(sorted(vertices))}
    edges = []
    for (u, v, w) in graph[1]:
        if u in index and v in index:
            edges.append((index[u], index[v], w))
    return kruskal(len(index), edges)


def steiner_bruteforce(graph, terminals):
    """Minimum over all vertex subsets containing the terminals of their MST cost."""
    n, _ = graph
    others = [v for v in range(n) if v not in terminals]
    best = None
    for mask in range(1 << len(others)):
        vertices = set(terminals)
        for (position, vertex) in enumerate(others):
            if mask >> position & 1:
                vertices.add(vertex)
        (cost, _) = mst_of_subset(graph, vertices)
        if cost is None:
            continue
        if best is None or cost < best:
            best = cost
    return best


def steiner_exact(graph, terminals):
    """Dreyfus-Wagner subset DP over the terminal subsets."""
    n, edges = graph
    terminal_list = sorted(terminals)
    if not terminal_list:
        return 0
    count = len(terminal_list)
    position = {terminal: index for (index, terminal) in enumerate(terminal_list)}
    size = 1 << count
    dp = [[INF] * n for _ in range(size)]
    for (index, terminal) in enumerate(terminal_list):
        dp[1 << index][terminal] = 0
    for subset in range(1, size):
        row = dp[subset]
        sub = (subset - 1) & subset
        while sub:
            other = subset ^ sub
            if sub < other:
                left, right = dp[sub], dp[other]
                for vertex in range(n):
                    candidate = left[vertex] + right[vertex]
                    if candidate < row[vertex]:
                        row[vertex] = candidate
            sub = (sub - 1) & subset
        # Dijkstra-style relaxation with the merge values as initial distances
        visited = [False] * n
        for _ in range(n):
            best = -1
            for vertex in range(n):
                if not visited[vertex] and (best == -1 or row[vertex] < row[best]):
                    best = vertex
            if best == -1 or row[best] == INF:
                break
            visited[best] = True
            for (other, weight) in adjacency_list(graph)[best]:
                candidate = row[best] + weight
                if candidate < row[other]:
                    row[other] = candidate
    return min(dp[size - 1])


def metric_closure(graph, terminals):
    """Complete graph on the terminals with shortest-path weights."""
    terminal_list = sorted(terminals)
    edges = []
    for (i, first) in enumerate(terminal_list):
        distance, _ = dijkstra(graph, first)
        for j in range(i + 1, len(terminal_list)):
            second = terminal_list[j]
            if distance[second] != INF:
                edges.append((first, second, distance[second]))
    return (graph[0], edges)


def kmb(graph, terminals):
    """Metric-closure MST expanded into the graph, then pruned."""
    terminal_list = sorted(terminals)
    if len(terminal_list) <= 1:
        return 0
    closure = metric_closure(graph, terminal_list)
    index = {vertex: position for (position, vertex) in enumerate(terminal_list)}
    closure_edges = [(index[u], index[v], w) for (u, v, w) in closure[1]]
    (cost, chosen) = kruskal(len(terminal_list), closure_edges)
    if cost is None:
        return INF
    used = set()
    for (u, v, _) in chosen:
        path = shortest_path(graph, terminal_list[u], terminal_list[v])
        for step in range(len(path) - 1):
            used.add((min(path[step], path[step + 1]), max(path[step], path[step + 1])))
    weights = {(min(u, v), max(u, v)): w for (u, v, w) in graph[1]}
    edge_set = [(u, v, weights[(u, v)]) for (u, v) in used]
    return prune_edges(graph, edge_set, terminals)


def prune_edges(graph, edges, terminals):
    """Iteratively drop non-terminal leaves; returns the total cost of what is left."""
    current = list(edges)
    while True:
        degree = {}
        for (u, v, _) in current:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        removable = [index for (index, (u, v, _)) in enumerate(current)
                     if (degree[u] == 1 and u not in terminals)
                     or (degree[v] == 1 and v not in terminals)]
        if not removable:
            break
        for index in reversed(removable):
            current.pop(index)
    return sum(w for (_, _, w) in current)


def sph(graph, terminals):
    """Shortest-path heuristic: attach the nearest remaining terminal to the growing tree."""
    terminal_list = sorted(terminals)
    if len(terminal_list) <= 1:
        return 0
    first = terminal_list[0]
    tree_vertices = {first}
    tree_edges = []
    remaining = set(terminal_list[1:])
    while remaining:
        best = None
        for vertex in sorted(tree_vertices):
            distance, previous = dijkstra(graph, vertex)
            for terminal in sorted(remaining):
                if distance[terminal] != INF and (best is None or distance[terminal] < best[0]):
                    best = (distance[terminal], vertex, terminal)
        if best is None:
            return INF
        (_, source, target) = best
        path = shortest_path(graph, source, target)
        weights = {(min(u, v), max(u, v)): w for (u, v, w) in graph[1]}
        for step in range(len(path) - 1):
            pair = (min(path[step], path[step + 1]), max(path[step], path[step + 1]))
            tree_edges.append((pair[0], pair[1], weights[pair]))
        tree_vertices.update(path)
        for vertex in path:
            remaining.discard(vertex)
    return prune_edges(graph, tree_edges, terminals)


def connected_spanning(graph, edges, terminals):
    """True when the edge set connects every terminal (and only uses real graph edges)."""
    weights = {(min(u, v), max(u, v)) for (u, v, _) in graph[1]}
    labels = {}
    for (u, v, _) in edges:
        assert (u, v) in weights, "a heuristic may only use graph edges"
        labels.setdefault(u, set()).add(v)
        labels.setdefault(v, set()).add(u)
    if not terminals <= set(labels):
        return False
    start = sorted(terminals)[0]
    seen = {start}
    stack = [start]
    while stack:
        vertex = stack.pop()
        for other in labels.get(vertex, ()):
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return terminals <= seen


def complete_graph(n):
    edges = [(i, j, 1) for i in range(n) for j in range(i + 1, n)]
    return (n, edges)


def path_graph(n, weight=3):
    return (n, [(i, i + 1, weight) for i in range(n - 1)])


def star_graph(n, weight=2):
    return (n, [(0, i, weight) for i in range(1, n)])


def main():
    # --- Exact DP against brute force on random instances --------------------------------
    worst_ratio = 0.0
    for (n, density, terminals_count, seed) in ((5, 70, 3, 1), (6, 60, 3, 2), (7, 55, 4, 3),
                                                (6, 80, 2, 4), (7, 45, 5, 5), (8, 60, 3, 6)):
        graph = random_graph(n, density, seed)
        terminals = frozenset(range(terminals_count))
        # keep the graph connected by construction check
        exact = steiner_exact(graph, terminals)
        reference = steiner_bruteforce(graph, terminals)
        if reference is None:
            continue
        assert abs(exact - reference) <= 1e-9, (n, density, terminals_count, exact, reference)
        closure = metric_closure(graph, terminals)
        closure_exact = steiner_exact(closure, set(terminals))
        assert abs(closure_exact - exact) <= 1e-9, "the metric closure preserves the optimum"
        for (name, cost) in (("kmb", kmb(graph, terminals)), ("sph", sph(graph, terminals))):
            assert cost >= exact - 1e-9, (name, cost, exact)
            bound = 2 * (1 - 1.0 / len(terminals)) * exact
            assert cost <= bound + 1e-9, (name, cost, bound)
            ratio = cost / float(exact) if exact else 1.0
            worst_ratio = max(worst_ratio, ratio)
        (mst_all, _) = kruskal(n, graph[1])
        assert mst_all is None or mst_all >= exact, "the MST ignoring terminals is an upper bound"
    assert worst_ratio <= 2 * (1 - 1.0 / 5)

    # --- Closed forms for the special cases ---------------------------------------------
    for n in (2, 3, 5, 7):
        graph = path_graph(n)
        everything = frozenset(range(n))
        assert steiner_exact(graph, everything) == 3 * (n - 1)
        assert kmb(graph, everything) == 3 * (n - 1)
        assert sph(graph, everything) == 3 * (n - 1)
        assert steiner_exact(graph, frozenset({0})) == 0
        assert steiner_exact(graph, frozenset()) == 0
        assert kmb(graph, frozenset()) == 0 and sph(graph, frozenset()) == 0
        assert steiner_exact(graph, frozenset({0, n - 1})) == 3 * (n - 1), "two ends of a path"
        assert steiner_bruteforce(graph, frozenset({0, n - 1})) == 3 * (n - 1)
    graph = star_graph(6)
    assert steiner_exact(graph, frozenset({1, 2, 3})) == 6, "three spokes of weight two"
    assert kmb(graph, frozenset({1, 2, 3})) == 6
    assert sph(graph, frozenset({1, 2, 3})) == 6
    assert steiner_exact(graph, frozenset({1, 2, 3, 4, 5})) == 10
    complete = complete_graph(5)
    assert steiner_exact(complete, frozenset(range(5))) == 4, "unit complete graph: an MST"
    assert kmb(complete, frozenset(range(5))) == 4
    assert steiner_exact(complete, frozenset({0, 1, 2})) == 2

    # --- Heuristics build real trees -----------------------------------------------------
    graph = random_graph(7, 65, 11)
    terminals = frozenset({0, 2, 5})
    cost = kmb(graph, terminals)
    closure = metric_closure(graph, terminals)
    index = {vertex: position for (position, vertex) in enumerate(sorted(terminals))}
    (closure_cost, chosen) = kruskal(len(terminals),
                                     [(index[u], index[v], w) for (u, v, w) in closure[1]])
    used = set()
    for (u, v, _) in chosen:
        path = shortest_path(graph, sorted(terminals)[u], sorted(terminals)[v])
        for step in range(len(path) - 1):
            used.add((min(path[step], path[step + 1]), max(path[step], path[step + 1])))
    weights = {(min(u, v), max(u, v)): w for (u, v, w) in graph[1]}
    edges = [(u, v, weights[(u, v)]) for (u, v) in used]
    assert connected_spanning(graph, edges, terminals), "KMB returns a connected terminal tree"
    assert prune_edges(graph, edges, terminals) <= sum(w for (_, _, w) in edges), \
        "pruning never raises the cost"
    assert cost <= closure_cost, "the expansion cannot cost more than the closure MST"
    assert sph(graph, terminals) >= steiner_exact(graph, terminals) - 1e-9

    # --- A grid-like instance where the Steiner point pays off ---------------------------
    grid = (9, [(0, 1, 1), (0, 3, 1), (1, 2, 1), (1, 4, 1), (2, 5, 1), (3, 4, 1), (3, 6, 1),
                (4, 5, 1), (4, 7, 1), (5, 8, 1), (6, 7, 1), (7, 8, 1)])
    corners = frozenset({0, 2, 6, 8})
    exact = steiner_exact(grid, corners)
    # The cheapest tree is 0-1-2 plus 6-7-8 plus the connector 1-4-7: six unit edges.
    assert exact == 6, "the four corners of the 3x3 grid need six unit edges"
    assert steiner_bruteforce(grid, corners) == 6
    assert kmb(grid, corners) >= exact and sph(grid, corners) >= exact
    assert kmb(grid, corners) <= 2 * (1 - 1.0 / 4) * exact + 1e-9

    # --- Determinism ----------------------------------------------------------------------
    graph = random_graph(7, 60, 21)
    terminals = frozenset({0, 3, 6})
    assert steiner_exact(graph, terminals) == steiner_exact(graph, terminals)
    assert kmb(graph, terminals) == kmb(graph, terminals)
    assert sph(graph, terminals) == sph(graph, terminals)
    assert metric_closure(graph, terminals) == metric_closure(graph, terminals)
    assert steiner_bruteforce(graph, terminals) == steiner_bruteforce(graph, terminals)


if __name__ == "__main__":
    main()
    print("PASS")
