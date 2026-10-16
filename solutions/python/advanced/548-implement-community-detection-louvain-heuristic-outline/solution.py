"""Task 548 - Implement community detection (Louvain heuristic outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

The Louvain heuristic for modularity maximisation, implemented in full rather than as an outline.
The working graph keeps integer edge weights in adjacency dictionaries, with self-loop weights
standing for contracted community interiors. Phase one visits the super-nodes in a fixed order and
moves each one into the neighbouring community (its own included) with the largest positive
modularity delta; the delta is derived exactly from the community-sum form of Newman's modularity
Q = sum_c [ in_c/(2m) - (tot_c/(2m))^2 ] (in_c the doubled within-community weight, tot_c the
degree sum, m the number of edges): moving v from community A to community B changes Q by
(a - b)/m + k_v * (tot_A - tot_B - k_v) / (2 m^2) with a = weight from v into B, b = weight from v
into A minus v's own self-loop and k_v = degree of v. That closed form is validated on a probe
graph against a from-scratch recomputation of the modularity before and after a move, and every
level's modularity is recomputed from the raw edge list rather than carried forward.
Phase two contracts each community into a super-node, summing edge weights and halving the diagonal
so that self-loops hold twice the internal weight, and the two phases repeat until a level produces
no merge. Ties go to the smallest community label and the vertex order is fixed, so runs are
reproducible; the file records every level's partition and modularity.

Checks: the ring of four 5-cliques recovered exactly (and beating the planted modularity); two
cliques joined by one edge; the complete graph collapsing to one community, the empty graph (each
vertex alone, Q = 0) and a forest; the two quadrature formulas (community sums versus the
(1/2m) sum over vertex pairs of A_uv - k_u k_v / 2m) agreeing to 1e-12 on every level; modularity
non-decreasing from level to level; communities connected, discovered by an independent traversal;
determinism; and brute-force optimal modularity over all set partitions (restricted growth strings)
of every tiny graph, where the heuristic must reach the optimum.
"""
import itertools

INF = float("inf")


def lcg(seed):
    return (seed * 75 + 74) % 65537


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def degrees(n, edges):
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def modularity_community_sums(n, edges, labels):
    """Q = sum_c [ in_c/(2m) - (tot_c/(2m))^2 ] with in_c twice the internal weight."""
    m = len(edges)
    if m == 0:
        return 0.0
    deg = degrees(n, edges)
    internal = {}
    total = {}
    for v in range(n):
        total[labels[v]] = total.get(labels[v], 0) + deg[v]
    for (u, v) in edges:
        if labels[u] == labels[v]:
            internal[labels[u]] = internal.get(labels[u], 0) + 2
    value = 0.0
    for community in set(labels):
        value += internal.get(community, 0) / (2.0 * m)
        value -= (total[community] / (2.0 * m)) ** 2
    return value


def modularity_vertex_pairs(n, edges, labels):
    """Q = 1/(2m) * sum_{u,v} [A_uv - k_u k_v/(2m)] * [c_u == c_v], computed independently."""
    m = len(edges)
    if m == 0:
        return 0.0
    deg = degrees(n, edges)
    matrix = [[0] * n for _ in range(n)]
    for (u, v) in edges:
        matrix[u][v] += 1
        matrix[v][u] += 1
    total = 0.0
    for u in range(n):
        for v in range(n):
            if labels[u] == labels[v]:
                total += matrix[u][v] - deg[u] * deg[v] / (2.0 * m)
    return total / (2.0 * m)


def modularity(n, edges, labels):
    first = modularity_community_sums(n, edges, labels)
    second = modularity_vertex_pairs(n, edges, labels)
    assert abs(first - second) <= 1e-12, "the two modularity formulas must agree"
    return first


def renumber(labels):
    mapping = {}
    out = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return out


def louvain(n, edges, max_levels=50):
    """Deterministic Louvain. Returns (final labels, per-level label vectors, modularities)."""
    graph = [dict() for _ in range(n)]                 # weighted adjacency with self-loops
    for (u, v) in edges:
        if u == v:
            graph[u][u] = graph[u].get(u, 0) + 1
        else:
            graph[u][v] = graph[u].get(v, 0) + 1
            graph[v][u] = graph[v].get(u, 0) + 1
    members = [[v] for v in range(n)]
    levels = [list(range(n))]
    modularities = [modularity(n, edges, list(range(n)))]
    moves_checked = 0
    for _ in range(max_levels):
        size = len(graph)
        # A self-loop of weight w contributes 2w to the degree of its vertex.
        degree = [sum(graph[v].values()) + graph[v].get(v, 0) for v in range(size)]
        double_m = sum(degree)
        if double_m == 0:
            break
        m = double_m / 2.0
        community = list(range(size))
        community_total = list(degree)
        moved = True
        while moved:
            moved = False
            for v in range(size):
                current = community[v]
                weight_to = {}
                own_weight = 0
                for (w, weight) in graph[v].items():
                    if w == v:
                        continue
                    weight_to[community[w]] = weight_to.get(community[w], 0) + weight
                    if community[w] == current:
                        own_weight += weight
                best_community = current
                best_delta = 0.0
                for candidate in sorted(weight_to):
                    if candidate == current:
                        continue
                    delta = (weight_to[candidate] - own_weight) / m \
                        + degree[v] * (community_total[current] - community_total[candidate]
                                       - degree[v]) / (2.0 * m * m)
                    moves_checked += 1
                    if delta > best_delta + 1e-12:
                        best_delta = delta
                        best_community = candidate
                if best_community != current:
                    community_total[current] -= degree[v]
                    community_total[best_community] += degree[v]
                    community[v] = best_community
                    moved = True
        labels = renumber(community)
        contracted = len(set(labels))
        # Aggregation: off-diagonal weights are summed directly, and the diagonal (the self-loop
        # of a super-node) must equal the *edge* weight internal to the community, so the ordered
        # internal pairs are halved while the members' own self-loops are kept as they are.
        loops = [0] * contracted
        ordered = [0] * contracted
        new_graph = [dict() for _ in range(contracted)]
        for v in range(size):
            a = labels[v]
            loops[a] += graph[v].get(v, 0)
            for (w, weight) in graph[v].items():
                b = labels[w]
                if a == b:
                    if w != v:
                        ordered[a] += weight
                else:
                    new_graph[a][b] = new_graph[a].get(b, 0) + weight
        for a in range(contracted):
            new_graph[a][a] = loops[a] + ordered[a] // 2
        new_members = [[] for _ in range(contracted)]
        for v in range(size):
            new_members[labels[v]].extend(members[v])
        projected = [0] * n
        for a in range(contracted):
            for v in new_members[a]:
                projected[v] = a
        levels.append(projected)
        modularities.append(modularity(n, edges, projected))
        members = new_members
        graph = new_graph
        if contracted == size:
            break
    return renumber(levels[-1]), levels, modularities, moves_checked


def community_partition(labels):
    groups = {}
    for v, label in enumerate(labels):
        groups.setdefault(label, set()).add(v)
    return frozenset(frozenset(members) for members in groups.values())


def community_sizes(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def each_community_connected(labels, adj):
    for members in community_partition(labels):
        start = min(members)
        seen = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w in members and w not in seen:
                    seen.add(w)
                    stack.append(w)
        if seen != set(members):
            return False
    return True


def ring_of_cliques(cliques, size):
    n = cliques * size
    edges = []
    for c in range(cliques):
        base = c * size
        for i in range(size):
            for j in range(i + 1, size):
                edges.append((base + i, base + j))
    for c in range(cliques):
        edges.append((c * size + 1, ((c + 1) % cliques) * size + 1))
    return n, edges


def all_partitions(n):
    """Every set partition's label vector, generated as restricted growth strings."""
    if n == 0:
        yield ()
        return
    labels = [0] * n

    def recurse(position, maximum):
        if position == n:
            yield tuple(labels)
            return
        for value in range(maximum + 2):
            labels[position] = value
            yield from recurse(position + 1, max(maximum, value))

    yield from recurse(1, 0)


def main():
    # --- The exact gain formula, checked against a from-scratch recomputation -----------
    probe_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    n = 4
    labels = [0, 0, 1, 1]
    before = modularity(n, probe_edges, labels)
    moved = list(labels)
    moved[1] = 1
    after = modularity(n, probe_edges, moved)
    m = len(probe_edges)
    deg = degrees(n, probe_edges)
    weight_to = 1                                # vertex 1 has one edge into community 1 (vertex 2)
    own_weight = 1                               # and one edge back into community 0 (vertex 0)
    tot_a = deg[0] + deg[1]
    tot_b = deg[2] + deg[3]
    predicted = (weight_to - own_weight) / m + deg[1] * (tot_a - tot_b - deg[1]) / (2.0 * m * m)
    assert abs((after - before) - predicted) <= 1e-12, \
        "the closed-form delta equals the recomputed modularity change"

    # --- Ring of four 5-cliques ---------------------------------------------------------
    n, edges = ring_of_cliques(4, 5)
    adj = adjacency(n, edges)
    labels, levels, modularities, moves = louvain(n, edges)
    expected = frozenset(frozenset(range(c * 5, c * 5 + 5)) for c in range(4))
    assert community_partition(labels) == expected, "the planted communities are recovered"
    planted_q = modularity(n, edges, [v // 5 for v in range(n)])
    assert modularities[-1] >= planted_q - 1e-12, "the heuristic beats the planted split"
    assert modularities[0] < planted_q, "the singleton partition is poor"
    assert all(modularities[i] <= modularities[i + 1] + 1e-12
               for i in range(len(modularities) - 1)), "modularity never decreases"
    assert len(levels) == len(modularities)
    assert levels[0] == list(range(n)), "level 0 is the singleton partition"
    assert community_sizes(labels) == {0: 5, 1: 5, 2: 5, 3: 5}
    assert each_community_connected(labels, adj), "communities are connected subgraphs"
    assert moves > 0
    again = louvain(n, edges)
    assert again[0] == labels and again[2] == modularities, "deterministic"

    # --- Two cliques joined by a single bridge -----------------------------------------
    n, edges = ring_of_cliques(2, 4)
    labels, _, modularities, _ = louvain(n, edges)
    assert community_partition(labels) == frozenset({frozenset({0, 1, 2, 3}),
                                                    frozenset({4, 5, 6, 7})})
    assert modularities[-1] >= modularity(n, edges, [0, 0, 0, 0, 1, 1, 1, 1]) - 1e-12
    assert modularities[-1] > 0.3, "two cliques with one bridge are strong communities"

    # --- Degenerate graphs --------------------------------------------------------------
    empty_labels, empty_levels, empty_q, _ = louvain(5, [])
    assert empty_labels == [0, 1, 2, 3, 4], "no edges means no merging"
    assert empty_q == [0.0], "an edgeless graph has modularity zero"
    complete = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    complete_labels, _, complete_q, _ = louvain(6, complete)
    assert complete_labels == [0] * 6, "K6 has no community structure to find"
    assert abs(complete_q[-1]) <= 1e-12
    assert complete_q[-1] >= complete_q[0] - 1e-12
    assert louvain(1, [])[0] == [0]
    assert louvain(1, [])[2] == [0.0]
    assert louvain(2, [(0, 1)])[2][-1] >= 0.0
    tree = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
    tree_labels, _, tree_q, _ = louvain(6, tree)
    assert all(tree_q[i] <= tree_q[i + 1] + 1e-12 for i in range(len(tree_q) - 1))
    assert modularity(6, tree, tree_labels) == tree_q[-1]

    # --- Brute-force optimality on tiny graphs -----------------------------------------
    seed = 9090
    checked = 0
    for n in (2, 3, 4, 5):
        for density in (30, 60, 90):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    seed = lcg(seed)
                    if seed % 100 < density:
                        edges.append((i, j))
            if not edges:
                continue
            labels, _, modularities, _ = louvain(n, edges)
            optimum = max(modularity(n, edges, list(partition)) for partition in all_partitions(n))
            assert modularities[-1] >= optimum - 1e-9, \
                "on graphs with at most five vertices the heuristic reaches the optimum"
            assert modularities[-1] <= optimum + 1e-9
            checked += 1
    assert checked >= 10, "the brute-force comparison really ran"

    # --- Random graphs: invariants on every level --------------------------------------
    seed = 31337
    for _ in range(8):
        n = 10 + seed % 5
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                seed = lcg(seed)
                if seed % 100 < 45:
                    edges.append((i, j))
        labels, levels, modularities, _ = louvain(n, edges)
        for partition in levels:
            modularity(n, edges, partition)          # asserts both formulas agree
        assert all(modularities[i] <= modularities[i + 1] + 1e-12
                   for i in range(len(modularities) - 1))
        assert labels == louvain(n, edges)[0], "deterministic"
        assert each_community_connected(labels, adjacency(n, edges))
        assert sum(community_sizes(labels).values()) == n
        assert len(set(labels)) <= n


if __name__ == "__main__":
    main()
    print("PASS")
