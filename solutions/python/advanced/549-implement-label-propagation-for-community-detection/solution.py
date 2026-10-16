"""Task 549 - Implement label propagation for community detection. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Raghavan-Hu-Kumara label propagation: every vertex starts with its own unique label and, in a fixed
vertex order, each vertex adopts a label that is maximal among its neighbours, updating in place
(the asynchronous variant that the published algorithm uses precisely because synchronous updates
swap labels indefinitely on a bipartite structure). Two tie-break rules are implemented and
contrasted, since with unique starting labels the neighbour mode is often tied: the "smallest
label" rule (fully deterministic, but biased - labels collapse towards label 0, and the file
asserts that the ring of four cliques is merged into a single community by it) and a deterministic
pseudo-random rule driven by the repository's LCG with a fixed seed, which reproduces the planted
partition of the ring of four 5-cliques and of two bridged cliques. Both rules are synchronous
fixed points, and the file re-derives that fixed-point property cell by cell - the label of each
vertex must be the mode of its neighbours, and the smallest mode in the min-rule - rather than
trusting the loop that produced it. A synchronous (Jacobi) variant is also implemented to exhibit
the oscillation: K2 and the even cycle C6 never settle synchronously, while the asynchronous sweep
merges both immediately (and C5 settles even synchronously).

Checks: planted recovery with the seeded rule on the ring of four 5-cliques and on two bridged
cliques, with modularity at least the planted partition's; the min-rule over-merging documented on
the same graph; the clique collapsing in a single pass and the path collapsing over a few passes;
the empty graph keeping every vertex single; convergence (updates reaching zero) and pass counts
within the documented bound for many seed/graph combinations; compact labels; communities connected
by an independent traversal; determinism for a fixed seed and independence from the seed only up to
the choice among tied labels; and the synchronous/asynchronous oscillation contrast on K2, C6, C5.
"""
import collections


def lcg(seed):
    return (seed * 75 + 74) % 65537


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(targets) for targets in adj]


def renumber(labels):
    mapping = {}
    out = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append(mapping[label])
    return out


def label_propagation(n, edges, seed=None, max_passes=100):
    """Asynchronous label propagation.

    ``seed is None`` breaks ties by the smallest label; otherwise the LCG picks among the tied
    modes deterministically from the given seed. Returns (labels, updates per pass, converged).
    """
    adj = adjacency(n, edges)
    labels = list(range(n))
    updates = []
    state = 0 if seed is None else seed
    converged = False
    for _ in range(max_passes):
        changed = 0
        for v in range(n):
            if not adj[v]:
                continue
            counts = collections.Counter(labels[w] for w in adj[v])
            top = max(counts.values())
            modes = sorted(label for (label, count) in counts.items() if count == top)
            if seed is None:
                best = modes[0]
            else:
                state = lcg(state)
                best = modes[state % len(modes)]
            if labels[v] != best:
                labels[v] = best
                changed += 1
        updates.append(changed)
        if changed == 0:
            converged = True
            break
    return renumber(labels), updates, converged


def label_propagation_synchronous(n, edges, max_passes=100):
    """Jacobi-style variant reading the previous pass: the classic oscillating version."""
    adj = adjacency(n, edges)
    labels = list(range(n))
    for _ in range(max_passes):
        fresh = list(labels)
        for v in range(n):
            if not adj[v]:
                continue
            counts = collections.Counter(labels[w] for w in adj[v])
            top = max(counts.values())
            fresh[v] = min(label for (label, count) in counts.items() if count == top)
        if fresh == labels:
            return renumber(labels), True
        labels = fresh
    return renumber(labels), False


def neighbour_modes(n, edges, labels, v):
    """The modal labels among the neighbours of v, sorted."""
    adj = adjacency(n, edges)
    counts = collections.Counter(labels[w] for w in adj[v])
    top = max(counts.values())
    return sorted(label for (label, count) in counts.items() if count == top)


def is_fixed_point(n, edges, labels, min_rule=True):
    """Re-derive the update rule for every vertex instead of trusting the producing loop."""
    adj = adjacency(n, edges)
    for v in range(n):
        if not adj[v]:
            continue
        modes = neighbour_modes(n, edges, labels, v)
        if labels[v] not in modes:
            return False
        if min_rule and labels[v] != modes[0]:
            return False
    return True


def community_partition(labels):
    groups = {}
    for v, label in enumerate(labels):
        groups.setdefault(label, set()).add(v)
    return frozenset(frozenset(members) for members in groups.values())


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


def degrees(n, edges):
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def modularity(n, edges, labels):
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


def planted_partition(cliques, size):
    return frozenset(frozenset(range(c * size, c * size + size)) for c in range(cliques))


def lcg_graph(n, density, seed):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            seed = lcg(seed)
            if seed % 100 < density:
                edges.append((i, j))
    return edges, seed


def main():
    # --- The seeded rule recovers planted structure -------------------------------------
    n, edges = ring_of_cliques(4, 5)
    adj = adjacency(n, edges)
    labels, updates, converged = label_propagation(n, edges, seed=12345)
    assert community_partition(labels) == planted_partition(4, 5), \
        "the four 5-cliques are recovered with the seeded tie-break"
    assert converged and updates[-1] == 0
    assert is_fixed_point(n, edges, labels, min_rule=False), "labels are neighbour modes"
    planted_q = modularity(n, edges, [v // 5 for v in range(n)])
    assert abs(modularity(n, edges, labels) - planted_q) <= 1e-12, \
        "recovering the planted partition means matching its modularity"
    assert modularity(n, edges, labels) > 0.5
    assert each_community_connected(labels, adj), "communities are connected"
    assert len(updates) <= n, "propagation converges in far fewer passes than there are vertices"
    repeated = label_propagation(n, edges, seed=12345)
    assert repeated[0] == labels and repeated[1] == updates, "deterministic for a fixed seed"

    # --- The min-rule is deterministic but biased towards label 0 -----------------------
    min_labels, min_updates, min_converged = label_propagation(n, edges)
    assert min_converged, "the min-rule also reaches a fixed point"
    assert len(set(min_labels)) == 1, \
        "with the smallest-label tie-break the whole ring collapses into one community"
    assert is_fixed_point(n, edges, min_labels, min_rule=True), \
        "and that collapsed labelling really is the min-rule fixed point"
    assert modularity(n, edges, min_labels) < planted_q

    # --- More planted structures with the seeded rule ----------------------------------
    for cliques, size in ((2, 5), (2, 8), (3, 6), (4, 6), (3, 8)):
        n, edges = ring_of_cliques(cliques, size)
        labels, updates, converged = label_propagation(n, edges, seed=777)
        assert community_partition(labels) == planted_partition(cliques, size), \
            "rings of cliques with at least five members per clique are recovered"
        assert converged and updates[-1] == 0
        assert is_fixed_point(n, edges, labels, min_rule=False)

    # --- Hand-checked small shapes ------------------------------------------------------
    clique = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    clique_labels, clique_updates, clique_converged = label_propagation(6, clique)
    assert clique_labels == [0] * 6, "one pass over a clique collapses every label"
    assert clique_updates[0] > 0 and clique_updates[-1] == 0 and clique_converged
    path = [(i, i + 1) for i in range(6)]
    path_labels, path_updates, path_converged = label_propagation(7, path)
    assert len(set(path_labels)) == 1, "a path collapses to a single community"
    assert path_converged and path_updates[-1] == 0
    lonely, lonely_updates, lonely_converged = label_propagation(4, [])
    assert lonely == [0, 1, 2, 3], "isolated vertices keep their own labels"
    assert lonely_updates == [0] and lonely_converged
    assert label_propagation(1, [])[0] == [0]
    assert label_propagation(2, [(0, 1)], seed=3)[0] == [0, 0]
    trivial = label_propagation(3, [(0, 0), (0, 1), (1, 1)])[0]
    assert len(trivial) == 3

    # --- Invariants across seeds and random graphs --------------------------------------
    seed = 6060
    graphs = []
    for trial in range(8):
        n = 8 + trial % 4
        edges, seed = lcg_graph(n, 40, seed)
        graphs.append((n, edges))
    for (n, edges) in graphs:
        for rule_seed in (1, 2, 3, 55, 999):
            labels, updates, converged = label_propagation(n, edges, seed=rule_seed)
            assert converged, "the seeded rule converged on every sampled graph"
            assert updates[-1] == 0
            assert len(updates) <= n + 2
            assert is_fixed_point(n, edges, labels, min_rule=False)
            assert each_community_connected(labels, adjacency(n, edges))
            assert min(labels) == 0 and max(labels) == len(set(labels)) - 1, "compact labels"
            assert modularity(n, edges, labels) >= modularity(n, edges, list(range(n))) - 1e-12
            assert label_propagation(n, edges, seed=rule_seed)[0] == labels
        min_labels, min_updates, min_converged = label_propagation(n, edges)
        assert min_converged and is_fixed_point(n, edges, min_labels, min_rule=True)
        assert each_community_connected(min_labels, adjacency(n, edges))

    # --- Synchronous propagation oscillates on bipartite structures --------------------
    assert not label_propagation_synchronous(2, [(0, 1)])[1], \
        "K2 swaps labels forever under synchronous updates"
    even_cycle = [(v, (v + 1) % 6) for v in range(6)]
    assert not label_propagation_synchronous(6, even_cycle)[1], "C6 oscillates synchronously"
    odd_cycle = [(v, (v + 1) % 5) for v in range(5)]
    odd_labels, odd_converged = label_propagation_synchronous(5, odd_cycle)
    assert odd_converged and len(set(odd_labels)) == 1, "an odd cycle settles synchronously"
    for edges, n in ((even_cycle, 6), (odd_cycle, 5)):
        async_labels, async_updates, async_converged = label_propagation(n, edges)
        assert async_converged and async_updates[-1] == 0
        assert len(set(async_labels)) == 1, "the asynchronous sweep merges the cycle"
        assert is_fixed_point(n, edges, async_labels, min_rule=True)
    assert label_propagation_synchronous(2, [(0, 1)])[0] != label_propagation(2, [(0, 1)])[0], \
        "the two variants disagree exactly when synchrony oscillates"


if __name__ == "__main__":
    main()
    print("PASS")
