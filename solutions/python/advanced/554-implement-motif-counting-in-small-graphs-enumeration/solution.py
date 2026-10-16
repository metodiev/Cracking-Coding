"""Task 554 - Implement motif counting in small graphs (enumeration). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Exhaustive counting of the induced k-vertex motifs of a small graph, with two independent
enumeration engines. The first walks all C(n, k) vertex subsets, builds the induced subgraph, and
canonicalises it by minimising the adjacency code over all k! relabelings - for k <= 4 the file
names the classes (4 for k = 3, the 11 labelled graphs for k = 4), and it validates the
canonicaliser itself against the known graph-class counts 1, 2, 4, 11, 34 on 1..5 vertices and the
connected-graph counts 1, 1, 2, 6, 21. The second engine is ESU (Wernicke): for each vertex v the
search starts from {v} with extension set N(v) ∩ {u > v} and exclusion set {u < v}, expands by
removing a vertex w from the extension set, adding N(w) minus exclusion minus the current
subgraph, recursing, and finally adding w to the exclusion set - which emits every connected
k-subset exactly once. The file checks that ESU's output is exactly the connected k-subsets, with
no duplicates and no omissions, and that its per-motif counts match the subset enumeration
restricted to connected patterns. Random instances come from a full-period 64-bit LCG (MMIX
constants) read in its high bits; the repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period
and low-bit structure that would bias the sampled edge density.

Checks: counts of the three-vertex motifs match closed forms on random graphs - induced triangles
equal trace(A^3)/6, the induced path-3 count equals sum_v C(d_v, 2) - 3T (sum_v C(d_v,2) counts
every path once at its centre and every triangle three times), the one-edge count equals
(n-2)m - 2(#P3) - 3T (each edge lies in n-2 triples), and the independent-triple count is the rest
of C(n,3) - so the four counts partition C(n,3) exactly; K_{3,3} gives 6 stars and 9 four-cycles
and nothing else, K5 gives five K4 and ten triangles, K3 ∪ K3 gives two triangles and eighteen
one-edge triples with no induced paths, and C6 gives no triangle, six induced paths and two
independent triples. Motif counts are invariant under vertex relabeling; connected motif
frequencies sum to one; ESU agrees with the subset enumeration on graphs up to 14 vertices.
Honest caveats: counting is of *induced* subgraphs - a triangle in K4 is four induced triangles
but 24 labelled occurrences, and P3 occurs 12 times in C6 as a subgraph but only 6 times induced -
and ESU only emits connected subsets, so motif frequencies have to be normalised over the connected
subgraphs found by ESU rather than over all subsets, which the file demonstrates on an instance
where dividing the connected counts by the count of *all* subsets gives frequencies that sum to
considerably more than one.
"""
import itertools

MASK64 = (1 << 64) - 1
GRAPH_CLASS_COUNTS = [1, 2, 4, 11, 34]
CONNECTED_CLASS_COUNTS = [1, 1, 2, 6, 21]


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_graph(n, density, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 9) & MASK64)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return (n, adj)


def all_graphs(n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    graphs = []
    for mask in range(1 << len(pairs)):
        adj = [0] * n
        for index, (i, j) in enumerate(pairs):
            if mask >> index & 1:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
        graphs.append((n, adj))
    return graphs


def canonical_key(local_adj, k):
    """Smallest adjacency code over all k! relabelings - the motif class of the subgraph."""
    best = None
    for perm in itertools.permutations(range(k)):
        code = 0
        shift = 0
        for i in range(k):
            row = local_adj[perm[i]]
            for j in range(i + 1, k):
                if (row >> perm[j]) & 1:
                    code |= 1 << shift
                shift += 1
        if best is None or code < best:
            best = code
    return best


def key_from_edges(k, edges):
    local = [0] * k
    for (a, b) in edges:
        local[a] |= 1 << b
        local[b] |= 1 << a
    return canonical_key(local, k)


PATTERNS_3 = {"independent-3": key_from_edges(3, []),
              "one-edge": key_from_edges(3, [(0, 1)]),
              "path-3": key_from_edges(3, [(0, 1), (1, 2)]),
              "triangle": key_from_edges(3, [(0, 1), (1, 2), (2, 0)])}
PATTERNS_4 = {"K4": key_from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
              "C4": key_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
              "K1,3": key_from_edges(4, [(0, 1), (0, 2), (0, 3)]),
              "paw": key_from_edges(4, [(0, 1), (1, 2), (2, 0), (1, 3)]),
              "diamond": key_from_edges(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]),
              "P4": key_from_edges(4, [(0, 1), (1, 2), (2, 3)]),
              "P3+isolated": key_from_edges(4, [(0, 1), (1, 2)]),
              "2K2": key_from_edges(4, [(0, 1), (2, 3)]),
              "K2+2isolated": key_from_edges(4, [(0, 1)]),
              "K3+isolated": key_from_edges(4, [(0, 1), (1, 2), (2, 0)]),
              "independent-4": key_from_edges(4, [])}
CONNECTED_KEYS_4 = {PATTERNS_4[name] for name in ("K4", "C4", "K1,3", "paw", "diamond", "P4")}
NAME_OF_KEY_3 = {key: name for (name, key) in PATTERNS_3.items()}
NAME_OF_KEY_4 = {key: name for (name, key) in PATTERNS_4.items()}


def subset_key(graph, subset):
    _, adj = graph
    k = len(subset)
    local = [0] * k
    for i in range(k):
        row = adj[subset[i]]
        for j in range(k):
            if i != j and (row >> subset[j]) & 1:
                local[i] |= 1 << j
    return canonical_key(local, k)


def mask_key(graph, mask):
    subset = [index for index in range(graph[0]) if (mask >> index) & 1]
    return subset_key(graph, subset)


def motif_counts(graph, k, connected_only=False):
    """Counts per motif class over all (or only the connected) k-vertex induced subsets."""
    n, _ = graph
    counts = {}
    for subset in itertools.combinations(range(n), k):
        if connected_only and not is_connected(graph, subset):
            continue
        key = subset_key(graph, subset)
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_connected(graph, subset):
    _, adj = graph
    if len(subset) <= 1:
        return True
    mask = 0
    for vertex in subset:
        mask |= 1 << vertex
    seen = 1 << subset[0]
    frontier = adj[subset[0]] & mask
    while frontier:
        seen |= frontier
        following = 0
        bits = frontier
        while bits:
            low = bits & -bits
            bits ^= low
            following |= adj[low.bit_length() - 1] & mask & ~seen
        frontier = following
    return seen == mask


def connected_subsets(graph, k):
    n, _ = graph
    return [subset for subset in itertools.combinations(range(n), k)
            if is_connected(graph, subset)]


def esu_counts(graph, k):
    """ESU enumeration of connected k-vertex induced subgraphs; returns (counts, subsets)."""
    n, adj = graph
    full = (1 << n) - 1
    counts = {}
    emitted = []

    def extend(subset, extension, excluded):
        if bin(subset).count("1") == k:
            key = mask_key(graph, subset)
            counts[key] = counts.get(key, 0) + 1
            emitted.append(subset)
            return
        while extension:
            low = extension & -extension
            extension ^= low
            vertex = low.bit_length() - 1
            grown = extension | (adj[vertex] & ~subset & ~excluded & full)
            extend(subset | low, grown, excluded)
            excluded |= low

    for start in range(n):
        if k == 1:
            counts[mask_key(graph, 1 << start)] = counts.get(mask_key(graph, 1 << start), 0) + 1
            emitted.append(1 << start)
            continue
        extend(1 << start, adj[start] & ~((1 << (start + 1)) - 1) & full, (1 << start) - 1)
    return counts, emitted


def count_monomorphisms(graph, pattern_edges, k):
    """Labelled occurrences of the pattern as a subgraph (extra target edges allowed)."""
    n, adj = graph
    pattern = [0] * k
    for (a, b) in pattern_edges:
        pattern[a] |= 1 << b
        pattern[b] |= 1 << a
    total = 0
    for assignment in itertools.permutations(range(n), k):
        ok = True
        for i in range(k):
            for j in range(i + 1, k):
                if (pattern[i] >> j) & 1 and not (adj[assignment[i]] >> assignment[j]) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            total += 1
    return total


def triangle_count_by_matrix(graph):
    n, adj = graph
    matrix = [[(adj[i] >> j) & 1 for j in range(n)] for i in range(n)]
    square = [[sum(matrix[i][k] * matrix[k][j] for k in range(n)) for j in range(n)]
              for i in range(n)]
    cube = [[sum(square[i][k] * matrix[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]
    return sum(cube[i][i] for i in range(n)) // 6


def relabelled(graph, perm):
    n, adj = graph
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if (adj[i] >> j) & 1:
                new_adj[perm[i]] |= 1 << perm[j]
    return (n, new_adj)


def complete(n):
    full = (1 << n) - 1
    return (n, [full & ~(1 << u) for u in range(n)])


def bipartite(a, b):
    adj = [0] * (a + b)
    for i in range(a):
        for j in range(a, a + b):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    return (a + b, adj)


def path_graph(n):
    adj = [0] * n
    for i in range(n - 1):
        adj[i] |= 1 << (i + 1)
        adj[i + 1] |= 1 << i
    return (n, adj)


def cycle_graph(n):
    graph = path_graph(n)
    adj = graph[1]
    adj[0] |= 1 << (n - 1)
    adj[n - 1] |= 1 << 0
    return (n, adj)


def disjoint_union(*graphs):
    total = sum(graph[0] for graph in graphs)
    adj = [0] * total
    offset = 0
    for graph in graphs:
        n, local = graph
        for i in range(n):
            row = local[i]
            for j in range(n):
                if (row >> j) & 1:
                    adj[offset + i] |= 1 << (offset + j)
        offset += n
    return (total, adj)


def frequencies(counts):
    total = float(sum(counts.values()))
    return {key: value / total for (key, value) in counts.items()}


def main():
    # --- The canonicaliser reproduces the known graph-class counts ------------------------
    assert [len({canonical_key(graph[1], graph[0]) for graph in all_graphs(n)})
            for n in (1, 2, 3, 4, 5)] == GRAPH_CLASS_COUNTS
    assert [len({canonical_key(graph[1], graph[0]) for graph in all_graphs(n)
                 if is_connected(graph, range(graph[0]))})
            for n in (1, 2, 3, 4, 5)] == CONNECTED_CLASS_COUNTS
    assert len(set(PATTERNS_3.values())) == 4 and len(set(PATTERNS_4.values())) == 11
    assert NAME_OF_KEY_4[PATTERNS_4["diamond"]] == "diamond"

    # --- Three-vertex motifs against the closed forms -------------------------------------
    for (n, density, seed) in ((9, 40, 1), (10, 55, 2), (12, 30, 3), (11, 70, 4), (14, 25, 5)):
        graph = random_graph(n, density, seed)
        _, adj = graph
        degrees = [bin(row).count("1") for row in adj]
        edges = sum(degrees) // 2
        counts = motif_counts(graph, 3)
        triangles = counts.get(PATTERNS_3["triangle"], 0)
        paths = counts.get(PATTERNS_3["path-3"], 0)
        one_edge = counts.get(PATTERNS_3["one-edge"], 0)
        independent = counts.get(PATTERNS_3["independent-3"], 0)
        assert triangles == triangle_count_by_matrix(graph)
        assert paths == sum(d * (d - 1) // 2 for d in degrees) - 3 * triangles
        assert one_edge == (n - 2) * edges - 2 * paths - 3 * triangles
        assert independent == n * (n - 1) * (n - 2) // 6 - one_edge - paths - triangles
        assert one_edge + paths + triangles + independent == n * (n - 1) * (n - 2) // 6

    # --- Named instances ------------------------------------------------------------------
    k33 = bipartite(3, 3)
    counts = motif_counts(k33, 4)
    assert counts.get(PATTERNS_4["K1,3"], 0) == 6, "a star is a centre plus its whole part"
    assert counts.get(PATTERNS_4["C4"], 0) == 9, "choose two vertices in each part"
    assert sum(counts.values()) == 15 and set(counts) == {PATTERNS_4["K1,3"], PATTERNS_4["C4"]}

    k5 = complete(5)
    assert motif_counts(k5, 4) == {PATTERNS_4["K4"]: 5}
    assert motif_counts(k5, 3) == {PATTERNS_3["triangle"]: 10}
    assert motif_counts(complete(4), 4) == {PATTERNS_4["K4"]: 1}
    assert motif_counts(complete(4), 3) == {PATTERNS_3["triangle"]: 4}

    c6 = cycle_graph(6)
    assert motif_counts(c6, 3) == {PATTERNS_3["path-3"]: 6, PATTERNS_3["one-edge"]: 12,
                                   PATTERNS_3["independent-3"]: 2}
    # The 15 four-vertex subsets of C6: 6 are paths of four consecutive vertices, 6 are a
    # two-edge path plus an isolated vertex and 3 are two disjoint edges.
    assert motif_counts(c6, 4) == {PATTERNS_4["P4"]: 6, PATTERNS_4["P3+isolated"]: 6,
                                   PATTERNS_4["2K2"]: 3}

    two_triangles = disjoint_union(complete(3), complete(3))
    three_counts = motif_counts(two_triangles, 3)
    assert three_counts == {PATTERNS_3["triangle"]: 2, PATTERNS_3["one-edge"]: 18}
    assert motif_counts(two_triangles, 4) == {PATTERNS_4["K3+isolated"]: 6,
                                              PATTERNS_4["2K2"]: 9}

    # --- Invariance under relabelling ------------------------------------------------------
    graph = random_graph(10, 45, 17)
    for perm in ((1, 0, 2, 3, 4, 5, 6, 7, 8, 9), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                 (3, 5, 7, 9, 1, 0, 2, 4, 6, 8)):
        assert motif_counts(relabelled(graph, perm), 4) == motif_counts(graph, 4)

    # --- ESU against the subset enumeration ----------------------------------------------
    for (n, density, seed, k) in ((12, 35, 21, 3), (12, 35, 21, 4), (14, 30, 22, 4),
                                  (13, 45, 23, 4), (11, 50, 24, 2)):
        graph = random_graph(n, density, seed)
        esu, emitted = esu_counts(graph, k)
        reference = motif_counts(graph, k, connected_only=True)
        assert esu == reference, (n, k, esu, reference)
        expected_masks = sorted(sum(1 << v for v in subset)
                                for subset in connected_subsets(graph, k))
        assert sorted(emitted) == expected_masks, "ESU emits every connected subset exactly once"
        assert len(set(emitted)) == len(emitted)
        if k == 4:
            assert set(reference) <= CONNECTED_KEYS_4
        elif k == 3:
            assert set(reference) <= set(PATTERNS_3.values())
        naive = motif_counts(graph, k)
        for key, value in reference.items():
            assert naive[key] == value, "connected patterns are counted the same way"

    # --- Frequencies ----------------------------------------------------------------------
    graph = random_graph(12, 40, 31)
    connected = motif_counts(graph, 4, connected_only=True)
    share = frequencies(connected)
    assert abs(sum(share.values()) - 1.0) <= 1e-12
    assert len(connected) >= 2
    all_subsets = sum(motif_counts(graph, 4).values())
    inflated = {key: value / float(all_subsets) for (key, value) in connected.items()}
    assert abs(sum(inflated.values()) - sum(share.values())) >= 0.1, \
        "using the count of all subsets as ESU's denominator inflates the frequencies" \
        "and they stop summing to one"

    # --- Documented caveat: induced counting is not occurrence counting -------------------
    assert motif_counts(complete(4), 3).get(PATTERNS_3["triangle"], 0) == 4
    assert count_monomorphisms(complete(4), [(0, 1), (1, 2), (2, 0)], 3) == 24, \
        "one induced triangle per triple, but 24 labelled occurrences in K4"
    assert motif_counts(c6, 3).get(PATTERNS_3["path-3"], 0) == 6
    assert count_monomorphisms(c6, [(0, 1), (1, 2)], 3) == 12, \
        "P3 occurs 12 times as a subgraph of C6 but only 6 times as an induced subgraph"
    assert count_monomorphisms(complete(3), [(0, 1), (1, 2)], 3) == 6
    assert motif_counts(complete(3), 3) == {PATTERNS_3["triangle"]: 1}

    # --- Degenerate inputs ---------------------------------------------------------------
    assert motif_counts(graph, 1) == {0: graph[0]}
    assert motif_counts(graph, 0) == {0: 1}
    assert motif_counts(graph, graph[0] + 1) == {}
    assert esu_counts((1, [0]), 1) == ({0: 1}, [1])
    assert esu_counts((3, [0, 0, 0]), 2) == ({}, [])
    assert motif_counts((3, [0, 0, 0]), 2) == {0: 3}, "no edge, so the empty 2-vertex motif"
    assert motif_counts(complete(2), 2) == {1: 1}


if __name__ == "__main__":
    main()
    print("PASS")
