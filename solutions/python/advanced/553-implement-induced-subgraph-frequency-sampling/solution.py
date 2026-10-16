"""Task 553 - Implement induced subgraph frequency sampling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Estimating the frequencies of the connected induced k-vertex patterns ("motifs") of a graph by
sampling subsets instead of enumerating them. Three components. (1) A canonical key: the induced
subgraph on a vertex subset is relabelled by every permutation of its k vertices and the smallest
adjacency code wins, so two subsets get the same key exactly when they induce the same graph (for
k = 4 this yields all 11 labelled-graph classes, which the file checks). (2) A uniform k-subset
sampler: partial Fisher-Yates driven by a full-period 64-bit LCG (MMIX constants) read in its high
bits, so the estimator count/trials is unbiased for the fraction of k-subsets inducing a pattern.
(3) A Metropolis random walk over the *connected* k-subsets, which is the distribution actually
wanted for motif frequencies: it proposes u outside the current subset and v ∈ S ∩ N(u) inside it,
moves to S \ {v} ∪ {u}, and accepts with probability min(1, |S ∩ N(u)| / |T ∩ N(v)|), which is
exactly the ratio of the two proposal densities - so the stationary distribution is uniform over
the connected k-subsets. The 16-bit repo LCG is not used anywhere here: for tens of thousands of
independent draws its short period and low-bit structure would show up as structure in the sampled
subsets.

Sizes, all fixed so the file stays fast: the uniformity and marginal checks spend 60000 uniform
draws on an 11-vertex instance (330 four-vertex subsets to compare against), the estimator uses 6
families of 2000 draws and the error-versus-trials study 3 families at 200, 800 and 3200 draws, the
Metropolis walk runs 40000 steps on a 9-vertex instance with a fifth of them discarded as burn-in,
and the transition matrix is built on a 9-vertex instance whose connected-subset space still has
more than 20 states.

Checks: canonical keys are invariant under relabelling and separate subsets with different edge
counts; exact enumeration matches closed forms - the number of induced triangles is trace(A^3)/6,
the induced P3 count is sum_v C(d_v, 2) - 3T, the one-edge count is (n-2)m - 2(#P3) - 3T and the
independent-triple count is C(n,3) minus the rest - on random graphs, and the 11 four-vertex keys
are shown to be complete and distinct. Sampling: the uniform sampler hits every vertex with
frequency k/n, its empirical distribution over subsets is within total variation 0.05 of uniform,
the pattern estimator averaged over 6 seed families (12000 draws) lands within three standard
errors of the exact frequency and shrinks as trials grow (200, 800 and 3200 draws over 3 families
each). The walk is checked to stay inside the connected
subsets and to be deterministic, its exact transition matrix is built and checked to be stochastic
and symmetric - which forces the uniform distribution over the connected subsets to be stationary,
an exact statement rather than a sampled one - and, after a burn-in, its visiting distribution is
within total variation 0.05 of the uniform distribution over connected subsets found by
enumeration. That last one is the only empirical statement: no mixing-time bound is proved or
claimed here, and the file says so.
Counterexample made explicit: connected-only sampling is not the same target as uniform subset
sampling - on the tested graph the connected-pattern frequencies differ substantially from the
frequencies over all subsets - so the two estimators are compared only against their own exact
reference. Degenerate inputs (k = 0, k = 1, k = n, k > n, graphs with no connected k-subset) are
pinned down.
"""
import itertools

MASK64 = (1 << 64) - 1

PERMUTATION_CACHE = {}


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def draw(state, bound):
    """Uniform integer in [0, bound) from the high bits of a 64-bit LCG state."""
    state = prng64(state)
    return state, (state >> 17) % bound


def random_graph(n, density, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 5) & MASK64)
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


def permutations_of(k):
    if k not in PERMUTATION_CACHE:
        PERMUTATION_CACHE[k] = list(itertools.permutations(range(k)))
    return PERMUTATION_CACHE[k]


RELABELING_TRIPLES = {}


RELABELINGS = {3: [(0, 1, 2), (2, 1, 0), (1, 2, 0)],
               4: [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)]}


def relabeling_triples(k):
    """For every relabeling, the (i, j, 1 << shift) triples of its adjacency-code bits."""
    if k not in RELABELING_TRIPLES:
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        RELABELING_TRIPLES[k] = [[(perm[i], perm[j], 1 << shift)
                                  for (shift, (i, j)) in enumerate(pairs)]
                                 for perm in permutations_of(k)]
    return RELABELING_TRIPLES[k]


def canonical_key(local_adj, k):
    """Smallest adjacency code over all k! relabelings.

    The k! relabelings are hoisted into a table of (i, j, bit) triples, so the inner loop is a
    flat sequence of bit tests instead of nested index arithmetic. ``canonical_key_reference``
    below is the straightforward formulation, and the checks assert the two agree on every
    4-vertex graph and on random relabelings, so the speed-up cannot have changed the keys.
    """
    best = 1 << 30
    for triples in relabeling_triples(k):
        code = 0
        for (i, j, bit) in triples:
            if (local_adj[i] >> j) & 1:
                code += bit
        if code < best:
            best = code
    return best


def canonical_key_reference(local_adj, k):
    """The unoptimised definition, kept as the cross-check for ``canonical_key``."""
    best = None
    for perm in permutations_of(k):
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


def induced_key(graph, subset):
    _, adj = graph
    k = len(subset)
    local = [0] * k
    for i in range(k):
        row = adj[subset[i]]
        for j in range(k):
            if i != j and (row >> subset[j]) & 1:
                local[i] |= 1 << j
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
NAME_OF_KEY_3 = {key: name for (name, key) in PATTERNS_3.items()}
NAME_OF_KEY_4 = {key: name for (name, key) in PATTERNS_4.items()}
CONNECTED_KEYS_4 = {PATTERNS_4[name] for name in ("K4", "C4", "K1,3", "paw", "diamond", "P4")}


def exact_pattern_counts(graph, k):
    """Counts of canonical induced-pattern keys over all C(n, k) vertex subsets."""
    n, _ = graph
    counts = {}
    for subset in itertools.combinations(range(n), k):
        key = induced_key(graph, subset)
        counts[key] = counts.get(key, 0) + 1
    return counts


def connected_within(graph, subset):
    _, adj = graph
    if not subset:
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
    """All k-subsets inducing a connected subgraph, in a deterministic order."""
    n, _ = graph
    return [subset for subset in itertools.combinations(range(n), k)
            if connected_within(graph, subset)]


def uniform_subsets(graph, k, trials, seed):
    """Uniform k-subsets by partial Fisher-Yates; returns (subsets, final rng state)."""
    n, _ = graph
    state = prng64((seed * 0x9E3779B97F4A7C15 + 7) & MASK64)
    drawn = []
    if k > n:
        return drawn, state
    for _ in range(trials):
        pool = list(range(n))
        for i in range(k):
            state, offset = draw(state, n - i)
            j = i + offset
            pool[i], pool[j] = pool[j], pool[i]
        drawn.append(tuple(pool[:k]))
    return drawn, state


def pattern_counts_of(graph, subsets):
    counts = {}
    for subset in subsets:
        key = induced_key(graph, subset)
        counts[key] = counts.get(key, 0) + 1
    return counts


def walk_connected(graph, k, steps, seed, start=None, burn_in=None):
    """Metropolis walk over connected k-subsets; uniform stationary distribution.

    Proposal: pick u uniformly outside S, then v uniformly in S ∩ N(u); the candidate is
    T = S \\ {v} ∪ {u}. For the reverse move the picked pair is (v, u), so the acceptance
    probability min(1, |S ∩ N(u)| / |T ∩ N(v)|) is exactly the proposal-density ratio.
    """
    n, adj = graph
    space = connected_subsets(graph, k)
    if not space:
        return [], space
    if start is None:
        start = space[0]
    current = frozenset(start)
    state = prng64((seed * 0x9E3779B97F4A7C15 + 11) & MASK64)
    if burn_in is None:
        burn_in = steps // 5
    visits = []
    for step in range(steps):
        outside = [v for v in range(n) if v not in current]
        if outside and len(outside) > 0 and k > 0:
            state, offset = draw(state, len(outside))
            u = outside[offset]
            neighbours = sorted(v for v in current if (adj[u] >> v) & 1)
            if neighbours:
                state, offset = draw(state, len(neighbours))
                v = neighbours[offset]
                candidate = (current - {v}) | {u}
                if connected_within(graph, tuple(sorted(candidate))):
                    beta = len(neighbours)                      # |S ∩ N(u)|
                    alpha = sum(1 for x in candidate if (adj[v] >> x) & 1)   # |T ∩ N(v)|
                    state, roll = draw(state, 1 << 20)
                    if alpha > 0 and roll / float(1 << 20) < min(1.0, beta / alpha):
                        current = frozenset(candidate)
        if step >= burn_in:
            visits.append(current)
    return visits, space


def total_variation(empirical, reference):
    keys = set(empirical) | set(reference)
    return 0.5 * sum(abs(empirical.get(key, 0.0) - reference.get(key, 0.0)) for key in keys)


def distribution(items, space):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = float(len(items))
    return {key: counts.get(key, 0) / total for key in space}


def transition_matrix(graph, k):
    """Exact transition matrix of the walk over the connected k-subsets (source order stable)."""
    n, adj = graph
    space = connected_subsets(graph, k)
    index = {frozenset(subset): i for i, subset in enumerate(space)}
    size = len(space)
    matrix = [[0.0] * size for _ in range(size)]
    for subset in space:
        current = frozenset(subset)
        row = index[current]
        outside = [v for v in range(n) if v not in current]
        if not outside:
            matrix[row][row] = 1.0
            continue
        probability_u = 1.0 / len(outside)
        for u in outside:
            neighbours = sorted(v for v in current if (adj[u] >> v) & 1)
            if not neighbours:
                matrix[row][row] += probability_u      # u is not adjacent to the current subset
                continue
            probability_v = probability_u / len(neighbours)
            for v in neighbours:
                candidate = (current - {v}) | {u}
                target = frozenset(candidate)
                if target not in index:
                    matrix[row][row] += probability_v
                    continue
                alpha = sum(1 for x in candidate if (adj[v] >> x) & 1)
                acceptance = min(1.0, len(neighbours) / float(alpha))
                matrix[row][index[target]] += probability_v * acceptance
                matrix[row][row] += probability_v * (1.0 - acceptance)
    return space, matrix


def matmul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def triangle_count_by_matrix(graph):
    n, adj = graph
    matrix = [[(adj[i] >> j) & 1 for j in range(n)] for i in range(n)]
    cube = matmul(matmul(matrix, matrix), matrix)
    return sum(cube[i][i] for i in range(n)) // 6


def main():
    # --- Canonical keys -------------------------------------------------------------------
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert PATTERNS_4["K4"] == key_from_edges(4, k4_edges[::-1]), "key is relabeling invariant"
    assert len(set(PATTERNS_3.values())) == 4
    assert len(set(PATTERNS_4.values())) == 11, "11 labelled graphs on 4 vertices"
    assert NAME_OF_KEY_4[PATTERNS_4["C4"]] == "C4"
    # The hoisted relabeling table has to reproduce the straightforward definition bit for bit on
    # every 3- and 4-vertex graph and on random relabelings of them.
    for graph in all_graphs(3) + all_graphs(4):
        k = graph[0]
        local = list(graph[1])
        assert canonical_key(local, k) == canonical_key_reference(local, k)
        for perm in RELABELINGS[k]:
            relabelled = [0] * k
            for (i, j) in ((i, j) for i in range(k) for j in range(k) if (local[i] >> j) & 1):
                relabelled[perm[i]] |= 1 << perm[j]
            assert canonical_key(relabelled, k) == canonical_key_reference(relabelled, k)
            assert canonical_key(relabelled, k) == canonical_key(local, k), "relabeling invariant"
    # every 4-vertex pattern seen in any small graph is one of the 11 named classes
    for graph in all_graphs(4) + all_graphs(5):
        keys = set(exact_pattern_counts(graph, 4))
        assert keys <= set(PATTERNS_4.values()), keys
    assert key_from_edges(4, []) != key_from_edges(4, [(0, 1)])

    # --- Exact enumeration against closed forms ------------------------------------------
    for (n, density, seed) in ((9, 40, 1), (10, 55, 2), (12, 30, 3), (11, 70, 4)):
        graph = random_graph(n, density, seed)
        _, adj = graph
        degrees = [bin(row).count("1") for row in adj]
        edges = sum(degrees) // 2
        counts = exact_pattern_counts(graph, 3)
        triangles = counts.get(PATTERNS_3["triangle"], 0)
        paths = counts.get(PATTERNS_3["path-3"], 0)
        one_edge = counts.get(PATTERNS_3["one-edge"], 0)
        independent = counts.get(PATTERNS_3["independent-3"], 0)
        assert triangles == triangle_count_by_matrix(graph)
        # sum_v C(d_v, 2) counts each ordered pair of neighbours of v, i.e. each induced P3 once
        # (at its centre) plus three times per triangle, so the induced-P3 count is that sum
        # minus 3T.
        assert paths == sum(d * (d - 1) // 2 for d in degrees) - 3 * triangles
        assert one_edge == (n - 2) * edges - 2 * paths - 3 * triangles
        assert independent == n * (n - 1) * (n - 2) // 6 - one_edge - paths - triangles
        assert one_edge + paths + triangles + independent == n * (n - 1) * (n - 2) // 6
    # 4-vertex counts are exhaustive and exhaustive in the 11 classes
    for (n, density, seed) in ((8, 40, 5), (9, 50, 6)):
        graph = random_graph(n, density, seed)
        counts = exact_pattern_counts(graph, 4)
        assert sum(counts.values()) == n * (n - 1) * (n - 2) * (n - 3) // 24
        assert set(counts) <= set(PATTERNS_4.values())

    # --- Known counts ---------------------------------------------------------------------
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

    k33 = bipartite(3, 3)
    counts = exact_pattern_counts(k33, 4)
    assert counts.get(PATTERNS_4["K1,3"], 0) == 6, "3 choices of part, 1 star centre each way"
    assert counts.get(PATTERNS_4["C4"], 0) == 9, "choose 2 vertices in each part"
    assert sum(counts.values()) == 15 == 6 + 9
    assert set(counts) == {PATTERNS_4["K1,3"], PATTERNS_4["C4"]}, "K3,3 is triangle-free"
    k5 = complete(5)
    counts5 = exact_pattern_counts(k5, 4)
    assert counts5 == {PATTERNS_4["K4"]: 5}
    assert exact_pattern_counts(k5, 3) == {PATTERNS_3["triangle"]: 10}
    assert exact_pattern_counts(complete(3), 3) == {PATTERNS_3["triangle"]: 1}

    # --- Uniform subset sampling ----------------------------------------------------------
    # The uniformity and marginal tests get their own instance: 330 four-vertex subsets are
    # enough cells for a sharp total-variation check at 60000 draws, and the estimator below
    # keeps the larger 12-vertex instance.
    sampler_graph = random_graph(11, 40, 21)
    n = sampler_graph[0]
    trials = 60000
    drawn, _ = uniform_subsets(sampler_graph, 4, trials, 99)
    membership = [0] * n
    for subset in drawn:
        for vertex in subset:
            membership[vertex] += 1
    for vertex in range(n):
        rate = membership[vertex] / float(trials)
        assert abs(rate - 4.0 / n) <= 4 * (4.0 / n * (1 - 4.0 / n) / trials) ** 0.5
    space = sorted(itertools.combinations(range(n), 4))
    empirical = distribution([frozenset(s) for s in drawn], [frozenset(s) for s in space])
    uniform_reference = {frozenset(s): 1.0 / len(space) for s in space}
    assert total_variation(empirical, uniform_reference) <= 0.05, "sampler is uniform"
    short_run = uniform_subsets(sampler_graph, 4, 4000, 99)
    assert drawn[:4000] == short_run[0], "the same seed replays the same prefix"
    assert uniform_subsets(sampler_graph, 4, 4000, 99) == short_run, "sampling is deterministic"

    # --- Sampling estimator versus the exact frequency ------------------------------------
    samples = 2000
    exact = exact_pattern_counts(graph, 4)
    reference = {key: value / float(sum(exact.values())) for (key, value) in exact.items()}
    key_of_interest = PATTERNS_4["K1,3"] if PATTERNS_4["K1,3"] in reference else \
        max(reference, key=lambda key: reference[key])
    truth = reference[key_of_interest]
    estimates = []
    for family in range(6):
        subset_sample, _ = uniform_subsets(graph, 4, samples, 500 + family)
        counts = pattern_counts_of(graph, subset_sample)
        estimates.append(counts.get(key_of_interest, 0) / float(samples))
    mean = sum(estimates) / len(estimates)
    standard_error = (truth * (1 - truth) / (samples * len(estimates))) ** 0.5
    assert abs(mean - truth) <= 3 * standard_error + 1e-9, "the estimator is unbiased in the mean"
    for estimate in estimates:
        assert abs(estimate - truth) <= 4 * (truth * (1 - truth) / samples) ** 0.5 + 1e-9
    errors = []
    for trials_count in (200, 800, 3200):
        total = 0.0
        for family in range(3):
            subset_sample, _ = uniform_subsets(graph, 4, trials_count, 900 + family)
            counts = pattern_counts_of(graph, subset_sample)
            for (key, value) in reference.items():
                total += abs(counts.get(key, 0) / float(trials_count) - value)
        errors.append(total / (5 * len(reference)))
    assert errors[0] > errors[2], "more draws means smaller average error"

    # --- Metropolis walk over connected subsets ------------------------------------------
    graph = random_graph(9, 45, 31)
    visits, connected_space = walk_connected(graph, 4, 40000, 1234)
    assert connected_space, "the instance has connected 4-subsets"
    assert len(connected_space) < len(list(itertools.combinations(range(graph[0]), 4)))
    assert all(connected_within(graph, tuple(sorted(subset))) for subset in visits), \
        "the walk stays inside the connected subsets"
    connected_lookup = {frozenset(s) for s in connected_space}
    assert all(subset in connected_lookup for subset in visits)
    visit_distribution = distribution(visits, [frozenset(s) for s in connected_space])
    reference_walk = {frozenset(s): 1.0 / len(connected_space) for s in connected_space}
    variation = total_variation(visit_distribution, reference_walk)
    walk_again, _ = walk_connected(graph, 4, 40000, 1234)
    assert walk_again == visits, "the walk is deterministic"
    assert variation <= 0.05, "the visit distribution is close to uniform"

    # The stationary distribution is uniform by construction, and that is verified exactly
    # rather than by sampling: with acceptance min(1, q(T->S)/q(S->T)) the transition matrix of
    # a Metropolis chain aimed at a uniform distribution is symmetric, so its rows summing to
    # one already forces the uniform vector to be stationary.
    small = random_graph(9, 50, 44)
    small_space, matrix = transition_matrix(small, 4)
    assert len(small_space) > 20
    for i, row in enumerate(matrix):
        assert abs(sum(row) - 1.0) <= 1e-12, "stochastic matrix"
        for j, value in enumerate(row):
            assert abs(value - matrix[j][i]) <= 1e-12, "the chain is reversible for uniform pi"
    size = len(small_space)
    distribution_vector = [1.0 / size] * size
    for _ in range(20):
        distribution_vector = [sum(distribution_vector[i] * matrix[i][j] for i in range(size))
                               for j in range(size)]
    assert all(abs(value - 1.0 / size) <= 1e-12 for value in distribution_vector), \
        "uniform is stationary"

    # --- Connected sampling is a different target than subset sampling -------------------
    connected_counts = pattern_counts_of(graph, connected_space)
    connected_reference = {key: value / float(sum(connected_counts.values()))
                           for (key, value) in connected_counts.items()}
    all_counts = exact_pattern_counts(graph, 4)
    share = {}
    for (key, value) in connected_reference.items():
        share[key] = value / (all_counts.get(key, 0) / float(sum(all_counts.values())))
    assert max(share.values()) > 1.2, \
        "connected-conditioned frequencies differ from unconditional ones on this instance"

    # --- Degenerate inputs ---------------------------------------------------------------
    assert exact_pattern_counts(graph, 0) == {0: 1}, "the empty subgraph is one pattern"
    assert exact_pattern_counts(graph, 1) == {0: graph[0]}
    singleton = (1, [0])
    assert exact_pattern_counts(singleton, 2) == {}
    assert pattern_counts_of(graph, []) == {}
    assert connected_subsets((3, [0, 0, 0]), 2) == []
    assert walk_connected((3, [0, 0, 0]), 2, 100, 5) == ([], [])
    triangle_graph = (3, [6, 5, 3])
    assert walk_connected(triangle_graph, 3, 200, 6)[0] == [frozenset({0, 1, 2})] * 160, \
        "the only connected 3-subset of K3 is K3 itself, minus the 40-step burn-in"


if __name__ == "__main__":
    main()
    print("PASS")
