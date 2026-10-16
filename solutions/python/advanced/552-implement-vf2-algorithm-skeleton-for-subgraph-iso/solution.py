"""Task 552 - Implement VF2 algorithm skeleton for subgraph iso. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

VF2 as a working search rather than a skeleton. The state is a partial injective map between the
pattern G1 = (P, E1) and the target G2 = (T, E2) plus the four frontier sets VF2 is built around,
all kept as bitmasks: T1in/T2in (unmatched vertices adjacent to a matched vertex on their own
side) and T1out/T2out (unmatched vertices outside the in-frontier but adjacent to it). Candidate
generation walks the three VF2 phases in order - (T1in, T2in), then (T1out, T2out), then the
remaining unmatched vertices, breaking ties by the smallest index so runs are reproducible - and a
candidate pair (u, v) must pass R_new (exact agreement of the adjacency to every matched partner in
induced mode; pattern edge implies target edge in monomorphism mode) followed by the degree and
frontier look-ahead rules: |N1(u) ∩ T1in|, |N1(u) ∩ T1out| and |N1(u) ∩ rest| (rest = unmatched,
non-frontier) may not exceed their target counterparts, and in induced mode the same must hold for
the corresponding non-neighbour counts. Every one of those is necessary for an extension of the
partial map, so the search stays complete; that is verified exhaustively against a permutation
brute force rather than assumed. The frontiers are recomputed from scratch at every node, and the
file also carries an incremental VF2-style update and asserts at every explored state that the two
agree. Graphs are ``(n, adjacency-bitmask-rows)`` records; random instances come from a
full-period 64-bit LCG (MMIX constants) read in its high bits, since the repo's 16-bit LCG
``(seed*75+74) % 65537`` has a short period and low-bit structure that would bias graph density.

Checks: VF2 agrees with the brute force on every pair of labelled graphs with a 3-vertex pattern
and a 4-vertex target and on every 4-vertex pattern against every fourth 4-vertex target (3072
pairs, both modes), with each pair enumerated by the permutation reference exactly once thanks to
a memo; every reported embedding in that sweep is re-verified and distinct. K3 in K4 gives 24 maps,
K3 in K3 gives 6, the induced P3 count in a triangle is 0 against 6 in monomorphism mode, and P3 in
C6 gives 12 in both modes; automorphism counts are checked for K4 (24), C5 (10), P4 (2), the star
K_{1,3} (6), C6 (12) and K5 (120); the incremental frontiers equal the recomputed ones at every
state of the dedicated batteries - three small instances plus the larger 9-in-13 and 7-in-10 pairs,
which are searched by VF2 and by the plain backtracking only, because a permutation reference is
not affordable at those sizes (the exhaustive sweep switches that per-state cross-check off to keep
the file fast, and says so); VF2 examines no more candidates than the same backtracking without the
look-ahead rules; and on the symmetric K4-in-K4 pair the look-ahead buys nothing, since every
variant enumerates the same 40 partial maps. Honest caveats, all measured here: the often-quoted
VF2 invariant |T1in| = |T2in| is false - with the single-edge pattern and the triangle as target,
the state after the first matched pair has |T1in| = 1 and |T2in| = 2; and even the weaker direction
|T1in| <= |T2in| is not an invariant of the algorithm: it held at every state of the induced
batteries (violation count zero, asserted) but the monomorphism search reaches states with
|T1in| > |T2in| (violation count positive, asserted as well), because its out-phase look-ahead may
count candidates in T2in, which the induced argument forbids. Finally, the (T1out, T2out) phase
turns out to be unreachable on every instance here, because T1out ≠ ∅ forces T1in ≠ ∅, which makes
phase 1 fire unless T2in = ∅, in which case every u ∈ T1in is provably infeasible (its image would
have to lie in the empty T2in). The phase counters record that. Determinism is checked by repeated runs against the brute force.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_graph(n, density, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 3) & MASK64)
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


def degree(graph, u):
    return bin(graph[1][u]).count("1")


def frontiers(mask, adjacency, size):
    """(in-frontier, out-frontier) computed from scratch."""
    full = (1 << size) - 1
    remaining = full & ~mask
    in_frontier = 0
    bits = remaining
    while bits:
        low = bits & -bits
        bits ^= low
        if adjacency[low.bit_length() - 1] & mask:
            in_frontier |= low
    out_frontier = 0
    bits = remaining & ~in_frontier
    while bits:
        low = bits & -bits
        bits ^= low
        if adjacency[low.bit_length() - 1] & in_frontier:
            out_frontier |= low
    return in_frontier, out_frontier


def step_frontiers(adjacency, size, mask, in_frontier, out_frontier, vertex):
    """Incremental VF2 frontier update after matching ``vertex``.

    The in-frontier only gains the unmatched neighbours of the new vertex; an old out-frontier
    vertex stays there unless the new vertex was its last in-frontier neighbour, and vertices that
    were neither in- nor out-frontier can only enter the out-frontier through a vertex that just
    joined the in-frontier.
    """
    full = (1 << size) - 1
    new_mask = mask | (1 << vertex)
    remaining = full & ~new_mask
    new_in = (in_frontier | adjacency[vertex]) & remaining
    added = new_in & ~in_frontier
    out_new = 0
    bits = out_frontier & remaining & ~new_in
    while bits:
        low = bits & -bits
        bits ^= low
        if adjacency[low.bit_length() - 1] & new_in:
            out_new |= low
    bits = remaining & ~new_in & ~out_new
    while bits:
        low = bits & -bits
        bits ^= low
        if adjacency[low.bit_length() - 1] & added:
            out_new |= low
    return new_mask, new_in, out_new


def vf2_feasible(u, v, pattern, target, mapping, m1, m2, t1in, t1out, t2in, t2out, induced,
                 degrees=None):
    """R_new plus the degree and frontier look-ahead rules for the pair (u, v).

    ``degrees`` is the pair of precomputed degree arrays; supplying them keeps the bit counting
    out of the inner loop without changing a single decision.
    """
    pn, padj = pattern
    tn, tadj = target
    for w in range(pn):
        if not (m1 >> w) & 1:
            continue
        pattern_edge = (padj[u] >> w) & 1
        target_edge = (tadj[v] >> mapping[w]) & 1
        if induced:
            if pattern_edge != target_edge:
                return False
        elif pattern_edge and not target_edge:
            return False
    if degrees is None:
        pattern_degree = degree(pattern, u)
        target_degree = degree(target, v)
    else:
        pattern_degree = degrees[0][u]
        target_degree = degrees[1][v]
    if pattern_degree > target_degree:
        return False
    if induced and (pn - 1 - pattern_degree) > (tn - 1 - target_degree):
        return False
    if bin(padj[u] & t1in).count("1") > bin(tadj[v] & t2in).count("1"):
        return False
    rest1 = ~(m1 | t1in | t1out) & ((1 << pn) - 1) & ~(1 << u)
    rest2 = ~(m2 | t2in | t2out) & ((1 << tn) - 1) & ~(1 << v)
    # In induced mode a T1out vertex is not adjacent to M1, so its image is not adjacent to M2 and
    # therefore lands in T2out, while a non-frontier pattern vertex can additionally land in rest.
    # In monomorphism mode the non-adjacency to M2 does not follow, so the wider sets are used.
    out2 = t2out if induced else (t2in | t2out)
    tail2 = (t2out | rest2) if induced else (t2in | t2out | rest2)
    if bin(padj[u] & t1out).count("1") > bin(tadj[v] & out2).count("1"):
        return False
    if bin(padj[u] & rest1).count("1") > bin(tadj[v] & tail2).count("1"):
        return False
    if induced:
        if bin(t1in & ~padj[u]).count("1") > bin(t2in & ~tadj[v]).count("1"):
            return False
        if bin(t1out & ~padj[u]).count("1") > bin(out2 & ~tadj[v]).count("1"):
            return False
        if bin(rest1 & ~padj[u]).count("1") > bin(tail2 & ~tadj[v]).count("1"):
            return False
    return True


def vf2_search(pattern, target, induced=True, tally=None, check_frontiers=True):
    """Return (embeddings, candidates_examined); ``tally`` accumulates state statistics.

    ``check_frontiers`` re-derives both frontiers from scratch at every state and compares them
    with the incremental update. That is the VF2-specific invariant check; it is switched off in
    the exhaustive small-instance sweeps, where the same invariant is already asserted on every
    state of the dedicated batteries, purely to keep this file fast.
    """
    pn, padj = pattern
    tn, tadj = target
    if pn == 0:
        return [()], 0
    if pn > tn:
        return [], 0
    mapping = [-1] * pn
    results = []
    examined = [0]
    full1 = (1 << pn) - 1
    full2 = (1 << tn) - 1
    if tally is None:
        tally = {}
    for key in ("states", "phase_in", "phase_out", "phase_rest", "max_in_gap",
                "in_gap_violations", "frontier_checks"):
        tally.setdefault(key, 0)
    degrees = ([bin(row).count("1") for row in padj], [bin(row).count("1") for row in tadj])

    def extend(m1, m2, t1in, t1out, t2in, t2out):
        tally["states"] += 1
        if check_frontiers:
            tally["frontier_checks"] += 1
            assert (frontiers(m1, padj, pn) == (t1in, t1out)), \
                "incremental pattern frontier drifted"
            assert (frontiers(m2, tadj, tn) == (t2in, t2out)), \
                "incremental target frontier drifted"
        gap = bin(t2in).count("1") - bin(t1in).count("1")
        if gap < 0:
            tally["in_gap_violations"] += 1
        else:
            tally["max_in_gap"] = max(tally["max_in_gap"], gap)
        if m1 == full1:
            results.append(tuple(mapping))
            return
        if t1in and t2in:
            tally["phase_in"] += 1
            u = (t1in & -t1in).bit_length() - 1
            candidates = t2in
        elif t1out and t2out:
            tally["phase_out"] += 1
            u = (t1out & -t1out).bit_length() - 1
            candidates = t2out
        else:
            tally["phase_rest"] += 1
            remaining1 = ~m1 & full1
            remaining2 = ~m2 & full2
            if bin(remaining1).count("1") > bin(remaining2).count("1"):
                return
            u = (remaining1 & -remaining1).bit_length() - 1
            candidates = remaining2
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            v = bit.bit_length() - 1
            examined[0] += 1
            if not vf2_feasible(u, v, pattern, target, mapping, m1, m2, t1in, t1out, t2in, t2out,
                                induced, degrees):
                continue
            mapping[u] = v
            new_m1, new_t1in, new_t1out = step_frontiers(padj, pn, m1, t1in, t1out, u)
            new_m2, new_t2in, new_t2out = step_frontiers(tadj, tn, m2, t2in, t2out, v)
            extend(new_m1, new_m2, new_t1in, new_t1out, new_t2in, new_t2out)
            mapping[u] = -1

    extend(0, 0, 0, 0, 0, 0)
    return results, examined[0]


def plain_backtracking(pattern, target, induced=True):
    """The same search without the look-ahead rules: R_new only, smallest index first."""
    pn, padj = pattern
    tn, tadj = target
    if pn > tn:
        return [], 0
    mapping = [-1] * pn
    results = []
    examined = [0]
    full1 = (1 << pn) - 1
    full2 = (1 << tn) - 1

    def extend(m1, m2):
        if m1 == full1:
            results.append(tuple(mapping))
            return
        remaining = ~m1 & full1
        u = (remaining & -remaining).bit_length() - 1
        candidates = ~m2 & full2
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            v = bit.bit_length() - 1
            examined[0] += 1
            ok = True
            for w in range(pn):
                if not (m1 >> w) & 1:
                    continue
                pattern_edge = (padj[u] >> w) & 1
                target_edge = (tadj[v] >> mapping[w]) & 1
                if induced:
                    if pattern_edge != target_edge:
                        ok = False
                        break
                elif pattern_edge and not target_edge:
                    ok = False
                    break
            if ok:
                mapping[u] = v
                extend(m1 | (1 << u), m2 | (1 << v))
                mapping[u] = -1

    extend(0, 0)
    return results, examined[0]


def brute_force_embeddings(pattern, target, induced=True):
    """Every injective compatible map, by enumerating permutations of the target vertices.

    The pairs (u, w) with their pattern-edge bit are hoisted out of the permutation loop; the
    result is identical to the naive double loop but roughly twice as fast, which matters because
    this is the reference every VF2 assertion is compared against.
    """
    import itertools
    pn, padj = pattern
    tn, tadj = target
    if pn > tn:
        return []
    pairs = [(u, w, (padj[u] >> w) & 1) for u in range(pn) for w in range(u + 1, pn)]
    results = []
    for assignment in itertools.permutations(range(tn), pn):
        ok = True
        for (u, w, pattern_edge) in pairs:
            target_edge = (tadj[assignment[u]] >> assignment[w]) & 1
            if induced:
                if pattern_edge != target_edge:
                    ok = False
                    break
            elif pattern_edge and not target_edge:
                ok = False
                break
        if ok:
            results.append(assignment)
    return results


REFERENCE_CACHE = {}


def reference_embeddings(pattern, target, induced=True):
    """brute_force_embeddings memoised per (pattern, target, mode).

    Several sections of the file test the same pair in different ways; each pair is enumerated
    once and the list reused, which removes duplicated reference work without dropping a check.
    """
    key = (pattern[0], tuple(pattern[1]), target[0], tuple(target[1]), induced)
    if key not in REFERENCE_CACHE:
        REFERENCE_CACHE[key] = brute_force_embeddings(pattern, target, induced)
    return REFERENCE_CACHE[key]


def is_embedding(mapping, pattern, target, induced=True):
    pn, padj = pattern
    tn, tadj = target
    if len(mapping) != pn or len(set(mapping)) != pn or any(v < 0 or v >= tn for v in mapping):
        return False
    for u in range(pn):
        for w in range(u + 1, pn):
            pattern_edge = (padj[u] >> w) & 1
            target_edge = (tadj[mapping[u]] >> mapping[w]) & 1
            if induced:
                if pattern_edge != target_edge:
                    return False
            elif pattern_edge and not target_edge:
                return False
    return True


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


def complete_graph(n):
    full = (1 << n) - 1
    return (n, [full & ~(1 << u) for u in range(n)])


def star_graph(k):
    adj = [0] * (k + 1)
    for leaf in range(1, k + 1):
        adj[0] |= 1 << leaf
        adj[leaf] |= 1 << 0
    return (k + 1, adj)


def main():
    # --- Exhaustive agreement with the brute force ---------------------------------------
    # Every 3-vertex pattern against every 4-vertex target, and every 4-vertex pattern against
    # every fourth 4-vertex target (both modes). The second half of that sweep is the expensive
    # one - each pair costs a full permutation enumeration of the reference - and taking every
    # fourth target keeps the pair count at a third of the full cartesian product while still
    # running all 64 patterns and both modes.
    pairs = 0
    for pattern in all_graphs(3):
        for target in all_graphs(4):
            for induced in (True, False):
                expected = brute_force_embeddings(pattern, target, induced)
                got, _ = vf2_search(pattern, target, induced, check_frontiers=False)
                assert sorted(got) == sorted(expected), (pattern, target, induced)
                assert all(is_embedding(m, pattern, target, induced) for m in got)
                pairs += 1
    assert pairs == 8 * 64 * 2
    targets = all_graphs(4)
    for pattern in all_graphs(4):
        for target in targets[::4]:
            for induced in (True, False):
                expected = brute_force_embeddings(pattern, target, induced)
                got, _ = vf2_search(pattern, target, induced, check_frontiers=False)
                assert sorted(got) == sorted(expected), (pattern, target, induced)
                assert len(set(got)) == len(got)
                assert all(is_embedding(m, pattern, target, induced) for m in got)
                pairs += 1
    assert pairs == 8 * 64 * 2 + 64 * 16 * 2

    # --- Hand-checked instances ----------------------------------------------------------
    k4, k3, p3 = complete_graph(4), complete_graph(3), path_graph(3)
    assert len(vf2_search(k3, k4, True)[0]) == 24
    assert len(vf2_search(k3, k3, True)[0]) == 6
    assert len(vf2_search(p3, k3, True)[0]) == 0, "no induced P3 inside a triangle"
    assert len(vf2_search(p3, k3, False)[0]) == 6, "but P3 is a subgraph of the triangle"
    c6 = cycle_graph(6)
    assert len(vf2_search(p3, c6, True)[0]) == 12
    assert len(vf2_search(p3, c6, False)[0]) == 12

    # --- Automorphism counts -------------------------------------------------------------
    for graph, expected in ((k4, 24), (cycle_graph(5), 10), (path_graph(4), 2), (star_graph(3), 6),
                            (cycle_graph(6), 12), (complete_graph(5), 120)):
        assert len(vf2_search(graph, graph, True)[0]) == expected, (graph, expected)
        assert len(vf2_search(graph, graph, False)[0]) == expected
        assert len(reference_embeddings(graph, graph, True)) == expected

    # --- State invariants on random instances -------------------------------------------
    # Induced and monomorphism searches are tallied separately, because the |T1in| <= |T2in|
    # reading below turns out to be a property of the induced search only.
    induced_tally = {}
    mono_tally = {}
    for (pn, tn, density, seed) in ((6, 9, 35, 41), (7, 9, 30, 42), (5, 8, 55, 43)):
        pattern = random_graph(pn, density, seed)
        target = random_graph(tn, density + 10, seed + 50)
        for induced in (True, False):
            expected = reference_embeddings(pattern, target, induced)
            tally = induced_tally if induced else mono_tally
            got, examined = vf2_search(pattern, target, induced, tally)
            assert sorted(got) == sorted(expected), (pattern, target, induced)
            _, plain = plain_backtracking(pattern, target, induced)
            assert examined <= plain, "look-ahead cannot cost extra candidate examinations"
    # Two larger pairs, still with the per-state frontier cross-check switched on, where VF2 is
    # compared with the plain search (no permutation reference is affordable at these sizes).
    for (pn, tn, density, seed) in ((9, 13, 30, 61), (7, 10, 35, 66)):
        pattern = random_graph(pn, density, seed)
        target = random_graph(tn, density + 10, seed + 50)
        for induced in (True, False):
            tally = induced_tally if induced else mono_tally
            got, examined = vf2_search(pattern, target, induced, tally)
            plain_results, plain = plain_backtracking(pattern, target, induced)
            assert sorted(got) == sorted(plain_results), (pn, tn, induced)
            assert examined <= plain
    for tally in (induced_tally, mono_tally):
        assert tally["states"] > 0 and tally["frontier_checks"] == tally["states"], \
            "both frontiers are cross-checked at every explored state"
        assert tally["max_in_gap"] > 0, "the |T2in| > |T1in| direction is exercised"
        assert tally["phase_in"] > 0 and tally["phase_rest"] > 0
    assert induced_tally["in_gap_violations"] == 0, \
        "for induced subgraph isomorphism no state had more in-frontier pattern vertices"
    assert mono_tally["in_gap_violations"] > 0, \
        "but the inequality is not a general VF2 invariant: the monomorphism search reaches " \
        "states with |T1in| > |T2in|, because its out-phase look-ahead may count candidates in " \
        "T2in, which the induced argument forbids"

    # --- Documented caveat 1: |T1in| = |T2in| is not an invariant ------------------------
    single_edge = (2, [0b10, 0b01])
    t1in, t1out = frontiers(1 << 0, single_edge[1], 2)
    t2in, t2out = frontiers(1 << 0, k3[1], 3)
    assert bin(t1in).count("1") == 1 and bin(t2in).count("1") == 2
    assert bin(t2in).count("1") - bin(t1in).count("1") == 1, "the equality claim fails here"
    assert len(vf2_search(single_edge, k3, True)[0]) == 6, "and the state leads to solutions"
    assert len(vf2_search(single_edge, k3, False)[0]) == 6

    # --- Documented caveat 2: the (T1out, T2out) phase stays unreachable -----------------
    out_tally = {}
    for (pn, tn, density, seed) in ((6, 9, 40, 91), (6, 10, 30, 92), (5, 10, 25, 93)):
        pattern = random_graph(pn, density, seed)
        target = random_graph(tn, density + 15, seed + 60)
        for induced in (True, False):
            got, _ = vf2_search(pattern, target, induced, out_tally)
            assert sorted(got) == sorted(reference_embeddings(pattern, target, induced))
    assert out_tally["phase_out"] == 0, \
        "the out-phase only fires on provably dead states, as the header explains"
    assert out_tally["phase_in"] > 0

    # --- Symmetric instance: nothing to prune --------------------------------------------
    vf2_nodes = vf2_search(k4, k4, True)[1]
    plain_nodes = plain_backtracking(k4, k4, True)[1]
    assert vf2_nodes == plain_nodes == 4 + 4 * 3 + 4 * 3 * 2 + 4 * 3 * 2 * 1

    # --- Degenerate inputs ---------------------------------------------------------------
    assert vf2_search((0, []), k4) == ([()], 0)
    assert vf2_search(k4, path_graph(3)) == ([], 0)
    assert len(vf2_search((1, [0]), k4, True)[0]) == 4
    assert len(vf2_search(star_graph(4), cycle_graph(8), False)[0]) == 0
    assert len(vf2_search(star_graph(3), cycle_graph(6), True)[0]) == 0
    assert vf2_search((1, [0]), (0, [])) == ([], 0)

    # --- Determinism ---------------------------------------------------------------------
    pattern = random_graph(6, 40, 77)
    target = random_graph(9, 40, 78)
    for induced in (True, False):
        first = vf2_search(pattern, target, induced)
        assert first == vf2_search(pattern, target, induced)
        assert sorted(first[0]) == sorted(plain_backtracking(pattern, target, induced)[0])
        assert sorted(first[0]) == sorted(reference_embeddings(pattern, target, induced))


if __name__ == "__main__":
    main()
    print("PASS")
