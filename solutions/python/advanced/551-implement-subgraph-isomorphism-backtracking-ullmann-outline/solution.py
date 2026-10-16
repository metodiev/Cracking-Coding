"""Task 551 - Implement subgraph isomorphism backtracking (Ullmann outline). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Subgraph isomorphism by backtracking in the shape of Ullmann's algorithm. Each graph is a record
``(n, adj)`` where ``adj[u]`` is a bitmask of the neighbours of ``u``, and the candidate matrix is
kept as one bitmask per pattern vertex. Before searching, the candidate matrix is refined to a
fixed point: a pair ``(u, v)`` is discarded when some other pattern vertex ``w`` has no remaining
candidate compatible with ``v`` under the graph relation - edge -> edge for a monomorphism
(non-induced subgraph isomorphism), and edge <-> edge, i.e. also non-edge -> non-edge, for an
induced subgraph isomorphism. Refinement is a necessary-condition filter: it can remove pairs no
embedding uses, but never a pair an embedding does use; the file validates that exhaustively
instead of trusting it. The search fixes a static order (fewest surviving candidates first, ties by
vertex index), tries candidates in increasing target index, checks each assignment against the
already assigned pattern vertices, then runs forward checking so that every unmatched pattern
vertex keeps at least one compatible, unused candidate.

Random instances come from a full-period 64-bit LCG (MMIX constants) read in its high bits; the
repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period and its low bits track the graph
density, which would make the "random" pattern/target pairs here systematically similar.

Checks: on every pair of labelled graphs with a 4-vertex pattern and a 4-vertex target (4096 pairs)
and on every pair with a 3-vertex pattern and a 4-vertex target, in both induced and monomorphism
mode, the embedding count equals a permutation brute force, and every returned mapping is
independently re-verified (injective, relation preserving, and in induced mode relation reflecting)
and distinct; K3 in K4 gives all 24 injective maps while the triangle in a triangle gives 6; the
induced count of P3 in a triangle is 0 against 6 for the monomorphism count, the standard
induced-versus-monomorphism discriminator; P3 in C6 is 12 in both modes (6 centres, 2 neighbour
choices, 1 remaining neighbour - C6 is triangle-free, so the induced and subgraph counts coincide)
and in P6 it is 8, and induced K3 in C6 is 0; automorphism counts come out as 4! = 24 for K4, 10 for C5, 2 for P4, 6 for the star
K_{1,3}, 12 for C6 and 24 for the star K_{1,4}; degenerate inputs (empty pattern, one-vertex
pattern, pattern larger than the target, a target whose degrees cannot host the pattern) are pinned
down; the refined matrix is a fixed point of ``refine``, stays inside the raw degree domain, never
grows, and every vertex of every brute-force embedding survives it; and the node counts show
refinement plus forward checking exploring no more nodes than plain domain backtracking while on
the fully symmetric K4-in-K4 pair every variant explores exactly the same 40 partial maps - the
honest caveat that refinement does not change the worst case. Repeated runs are identical.
"""
import itertools

MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_graph(n, density, seed):
    """Each of the C(n, 2) vertex pairs is present with probability density/100."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 1) & MASK64)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            state = prng64(state)
            if (state >> 33) % 100 < density:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return (n, adj)


def all_graphs(n):
    """Every labelled loop-free graph on n vertices, in a fixed order."""
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


def initial_domain(pattern, target, induced=True):
    """Per-pattern-vertex candidate bitmask after the degree tests only."""
    pn, _ = pattern
    tn, _ = target
    domain = []
    for u in range(pn):
        mask = 0
        for v in range(tn):
            if degree(pattern, u) > degree(target, v):
                continue
            if induced and (pn - 1 - degree(pattern, u)) > (tn - 1 - degree(target, v)):
                continue
            mask |= 1 << v
        domain.append(mask)
    return domain


def refine(pattern, target, induced=True, domain=None):
    """Ullmann refinement, iterated to a fixed point.

    A pair (u, v) survives only if every other pattern vertex w has at least one candidate left
    that is compatible with v: adjacent to v when u ~ w, and (induced mode) non-adjacent to v when
    u is not adjacent to w. Each single condition is necessary for an injective
    relation-preserving map, so no embedding is lost.
    """
    pn, padj = pattern
    tn, tadj = target
    if domain is None:
        domain = initial_domain(pattern, target, induced)
    else:
        domain = list(domain)
    full = (1 << tn) - 1
    changed = True
    while changed:
        changed = False
        for u in range(pn):
            mask = domain[u]
            for v in range(tn):
                bit = 1 << v
                if not mask & bit:
                    continue
                keep = True
                for w in range(pn):
                    if w == u:
                        continue
                    others = domain[w] & ~bit & full
                    if (padj[u] >> w) & 1:
                        if not others & tadj[v]:
                            keep = False
                            break
                    elif induced and not others & ~tadj[v] & full:
                        keep = False
                        break
                if not keep:
                    mask &= ~bit
                    changed = True
            domain[u] = mask
    return domain


def ullmann_search(pattern, target, induced=True, refinement=True, forward_checking=True):
    """Return (embeddings, nodes); an embedding is a pattern-index -> target-index tuple."""
    pn, padj = pattern
    tn, tadj = target
    if pn == 0:
        return [()], 0
    if pn > tn:
        return [], 0
    domain = refine(pattern, target, induced) if refinement else \
        initial_domain(pattern, target, induced)
    if any(mask == 0 for mask in domain):
        return [], 0
    order = sorted(range(pn), key=lambda u: (bin(domain[u]).count("1"), u))
    mapping = [-1] * pn
    results = []
    nodes = [0]
    full = (1 << tn) - 1

    def compatible(u, v):
        for w in range(pn):
            vw = mapping[w]
            if w == u or vw < 0:
                continue
            pattern_edge = (padj[u] >> w) & 1
            target_edge = (tadj[v] >> vw) & 1
            if induced:
                if pattern_edge != target_edge:
                    return False
            elif pattern_edge and not target_edge:
                return False
        return True

    def frontier_ok(step, used):
        for index in range(step, pn):
            w = order[index]
            candidates = domain[w] & ~used & full
            while candidates:
                bit = candidates & -candidates
                candidates ^= bit
                mapping[w] = bit.bit_length() - 1
                if compatible(w, mapping[w]):
                    break
                mapping[w] = -1
            else:
                mapping[w] = -1
                return False
            mapping[w] = -1
        return True

    def extend(step, used):
        u = order[step]
        candidates = domain[u] & ~used & full
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            mapping[u] = bit.bit_length() - 1
            nodes[0] += 1
            if compatible(u, mapping[u]) and (not forward_checking or step + 1 == pn
                                             or frontier_ok(step + 1, used | bit)):
                if step + 1 == pn:
                    results.append(tuple(mapping))
                else:
                    extend(step + 1, used | bit)
            mapping[u] = -1

    extend(0, 0)
    return results, nodes[0]


def brute_force_embeddings(pattern, target, induced=True):
    """All injective compatible maps, by enumerating every injection explicitly."""
    pn, padj = pattern
    tn, tadj = target
    if pn > tn:
        return []
    results = []
    for assignment in itertools.permutations(range(tn), pn):
        ok = True
        for u in range(pn):
            for w in range(u + 1, pn):
                pattern_edge = (padj[u] >> w) & 1
                target_edge = (tadj[assignment[u]] >> assignment[w]) & 1
                if induced:
                    if pattern_edge != target_edge:
                        ok = False
                        break
                elif pattern_edge and not target_edge:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            results.append(assignment)
    return results


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
    # --- Exhaustive agreement with the brute force, induced and monomorphism modes --------
    pairs_tested = 0
    for pattern in all_graphs(3):
        for target in all_graphs(4):
            for induced in (True, False):
                expected = brute_force_embeddings(pattern, target, induced)
                got, nodes = ullmann_search(pattern, target, induced)
                assert sorted(got) == sorted(expected), (pattern, target, induced)
                pairs_tested += 1
    for pattern in all_graphs(4):
        for target in all_graphs(4):
            for induced in (True, False):
                expected = brute_force_embeddings(pattern, target, induced)
                got, _ = ullmann_search(pattern, target, induced)
                assert sorted(got) == sorted(expected), (pattern, target, induced)
                assert all(is_embedding(m, pattern, target, induced) for m in got)
                assert len(set(got)) == len(got), "each embedding is reported once"
                pairs_tested += 1
    assert pairs_tested == 8 * 64 * 2 + 64 * 64 * 2 == 9216

    # --- Hand-checked instances ----------------------------------------------------------
    k4, k3, p3 = complete_graph(4), complete_graph(3), path_graph(3)
    assert len(ullmann_search(k3, k4, induced=True)[0]) == 24
    assert len(brute_force_embeddings(k3, k4, induced=True)) == 24
    assert len(ullmann_search(k3, k3, induced=True)[0]) == 6

    assert len(ullmann_search(p3, k3, induced=True)[0]) == 0, "a triangle has no induced P3"
    assert len(ullmann_search(p3, k3, induced=False)[0]) == 6, "but it contains P3 as a subgraph"
    assert len(brute_force_embeddings(p3, k3, induced=True)) == 0
    assert len(brute_force_embeddings(p3, k3, induced=False)) == 6

    c6 = cycle_graph(6)
    # P3 in C6: 6 centres * 2 neighbour choices * 1 remaining neighbour = 12. Both modes agree
    # because C6 has no triangle, so no candidate third vertex is adjacent to the first one.
    assert len(ullmann_search(p3, c6, induced=True)[0]) == 12
    assert len(ullmann_search(p3, c6, induced=False)[0]) == 12
    # P3 in the 6-vertex path: the 4 non-end vertices are centres with exactly 2 neighbours,
    # each giving one remaining choice - 4 * 2 = 8.
    assert len(ullmann_search(p3, path_graph(6), induced=True)[0]) == 8
    assert len(ullmann_search(p3, path_graph(6), induced=False)[0]) == 8
    assert len(ullmann_search(k3, c6, induced=True)[0]) == 0
    assert len(ullmann_search(k3, path_graph(5), induced=False)[0]) == 0
    assert len(ullmann_search(k3, complete_graph(5), induced=True)[0]) == 5 * 4 * 3

    # --- Automorphism counts against the textbook values ---------------------------------
    for graph, expected in ((k4, 24), (cycle_graph(5), 10), (path_graph(4), 2), (star_graph(3), 6),
                            (cycle_graph(6), 12), (star_graph(4), 24), (complete_graph(5), 120)):
        assert len(ullmann_search(graph, graph, induced=True)[0]) == expected, (graph, expected)
        assert len(ullmann_search(graph, graph, induced=False)[0]) == expected
        assert len(brute_force_embeddings(graph, graph, induced=True)) == expected

    # --- Degenerate inputs ---------------------------------------------------------------
    empty = (0, [])
    assert ullmann_search(empty, k4) == ([()], 0)
    assert brute_force_embeddings(empty, k4) == [()]
    assert len(ullmann_search((1, [0]), k4)[0]) == 4
    assert ullmann_search(k4, path_graph(3)) == ([], 0), "pattern larger than the target"
    assert brute_force_embeddings(k4, path_graph(3)) == []
    assert len(ullmann_search(star_graph(3), c6, induced=True)[0]) == 0
    assert len(ullmann_search(star_graph(3), c6, induced=False)[0]) == 0
    assert len(brute_force_embeddings(star_graph(3), c6, induced=False)) == 0

    # --- Refinement is a valid, necessary-only filter ------------------------------------
    for (pn, tn, density, seed) in ((6, 9, 35, 11), (5, 8, 50, 12), (7, 10, 30, 13)):
        pattern = random_graph(pn, density, seed)
        target = random_graph(tn, density + 10, seed + 100)
        raw = initial_domain(pattern, target)
        once = refine(pattern, target, True, raw)
        twice = refine(pattern, target, True, once)
        assert once == twice, "the refinement is a fixed point of itself"
        for u in range(pn):
            assert once[u] & ~raw[u] == 0, "refinement only removes candidates"
        expected = brute_force_embeddings(pattern, target, True)
        got, _ = ullmann_search(pattern, target, True, refinement=True)
        assert sorted(got) == sorted(expected), "refinement never loses an embedding"
        for m in got:
            for u in range(pn):
                assert once[u] >> m[u] & 1, "a pair used by an embedding was refined away"
    assert refine(k4, k4, True) == initial_domain(k4, k4, True), \
        "nothing to remove on the complete pair"

    # --- Node counts: pruning helps, the worst case does not shrink ----------------------
    pattern = random_graph(6, 30, 21)
    target = random_graph(10, 35, 22)
    _, refined_nodes = ullmann_search(pattern, target, True, refinement=True)
    _, plain_nodes = ullmann_search(pattern, target, True, refinement=False)
    _, degree_only = ullmann_search(pattern, target, True, refinement=False, forward_checking=False)
    assert refined_nodes <= plain_nodes and plain_nodes <= degree_only
    assert refined_nodes < degree_only, "pruning must actually prune on this instance"
    k4_nodes = ullmann_search(k4, k4, True, refinement=True)[1]
    assert k4_nodes == ullmann_search(k4, k4, True, refinement=False, forward_checking=False)[1]
    assert k4_nodes == 4 + 4 * 3 + 4 * 3 * 2 + 4 * 3 * 2 * 1, \
        "K4 in K4 explores 4P3 = 40 partial maps however it is pruned"

    # --- Determinism --------------------------------------------------------------------
    for (pn, tn, seed) in ((5, 9, 31), (6, 9, 32)):
        pattern = random_graph(pn, 40, seed)
        target = random_graph(tn, 40, seed + 7)
        for induced in (True, False):
            first = ullmann_search(pattern, target, induced)
            assert first == ullmann_search(pattern, target, induced)
            assert sorted(first[0]) == sorted(brute_force_embeddings(pattern, target, induced))


if __name__ == "__main__":
    main()
    print("PASS")
