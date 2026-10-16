"""Task 542 - Implement tree isomorphism hashing (AHU algorithm). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Canonical forms for trees, following Aho-Hopcroft-Ullman. A rooted tree is encoded by
``canonical_rooted``: the neighbours are ordered by parent pointers, the children's codes are
collected bottom-up and sorted before being wrapped in parentheses, so two rooted trees get the
same string exactly when they are isomorphic as rooted trees - the sorted multiset of child codes
is a complete invariant at every node. ``canonical_ids`` computes the same thing as nested integer
tuples (a completely different representation, compared against the string version), and
``canonical_unrooted`` roots the tree at its centre (one centre, or two joined by an edge, found by
iterative leaf peeling) and takes the lexicographically smaller of the resulting codes, which is a
complete invariant for free trees.

The reference implementation is a brute-force isomorphism search - every bijection of the vertex
sets is tried, with the root image fixed for the rooted case - so the canonical verdict is compared
against an definition-level search rather than against itself, and the AHU trick is shown to be a
strictly cheaper way of deciding the same predicate.

Checks: hand-built examples (P3 rooted at its centre versus at an end shares the rooted-iso
verdict "false" but the unrooted one "true"; the star and the path; and the documented pair of
seven-vertex trees with the same degree sequence {3, 2, 2, 2, 1, 1, 1} - a spider with three legs
of length two versus one with legs 3, 2, 1 - which the canonical form distinguishes even though a
degree-sequence fingerprint cannot); all 1296 labelled trees on six vertices generated from Prüfer
sequences collapse to exactly the six known free trees on six vertices (and the 16807 sequences on
seven vertices to the eleven known ones); Prüfer encode/decode round trips; canonical forms
invariant under relabelling and under shuffling the adjacency lists; brute-force agreement on all 100
pairs of ten random small trees; centre detection on paths, stars and the two-centre path P4; and
determinism of repeated runs.
"""
import heapq
import itertools


def neighbours(adj, v):
    return list(adj[v])


def parent_order(adj, root):
    """BFS/DFS order from ``root`` with parents; used by the bottom-up encoders."""
    parent = {root: -1}
    order = [root]
    stack = [root]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w != parent[v] and w not in parent:
                parent[w] = v
                order.append(w)
                stack.append(w)
    assert len(order) == len(adj), "the input must be a single connected tree"
    return order, parent


def canonical_rooted(adj, root):
    """AHU string code of the tree rooted at ``root``."""
    order, parent = parent_order(adj, root)
    code = {}
    for v in reversed(order):
        children = sorted(code[w] for w in adj[v] if w != parent[v])
        code[v] = "(" + "".join(children) + ")"
    return code[root]


def canonical_ids(adj, root):
    """Independent encoding: nested tuples of integers, depth by depth."""
    order, parent = parent_order(adj, root)
    code = {}
    for v in reversed(order):
        code[v] = tuple(sorted(code[w] for w in adj[v] if w != parent[v]))
    return code[root]


def ids_to_text(ids):
    """Textual form of the nested-tuple encoding, so the two encodings can be compared."""
    return "(" + "".join(ids_to_text(child) for child in ids) + ")"


def tree_centres(adj):
    """One or two centres, found by peeling leaves until at most two vertices remain."""
    n = len(adj)
    if n <= 2:
        return list(range(n))
    degree = [len(adj[v]) for v in range(n)]
    leaves = [v for v in range(n) if degree[v] <= 1]
    remaining = n
    while remaining > 2:
        new_leaves = []
        for leaf in leaves:
            remaining -= 1
            for w in adj[leaf]:
                degree[w] -= 1
                if degree[w] == 1:
                    new_leaves.append(w)
        leaves = new_leaves
    return sorted(leaves)


def canonical_unrooted(adj):
    return min(canonical_rooted(adj, centre) for centre in tree_centres(adj))


def canonical_unrooted_ids(adj):
    return min(ids_to_text(canonical_ids(adj, centre)) for centre in tree_centres(adj))


def edges_of(adj):
    return [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]


def rooted_isomorphic_brute(adj1, root1, adj2, root2):
    """Exhaustive rooted isomorphism: root1 must map to root2, everything else is permuted."""
    n = len(adj1)
    if len(adj2) != n:
        return False
    targets = [v for v in range(n) if v != root2]
    sources = [v for v in range(n) if v != root1]
    edges = edges_of(adj1)
    for assignment in itertools.permutations(targets):
        mapping = {root1: root2}
        for (source, target) in zip(sources, assignment):
            mapping[source] = target
        if all(mapping[v] in adj2[mapping[u]] for (u, v) in edges):
            return True
    return False


def isomorphic_brute(adj1, adj2):
    """Exhaustive unrooted isomorphism by trying every choice of roots."""
    if len(adj1) != len(adj2):
        return False
    for root1 in range(len(adj1)):
        for root2 in range(len(adj2)):
            if rooted_isomorphic_brute(adj1, root1, adj2, root2):
                return True
    return False


def prufer_to_tree(sequence, n):
    """Decode a Prüfer sequence into an adjacency list (n >= 2 for a non-empty sequence)."""
    if n == 1:
        return [[]]
    degree = [1] * n
    for value in sequence:
        degree[value] += 1
    adj = [[] for _ in range(n)]
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)
    for value in sequence:
        leaf = heapq.heappop(leaves)
        adj[leaf].append(value)
        adj[value].append(leaf)
        degree[value] -= 1
        if degree[value] == 1:
            heapq.heappush(leaves, value)
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adj[u].append(v)
    adj[v].append(u)
    return adj


def tree_to_prufer(adj):
    """Encode a labelled tree with at least two vertices into its Prüfer sequence."""
    n = len(adj)
    assert n >= 2
    degree = [len(adj[v]) for v in range(n)]
    leaves = [v for v in range(n) if degree[v] == 1]
    heapq.heapify(leaves)
    sequence = []
    removed = 0
    while removed < n - 2:
        leaf = heapq.heappop(leaves)
        for w in adj[leaf]:
            if degree[w] > 0:
                neighbour = w
                break
        else:
            raise AssertionError("a leaf must have a live neighbour")
        sequence.append(neighbour)
        degree[leaf] = 0
        degree[neighbour] -= 1
        if degree[neighbour] == 1:
            heapq.heappush(leaves, neighbour)
        removed += 1
    return sequence


def relabel(adj, permutation):
    """Relabel the tree: vertex i becomes permutation[i]."""
    n = len(adj)
    out = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            out[permutation[u]].append(permutation[v])
    return out


def lcg_permutation(n, seed):
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        seed = (seed * 75 + 74) % 65537
        j = seed % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm, seed


def lcg_tree(n, seed):
    sequence = []
    for _ in range(max(0, n - 2)):
        seed = (seed * 75 + 74) % 65537
        sequence.append(seed % n)
    return prufer_to_tree(sequence, n), seed


def spider(legs):
    """Tree: a hub with one path of the given length hanging from it."""
    adj = [[]]
    for length in legs:
        previous = 0
        for _ in range(length):
            new = len(adj)
            adj.append([])
            adj[previous].append(new)
            adj[new].append(previous)
            previous = new
    return adj


def degrees_sorted(adj):
    return sorted(len(adj[v]) for v in range(len(adj)))


def main():
    # --- Rooted versus unrooted ---------------------------------------------------------
    path3 = [[1], [0, 2], [1]]
    assert canonical_rooted(path3, 0) != canonical_rooted(path3, 1), \
        "P3 rooted at an end differs from P3 rooted at the centre"
    assert canonical_unrooted(path3) == canonical_rooted(path3, 1), "the centre is used"
    assert canonical_unrooted_ids(path3) == ids_to_text(canonical_ids(path3, 1)), \
        "the tuple encoding picks the same centre"
    assert isomorphic_brute(path3, path3)
    assert rooted_isomorphic_brute(path3, 0, path3, 2), "the two ends are interchangeable"
    assert not rooted_isomorphic_brute(path3, 0, path3, 1)

    # --- Known small families -----------------------------------------------------------
    star4 = [[1, 2, 3], [0], [0], [0]]
    path4 = [[1], [0, 2], [1, 3], [2]]
    assert canonical_unrooted(star4) != canonical_unrooted(path4)
    assert not isomorphic_brute(star4, path4)
    assert canonical_unrooted(star4) == canonical_rooted(star4, 0), "the star has one centre"
    assert tree_centres(path4) == [1, 2], "P4 has two centres joined by an edge"
    assert len(tree_centres(path3)) == 1 and tree_centres([[1], [0]]) == [0, 1]

    # --- Same degree sequence, different trees -----------------------------------------
    spider222 = spider([2, 2, 2])          # three legs of length two: 7 vertices
    spider321 = spider([3, 2, 1])          # legs of length three, two and one: 7 vertices
    assert degrees_sorted(spider222) == degrees_sorted(spider321) == [1, 1, 1, 2, 2, 2, 3], \
        "the degree sequence alone cannot tell these apart"
    assert canonical_unrooted(spider222) != canonical_unrooted(spider321), \
        "AHU separates them"
    assert not isomorphic_brute(spider222, spider321)
    assert len(tree_centres(spider222)) == 1 and tree_centres(spider222) == [0]
    assert isomorphic_brute(spider222, relabel(spider222, [0, 2, 1, 3, 5, 4, 6]))

    # --- Prüfer round trips -------------------------------------------------------------
    seed = 123456
    for n in range(2, 10):
        for _ in range(5):
            tree, seed = lcg_tree(n, seed)
            sequence = tree_to_prufer(tree)
            assert len(sequence) == n - 2
            rebuilt = prufer_to_tree(sequence, n)
            assert canonical_unrooted(rebuilt) == canonical_unrooted(tree), "round trip"

    # --- Invariance under relabelling and neighbour order ------------------------------
    seed = 8191
    for n in range(3, 12):
        tree, seed = lcg_tree(n, seed)
        reference = canonical_unrooted(tree)
        reference_ids = canonical_unrooted_ids(tree)
        for _ in range(3):
            permutation, seed = lcg_permutation(n, seed)
            moved = relabel(tree, permutation)
            assert canonical_unrooted(moved) == reference, "relabelling cannot change the code"
            assert canonical_unrooted_ids(moved) == reference_ids, \
                "the tuple encoding is invariant too"
            shuffled = [list(reversed(adj)) for adj in moved]
            assert canonical_unrooted(shuffled) == reference, "adjacency order is irrelevant"
            assert canonical_unrooted(shuffled) == canonical_unrooted(moved)

    # --- Agreement with the exhaustive search ------------------------------------------
    seed = 24680
    small = []
    for n in range(2, 7):
        for _ in range(2):
            tree, seed = lcg_tree(n, seed)
            small.append(tree)
    comparisons = 0
    for i in range(len(small)):
        for j in range(len(small)):
            a, b = small[i], small[j]
            same_canonical = len(a) == len(b) and canonical_unrooted(a) == canonical_unrooted(b)
            assert same_canonical == isomorphic_brute(a, b), \
                "the canonical form must decide exactly what brute force decides"
            assert same_canonical == (canonical_unrooted_ids(a) == canonical_unrooted_ids(b))
            comparisons += 1
    assert comparisons == 100

    # --- All trees on six and seven vertices -------------------------------------------
    classes_six = {}
    for sequence in itertools.product(range(6), repeat=4):
        code = canonical_unrooted(prufer_to_tree(list(sequence), 6))
        classes_six[code] = classes_six.get(code, 0) + 1
    assert len(classes_six) == 6, "there are exactly six free trees on six vertices"
    assert sum(classes_six.values()) == 6 ** 4, "every labelled tree was generated once"
    # The class multiplicities are the orbit sizes 6! / |Aut(T)|: the star K1,5 has 6 labelled
    # copies, the path P6 and two others have automorphism group of order 2 (360 each), and the
    # remaining two trees contribute 90 and 120.
    assert sorted(classes_six.values()) == [6, 90, 120, 360, 360, 360]
    classes_seven = set()
    for sequence in itertools.product(range(7), repeat=5):
        classes_seven.add(canonical_unrooted(prufer_to_tree(list(sequence), 7)))
    assert len(classes_seven) == 11, "there are exactly eleven free trees on seven vertices"

    # --- Determinism --------------------------------------------------------------------
    tree, _ = lcg_tree(15, 13579)
    assert canonical_unrooted(tree) == canonical_unrooted(tree)
    assert tree_centres(tree) == tree_centres(tree)
    assert canonical_unrooted_ids(tree) == canonical_unrooted_ids(tree)


if __name__ == "__main__":
    main()
    print("PASS")
