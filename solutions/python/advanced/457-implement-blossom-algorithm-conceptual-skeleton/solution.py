"""Task 457 - Implement blossom algorithm (conceptual/skeleton). (Tier: Advanced, CTCI Ch 4)

Emdonds' blossom algorithm as a labelled BFS skeleton: vertices in the alternating forest are
labelled outer (even, S) via `used` and inner (odd, T) via `p` (p[v] is the outer vertex that
reached v). When a BFS step meets an edge joining two outer vertices, the odd cycle it closes is
a BLOSSOM: lca() finds the blossom base (following p/mate toward the roots, comparing base[]),
mark_path() relabels the stem and blossom edges (p[v] = child), and every vertex whose base lies
on the blossom is contracted to the base so the search continues "as if" the blossom were one
vertex. If instead the edge reaches an unvisited vertex with p[to] == -1 we grow the forest; a
free vertex ends the search and the augmenting path is flipped through the p pointers.

Assertions: contractions actually happen (C5 needs one), every produced matching is VALID, and
the matching equals known optima (triangle -> 1, C5 -> 2, K4 -> 2, star -> 1, C6 -> 3, C7 -> 3,
K3,3 -> 3, two triangles -> 2, diamond/P4 -> 2) and the brute-force maximum matching on LCG
general graphs with n <= 8.
"""

from collections import deque


def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def check_matching(n, adj, mate):
    """Asserts validity and returns the matching size."""
    size = 0
    neighbour_sets = [set(a) for a in adj]
    for v in range(n):
        if mate[v] != -1:
            assert mate[mate[v]] == v, "symmetric mates"
            assert mate[v] != v, "no self match"
            assert mate[v] in neighbour_sets[v], "matched pair must be an edge"
            size += 1
    assert size % 2 == 0
    return size // 2


def blossom_matching(n, adj):
    """Returns (mate, contractions): Edmonds' algorithm with labelled BFS + contraction."""
    mate = [-1] * n
    base = list(range(n))
    p = [-1] * n
    used = [False] * n
    blossom = [False] * n
    contractions = 0

    def lca(a, b):
        seen = [False] * n
        while True:
            a = base[a]
            seen[a] = True
            if mate[a] == -1:
                break
            a = p[mate[a]]
        while True:
            b = base[b]
            if seen[b]:
                return b
            b = p[mate[b]]

    def mark_path(v, b, child):
        while base[v] != b:
            blossom[base[v]] = True
            blossom[base[mate[v]]] = True
            p[v] = child
            child = mate[v]
            v = p[mate[v]]

    def find_path(root):
        nonlocal contractions
        for i in range(n):
            used[i] = False
            p[i] = -1
            base[i] = i
        used[root] = True
        queue = deque([root])
        while queue:
            v = queue.popleft()
            for to in adj[v]:
                if base[v] == base[to] or mate[v] == to:
                    continue
                if to == root or (mate[to] != -1 and p[mate[to]] != -1):
                    contractions += 1
                    curbase = lca(v, to)
                    for i in range(n):
                        blossom[i] = False
                    mark_path(v, curbase, to)
                    mark_path(to, curbase, v)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i] = curbase
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif p[to] == -1:
                    p[to] = v
                    if mate[to] == -1:
                        return to
                    nxt = mate[to]
                    used[nxt] = True
                    queue.append(nxt)
        return -1

    for i in range(n):
        if mate[i] == -1:
            v = find_path(i)
            while v != -1:
                pv = p[v]
                ppv = mate[pv]
                mate[v] = pv
                mate[pv] = v
                v = ppv
    return mate, contractions


def brute_max_matching(n, adj):
    """Exact maximum matching by exhaustive backtracking (first-free-vertex branching)."""
    mate = [-1] * n

    def rec(i):
        while i < n and mate[i] != -1:
            i += 1
        if i == n:
            return 0
        best = rec(i + 1)
        for v in adj[i]:
            if v > i and mate[v] == -1:
                mate[i] = v
                mate[v] = i
                best = max(best, 1 + rec(i + 1))
                mate[i] = -1
                mate[v] = -1
        return best

    return rec(0)


def lcg_general_graphs(count, seed, min_n=2, max_n=8):
    graphs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            edges.append((node, seed % node))
        for _ in range(n + 1):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
        graphs.append((n, edges))
    return graphs


def main():
    c5 = (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    adj = build_adj(*c5)
    mate, contractions = blossom_matching(5, adj)
    assert check_matching(5, adj, mate) == 2
    assert contractions >= 1
    assert brute_max_matching(5, adj) == 2

    hand_graphs = [
        ("triangle", 3, [(0, 1), (1, 2), (0, 2)], 1),
        ("C5", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 2),
        ("K4", 4, [(u, v) for u in range(4) for v in range(u + 1, 4)], 2),
        ("star", 5, [(0, 1), (0, 2), (0, 3), (0, 4)], 1),
        ("C6", 6, [(i, (i + 1) % 6) for i in range(6)], 3),
        ("C7", 7, [(i, (i + 1) % 7) for i in range(7)], 3),
        ("K3,3", 6, [(u, v + 3) for u in range(3) for v in range(3)], 3),
        ("two triangles", 6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 2),
        ("diamond", 4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], 2),
        ("P4", 4, [(0, 1), (1, 2), (2, 3)], 2),
        ("empty", 5, [], 0),
        ("single", 2, [(0, 1)], 1),
    ]
    for label, n, edges, want in hand_graphs:
        adj = build_adj(n, edges)
        assert brute_max_matching(n, adj) == want, label
        mate, contractions = blossom_matching(n, adj)
        assert check_matching(n, adj, mate) == want, label

    checked = 0
    total_contractions = 0
    for seed in (4571, 4572, 4573, 4574):
        for n, edges in lcg_general_graphs(6, seed):
            adj = build_adj(n, edges)
            mate, contractions = blossom_matching(n, adj)
            total_contractions += contractions
            size = check_matching(n, adj, mate)
            assert size == brute_max_matching(n, adj), (n, edges, size)
            checked += 1
    assert checked == 24
    assert total_contractions >= 1


if __name__ == "__main__":
    main()
    print("PASS")
