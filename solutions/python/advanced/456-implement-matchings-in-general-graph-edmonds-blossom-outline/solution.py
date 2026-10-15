"""Task 456 - Implement matchings in general graph (Edmonds blossom outline). (Tier: Advanced, CTCI Ch 4)

Outline of general-graph matching: start from a greedy maximal matching, then repeatedly look
for an augmenting path from every free vertex and flip it. Finding augmenting paths is the hard
part on general graphs because of odd cycles ("blossoms"); the polynomial blossom algorithm
contracts them, which is the subject of task 457. Here the search is kept as an explicit
exhaustive depth-first search over *simple alternating paths*: from an even vertex we walk an
unmatched edge; if the neighbour is free we have an augmenting path, otherwise we continue
along its matched edge to the next even vertex (each even vertex is visited once). By Berge's
theorem a matching is maximum iff it has no augmenting path, and the exhaustive search finds
one whenever it exists, so the outline is exact for the small graphs it is meant for.

Assertions: every produced matching is VALID (mate symmetric, no shared vertices, every pair is
a real edge), equals the known optima on hand graphs (triangle -> 1, C5 -> 2, K4 -> 2, star -> 1,
C6 -> 3, K3,3 -> 3, two triangles -> 2), and equals a brute-force maximum matching on LCG general
graphs with n <= 8. At least one LCG graph must be strictly improved by an augmenting path over
the plain greedy matching, so the augmentation loop is documented as doing real work.
"""


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


def greedy_matching(n, adj):
    mate = [-1] * n
    for u in range(n):
        if mate[u] == -1:
            for v in adj[u]:
                if v > u and mate[v] == -1:
                    mate[u] = v
                    mate[v] = u
                    break
    return mate


def find_augmenting_path(n, adj, mate, root):
    """DFS over simple alternating paths from a free vertex; returns the edge list or None."""
    on_path = [False] * n
    on_path[root] = True
    path = []

    def dfs(u):
        for v in adj[u]:
            if mate[u] == v or on_path[v]:
                continue
            if mate[v] == -1:
                path.append((u, v))
                return True
            w = mate[v]
            if on_path[w]:
                continue
            on_path[v] = True
            on_path[w] = True
            path.append((u, v))
            path.append((v, w))
            if dfs(w):
                return True
            path.pop()
            path.pop()
            on_path[v] = False
            on_path[w] = False
        return False

    return path if dfs(root) else None


def apply_augmenting(mate, path):
    # First remove the matched edges on the path, then add the previously unmatched ones.
    for i, (a, b) in enumerate(path):
        if i % 2 == 1:
            assert mate[a] == b and mate[b] == a, "path alternates"
            mate[a] = -1
            mate[b] = -1
    for i, (a, b) in enumerate(path):
        if i % 2 == 0:
            mate[a] = b
            mate[b] = a


def max_matching_outline(n, adj):
    """Greedy + augmenting paths; returns (mate, augmentations_performed)."""
    mate = greedy_matching(n, adj)
    augmentations = 0
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if mate[u] == -1:
                path = find_augmenting_path(n, adj, mate, u)
                if path is not None:
                    apply_augmenting(mate, path)
                    augmentations += 1
                    changed = True
    return mate, augmentations


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
    triangle = (3, [(0, 1), (1, 2), (0, 2)])
    c5 = (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    k4 = (4, [(u, v) for u in range(4) for v in range(u + 1, 4)])
    star = (5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    c6 = (6, [(i, (i + 1) % 6) for i in range(6)])
    c7 = (7, [(i, (i + 1) % 7) for i in range(7)])
    two_triangles = (6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    k33 = (6, [(u, v + 3) for u in range(3) for v in range(3)])
    diamond = (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    p4 = (4, [(0, 1), (1, 2), (2, 3)])
    empty = (5, [])
    single = (2, [(0, 1)])

    expected = {id(triangle): 1, id(c5): 2, id(k4): 2, id(star): 1, id(c6): 3, id(c7): 3,
                id(two_triangles): 2, id(k33): 3, id(diamond): 2, id(p4): 2, id(empty): 0,
                id(single): 1}
    for graph in (triangle, c5, k4, star, c6, c7, two_triangles, k33, diamond, p4, empty, single):
        n, edges = graph
        adj = build_adj(n, edges)
        want = expected[id(graph)]
        assert brute_max_matching(n, adj) == want, (n, edges)
        mate, augmentations = max_matching_outline(n, adj)
        assert check_matching(n, adj, mate) == want, (n, edges, mate)
        assert augmentations >= 0

    improved_graphs = 0
    checked = 0
    total_augmentations = 0
    for seed in (4561, 4562, 4563, 4564):
        for n, edges in lcg_general_graphs(6, seed):
            adj = build_adj(n, edges)
            greedy = greedy_matching(n, adj)
            greedy_size = check_matching(n, adj, greedy)
            mate, augmentations = max_matching_outline(n, adj)
            size = check_matching(n, adj, mate)
            assert size >= greedy_size
            if size > greedy_size:
                improved_graphs += 1
            want = brute_max_matching(n, adj)
            assert size == want, (n, edges, size, want)
            total_augmentations += augmentations
            checked += 1
    assert checked == 24
    assert improved_graphs >= 1
    assert total_augmentations >= improved_graphs


if __name__ == "__main__":
    main()
    print("PASS")
