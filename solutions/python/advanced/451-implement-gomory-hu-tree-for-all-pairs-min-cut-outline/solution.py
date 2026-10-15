"""Task 451 - Implement Gomory-Hu tree for all-pairs min-cut (outline). (Tier: Advanced, CTCI Ch 4)

Outline: an all-pairs min-cut summary of an undirected graph needs only n-1 max-flow runs
instead of O(n^2). Gusfield's simplification keeps a parent array: for s = 1..n-1 we run one
max-flow between s and t = parent[s] on the ORIGINAL graph, store the value as the tree edge
weight of (s, parent[s]), and re-parent the still-unprocessed vertices i > s that sit on s's
side of the cut. After n-1 runs, the min cut between u and v equals the minimum edge weight on
the tree path u..v. Documented simplification vs. the original Gomory-Hu contraction scheme:
no vertex contraction is ever done; the flow network is always the original one.

The outline's tree is checked against a brute-force oracle (all vertex partitions) for every
pair on small LCG graphs, plus a single direct max-flow value on the (0, n-1) pair of each
graph, and symmetry of every reported pair.
"""

from collections import deque

INF = float("inf")


def lcg_graphs(count, seed, min_n=4, max_n=7):
    """Undirected connected LCG graphs: a random tree plus a few extra edges."""
    graphs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            parent = seed % node
            seed = (seed * 75 + 74) % 65537
            edges.append((node, parent, 1 + seed % 9))
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            if u != v:
                edges.append((u, v, 1 + seed % 9))
        graphs.append((n, edges))
    return graphs


def capacity_matrix(n, edges):
    cap = [[0] * n for _ in range(n)]
    for u, v, w in edges:
        if u != v:
            cap[u][v] += w
            cap[v][u] += w
    return cap


def max_flow_undirected(n, edges, s, t):
    """Edmonds-Karp on the directed model of an undirected graph.

    Undirected edge {u,v,w} becomes arcs u->v and v->u of capacity w. Returns
    (max flow value, set of vertices reachable from s in the final residual graph).
    """
    cap = capacity_matrix(n, edges)
    flow = 0
    while True:
        parent = [-1] * n
        parent[s] = s
        q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if parent[t] == -1:
            reachable = {i for i in range(n) if parent[i] != -1}
            return flow, reachable
        aug = INF
        v = t
        while v != s:
            u = parent[v]
            if cap[u][v] < aug:
                aug = cap[u][v]
            v = u
        v = t
        while v != s:
            u = parent[v]
            cap[u][v] -= aug
            cap[v][u] += aug
            v = u
        flow += aug


def gomory_hu(n, edges):
    """Gusfield simplification: n-1 max-flow runs. Returns (parent, weight, flow_runs)."""
    parent = [0] * n
    weight = [0] * n
    runs = 0
    for s in range(1, n):
        t = parent[s]
        value, side = max_flow_undirected(n, edges, s, t)
        runs += 1
        weight[s] = value
        for i in range(s + 1, n):
            if parent[i] == t and i in side:
                parent[i] = s
    return parent, weight, runs


def tree_min_cut(n, parent, weight, u, v):
    """Minimum edge weight on the tree path u..v (BFS over tree edges)."""
    tree = [[] for _ in range(n)]
    for i in range(1, n):
        tree[i].append((parent[i], weight[i]))
        tree[parent[i]].append((i, weight[i]))
    seen = [False] * n
    seen[u] = True
    q = deque([(u, INF)])
    while q:
        node, best = q.popleft()
        if node == v:
            return best
        for nxt, w in tree[node]:
            if not seen[nxt]:
                seen[nxt] = True
                q.append((nxt, min(best, w)))
    return None


def brute_pair_min_cut(n, edges, u, v):
    """Exact min u-v cut: minimum crossing weight over all vertex partitions."""
    best = INF
    for mask in range(1 << n):
        if ((mask >> u) & 1) == ((mask >> v) & 1):
            continue
        cut = 0
        for a, b, w in edges:
            if ((mask >> a) & 1) != ((mask >> b) & 1):
                cut += w
        if cut < best:
            best = cut
    return best


def main():
    triangle = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
    parent, weight, runs = gomory_hu(3, triangle)
    assert runs == 2
    for u in range(3):
        for v in range(u + 1, 3):
            assert tree_min_cut(3, parent, weight, u, v) == 2
            assert tree_min_cut(3, parent, weight, u, v) == tree_min_cut(3, parent, weight, v, u)
            assert brute_pair_min_cut(3, triangle, u, v) == 2

    path = [(0, 1, 5), (1, 2, 3), (2, 3, 4)]
    expected = {(0, 1): 5, (0, 2): 3, (0, 3): 3, (1, 2): 3, (1, 3): 3, (2, 3): 4}
    parent, weight, runs = gomory_hu(4, path)
    assert runs == 3
    for (u, v), want in expected.items():
        assert tree_min_cut(4, parent, weight, u, v) == want
        assert brute_pair_min_cut(4, path, u, v) == want

    k4 = [(u, v, 1) for u in range(4) for v in range(u + 1, 4)]
    parent, weight, runs = gomory_hu(4, k4)
    for u in range(4):
        for v in range(u + 1, 4):
            assert tree_min_cut(4, parent, weight, u, v) == 3
            assert brute_pair_min_cut(4, k4, u, v) == 3

    parallel = [(0, 1, 3), (1, 0, 4)]
    parent, weight, runs = gomory_hu(2, parallel)
    assert tree_min_cut(2, parent, weight, 0, 1) == 7
    assert brute_pair_min_cut(2, parallel, 0, 1) == 7

    pair_checks = 0
    graph_count = 0
    for seed in (4511, 4512, 4513, 4514):
        for n, edges in lcg_graphs(4, seed):
            parent, weight, runs = gomory_hu(n, edges)
            assert runs == n - 1
            for u in range(n):
                for v in range(u + 1, n):
                    got = tree_min_cut(n, parent, weight, u, v)
                    want = brute_pair_min_cut(n, edges, u, v)
                    assert got == want, (n, edges, u, v, got, want)
                    assert tree_min_cut(n, parent, weight, v, u) == got
                    pair_checks += 1
            direct, _ = max_flow_undirected(n, edges, 0, n - 1)
            assert direct == brute_pair_min_cut(n, edges, 0, n - 1), (n, edges)
            graph_count += 1
    assert graph_count == 16
    assert pair_checks >= 16 * 6


if __name__ == "__main__":
    main()
    print("PASS")
