"""Task 282 - Bridge edges via Tarjan's disc/low values. (Tier: Intermediate, CTCI Ch 4)

An edge (u, v) is a bridge when the subtree below v cannot reach u or higher, i.e.
low[v] > disc[u]. One DFS pass; parallel edges are handled by skipping only the exact
edge id we arrived on. Cross-checked by deleting each edge and recounting components.
"""

def find_bridges(n, edges):
    """Returns bridges as canonical sorted (min, max) pairs."""
    adjacency = [[] for _ in range(n)]
    for eid, (a, b) in enumerate(edges):
        if a == b:
            continue                              # self loops are never bridges
        adjacency[a].append((b, eid))
        adjacency[b].append((a, eid))

    disc = [-1] * n
    low = [0] * n
    timer = [0]
    bridges = []

    def dfs(u, parent_edge):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v, eid in adjacency[u]:
            if eid == parent_edge:
                continue                          # skip only the arrival edge itself
            if disc[v] != -1:
                low[u] = min(low[u], disc[v])     # back edge to an ancestor
            else:
                dfs(v, eid)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:              # v's subtree cannot escape above u
                    bridges.append((min(u, v), max(u, v)))

    for v in range(n):
        if disc[v] == -1:
            dfs(v, -1)
    bridges.sort()
    return bridges


def count_components(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = [False] * n
    total = 0
    for s in range(n):
        if seen[s]:
            continue
        total += 1
        stack = [s]
        seen[s] = True
        while stack:
            x = stack.pop()
            for y in adjacency[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
    return total


def brute_force_bridges(n, edges):
    """Oracle: deleting a bridge increases the number of connected components."""
    base = count_components(n, edges)
    found = set()
    for i in range(len(edges)):
        without = edges[:i] + edges[i + 1:]
        if count_components(n, without) > base:
            a, b = edges[i]
            found.add((min(a, b), max(a, b)))
    return sorted(found)


def random_undirected_graphs(n, count, seed=13579):
    """Simple graphs: each unordered pair {u, v} considered once."""
    graphs = []
    for _ in range(count):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                seed = (seed * 75 + 74) % 65537
                if seed % 3 == 0:
                    edges.append((u, v))
        graphs.append(edges)
    return graphs


def random_multigraphs(n, count, seed=13579):
    """Multigraphs: (u, v) and (v, u) may both appear as parallel edges."""
    graphs = []
    for _ in range(count):
        edges = []
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                seed = (seed * 75 + 74) % 65537
                if seed % 3 == 0:
                    edges.append((u, v))
        graphs.append(edges)
    return graphs


def main():
    assert find_bridges(3, [(0, 1), (1, 2)]) == [(0, 1), (1, 2)]                    # path: 2 bridges
    assert find_bridges(3, [(0, 1), (1, 2), (2, 0)]) == []                          # triangle
    assert find_bridges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == []                  # square
    assert find_bridges(5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]) == [(2, 4)]    # square + tail
    assert find_bridges(5, [(0, 1), (1, 2), (3, 4)]) == [(0, 1), (1, 2), (3, 4)]    # disconnected
    assert find_bridges(3, [(0, 1), (0, 1), (1, 2)]) == [(1, 2)]                    # parallel edges
    assert find_bridges(2, [(0, 0), (0, 1)]) == [(0, 1)]                            # self loop ignored
    assert find_bridges(3, [(0, 1)]) == [(0, 1)]                                    # isolated vertex 2

    for label, graphs in (("simple", random_undirected_graphs(6, 120)),
                          ("multigraph", random_multigraphs(6, 120))):
        seen_bridge = False
        seen_no_bridge = False
        for edges in graphs:
            bridges = find_bridges(6, edges)
            assert bridges == brute_force_bridges(6, edges), (label, edges)
            edge_set = {(min(a, b), max(a, b)) for a, b in edges if a != b}
            assert all(bridge in edge_set for bridge in bridges), (label, edges)
            if bridges:
                seen_bridge = True
            else:
                seen_no_bridge = True
        assert seen_bridge and seen_no_bridge, label


if __name__ == "__main__":
    main()
    print("PASS")
