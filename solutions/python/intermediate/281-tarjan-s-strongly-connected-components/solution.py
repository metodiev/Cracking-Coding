"""Task 281 - Tarjan's strongly connected components in one DFS pass. (Tier: Intermediate, CTCI Ch 4)

Tarjan: one DFS assigning discovery indices; a vertex is the root of an SCC when its
low-link equals its index. The resulting partition is cross-checked against the
Kosaraju result (task 280) on the same deterministic LCG graphs. Components are
returned as sorted vertex lists, ordered by their smallest vertex.
"""

def tarjan_scc(n, edges):
    """Single-pass Tarjan; returns components sorted by smallest vertex."""
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)

    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    components = []
    counter = [0]

    def strongconnect(u):
        index[u] = low[u] = counter[0]
        counter[0] += 1
        stack.append(u)
        on_stack[u] = True
        for v in adjacency[u]:
            if index[v] == -1:                    # tree edge: descend
                strongconnect(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:                     # back/cross edge to an open SCC
                low[u] = min(low[u], index[v])
        if low[u] == index[u]:                    # u closes an SCC: pop until u
            group = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                group.append(w)
                if w == u:
                    break
            components.append(sorted(group))

    for v in range(n):
        if index[v] == -1:
            strongconnect(v)
    components.sort(key=lambda group: group[0])
    return components


def kosaraju_scc(n, edges):
    """Task 280's two-pass algorithm, used here only as an independent oracle."""
    adjacency = [[] for _ in range(n)]
    reversed_adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        reversed_adjacency[b].append(a)

    visited = [False] * n
    order = []

    def dfs_finish(u):
        visited[u] = True
        for v in adjacency[u]:
            if not visited[v]:
                dfs_finish(v)
        order.append(u)

    for v in range(n):
        if not visited[v]:
            dfs_finish(v)

    component = [-1] * n
    components = []
    for u in reversed(order):
        if component[u] != -1:
            continue
        index = len(components)
        component[u] = index
        group = [u]
        stack = [u]
        while stack:
            x = stack.pop()
            for y in reversed_adjacency[x]:
                if component[y] == -1:
                    component[y] = index
                    group.append(y)
                    stack.append(y)
        components.append(sorted(group))
    return components


def random_graphs(n, count, seed=13579):
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


def canonical(components):
    return sorted((sorted(group) for group in components), key=lambda group: group[0])


def main():
    classic = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (4, 5)]
    assert tarjan_scc(6, classic) == [[0, 1, 2], [3, 4], [5]]

    assert tarjan_scc(3, [(0, 1), (1, 2), (2, 0)]) == [[0, 1, 2]]
    assert tarjan_scc(3, [(0, 1), (1, 2)]) == [[0], [1], [2]]          # DAG: all singletons
    assert tarjan_scc(3, []) == [[0], [1], [2]]
    assert tarjan_scc(3, [(0, 1), (1, 0), (1, 2)]) == [[0, 1], [2]]    # mutual pair plus tail
    assert tarjan_scc(2, [(0, 1)]) == [[0], [1]]                       # one-way edge
    assert tarjan_scc(1, [(0, 0)]) == [[0]]                            # self loop is its own SCC

    graphs = random_graphs(6, 120)
    nontrivial_seen = False
    for edges in graphs:
        components = tarjan_scc(6, edges)
        kosaraju = kosaraju_scc(6, edges)
        assert components == canonical(kosaraju), edges
        assert sorted(v for group in components for v in group) == list(range(6)), edges
        assert components == canonical(components), edges
        if any(len(group) > 1 for group in components):
            nontrivial_seen = True
    assert nontrivial_seen, "LCG graphs should contain real SCCs"


if __name__ == "__main__":
    main()
    print("PASS")
