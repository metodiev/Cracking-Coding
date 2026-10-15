"""Task 280 - Count strongly connected components of a directed graph. (Tier: Intermediate, CTCI Ch 4)

Kosaraju: DFS the graph to get finish order, then DFS the reversed graph in reverse finish
order - each tree is one SCC. Cross-checked against a transitive-closure oracle.
"""

def strongly_connected_components(n, edges):
    """Returns components as a list of sorted vertex lists."""
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


def count_components(n, edges):
    return len(strongly_connected_components(n, edges))


def brute_force_components(n, edges):
    """Transitive closure oracle: u ~ v iff each reaches the other."""
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for a, b in edges:
        reach[a][b] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True

    groups = {}
    for v in range(n):
        key = tuple(i for i in range(n) if reach[v][i] and reach[i][v])
        groups.setdefault(key, []).append(v)
    return sorted(sorted(group) for group in groups.values())


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


def main():
    cycle = [(0, 1), (1, 2), (2, 0)]
    assert count_components(3, cycle) == 1
    assert count_components(4, cycle + [(3, 3)]) == 2                     # isolated self-loop vertex
    assert count_components(3, [(0, 1), (1, 2)]) == 3                     # DAG -> all singletons
    assert count_components(3, []) == 3
    assert count_components(3, [(0, 1), (1, 0), (1, 2)]) == 2             # mutual pair plus tail

    classic = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (4, 5)]
    components = strongly_connected_components(6, classic)
    assert components == [[0, 1, 2], [3, 4], [5]] or sorted(components) == [[0, 1, 2], [3, 4], [5]]
    assert count_components(6, classic) == 3

    # a two-node cycle where only one direction exists is NOT strongly connected
    assert count_components(2, [(0, 1)]) == 2

    for edges in random_graphs(6, 120):
        components = strongly_connected_components(6, edges)
        assert sorted(components) == brute_force_components(6, edges), edges
        assert count_components(6, edges) == len(brute_force_components(6, edges)), edges
        assert count_components(6, edges) <= 6


if __name__ == "__main__":
    main()
    print("PASS")
