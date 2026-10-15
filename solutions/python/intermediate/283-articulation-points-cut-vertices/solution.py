"""Task 283 - Articulation points (cut vertices). (Tier: Intermediate, CTCI Ch 4)

An undirected vertex is an articulation point when removing it raises the number of
connected components. One DFS with disc/low values; the DFS root needs the special
rule of having more than one child. Cross-checked with a delete-each-vertex oracle.
"""

def articulation_points(n, edges):
    """Returns cut vertices in increasing order."""
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    disc = [-1] * n
    low = [0] * n
    timer = [0]
    points = set()

    def dfs(u, parent):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in adjacency[u]:
            if disc[v] == -1:                     # tree edge
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent != -1 and low[v] >= disc[u]:
                    points.add(u)                 # no back edge out of v's subtree
            elif v != parent:                     # back edge to an ancestor
                low[u] = min(low[u], disc[v])
        if parent == -1 and children > 1:         # root special case
            points.add(u)

    for v in range(n):
        if disc[v] == -1:
            dfs(v, -1)
    return sorted(points)


def count_components(n, edges, skipped=-1):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        if a == skipped or b == skipped:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = [False] * n
    total = 0
    for s in range(n):
        if s == skipped or seen[s]:
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


def brute_force_articulation_points(n, edges):
    """Oracle: remove each vertex and count connected components."""
    base = count_components(n, edges)
    return [v for v in range(n) if count_components(n, edges, skipped=v) > base]


def random_simple_graphs(n, count, divisor=3, seed=13579):
    """Each unordered pair {u, v} appears at most once; edge when seed % divisor == 0."""
    graphs = []
    for _ in range(count):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                seed = (seed * 75 + 74) % 65537
                if seed % divisor == 0:
                    edges.append((u, v))
        graphs.append(edges)
    return graphs


def main():
    assert articulation_points(4, [(0, 1), (1, 2), (2, 3)]) == [1, 2]                # path middles
    assert articulation_points(4, [(0, 1), (1, 2), (2, 3), (3, 0)]) == []            # cycle
    assert articulation_points(5, [(0, 1), (0, 2), (0, 3), (0, 4)]) == [0]           # star center
    assert articulation_points(3, [(0, 1), (0, 2)]) == [0]                           # root with 2 children
    assert articulation_points(6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]) == []  # two triangles
    assert articulation_points(4, [(0, 1), (1, 2), (2, 0), (2, 3)]) == [2]           # triangle plus tail
    assert articulation_points(2, [(0, 1)]) == []                                    # single edge
    assert articulation_points(3, []) == []                                          # isolated vertices
    assert articulation_points(3, [(0, 1), (0, 1), (1, 2)]) == [1]                   # parallel edges
    assert articulation_points(7, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 6)]) == [2]

    for divisor in (3, 5):
        seen_cut = seen_none = False
        for edges in random_simple_graphs(6, 120, divisor):
            points = articulation_points(6, edges)
            assert points == brute_force_articulation_points(6, edges), (divisor, edges)
            if points:
                seen_cut = True
            else:
                seen_none = True
        assert seen_cut and seen_none, divisor

    # a path graph: every interior vertex is a cut vertex
    path_edges = [(i, i + 1) for i in range(9)]
    assert articulation_points(10, path_edges) == list(range(1, 9))


if __name__ == "__main__":
    main()
    print("PASS")
