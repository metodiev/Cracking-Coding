"""Task 278 - Dinic's algorithm for max flow (basic). (Tier: Intermediate, CTCI Ch 4)

Each phase builds the BFS level graph and then repeatedly pushes a blocking flow with DFS
plus a per-vertex pointer, so a vertex is scanned once per phase. O(V^2 * E).
Cross-checked against Edmonds-Karp on deterministic pseudo-random graphs.
"""

from collections import deque

def dinic(n, capacity, source, sink):
    """capacity[i][j] is the capacity of i -> j; the residual matrix is updated in place."""
    residual = [row[:] for row in capacity]
    total = 0
    while True:
        level = [-1] * n
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if level[v] == -1 and residual[u][v] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[sink] == -1:
            return total

        pointer = [0] * n

        def dfs(u, pushed):
            if u == sink:
                return pushed
            while pointer[u] < n:
                v = pointer[u]
                if residual[u][v] > 0 and level[v] == level[u] + 1:
                    pushed_here = dfs(v, min(pushed, residual[u][v]))
                    if pushed_here > 0:
                        residual[u][v] -= pushed_here
                        residual[v][u] += pushed_here
                        return pushed_here
                pointer[u] += 1
            return 0

        while True:
            pushed = dfs(source, float("inf"))
            if pushed == 0:
                break
            total += pushed

def edmonds_karp(n, capacity, source, sink):
    """BFS augmenting-path reference implementation used to cross-check Dinic."""
    residual = [row[:] for row in capacity]
    total = 0
    while True:
        parent = [-1] * n
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            u = queue.popleft()
            for v in range(n):
                if parent[v] == -1 and residual[u][v] > 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] == -1:
            return total
        bottleneck = float("inf")
        v = sink
        while v != source:
            bottleneck = min(bottleneck, residual[parent[v]][v])
            v = parent[v]
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= bottleneck
            residual[v][u] += bottleneck
            v = u
        total += bottleneck


def matrix_from_edges(n, edges):
    capacity = [[0] * n for _ in range(n)]
    for u, v, cap in edges:
        capacity[u][v] += cap
    return capacity


def lcg_capacities(n, seed):
    """Deterministic LCG so all four languages test the same graphs."""
    capacity = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            seed = (seed * 75 + 74) % 65537
            if seed % 4 == 0:
                capacity[u][v] = seed % 10
    return capacity, seed


def min_cut_brute_force(n, capacity, source, sink):
    """Independent oracle: enumerate every s-t cut (max-flow = min-cut)."""
    others = [v for v in range(n) if v not in (source, sink)]
    best = float("inf")
    for mask in range(1 << len(others)):
        side = {source}
        for i, v in enumerate(others):
            if mask >> i & 1:
                side.add(v)
        cut = sum(capacity[u][v] for u in side for v in range(n) if v not in side)
        best = min(best, cut)
    return best


def main():
    clrs = matrix_from_edges(6, [
        (0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4),
        (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
        (3, 5, 20), (4, 5, 4),
    ])
    assert dinic(6, clrs, 0, 5) == 23
    assert edmonds_karp(6, clrs, 0, 5) == 23

    diamond = matrix_from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)])
    assert dinic(4, diamond, 0, 3) == 4

    crossing = matrix_from_edges(4, [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)])
    assert dinic(4, crossing, 0, 3) == 2
    assert dinic(3, matrix_from_edges(3, [(0, 1, 1)]), 0, 2) == 0

    seed = 97531
    for _ in range(150):
        capacity, seed = lcg_capacities(6, seed)
        assert dinic(6, capacity, 0, 5) == edmonds_karp(6, capacity, 0, 5), capacity

    # a 2-layer network where the blocking flow needs several pushes per phase
    layered = matrix_from_edges(8, [
        (0, 1, 10), (0, 2, 10), (1, 3, 5), (1, 4, 6), (2, 5, 6), (2, 6, 5),
        (3, 7, 4), (4, 7, 7), (5, 7, 5), (6, 7, 4),
    ])
    assert dinic(8, layered, 0, 7) == edmonds_karp(8, layered, 0, 7)
    assert dinic(8, layered, 0, 7) == min_cut_brute_force(8, layered, 0, 7)
    assert dinic(6, clrs, 0, 5) == min_cut_brute_force(6, clrs, 0, 5)
    seed = 1
    for _ in range(40):
        capacity, seed = lcg_capacities(6, seed)
        assert dinic(6, capacity, 0, 5) == min_cut_brute_force(6, capacity, 0, 5), capacity


if __name__ == "__main__":
    main()
    print("PASS")
