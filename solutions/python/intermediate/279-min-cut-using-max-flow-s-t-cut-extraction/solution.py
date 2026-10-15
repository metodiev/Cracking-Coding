"""Task 279 - Min cut using max flow (s-t cut extraction). (Tier: Intermediate, CTCI Ch 4)

Run max flow, then take the vertices still reachable from s in the residual graph as S.
The edges leaving S form the minimum cut and max-flow = min-cut is asserted against an
independent brute-force enumeration of every s-t cut.
"""

from collections import deque


def max_flow(n, capacity, source, sink):
    """Edmonds-Karp; returns the flow value and the final residual matrix."""
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
            return total, residual
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


def min_cut(n, capacity, source, sink):
    """Returns (value, S, T, cut_edges) where cut_edges are (u, v, capacity)."""
    value, residual = max_flow(n, capacity, source, sink)
    side = [False] * n
    side[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if not side[v] and residual[u][v] > 0:
                side[v] = True
                queue.append(v)
    in_s = {v for v in range(n) if side[v]}
    in_t = {v for v in range(n) if not side[v]}
    cut_edges = [(u, v, capacity[u][v]) for u in in_s for v in in_t if capacity[u][v] > 0]
    return value, in_s, in_t, cut_edges


def min_cut_brute_force(n, capacity, source, sink):
    others = [v for v in range(n) if v not in (source, sink)]
    best = float("inf")
    for mask in range(1 << len(others)):
        side = {source}
        for i, v in enumerate(others):
            if mask >> i & 1:
                side.add(v)
        best = min(best, sum(capacity[u][v] for u in side for v in range(n) if v not in side))
    return best


def matrix_from_edges(n, edges):
    capacity = [[0] * n for _ in range(n)]
    for u, v, cap in edges:
        capacity[u][v] += cap
    return capacity


def lcg_capacities(n, seed):
    capacity = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            seed = (seed * 75 + 74) % 65537
            if seed % 4 == 0:
                capacity[u][v] = seed % 10
    return capacity, seed


def main():
    clrs = matrix_from_edges(6, [
        (0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4),
        (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
        (3, 5, 20), (4, 5, 4),
    ])
    value, in_s, in_t, cut_edges = min_cut(6, clrs, 0, 5)
    assert value == 23
    assert 0 in in_s and 5 in in_t
    assert sum(cap for _, _, cap in cut_edges) == value          # max-flow = min-cut
    assert value == min_cut_brute_force(6, clrs, 0, 5)
    assert all(v in in_t for _, v, _ in cut_edges)

    diamond = matrix_from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)])
    value, in_s, in_t, cut_edges = min_cut(4, diamond, 0, 3)
    assert value == 4 and sum(cap for _, _, cap in cut_edges) == 4
    assert value == min_cut_brute_force(4, diamond, 0, 3)

    # sink unreachable: S is everything reachable, the cut has capacity 0
    stranded = matrix_from_edges(3, [(0, 1, 7)])
    value, in_s, in_t, cut_edges = min_cut(3, stranded, 0, 2)
    assert value == 0 and in_s == {0, 1} and in_t == {2} and cut_edges == []

    # no residual edge may leave S in either direction of the theorem
    seed = 24680
    for _ in range(60):
        capacity, seed = lcg_capacities(6, seed)
        value, in_s, in_t, cut_edges = min_cut(6, capacity, 0, 5)
        assert sum(cap for _, _, cap in cut_edges) == value, capacity
        assert value == min_cut_brute_force(6, capacity, 0, 5), capacity


if __name__ == "__main__":
    main()
    print("PASS")
