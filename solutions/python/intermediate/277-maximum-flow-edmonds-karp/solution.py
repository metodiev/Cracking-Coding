"""Task 277 - Maximum flow (Edmonds-Karp). (Tier: Intermediate, CTCI Ch 4)

BFS finds the augmenting path with the fewest edges in the residual network, which makes the
bound O(V*E^2) and also makes the algorithm immune to the classic bad-path blowup.
"""

from collections import deque


def edmonds_karp(n, capacity, source, sink):
    """capacity[i][j] is the capacity of the edge i -> j (0 when there is none)."""
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
            u = parent[v]
            bottleneck = min(bottleneck, residual[u][v])
            v = u
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


def main():
    # CLRS figure: the canonical 23-unit flow
    clrs = matrix_from_edges(6, [
        (0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4),
        (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
        (3, 5, 20), (4, 5, 4),
    ])
    assert edmonds_karp(6, clrs, 0, 5) == 23

    diamond = matrix_from_edges(4, [(0, 1, 3), (0, 2, 2), (1, 3, 2), (2, 3, 3)])
    assert edmonds_karp(4, diamond, 0, 3) == 4

    path = matrix_from_edges(3, [(0, 1, 3), (1, 2, 5)])
    assert edmonds_karp(3, path, 0, 2) == 3

    assert edmonds_karp(3, matrix_from_edges(3, [(0, 1, 1)]), 0, 2) == 0
    assert edmonds_karp(2, matrix_from_edges(2, []), 0, 1) == 0

    # two disjoint routes; the middle edge forces a residual/back-edge augmentation
    crossing = matrix_from_edges(4, [(0, 1, 1), (0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 3, 1)])
    assert edmonds_karp(4, crossing, 0, 3) == 2

    # widening an existing edge just adds capacity
    doubled = matrix_from_edges(3, [(0, 1, 2), (0, 1, 3), (1, 2, 4)])
    assert edmonds_karp(3, doubled, 0, 2) == 4

    # flow never exceeds the capacity of any cut through the source
    total_out = sum(clrs[0][v] for v in range(6))
    assert edmonds_karp(6, clrs, 0, 5) <= total_out


if __name__ == "__main__":
    main()
    print("PASS")
