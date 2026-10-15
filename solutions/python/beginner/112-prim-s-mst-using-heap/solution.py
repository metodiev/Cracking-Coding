"""Task 112 - Prim's MST using heap. (Tier: Beginner, CTCI Ch 4)"""

import heapq


def prim(n, adj, start=0):
    """adj[u] = list of (v, weight). Returns (total, mst edges) or None if disconnected."""
    visited = [False] * n
    heap = [(0, start, -1)]          # weight, node, parent
    total = 0
    chosen = []
    while heap and len(chosen) < n:
        weight, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total += weight
        if parent != -1:
            chosen.append((parent, u, weight))
        for v, w in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (w, v, u))
    return (total, chosen) if len(chosen) == n - 1 else None


def main():
    adj = [
        [(1, 4), (2, 3)],
        [(0, 4), (2, 1), (3, 2)],
        [(0, 3), (1, 1), (3, 4), (4, 5)],
        [(1, 2), (2, 4), (4, 2)],
        [(3, 2), (2, 5)],
    ]
    total, chosen = prim(5, adj)
    assert total == 8, total                 # same MST weight as Kruskal (task 111)
    assert len(chosen) == 4

    assert prim(1, [[]]) == (0, [])
    assert prim(3, [[(1, 1)], [(0, 1)], []]) is None    # vertex 2 unreachable


if __name__ == "__main__":
    main()
    print("PASS")
