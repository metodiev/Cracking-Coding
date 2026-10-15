"""Task 113 - Simple Dijkstra's algorithm (non-negative weights). (Tier: Beginner, CTCI Ch 4)"""

import heapq


def dijkstra(n, adj, src):
    """adj[u] = list of (v, weight >= 0). Returns distances (inf = unreachable)."""
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:                  # stale heap entry
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def main():
    adj = [
        [(1, 1), (2, 4)],
        [(2, 2), (3, 6)],
        [(3, 3)],
        [],
    ]
    dist = dijkstra(4, adj, 0)
    assert dist == [0, 1, 3, 6], dist

    # unreachable vertex stays infinity
    adj2 = [[(1, 5)], [(0, 5)], []]
    dist2 = dijkstra(3, adj2, 0)
    assert dist2[0] == 0 and dist2[1] == 5 and dist2[2] == float("inf")

    # agreement check against Bellman-Ford (task 114) on a DAG
    adj3 = [[(1, 2), (2, 5)], [(2, 1), (3, 4)], [(3, 1)], []]
    assert dijkstra(4, adj3, 0) == [0, 2, 3, 4]


if __name__ == "__main__":
    main()
    print("PASS")
