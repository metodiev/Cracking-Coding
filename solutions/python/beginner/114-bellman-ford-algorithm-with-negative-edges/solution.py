"""Task 114 - Bellman-Ford algorithm (with negative edges). (Tier: Beginner, CTCI Ch 4)"""


def bellman_ford(n, edges, src):
    """edges: (u, v, w). Returns distances or None when a negative cycle is reachable."""
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break                        # early exit when stable
    for u, v, w in edges:                # one more pass detects negative cycles
        if dist[u] != INF and dist[u] + w < dist[v]:
            return None
    return dist


def main():
    # graph with a negative edge but no negative cycle
    edges = [(0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 2)]
    assert bellman_ford(4, edges, 0) == [0, 4, 1, 3]

    # agrees with Dijkstra (task 113) on non-negative graphs
    non_neg = [(0, 1, 1), (1, 2, 2), (0, 2, 4)]
    assert bellman_ford(3, non_neg, 0) == [0, 1, 3]

    # negative cycle reachable from src -> None
    cycle = [(0, 1, 1), (1, 2, -2), (2, 1, -2)]
    assert bellman_ford(3, cycle, 0) is None

    # no path -> infinity
    far = [(1, 2, 1)]
    assert bellman_ford(3, far, 0)[2] == float("inf")


if __name__ == "__main__":
    main()
    print("PASS")
