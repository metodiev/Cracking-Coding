"""Task 264 - Find longest path in DAG (DP on topo order). (Tier: Intermediate, CTCI Ch 8)

Longest path in an UNWEIGHTED directed acyclic graph, measured in EDGES (documented;
no weighted variant here). Kahn's topological order feeds the DP: when u is popped,
dist[u] is final because all its predecessors have been processed; every edge u->v
relaxes dist[v] = max(dist[v], dist[u] + 1). Isolated/disconnected vertices stay at 0.

Cycle behaviour (documented): if the graph is cyclic, Kahn's queue drains before all
vertices are processed and the function raises ValueError instead of reporting a bogus
longest "path" (a longest path in a cyclic graph is ill-defined / unbounded).

longest_path_dfs is an independent memoized-DFS oracle used only for cross-checks on
acyclic inputs. Complexity: O(V + E) time, O(V + E) space.
"""

from collections import deque


def longest_path(n, edges):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(v for v in range(n) if indeg[v] == 0)
    dist = [0] * n
    processed = 0
    while q:
        u = q.popleft()
        processed += 1
        for v in adj[u]:
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if processed != n:
        raise ValueError("graph contains a cycle")
    return max(dist) if n else 0


def longest_path_dfs(n, edges):
    """Memoized DFS reference (valid on acyclic inputs); used only for cross-checks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    memo = [None] * n

    def depth(u):
        if memo[u] is None:
            best = 0
            for v in adj[u]:
                best = max(best, 1 + depth(v))
            memo[u] = best
        return memo[u]

    return max((depth(u) for u in range(n)), default=0)


def main():
    dag = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (2, 5)]
    assert longest_path(6, dag) == 3                    # 0 -> 2 -> 3 -> 4 (also 0 -> 1 -> 3 -> 4)
    assert longest_path(8, dag) == 3                    # vertices 6 and 7 are disconnected
    assert longest_path(5, [(0, 1), (1, 2), (2, 3), (3, 4)]) == 4       # linear chain
    assert longest_path(1, []) == 0                     # single vertex
    assert longest_path(0, []) == 0                     # no vertices
    assert longest_path(5, [(0, 1), (1, 2), (3, 4)]) == 2               # disconnected chains
    assert longest_path(4, [(0, 1), (0, 2), (1, 3), (2, 3)]) == 2       # diamond
    assert longest_path(3, [(0, 2), (2, 1)]) == 2       # topological order != index order
    assert longest_path(4, [(3, 2), (2, 1), (1, 0)]) == 3               # reversed chain

    for bad_n, bad_edges in ((3, [(0, 1), (1, 2), (2, 0)]),
                             (4, [(0, 1), (1, 2), (2, 0), (2, 3)])):
        try:
            longest_path(bad_n, bad_edges)
            raise AssertionError("cycle must be rejected")
        except ValueError:
            pass

    import random

    rng = random.Random(264)
    for _ in range(300):                            # cross-check vs the memoized-DFS oracle
        n = rng.randint(0, 12)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.25]
        assert longest_path(n, edges) == longest_path_dfs(n, edges)


if __name__ == "__main__":
    main()
    print("PASS")
