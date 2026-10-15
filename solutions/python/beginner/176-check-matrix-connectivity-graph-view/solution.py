
"""Task 176 - Check matrix connectivity (graph view). (Tier: Beginner, CTCI Ch 4)

An adjacency matrix defines an undirected graph: adjacency[i][j] = 1 means an edge.
The graph is connected iff BFS/DFS from 0 reaches every vertex. O(n^2).
"""

from collections import deque


def is_connected(adj):
    """adj is a square 0/1 matrix. Empty/1x1 graphs count as connected."""
    n = len(adj)
    if n <= 1:
        return True
    seen = [False] * n
    seen[0] = True
    q = deque([0])
    reached = 1
    while q:
        u = q.popleft()
        for v in range(n):
            if adj[u][v] and not seen[v]:
                seen[v] = True
                reached += 1
                q.append(v)
    return reached == n


def connected_components(adj):
    """Component count via repeated BFS (bonus view of the same structure)."""
    n = len(adj)
    seen = [False] * n
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        seen[start] = True
        q = deque([start])
        while q:
            u = q.popleft()
            for v in range(n):
                if adj[u][v] and not seen[v]:
                    seen[v] = True
                    q.append(v)
    return components


def main():
    # path 0-1-2-3
    path = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]
    assert is_connected(path) and connected_components(path) == 1

    # two separate pairs
    split = [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    assert not is_connected(split) and connected_components(split) == 2

    # isolated vertex
    star_with_lonely = [
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert not is_connected(star_with_lonely)
    assert connected_components(star_with_lonely) == 2

    assert is_connected([]) and is_connected([[0]])
    assert is_connected([[0, 1], [1, 0]])


if __name__ == "__main__":
    main()
    print("PASS")
