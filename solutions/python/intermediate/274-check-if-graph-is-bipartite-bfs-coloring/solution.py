"""Task 274 - Check if a graph is bipartite (BFS 2-coloring). (Tier: Intermediate, CTCI Ch 4)

Grow a BFS forest per component; every edge must join opposite colours. A self-loop or an odd
cycle breaks the coloring, which is exactly the bipartite obstruction.
"""

from collections import deque


def two_color(n, edges):
    """Returns the colour list (0/1) or None when the graph is not bipartite."""
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    color = [-1] * n
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    return None
    return color


def is_bipartite(n, edges):
    return two_color(n, edges) is not None


def main():
    assert is_bipartite(4, [(0, 1), (1, 2), (2, 3), (3, 0)])          # even cycle
    assert not is_bipartite(3, [(0, 1), (1, 2), (2, 0)])              # triangle
    assert is_bipartite(3, [(0, 1), (1, 2)])                          # path
    assert not is_bipartite(1, [(0, 0)])                              # self-loop
    assert is_bipartite(5, [])                                        # no edges
    # two components, one of them odd
    assert not is_bipartite(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)])
    assert is_bipartite(6, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)])
    # K3,3 is bipartite, K4 is not
    assert is_bipartite(6, [(i, j) for i in range(3) for j in range(3, 6)])
    assert not is_bipartite(4, [(i, j) for i in range(4) for j in range(i + 1, 4)])

    colors = two_color(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert colors is not None and colors[0] != colors[1] and colors[1] != colors[2]
    assert all(colors[a] != colors[b] for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert two_color(3, [(0, 1), (1, 2), (2, 0)]) is None


if __name__ == "__main__":
    main()
    print("PASS")
