"""Task 104 - Implement BFS on adjacency list graph. (Tier: Beginner, CTCI Ch 4)

Breadth-first search returns the order vertices are dequeued (level order)
and the distance in edges from the source.  Vertices in other components are
never reached, so they simply stay out of the distances map.
"""
from collections import deque


def bfs_order(graph, source):
    if source not in graph:
        return []
    seen = {source}
    order = []
    queue = deque([source])
    while queue:
        v = queue.popleft()
        order.append(v)
        for w in graph.get(v, []):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    return order


def bfs_distances(graph, source):
    if source not in graph:
        return {}
    dist = {source: 0}
    queue = deque([source])
    while queue:
        v = queue.popleft()
        for w in graph.get(v, []):
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


def main():
    g = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
    assert bfs_order(g, 0) == [0, 1, 2, 3]              # cycle visited once each
    assert bfs_distances(g, 0) == {0: 0, 1: 1, 2: 1, 3: 2}

    parts = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    assert bfs_order(parts, 0) == [0, 1]                # disconnected component
    assert bfs_distances(parts, 0) == {0: 0, 1: 1}
    assert bfs_order(parts, 4) == [4]                   # loner vertex
    assert bfs_distances(parts, 4) == {4: 0}

    assert bfs_order(parts, 99) == []                   # source not in graph
    assert bfs_distances(parts, 99) == {}
    assert bfs_order({}, 0) == []
    assert bfs_distances({}, 0) == {}

    assert bfs_order({5: [6], 6: [5, 7], 7: [5]}, 5) == [5, 6, 7]


if __name__ == "__main__":
    main()
    print("PASS")
