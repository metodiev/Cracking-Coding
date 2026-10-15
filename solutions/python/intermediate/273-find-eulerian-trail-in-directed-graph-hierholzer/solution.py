"""Task 273 - Find an Eulerian trail in a directed graph (Hierholzer). (Tier: Intermediate, CTCI Ch 4)

A directed trail exists iff at most one vertex has out-in == 1 (the start), at most one has
in-out == 1 (the end), the rest are balanced, and every edge lies in one connected component.
The trail itself comes from Hierholzer with lexicographically ordered neighbours.
"""

def find_eulerian_trail(n, edges, start=None):
    """Returns the vertex sequence of an Eulerian trail, [] for no edges, or None if impossible."""
    if not edges:
        return []
    out_degree = [0] * n
    in_degree = [0] * n
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        out_degree[a] += 1
        in_degree[b] += 1
    for neighbours in adjacency:
        neighbours.sort()

    starts = [v for v in range(n) if out_degree[v] - in_degree[v] == 1]
    ends = [v for v in range(n) if in_degree[v] - out_degree[v] == 1]
    if len(starts) > 1 or len(ends) > 1:
        return None
    if any(out_degree[v] != in_degree[v] and v not in starts and v not in ends for v in range(n)):
        return None

    if start is None:
        start = starts[0] if starts else min(v for v in range(n) if out_degree[v] > 0)
    elif starts and start != starts[0]:
        return None
    elif out_degree[start] - in_degree[start] not in (0, 1):
        return None

    # every edge must sit in one weakly connected component
    active = {v for v in range(n) if out_degree[v] or in_degree[v]}
    if active:
        undirected = [[] for _ in range(n)]
        for a, b in edges:
            undirected[a].append(b)
            undirected[b].append(a)
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in undirected[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if active - seen:
            return None

    pointer = [0] * n
    trail = []
    stack = [start]
    while stack:
        u = stack[-1]
        if pointer[u] < len(adjacency[u]):
            v = adjacency[u][pointer[u]]
            pointer[u] += 1
            stack.append(v)
        else:
            trail.append(stack.pop())
    trail.reverse()
    return trail if len(trail) == len(edges) + 1 else None


def uses_all_edges(trail, edges):
    from collections import Counter

    return Counter(zip(trail, trail[1:])) == Counter(edges)


def main():
    cycle = [(0, 1), (1, 2), (2, 0)]
    assert find_eulerian_trail(3, cycle) == [0, 1, 2, 0]
    assert uses_all_edges(find_eulerian_trail(3, cycle), cycle)
    assert find_eulerian_trail(3, cycle, start=1) == [1, 2, 0, 1]   # balanced graph: any vertex may start

    path = [(0, 1), (1, 2)]
    assert find_eulerian_trail(3, path) == [0, 1, 2]

    mixed = [(0, 1), (1, 2), (2, 0), (0, 3)]
    trail = find_eulerian_trail(4, mixed)
    assert trail == [0, 1, 2, 0, 3], trail
    assert uses_all_edges(trail, mixed)

    assert find_eulerian_trail(4, [(0, 1), (0, 2), (2, 0), (0, 3)]) is None   # out-in == 2
    assert find_eulerian_trail(4, [(0, 1), (2, 3)]) is None                    # disconnected
    assert find_eulerian_trail(3, []) == []                                    # nothing to do
    assert find_eulerian_trail(3, cycle, start=2) == [2, 0, 1, 2]              # ... and the walk still covers every edge

    # an explicit valid start on an open trail
    open_trail = [(1, 2), (2, 0)]
    assert find_eulerian_trail(3, open_trail, start=1) == [1, 2, 0]


if __name__ == "__main__":
    main()
    print("PASS")
