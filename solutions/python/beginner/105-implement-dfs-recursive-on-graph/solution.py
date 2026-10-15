"""Task 105 - Implement DFS (recursive) on graph. (Tier: Beginner, CTCI Ch 8)

Recursive preorder DFS.  The visited set is marked on entry, so back edges
into the current path cannot loop forever; dfs_all restarts the walk for
every component so disconnected graphs are covered too.
"""


def dfs_order(graph, source):
    if source not in graph:
        return []
    visited = set()
    order = []
    _dfs(graph, source, visited, order)
    return order


def dfs_all(graph):
    visited = set()
    order = []
    for v in sorted(graph):
        if v not in visited:
            _dfs(graph, v, visited, order)
    return order


def _dfs(graph, v, visited, order):
    visited.add(v)
    order.append(v)
    for w in graph.get(v, []):
        if w not in visited:
            _dfs(graph, w, visited, order)


def main():
    cyclic = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
    assert dfs_order(cyclic, 0) == [0, 1, 2, 3]          # cycle guard holds

    two_cycles = {0: [1], 1: [0], 2: [3], 3: [2]}
    assert dfs_order(two_cycles, 0) == [0, 1]            # other component untouched
    assert dfs_all(two_cycles) == [0, 1, 2, 3]           # disconnected covered

    assert dfs_order({5: []}, 5) == [5]                  # single vertex
    assert dfs_order({}, 0) == []
    assert dfs_order(cyclic, 99) == []                   # source not in graph
    assert dfs_all({}) == []
    assert dfs_all({7: []}) == [7]

    chain = {i: [i + 1] for i in range(50)}
    chain[50] = []
    assert len(dfs_order(chain, 0)) == 51                # deep chain, no repeats


if __name__ == "__main__":
    main()
    print("PASS")
