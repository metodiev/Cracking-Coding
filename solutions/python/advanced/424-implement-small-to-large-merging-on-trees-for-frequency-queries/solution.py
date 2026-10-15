"""Task 424 - Implement small-to-large merging on trees for frequency queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Roots the tree at 0 and processes nodes bottom-up. Each node's colour counter starts as the
counter of its LARGEST child; smaller sibling counters are merged into it, which keeps the
total merge work at O(n log n) (every element moves only into a counter at least twice as
big). From each counter we read the number of distinct colours and the highest frequency in
the subtree; both are cross-checked against a naive DFS collection for every node.
"""


def root_tree(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = [-1] * n
    parent[0] = 0
    order = [0]
    for u in order:
        for v in adjacency[u]:
            if v != parent[u]:
                parent[v] = u
                order.append(v)
    children = [[] for _ in range(n)]
    for u in order[1:]:
        children[parent[u]].append(u)
    return children, order


def subtree_frequency_summary(n, edges, colors):
    """Returns (distinct colours, highest frequency) for every subtree."""
    children, order = root_tree(n, edges)
    counters = [None] * n
    distinct = [0] * n
    top_frequency = [0] * n
    for u in reversed(order):
        biggest = -1
        for v in children[u]:
            if biggest == -1 or len(counters[v]) > len(counters[biggest]):
                biggest = v
        if biggest == -1:
            table = {colors[u]: 1}
            best = 1
        else:
            table = counters[biggest]              # reuse the largest counter
            best = top_frequency[biggest]
            count = table.get(colors[u], 0) + 1
            table[colors[u]] = count
            if count > best:
                best = count
            for v in children[u]:
                if v == biggest:
                    continue
                for color, amount in counters[v].items():
                    count = table.get(color, 0) + amount
                    table[color] = count
                    if count > best:
                        best = count
        counters[u] = table
        distinct[u] = len(table)
        top_frequency[u] = best
    return distinct, top_frequency


def naive_frequency_summary(n, edges, colors):
    children, _ = root_tree(n, edges)
    distinct = [0] * n
    top_frequency = [0] * n
    for u in range(n):
        counts = {}
        stack = [u]
        while stack:
            node = stack.pop()
            counts[colors[node]] = counts.get(colors[node], 0) + 1
            stack.extend(children[node])
        distinct[u] = len(counts)
        top_frequency[u] = max(counts.values())
    return distinct, top_frequency


def main():
    # tree 0-1, 0-2, 1-3 with colours [0, 1, 0, 1]
    distinct, top = subtree_frequency_summary(4, [(0, 1), (0, 2), (1, 3)], [0, 1, 0, 1])
    assert (distinct, top) == ([2, 1, 1, 1], [2, 2, 1, 1])
    assert (distinct, top) == naive_frequency_summary(4, [(0, 1), (0, 2), (1, 3)], [0, 1, 0, 1])

    # path 0-1-2-3-4 with colours [1, 2, 2, 3, 3]
    path_edges = [(i, i + 1) for i in range(4)]
    distinct, top = subtree_frequency_summary(5, path_edges, [1, 2, 2, 3, 3])
    assert (distinct, top) == ([3, 2, 2, 1, 1], [2, 2, 2, 2, 1])
    assert (distinct, top) == naive_frequency_summary(5, path_edges, [1, 2, 2, 3, 3])

    # single node
    assert subtree_frequency_summary(1, [], [7]) == ([1], [1])

    # star with five nodes, all the same colour
    star = [(0, i) for i in range(1, 5)]
    assert subtree_frequency_summary(5, star, [3, 3, 3, 3, 3]) == ([1, 1, 1, 1, 1], [5, 1, 1, 1, 1])

    for seed in (424, 2024, 77777):
        local = seed
        for _ in range(20):
            local = (local * 75 + 74) % 65537
            n = 2 + local % 59
            edges = []
            for node in range(1, n):
                local = (local * 75 + 74) % 65537
                edges.append((local % node, node))
            colors = []
            for _ in range(n):
                local = (local * 75 + 74) % 65537
                colors.append(local % (2 + seed % 5))
            got = subtree_frequency_summary(n, edges, colors)
            want = naive_frequency_summary(n, edges, colors)
            assert got == want, (seed, n, got, want)

    # a mostly-unique palette stresses distinct counts instead of frequencies
    local = 909090
    for _ in range(10):
        local = (local * 75 + 74) % 65537
        n = 2 + local % 59
        edges = []
        for node in range(1, n):
            local = (local * 75 + 74) % 65537
            edges.append((local % node, node))
        colors = []
        for _ in range(n):
            local = (local * 75 + 74) % 65537
            colors.append(local % 1000)
        assert subtree_frequency_summary(n, edges, colors) == naive_frequency_summary(n, edges, colors), n


if __name__ == "__main__":
    main()
    print("PASS")
