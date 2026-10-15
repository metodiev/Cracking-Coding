"""Task 430 - Implement Mo's algorithm on trees (Euler tour + Mo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

An Euler tour (tin/tout) turns every subtree into a contiguous interval of the flattened
array, so many offline "distinct colours in subtree u" queries fit into Mo's algorithm:
queries are ordered by block of the left end with an alternating right-end direction, and a
sliding window maintains the colour counts. Answers are cross-checked against a naive
recount (DFS per query) on LCG trees.
"""


def euler_tour(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    children = [[] for _ in range(n)]
    tin = [0] * n
    tout = [0] * n
    euler = [0] * n                       # euler[tin[u]] = u
    timer = 0

    def dfs(u, parent):
        nonlocal timer
        tin[u] = timer
        euler[timer] = u
        timer += 1
        for v in adjacency[u]:
            if v != parent:
                children[u].append(v)
                dfs(v, u)
        tout[u] = timer - 1

    dfs(0, -1)
    return tin, tout, euler, children


def subtree_distinct_counts_mo(n, edges, colors, queries):
    tin, tout, euler, _ = euler_tour(n, edges)
    block = max(1, int(n ** 0.5))

    def order_key(index):
        slot = tin[queries[index]] // block
        return (slot, tout[queries[index]] if slot % 2 == 0 else -tout[queries[index]])

    counts = {}
    distinct = 0
    current_left, current_right = 0, -1
    answers = [0] * len(queries)

    def add(position):
        nonlocal distinct
        color = colors[euler[position]]
        counts[color] = counts.get(color, 0) + 1
        if counts[color] == 1:
            distinct += 1

    def remove(position):
        nonlocal distinct
        color = colors[euler[position]]
        counts[color] -= 1
        if counts[color] == 0:
            distinct -= 1

    for index in sorted(range(len(queries)), key=order_key):
        left = tin[queries[index]]
        right = tout[queries[index]]
        while current_left > left:
            current_left -= 1
            add(current_left)
        while current_right < right:
            current_right += 1
            add(current_right)
        while current_left < left:
            remove(current_left)
            current_left += 1
        while current_right > right:
            remove(current_right)
            current_right -= 1
        answers[index] = distinct
    return answers


def naive_subtree_distinct(n, edges, colors, queries):
    _, _, _, children = euler_tour(n, edges)
    answers = []
    for u in queries:
        seen = set()
        stack = [u]
        while stack:
            node = stack.pop()
            seen.add(colors[node])
            stack.extend(children[node])
        answers.append(len(seen))
    return answers


def main():
    edges = [(0, 1), (0, 2), (1, 3)]
    colors = [0, 1, 0, 1]
    assert subtree_distinct_counts_mo(4, edges, colors, [0, 1, 2, 3]) == [2, 1, 1, 1]
    assert subtree_distinct_counts_mo(4, edges, colors, [3, 2, 1, 0]) == [1, 1, 1, 2]
    assert subtree_distinct_counts_mo(4, edges, colors, [2, 0, 2, 0]) == [1, 2, 1, 2]
    assert subtree_distinct_counts_mo(4, edges, [5, 5, 5, 5], [0, 1, 2, 3]) == [1, 1, 1, 1]
    assert subtree_distinct_counts_mo(1, [], [9], [0]) == [1]

    for seed in (430, 24680, 13579):
        local = seed
        for _ in range(15):
            local = (local * 75 + 74) % 65537
            n = 2 + local % 49
            edges = []
            for node in range(1, n):
                local = (local * 75 + 74) % 65537
                edges.append((local % node, node))
            colors = []
            for _ in range(n):
                local = (local * 75 + 74) % 65537
                colors.append(local % (2 + seed % 6))
            queries = list(range(n))
            for _ in range(12):
                local = (local * 75 + 74) % 65537
                queries.append(local % n)
            got = subtree_distinct_counts_mo(n, edges, colors, queries)
            want = naive_subtree_distinct(n, edges, colors, queries)
            assert got == want, (seed, n, got, want)

    # a mostly-unique palette makes nearly every subtree fully distinct
    local = 123123
    for _ in range(10):
        local = (local * 75 + 74) % 65537
        n = 2 + local % 49
        edges = []
        for node in range(1, n):
            local = (local * 75 + 74) % 65537
            edges.append((local % node, node))
        colors = []
        for _ in range(n):
            local = (local * 75 + 74) % 65537
            colors.append(local % 1000)
        queries = list(range(n))
        assert subtree_distinct_counts_mo(n, edges, colors, queries) == naive_subtree_distinct(n, edges, colors, queries), n


if __name__ == "__main__":
    main()
    print("PASS")
