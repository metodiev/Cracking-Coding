"""Task 427 - Implement rerooting DP technique on trees. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Rerooting example: the sum of distances from every node to all other nodes. The downward
pass computes subtree sizes and subtree distance sums; the rerooting pass slides the root
across each edge with ans[child] = ans[parent] + n - 2 * subtree_size[child] (the child's
own subtree gets one step closer, every other node one step farther). All n values are
checked against a BFS brute force from every node on LCG trees.
"""


def sum_of_distances_from_each_node(n, edges):
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
    size = [1] * n
    down = [0] * n                          # sum of distances from u to its subtree nodes
    for u in reversed(order):
        for v in adjacency[u]:
            if v != parent[u]:
                size[u] += size[v]
                down[u] += down[v] + size[v]
    result = [0] * n
    result[0] = down[0]
    for u in order:                         # reroot across every tree edge
        for v in adjacency[u]:
            if v != parent[u]:
                result[v] = result[u] + n - 2 * size[v]
    return result


def brute_force(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    totals = []
    for source in range(n):
        distance = [-1] * n
        distance[source] = 0
        queue = [source]
        for u in queue:
            for v in adjacency[u]:
                if distance[v] == -1:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        totals.append(sum(distance))
    return totals


def lcg_trees(count, seed, max_n=60):
    trees = []
    for _ in range(count):
        n = 2 + seed % (max_n - 1)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            edges.append((seed % node, node))
        trees.append((n, edges))
    return trees


def main():
    assert sum_of_distances_from_each_node(1, []) == [0]
    assert sum_of_distances_from_each_node(2, [(0, 1)]) == [1, 1]
    assert sum_of_distances_from_each_node(3, [(0, 1), (1, 2)]) == [3, 2, 3]
    assert sum_of_distances_from_each_node(4, [(i, i + 1) for i in range(3)]) == [6, 4, 4, 6]
    assert sum_of_distances_from_each_node(4, [(0, 1), (0, 2), (0, 3)]) == [3, 5, 5, 5]
    assert sum_of_distances_from_each_node(4, [(0, 1), (0, 2), (1, 3)]) == [4, 4, 6, 6]

    for n, edges in lcg_trees(30, 427):
        got = sum_of_distances_from_each_node(n, edges)
        expected = brute_force(n, edges)
        assert got == expected, (n, edges, got, expected)


if __name__ == "__main__":
    main()
    print("PASS")
