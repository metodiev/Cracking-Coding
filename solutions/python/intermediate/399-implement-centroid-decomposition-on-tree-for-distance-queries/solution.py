"""Task 399 - Implement centroid decomposition for distance queries. (Tier: Intermediate, CTCI Ch 4)

Counts unordered pairs of nodes at distance exactly k. Each centroid, once removed, counts
the pairs whose path passes through it by pairing subtree distance lists against everything
seen before (the centroid itself sits at distance 0). Every pair is counted exactly once -
at the first removed centroid on its path - and the decomposition depth stays O(log n), so
the total work is O(n log n) with the naive distance gathering. Cross-checked against a
BFS-from-every-node brute force.
"""

def count_pairs_at_distance(n, edges, k):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    removed = [False] * n
    size = [0] * n

    def sizes(u, parent):
        size[u] = 1
        for v in adjacency[u]:
            if v != parent and not removed[v]:
                sizes(v, u)
                size[u] += size[v]

    def find_centroid(u, parent, component_size):
        for v in adjacency[u]:
            if v != parent and not removed[v] and size[v] > component_size // 2:
                return find_centroid(v, u, component_size)
        return u

    def collect(u, parent, distance, out):
        out.append(distance)
        for v in adjacency[u]:
            if v != parent and not removed[v]:
                collect(v, u, distance + 1, out)

    total = 0

    def decompose(start):
        nonlocal total
        sizes(start, -1)
        centroid = find_centroid(start, -1, size[start])
        removed[centroid] = True
        seen = {0: 1}                          # the centroid itself is at distance 0
        for v in adjacency[centroid]:
            if removed[v]:
                continue
            subtree = []
            collect(v, centroid, 1, subtree)
            for d in subtree:
                total += seen.get(k - d, 0)
            for d in subtree:
                seen[d] = seen.get(d, 0) + 1
        for v in adjacency[centroid]:
            if not removed[v]:
                decompose(v)

    if n > 0:
        decompose(0)
    return total


def brute_force(n, edges, k):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    total = 0
    for source in range(n):
        distance = [-1] * n
        distance[source] = 0
        queue = [source]
        for u in queue:
            for v in adjacency[u]:
                if distance[v] == -1:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        for target in range(source + 1, n):
            if distance[target] == k:
                total += 1
    return total


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
    path3 = (3, [(0, 1), (1, 2)])
    assert count_pairs_at_distance(3, path3[1], 1) == 2
    assert count_pairs_at_distance(3, path3[1], 2) == 1
    assert count_pairs_at_distance(3, path3[1], 3) == 0

    star = (3, [(0, 1), (0, 2)])
    assert count_pairs_at_distance(3, star[1], 2) == 1
    assert count_pairs_at_distance(3, star[1], 1) == 2

    assert count_pairs_at_distance(1, [], 0) == 0
    assert count_pairs_at_distance(2, [(0, 1)], 1) == 1
    assert count_pairs_at_distance(2, [(0, 1)], 2) == 0

    path6 = (6, [(i, i + 1) for i in range(5)])
    for k in range(0, 7):
        assert count_pairs_at_distance(6, path6[1], k) == brute_force(6, path6[1], k), k

    for n, edges in lcg_trees(30, seed=600):
        for k in range(0, 6):
            got = count_pairs_at_distance(n, edges, k)
            expected = brute_force(n, edges, k)
            assert got == expected, (n, edges, k, got, expected)


if __name__ == "__main__":
    main()
    print("PASS")
