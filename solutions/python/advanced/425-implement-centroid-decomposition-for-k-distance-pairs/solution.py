"""Task 425 - Implement centroid decomposition for k-distance pairs. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Counts unordered pairs of nodes whose tree distance is exactly k. At every centroid the
pairs passing through it are counted by inclusion-exclusion: pairs over ALL distances to the
centroid (the centroid itself sits at distance 0) minus the pairs lying inside a single
branch, whose path does not really pass through the centroid. Each pair is decided exactly
once - at the first removed centroid on its path - and the naive distance gathering keeps
the total work at O(n log n). All expectations are derived from the BFS brute force oracle.
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

    def find_centroid(u, parent, component):
        for v in adjacency[u]:
            if v != parent and not removed[v] and size[v] > component // 2:
                return find_centroid(v, u, component)
        return u

    def collect(u, parent, distance, distances):
        distances.append(distance)
        for v in adjacency[u]:
            if v != parent and not removed[v]:
                collect(v, u, distance + 1, distances)

    def pairs_with_sum(distances):
        seen = {}
        pairs = 0
        for d in distances:
            pairs += seen.get(k - d, 0)
            seen[d] = seen.get(d, 0) + 1
        return pairs

    total = 0

    def decompose(start):
        nonlocal total
        sizes(start, -1)
        centroid = find_centroid(start, -1, size[start])
        removed[centroid] = True
        all_distances = [0]                    # the centroid itself
        branches = []
        for v in adjacency[centroid]:
            if removed[v]:
                continue
            distances = []
            collect(v, centroid, 1, distances)
            branches.append(distances)
            all_distances.extend(distances)
        total += pairs_with_sum(all_distances)
        for distances in branches:             # remove pairs inside one branch
            total -= pairs_with_sum(distances)
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
    assert count_pairs_at_distance(0, [], 0) == 0
    assert count_pairs_at_distance(1, [], 0) == 0
    assert count_pairs_at_distance(1, [], 1) == 0
    assert count_pairs_at_distance(2, [(0, 1)], 1) == 1
    assert count_pairs_at_distance(2, [(0, 1)], 0) == 0
    assert count_pairs_at_distance(2, [(0, 1)], 2) == 0

    path3 = [(0, 1), (1, 2)]
    assert count_pairs_at_distance(3, path3, 0) == 0      # no two nodes share a position
    assert count_pairs_at_distance(3, path3, 1) == 2      # {0-1, 1-2}
    assert count_pairs_at_distance(3, path3, 2) == 1      # {0-2}
    assert count_pairs_at_distance(3, path3, 3) == 0

    star = [(0, 1), (0, 2), (0, 3)]
    assert count_pairs_at_distance(4, star, 1) == 3       # centre to each leaf
    assert count_pairs_at_distance(4, star, 2) == 3       # C(3,2) leaf pairs
    assert count_pairs_at_distance(4, star, 3) == 0

    path4 = [(i, i + 1) for i in range(3)]
    assert count_pairs_at_distance(4, path4, 1) == 3
    assert count_pairs_at_distance(4, path4, 2) == 2      # {0-2, 1-3}
    assert count_pairs_at_distance(4, path4, 3) == 1      # {0-3}
    assert count_pairs_at_distance(4, path4, 4) == 0

    path6 = [(i, i + 1) for i in range(5)]
    for k, expected in enumerate([0, 5, 4, 3, 2, 1, 0]):
        assert count_pairs_at_distance(6, path6, k) == expected, k

    for n, edges in lcg_trees(30, 425):
        for k in range(0, 13):
            got = count_pairs_at_distance(n, edges, k)
            expected = brute_force(n, edges, k)
            assert got == expected, (n, edges, k, got, expected)


if __name__ == "__main__":
    main()
    print("PASS")
