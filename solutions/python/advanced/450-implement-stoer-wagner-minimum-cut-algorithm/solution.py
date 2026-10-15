"""Task 450 - Implement stoer-wagner minimum cut algorithm. (Tier: Advanced, CTCI Ch 4)

Deterministic O(n^3) global min cut. Each phase runs a maximum-adjacency search (always add the
still-unadded vertex most tightly connected to the added set), records the connection weight of
the last added vertex as a candidate cut, then merges the last two vertices; the minimum over
all phases is the answer. Hand cases: the unit triangle gives 2 and K4 gives 3 (single-vertex
cuts; every 2+2 split of K4 costs 4). Verified against brute-force enumeration of all vertex
partitions on LCG graphs with n <= 8, including a disconnected case (answer 0) and parallel
edges. Weights are assumed non-negative, as the algorithm requires.
"""


def stoer_wagner(n, edges):
    """Returns (min_cut_weight, one_side_of_the_best_partition)."""
    weight = [[0] * n for _ in range(n)]
    for u, v, w in edges:
        if u == v:
            continue
        weight[u][v] += w
        weight[v][u] += w
    vertices = list(range(n))
    groups = [[v] for v in range(n)]
    best = None
    best_side = None
    while len(vertices) > 1:
        remaining = set(vertices)
        connection = dict.fromkeys(vertices, 0)
        order = []
        while remaining:
            pick = max(remaining, key=lambda v: (connection[v], -v))
            remaining.discard(pick)
            order.append(pick)
            for other in remaining:
                connection[other] += weight[pick][other]
        t = order[-1]
        s = order[-2]
        cut = connection[t]
        if best is None or cut < best:
            best = cut
            best_side = list(groups[t])
        for v in vertices:
            if v != s and v != t:
                weight[s][v] += weight[t][v]
                weight[v][s] = weight[s][v]
        for v in vertices:
            weight[t][v] = 0
            weight[v][t] = 0
        vertices.remove(t)
        groups[s] = groups[s] + groups[t]
    return best, best_side


def brute_min_cut(n, edges):
    best = None
    for mask in range(1, 1 << (n - 1)):
        cut = 0
        for u, v, w in edges:
            bu = (mask >> (u - 1)) & 1 if u > 0 else 0
            bv = (mask >> (v - 1)) & 1 if v > 0 else 0
            if bu != bv:
                cut += w
        if best is None or cut < best:
            best = cut
    return best


def crossing_weight(edges, side):
    on_side = set(side)
    return sum(w for u, v, w in edges if (u in on_side) != (v in on_side))


def lcg_graphs(count, seed, max_n, connected):
    graphs = []
    for _ in range(count):
        n = 2 + seed % (max_n - 1)
        edges = []
        start = 1 if connected else 0
        for node in range(start, n):
            seed = (seed * 75 + 74) % 65537
            parent = seed % node if node > 0 else 0
            seed = (seed * 75 + 74) % 65537
            edges.append((node, parent, 1 + seed % 20))
        for _ in range(n + 2):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            w = 1 + seed % 20
            if u != v:
                edges.append((u, v, w))
            elif n == 1:
                edges.append((0, 0, w))
        graphs.append((n, edges))
    return graphs


def main():
    triangle = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
    k4 = [(u, v, 1) for u in range(4) for v in range(u + 1, 4)]
    assert brute_min_cut(3, triangle) == 2
    assert brute_min_cut(4, k4) == 3
    value, side = stoer_wagner(3, triangle)
    assert value == 2
    assert crossing_weight(triangle, side) == 2
    value, side = stoer_wagner(4, k4)
    assert value == 3
    assert crossing_weight(k4, side) == 3

    cycle4 = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
    assert stoer_wagner(4, cycle4)[0] == 2
    assert brute_min_cut(4, cycle4) == 2

    disjoint = [(0, 1, 5), (1, 2, 5), (0, 2, 5), (3, 4, 7), (4, 5, 7), (3, 5, 7)]
    assert brute_min_cut(6, disjoint) == 0
    value, side = stoer_wagner(6, disjoint)
    assert value == 0
    assert crossing_weight(disjoint, side) == 0

    parallel = [(0, 1, 3), (1, 0, 4), (0, 1, 5)]
    assert stoer_wagner(2, parallel)[0] == 12
    weighted_triangle = [(0, 1, 5), (1, 2, 5), (0, 2, 5)]
    assert stoer_wagner(3, weighted_triangle)[0] == 10

    checked = 0
    seeds = (450, 451, 452, 453, 454, 455, 456, 457)
    for conn in (True, False):
        for n, edges in lcg_graphs(8, seeds[0 if conn else 1], 8, conn):
            # enlarge the stream deterministically per graph so graphs differ
            expected = brute_min_cut(n, edges)
            value, side = stoer_wagner(n, edges)
            assert crossing_weight(edges, side) == value
            assert value == expected, (conn, n, edges)
            checked += 1
    assert checked == 16
    for extra in seeds[2:]:
        for n, edges in lcg_graphs(3, extra, 7, True):
            assert stoer_wagner(n, edges)[0] == brute_min_cut(n, edges)
            checked += 1
    assert checked == 34


if __name__ == "__main__":
    main()
    print("PASS")
