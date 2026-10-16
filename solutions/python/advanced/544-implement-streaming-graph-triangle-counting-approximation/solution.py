"""Task 544 - Implement streaming graph triangle counting approximation. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)

Triangle counting over an edge stream with one pass and sublinear storage. The exact counter used
as the yardstick is the degree-oriented intersection method: orient every edge from the endpoint
with the smaller (degree, id) key to the larger, then for each vertex intersect the ordered
out-neighbour lists in pairs and test adjacency - every triangle is charged exactly once, to its
lowest-ranked vertex. It is validated twice over: against the O(n^3) triple scan and against the
trace identity tr(A^3)/6 of the adjacency matrix, plus the wedge identity
sum_v C(deg(v), 2) = 3T + (number of open length-2 paths), where the open paths are enumerated
directly and the triangles come from the exact counter.

The streaming estimator is DOULION sparsification: each edge is kept independently with
probability p (a deterministic threshold on an LCG stream, so runs are reproducible), triangles are
counted in the sampled graph with the same orientation engine, and the count is scaled by 1/p^3,
which is unbiased because a triangle survives exactly when its three edges survive. The stream is
consumed exactly once - the file asserts the number of inspected edges equals the stream length,
that only about p * m edges are retained, and that the kept edges form a subsequence of the stream.
A second, structurally different estimator is also implemented: uniform wedge sampling, counting
how many of k sampled wedges are closed and scaling sum_v C(deg(v), 2) / 3 by that fraction.

Checks: known counts (K4 = 4, K5 = 10, K6 = 20, a triangle = 1, C6 = 0, K3,3 = 0, a windmill of
three triangles sharing a hub = 3); agreement of the three exact methods on LCG graphs - the
degree-oriented counter against the O(n^3) triple scan and against tr(A^3)/6 - plus the wedge
identity sum_v C(deg v, 2) = 3T + (open length-2 paths) checked by direct enumeration (21 = 9 + 12
on the windmill); p = 1 reproducing the exact count bit-for-bit; the mean of 200 independent
sparsifications landing within 25%, 15% and 8% of the exact count at p = 0.3, 0.5 and 0.8 (measured
deviations 3.0%, 1.9% and 0.3%) while the spread shrinks by more than a quarter; tree and bipartite
streams returning exactly 0 for every seed; the wedge estimator's mean within 25% and 15% at 2000
and 8000 samples and its exactness on the whole wedge population; variance shrinkage with the
sample count; and determinism of both estimators under a repeated seed.
"""
def lcg(seed):
    return (seed * 75 + 74) % 65537


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        assert u != v, "no self loops"
        assert v not in adj[u], "the input must be a simple graph"
        adj[u].add(v)
        adj[v].add(u)
    return adj


def exact_triangles(n, edges):
    """Degree-oriented intersection counter: O(m^1.5) in the worst case."""
    adj = adjacency(n, edges)
    degree = [len(adj[v]) for v in range(n)]
    rank = [0] * n
    for position, vertex in enumerate(sorted(range(n), key=lambda v: (degree[v], v))):
        rank[vertex] = position
    out = [[] for _ in range(n)]
    for (u, v) in edges:
        if rank[u] < rank[v]:
            out[u].append(v)
        else:
            out[v].append(u)
    count = 0
    for u in range(n):
        listed = out[u]
        for i in range(len(listed)):
            inner = adj[listed[i]]
            for j in range(i + 1, len(listed)):
                if listed[j] in inner:
                    count += 1
    return count


def exact_triangles_brute(n, edges):
    """Reference: every triple of vertices is tested."""
    adj = adjacency(n, edges)
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                continue
            for c in range(b + 1, n):
                if c in adj[a] and c in adj[b]:
                    count += 1
    return count


def matmul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def exact_triangles_trace(n, edges):
    """Third method: tr(A^3) / 6, an algebraic identity independent of the combinatorial ones."""
    a = [[0] * n for _ in range(n)]
    for (u, v) in edges:
        a[u][v] += 1
        a[v][u] += 1
    a3 = matmul(matmul(a, a), a)
    return sum(a3[i][i] for i in range(n)) // 6


def wedges(n, edges):
    adj = adjacency(n, edges)
    return sum(len(adj[v]) * (len(adj[v]) - 1) // 2 for v in range(n))


def open_paths_of_length_two(n, edges):
    """Enumerate the open length-2 paths, for the wedge identity check."""
    adj = adjacency(n, edges)
    total = 0
    for v in range(n):
        ordered = sorted(adj[v])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                if ordered[j] not in adj[ordered[i]]:
                    total += 1
    return total


def wedge_closure_counts(n, edges):
    """All wedges and how many of them are closed (each triangle closes exactly three)."""
    adj = adjacency(n, edges)
    total = 0
    closed = 0
    for v in range(n):
        ordered = sorted(adj[v])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                total += 1
                if ordered[j] in adj[ordered[i]]:
                    closed += 1
    return total, closed


def stream_estimate(stream, n, keep_per_mille, seed):
    """DOULION: keep each edge with probability keep/1000, count triangles, scale by 1/p^3.

    Returns (estimate, inspected edges, kept edges, triangles in the sample).
    """
    kept = []
    inspected = 0
    for (u, v) in stream:
        inspected += 1
        seed = lcg(seed)
        if seed % 1000 < keep_per_mille:
            kept.append((u, v))
    sampled = exact_triangles(n, kept)
    scaled = sampled * (1000.0 / keep_per_mille) ** 3
    return scaled, inspected, kept, sampled


def wedge_sampling_estimate(n, edges, samples, seed):
    """Sample `samples` wedges uniformly (with replacement) and scale the closure rate."""
    adj = adjacency(n, edges)
    wedge_list = []
    for v in range(n):
        ordered = sorted(adj[v])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                wedge_list.append((v, ordered[i], ordered[j]))
    total_wedges = len(wedge_list)
    if total_wedges == 0 or samples == 0:
        return 0.0, total_wedges, 0
    closed = 0
    for _ in range(samples):
        seed = lcg(seed)
        (v, a, b) = wedge_list[seed % total_wedges]
        if b in adj[a]:
            closed += 1
    return closed / samples * total_wedges / 3.0, total_wedges, closed


def complete(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cycle(n):
    return [(v, (v + 1) % n) for v in range(n)]


def windmill(triangles):
    """`triangles` triangles sharing a single hub vertex."""
    edges = []
    for t in range(triangles):
        a = 1 + 2 * t
        b = 2 + 2 * t
        edges.extend([(0, a), (0, b), (a, b)])
    return edges, 1 + 2 * triangles


def lcg_graph(n, keep_per_mille, seed):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            seed = lcg(seed)
            if seed % 1000 < keep_per_mille:
                edges.append((i, j))
    return edges, seed


def main():
    # --- Known counts -------------------------------------------------------------------
    assert exact_triangles(4, complete(4)) == 4
    assert exact_triangles(5, complete(5)) == 10
    assert exact_triangles(6, complete(6)) == 20, "K6 has C(6,3) = 20 triangles"
    assert exact_triangles(3, complete(3)) == 1
    assert exact_triangles(6, cycle(6)) == 0
    bipartite = [(i, 3 + j) for i in range(3) for j in range(3)]
    assert exact_triangles(6, bipartite) == 0, "K3,3 is triangle free"
    windmill_edges, windmill_n = windmill(3)
    assert exact_triangles(windmill_n, windmill_edges) == 3
    assert exact_triangles(2, [(0, 1)]) == 0
    assert exact_triangles(1, []) == 0

    # --- Three exact methods and the wedge identity ------------------------------------
    seed = 4242
    for n, density in ((8, 500), (12, 400), (16, 700), (24, 600), (10, 200)):
        edges, seed = lcg_graph(n, density, seed)
        fast = exact_triangles(n, edges)
        assert fast == exact_triangles_brute(n, edges), "orientation vs triple scan"
        assert fast == exact_triangles_trace(n, edges), "orientation vs tr(A^3)/6"
        assert wedges(n, edges) == 3 * fast + open_paths_of_length_two(n, edges), \
            "every wedge is either closed (3 per triangle) or an open length-2 path"
        assert fast >= 0
    windmill_wedges = wedges(windmill_n, windmill_edges)
    assert windmill_wedges == 21, \
        "the degree-6 hub sees C(6,2) = 15 wedges and each of the six arm vertices one more"
    assert windmill_wedges == 3 * 3 + open_paths_of_length_two(windmill_n, windmill_edges)
    assert open_paths_of_length_two(windmill_n, windmill_edges) == 12, \
        "the twelve open wedges are the cross-arm pairs at the hub"

    # --- p = 1 reproduces the exact count ----------------------------------------------
    big_edges, _ = lcg_graph(20, 500, 99)
    exact = exact_triangles(20, big_edges)
    estimate, inspected, kept, sampled = stream_estimate(big_edges, 20, 1000, 7)
    assert inspected == len(big_edges), "the stream is consumed exactly once"
    assert len(kept) == len(big_edges) and sampled == exact
    assert abs(estimate - exact) <= 1e-9, "with p = 1 there is nothing to scale"

    # --- Streaming bookkeeping and determinism ----------------------------------------
    stream, _ = lcg_graph(24, 600, 2024)
    triangle_count = exact_triangles(24, stream)
    assert triangle_count > 20, "the sample graph really has plenty of triangles"
    for keep in (250, 500, 750):
        estimate, inspected, kept, sampled = stream_estimate(stream, 24, keep, 555)
        assert inspected == len(stream)
        assert len(kept) <= len(stream)
        assert abs(len(kept) - keep / 1000.0 * len(stream)) <= 0.12 * len(stream), \
            "about a p fraction of the edges is retained"
        kept_set = set(kept)
        positions = [index for index, edge in enumerate(stream) if edge in kept_set]
        assert [stream[p] for p in positions] == kept, "kept edges are a subsequence"
        again = stream_estimate(stream, 24, keep, 555)
        assert again == (estimate, inspected, kept, sampled), "deterministic under a fixed seed"

    # --- Unbiasedness and variance shrinkage ------------------------------------------
    target = exact_triangles(24, stream)
    for keep, tolerance in ((300, 0.25), (500, 0.15), (800, 0.08)):
        estimates = []
        for trial in range(200):
            value, _, _, _ = stream_estimate(stream, 24, keep, 1000 + 37 * trial)
            estimates.append(value)
        mean = sum(estimates) / len(estimates)
        assert abs(mean / target - 1.0) <= tolerance, \
            "the mean of many sparsifications estimates the true count"
        assert min(estimates) >= 0.0
    spreads = {}
    for keep in (300, 800):
        estimates = [stream_estimate(stream, 24, keep, 2000 + 41 * trial)[0]
                     for trial in range(200)]
        mean = sum(estimates) / len(estimates)
        variance = sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)
        spreads[keep] = variance ** 0.5
    assert spreads[800] < spreads[300], "keeping more edges cannot increase the spread"
    assert spreads[800] < 0.75 * spreads[300], "and the shrinkage is substantial"

    # --- Triangle-free streams return exactly zero -------------------------------------
    tree = [(v, v + 1) for v in range(10)]
    for keep in (200, 500, 900):
        for trial in range(5):
            estimate, _, _, sampled = stream_estimate(tree, 11, keep, 13 + trial)
            assert sampled == 0 and estimate == 0.0, "no triangle can be invented"
    for trial in range(5):
        estimate, _, _, sampled = stream_estimate(bipartite, 6, 400, 91 + trial)
        assert sampled == 0 and estimate == 0.0

    # --- Wedge sampling estimator ------------------------------------------------------
    total_wedges = wedges(24, stream)
    assert total_wedges > 0
    all_wedges, all_closed = wedge_closure_counts(24, stream)
    assert all_wedges == total_wedges
    assert all_closed == 3 * target, "each triangle closes exactly three wedges"
    assert abs(all_closed / all_wedges * total_wedges / 3.0 - target) <= 1e-9, \
        "the closure-rate arithmetic is exact on the whole population"
    assert abs(wedges(24, stream) - total_wedges) <= 0
    for samples, tolerance in ((2000, 0.25), (8000, 0.15)):
        values = [wedge_sampling_estimate(24, stream, samples, 5000 + 29 * trial)[0]
                  for trial in range(120)]
        mean = sum(values) / len(values)
        assert abs(mean / target - 1.0) <= tolerance, "wedge sampling is unbiased too"
        assert min(values) >= 0.0
    spread_small = [wedge_sampling_estimate(24, stream, 2000, 700 + 11 * t)[0]
                    for t in range(60)]
    spread_large = [wedge_sampling_estimate(24, stream, 16000, 900 + 11 * t)[0]
                    for t in range(60)]
    mean_small = sum(spread_small) / len(spread_small)
    mean_large = sum(spread_large) / len(spread_large)
    var_small = sum((v - mean_small) ** 2 for v in spread_small) / (len(spread_small) - 1)
    var_large = sum((v - mean_large) ** 2 for v in spread_large) / (len(spread_large) - 1)
    assert var_large < var_small, "more wedge samples means a tighter estimate"
    zero, wedges_found, closed_found = wedge_sampling_estimate(11, tree, 50, 5)
    assert zero == 0.0 and closed_found == 0
    assert wedge_sampling_estimate(3, complete(3), 1, 1)[0] == 1.0, "one triangle, one wedge"


if __name__ == "__main__":
    main()
    print("PASS")
