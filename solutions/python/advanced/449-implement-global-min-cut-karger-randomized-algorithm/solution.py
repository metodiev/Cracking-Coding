"""Task 449 - Implement global min-cut (Karger randomized algorithm). (Tier: Advanced, CTCI Ch 4)

Karger's algorithm repeatedly picks a uniformly random edge of the current multigraph and
contracts its endpoints until two supernodes remain; the surviving parallel edges between the
two sides are one cut. A single run succeeds with probability >= 2/(n*(n-1)), so the demo
repeats runs with several FIXED seeds (randomness is the deterministic LCG stream
seed = (seed * 75 + 74) % 65537, documented for reproducibility) and keeps the best cut, which
must equal the true min cut from brute-force enumeration over all vertex partitions; it also
proves the trial minimum is never below that bound. Every individual run's witness is checked:
the two supernode vertex sets partition the graph and the number of crossing original edges
equals the reported value.
"""


class Lcg:
    def __init__(self, seed):
        self.state = seed % 65537

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state

    def below(self, bound):
        return self.next() % bound


def karger_min_cut(n, edges, seed):
    """Returns (cut_value, side_a, side_b); edges are (u, v) pairs, multiplicities respected."""
    rng = Lcg(seed)
    label = list(range(n))
    remaining = [(u, v) for u, v in edges]
    components = n
    while components > 2:
        u, v = remaining[rng.below(len(remaining))]
        a = label[u]
        b = label[v]
        for i in range(n):
            if label[i] == b:
                label[i] = a
        remaining = [(x, y) for x, y in remaining if label[x] != label[y]]
        components -= 1
    side_a = sorted(i for i in range(n) if label[i] == label[0])
    side_b = sorted(i for i in range(n) if label[i] != label[0])
    return len(remaining), side_a, side_b


def brute_min_cut(n, edges):
    """Enumerates all nontrivial partitions (vertex 0 pinned to one side)."""
    best = None
    for mask in range(1, 1 << (n - 1)):
        cut = 0
        for u, v in edges:
            bu = (mask >> (u - 1)) & 1 if u > 0 else 0
            bv = (mask >> (v - 1)) & 1 if v > 0 else 0
            if bu != bv:
                cut += 1
        if best is None or cut < best:
            best = cut
    return best


def check_witness(n, edges, value, side_a, side_b):
    assert sorted(side_a + side_b) == list(range(n))
    assert side_a and side_b
    on_a = set(side_a)
    crossing = sum(1 for u, v in edges if (u in on_a) != (v in on_a))
    assert crossing == value, (crossing, value)


def karger_best_of(n, edges, base_seed, trials, expected):
    """Runs `trials` fixed-seed contractions; returns (best, successes) with assertions."""
    best = None
    successes = 0
    for t in range(trials):
        seed = (base_seed * 31 + t * 17 + 1) % 65537
        value, side_a, side_b = karger_min_cut(n, edges, seed)
        check_witness(n, edges, value, side_a, side_b)
        assert value >= expected, (base_seed, t, value, expected)
        if value == expected:
            successes += 1
        if best is None or value < best:
            best = value
    return best, successes


def lcg_graphs(count, seed, max_n):
    graphs = []
    for _ in range(count):
        n = 3 + seed % (max_n - 2)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            parent = seed % node
            edges.append((node, parent))
        for _ in range(seed % (n + 2)):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            if u != v:
                edges.append((u, v))
        graphs.append((n, edges))
    return graphs


def main():
    triangle = [(0, 1), (1, 2), (0, 2)]
    assert brute_min_cut(3, triangle) == 2
    for seed in (1, 2, 3, 65536):
        value, side_a, side_b = karger_min_cut(3, triangle, seed)
        check_witness(3, triangle, value, side_a, side_b)
        assert value == 2

    k4 = [(u, v) for u in range(4) for v in range(u + 1, 4)]
    assert brute_min_cut(4, k4) == 3
    best, successes = karger_best_of(4, k4, 42, 60, 3)
    assert best == 3
    assert successes > 0

    total_trials = 0
    for n, edges in lcg_graphs(3, 449, 6):
        expected = brute_min_cut(n, edges)
        for base_seed in (1, 42, 1234, 65536):
            best, successes = karger_best_of(n, edges, base_seed, 400, expected)
            assert best == expected, (n, base_seed)
            assert successes > 0, (n, base_seed)
            total_trials += 400
    assert total_trials == 4800

    k6 = [(u, v) for u in range(6) for v in range(u + 1, 6)]
    assert brute_min_cut(6, k6) == 5
    best, successes = karger_best_of(6, k6, 1234, 400, 5)
    assert best == 5
    assert successes > 0

    # Parallel edges raise the min cut: three copies between two vertices.
    doubled = [(0, 1), (1, 0), (0, 1)]
    assert brute_min_cut(2, doubled) == 3
    best, successes = karger_best_of(2, doubled, 7, 25, 3)
    assert best == 3
    assert successes == 25


if __name__ == "__main__":
    main()
    print("PASS")
