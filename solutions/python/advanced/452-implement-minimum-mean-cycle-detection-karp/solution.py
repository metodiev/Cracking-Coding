"""Task 452 - Implement minimum mean cycle detection (Karp). (Tier: Advanced, CTCI Ch 4)

Karp's DP: d[k][v] = minimum weight of a walk with EXACTLY k edges ending at v, where d[0][v] = 0
for every v (equivalent to a super-source joined to all vertices). Karp's theorem gives

    lambda* = min_v max_{0 <= k <= n-1} (d[n][v] - d[k][v]) / (n - k)

for every v with d[n][v] finite; if no such v exists the digraph has no cycle. We keep the mean
as an exact reduced fraction so comparisons are integer-exact. Cross-checked by brute-force
enumeration of all simple cycles (each cycle generated once from its minimum-index vertex).

Hand case re-derived (4 nodes): 0->1(1), 1->2(1), 2->3(1), 3->0(1) is a 4-cycle of mean
4/4 = 1; the shortcut 2->0(5) forms the triangle 0->1->2->0 of mean (1+1+5)/3 = 7/3 ~ 2.33,
so the minimum mean is 1 and the longer cycle wins.
"""

from math import gcd

INF = 10 ** 9


def reduce_fraction(num, den):
    g = gcd(abs(num), den)
    return (num // g, den // g)


def frac_less(a, b):
    return a[0] * b[1] < b[0] * a[1]


def frac_greater(a, b):
    return a[0] * b[1] > b[0] * a[1]


def karp_min_mean_cycle(n, edges):
    """Returns the min mean cycle weight as a reduced (num, den) fraction, or None."""
    d = [[INF] * n for _ in range(n + 1)]
    for v in range(n):
        d[0][v] = 0
    for k in range(1, n + 1):
        for u, v, w in edges:
            if d[k - 1][u] < INF:
                cand = d[k - 1][u] + w
                if cand < d[k][v]:
                    d[k][v] = cand
    best = None
    for v in range(n):
        if d[n][v] >= INF:
            continue
        local = None
        for k in range(n):
            if d[k][v] >= INF:
                continue
            cand = reduce_fraction(d[n][v] - d[k][v], n - k)
            if local is None or frac_greater(cand, local):
                local = cand
        if best is None or frac_less(local, best):
            best = local
    return best


def brute_min_mean_cycle(n, edges):
    """Minimum mean over all simple cycles by enumeration (each cycle from its min vertex)."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    best = None

    def walk(path, total):
        nonlocal best
        cur = path[-1]
        for v, w in adj[cur]:
            if v == path[0]:
                cand = reduce_fraction(total + w, len(path))
                if best is None or frac_less(cand, best):
                    best = cand
            elif v > path[0] and v not in path:
                path.append(v)
                walk(path, total + w)
                path.pop()

    for s in range(n):
        walk([s], 0)
    return best


def lcg_digraphs(count, seed, min_n=3, max_n=6):
    """Digraphs with a guaranteed cycle: chain edges, a closing edge and random extras."""
    graphs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        n = min_n + seed % (max_n - min_n + 1)
        edges = []
        for node in range(n - 1):
            seed = (seed * 75 + 74) % 65537
            edges.append((node, node + 1, seed % 15 - 5))
        seed = (seed * 75 + 74) % 65537
        edges.append((n - 1, 0, seed % 15 - 5))
        for _ in range(2 * n):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            if seed % 100 < 20:
                seed = (seed * 75 + 74) % 65537
                edges.append((u, u, seed % 9 - 4))
                continue
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            if u != v:
                edges.append((u, v, seed % 15 - 5))
        graphs.append((n, edges))
    return graphs


def main():
    four_cycle = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (2, 0, 5)]
    assert karp_min_mean_cycle(4, four_cycle) == (1, 1)
    assert brute_min_mean_cycle(4, four_cycle) == (1, 1)

    negative = [(0, 1, -2), (1, 2, -1), (2, 0, -3)]
    assert karp_min_mean_cycle(3, negative) == (-2, 1)
    assert brute_min_mean_cycle(3, negative) == (-2, 1)

    self_loop = [(0, 0, 7)]
    assert karp_min_mean_cycle(1, self_loop) == (7, 1)
    assert brute_min_mean_cycle(1, self_loop) == (7, 1)

    two_cycles = [(0, 1, 2), (1, 0, 2), (1, 2, -1), (2, 1, -1)]
    assert karp_min_mean_cycle(3, two_cycles) == (-1, 1)
    assert brute_min_mean_cycle(3, two_cycles) == (-1, 1)

    half = [(0, 1, 1), (1, 0, 0)]
    assert karp_min_mean_cycle(2, half) == (1, 2)
    assert brute_min_mean_cycle(2, half) == (1, 2)

    dag = [(0, 1, 1), (1, 2, 2)]
    assert karp_min_mean_cycle(3, dag) is None
    assert brute_min_mean_cycle(3, dag) is None

    checked = 0
    for seed in (4521, 4522, 4523, 4524):
        for n, edges in lcg_digraphs(5, seed):
            expected = brute_min_mean_cycle(n, edges)
            got = karp_min_mean_cycle(n, edges)
            assert expected is not None, (n, edges)
            assert got == expected, (n, edges, got, expected)
            checked += 1
    assert checked == 20


if __name__ == "__main__":
    main()
    print("PASS")
