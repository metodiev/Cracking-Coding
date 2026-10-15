"""Task 448 - Implement parametric search for optimizing monotone predicates. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Parametric search: when a predicate P(d) over a numeric parameter is monotone (false...false,
then true...true), the smallest d with P(d) true is located by binary searching a sorted
candidate list instead of testing every value. Two example instances on weighted graphs:
P1(d) = "every node is reachable from a source using only edges of weight <= d" (bottleneck
connectivity), P2(d) = "the graph has at most k connected components when only edges of weight
<= d are kept". Both are monotone; the parametric binary search is asserted against a linear
scan over the same candidates, and the monotonicity itself is asserted on the samples.
"""


def parametric_min(candidates, predicate):
    """Smallest candidate c with predicate(c) true, or None when the predicate never fires."""
    lo = 0
    hi = len(candidates)
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return candidates[lo] if lo < len(candidates) else None


def linear_scan(candidates, predicate):
    for candidate in candidates:
        if predicate(candidate):
            return candidate
    return None


def reachable_all(n, edges, d, source):
    adjacency = [[] for _ in range(n)]
    for u, v, w in edges:
        if w <= d:
            adjacency[u].append(v)
            adjacency[v].append(u)
    seen = [False] * n
    seen[source] = True
    stack = [source]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            if not seen[y]:
                seen[y] = True
                stack.append(y)
    return all(seen)


def components_leq(n, edges, d, k):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    for u, v, w in edges:
        if w <= d:
            ru = find(u)
            rv = find(v)
            if ru != rv:
                parent[ru] = rv
                components -= 1
    return components <= k


def lcg_graphs(count, seed, max_n):
    """Connected graphs (random spanning tree plus extra edges) with distinct weights not needed."""
    graphs = []
    for _ in range(count):
        n = 3 + seed % (max_n - 2)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            parent = seed % node
            seed = (seed * 75 + 74) % 65537
            weight = 1 + seed % 30
            edges.append((parent, node, weight))
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            seed = (seed * 75 + 74) % 65537
            weight = 1 + seed % 30
            if u != v:
                edges.append((u, v, weight))
        graphs.append((n, edges))
    return graphs


def assert_monotone(candidates, predicate):
    values = [predicate(c) for c in candidates]
    assert values == sorted(values), values


def main():
    # Hand case: path 0-1 (4), 1-2 (1), 2-3 (7); candidates are the distinct weights.
    edges = [(0, 1, 4), (1, 2, 1), (2, 3, 7)]
    candidates = [1, 4, 7]
    reach = lambda d: reachable_all(4, edges, d, 0)
    comps2 = lambda d: components_leq(4, edges, d, 2)
    comps1 = lambda d: components_leq(4, edges, d, 1)
    assert [reach(d) for d in candidates] == [False, False, True]
    assert [comps2(d) for d in candidates] == [False, True, True]
    assert parametric_min(candidates, reach) == 7
    assert parametric_min(candidates, comps2) == 4
    assert parametric_min(candidates, comps1) == 7
    assert_monotone(candidates, reach)
    assert_monotone(candidates, comps2)

    edge_case = [(0, 1, 5)]
    assert parametric_min([5], lambda d: reachable_all(2, edge_case, d, 0)) == 5
    assert parametric_min([1, 2, 3], lambda d: False) is None
    assert parametric_min([1, 2, 3], lambda d: True) == 1

    checked = 0
    for n, edges in lcg_graphs(20, 448, 12):
        candidates = sorted({w for _, _, w in edges})
        for source in (0, n - 1):
            reach = lambda d, s=source: reachable_all(n, edges, d, s)
            assert_monotone(candidates, reach)
            assert parametric_min(candidates, reach) == linear_scan(candidates, reach)
            checked += 1
        for k in (1, 2, 4):
            comps = lambda d, k=k: components_leq(n, edges, d, k)
            assert_monotone(candidates, comps)
            assert parametric_min(candidates, comps) == linear_scan(candidates, comps)
            checked += 1
    assert checked == 100


if __name__ == "__main__":
    main()
    print("PASS")
