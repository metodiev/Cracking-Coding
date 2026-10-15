"""Task 275 - Maximum bipartite matching (Hopcroft-Karp). (Tier: Intermediate, CTCI Ch 4)

BFS builds the alternating layer graph, DFS augments along vertex-disjoint shortest paths,
so one phase per phase-number instead of one augment per pass. Validated against the simpler
Kuhn augmenting-path algorithm on deterministic pseudo-random graphs.
"""

from collections import deque


def hopcroft_karp(nl, adj, nr):
    """Returns (matching size, list of (left, right) pairs)."""
    pair_u = [-1] * nl
    pair_v = [-1] * nr
    dist = [-1] * nl

    def bfs():
        queue = deque()
        for u in range(nl):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        reachable = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = pair_v[v]
                if w == -1:
                    reachable = True
                elif dist[w] == -1:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return reachable

    def dfs(u):
        for v in adj[u]:
            w = pair_v[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(nl):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching, [(u, pair_u[u]) for u in range(nl) if pair_u[u] != -1]


def kuhn(nl, adj, nr):
    """Simple O(V*E) augmenting-path oracle."""
    match_v = [-1] * nr

    def try_kuhn(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_v[v] == -1 or try_kuhn(match_v[v], visited):
                    match_v[v] = u
                    return True
        return False

    count = 0
    for u in range(nl):
        if try_kuhn(u, [False] * nr):
            count += 1
    return count


def is_valid_matching(pairs, adj, nl, nr):
    lefts, rights = set(), set()
    for u, v in pairs:
        if not (0 <= u < nl and 0 <= v < nr):
            return False
        if u in lefts or v in rights:
            return False
        if v not in adj[u]:
            return False
        lefts.add(u)
        rights.add(v)
    return True


def random_graphs(nl, nr, count, seed=12345):
    """Deterministic LCG so every language generates the same graphs."""
    for _ in range(count):
        adj = [[] for _ in range(nl)]
        for u in range(nl):
            for v in range(nr):
                seed = (seed * 75 + 74) % 65537
                if seed % 3 == 0:
                    adj[u].append(v)
        yield adj


def main():
    k33 = [[0, 1, 2]] * 3
    assert hopcroft_karp(3, k33, 3)[0] == 3
    assert hopcroft_karp(3, [[], [], []], 3)[0] == 0
    assert hopcroft_karp(3, [[0, 1], [0], []], 2)[0] == 2
    assert hopcroft_karp(3, [[0], [0], []], 1)[0] == 1
    assert hopcroft_karp(3, [[0], [0, 1], [1]], 2)[0] == 2
    assert hopcroft_karp(3, [[0, 1], [1], [2]], 3)[0] == 3
    assert hopcroft_karp(2, [[0, 1], [0]], 2)[0] == 2          # greedy would stop at 1
    assert hopcroft_karp(2, [[0, 1, 2, 3, 4], [0]], 5)[0] == 2

    size, pairs = hopcroft_karp(3, k33, 3)
    assert is_valid_matching(pairs, k33, 3, 3) and len(pairs) == size

    for adj in random_graphs(6, 6, 200):
        size, pairs = hopcroft_karp(6, adj, 6)
        assert is_valid_matching(pairs, adj, 6, 6), adj
        assert len(pairs) == size
        assert size == kuhn(6, adj, 6), adj
        assert size <= 6


if __name__ == "__main__":
    main()
    print("PASS")
