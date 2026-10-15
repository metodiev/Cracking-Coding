
"""Task 178 - Compute transitive closure (Floyd-Warshall naive). (Tier: Beginner, CTCI Ch 4)

reach[i][j] = True iff j is reachable from i. O(n^3) in-place Floyd-Warshall-style
relaxation: reach[i][j] |= reach[i][k] and reach[k][j]. O(n^2) space.
The diagonal starts as identity, so every node reaches itself.
"""


def transitive_closure(adj):
    """adj: square 0/1 adjacency matrix -> boolean reachability matrix."""
    n = len(adj)
    reach = [[bool(adj[i][j]) or i == j for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return reach


def main():
    # chain 0 -> 1 -> 2
    chain = [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
    reach = transitive_closure(chain)
    assert reach[0] == [True, True, True]      # 0 reaches everyone
    assert reach[1] == [False, True, True]
    assert reach[2] == [False, False, True]    # only itself

    # cycle 0 <-> 1, 2 isolated
    cycle = [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]
    reach2 = transitive_closure(cycle)
    assert reach2[0][1] and reach2[1][0] and not reach2[0][2]

    assert transitive_closure([]) == []
    assert transitive_closure([[0]]) == [[True]]           # identity diagonal
    # closure agrees with BFS reachability on a random-ish graph
    import random
    rng = random.Random(3)
    n = 8
    adj = [[1 if i != j and rng.random() < 0.2 else 0 for j in range(n)] for i in range(n)]
    reach3 = transitive_closure(adj)
    for src in range(n):
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u][v] and v not in seen:
                    seen.add(v)
                    stack.append(v)
        assert all(reach3[src][v] == (v in seen) for v in range(n)), src


if __name__ == "__main__":
    main()
    print("PASS")
