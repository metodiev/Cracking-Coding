"""Task 108 - Detect cycle in undirected graph (union-find). (Tier: Beginner, CTCI Ch 4)

An edge between two vertices that already share a component closes a cycle;
otherwise the union merges the components.  Self-loops and parallel edges are
cycles too.
"""


def has_cycle_undirected(n, edges):
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path halving keeps find cheap
            x = parent[x]
        return x

    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint out of range")
        ru, rv = find(u), find(v)
        if ru == rv:
            return True
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
    return False


def main():
    assert has_cycle_undirected(5, [(0, 1), (0, 2), (1, 3), (1, 4)]) is False      # tree
    assert has_cycle_undirected(3, [(0, 1), (1, 2), (2, 0)]) is True               # triangle
    assert has_cycle_undirected(6, [(0, 1), (2, 3), (3, 4), (4, 5)]) is False      # forest
    assert has_cycle_undirected(6, [(0, 1), (2, 3), (3, 4), (4, 5), (5, 2)]) is True
    assert has_cycle_undirected(1, []) is False                                     # single vertex
    assert has_cycle_undirected(2, [(0, 0)]) is True                                # self-loop
    assert has_cycle_undirected(2, [(0, 1), (1, 0)]) is True                        # parallel edges
    try:
        has_cycle_undirected(2, [(0, 5)])
        raise AssertionError("expected ValueError for out-of-range edge")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
