"""Task 447 - Implement dynamic MST under edge insertions (conceptual). (Tier: Advanced, CTCI Ch 4)

Edges arrive one at a time and the MST weight must stay correct after each insertion. The
documented incremental rule: if the new edge joins two different components of the current
forest, add it; otherwise the unique tree path between its endpoints plus the new edge forms a
fundamental cycle, and the new edge only belongs to an MST when it is strictly lighter than the
heaviest tree edge on that path, in which case the two swap (cut and link). `DynamicMST` applies
that rule; the test harness recomputes Kruskal from scratch after every insertion and asserts
the weights agree, so the rule is never trusted blindly. O(1) insertions become O(n) path scans
here because graphs in the tests are small; the rule itself is the conceptual core.
"""


class DynamicMST:
    def __init__(self, n):
        self.n = n
        self.edges = []
        self.tree_edges = []
        self.swaps = 0

    def _adjacency(self):
        adjacency = {}
        for u, v, w in self.tree_edges:
            adjacency.setdefault(u, []).append((v, (u, v, w)))
            adjacency.setdefault(v, []).append((u, (u, v, w)))
        return adjacency

    def _heaviest_on_path(self, u, v):
        """Heaviest tree edge on the u-v path, or None when they are in different components."""
        if u == v:
            return None
        adjacency = self._adjacency()
        if u not in adjacency:
            return None
        parent = {u: None}
        stack = [u]
        while stack:
            x = stack.pop()
            for y, edge in adjacency.get(x, []):
                if y not in parent:
                    parent[y] = (x, edge)
                    stack.append(y)
        if v not in parent:
            return None
        heaviest = None
        node = v
        while parent[node] is not None:
            x, edge = parent[node]
            if heaviest is None or edge[2] > heaviest[2]:
                heaviest = edge
            node = x
        return heaviest

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))
        heaviest = self._heaviest_on_path(u, v)
        if heaviest is None:
            self.tree_edges.append((u, v, w))
        elif heaviest[2] > w:
            self.tree_edges.remove(heaviest)
            self.tree_edges.append((u, v, w))
            self.swaps += 1

    def mst_weight(self):
        return sum(w for _, _, w in self.tree_edges)


def kruskal_weight(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[ru] = rv
            total += w
    return total


def lcg_edge_stream(count, seed, n):
    edges = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        u = seed % n
        seed = (seed * 75 + 74) % 65537
        v = seed % n
        if u == v:
            v = (v + 1) % n
        seed = (seed * 75 + 74) % 65537
        w = 1 + seed % 30
        edges.append((u, v, w))
    return edges


def main():
    # Hand case: 0-1 (5), 1-2 (3), 2-3 (7), then 0-3 (1) swaps out the heaviest edge 2-3 (7).
    dynamic = DynamicMST(4)
    dynamic.add_edge(0, 1, 5)
    assert dynamic.mst_weight() == 5
    dynamic.add_edge(1, 2, 3)
    assert dynamic.mst_weight() == 8
    dynamic.add_edge(2, 3, 7)
    assert dynamic.mst_weight() == 15
    dynamic.add_edge(0, 3, 1)
    assert dynamic.mst_weight() == 9
    assert dynamic.swaps == 1
    dynamic.add_edge(0, 2, 9)
    assert dynamic.mst_weight() == 9

    swaps_seen = 0
    for trial in range(25):
        n = 4 + trial % 5
        stream = lcg_edge_stream(2 * n, 900 + trial * 37, n)
        dynamic = DynamicMST(n)
        inserted = []
        for edge in stream:
            dynamic.add_edge(*edge)
            inserted.append(edge)
            assert dynamic.mst_weight() == kruskal_weight(n, inserted), (trial, edge)
        swaps_seen += dynamic.swaps
    assert swaps_seen > 0

    # Over a connected stream the maintained tree stays a spanning forest of n - 1 edges.
    n = 6
    stream = lcg_edge_stream(3 * n, 321, n)
    dynamic = DynamicMST(n)
    for edge in stream:
        dynamic.add_edge(*edge)
    assert dynamic.mst_weight() == kruskal_weight(n, stream)
    assert len(dynamic.tree_edges) == n - 1


if __name__ == "__main__":
    main()
    print("PASS")
