"""Task 400 - Implement heavy-light decomposition basics for path queries. (Tier: Intermediate, CTCI Ch 4)

HLD labels every node with a chain head, a position and a depth. A u-v path decomposes into
O(log n) contiguous chain intervals, so a Fenwick tree over the positions answers path sums
in O(log^2 n) and supports point updates. The heavy child is the largest subtree; every light
edge halves the subtree, which bounds the number of chains on any root path. Cross-checked
against a BFS-parent brute force on deterministic random trees.
"""

class Fenwick:
    """Point add, inclusive range sum."""

    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, index, delta):
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    def prefix(self, count):
        total = 0
        i = count
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total

    def range_sum(self, left, right):
        return self.prefix(right + 1) - self.prefix(left)


class HeavyLight:
    """Path sums with point updates: every root path is O(log n) contiguous chains."""

    def __init__(self, n, edges, values):
        adjacency = [[] for _ in range(n)]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        self.parent = [-1] * n
        self.depth = [0] * n
        order = [0]
        for u in order:
            for v in adjacency[u]:
                if v != self.parent[u]:
                    self.parent[v] = u
                    self.depth[v] = self.depth[u] + 1
                    order.append(v)
        self.size = [1] * n
        self.heavy = [-1] * n
        for u in reversed(order):
            best = 0
            for v in adjacency[u]:
                if v != self.parent[u]:
                    self.size[u] += self.size[v]
                    if self.size[v] > best:
                        best = self.size[v]
                        self.heavy[u] = v
        self.head = [0] * n
        self.pos = [0] * n
        position = 0
        for u in order:
            if u == 0 or self.heavy[self.parent[u]] != u:      # starts a new chain
                x = u
                while x != -1:
                    self.head[x] = u
                    self.pos[x] = position
                    position += 1
                    x = self.heavy[x]
        self.values = values[:]
        self.fenwick = Fenwick(n)
        for node in range(n):
            self.fenwick.add(self.pos[node], values[node])

    def update(self, node, value):
        delta = value - self.values[node]
        self.values[node] = value
        self.fenwick.add(self.pos[node], delta)

    def path_sum(self, u, v):
        total = 0
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] < self.depth[self.head[v]]:
                u, v = v, u
            total += self.fenwick.range_sum(self.pos[self.head[u]], self.pos[u])
            u = self.parent[self.head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        total += self.fenwick.range_sum(self.pos[u], self.pos[v])
        return total


def brute_path_sum(n, edges, values, u, v):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = [-1] * n
    parent[u] = u
    queue = [u]
    for x in queue:
        if x == v:
            break
        for y in adjacency[x]:
            if parent[y] == -1:
                parent[y] = x
                queue.append(y)
    total = 0
    x = v
    while x != u:
        total += values[x]
        x = parent[x]
    return total + values[u]


def lcg_trees(count, seed, max_n=50):
    trees = []
    for _ in range(count):
        n = 2 + seed % (max_n - 1)
        edges = []
        values = []
        for node in range(n):
            seed = (seed * 75 + 74) % 65537
            values.append(seed % 20 - 5)
            if node > 0:
                seed = (seed * 75 + 74) % 65537
                edges.append((seed % node, node))
        trees.append((n, edges, values))
    return trees


def main():
    path_edges = [(0, 1), (1, 2), (2, 3)]
    values = [1, 2, 3, 4]
    hld = HeavyLight(4, path_edges, values)
    assert hld.path_sum(0, 3) == 10
    assert hld.path_sum(0, 0) == 1
    assert hld.path_sum(1, 2) == 5

    hld.update(2, 10)
    assert hld.path_sum(0, 3) == 17
    assert hld.path_sum(1, 2) == 12
    assert hld.path_sum(2, 2) == 10

    star = HeavyLight(3, [(0, 1), (0, 2)], [5, 1, 1])
    assert star.path_sum(1, 2) == 7
    assert star.path_sum(0, 0) == 5

    for n, edges, values in lcg_trees(25, seed=400):
        hld = HeavyLight(n, edges, values)
        seed = n
        for _ in range(12):
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            assert hld.path_sum(u, v) == brute_path_sum(n, edges, values, u, v), (n, u, v)
            seed = (seed * 75 + 74) % 65537
            node = seed % n
            seed = (seed * 75 + 74) % 65537
            new_value = seed % 40 - 20
            values = values[:]
            values[node] = new_value
            hld.update(node, new_value)
            assert hld.path_sum(u, v) == brute_path_sum(n, edges, values, u, v), ("after update", n, u, v)


if __name__ == "__main__":
    main()
    print("PASS")
