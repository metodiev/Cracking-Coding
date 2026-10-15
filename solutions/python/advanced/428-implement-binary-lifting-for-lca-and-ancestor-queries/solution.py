"""Task 428 - Implement binary lifting for LCA and ancestor queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Roots the tree at 0 (the root's parent is itself) and stores up[j][u] = the 2^j-th ancestor
of u. lca lifts the deeper node to the other's depth, then both climb together with the
largest jumps that keep them apart; kth_ancestor walks the set bits of k and returns -1 when
k exceeds the node's depth. Both are cross-checked against a naive parent walk for every
pair of nodes and every k on LCG trees.
"""


class BinaryLifting:
    def __init__(self, n, edges):
        self.n = n
        self.log = max(1, n.bit_length())
        adjacency = [[] for _ in range(n)]
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        self.depth = [0] * n
        self.parent = [0] * n
        visited = [False] * n
        visited[0] = True
        order = [0]
        for u in order:
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    self.parent[v] = u
                    self.depth[v] = self.depth[u] + 1
                    order.append(v)
        self.up = [self.parent[:]]
        for _ in range(1, self.log):
            previous = self.up[-1]
            self.up.append([previous[previous[u]] for u in range(n)])

    def kth_ancestor(self, node, k):
        if k > self.depth[node]:
            return -1
        bit = 0
        while k:
            if k & 1:
                node = self.up[bit][node]
            k >>= 1
            bit += 1
        return node

    def lca(self, a, b):
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        a = self.kth_ancestor(a, self.depth[a] - self.depth[b])
        if a == b:
            return a
        for j in range(self.log - 1, -1, -1):
            if self.up[j][a] != self.up[j][b]:
                a = self.up[j][a]
                b = self.up[j][b]
        return self.parent[a]


def rooted_tree(n, edges):
    adjacency = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = [0] * n
    depth = [0] * n
    visited = [False] * n
    visited[0] = True
    order = [0]
    for u in order:
        for v in adjacency[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                order.append(v)
    return parent, depth


def naive_lca(parent, depth, a, b):
    while depth[a] > depth[b]:
        a = parent[a]
    while depth[b] > depth[a]:
        b = parent[b]
    while a != b:
        a = parent[a]
        b = parent[b]
    return a


def naive_kth_ancestor(parent, depth, node, k):
    if k > depth[node]:
        return -1
    for _ in range(k):
        node = parent[node]
    return node


def lcg_trees(count, seed, max_n=30):
    trees = []
    for _ in range(count):
        n = 2 + seed % (max_n - 1)
        edges = []
        for node in range(1, n):
            seed = (seed * 75 + 74) % 65537
            edges.append((seed % node, node))
        trees.append((n, edges))
    return trees


def main():
    path = BinaryLifting(4, [(0, 1), (1, 2), (2, 3)])
    assert path.lca(0, 3) == 0
    assert path.lca(2, 3) == 2
    assert path.lca(1, 2) == 1
    assert path.lca(3, 3) == 3
    assert path.lca(0, 0) == 0
    assert path.kth_ancestor(3, 0) == 3
    assert path.kth_ancestor(3, 1) == 2
    assert path.kth_ancestor(3, 2) == 1
    assert path.kth_ancestor(3, 3) == 0
    assert path.kth_ancestor(3, 4) == -1
    assert path.kth_ancestor(0, 1) == -1

    star = BinaryLifting(4, [(0, 1), (0, 2), (0, 3)])
    assert star.lca(1, 2) == 0
    assert star.lca(1, 3) == 0
    assert star.lca(0, 3) == 0
    assert star.lca(2, 2) == 2
    assert star.kth_ancestor(2, 1) == 0
    assert star.kth_ancestor(2, 2) == -1

    single = BinaryLifting(1, [])
    assert single.lca(0, 0) == 0
    assert single.kth_ancestor(0, 0) == 0
    assert single.kth_ancestor(0, 1) == -1

    for n, edges in lcg_trees(25, 428):
        bl = BinaryLifting(n, edges)
        parent, depth = rooted_tree(n, edges)
        for a in range(n):
            for b in range(n):
                expected = naive_lca(parent, depth, a, b)
                assert bl.lca(a, b) == expected, (n, edges, a, b)
                assert bl.lca(a, b) == bl.lca(b, a)
        for u in range(n):
            for k in range(0, depth[u] + 3):
                expected = naive_kth_ancestor(parent, depth, u, k)
                assert bl.kth_ancestor(u, k) == expected, (n, edges, u, k)


if __name__ == "__main__":
    main()
    print("PASS")
