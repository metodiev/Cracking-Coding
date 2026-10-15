"""Task 419 - Implement dynamic tree (link-cut tree) basic operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A link-cut tree represents a forest of rooted trees with splay trees over preferred
paths. Nodes are 1-based here, 0 is the null child. Arrays: left/right/parent (the
parent entry doubles as the splay parent and the path-parent), rev (lazy subtree flip),
val (node value) and total (splay-subtree aggregate = path sum of the preferred path).

Splay details:
  * is_root(x): parent[x] == 0 or x is neither the left nor the right child of parent[x]
    — that is where a preferred path ends and a path-parent edge starts,
  * every splay first pushes the rev flags from the splay root down to x, so rotations
    never see stale orientations,
  * rotations fix three families of links (b-subtree, p, g) and pull p then x.
Operations: access (make the x-root path preferred), makeroot (access + rev toggle),
findroot (access + descend leftmost + splay), link/cut (both guarded), connected, and
path_sum(u, v) = makeroot(u) + access(v) -> total[v]. Ops in the tests stay in the
hundreds; every result is validated against an explicit adjacency+union-find oracle.
"""


class LinkCutTree:
    def __init__(self, values):
        n = len(values)
        self.left = [0] * (n + 1)
        self.right = [0] * (n + 1)
        self.parent = [0] * (n + 1)
        self.rev = [False] * (n + 1)
        self.val = [0] + list(values)
        self.total = [0] + list(values)

    def _is_root(self, x):
        p = self.parent[x]
        return p == 0 or (self.left[p] != x and self.right[p] != x)

    def _push(self, x):
        if self.rev[x]:
            self.left[x], self.right[x] = self.right[x], self.left[x]
            if self.left[x]:
                self.rev[self.left[x]] = not self.rev[self.left[x]]
            if self.right[x]:
                self.rev[self.right[x]] = not self.rev[self.right[x]]
            self.rev[x] = False

    def _pull(self, x):
        self.total[x] = self.total[self.left[x]] + self.total[self.right[x]] + self.val[x]

    def _rotate(self, x):
        p = self.parent[x]
        g = self.parent[p]
        if self.left[p] == x:
            b = self.right[x]
            self.right[x] = p
            self.left[p] = b
        else:
            b = self.left[x]
            self.left[x] = p
            self.right[p] = b
        if b:
            self.parent[b] = p
        self.parent[p] = x
        self.parent[x] = g
        if g:
            if self.left[g] == p:
                self.left[g] = x
            elif self.right[g] == p:
                self.right[g] = x
        self._pull(p)
        self._pull(x)

    def _splay(self, x):
        stack = [x]
        y = x
        while not self._is_root(y):
            y = self.parent[y]
            stack.append(y)
        for v in reversed(stack):
            self._push(v)
        while not self._is_root(x):
            p = self.parent[x]
            g = self.parent[p]
            if not self._is_root(p):
                if (self.left[p] == x) == (self.left[g] == p):
                    self._rotate(p)
                else:
                    self._rotate(x)
            self._rotate(x)

    def access(self, x):
        last = 0
        y = x
        while y:
            self._splay(y)
            self.right[y] = last
            if last:
                self.parent[last] = y
            self._pull(y)
            last = y
            y = self.parent[y]
        self._splay(x)

    def makeroot(self, x):
        self.access(x)
        self.rev[x] = not self.rev[x]

    def findroot(self, x):
        self.access(x)
        while True:
            self._push(x)
            if self.left[x] == 0:
                break
            x = self.left[x]
        self._splay(x)
        return x

    def connected(self, u, v):
        if u == v:
            return True
        return self.findroot(u) == self.findroot(v)

    def link(self, u, v):
        self.makeroot(u)
        if self.findroot(v) != u:
            self.parent[u] = v

    def cut(self, u, v):
        self.makeroot(u)
        self.access(v)
        if self.left[v] == u and self.right[u] == 0:
            self.left[v] = 0
            self.parent[u] = 0
            self._pull(v)

    def path_sum(self, u, v):
        self.makeroot(u)
        self.access(v)
        return self.total[v]


class Forest:
    """Oracle: explicit adjacency plus BFS for connectivity and path sums."""

    def __init__(self, n):
        self.adj = {v: set() for v in range(1, n + 1)}

    def link(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)

    def cut(self, u, v):
        self.adj[u].remove(v)
        self.adj[v].remove(u)

    def has_edge(self, u, v):
        return v in self.adj[u]

    def path(self, u, v):
        """Unique forest path u -> v as a vertex list, or None when disconnected."""
        prev = {u: 0}
        queue = [u]
        while queue:
            cur = queue.pop(0)
            if cur == v:
                path = []
                node = v
                while node != 0:
                    path.append(node)
                    node = prev[node]
                return path[::-1]
            for nxt in sorted(self.adj[cur]):
                if nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
        return None

    def connected(self, u, v):
        return self.path(u, v) is not None


class Lcg:
    def __init__(self, state):
        self.state = state

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def main():
    values = [3, 1, 4, 1, 5, 9]
    lct = LinkCutTree(values)

    # isolated vertices: every node is its own root, no paths connect them
    for v in range(1, 7):
        assert lct.findroot(v) == v
    assert not lct.connected(1, 2)
    for (u, v) in [(1, 2), (2, 3), (4, 5)]:
        assert not lct.connected(u, v)

    # link two singletons and read the path
    lct.link(1, 2)
    assert lct.connected(1, 2) and lct.connected(2, 1)
    assert lct.path_sum(1, 2) == 4 and lct.path_sum(2, 1) == 4      # 3 + 1
    assert lct.findroot(2) == lct.findroot(1)
    assert lct.path_sum(1, 1) == 3

    # extend to a chain 1 - 2 - 3 and query across it
    lct.link(2, 3)
    assert lct.path_sum(1, 3) == 8 == values[0] + values[1] + values[2]
    assert lct.path_sum(3, 1) == 8
    assert lct.path_sum(2, 2) == 1
    # path_sum calls makeroot internally; an explicit makeroot(3) makes 3 the root
    lct.makeroot(3)
    assert lct.findroot(1) == 3 and lct.findroot(3) == 3

    # cutting an existing edge splits the tree; missing edges are left untouched
    lct.cut(2, 3)
    assert not lct.connected(1, 3)
    assert lct.connected(1, 2)
    assert lct.findroot(3) == 3
    lct.cut(1, 3)                     # no such edge: guarded no-op
    assert lct.connected(1, 2) and not lct.connected(1, 3)

    # build two trees, then join them through single edges
    lct.link(3, 4)
    lct.link(5, 6)
    assert lct.connected(3, 4) and lct.connected(5, 6)
    assert not lct.connected(1, 4)
    lct.link(2, 3)                    # re-join {1,2} with {3,4}
    assert lct.connected(1, 4)
    assert lct.path_sum(1, 4) == 3 + 1 + 4 + 1 == 9
    lct.link(4, 5)
    assert lct.connected(1, 6)
    assert lct.path_sum(1, 6) == 3 + 1 + 4 + 1 + 5 + 9 == 23
    assert lct.path_sum(6, 1) == 23
    assert lct.path_sum(3, 5) == 4 + 1 + 5 == 10

    # cutting the bridge splits the forest again
    lct.cut(4, 5)
    assert lct.connected(1, 4) and lct.connected(5, 6)
    assert not lct.connected(1, 5)

    # LCG-driven script against the adjacency oracle (hundreds of operations)
    n = 30
    gen = Lcg(4190)
    vals = []
    for _ in range(n):
        vals.append(1 + gen.next() % 100)
    lct = LinkCutTree(vals)
    forest = Forest(n)

    for step in range(350):
        kind = gen.next() % 4
        if kind == 0:
            u = 1 + gen.next() % n
            v = 1 + gen.next() % n
            if u != v and forest.connected(u, v) and forest.has_edge(u, v):
                lct.cut(u, v)
                forest.cut(u, v)
                assert not lct.connected(u, v), step
        elif kind == 1:
            u = 1 + gen.next() % n
            v = 1 + gen.next() % n
            if u != v and not forest.connected(u, v):
                lct.link(u, v)
                forest.link(u, v)
                assert lct.connected(u, v), step
        elif kind == 2:
            u = 1 + gen.next() % n
            v = 1 + gen.next() % n
            assert lct.connected(u, v) == forest.connected(u, v), step
            if forest.connected(u, v):
                path = forest.path(u, v)
                assert lct.path_sum(u, v) == sum(vals[node - 1] for node in path), step
                assert lct.path_sum(v, u) == lct.path_sum(u, v), step
        else:
            # full pairwise connectivity audit
            for u in range(1, n + 1):
                for v in range(u + 1, n + 1):
                    assert lct.connected(u, v) == forest.connected(u, v), (step, u, v)

    # final audit: every path sum across the whole forest matches the oracle
    checks = 0
    for u in range(1, n + 1):
        for v in range(u, n + 1):
            assert lct.connected(u, v) == forest.connected(u, v)
            if forest.connected(u, v):
                path = forest.path(u, v)
                assert lct.path_sum(u, v) == sum(vals[node - 1] for node in path)
                checks += 1
    assert checks > n  # the script must leave a meaningfully connected forest

    # a star built by links from one hub answers all leaf-to-leaf paths through it
    gen = Lcg(777)
    hub_values = [1 + gen.next() % 9 for _ in range(9)]
    lct = LinkCutTree(hub_values)
    for leaf in range(2, 10):
        lct.link(1, leaf)
    for leaf in range(2, 10):
        assert lct.path_sum(1, leaf) == hub_values[0] + hub_values[leaf - 1]
    for a in range(2, 10):
        for b in range(a + 1, 10):
            want = hub_values[0] + hub_values[a - 1] + hub_values[b - 1]
            assert lct.path_sum(a, b) == want, (a, b)
    # each path query re-roots through makeroot; after makeroot(9), 9 is the tree root
    lct.makeroot(9)
    assert lct.findroot(1) == 9 and lct.findroot(9) == 9

    # cutting every leaf leaves a single node
    for leaf in range(2, 10):
        lct.cut(1, leaf)
    for leaf in range(2, 10):
        assert not lct.connected(1, leaf)
        assert lct.findroot(leaf) == leaf


if __name__ == "__main__":
    main()
    print("PASS")
