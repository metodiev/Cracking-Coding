"""Task 110 - Path compression in union-find. (Tier: Beginner, CTCI Ch 4)

find is a two-pass iterative search: walk to the root, then re-point every
node on the walked path straight at that root.  A deliberately deep chain
collapses to depth 1, so later finds do near-constant work.
"""


class Dsu:
    def __init__(self, n):
        self.parent = list(range(n))
        self.components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # second pass flattens the path
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        """Plain linking on purpose, so deep chains can form."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.components -= 1
        return True

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def count(self):
        return self.components

    def depth(self, x):
        d = 0
        while self.parent[x] != x:
            x = self.parent[x]
            d += 1
        return d


def main():
    d = Dsu(9)
    assert d.count() == 9
    for i in range(7):
        assert d.union(i, i + 1) is True
    assert d.count() == 2
    assert d.depth(0) == 7                     # plain linking built a long chain
    assert d.find(0) == 7
    assert d.depth(0) == 1                     # find flattened the whole chain
    for i in range(1, 7):
        assert d.depth(i) == 1
    assert d.depth(7) == 0
    assert d.connected(0, 6)
    assert d.find(0) == d.find(0)              # idempotent
    assert d.depth(0) == 1
    assert d.union(7, 8) is True
    assert d.count() == 1
    assert d.find(0) == 8
    assert d.depth(0) <= 2                     # near-constant after compression
    assert d.find(0) == 8 and d.depth(0) == 1  # compressed onto the new root
    assert max(d.depth(i) for i in range(9)) <= 2

    one = Dsu(1)
    assert one.count() == 1 and one.depth(0) == 0 and one.find(0) == 0

    flat = Dsu(3)
    assert flat.union(0, 1) and flat.union(1, 2)
    assert flat.depth(0) == 2                  # chain of three before the find
    assert flat.find(0) == 2
    assert flat.depth(0) == 1 and flat.depth(1) == 1


if __name__ == "__main__":
    main()
    print("PASS")
