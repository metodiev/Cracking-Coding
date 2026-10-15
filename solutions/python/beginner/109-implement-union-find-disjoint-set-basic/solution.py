"""Task 109 - Implement union-find / disjoint set (basic). (Tier: Beginner, CTCI Ch 4)

Starts from the textbook parent array with plain linking (root of a is hung
under root of b), then adds union-by-rank so the taller tree keeps the root
and depths stay logarithmic.  find is a bare pointer walk; path compression
arrives in task 110.
"""


class Dsu:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Plain linking: hang a's root under b's root."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        self.components -= 1
        return True

    def union_by_rank(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
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
    d = Dsu(6)
    assert d.count() == 6
    assert d.find(3) == 3
    assert d.connected(0, 0)
    assert d.union(0, 1) is True
    assert d.union(1, 2) is True
    assert d.count() == 4
    assert d.connected(0, 2)
    assert d.depth(0) == 2                        # plain linking leaves a chain
    assert d.union(0, 2) is False                 # already connected
    assert d.count() == 4
    assert d.union(5, 5) is False                 # self union is a no-op

    assert d.union_by_rank(3, 4) is True
    assert d.union_by_rank(4, 5) is True
    assert d.connected(3, 5)
    assert not d.connected(2, 3)
    assert d.count() == 2

    r = Dsu(8)
    for i in range(7):
        assert r.union_by_rank(i, i + 1) is True
    assert r.count() == 1
    assert max(r.depth(i) for i in range(8)) <= 1  # rank keeps the tree flat

    one = Dsu(1)
    assert one.count() == 1 and one.find(0) == 0 and one.depth(0) == 0


if __name__ == "__main__":
    main()
    print("PASS")
