"""Task 111 - Kruskal's MST using union-find. (Tier: Beginner, CTCI Ch 4)"""


class Dsu:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal(n, edges):
    """edges: list of (u, v, weight). Returns (total, chosen edges) or None if disconnected."""
    dsu = Dsu(n)
    total = 0
    chosen = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            chosen.append((u, v, w))
            total += w
            if len(chosen) == n - 1:
                break
    return (total, chosen) if len(chosen) == n - 1 else None


def main():
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1),
        (1, 3, 2), (2, 3, 4), (3, 4, 2), (2, 4, 5),
    ]
    result = kruskal(5, edges)
    assert result is not None
    total, chosen = result
    assert total == 8, total                       # 1 + 2 + 2 + 3
    assert len(chosen) == 4                        # n - 1 edges

    assert kruskal(1, []) == (0, [])               # single vertex
    assert kruskal(3, [(0, 1, 1)]) is None         # disconnected


if __name__ == "__main__":
    main()
    print("PASS")
