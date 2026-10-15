"""Task 422 - Persistent union-find (partial persistence) basic. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Union operations record parent and root-size changes together with the time they happened,
so past states stay queryable ("what was the representative of node 5 after the first 10
unions?"). find(node, time) walks the recorded history, and component_count(time) counts
distinct representatives. Every past state is cross-checked against replaying the first
`time` operations on a fresh union-find that makes identical union-by-size decisions.
"""


class PersistentUnionFind:
    def __init__(self, n):
        self.n = n
        self.parent_history = [[] for _ in range(n)]    # (time, new parent)
        self.size_history = [[] for _ in range(n)]      # (time, new root size)

    @staticmethod
    def _value_at(history, time):
        """Last recorded value with timestamp <= time, or None."""
        lo, hi = 0, len(history)
        while lo < hi:
            mid = (lo + hi) // 2
            if history[mid][0] <= time:
                lo = mid + 1
            else:
                hi = mid
        return history[lo - 1][1] if lo else None

    def find(self, node, time):
        while True:
            parent = self._value_at(self.parent_history[node], time)
            if parent is None:
                return node
            node = parent

    def size(self, node, time):
        recorded = self._value_at(self.size_history[node], time)
        return recorded if recorded is not None else 1

    def union(self, a, b, time):
        """Merge at the given (strictly increasing) timestamp; True if a merge happened."""
        root_a = self.find(a, time - 1)
        root_b = self.find(b, time - 1)
        if root_a == root_b:
            return False
        size_a = self.size(root_a, time - 1)
        size_b = self.size(root_b, time - 1)
        if size_a < size_b:
            root_a, root_b = root_b, root_a
            size_a, size_b = size_b, size_a
        self.parent_history[root_b].append((time, root_a))
        self.size_history[root_a].append((time, size_a + size_b))
        return True

    def component_count(self, time):
        return len({self.find(node, time) for node in range(self.n)})


def replay_union_find(n, operations, time):
    """Oracle: the plain parent array after applying the first `time` operations."""
    parent = list(range(n))
    size = [1] * n
    for step in range(1, time + 1):
        a, b = operations[step - 1]
        root_a = root_of(parent, a)
        root_b = root_of(parent, b)
        if root_a != root_b:
            if size[root_a] < size[root_b]:
                root_a, root_b = root_b, root_a
            parent[root_b] = root_a
            size[root_a] += size[root_b]
    return parent


def root_of(parent, node):
    while parent[node] != node:
        node = parent[node]
    return node


def oracle_roots(n, operations, time):
    parent = replay_union_find(n, operations, time)
    return [root_of(parent, node) for node in range(n)]


def main():
    operations = [(0, 1), (2, 3), (1, 3), (0, 2)]
    uf = PersistentUnionFind(4)
    merges = [uf.union(a, b, time) for time, (a, b) in enumerate(operations, start=1)]
    assert merges == [True, True, True, False]

    # time t = number of operations applied so far
    assert uf.component_count(0) == 4
    assert uf.component_count(1) == 3
    assert uf.component_count(2) == 2
    assert uf.component_count(3) == 1
    assert uf.component_count(4) == 1

    assert uf.find(1, 1) == 0 and uf.find(0, 1) == 0
    assert uf.find(3, 1) == 3                      # second union has not happened yet
    assert uf.find(3, 2) == 2                      # after the second union 3 -> 2
    assert uf.find(3, 3) == 0
    assert uf.find(3, 4) == 0
    assert uf.find(2, 2) == 2
    assert uf.find(1, 0) == 1 and uf.find(2, 0) == 2

    for time in range(0, 5):
        expected = oracle_roots(4, operations, time)
        assert uf.component_count(time) == len(set(expected)), time
        for node in range(4):
            assert uf.find(node, time) == expected[node], (time, node)

    for seed in (422, 31415, 271828):
        n = 2 + seed % 9
        uf = PersistentUnionFind(n)
        operations = []
        local = seed
        for _ in range(40):
            local = (local * 75 + 74) % 65537
            a = local % n
            local = (local * 75 + 74) % 65537
            operations.append((a, local % n))
        applied = 0
        for time, (a, b) in enumerate(operations, start=1):
            if uf.union(a, b, time):
                applied += 1
            assert uf.component_count(time) == n - applied, (seed, time)
        for _ in range(60):
            local = (local * 75 + 74) % 65537
            time = local % (len(operations) + 1)
            expected = oracle_roots(n, operations, time)
            assert uf.component_count(time) == len(set(expected)), (seed, time)
            for node in range(n):
                assert uf.find(node, time) == expected[node], (seed, time, node)


if __name__ == "__main__":
    main()
    print("PASS")
