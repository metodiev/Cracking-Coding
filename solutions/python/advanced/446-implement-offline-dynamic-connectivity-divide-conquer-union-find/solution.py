"""Task 446 - Implement offline dynamic connectivity (divide & conquer + union find). (Tier: Advanced, CTCI Ch 4)

All operations are read first; every edge that is inserted and later deleted becomes an active
time interval. A divide-and-conquer over the timeline keeps only the intervals that fully cover
the current segment, unions them into a rollback union-find (union by size, no path
compression), recurses on both halves, then undoes exactly the unions it made. Queries are
answered at the leaves where their moment lies. Intervals that never cover a node are pushed
down, so an edge is handled O(log q) times and the whole run is O(q log q * alpha) with small
memory. Connectivity answers are compared against a full replay brute force for every script.
"""


class RollbackDSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.history = []

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] > self.size[rb]:
            ra, rb = rb, ra
        self.parent[ra] = rb
        self.size[rb] += self.size[ra]
        self.history.append((ra, rb))
        return True

    def same(self, a, b):
        return self.find(a) == self.find(b)

    def snapshot(self):
        return len(self.history)

    def rollback(self, snapshot):
        while len(self.history) > snapshot:
            ra, rb = self.history.pop()
            self.parent[ra] = ra
            self.size[rb] -= self.size[ra]


def offline_dynamic_connectivity(n, ops):
    q = len(ops)
    if q == 0:
        return []
    open_edges = {}
    counts = {}
    intervals = []
    for t, (kind, u, v) in enumerate(ops):
        key = (min(u, v), max(u, v))
        if kind == 0:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 1:
                open_edges[key] = t
        elif kind == 1:
            if counts.get(key, 0) > 0:
                counts[key] -= 1
                if counts[key] == 0:
                    # A removal at time t kills the edge *before* any query at t, so the
                    # interval of the last copy of the edge ends at t - 1.
                    start = open_edges.pop(key)
                    if start <= t - 1:
                        intervals.append((key[0], key[1], start, t - 1))
    for key, start in open_edges.items():
        intervals.append((key[0], key[1], start, q - 1))

    dsu = RollbackDSU(n)
    answers = [None] * q

    def solve(lo, hi, active):
        snapshot = dsu.snapshot()
        rest = []
        for u, v, a, b in active:
            if a <= lo and hi <= b:
                dsu.union(u, v)
            else:
                rest.append((u, v, a, b))
        if lo == hi:
            kind, u, v = ops[lo]
            if kind == 2:
                answers[lo] = dsu.same(u, v)
        else:
            mid = (lo + hi) // 2
            solve(lo, mid, [e for e in rest if e[2] <= mid and e[3] >= lo])
            solve(mid + 1, hi, [e for e in rest if e[2] <= hi and e[3] >= mid + 1])
        dsu.rollback(snapshot)

    solve(0, q - 1, intervals)
    return answers


def brute_answers(n, ops):
    counts = {}
    answers = [None] * len(ops)
    for t, (kind, u, v) in enumerate(ops):
        key = (min(u, v), max(u, v))
        if kind == 0:
            counts[key] = counts.get(key, 0) + 1
        elif kind == 1:
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
        else:
            adjacency = [[] for _ in range(n)]
            for a, b in counts:
                adjacency[a].append(b)
                adjacency[b].append(a)
            seen = [False] * n
            seen[u] = True
            stack = [u]
            while stack:
                x = stack.pop()
                for y in adjacency[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
            answers[t] = seen[v]
    return answers


def lcg_op_scripts(trials, seed, n, q):
    """Valid insert/delete/query scripts: a delete only ever removes a currently present edge."""
    scripts = []
    for _ in range(trials):
        counts = {}
        ops = []
        for _ in range(q):
            seed = (seed * 75 + 74) % 65537
            kind = seed % 3
            seed = (seed * 75 + 74) % 65537
            u = seed % n
            seed = (seed * 75 + 74) % 65537
            v = seed % n
            if u == v:
                v = (v + 1) % n
            key = (min(u, v), max(u, v))
            if kind == 1 and not counts:
                kind = 0
            if kind == 0:
                ops.append((0, u, v))
                counts[key] = counts.get(key, 0) + 1
            elif kind == 1:
                keys = sorted(counts)
                seed = (seed * 75 + 74) % 65537
                key = keys[seed % len(keys)]
                ops.append((1, key[0], key[1]))
                counts[key] -= 1
                if counts[key] == 0:
                    del counts[key]
            else:
                ops.append((2, u, v))
        scripts.append(ops)
    return scripts


def main():
    # Hand case: 0-1 added, query connected; 1-2 added, 0-2 connected via 1; 0-1 removed.
    ops = [(0, 0, 1), (2, 0, 1), (0, 1, 2), (2, 0, 2), (1, 0, 1), (2, 0, 2), (2, 0, 0)]
    assert offline_dynamic_connectivity(3, ops) == [None, True, None, True, None, False, True]
    ops = [(2, 0, 1), (0, 0, 1), (2, 0, 1), (1, 0, 1), (2, 0, 1), (2, 1, 1)]
    assert offline_dynamic_connectivity(2, ops) == [False, None, True, None, False, True]

    total_queries = 0
    for script in lcg_op_scripts(30, 446, 8, 40):
        expected = brute_answers(8, script)
        total_queries += sum(1 for a in expected if a is not None)
        assert offline_dynamic_connectivity(8, script) == expected
    assert total_queries > 100

    for script in lcg_op_scripts(10, 777, 12, 60):
        assert offline_dynamic_connectivity(12, script) == brute_answers(12, script)


if __name__ == "__main__":
    main()
    print("PASS")
