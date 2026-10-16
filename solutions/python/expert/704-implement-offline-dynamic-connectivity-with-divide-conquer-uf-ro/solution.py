"""Task 704 - Implement offline dynamic connectivity with divide & conquer + UF rollback.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

Fully dynamic connectivity is hard, but when all edge lifetimes are known up front ("offline")
it collapses to a divide & conquer over time:

  1. build the segment tree over the T time steps and place each edge (u, v, l, r) into the
     O(log T) canonical nodes whose ranges tile [l, r] exactly;
  2. DFS that tree with a *rollback* union-find -- union by size, no path compression -- so the
     state at a leaf contains exactly the edges alive at that time step;
  3. after visiting a subtree, undo every union made on the way in (the DSU keeps a history
     stack of single link operations, and undo is O(1) per link because nothing else changed).

The invariant this rests on is double: the canonical placement is exact (for every time t the
set of edges sitting on the root-to-leaf path is precisely {edges with l <= t <= r}), and the
rolling-back DSU restores the parent/size arrays and the component counter *bit for bit* at every
node of the recursion.  Union by size without path compression keeps the forest depth at most
floor(log2 n), which is what makes the O(1) undo valid.

Simplified versus production use: edges are given as intervals rather than as an add/remove event
stream (no online-to-offline conversion is implemented), the queries are component counts and a
few (time, u, v) connectivity questions, and the DSU is a plain Python object instead of the
usual integer-array micro-optimised version.  Verified against a brute-force per-time-step BFS
oracle on randomised instances, plus exact rollback snapshots taken at every recursion node.
"""
import math
import sys


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class RollbackDSU:
    """Union-find with union by size and no path compression, so every union can be undone."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n
        self.history = []              # None for a no-op union, else (child, root, child_size)
        self.unions = 0
        self.rollbacks = 0
        self.max_depth = 0

    def find(self, x):
        depth = 0
        while self.parent[x] != x:
            x = self.parent[x]
            depth += 1
        if depth > self.max_depth:
            self.max_depth = depth
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            self.history.append(None)  # still recorded: callers undo by history length
            return False
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        child_size = self.size[root_b]
        self.parent[root_b] = root_a
        self.size[root_a] += child_size
        self.components -= 1
        self.history.append((root_b, root_a, child_size))
        self.unions += 1
        return True

    def mark(self):
        return len(self.history)

    def rollback(self, mark):
        while len(self.history) > mark:
            entry = self.history.pop()
            self.rollbacks += 1
            if entry is None:
                continue
            child, root, child_size = entry
            self.size[root] -= child_size
            self.parent[child] = child
            self.components += 1

    def snapshot(self):
        return (self.parent[:], self.size[:], self.components)


def place_interval(tree, node, lo, hi, left, right, edge):
    """Add `edge` to the canonical segment-tree nodes tiling [left, right]."""
    if right < lo or hi < left:
        return 0
    if left <= lo and hi <= right:
        tree[node].append(edge)
        return 1
    mid = (lo + hi) // 2
    return (place_interval(tree, 2 * node, lo, mid, left, right, edge)
            + place_interval(tree, 2 * node + 1, mid + 1, hi, left, right, edge))


def edges_at_time(tree, time, spans):
    """The edges on the root-to-leaf path of `time` -- the placement invariant, made explicit."""
    found = []
    lo, hi, node = 0, spans - 1, 1
    while True:
        found.extend(tree[node])
        if lo == hi:
            break
        mid = (lo + hi) // 2
        if time <= mid:
            hi, node = mid, 2 * node
        else:
            lo, node = mid + 1, 2 * node + 1
    return found


def solve(n, edges, steps, queries=(), verify_rollback=True):
    """Component counts per time step, connectivity answers, and measured DSU work.

    `edges` is a list of (u, v, first, last) with 0 <= first <= last < steps; `queries` is a
    list of (time, u, v).  Returns (counts, answers, stats).
    """
    for u, v, first, last in edges:
        check(0 <= first <= last < steps, f"bad interval {(u, v, first, last)}")
        check(0 <= u < n and 0 <= v < n, f"bad endpoints {(u, v)}")
    tree = [[] for _ in range(4 * steps)]
    placements = 0
    for u, v, first, last in edges:
        placements += place_interval(tree, 1, 0, steps - 1, first, last, (u, v))

    wanted = {}
    for time, u, v in queries:
        wanted.setdefault(time, []).append((u, v))
    counts = [0] * steps
    answers = {}
    dsu = RollbackDSU(n)
    nodes_visited = 0

    def dfs(node, lo, hi):
        nonlocal nodes_visited
        nodes_visited += 1
        mark = dsu.mark()
        before = dsu.snapshot() if verify_rollback else None
        for u, v in tree[node]:
            dsu.union(u, v)
        if lo == hi:
            counts[lo] = dsu.components
            for u, v in wanted.get(lo, ()):
                answers[(lo, u, v)] = dsu.find(u) == dsu.find(v)
        else:
            mid = (lo + hi) // 2
            dfs(2 * node, lo, mid)
            dfs(2 * node + 1, mid + 1, hi)
        dsu.rollback(mark)
        if verify_rollback:
            check(dsu.snapshot() == before,
                  f"rollback did not restore the DSU state at segment node {node}")

    dfs(1, 0, steps - 1)
    check(not dsu.history, "the history stack must be empty when the DFS finishes")
    stats = {"placements": placements, "nodes": nodes_visited, "unions": dsu.unions,
             "rollbacks": dsu.rollbacks, "max_depth": dsu.max_depth}
    return counts, answers, stats


# --------------------------------------------------------------------------- oracle
def brute_counts(n, edges, steps):
    """Component count per time step, recomputed from scratch with BFS."""
    counts = []
    for time in range(steps):
        adjacency = [[] for _ in range(n)]
        for u, v, first, last in edges:
            if first <= time <= last:
                adjacency[u].append(v)
                adjacency[v].append(u)
        seen = [False] * n
        components = 0
        for start in range(n):
            if seen[start]:
                continue
            components += 1
            seen[start] = True
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbour in adjacency[node]:
                    if not seen[neighbour]:
                        seen[neighbour] = True
                        stack.append(neighbour)
        counts.append(components)
    return counts


def brute_connected(n, edges, time, u, v):
    adjacency = [[] for _ in range(n)]
    for a, b, first, last in edges:
        if first <= time <= last:
            adjacency[a].append(b)
            adjacency[b].append(a)
    seen = {u}
    stack = [u]
    while stack:
        node = stack.pop()
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return v in seen


# --------------------------------------------------------------------------- checks
def test_handcrafted():
    # a triangle that exists during 0..3, then a bridge 2-3 that exists only at time 4
    edges = [(0, 1, 0, 3), (1, 2, 0, 3), (0, 2, 0, 3), (2, 3, 4, 4), (3, 4, 4, 6)]
    counts, answers, stats = solve(5, edges, 7, queries=[(0, 0, 2), (3, 2, 3), (4, 2, 3), (4, 0, 4)])
    check(counts == [3, 3, 3, 3, 3, 4, 4], f"handcrafted component counts: {counts}")
    check(counts == brute_counts(5, edges, 7), "handcrafted counts must match the oracle")
    check(answers[(0, 0, 2)] is True, "the triangle connects 0 and 2")
    check(answers[(3, 2, 3)] is False, "the bridge is not alive at time 3")
    check(answers[(4, 2, 3)] is True, "the bridge is alive at time 4")
    check(answers[(4, 0, 4)] is False, "0 is isolated at time 4, so it cannot reach 4")
    check(stats["placements"] >= len(edges), "each edge needs at least one canonical node")
    return stats


def test_rollback_dsu_unit():
    dsu = RollbackDSU(8)
    initial = dsu.snapshot()
    check(dsu.components == 8, "eight singleton components")
    check(dsu.union(0, 1) and dsu.components == 7, "first union merges two components")
    mark = dsu.mark()
    check(dsu.union(2, 3) and dsu.union(4, 5) and dsu.union(5, 6), "three more unions")
    check(dsu.union(0, 1) is False, "a repeated union is a no-op but still recorded")
    check(dsu.components == 4, f"four components after the merges, got {dsu.components}")
    dsu.rollback(mark)
    check(dsu.components == 7, "rollback restores the component counter")
    check(dsu.find(2) == 2 and dsu.find(6) == 6, "the rolled-back nodes are singletons again")
    dsu.rollback(0)
    check(dsu.snapshot() == initial, "rollback to zero restores the initial arrays exactly")
    check(dsu.parent == list(range(8)) and dsu.size == [1] * 8, "identity arrays restored")

    # union by size keeps the find depth at most floor(log2 n)
    dsu = RollbackDSU(64)
    for i in range(1, 64):
        dsu.union(i % 7, i)
    deepest = max(_depth(dsu, x) for x in range(64))
    check(deepest <= 6, f"depth must stay logarithmic for 64 nodes, got {deepest}")
    check(dsu.max_depth <= 6, f"the DSU agreed, max depth {dsu.max_depth}")


def _depth(dsu, x):
    depth = 0
    while dsu.parent[x] != x:
        x = dsu.parent[x]
        depth += 1
    return depth


def test_canonical_placement_invariant():
    steps = 12
    edges = [(0, 1, 0, 11), (1, 2, 2, 5), (2, 3, 5, 5), (3, 4, 7, 11), (4, 0, 3, 9)]
    tree = [[] for _ in range(4 * steps)]
    placements = 0
    for u, v, first, last in edges:
        placements += place_interval(tree, 1, 0, steps - 1, first, last, (u, v))
    for time in range(steps):
        expected = {(u, v) for u, v, first, last in edges if first <= time <= last}
        got = set(edges_at_time(tree, time, steps))
        check(got == expected, f"placement invariant broken at time {time}: {got} != {expected}")
    import math
    check(placements <= 2 * len(edges) * max(1, math.ceil(math.log2(steps))),          f"canonical decomposition must be O(m log T), got {placements}")


def test_random_instances_against_brute_force():
    mask = (1 << 64) - 1
    seed = 0x1234567890ABCDEF
    for trial in range(25):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        n = 2 + seed % 7
        steps = 1 + (seed >> 11) % 8
        edge_count = (seed >> 23) % 10
        edges = []
        for k in range(edge_count):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            u = seed % n
            v = (seed >> 13) % n
            first = (seed >> 27) % steps
            last = first + (seed >> 41) % (steps - first)
            edges.append((u, v, first, last))
        queries = []
        for time in range(steps):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            queries.append((time, seed % n, (seed >> 17) % n))
        counts, answers, _ = solve(n, edges, steps, queries=queries)
        check(counts == brute_counts(n, edges, steps),
              f"trial {trial}: counts {counts} != oracle {brute_counts(n, edges, steps)}")
        for time, u, v in queries:
            check(answers[(time, u, v)] == brute_connected(n, edges, time, u, v),
                  f"trial {trial}: connectivity wrong at time {time} for {u},{v}")


def test_boundaries():
    counts, answers, stats = solve(1, [], 1)
    check(counts == [1], "a single node, a single step: one component")
    check(answers == {} and stats["unions"] == 0, "no unions happen without edges")

    counts, _, _ = solve(4, [], 5)
    check(counts == [4] * 5, "no edges means n components at every step")

    counts, _, _ = solve(4, [(0, 1, 0, 4), (2, 3, 0, 4)], 5)
    check(counts == [2] * 5, "two fixed edges, two components")

    counts, _, _ = solve(4, [(0, 1, 2, 2)], 5)
    check(counts == [4, 4, 3, 4, 4], f"an edge alive for one step only: {counts}")

    counts, _, _ = solve(3, [(0, 0, 0, 2), (1, 1, 0, 2)], 3)
    check(counts == [3, 3, 3], "self loops never merge anything")

    counts, _, _ = solve(3, [(0, 1, 0, 0), (0, 1, 1, 2)], 3)
    check(counts == [2, 2, 2], "parallel edges across time steps are merged correctly")

    counts, _, _ = solve(3, [(0, 1, 0, 2), (0, 1, 0, 2)], 3)
    check(counts == [2, 2, 2], "a duplicated interval must not double count")

    try:
        solve(3, [(0, 1, 2, 1)], 3)
        raise AssertionError("an interval with first > last must be rejected")
    except AssertionError as error:
        check("bad interval" in str(error), f"unexpected rejection message: {error}")
    try:
        solve(3, [(0, 5, 0, 1)], 3)
        raise AssertionError("an out-of-range endpoint must be rejected")
    except AssertionError as error:
        check("bad endpoints" in str(error), f"unexpected rejection message: {error}")


def test_larger_instance_and_state_restoration():
    n, steps, edge_count = 30, 64, 70
    mask = (1 << 64) - 1
    seed = 0xFACEFEED12345678
    edges = []
    for _ in range(edge_count):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        u = seed % n
        v = (seed >> 19) % n
        first = (seed >> 37) % steps
        last = first + (seed >> 47) % (steps - first)
        edges.append((u, v, first, last))
    counts, _, stats = solve(n, edges, steps, verify_rollback=True)
    check(counts == brute_counts(n, edges, steps), "larger instance counts vs the oracle")
    check(all(1 <= value <= n for value in counts), "component counts stay in range")
    check(stats["unions"] > 0 and stats["rollbacks"] >= stats["unions"],
          "every union must eventually be undone")
    check(stats["max_depth"] <= 5, f"depth bound for n = 30 is 5, got {stats['max_depth']}")
    check(counts[steps - 1] == brute_counts(n, edges, steps)[steps - 1],
          "spot check on the last time step")
    return stats


def main():
    hand = test_handcrafted()
    test_rollback_dsu_unit()
    test_canonical_placement_invariant()
    test_random_instances_against_brute_force()
    test_boundaries()
    large = test_larger_instance_and_state_restoration()

    print(f"handcrafted_placements={hand['placements']} large_nodes={large['nodes']} "
          f"unions={large['unions']} rollbacks={large['rollbacks']} "
          f"max_depth={large['max_depth']}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
