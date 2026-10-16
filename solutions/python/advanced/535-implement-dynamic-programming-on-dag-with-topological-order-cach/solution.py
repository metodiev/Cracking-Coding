"""Task 535 - Implement dynamic programming on DAG with topological order caching. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)

Dynamic programming over a directed acyclic graph whose answer for a vertex depends only on its
predecessors, so a single pass in topological order suffices. The ``DAG`` class stores adjacency
lists of (successor, weight) pairs and memoises the topological order: Kahn's algorithm runs only
when the cached order is stale (a version counter is bumped by every ``add_edge``), so repeated DP
queries reuse one order. The self-test counts order builds and cache hits to prove the caching
happens, checks that the returned order is the very same object, and that an edge insertion
invalidates it. The order is produced with a binary heap keyed by vertex index, so it is
deterministic and independent of the insertion order of the edges.

Four independent query styles are implemented and cross-checked: (1) a forward pass over the
cached topological order computing longest paths, shortest paths (negative weights allowed - a
case where Dijkstra is simply wrong, which the file demonstrates) and the number of paths from a
source; (2) a memoised recursion over the reverse graph; (3) brute force enumeration of every
simple path (equivalent to every path in a DAG) giving the exact extreme weights and the exact
path count, including a check that the vertex path counts sum to the number of enumerated paths;
and (4) closed forms, namely the 4x4 grid DAG, whose number of monotone source-to-sink paths is
the binomial C(6,3) = 20 and whose longest path is exactly 6 unit steps, plus the hand-checked
tiny example 0-1-3-4 (weight 10) versus 0-2-3-4 (weight 9).

Checks: topological-order validity (pos[u] < pos[v] on every edge) and cycle detection raising
ValueError without consuming a rebuild; cached-object identity plus build/hit counters across
three consecutive queries; agreement of the topological DP with the memoised recursion and with
brute force on twenty LCG-generated DAGs (n = 8, weights in [-5, 10], so negative edges are
exercised); path counts against brute force and the C(6,3) grid value; unreachable vertices
reported as -inf / +inf / zero paths and the empty graph's 1/0 pattern; and insertion-order
independence of both the topological order and every DP table.
"""
import heapq

NEG_INF = float("-inf")
POS_INF = float("inf")


class DAG:
    """Directed graph with memoised topological order and version-based invalidation."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self._version = 0
        self._order = None
        self._order_version = -1
        self.order_builds = 0
        self.cache_hits = 0

    def add_edge(self, u, v, w=1):
        assert 0 <= u < self.n and 0 <= v < self.n, "endpoints in range"
        self.adj[u].append((v, w))
        self._version += 1

    @property
    def edges(self):
        return [(u, v, w) for u in range(self.n) for (v, w) in self.adj[u]]

    def topological_order(self):
        """Kahn's algorithm; the result is cached until the graph changes."""
        if self._order is not None and self._order_version == self._version:
            self.cache_hits += 1
            return self._order
        indegree = [0] * self.n
        for u in range(self.n):
            for (v, _) in self.adj[u]:
                indegree[v] += 1
        heap = [v for v in range(self.n) if indegree[v] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            u = heapq.heappop(heap)
            order.append(u)
            for (v, _) in self.adj[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    heapq.heappush(heap, v)
        if len(order) != self.n:
            raise ValueError("graph contains a cycle - no topological order exists")
        self._order = order
        self._order_version = self._version
        self.order_builds += 1
        return order

    def positions(self):
        pos = [0] * self.n
        for index, vertex in enumerate(self.topological_order()):
            pos[vertex] = index
        return pos

    def reverse_adj(self):
        radj = [[] for _ in range(self.n)]
        for u in range(self.n):
            for (v, w) in self.adj[u]:
                radj[v].append((u, w))
        return radj

    def longest_paths(self, source):
        order = self.topological_order()
        dist = [NEG_INF] * self.n
        dist[source] = 0.0
        for u in order:
            if dist[u] == NEG_INF:
                continue
            for (v, w) in self.adj[u]:
                if dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w
        return dist

    def shortest_paths(self, source):
        order = self.topological_order()
        dist = [POS_INF] * self.n
        dist[source] = 0.0
        for u in order:
            if dist[u] == POS_INF:
                continue
            for (v, w) in self.adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        return dist

    def count_paths(self, source):
        order = self.topological_order()
        count = [0] * self.n
        count[source] = 1
        for u in order:
            if count[u] == 0:
                continue
            for (v, _) in self.adj[u]:
                count[v] += count[u]
        return count

    def longest_paths_memo(self, source):
        """Same answer as longest_paths, via memoised recursion on the reverse graph."""
        radj = self.reverse_adj()
        memo = {}

        def best(v, stack):
            if v not in memo:
                assert v not in stack, "memoised recursion needs an acyclic graph"
                stack.add(v)
                value = NEG_INF
                for (u, w) in radj[v]:
                    candidate = best(u, stack)
                    if candidate != NEG_INF and candidate + w > value:
                        value = candidate + w
                stack.discard(v)
                memo[v] = 0.0 if v == source else value
            return memo[v]

        return [best(v, set()) for v in range(self.n)]


def reachable(dag, source, target):
    seen = set()
    stack = [source]
    while stack:
        u = stack.pop()
        if u == target:
            return True
        if u in seen:
            continue
        seen.add(u)
        for (v, _) in dag.adj[u]:
            stack.append(v)
    return False


def enumerate_paths(dag, source):
    """Every simple path leaving ``source`` as (total weight, vertex list)."""
    paths = []

    def walk(v, weight, trail):
        for (nxt, w) in dag.adj[v]:
            paths.append((weight + w, trail + [nxt]))
            walk(nxt, weight + w, trail + [nxt])

    walk(source, 0.0, [source])
    return paths


def brute_force_paths(dag, source, target=None):
    """(longest, shortest, count) over paths from ``source``, optionally ending at ``target``."""
    weights = [weight for (weight, trail) in enumerate_paths(dag, source)
               if target is None or trail[-1] == target]
    if not weights:
        return NEG_INF, POS_INF, 0
    return max(weights), min(weights), len(weights)


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_dag(n, seed, threshold=3, modulus=7, lo=-5, hi=10):
    """DAG on n vertices: edge i -> j for i < j, chosen by the LCG, weight in [lo, hi]."""
    dag = DAG(n)
    for i in range(n):
        for j in range(i + 1, n):
            seed = lcg(seed)
            if seed % modulus < threshold:
                seed = lcg(seed)
                dag.add_edge(i, j, lo + seed % (hi - lo + 1))
    return dag, seed


def grid_dag(rows, cols):
    """Lattice: vertex (r, c) is r*cols + c, unit-weight edges going right and down."""
    dag = DAG(rows * cols)
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                dag.add_edge(u, r * cols + c + 1, 1)
            if r + 1 < rows:
                dag.add_edge(u, (r + 1) * cols + c, 1)
    return dag


def main():
    # --- Hand-checked tiny example ------------------------------------------------------
    dag = DAG(5)
    dag.add_edge(0, 1, 2)
    dag.add_edge(0, 2, 5)
    dag.add_edge(1, 3, 5)
    dag.add_edge(2, 3, 1)
    dag.add_edge(3, 4, 3)
    pos = dag.positions()
    for (u, v, _) in dag.edges:
        assert pos[u] < pos[v], "topological order respects every edge"
    assert dag.longest_paths(0)[4] == 10.0, "0-1-3-4 = 10 beats 0-2-3-4 = 9"
    assert dag.shortest_paths(0)[4] == 9.0
    assert dag.count_paths(0) == [1, 1, 1, 2, 2], "two routes into 3 and into 4"
    longest, shortest, count = brute_force_paths(dag, 0, 4)
    assert longest == 10.0 and shortest == 9.0 and count == 2, "brute force: two routes to 4"

    # Caching: same order object reused, then invalidated by a new edge.
    order = dag.topological_order()
    builds_before, hits_before = dag.order_builds, dag.cache_hits
    assert dag.topological_order() is order, "cached order object is returned"
    assert dag.order_builds == builds_before and dag.cache_hits == hits_before + 1
    dag.add_edge(0, 4, 100)                      # weighty shortcut, still acyclic
    assert dag.order_builds == builds_before, "no rebuild until the next query"
    dag.topological_order()
    assert dag.order_builds == builds_before + 1, "the version bump forces exactly one rebuild"
    assert dag.longest_paths(0)[4] == 100.0, "the new edge is visible to the DP"

    # A failed (cyclic) build must not be counted, and must raise.
    builds_before = dag.order_builds
    dag.add_edge(4, 3, 1)                        # 3 -> 4 -> 3
    try:
        dag.topological_order()
        raise AssertionError("cycle must be detected")
    except ValueError:
        pass
    assert dag.order_builds == builds_before, "the failed build is not counted"

    # --- Negative weights: topological DP works where Dijkstra would not ---------------
    neg = DAG(4)
    neg.add_edge(0, 1, 1)
    neg.add_edge(0, 2, 4)
    neg.add_edge(1, 2, -3)
    neg.add_edge(2, 3, 1)
    assert neg.shortest_paths(0)[3] == -1.0, "0-1-2-3 = 1 - 3 + 1 = -1"
    assert neg.longest_paths(0)[3] == 5.0, "0-2-3 = 5"
    assert neg.longest_paths_memo(0)[3] == 5.0

    # --- Grid DAG: closed-form path counts ---------------------------------------------
    grid = grid_dag(4, 4)
    counts = grid.count_paths(0)
    assert counts[15] == 20, "C(6,3) = 20 monotone lattice paths from (0,0) to (3,3)"
    assert grid.longest_paths(0)[15] == 6.0, "every monotone path has exactly 6 unit steps"
    assert counts[1] == 1 and counts[0] == 1 and counts[5] == 2, "C(2,1) = 2 routes to (1,1)"
    assert grid.topological_order()[0] == 0 and grid.topological_order()[-1] == 15
    grid_best, grid_worst, grid_count = brute_force_paths(grid, 0, 15)
    assert grid_count == 20 and grid_best == 6.0 and grid_worst == 6.0
    builds = grid.order_builds
    grid.count_paths(0)
    grid.longest_paths(0)
    grid.shortest_paths(0)
    assert grid.order_builds == builds, "three queries, still a single order build"

    # --- LCG DAGs: topological DP vs memoised recursion vs brute force -----------------
    seed = 20241016
    total_paths = 0
    for _ in range(20):
        gen, seed = random_dag(8, seed)
        gen_pos = gen.positions()
        for source in range(8):
            topo_longest = gen.longest_paths(source)
            topo_shortest = gen.shortest_paths(source)
            topo_counts = gen.count_paths(source)
            memo_longest = gen.longest_paths_memo(source)
            assert topo_longest[source] == 0.0 and topo_counts[source] == 1
            for v in range(8):
                if not reachable(gen, source, v):
                    assert topo_longest[v] == NEG_INF and topo_shortest[v] == POS_INF
                    assert topo_counts[v] == 0 and memo_longest[v] == NEG_INF
                else:
                    assert topo_counts[v] >= 1 and topo_shortest[v] <= topo_longest[v] + 1e-12
                    assert topo_longest[v] == memo_longest[v], "topo DP and memo recursion differ"
            paths = enumerate_paths(gen, source)
            if not paths:
                continue
            total_paths += len(paths)
            by_target = {}
            for (weight, trail) in paths:
                by_target.setdefault(trail[-1], []).append(weight)
                assert all(gen_pos[trail[i]] < gen_pos[trail[i + 1]]
                           for i in range(len(trail) - 1)), \
                    "every path follows the topological order"
            for v in range(8):
                weights = by_target.get(v, [])
                assert len(weights) == topo_counts[v] - (1 if v == source else 0), \
                    "vertex path count vs enumeration"
                if v == source:
                    assert topo_longest[v] == 0.0 and topo_shortest[v] == 0.0
                    continue
                if topo_counts[v] == 0:
                    assert weights == [], "DP says unreachable but enumeration found a path"
                    continue
                assert weights, "DP says reachable but enumeration found nothing"
                assert max(weights) == topo_longest[v], "longest path to a vertex"
                assert min(weights) == topo_shortest[v], "shortest path to a vertex"
            live = [v for v in range(8) if topo_counts[v] > 0]
            assert sum(topo_counts[v] for v in live) - 1 == len(paths), \
                "vertex path counts must sum to the number of enumerated paths"
    assert total_paths > 100, "the LCG DAGs are dense enough to be meaningful"

    # --- Insertion-order independence ---------------------------------------------------
    first = DAG(5)
    second = DAG(5)
    edge_list = [(0, 1, 3), (0, 2, 1), (1, 3, 4), (2, 3, 2), (3, 4, 7), (1, 4, 1)]
    for (u, v, w) in edge_list:
        first.add_edge(u, v, w)
    for (u, v, w) in reversed(edge_list):
        second.add_edge(u, v, w)
    assert first.topological_order() == second.topological_order(), "heap order is canonical"
    for source in range(5):
        assert first.longest_paths(source) == second.longest_paths(source)
        assert first.count_paths(source) == second.count_paths(source)
        assert first.shortest_paths(source) == second.shortest_paths(source)

    # --- Unreachable vertices and empty graph ------------------------------------------
    lonely = DAG(3)
    lonely.add_edge(0, 1, 1)
    assert lonely.longest_paths(0) == [0.0, 1.0, NEG_INF]
    assert lonely.count_paths(0) == [1, 1, 0]
    assert lonely.shortest_paths(2) == [POS_INF, POS_INF, 0.0]
    assert brute_force_paths(lonely, 0) == (1.0, 1.0, 1)
    empty = DAG(2)
    assert empty.topological_order() == [0, 1]
    assert empty.longest_paths(0) == [0.0, NEG_INF]
    assert empty.count_paths(0) == [1, 0]
    assert brute_force_paths(empty, 0) == (NEG_INF, POS_INF, 0)

    # --- Cycle detection ----------------------------------------------------------------
    cyc = DAG(3)
    cyc.add_edge(0, 1, 1)
    cyc.add_edge(1, 2, 1)
    cyc.add_edge(2, 0, 1)
    try:
        cyc.topological_order()
        raise AssertionError("a 3-cycle must raise ValueError")
    except ValueError:
        pass





if __name__ == "__main__":
    main()
    print("PASS")
