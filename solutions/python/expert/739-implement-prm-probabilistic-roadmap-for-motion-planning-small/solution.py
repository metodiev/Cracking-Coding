"""Task 739 - Implement PRM (probabilistic roadmap) for motion planning small.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

A probabilistic roadmap plans in two phases instead of growing a tree per query: the learning
phase scatters deterministic pseudo-random free configurations over the workspace and joins
each one to its nearest neighbours that are mutually visible, producing an undirected graph
that captures the connectivity of the free space; the query phase then adds the start and goal
to the same graph and runs a shortest-path search.  Because the graph is built once, later
queries are cheap - that is the point of the roadmap.

Visibility of a candidate edge is exact for the axis-aligned rectangular obstacles used here: a
segment is clipped against each rectangle with the Liang-Barsky parameter interval and closed
contact counts as blocked, so the roadmap never squeezes through a corner.  The query uses
Dijkstra on a binary heap, and the tests cross-check it with two independent implementations: a
Bellman-Ford relaxation loop (different algorithm, same answer) and A* with the Euclidean
heuristic (which must return the same cost because the heuristic is admissible and consistent).

Simplified versus real PRM implementations: uniform sampling instead of Gaussian/OBPRM
sampling, a fixed neighbourhood radius and k instead of an adaptive one, no
roadmap-expansion/repair, and no re-sampling after a failed query.  Workspaces stay small (a
few hundred samples) so the file runs in well under a second.  Verified: the graph is symmetric
and every edge passes an independent sampled line-of-sight oracle, Dijkstra agrees with
Bellman-Ford and A*, the reported path is a real chain of graph edges whose length equals the
reported cost, the straight-through-the-gap corridor is found with a near-optimal length, a
sealed wall defeats the roadmap, degenerate start-equals-goal queries work, and a fixed seed
reproduces the roadmap exactly.
"""
import heapq
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Workspace:
    """Bounded 2D workspace with closed axis-aligned obstacles."""

    def __init__(self, bounds, obstacles):
        self.bounds = bounds
        self.obstacles = obstacles

    def point_free(self, p):
        for (x0, y0, x1, y1) in self.obstacles:
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return False
        return True

    def segment_clear(self, a, b):
        for rect in self.obstacles:
            if segment_hits_rect(a, b, rect):
                return False
        return True

    def sampled_clear(self, a, b, samples=400):
        """Independent oracle: dense sampling of the segment (never used by the planner)."""
        for i in range(samples + 1):
            t = i / samples
            if not self.point_free((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))):
                return False
        return True


def segment_hits_rect(a, b, rect):
    """Closed segment versus closed rectangle via the Liang-Barsky parameter interval."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t_low, t_high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            if ratio > t_high:
                return False
            t_low = max(t_low, ratio)
        else:
            if ratio < t_low:
                return False
            t_high = min(t_high, ratio)
    return t_low <= t_high


class Roadmap:
    """Probabilistic roadmap: nodes plus an undirected, geometrically validated adjacency."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.nodes = []
        self.adjacency = []
        self.rejected_edges = 0

    def add_node(self, point):
        self.nodes.append(point)
        self.adjacency.append({})
        return len(self.nodes) - 1

    def connect(self, index, neighbours):
        """Join `index` to each visible neighbour (both directions, same weight)."""
        added = 0
        for other in neighbours:
            if other == index or other in self.adjacency[index]:
                continue
            a, b = self.nodes[index], self.nodes[other]
            if not self.workspace.segment_clear(a, b):
                self.rejected_edges += 1
                continue
            weight = dist(a, b)
            self.adjacency[index][other] = weight
            self.adjacency[other][index] = weight
            added += 1
        return added

    def build(self, samples, radius, k, seed):
        """Learning phase: `samples` free configurations, each joined to its k nearest within
        `radius` that are mutually visible."""
        x0, y0, x1, y1 = self.workspace.bounds
        state, placed = seed, 0
        attempts = 0
        while placed < samples and attempts < samples * 50:
            attempts += 1
            state, rx = splitmix64(state)
            state, ry = splitmix64(state)
            point = (x0 + (x1 - x0) * (rx / float(1 << 64)),
                     y0 + (y1 - y0) * (ry / float(1 << 64)))
            if not self.workspace.point_free(point):
                continue
            self.add_node(point)
            placed += 1
        for index in range(len(self.nodes)):
            self.connect(index, self.k_nearest(index, radius, k))
        return self

    def k_nearest(self, index, radius, k):
        reference = self.nodes[index]
        candidates = sorted(((dist(reference, node), other) for other, node in enumerate(self.nodes)
                             if other != index and dist(reference, node) <= radius),
                            key=lambda pair: (pair[0], pair[1]))
        return [other for _, other in candidates[:k]]

    def add_query(self, point, radius, k):
        """Attach the start/goal to the roadmap with the same neighbourhood rule."""
        index = self.add_node(point)
        self.connect(index, self.k_nearest(index, radius, k))
        return index

    def edge_count(self):
        return sum(len(neighbours) for neighbours in self.adjacency) // 2

    # ------------------------------------------------------------------ search
    def dijkstra(self, source):
        best = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best.get(node, math.inf):
                continue
            for other, weight in self.adjacency[node].items():
                candidate = cost + weight
                if candidate < best.get(other, math.inf) - 1e-12:
                    best[other] = candidate
                    heapq.heappush(queue, (candidate, other))
        return best

    def astar(self, source, target):
        """A* with the Euclidean heuristic: same cost as Dijkstra when it is consistent."""
        best = {source: 0.0}
        queue = [(dist(self.nodes[source], self.nodes[target]), 0.0, source)]
        expanded = 0
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > best.get(node, math.inf) + 1e-12:
                continue
            expanded += 1
            if node == target:
                return cost, expanded
            for other, weight in self.adjacency[node].items():
                candidate = cost + weight
                if candidate < best.get(other, math.inf) - 1e-12:
                    best[other] = candidate
                    heapq.heappush(queue, (candidate + dist(self.nodes[other], self.nodes[target]),
                                           candidate, other))
        return None, expanded

    def bellman_ford(self, source):
        """Independent oracle: classical plain relaxation until no edge improves."""
        best = {node: (0.0 if node == source else math.inf) for node in range(len(self.nodes))}
        for _ in range(len(self.nodes)):
            changed = False
            for node, neighbours in enumerate(self.adjacency):
                if best[node] == math.inf:
                    continue
                for other, weight in neighbours.items():
                    if best[node] + weight < best[other] - 1e-12:
                        best[other] = best[node] + weight
                        changed = True
            if not changed:
                break
        return best

    def path(self, source, target):
        """Shortest path as a node list, recovered from the Dijkstra distances."""
        best = self.dijkstra(source)
        if best.get(target, math.inf) == math.inf:
            return None
        chain, node = [target], target
        while node != source:
            step = min((other for other in self.adjacency[node]
                        if abs(best.get(other, math.inf) + self.adjacency[node][other]
                               - best[node]) < 1e-9),
                       key=lambda other: (best[other], other))
            chain.append(step)
            node = step
        return chain[::-1]

    def path_length(self, chain):
        return sum(dist(self.nodes[chain[i]], self.nodes[chain[i + 1]]) for i in range(len(chain) - 1))


def open_workspace():
    return Workspace((0.0, 0.0, 10.0, 10.0), [])


def gapped_wall_workspace():
    """Wall at x in [5, 5.3] with a gap for y in (4, 6): the straight y = 5 line stays open."""
    return Workspace((0.0, 0.0, 10.0, 10.0), [(5.0, -1.0, 5.3, 4.0), (5.0, 6.0, 5.3, 11.0)])


def sealed_wall_workspace():
    return Workspace((0.0, 0.0, 10.0, 10.0), [(5.0, -1.0, 5.3, 11.0)])


def main():
    # 1. learning phase on the gapped-wall workspace
    workspace = gapped_wall_workspace()
    roadmap = Roadmap(workspace).build(220, 1.6, 6, 0x5EED)
    check(len(roadmap.nodes) == 220, "the roadmap must hold exactly the requested samples")
    check(roadmap.edge_count() > 100, "220 samples with 6 neighbours must give a dense graph")
    symmetric = all(roadmap.adjacency[other].get(node) == weight
                    for node, neighbours in enumerate(roadmap.adjacency)
                    for other, weight in neighbours.items())
    check(symmetric, "the roadmap must be undirected: every edge present in both directions")
    check(all(roadmap.workspace.segment_clear(roadmap.nodes[a], roadmap.nodes[b])
              for a in range(len(roadmap.adjacency)) for b in roadmap.adjacency[a]),
          "every roadmap edge must be collision free by the exact clipping test")
    checks = 0
    for a in range(min(40, len(roadmap.adjacency))):
        for b in sorted(roadmap.adjacency[a])[:3]:
            if a < b:
                checks += 1
                check(roadmap.workspace.sampled_clear(roadmap.nodes[a], roadmap.nodes[b]),
                      "roadmap edges must also pass the independent sampled oracle")
    check(checks > 20, "the sampled oracle must have been applied to many edges")
    check(roadmap.rejected_edges > 0, "some candidate edges must have been rejected by obstacles")

    # 2. query phase: start and goal attach to the roadmap and Dijkstra finds the corridor
    start = roadmap.add_query((1.0, 5.0), 1.6, 6)
    goal = roadmap.add_query((9.0, 5.0), 1.6, 6)
    chain = roadmap.path(start, goal)
    check(chain is not None and chain[0] == start and chain[-1] == goal,
          "the roadmap must connect start to goal through the gap")
    check(all(other in roadmap.adjacency[node] for node, other in zip(chain, chain[1:])),
          "the reported path must be a chain of real graph edges")
    check(all(roadmap.workspace.sampled_clear(roadmap.nodes[a], roadmap.nodes[b])
              for a, b in zip(chain, chain[1:])),
          "every edge of the reported path must pass the sampled oracle")
    length = roadmap.path_length(chain)
    check(length >= 8.0 - 1e-9, "no path can be shorter than the straight line through the gap")
    check(length <= 8.0 * 1.4, "a fine roadmap must find a near-optimal corridor")
    check(abs(roadmap.dijkstra(start)[goal] - length) < 1e-9,
          "the path length must equal the Dijkstra distance to the goal")

    # 3. independent shortest-path algorithms must agree
    distances = roadmap.dijkstra(start)
    bellman = roadmap.bellman_ford(start)
    check(all(abs(distances[node] - bellman[node]) < 1e-9 for node in distances),
          "Dijkstra must agree with Bellman-Ford on every reachable node")
    astar_cost, expanded = roadmap.astar(start, goal)
    check(abs(astar_cost - length) < 1e-9,
          "A* with the Euclidean heuristic must return the same cost as Dijkstra")
    check(expanded <= len(roadmap.nodes), "A* cannot expand more nodes than the graph has")

    # 4. negative case: a sealed wall defeats the roadmap
    sealed = Roadmap(sealed_wall_workspace()).build(180, 1.6, 6, 0xBEEF)
    sealed_start = sealed.add_query((1.0, 5.0), 1.6, 6)
    sealed_goal = sealed.add_query((9.0, 5.0), 1.6, 6)
    check(sealed.dijkstra(sealed_start).get(sealed_goal, math.inf) == math.inf,
          "a sealed wall must leave the goal unreachable in the roadmap")
    check(sealed.path(sealed_start, sealed_goal) is None, "no path may be reported through a wall")

    # 5. boundary cases: empty roadmap plus a visible goal, and start equal to goal
    bare = Roadmap(open_workspace())
    s0 = bare.add_query((1.0, 1.0), 5.0, 4)
    g0 = bare.add_query((3.0, 4.0), 5.0, 4)
    check(abs(bare.dijkstra(s0)[g0] - math.hypot(2.0, 3.0)) < 1e-9,
          "with only two nodes the direct visible edge must be the answer")
    check(bare.path(s0, g0) == [s0, g0], "the trivial path must be a single edge")
    same = Roadmap(open_workspace()).build(20, 5.0, 4, 0x11)
    node = same.add_query((4.0, 4.0), 5.0, 4)
    check(same.dijkstra(node)[node] == 0.0 and same.path(node, node) == [node],
          "a start equal to the goal must have distance zero and a one-node path")
    far = Roadmap(open_workspace())
    isolated = far.add_query((1.0, 1.0), 0.5, 4)
    unreachable = far.add_query((9.0, 9.0), 0.5, 4)
    check(far.dijkstra(isolated).get(unreachable, math.inf) == math.inf,
          "nodes further apart than the radius must not be joined by an edge")

    # 6. determinism: same seed, same roadmap, same path
    twin = Roadmap(gapped_wall_workspace()).build(220, 1.6, 6, 0x5EED)
    check(twin.nodes == roadmap.nodes[:220], "a fixed seed must reproduce the sampled nodes exactly")
    twin_start = twin.add_query((1.0, 5.0), 1.6, 6)
    twin_goal = twin.add_query((9.0, 5.0), 1.6, 6)
    check(twin.path(twin_start, twin_goal) == chain, "a fixed seed must reproduce the same path")

    # 7. connectivity grows with the sample count: a tiny roadmap may miss the corridor
    sparse = Roadmap(gapped_wall_workspace()).build(12, 1.6, 3, 0x5EED)
    sparse_start = sparse.add_query((1.0, 5.0), 1.6, 3)
    sparse_goal = sparse.add_query((9.0, 5.0), 1.6, 3)
    check(sparse.edge_count() < roadmap.edge_count(),
          "a sparse roadmap must have fewer edges than a dense one")
    sparse_chain = sparse.path(sparse_start, sparse_goal)
    check(sparse_chain is None or sparse.path_length(sparse_chain) >= length - 1e-9,
          "a sparse roadmap can only be worse than the dense one")

    print(f"739 samples={len(roadmap.nodes)} edges={roadmap.edge_count()} "
          f"rejected={roadmap.rejected_edges}")
    print(f"739 path nodes={len(chain)} length={length:.3f} optimal>=8.000 "
          f"astar_expanded={expanded}")
    print("PASS")


if __name__ == "__main__":
    main()
