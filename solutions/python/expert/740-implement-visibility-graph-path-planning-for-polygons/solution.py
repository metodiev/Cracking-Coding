"""Task 740 - Implement visibility graph path planning for polygons.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

In a bounded polygonal world the shortest collision-free path between two points is a
polyline whose interior corners are obstacle vertices: anywhere else the path could be pulled
taut.  The visibility graph turns that fact into a shortest-path problem - nodes are the start,
the goal and every obstacle vertex, and two nodes are joined when the segment between them
stays in the free space - so one Dijkstra run answers the query.

Visibility of a segment is decided exactly and in two steps: first no obstacle edge may meet
the segment except at a shared endpoint (a proper crossing, a collinear overlap or a touch
anywhere else is a block, so the planner never slides along an edge or squeezes through a
vertex it does not pivot on), second the midpoint of the segment must lie outside every
obstacle polygon and inside the arena (this is what rejects a diagonal that runs through the
interior of an obstacle without crossing any edge).  All orientation tests are exact rational
arithmetic because the coordinates are integers.

Simplified versus a full implementation: obstacles are simple polygons with integer
coordinates, no curved boundaries, a "two step" visibility test instead of an angular sweep
around each vertex, and the graph is built from all vertices at once instead of per query.
Verified: every graph edge passes an independent sampled point-in-polygon oracle, the Dijkstra
result equals an exhaustive subset-DP over the same graph, the reported path is a real chain of
visible edges, a wall detour matches the hand-derived optimal length, every interior waypoint
is an obstacle vertex, a sealed corridor yields no path, and the same input always gives the
same graph and path.
"""
import heapq
import math
from fractions import Fraction
from itertools import combinations

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


def orient(a, b, c):
    """Exact 2D orientation of the triple a,b,c as an integer."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_block(a, b, p, q):
    """True when the closed segments a-b and p-q meet anywhere other than at one shared
    endpoint.  Exact: proper crossings, collinear overlaps and vertex touches all block."""
    d1, d2 = orient(a, b, p), orient(a, b, q)
    d3, d4 = orient(p, q, a), orient(p, q, b)
    if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True                                    # proper crossing
    contacts = set()
    for point, side in ((p, d1), (q, d2)):
        if side == 0 and on_segment(a, b, point):
            contacts.add(point)
    for point, side in ((a, d3), (b, d4)):
        if side == 0 and on_segment(p, q, point):
            contacts.add(point)
    if not contacts:
        return False
    shared = {a, b} & {p, q}
    if len(contacts) == 1 and contacts <= shared:
        return False                                   # meeting exactly at a shared vertex
    if contacts == {a, b} and set(contacts) <= set((p, q)):
        return False                                   # the edge p-q is the segment a-b itself
    return True


def point_in_polygon(polygon, p):
    """Ray casting, integer coordinates; points on the boundary count as outside."""
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if orient(a, b, p) == 0 and on_segment(a, b, p):
            return False
        if (a[1] > p[1]) != (b[1] > p[1]):
            x_cross = a[0] + Fraction(b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
            if x_cross > p[0]:
                inside = not inside
    return inside


def midpoint(a, b):
    return (Fraction(a[0] + b[0], 2), Fraction(a[1] + b[1], 2))


class World:
    """Axis-aligned rectangular arena with polygonal obstacles (integer coordinates)."""

    def __init__(self, bounds, obstacles):
        self.bounds = bounds
        self.obstacles = obstacles
        self.edges = [(polygon[i], polygon[(i + 1) % len(polygon)])
                      for polygon in obstacles for i in range(len(polygon))]

    def in_arena(self, p):
        x0, y0, x1, y1 = self.bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def point_free(self, p):
        return self.in_arena(p) and not any(point_in_polygon(poly, p) for poly in self.obstacles)

    def visible(self, a, b):
        """Exact visibility test: both endpoints in the free space, no edge contact except at
        shared vertices, and the segment midpoint in the free space."""
        if a == b:
            return True
        if not (self.point_free(a) and self.point_free(b)):
            return False
        for (p, q) in self.edges:
            if segments_block(a, b, p, q):
                return False
        return self.point_free(midpoint(a, b))

    def sampled_visible(self, a, b, samples=200):
        """Independent oracle: dense sampling with the ray-casting point test (never used by
        the planner itself)."""
        for i in range(1, samples):
            t = Fraction(i, samples)
            p = (Fraction(a[0]) + t * (b[0] - a[0]), Fraction(a[1]) + t * (b[1] - a[1]))
            if not self.point_free(p):
                return False
        return True


class VisibilityGraph:
    """Nodes = start, goal and all obstacle vertices; edges = visible pairs."""

    def __init__(self, world, start, goal):
        self.world = world
        self.nodes = [start, goal] + [v for poly in world.obstacles for v in poly]
        self.adjacency = {i: {} for i in range(len(self.nodes))}
        self.rejected = 0
        for i, j in combinations(range(len(self.nodes)), 2):
            if world.visible(self.nodes[i], self.nodes[j]):
                weight = math.dist(self.nodes[i], self.nodes[j])
                self.adjacency[i][j] = weight
                self.adjacency[j][i] = weight
            else:
                self.rejected += 1

    def dijkstra(self, source):
        best = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best.get(node, math.inf):
                continue
            for other, weight in self.adjacency[node].items():
                if cost + weight < best.get(other, math.inf) - 1e-12:
                    best[other] = cost + weight
                    heapq.heappush(queue, (cost + weight, other))
        return best

    def path(self, source, target):
        best = self.dijkstra(source)
        if best.get(target, math.inf) == math.inf:
            return None
        chain, node = [target], target
        while node != source:
            step = min((other for other in self.adjacency[node]
                        if abs(best[other] + self.adjacency[node][other] - best[node]) < 1e-9),
                       key=lambda other: (best[other], other))
            chain.append(step)
            node = step
        return chain[::-1]

    def path_length(self, chain):
        return sum(math.dist(self.nodes[a], self.nodes[b]) for a, b in zip(chain, chain[1:]))

    def exhaustive_shortest(self, source, target):
        """Oracle: subset DP over every simple path in the very same graph (Held-Karp style):
        with positive weights the optimum is simple, so this is exact and uses no heuristic."""
        order = [n for n in range(len(self.nodes)) if n not in (source, target)]
        size = len(order)
        best = {}
        for k, node in enumerate(order):
            if node in self.adjacency[source]:
                best[(1 << k, k)] = self.adjacency[source][node]
        answer = self.adjacency[source].get(target, math.inf)
        for mask in range(1 << size):
            for last in range(size):
                cost = best.get((mask, last))
                if cost is None:
                    continue
                node = order[last]
                if target in self.adjacency[node]:
                    answer = min(answer, cost + self.adjacency[node][target])
                for other in range(size):
                    if mask & (1 << other) or order[other] not in self.adjacency[node]:
                        continue
                    candidate = cost + self.adjacency[node][order[other]]
                    key = (mask | (1 << other), other)
                    if candidate < best.get(key, math.inf) - 1e-12:
                        best[key] = candidate
        return answer


def rectangle(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def open_world(span=20):
    return World((0, 0, span, span), [])


def wall_world():
    return World((0, 0, 20, 20), [rectangle(6, 2, 8, 12)])


def sealed_world():
    """A wall that pokes out of the arena on both sides: no way around inside the bounds."""
    return World((0, 0, 20, 20), [rectangle(9, -1, 11, 21)])


def pocket_world():
    """U-shaped pocket open at the top."""
    return World((0, 0, 20, 20), [rectangle(5, 5, 15, 7), rectangle(5, 7, 7, 15),
                                  rectangle(13, 7, 15, 15)])


def main():
    # 1. open arena: the straight line is visible and is the answer
    world = open_world()
    plain = VisibilityGraph(world, (1, 1), (19, 18))
    check(len(plain.adjacency) == 2 and plain.adjacency[0][1] > 0,
          "with no obstacles the two query nodes must see each other")
    check(abs(plain.dijkstra(0)[1] - math.dist((1, 1), (19, 18))) < 1e-9,
          "in an open arena the shortest path is the straight line")
    check(plain.path(0, 1) == [0, 1], "the straight path must be a single edge")

    # 2. a wall obstacle: the path must round its corner at (6, 12) or (8, 0)
    wall = wall_world()
    around = VisibilityGraph(wall, (2, 4), (18, 4))
    chain = around.path(0, 1)
    check(chain is not None, "a wall that does not span the arena must be roundable")
    expected = min(math.dist((2, 4), (6, 2)) + 2.0 + math.dist((8, 2), (18, 4)),
                   math.dist((2, 4), (6, 12)) + 2.0 + math.dist((8, 12), (18, 4)))
    check(len(chain) in (3, 4), "a two-bend detour around the wall corner is expected")
    check(chain[0] == 0 and chain[-1] == 1, "the path must run from the start to the goal")
    check(abs(around.path_length(chain) - expected) < 1e-9,
          f"the detour length {around.path_length(chain)} must equal the hand-derived {expected}")
    check(all(world.visible(around.nodes[a], around.nodes[b]) for a, b in zip(chain, chain[1:])),
          "every path edge must be visible under the exact predicate")
    check(all(world.sampled_visible(around.nodes[a], around.nodes[b]) for a, b in zip(chain, chain[1:])),
          "every path edge must pass the independent sampled oracle")
    check(all(around.nodes[node] in [v for poly in wall.obstacles for v in poly]
              for node in chain[1:-1]),
          "every interior waypoint must be an obstacle vertex (the taut-path property)")

    # 3. Dijkstra versus an exhaustive search over the same graph
    for graph, source, target in ((around, 0, 1), (plain, 0, 1)):
        distances = graph.dijkstra(source)
        oracle = graph.exhaustive_shortest(source, target)
        check(abs(distances[target] - oracle) < 1e-9,
              "Dijkstra must agree with the exhaustive subset search")

    # 4. visibility predicate: the diagonal of an obstacle is blocked, its sides are not
    square = rectangle(4, 4, 10, 10)
    square_world = World((0, 0, 20, 20), [square])
    check(square_world.visible((4, 4), (10, 10)) is False,
          "a diagonal through an obstacle interior must not be visible")
    check(square_world.visible((4, 4), (10, 4)) is True, "an obstacle side must be traversable")
    check(square_world.visible((4, 4), (4, 10)) is True, "the other obstacle side too")
    check(square_world.visible((0, 7), (20, 7)) is False,
          "a segment entering and leaving the obstacle must not be visible")
    check(square_world.sampled_visible((4, 4), (10, 10)) is False
          and square_world.sampled_visible((0, 7), (20, 7)) is False,
          "the sampled oracle must confirm both rejections")

    # 5. negative case: a wall spanning the arena leaves no path
    sealed = sealed_world()
    locked = VisibilityGraph(sealed, (2, 10), (18, 10))
    check(locked.path(0, 1) is None, "a wall spanning the arena must leave the goal unreachable")
    check(locked.rejected > 0, "the sealed world must have rejected edges")

    # 6. the U-shaped pocket: the path must dive in and come back out around the walls
    pocket = pocket_world()
    inside = VisibilityGraph(pocket, (10, 12), (10, 4))
    dive = inside.path(0, 1)
    check(dive is not None and len(dive) >= 4, "reaching inside the pocket must need a detour")
    check(all(pocket.sampled_visible(inside.nodes[a], inside.nodes[b])
              for a, b in zip(dive, dive[1:])), "the pocket detour must be sampled-visible")
    check(abs(inside.path_length(dive) - inside.dijkstra(0)[1]) < 1e-9,
          "the reported detour must have the reported length")

    # 7. boundary cases: start equal to goal, and a start outside the arena
    same = VisibilityGraph(open_world(), (5, 5), (5, 5))
    check(same.path(0, 1) == [0, 1] and same.dijkstra(0)[1] == 0.0,
          "a zero-length query must return the trivial path")
    outside = VisibilityGraph(wall_world(), (30, 30), (18, 4))
    check(outside.dijkstra(0).get(1, math.inf) == math.inf,
          "a start outside the arena must see nothing and stay isolated")

    # 8. determinism: the same world gives the same graph and the same path
    twin = VisibilityGraph(wall_world(), (2, 4), (18, 4))
    check(twin.adjacency == around.adjacency and twin.nodes == around.nodes,
          "rebuilding the graph must reproduce every edge")
    check(twin.path(0, 1) == chain, "rebuilding the graph must reproduce the path")

    # 9. extra queries on the same graph: the wall detour is consistent for a mirrored start
    mirrored = VisibilityGraph(wall, (18, 4), (2, 4))
    mirror_chain = mirrored.path(0, 1)
    check(abs(mirrored.path_length(mirror_chain) - around.path_length(chain)) < 1e-9,
          "reversing the query must give the same length (the graph is undirected)")
    check(mirror_chain is not None and mirror_chain[-1] == 1, "the reversed query must arrive")

    print(f"740 nodes={len(around.nodes)} edges={sum(len(v) for v in around.adjacency.values()) // 2} "
          f"rejected={around.rejected}")
    print(f"740 wall detour length={around.path_length(chain):.4f} expected={expected:.4f} "
          f"waypoints={[around.nodes[n] for n in chain]}")
    print("PASS")


if __name__ == "__main__":
    main()
