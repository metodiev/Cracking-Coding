"""Task 948 - Implement sublinear time algorithms for property testing on graphs (e.g., bipartiteness).  (Tier: Master, CTCI Ch 4 - Trees and Graphs)

An adjacency-list query model: a graph is only reachable through `degree(v)` and `neighbour(v, i)`, and
every such call is counted, so the tester's cost is measured in probes rather than in edges.  The tester
does bounded BFS from pseudo-random seeds with an explicit probe budget; it reports "non-bipartite" with
an odd-cycle witness as soon as two same-coloured neighbours collide, "unknown" when the budget runs out
before it has seen enough of the graph, and "bipartite" only when it finished its probes without finding
an odd cycle.  An exact DFS oracle, a triangle-union instance (whose distance from bipartiteness is
exactly one edge per triangle) and a near-bipartite instance make the accept/reject and the failure
probability measurable.

Simplified versus the full O(1/epsilon)-probe testers (Goldreich-Ron, Alon-Krivelevich): the tester
explores a bounded local window around each seed instead of scanning 1/epsilon-sized vertex subsets, and
the instances are sparse rather than dense.  Verified: acceptance of bipartite instances and rejection
of a triangle union with a witness that is an independently validated odd cycle; agreement with the
exact DFS oracle where the tester must agree (accept bipartite, reject instances that are far) and
acceptance of a near-bipartite instance, which is the bounded-error behaviour; the probe count stays far
below the edge count and does not grow when the instance doubles; a budget below one edge yields
"unknown" and never a false accept; the empty, single-vertex and single-edge boundaries; detection over
many differently-sized instances against the exact distance from bipartiteness.
"""
import collections

SEEDS = 12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def pseudo(value):
    """Deterministic 64-bit mix, so the pseudo-random seeds are reproducible."""
    value = (value * 0x9E3779B97F4A7C15 + 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 29
    return ((value * 0x94D049BB133111EB) & ((1 << 64) - 1)) >> 11


class QueryGraph:
    """Adjacency lists behind a query counter (the model the tester is charged for)."""

    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.vertices = len(adjacency)
        self.edges = sum(len(neighbours) for neighbours in adjacency.values()) // 2
        self.queries = 0

    def degree(self, vertex):
        self.queries += 1
        return len(self.adjacency.get(vertex, ()))

    def neighbour(self, vertex, index):
        self.queries += 1
        return self.adjacency[vertex][index]


def build_bipartite(pairs, extra, seed):
    """Two sides of `pairs` vertices with deterministic cross edges."""
    adjacency = {vertex: [] for vertex in range(2 * pairs)}
    added = set()
    for step in range(extra):
        left = pseudo(seed * 4099 + step * 131) % pairs
        right = pairs + pseudo(seed * 6151 + step * 197) % pairs
        if (left, right) in added:
            continue
        added.add((left, right))
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def triangle_union(triangles):
    """Disjoint union of triangles: every triangle needs one edge deletion, so its distance is 1/3."""
    adjacency = {vertex: [] for vertex in range(3 * triangles)}
    for index in range(triangles):
        first, second, third = 3 * index, 3 * index + 1, 3 * index + 2
        for source, target in ((first, second), (second, third), (third, first)):
            adjacency[source].append(target)
            adjacency[target].append(source)
    return adjacency


def add_triangle(adjacency, first, second, third):
    copied = {vertex: list(neighbours) for vertex, neighbours in adjacency.items()}
    for source, target in ((first, second), (second, third), (third, first)):
        copied[source].append(target)
        copied[target].append(source)
    return copied


def exact_bipartite(adjacency):
    """Independent oracle: full DFS two-colouring, no query budget."""
    colour = {}
    for start in adjacency:
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            vertex = stack.pop()
            for other in adjacency[vertex]:
                if other not in colour:
                    colour[other] = 1 - colour[vertex]
                    stack.append(other)
                elif colour[other] == colour[vertex]:
                    return False
    return True


def witness_cycle(parents, first, second):
    """Closing walk: the two BFS paths from their junction plus the conflicting edge (first, second)."""
    left, right = [], []
    node = first
    while node is not None:
        left.append(node)
        node = parents[node][1]
    node = second
    while node is not None:
        right.append(node)
        node = parents[node][1]
    while left and right and left[-1] == right[-1]:
        junction = left.pop()
        right.pop()
    return tuple(left + [junction] + list(reversed(right)) + [left[0]])


def test_bipartite(graph, epsilon, seeds=SEEDS, budget_per_seed=None):
    """Bounded local search: a few seeds, each strictly under a probe quota."""
    window = max(1, int(1 / max(epsilon, 1e-9)))
    quota = budget_per_seed if budget_per_seed is not None else 4 * window
    if quota < 2:                      # fewer probes than one edge needs: no verdict is possible
        return "unknown", None, graph.queries
    total = seeds * quota
    for attempt in range(seeds):
        seed = pseudo(attempt * 7919 + graph.vertices) % max(1, graph.vertices)
        parents, queue, start = {seed: (0, None)}, collections.deque([seed]), graph.queries
        while queue and graph.queries - start < quota:
            vertex = queue.popleft()
            for index in range(graph.degree(vertex)):
                if graph.queries - start >= quota:
                    break
                other = graph.neighbour(vertex, index)
                if other in parents:
                    if parents[other][0] == parents[vertex][0]:
                        return ("non-bipartite", witness_cycle(parents, vertex, other),
                                graph.queries)
                    continue
                parents[other] = (1 - parents[vertex][0], vertex)
                queue.append(other)
        if graph.queries >= total and attempt + 1 < seeds:
            return "unknown", None, graph.queries
    return "bipartite", None, graph.queries


def validate_witness(adjacency, cycle):
    """Independent check: the witness must be a closed walk with an odd number of edges."""
    if not cycle or cycle[0] != cycle[-1]:
        return False
    edges = 0
    for source, target in zip(cycle, cycle[1:]):
        if target not in adjacency[source]:
            return False
        edges += 1
    return edges % 2 == 1


def main():
    bipartite = build_bipartite(1500, 9000, 5)
    far = triangle_union(1000)                     # 3000 vertices, distance 1000/3000 = 1/3
    near = add_triangle(build_bipartite(1500, 9000, 5), 0, 3, 7)

    # 1) acceptance on bipartite graphs and agreement with the exact DFS oracle, with probes far
    #    below the edge count.
    graph = QueryGraph(bipartite)
    verdict, witness, probes = test_bipartite(graph, 0.25)
    check(verdict == "bipartite" and witness is None and exact_bipartite(bipartite)
          and probes < graph.edges / 20,
          f"a bipartite graph must be accepted with sublinear probes: {verdict}, {probes} vs "
          f"{graph.edges} edges")

    # 2) rejection of the triangle union with an independently validated odd-cycle witness.
    far_graph = QueryGraph(far)
    verdict, witness, probes = test_bipartite(far_graph, 0.25)
    check(verdict == "non-bipartite" and validate_witness(far, witness)
          and len(witness) % 2 == 0 and len(set(witness)) == 3,
          f"the triangle union must be rejected with an odd cycle: {verdict} {witness}")

    # 3) agreement with the exact oracle where it is required: bipartite is accepted and far is
    #    rejected, while a near-bipartite instance (one triangle in a large bipartite graph) is
    #    accepted - the bounded-error behaviour a tester is allowed to show.
    near_verdict = test_bipartite(QueryGraph(near), 0.25)[0]
    check(exact_bipartite(bipartite) and not exact_bipartite(near) and near_verdict == "bipartite"
          and not exact_bipartite(far)
          and test_bipartite(QueryGraph(far), 0.25)[0] == "non-bipartite",
          f"tester vs exact oracle: near={near_verdict} must be accepted although the exact "
          f"oracle reports non-bipartite")

    # 4) the distance from bipartiteness of the triangle union is exactly one edge per triangle.
    degrees = sum(len(neighbours) for neighbours in far.values())
    deletions = len(far) // 3
    check(degrees == 2 * deletions * 3 and deletions / (degrees / 2) >= 0.3,
          f"the triangle union needs {deletions} deletions out of {degrees // 2} edges, i.e. it is "
          f"far from bipartite")

    # 5) probe cost grows sublinearly when the instance doubles; the budget is respected.
    small, double = QueryGraph(triangle_union(500)), QueryGraph(triangle_union(1000))
    small_verdict, _, small_probes = test_bipartite(small, 0.25)
    double_verdict, _, double_probes = test_bipartite(double, 0.25)
    check(small_verdict == double_verdict == "non-bipartite"
          and double_probes < 2 * small_probes and double_probes <= double.edges / 20,
          f"probes must grow sublinearly: {small_probes} -> {double_probes}")

    # 6) a budget below the two probes one edge needs reports "unknown", never a false accept.
    starved = QueryGraph(far)
    quiet = QueryGraph(bipartite)
    check(test_bipartite(starved, 0.25, seeds=1, budget_per_seed=1)[0] == "unknown"
          and test_bipartite(quiet, 0.25, seeds=1, budget_per_seed=1)[0] == "unknown",
          "a one-probe budget must stay inconclusive")

    # 7) boundaries: empty graph, isolated vertices and a single edge.
    empty, single = QueryGraph({}), QueryGraph({7: [9], 9: [7]})
    isolated = QueryGraph({vertex: [] for vertex in range(5)})
    empty_verdict, _, empty_probes = test_bipartite(empty, 0.5)
    check(empty_verdict == "bipartite" and empty_probes <= SEEDS
          and test_bipartite(single, 0.5)[0] == "bipartite"
          and test_bipartite(isolated, 0.5)[0] == "bipartite"
          and exact_bipartite({7: [9], 9: [7]}),
          f"the empty, isolated and single-edge boundaries must all be bipartite ({empty_probes} "
          f"probes on the empty graph)")

    # 8) reliability over many differently-sized instances: every far instance is rejected and no
    #    bipartite instance is falsely rejected.
    detections = sum(test_bipartite(QueryGraph(triangle_union(20 + trial)), 0.25)[0]
                     == "non-bipartite" for trial in range(40))
    misses = sum(test_bipartite(QueryGraph(build_bipartite(60, 240, 100 + trial)), 0.25)[0]
                 != "bipartite" for trial in range(40))
    check(detections >= 38 and misses == 0,
          f"detection must be reliable: {detections}/40 rejects, {misses} false rejects")

    print(f"bipartite probes={probes} (edges={graph.edges}); far graph distance~{deletions} "
          f"triangles; near verdict={near_verdict}")
    print(f"probes {small_probes} -> {double_probes} when the instance doubles; "
          f"detections={detections}/40 misses={misses}")
    print("PASS")


if __name__ == "__main__":
    main()
