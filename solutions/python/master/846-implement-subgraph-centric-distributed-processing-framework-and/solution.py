"""Task 846 - Implement subgraph-centric distributed processing framework and test on motifs.  (Tier: Master, CTCI Beyond — System Design and Scalability)

Subgraph-centric (a.k.a. partition-centric) computation: a graph is split into parts, each part
owns a contiguous block of vertices and *replicates* every neighbour that lives elsewhere as a
ghost, then each part runs a sequential algorithm over its own vertices plus those ghosts.  All
the parallelism is expressed as a deterministic schedule over plain data structures - no
threads, sockets or clocks - so results are reproducible and can be compared with a brute-force
central computation.

What the framework implements:
  * `Partitioner` - balanced contiguous parts plus ghost replication (each ghost carries its
    full adjacency, the one-hop replication a subgraph-centric engine relies on);
  * `part_cost` - the metadata a scheduler would use: part sizes, ghost counts, cut edges and
    the replication factor;
  * motif counting with the *canonical owner* rule: a triangle is attributed to the part owning
    its smallest vertex, a star to the part owning its center, a 3-path to the part owning the
    smaller endpoint of its middle edge.  Every motif therefore has exactly one owner, so the
    parts' totals sum to the exact global count - no double counting and no coordination;
  * boundary label propagation (connected components) where ghosts forward labels across cut
    edges for a bounded number of rounds until the local labels stop changing.

Simplified versus production: instead of a coordinator that touches a shared graph table, the
parts read the same immutable adjacency dictionary; the "network" is the ghost replication
itself and the label exchange is done in rounds inside one process.  The measured cost model is
the one that matters at scale: ghost replication, cut edges and messages per round.

Verified: the partition is a partition (every vertex owned exactly once, ghosts exactly the
outside neighbours), motif counts match a brute-force central count on path, cycle, clique, star,
bipartite, random, disconnected and empty graphs, the ownership rule is *necessary* (the naive
count-everywhere total is strictly larger on graphs with cut triangles - the negative control),
every triangle a part counts lies inside that part's local view, connected components match a
union-find reference with a bounded number of message rounds, and the whole run is reproducible.
"""

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        z = (self.state + GOLDEN) & MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        self.state = (z ^ (z >> 31)) & MASK64
        return self.state

    def below(self, limit):
        return self.next() % limit


class Partitioner:
    """Splits an undirected graph into balanced parts with one-hop ghost replication."""

    def __init__(self, graph, parts):
        self.graph = {node: set(neighbours) for node, neighbours in graph.items()}
        for node in list(self.graph):
            for neighbour in list(self.graph[node]):
                self.graph.setdefault(neighbour, set()).add(node)
        self.vertices = sorted(self.graph)
        self.parts = parts
        self.owner = {}
        self.owned = {index: [] for index in range(parts)}
        for position, vertex in enumerate(self.vertices):
            block = position * parts // max(1, len(self.vertices)) if self.vertices else 0
            block = min(block, parts - 1)
            self.owner[vertex] = block
            self.owned[block].append(vertex)
        self.ghosts = {}
        for index in range(parts):
            local = set(self.owned[index])
            outside = set()
            for vertex in self.owned[index]:
                outside.update(self.graph[vertex] - local)
            self.ghosts[index] = outside

    def local_view(self, index):
        """Everything the part can see: its vertices plus the replicated frontier."""
        return set(self.owned[index]) | self.ghosts[index]

    def cut_edges(self):
        return sorted({(min(a, b), max(a, b)) for a in self.graph for b in self.graph[a]
                       if self.owner[a] != self.owner[b]})

    def cost(self):
        return {"parts": self.parts, "owned": {index: len(self.owned[index])
                                               for index in range(self.parts)},
                "ghosts": {index: len(self.ghosts[index]) for index in range(self.parts)},
                "cut_edges": len(self.cut_edges()),
                "replication": sum(len(self.ghosts[index]) for index in range(self.parts))}

    def neighbour_view(self, index, vertex):
        """The part's view of a vertex's neighbours (complete, thanks to replication)."""
        return self.graph[vertex]

    # -- motifs ------------------------------------------------------------
    def triangles_owned(self, index, naive=False):
        """Triangles attributed to this part (or all triangles it can see, for the control).

        The apex ordering `vertex < first < second` enumerates every triangle exactly once, at
        its smallest vertex, so the ownership test really is "the part owning the smallest
        vertex counts it".
        """
        view = self.local_view(index)
        found = []
        for vertex in sorted(view):
            for first in sorted(self.graph[vertex]):
                for second in sorted(self.graph[vertex]):
                    if not vertex < first < second:
                        continue
                    if second not in self.graph[first]:
                        continue
                    if first not in view or second not in view:
                        continue
                    if naive or self.owner[vertex] == index:
                        found.append((vertex, first, second))
        return found

    def stars_owned(self, index):
        stars = 0
        for vertex in self.owned[index]:
            degree = len(self.graph[vertex])
            stars += degree * (degree - 1) // 2
        return stars

    def paths_owned(self, index):
        total = 0
        for vertex in self.owned[index]:
            for neighbour in self.graph[vertex]:
                if neighbour <= vertex:
                    continue
                total += (len(self.graph[vertex]) - 1) * (len(self.graph[neighbour]) - 1)
        return total

    # -- boundary label propagation ----------------------------------------
    def components(self, max_rounds=None):
        """Local union-find per part, then min-label propagation across the cut edges.

        The local phase can only use edges that lie entirely inside the part's view; whatever
        crosses a part boundary is resolved by exchanging labels along the cut edges, and every
        part then pushes the resulting minimum down through its *local* components (its own
        vertices and the ghosts they are connected to).  The result is the global component
        labelling, plus the message cost.
        """
        labels = {vertex: vertex for vertex in self.vertices}
        groups = []
        for index in range(self.parts):
            view = self.local_view(index)
            parent = {vertex: vertex for vertex in view}

            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            for vertex in sorted(view):
                for neighbour in sorted(self.graph[vertex]):
                    if neighbour not in view:
                        continue
                    left, right = find(vertex), find(neighbour)
                    if left != right:
                        parent[max(left, right)] = min(left, right)
            roots = {vertex: find(vertex) for vertex in view}
            groups.append(roots)
            for vertex in self.owned[index]:
                labels[vertex] = roots[vertex]

        cut = self.cut_edges()
        messages = 0
        rounds = 0
        limit = max_rounds or max(1, 4 * len(self.vertices) + 4)
        while rounds < limit:
            rounds += 1
            changed = False
            for first, second in cut:
                messages += 2
                if labels[first] < labels[second]:
                    labels[second] = labels[first]
                    changed = True
                elif labels[second] < labels[first]:
                    labels[first] = labels[second]
                    changed = True
            for roots in groups:
                members = {}
                for vertex, root in roots.items():
                    members.setdefault(root, []).append(vertex)
                for vertices in members.values():
                    smallest = min(labels[vertex] for vertex in vertices)
                    for vertex in vertices:
                        if labels[vertex] != smallest:
                            labels[vertex] = smallest
                            changed = True
            if not changed:
                break
        return labels, {"rounds": rounds, "messages": messages}


# --------------------------------------------------------------------------------------
# brute-force references
# --------------------------------------------------------------------------------------
def undirected(graph):
    edges = {node: set(neighbours) for node, neighbours in graph.items()}
    for node in list(edges):
        for neighbour in list(edges[node]):
            edges.setdefault(neighbour, set()).add(node)
    return edges


def reference_triangles(graph):
    edges = undirected(graph)
    count = 0
    for vertex in sorted(edges):
        for first in sorted(edges[vertex]):
            for second in sorted(edges[vertex]):
                if vertex < first < second and second in edges[first]:
                    count += 1
    return count


def reference_stars(graph):
    return sum(len(neighbours) * (len(neighbours) - 1) // 2 for neighbours in graph.values())


def reference_paths(graph):
    edges = undirected(graph)
    total = 0
    for vertex in sorted(edges):
        for neighbour in edges[vertex]:
            if neighbour <= vertex:
                continue
            total += (len(edges[vertex]) - 1) * (len(edges[neighbour]) - 1)
    return total


def reference_components(graph):
    edges = undirected(graph)
    parent = {node: node for node in edges}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for node in sorted(edges):
        for neighbour in edges[node]:
            left, right = find(node), find(neighbour)
            if left != right:
                parent[max(left, right)] = min(left, right)
    return {node: find(node) for node in edges}


def gallery():
    """Small graphs whose motif counts are known by construction."""
    graphs = {
        "empty": {},
        "single": {0: []},
        "edge": {0: [1], 1: [0]},
        "path4": {0: [1], 1: [2], 2: [3], 3: []},
        "cycle6": {index: [(index + 1) % 6] for index in range(6)},
        "clique5": {index: [other for other in range(5) if other != index] for index in range(5)},
        "star6": {0: list(range(1, 7))},
        "bipartite": {0: [3, 4], 1: [3, 4], 2: [4, 5], 3: [], 4: [], 5: []},
        "two_components": {0: [1], 1: [0], 2: [3], 3: [2]},
    }
    rng = Rng(4242)
    random_graph = {index: [] for index in range(14)}
    for index in range(14):
        for other in range(index + 1, 14):
            if rng.below(100) < 22:
                random_graph[index].append(other)
    graphs["random14"] = random_graph
    return graphs


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    graphs = gallery()
    total_parts = 0
    total_triangles = 0

    # ---- 1. the partition itself -----------------------------------------
    for name, graph in graphs.items():
        edges = undirected(graph)
        for parts in (1, 2, 3, 5):
            partitioner = Partitioner(graph, parts)
            check(sorted(partitioner.owned[index] for index in range(parts)) ==
                  sorted([[vertex for vertex in partitioner.vertices
                           if partitioner.owner[vertex] == index] for index in range(parts)]),
                  "%s/%d: every vertex must be owned by exactly one part" % (name, parts))
            for index in range(parts):
                outside = set()
                for vertex in partitioner.owned[index]:
                    outside.update(edges[vertex] - set(partitioner.owned[index]))
                check(partitioner.ghosts[index] == outside,
                      "%s/%d: ghosts must be exactly the outside neighbours" % (name, parts))
            if partitioner.vertices:
                sizes = [len(partitioner.owned[index]) for index in range(parts)]
                check(max(sizes) - min(sizes) <= 1,
                      "%s/%d: the partition must be balanced (%r)" % (name, parts, sizes))
            total_parts += parts

    # ---- 2. motif counts against the central brute force ------------------
    for name, graph in graphs.items():
        edges = undirected(graph)
        expected_triangles = reference_triangles(graph)
        expected_stars = reference_stars(edges)
        expected_paths = reference_paths(graph)
        for parts in (1, 2, 3, 5):
            partitioner = Partitioner(graph, parts)
            seen = set()
            raw = 0
            for index in range(parts):
                for triangle in partitioner.triangles_owned(index):
                    raw += 1
                    seen.add(tuple(sorted(triangle)))
                    check(all(vertex in partitioner.local_view(index) for vertex in triangle),
                          "%s/%d: a counted triangle must lie in the local view"
                          % (name, parts))
                check(partitioner.stars_owned(index) >= 0 and
                      partitioner.paths_owned(index) >= 0, "counts are non-negative")
            triangles = len(seen)
            check(raw == triangles,
                  "%s/%d: no triangle may be counted by two parts (%d vs %d)"
                  % (name, parts, raw, triangles))
            stars_total = sum(partitioner.stars_owned(index) for index in range(parts))
            paths_total = sum(partitioner.paths_owned(index) for index in range(parts))
            check(triangles == expected_triangles,
                  "%s/%d: triangles %d vs %d" % (name, parts, triangles, expected_triangles))
            check(stars_total == expected_stars,
                  "%s/%d: 2-stars %d vs %d" % (name, parts, stars_total, expected_stars))
            check(paths_total == expected_paths,
                  "%s/%d: 3-paths %d vs %d" % (name, parts, paths_total, expected_paths))
            total_triangles += triangles

    # ---- 3. the ownership rule is necessary (negative control) -----------
    clique = Partitioner(gallery()["clique5"], 3)
    naive = sum(len(clique.triangles_owned(index, naive=True)) for index in range(3))
    owned = sum(len(clique.triangles_owned(index)) for index in range(3))
    check(owned == reference_triangles(gallery()["clique5"]) == 10,
          "the clique holds ten triangles")
    check(naive > owned,
          "counting every visible triangle must double count (%d vs %d)" % (naive, owned))
    single = Partitioner(gallery()["clique5"], 1)
    check(sum(len(single.triangles_owned(index, naive=True)) for index in range(1)) ==
          sum(len(single.triangles_owned(index)) for index in range(1)),
          "with a single part the two rules coincide")

    # ---- 4. the cost model ------------------------------------------------
    random_graph = gallery()["random14"]
    for parts in (1, 2, 4, 7):
        partitioner = Partitioner(random_graph, parts)
        cost = partitioner.cost()
        check(cost["replication"] == sum(len(partitioner.ghosts[index])
                                         for index in range(parts)),
              "the replication factor is the number of ghost copies")
        check(cost["cut_edges"] <= sum(len(neighbours) for neighbours in
                                       undirected(random_graph).values()) // 2,
              "cut edges cannot exceed all edges")
        if parts == 1:
            check(cost["replication"] == 0 and cost["cut_edges"] == 0,
                  "a single part has no ghosts and no cut edges")
    many = Partitioner(random_graph, 7).cost()
    none = Partitioner(random_graph, 1).cost()
    check(many["replication"] > none["replication"] == 0,
          "more parts must replicate more frontier vertices")

    # ---- 5. boundary label propagation -----------------------------------
    for name, graph in graphs.items():
        expected = reference_components(graph)
        for parts in (1, 2, 3, 4):
            partitioner = Partitioner(graph, parts)
            labels, stats = partitioner.components()
            check(labels == expected,
                  "%s/%d: components must match union-find" % (name, parts))
            check(stats["rounds"] <= max(1, 4 * len(partitioner.vertices) + 4),
                  "propagation must terminate")
            check(stats["messages"] <= stats["rounds"] * 2 * max(1, len(partitioner.cut_edges())),
                  "messages per round are bounded by the cut edges")

    # ---- 6. determinism ---------------------------------------------------
    first = Partitioner(gallery()["random14"], 4)
    second = Partitioner(gallery()["random14"], 4)
    check(first.cost() == second.cost() and first.ghosts == second.ghosts,
          "partitioning must be reproducible")
    check([first.triangles_owned(index) for index in range(4)] ==
          [second.triangles_owned(index) for index in range(4)],
          "motif enumeration must be reproducible")
    check(first.components()[0] == second.components()[0], "labels must be reproducible")
    check(first.cost()["replication"] > 0 and total_parts > 8,
          "the sweep must cover several partitionings (%d)" % total_parts)
    check(total_triangles > 0, "the gallery must contain triangles (%d counted)" % total_triangles)

    print("gallery: %d graphs, %d partitionings, %d triangles counted in total"
          % (len(graphs), total_parts, total_triangles))
    print("clique5: owned %d vs naive %d (double counting without the ownership rule)"
          % (owned, naive))
    print("random14 cost: %s" % (Partitioner(random_graph, 7).cost(),))
    print("PASS")


if __name__ == "__main__":
    main()
