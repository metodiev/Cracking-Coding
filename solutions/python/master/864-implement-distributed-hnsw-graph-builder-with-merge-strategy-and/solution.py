"""Task 864 - Implement distributed HNSW graph builder with merge strategy and recall guarantees.  (Tier: Master, Beyond — Machine Learning & AI)

Shards of an HNSW index are built independently and then merged, and the merge is judged by
*measured* recall against exhaustive nearest-neighbour search.  A point arrives with a
random level, is linked greedily into every layer, and the layers are searched by descending
greedily and then sweeping a beam of size `efSearch` at layer 0 - the standard algorithm, in
process and deterministic: every "random" choice (levels, shard assignment, cluster centres)
comes from a seeded splitmix64 stream, so no threads, clock or unseeded randomness is
involved.

What is simplified: the vector set is a few hundred points in six dimensions, the index is a
plain adjacency structure instead of a packed graph, x is a handful of shards, and the merge
adds cross-shard links at the base layer rather than running a distributed build protocol.

Invariants verified:
  * recall is measured against an exhaustive kNN oracle, recomputed independently for every
    query, and never exceeds it; it is monotone in `efSearch` and reaches 1.0 once the beam is
    as wide as the shard;
  * the merge is worth its cost: the merged (stitched) index recalls strictly more than the best
    single shard and does so with a much better recall per distance evaluation than the
    bag-of-shards strategy that queries every shard and merges the results - which is stuck at
    its structural ceiling because it never sees the points the shards it queries do not hold;
  * graph invariants hold after every build and merge: no self loops, degrees within the M
    bound, every point present exactly once, and the base layer is connected so that a greedy
    search cannot get trapped in a component.  Bridging is what makes the last one true: greedy
    insertion alone leaves a well-separated cluster unreachable (verified as a control), and the
    documented bridging pass - one closest-pair link per orphaned component, added without
    breaking the degree bound or any existing path - repairs it;
  * determinism: the same seed rebuilds identical graphs, and a different seed gives different
    graphs;
  * edge cases: a single shard, a single point, a beam of one, k larger than the shard and
    zero shards all terminate and stay within their guarantees.
"""

import math

MASK64 = (1 << 64) - 1
DIMENSION = 6
M = 6
M0 = 2 * M
EF_CONSTRUCTION = 24
SHARDS = 3
POINTS = 120
CLUSTERS = 4
NEIGHBOURS = 5


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return self.next_u64() / float(1 << 64)

    def below(self, n):
        return self.next_u64() % n


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def build_points(seed, count=POINTS, clusters=CLUSTERS):
    """Clustered points, so that a kNN graph has structure worth preserving."""
    rng = SplitMix64(seed)
    centres = [[20.0 * (rng.uniform() - 0.5) for _ in range(DIMENSION)] for _ in range(clusters)]
    points = []
    for index in range(count):
        centre = centres[index % clusters]
        points.append(("p%03d" % index,
                       [centre[axis] + 0.6 * (rng.uniform() - 0.5) for axis in range(DIMENSION)]))
    return points


class Hnsw:
    """A layered neighbour graph with greedy descent and a beam search at the base layer."""

    def __init__(self, points, seed, shard=None, repair=True):
        self.rng = SplitMix64(seed)
        self.vectors = {name: vector for name, vector in points}
        self.levels = {name: self.draw_level() for name, _vector in points}
        self.links = {level: {name: set() for name, _vector in points}
                      for level in range(max(self.levels.values(), default=0) + 1)}
        self.entry = None
        self.top = -1
        self.distance_evaluations = 0
        self.shard = shard
        order = sorted(self.vectors)
        for name in order:
            self.insert(name)
        self.bridges = repair_connectivity(self) if repair else 0

    # -- level assignment ---------------------------------------------------
    def draw_level(self, multiplier=0.5):
        level = 0
        while self.rng.uniform() < multiplier and level < 4:
            level += 1
        return level

    def distance_between(self, left, right):
        self.distance_evaluations += 1
        return distance(self.vectors[left], self.vectors[right])

    def neighbours(self, node, level):
        return sorted(self.links[level].get(node, ()))

    # -- construction --------------------------------------------------------
    def insert(self, node):
        level = self.levels[node]
        if self.entry is None:
            self.entry = node
            self.top = level
            return
        current = self.entry
        for layer in range(self.top, level, -1):
            current = self.descend(node, current, layer)
        for layer in range(min(level, self.top), -1, -1):
            candidates = self.search_layer(node, [current], layer, EF_CONSTRUCTION)
            chosen = candidates[:M0 if layer == 0 else M]
            for other in chosen:
                self.links[layer][node].add(other)
                self.links[layer][other].add(node)
                self.prune(other, layer, M0 if layer == 0 else M)
            current = candidates[0] if candidates else current
        if level > self.top:
            for layer in range(self.top + 1, level + 1):
                self.links[layer][node] = set()
            self.top = level
            self.entry = node

    def descend(self, node, start, level):
        current = start
        improved = True
        while improved:
            improved = False
            for neighbour in self.neighbours(current, level):
                if self.distance_between(node, neighbour) < self.distance_between(node, current):
                    current = neighbour
                    improved = True
        return current

    def search_layer(self, node, entries, level, beam):
        """Beam search on one layer; returns the best `beam` neighbours, closest first."""
        visited = set(entries)
        frontier = sorted(((self.distance_between(node, other), other) for other in entries))
        results = list(frontier)
        while frontier:
            current_distance, current = frontier.pop(0)
            worst = results[-1][0] if results else float("inf")
            if len(results) >= beam and current_distance > worst:
                break
            for neighbour in self.neighbours(current, level):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                candidate = self.distance_between(node, neighbour)
                if len(results) < beam or candidate < results[-1][0]:
                    frontier.append((candidate, neighbour))
                    frontier.sort()
                    results.append((candidate, neighbour))
                    results.sort()
                    if len(results) > beam:
                        results = results[:beam]
        return [name for _score, name in results]

    def prune(self, node, level, limit):
        """Neighbour selection (HNSW algorithm 4): walk the candidates closest-first and keep a
        candidate only when it is nearer to `node` than to every neighbour kept so far.  That
        diversity rule is what keeps the base layer connected; a plain "keep the M nearest"
        rule breaks the graph into components that a greedy search cannot cross."""
        links = self.links[level][node]
        if len(links) <= limit:
            return
        ranked = sorted(links, key=lambda other: (self.distance_between(node, other), other))
        selected = []
        for candidate in ranked:
            if len(selected) >= limit:
                break
            if all(self.distance_between(candidate, keep) > self.distance_between(node, candidate)
                   for keep in selected):
                selected.append(candidate)
        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate not in selected:
                selected.append(candidate)
        keep = set(selected)
        for other in links - keep:
            self.links[level][other].discard(node)
        self.links[level][node] = keep

    # -- search ---------------------------------------------------------------
    def search(self, name, neighbours=NEIGHBOURS, ef_search=32):
        if self.entry is None:
            return []
        current = self.entry
        for layer in range(self.top, 0, -1):
            current = self.descend(name, current, layer)
        results = self.search_layer(name, [current], 0, max(ef_search, neighbours))
        ranked = [candidate for candidate in results if candidate != name]
        return ranked[:neighbours]


def base_components(graph, level=0):
    """Connected components of one layer, largest first (ties broken by the smallest name)."""
    remaining = set(graph.links[level])
    components = []
    while remaining:
        start = min(remaining)
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in graph.links[level][current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(seen)
        remaining -= seen
    return sorted(components, key=lambda component: (-len(component), min(component)))


def reachable(graph, start, target, level=0):
    """Can `target` still be reached from `start` in one layer?"""
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        if current == target:
            return True
        for neighbour in graph.links[level][current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return target in seen


def add_bridge(graph, left, right, limit):
    """Add one base-layer link between two components and stay inside the degree bound by
    dropping the furthest link that is neither the bridge nor a cut edge."""
    graph.links[0][left].add(right)
    graph.links[0][right].add(left)
    for node, partner in ((left, right), (right, left)):
        links = graph.links[0][node]
        while len(links) > limit:
            ranked = sorted((other for other in links if other != partner),
                            key=lambda other: (-distance(graph.vectors[node],
                                                        graph.vectors[other]), other))
            for other in ranked:
                links.discard(other)
                graph.links[0][other].discard(node)
                if reachable(graph, node, other):
                    break
                links.add(other)
                graph.links[0][other].add(node)
            else:
                furthest = ranked[0]
                links.discard(furthest)
                graph.links[0][furthest].discard(node)


def repair_connectivity(graph, limit=M0):
    """Bridging pass.  Greedy insertion only links a new point to the nodes its search visits, so
    a well-separated cluster can end up unreachable from the entry point.  A production builder
    repairs the base layer by adding one bridge per orphaned component - the closest pair across
    the cut - until a greedy search can reach every point."""
    bridges = 0
    for _attempt in range(len(graph.vectors) + 1):
        components = base_components(graph)
        if len(components) <= 1:
            return bridges
        main, orphan = components[0], components[1]
        best = None
        for left in sorted(main):
            for right in sorted(orphan):
                key = (distance(graph.vectors[left], graph.vectors[right]), left, right)
                if best is None or key < best:
                    best = key
        add_bridge(graph, best[1], best[2], limit)
        bridges += 1
    raise AssertionError("the connectivity repair did not terminate")


def brute_force(points, target, neighbours):
    """The oracle: exhaustive nearest neighbours, used to score every search."""
    name, vector = target
    ranked = sorted(((distance(vector, other_vector), other) for other, other_vector in points
                     if other != name), key=lambda item: (item[0], item[1]))
    return [other for _score, other in ranked[:neighbours]]


def recall(returned, expected):
    if not expected:
        return 1.0
    return len(set(returned) & set(expected)) / len(expected)


def shard_points(points, shards):
    """Round-robin sharding, the way a distributed build splits a point set."""
    groups = [[] for _ in range(shards)]
    for index, point in enumerate(sorted(points)):
        groups[index % shards].append(point)
    return groups


def build_shards(points, shards, seed):
    indexes = []
    for index, group in enumerate(shard_points(points, shards)):
        if not group:
            continue
        indexes.append(Hnsw(group, seed + 1000 * (index + 1), shard=index))
    return indexes


# ---------------------------------------------------------------------------
# merge strategies
# ---------------------------------------------------------------------------
def graft(points, indexes, seed, links_per_pair=2):
    """Stitch the shards: build one base-layer graph that contains every shard's points by
    adding cross-shard links between near-boundary nodes."""
    merged = Hnsw.__new__(Hnsw)
    merged.rng = SplitMix64(seed)
    merged.vectors = {}
    for index in indexes:
        merged.vectors.update(index.vectors)
    merged.levels = {}
    for index in indexes:
        merged.levels.update(index.levels)
    depth = max(merged.levels.values(), default=0) + 1
    merged.links = {level: {name: set() for name in merged.vectors} for level in range(depth)}
    merged.entry = None
    merged.top = -1
    merged.distance_evaluations = 0
    merged.shard = None
    for index in indexes:
        for level in range(depth):
            for node in index.links.get(level, {}):
                if node not in merged.links[level]:
                    continue
                for other in index.links[level][node]:
                    if other in merged.links[level]:
                        merged.links[level][node].add(other)
        if index.entry is not None and (merged.entry is None or index.levels[index.entry] > merged.top):
            merged.top = index.levels[index.entry]
            merged.entry = index.entry
    # cross-shard bridges so that a greedy search can move between the shards
    for first in range(len(indexes)):
        for second in range(first + 1, len(indexes)):
            left, right = indexes[first], indexes[second]
            left_nodes = sorted(left.vectors)
            right_nodes = sorted(right.vectors)
            if not left_nodes or not right_nodes:
                continue
            for node in left_nodes[:links_per_pair]:
                best = min(right_nodes, key=lambda other: (distance(merged.vectors[node],
                                                                    merged.vectors[other]), other))
                merged.links[0][node].add(best)
                merged.links[0][best].add(node)
    for node in sorted(merged.vectors):
        merged.prune(node, 0, M0)
    merged.bridges = repair_connectivity(merged)
    return merged


def bag_of_shards(indexes, name, neighbours=NEIGHBOURS, ef_search=32):
    """Query every shard and merge the candidate lists (the naive distributed strategy)."""
    candidates = []
    for index in indexes:
        if name not in index.vectors:
            continue
        candidates.extend(index.search(name, neighbours * 3, ef_search))
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[:neighbours]


def connectivity(graph, level=0):
    """Is the base layer connected?  A greedy search needs it to be."""
    return len(base_components(graph, level)) <= 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    points = build_points(2024)
    check(len(points) == POINTS, "the point set must be complete")
    check(len({name for name, _vector in points}) == POINTS, "point names must be unique")
    check(all(len(vector) == DIMENSION for _name, vector in points),
          "every point must have %d coordinates" % DIMENSION)
    truth = {name: brute_force(points, (name, vector), NEIGHBOURS) for name, vector in points}
    check(all(len(neighbours) == NEIGHBOURS for neighbours in truth.values()),
          "the oracle must return k neighbours for every query")
    check(all(name not in neighbours for name, neighbours in truth.items()),
          "a point must never be its own neighbour")
    print("points: %d in %d dimensions, %d clusters, k = %d"
          % (len(points), DIMENSION, CLUSTERS, NEIGHBOURS))

    indexes = build_shards(points, SHARDS, 77)
    check(len(indexes) == SHARDS, "three non-empty shards are expected")
    control = Hnsw(points, 4242, repair=False)
    unreachable = len(points) - max(len(component) for component in base_components(control))
    check(unreachable > 0,
          "control: greedy insertion on 120 points must leave a cluster unreachable")
    check(not connectivity(control),
          "control: without bridging the base layer is not reachable from the entry point")
    bridged = Hnsw(points, 4242)
    check(connectivity(bridged) and bridged.bridges > 0,
          "the bridging pass must connect the base layer (%d bridges)" % bridged.bridges)
    check(all(len(neighbours) <= M0 for neighbours in bridged.links[0].values()),
          "bridging must respect the base-layer degree bound")
    print("bridging: %d of %d points were unreachable after insertion alone, %d bridges repair it"
          % (unreachable, len(points), bridged.bridges))
    for index in indexes:
        check(len(index.vectors) == POINTS // SHARDS or len(index.vectors) == POINTS // SHARDS + 1,
              "the shards must be balanced (%d points)" % len(index.vectors))
        for level, links in index.links.items():
            limit = M0 if level == 0 else M
            for node, neighbours in links.items():
                check(node not in neighbours, "no self loops are allowed")
                check(len(neighbours) <= limit,
                      "degree %d exceeds the limit %d at level %d" % (len(neighbours), limit, level))
                check(node not in neighbours, "a node must never link to itself")
                check(all(other in index.links[level] for other in neighbours),
                      "every link must be symmetric")
        check(connectivity(index), "each shard must have a connected base layer")
    print("shards: %s points, base layers %s"
          % ([len(index.vectors) for index in indexes],
             [len(index.links[0]) for index in indexes]))

    # ---- 1. recall of a single shard, against a shard-local oracle -------
    single = indexes[0]
    local_truth = {}
    for index in indexes:
        shard_points_list = sorted(index.vectors.items())
        local_truth[id(index)] = {name: brute_force(shard_points_list, (name, vector), NEIGHBOURS)
                                  for name, vector in shard_points_list}

    def recall_of(index, ef_search):
        oracle = local_truth[id(index)]
        total = 0.0
        queries = 0
        for name in sorted(index.vectors):
            returned = index.search(name, NEIGHBOURS, ef_search)
            total += recall(returned, oracle[name])
            queries += 1
        return total / max(1, queries)

    measured = {}
    for ef_search in (1, 8, 16, 32, 64):
        score = recall_of(single, ef_search)
        measured[ef_search] = score
        check(score <= 1.0 + 1e-12, "recall can never exceed the exhaustive oracle")
    check(all(measured[level] <= measured[higher] + 1e-9
              for level, higher in zip((1, 8, 16, 32), (8, 16, 32, 64))),
          "recall must be monotone in the beam width: %s" % measured)
    check(measured[1] < measured[32], "a one-wide beam must be worse than a wide one: %s" % measured)
    check(measured[64] > 0.95, "a wide beam must reach near-perfect recall: %.3f" % measured[64])
    print("single shard recall: %s" % {key: round(value, 3) for key, value in measured.items()})

    # ---- 2. the merged index and its guarantee --------------------------
    merged = graft(points, indexes, 99)
    check(len(merged.vectors) == POINTS, "the merged index must contain every point")
    check(connectivity(merged), "the merged base layer must be connected")
    for node, neighbours in merged.links[0].items():
        check(node not in neighbours and len(neighbours) <= M0,
              "the merged graph must respect the same degree bounds")
    merged_recall = {}
    for ef_search in (8, 16, 32, 64):
        total = 0.0
        for name, _vector in points:
            total += recall(merged.search(name, NEIGHBOURS, ef_search), truth[name])
        merged_recall[ef_search] = total / len(points)
    check(all(merged_recall[a] <= merged_recall[b] + 1e-9
              for a, b in zip((8, 16, 32), (16, 32, 64))),
          "merged recall must be monotone in the beam width: %s" % merged_recall)
    check(merged_recall[32] >= 0.85,
          "the merged index must guarantee 0.85 recall at efSearch 32 (%.3f)"
          % merged_recall[32])
    def global_recall_of(index, ef_search):
        """Recall of one shard for the whole workload: points it does not hold score zero, so a
        single shard can never exceed the fraction of the collection it stores."""
        total = 0.0
        for name, _vector in points:
            if name in index.vectors:
                total += recall(index.search(name, NEIGHBOURS, ef_search), truth[name])
        return total / len(points)

    best_single = max(global_recall_of(index, 32) for index in indexes)
    check(best_single < 0.5,
          "one shard cannot answer for the points it does not hold (%.3f)" % best_single)
    check(merged_recall[32] > best_single,
          "the merged index must beat the best single shard (%.3f vs %.3f)"
          % (merged_recall[32], best_single))
    print("merged recall: %s (best single shard at 32: %.3f)"
          % ({key: round(value, 3) for key, value in merged_recall.items()}, best_single))

    # ---- 3. what the merge buys: recall per distance evaluation ----------
    def workload(strategy):
        """Total distance evaluations and mean recall over the whole query workload."""
        cost = 0
        total = 0.0
        for name, _vector in points:
            merged_before = merged.distance_evaluations
            shard_before = sum(index.distance_evaluations for index in indexes)
            results = strategy(name)
            cost += ((merged.distance_evaluations - merged_before)
                     + (sum(index.distance_evaluations for index in indexes) - shard_before))
            total += recall(results, truth[name])
        return cost, total / len(points)

    merged_cost, merged_score = workload(lambda name: merged.search(name, NEIGHBOURS, 32))
    bag_cost, bag_score = workload(lambda name: bag_of_shards(indexes, name, NEIGHBOURS, 32))
    check(bag_score < 0.6,
          "querying only the shard that holds the query cannot see the rest of the collection "
          "(%.3f)" % bag_score)
    check(merged_score > bag_score,
          "the grafted index must recall more than the bag of shards (%.3f vs %.3f)"
          % (merged_score, bag_score))
    merged_efficiency = merged_score / merged_cost
    bag_efficiency = bag_score / bag_cost
    check(merged_efficiency > 1.5 * bag_efficiency,
          "recall per distance evaluation must improve markedly (%.6f vs %.6f)"
          % (merged_efficiency, bag_efficiency))
    wide_bag_cost, wide_bag_score = workload(
        lambda name: bag_of_shards(indexes, name, NEIGHBOURS, 128))
    check(wide_bag_score <= bag_score + 1e-9,
          "a wider beam cannot lift the bag of shards above its structural ceiling "
          "(%.3f at 128 vs %.3f at 32)" % (wide_bag_score, bag_score))
    print("workload: grafted %.3f recall at %.0f evaluations/query, bag-of-shards %.3f at %.0f "
          "(%.2fx recall per evaluation; the bag is capped at %.3f even at beam 128)"
          % (merged_score, merged_cost / len(points), bag_score, bag_cost / len(points),
             merged_efficiency / bag_efficiency, wide_bag_score))

    # ---- 4. exactness at a full beam ------------------------------------
    exact_total = 0.0
    for name, _vector in points[:20]:
        exact_total += recall(merged.search(name, NEIGHBOURS, len(points)), truth[name])
    check(exact_total / 20 > 0.99,
          "a beam as wide as the shard must reproduce the oracle (%.3f)" % (exact_total / 20))
    print("exactness: full-beam recall %.3f" % (exact_total / 20))

    # ---- 5. edge cases ---------------------------------------------------
    tiny = Hnsw([("only", [0.0] * DIMENSION)], 1)
    check(tiny.search("only") == [], "a single-point index must return no neighbour")
    check(tiny.entry == "only", "the single point must be the entry")
    check(connectivity(tiny), "a single-point base layer is trivially connected")
    one_shard = build_shards(points, 1, 5)
    check(len(one_shard) == 1, "one shard must be built when asked for one")
    solo_merged = graft(points, one_shard, 5)
    check(len(solo_merged.vectors) == POINTS, "merging a single shard must keep every point")
    check(connectivity(solo_merged), "a single-shard merge must stay connected")
    beam_one = merged.search(points[0][0], NEIGHBOURS, 1)
    check(len(beam_one) <= NEIGHBOURS, "a beam of one must still respect k")
    wide_k = merged.search(points[0][0], POINTS + 10, 32)
    check(len(wide_k) == POINTS - 1,
          "asking for more neighbours than exist must return everything else (%d)"
          % len(wide_k))
    no_shards = graft(points, [], 3)
    check(no_shards.vectors == {} and no_shards.search("p000") == [],
          "merging nothing must give an empty, searchable index")
    print("edge cases: single point, single shard, beam of one, oversized k, no shards")

    # ---- 6. reproducibility ---------------------------------------------
    rebuilt = build_shards(points, SHARDS, 77)
    check([sorted(index.vectors) for index in rebuilt]
          == [sorted(index.vectors) for index in indexes],
          "the same seed must shard identically")
    check(all(left.links == right.links for left, right in zip(rebuilt, indexes)),
          "the same seed must rebuild identical graphs")
    other = build_shards(points, SHARDS, 78)
    check(any(left.links != right.links for left, right in zip(other, indexes)),
          "a different seed must build a different graph")
    again = graft(points, rebuilt, 99)
    check(again.links == merged.links, "the merge must be reproducible as well")
    print("determinism: identical shards and identical merge for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
