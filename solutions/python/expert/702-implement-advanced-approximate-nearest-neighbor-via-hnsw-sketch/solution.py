"""Task 702 - Implement advanced approximate nearest neighbor via HNSW sketch.  (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)

Hierarchical Navigable Small World graphs (Malkov & Yashunin) index vectors in a stack of
proximity graphs: every point lives on level 0 and is promoted to level L with probability
exp(-L / mL), mL = 1 / ln(M).  Search descends greedily through the sparse upper levels (which
act as a coarse skip-list over the data) and then runs a best-first beam search with width `ef`
on level 0, which is the dense navigable graph that makes the answer good.

Insertion follows the same route: the new point descends greedily to its own top level, and at
every level from there down it runs the beam search, keeps the `M` closest candidates as
neighbours (M_max0 = 2M on level 0, the paper's convention) and adds the back links, shrinking
any neighbour that exceeds its degree cap.  Layer 0 stays connected because every inserted point
receives at least one back-link, which is what makes the recall@k measurable against an exact
brute-force oracle.

Simplified versus the paper: the neighbour selection keeps the M nearest candidates instead of
running the diversity heuristic (Algorithm 4), deletion is not implemented, and distances are
plain integer squared Euclidean distances instead of SIMD/FDA code.  Stored vectors are the
exact inputs, so "sketch" here means the layered proximity graph, not a lossy code.

Verified: search output shape (k results, ascending distance, no duplicates), the graph
invariants after every insertion (no self loops, degree caps per level, nested levels matching
each node's assigned level, level 0 connected), exact self-queries return the query point first,
recall@10 against the brute-force oracle on both a smooth and a deliberately tie-heavy instance
(where a narrow beam is measurably approximate and the recall rises to exactly 1.0 as ef grows),
and the boundary cases (empty index, single point, k > n, k <= 0, duplicate vectors).
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


def random_vectors(count, dim, limit, seed):
    """Deterministic integer vectors; floats would only add rounding noise to the checks."""
    data = []
    for _ in range(count):
        vector = []
        for _ in range(dim):
            seed, r = splitmix64(seed)
            vector.append(r % limit)
        data.append(vector)
    return data, seed


def squared_distance(a, b):
    total = 0
    for x, y in zip(a, b):
        d = x - y
        total += d * d
    return total


class HNSW:
    def __init__(self, m=8, ef_construction=32, seed=0x123456789ABCDEF):
        self.m = m
        self.m_max0 = 2 * m
        self.ef_construction = ef_construction
        self.vectors = []
        self.node_level = []
        self.layers = [{}]                 # layers[level][node] -> set of neighbours
        self.entry = None
        self.max_level = -1
        self.state = seed
        self.ml = 1.0 / math.log(m)

    # ------------------------------------------------------------------ internals
    def _draw_level(self):
        self.state, r = splitmix64(self.state)
        u = (r + 1) / (2 ** 64 + 1)        # uniform in (0, 1), never 0 or 1
        return int(-math.log(u) * self.ml)

    def _dist(self, vector, node):
        return squared_distance(vector, self.vectors[node])

    def _greedy(self, vector, node, level):
        """Greedy descent on one level: walk to the locally closest neighbour, repeat."""
        best_node, best_dist = node, self._dist(vector, node)
        improved = True
        while improved:
            improved = False
            for neighbour in self.layers[level].get(best_node, ()):
                distance = self._dist(vector, neighbour)
                if distance < best_dist:
                    best_node, best_dist, improved = neighbour, distance, True
        return best_node, best_dist

    def _search_layer(self, vector, entry_points, ef, level):
        """Beam search of width ef; returns [(distance, node)] sorted by distance."""
        visited = set(entry_points)
        frontier = []                      # min-heap over (distance, node)
        beam = []                          # max-heap of the ef best seen, via (-distance, node)
        for node in entry_points:
            distance = self._dist(vector, node)
            heapq.heappush(frontier, (distance, node))
            heapq.heappush(beam, (-distance, node))
        while frontier:
            distance, node = heapq.heappop(frontier)
            if len(beam) >= ef and distance > -beam[0][0]:
                break                       # nothing better can be reached from here
            for neighbour in self.layers[level].get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                neighbour_dist = self._dist(vector, neighbour)
                if len(beam) < ef or neighbour_dist < -beam[0][0]:
                    heapq.heappush(frontier, (neighbour_dist, neighbour))
                    heapq.heappush(beam, (-neighbour_dist, neighbour))
                    while len(beam) > ef:
                        heapq.heappop(beam)
        result = [(-neg, node) for neg, node in beam]
        result.sort(key=lambda item: (item[0], item[1]))
        return result

    def _shrink(self, node, level):
        cap = self.m_max0 if level == 0 else self.m
        neighbours = self.layers[level][node]
        if len(neighbours) <= cap:
            return
        ranked = sorted(neighbours, key=lambda n: (self._dist(self.vectors[node], n), n))
        for dropped in ranked[cap:]:
            self.layers[level][dropped].discard(node)
        self.layers[level][node] = set(ranked[:cap])

    # ------------------------------------------------------------------ public API
    def add(self, vector):
        node = len(self.vectors)
        self.vectors.append(vector)
        level = self._draw_level()
        self.node_level.append(level)
        while len(self.layers) <= level:
            self.layers.append({})
        for lev in range(level + 1):
            self.layers[lev][node] = set()
        if self.entry is None:
            self.entry, self.max_level = node, level
            return node

        current = self.entry
        for lev in range(self.max_level, level, -1):        # greedy phase, above the new level
            current, _ = self._greedy(vector, current, lev)
        for lev in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(vector, [current], self.ef_construction, lev)
            chosen = [n for _, n in candidates[:self.m]]
            self.layers[lev][node] = set(chosen)
            for neighbour in chosen:
                self.layers[lev][neighbour].add(node)
                self._shrink(neighbour, lev)
            if chosen:
                current = chosen[0]
        if level > self.max_level:
            self.max_level, self.entry = level, node
        return node

    def build(self, vectors):
        for vector in vectors:
            self.add(vector)
        return self

    def search(self, vector, k=10, ef=None):
        if self.entry is None or k <= 0:
            return []
        ef = max(ef if ef is not None else k, k)
        current = self.entry
        for lev in range(self.max_level, 0, -1):
            current, _ = self._greedy(vector, current, lev)
        return self._search_layer(vector, [current], ef, 0)[:k]

    def brute_force(self, vector, k=10):
        ranked = sorted((squared_distance(vector, v), n) for n, v in enumerate(self.vectors))
        return ranked[:k]


def recall_at_k(index, queries, k, ef):
    hits, total = 0, 0
    for query in queries:
        truth = {node for _, node in index.brute_force(query, k)}
        found = {node for _, node in index.search(query, k, ef)}
        hits += len(truth & found)
        total += len(truth)
    return hits / total if total else 1.0


# --------------------------------------------------------------------------- checks
def make_index(count=200, dim=6, limit=1000, seed=0xC0FFEE):
    data, _ = random_vectors(count, dim, limit, seed)
    index = HNSW(m=8, ef_construction=32, seed=0xABCDEF1234567890)
    index.build(data)
    return index, data


def test_distance_basics():
    a, b = [1, 2, 3], [4, 0, 3]
    check(squared_distance(a, b) == 9 + 4 + 0, "squared euclidean distance")
    check(squared_distance(a, b) == squared_distance(b, a), "distance must be symmetric")
    check(squared_distance(a, a) == 0, "distance to self is zero")


def test_graph_invariants():
    index, data = make_index()
    check(len(index.vectors) == len(data), "every vector must be stored")
    check(len(index.layers[0]) == len(data), "all nodes live on level 0")
    check(index.node_level[index.entry] == index.max_level, "entry point sits on the top level")
    check(len(index.layers) == index.max_level + 1, "layer count matches the top level")

    for node, level in enumerate(index.node_level):
        for lev in range(index.max_level + 1):
            present = node in index.layers[lev]
            check(present == (lev <= level),
                  f"node {node} (level {level}) presence on layer {lev} is {present}")
        for lev in range(level + 1):
            neighbours = index.layers[lev][node]
            check(node not in neighbours, f"self loop at node {node} level {lev}")
            cap = index.m_max0 if lev == 0 else index.m
            check(len(neighbours) <= cap,
                  f"degree cap violated at node {node} level {lev}: {len(neighbours)} > {cap}")
            for neighbour in neighbours:
                check(neighbour in index.layers[lev], "neighbour must exist on the same layer")
                check(node in index.layers[lev][neighbour],
                      f"back-link missing {node} <-> {neighbour} on level {lev}")

    seen, stack = {0}, [0]                            # level 0 must be one component
    while stack:
        node = stack.pop()
        for neighbour in index.layers[0][node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    check(len(seen) == len(data), f"level 0 must be connected, reached {len(seen)} of {len(data)}")


def test_self_queries_are_exact():
    index, data = make_index()
    for node in range(0, len(data), 7):
        results = index.search(data[node], k=1, ef=16)
        check(results[0][1] == node, f"self query for node {node} returned {results[0][1]}")
        check(results[0][0] == 0, "a self query has distance zero")


def test_recall_against_brute_force():
    # A deliberately hard instance: two dimensions with only eight distinct coordinates, so the
    # neighbour lists are full of near-ties and approximation quality becomes visible.
    data, _ = random_vectors(300, 2, 8, 0xC0FFEE)
    index = HNSW(m=8, ef_construction=32, seed=0xABCDEF1234567890).build(data)
    queries, _ = random_vectors(40, 2, 8, 0x5EED)
    narrow = recall_at_k(index, queries, 10, 8)
    medium = recall_at_k(index, queries, 10, 16)
    wide = recall_at_k(index, queries, 10, 32)
    full = recall_at_k(index, queries, 10, 200)
    check(0.4 <= narrow <= 0.95, f"a narrow beam must be visibly approximate, got {narrow}")
    check(narrow <= medium <= wide, f"recall must not fall as ef grows: {narrow} {medium} {wide}")
    check(wide == 1.0, f"ef = 32 already reaches the exact answer here, got {wide}")
    check(full == 1.0, f"ef = 200 must be exhaustive on this instance, got {full}")
    exact = recall_at_k(index, queries, 1, 16)
    check(exact == 1.0, f"recall@1 at ef = 16 should be perfect here, got {exact}")
    return narrow, medium, wide, full, exact


def test_recall_on_smooth_data():
    index, data = make_index(count=250, dim=6)
    queries, _ = random_vectors(40, 6, 1000, 0x5EED)
    recall_mid = recall_at_k(index, queries, 10, 50)
    check(recall_mid == 1.0, f"well-spread data must be answered exactly, got {recall_mid}")
    return recall_mid


def test_result_shape():
    index, data = make_index(count=120, dim=4, limit=50)
    query = [7, 7, 7, 7]
    for k in (1, 3, 10, 120, 200):
        results = index.search(query, k=k, ef=40)
        expected_len = min(k, len(data))
        check(len(results) == expected_len,
              f"k={k} must return {expected_len} results, got {len(results)}")
        distances = [d for d, _ in results]
        check(distances == sorted(distances), f"k={k} results must be ascending by distance")
        nodes = [n for _, n in results]
        check(len(set(nodes)) == len(nodes), f"k={k} results must be distinct")
        check(all(0 <= n < len(data) for n in nodes), f"k={k} node ids in range")
    check(index.search(query, k=0) == [], "k = 0 returns nothing")
    check(index.search(query, k=-3) == [], "negative k returns nothing")


def test_boundaries():
    empty = HNSW(m=8)
    check(empty.search([1, 2, 3], k=5) == [], "an empty index has no answer")
    check(empty.brute_force([1, 2, 3], k=5) == [], "the oracle agrees on the empty index")

    single = HNSW(m=8).build([[5, 5]])
    check(len(single.search([0, 0], k=4, ef=8)) == 1, "a single point is always found")
    check(single.search([0, 0], k=1)[0][0] == 50, "distance 25 + 25 to the only point")

    zero_vector = HNSW(m=8).build([[0, 0], [0, 0], [1, 1]])
    results = zero_vector.search([0, 0], k=2, ef=8)
    check(len(results) == 2 and results[0][0] == 0, "duplicate vectors are reachable at distance 0")
    check(all(d == 0 for d, _ in results[:2]), "both duplicates must be found first")
    check(zero_vector.search([1, 1], k=1, ef=8)[0][0] == 0, "exact hit on the non-duplicate")


def test_level_distribution():
    index, data = make_index(count=300, dim=3, limit=7)
    counts = [len(index.layers[lev]) for lev in range(index.max_level + 1)]
    check(counts[0] == len(data), "level 0 holds every node (exp(0) = 1)")
    check(all(a >= b for a, b in zip(counts, counts[1:])), f"levels must shrink: {counts}")
    check(1 <= index.max_level <= 10, f"max level should stay small, got {index.max_level}")
    check(counts[-1] >= 1, "the top level is non-empty")
    check(sum(1 for level in index.node_level if level == 0) > 0, "level 0 nodes exist")


def main():
    test_distance_basics()
    test_graph_invariants()
    test_self_queries_are_exact()
    narrow, medium, wide, full, exact = test_recall_against_brute_force()
    smooth = test_recall_on_smooth_data()
    test_result_shape()
    test_boundaries()
    test_level_distribution()

    index, _ = make_index(count=200, dim=6)
    print(f"nodes={len(index.vectors)} layers={len(index.layers)} m={index.m} "
          f"max_level={index.max_level} smooth_recall@10(ef=50)={smooth:.3f}")
    print(f"hard_recall@10 ef=8:{narrow:.3f} ef=16:{medium:.3f} ef=32:{wide:.3f} ef=200:{full:.3f} "
          f"recall@1(ef=16)={exact:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
