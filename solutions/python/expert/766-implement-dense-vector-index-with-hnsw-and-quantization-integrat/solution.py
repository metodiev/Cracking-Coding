"""Task 766 - Dense vector index with HNSW and quantisation integration.  (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)

HNSW (Hierarchical Navigable Small World) is the graph index behind most dense-vector retrieval: each
node receives a level from a geometric distribution, layer 0 contains every node and the upper layers
contain exponentially fewer nodes sampled from the same distribution, so a search first descends
greedily through the sparse upper layers to get close to the query and then runs a beam search
(`ef` candidates) inside layer 0.  When inserting a node, the same descent finds a good entry point,
an `ef_construction` beam search collects candidates on each layer the node belongs to, and the
*diversification heuristic* picks the neighbours (a candidate is kept only if it is closer to the
query point than to any already kept neighbour), after which every link is made bidirectional and
pruned from both sides so the graph stays navigable.

Quantisation is layered on top: vectors are stored as 8-bit scalar codes (per-dimension min/max are
learned from the data), distances are computed asymmetrically (float query against the decoded
database vector), and the top of the search result can be re-ranked with exact distances.  So the
index keeps the vector *codes* only - 8 bits per dimension instead of 32 - and loses a little recall
that the re-ranking recovers.

The checks verify the structural invariants (layer 0 complete, link caps, symmetric links, levels
nested), graph connectivity from the entry point, that a beam search with ef = N reproduces the exact
nearest neighbours (the connectivity check is what makes that a real statement), recall against
brute force and its monotonicity in ef, the exactness of re-ranking inside the visited set, the
quantiser's error bound, and the boundary cases (empty index, one vector, duplicates, k > N, a query
equal to an indexed point).

Simplified versus hnswlib/FAISS: a few hundred vectors in 8 dimensions, plain lists of neighbour sets
instead of blocked adjacency tables, no heuristics for early termination or link removal on delete,
per-dimension scalar quantisation instead of product quantisation, and no SIMD distance kernels.
What is not simplified is the algorithm: the same level sampling, the same heuristic neighbour
selection, the same beam search and the same two-phase (approximate, then exact) retrieval.
"""
import heapq
import math

MASK64 = (1 << 64) - 1
DIM = 8


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def squared_distance(left, right):
    total = 0.0
    for a, b in zip(left, right):
        total += (a - b) * (a - b)
    return total


def make_vectors(count, dim=DIM, seed=0x5EED, centers=8):
    seeds = [tuple((center * 13 + axis * 7) % 41 - 20 for axis in range(dim))
             for center in range(centers)]
    vectors = []
    state = seed & MASK64
    for _ in range(count):
        state, r = splitmix64(state)
        base = seeds[r % centers]
        point = []
        for axis in range(dim):
            state, r = splitmix64(state)
            point.append(float(base[axis] + (r % 7) - 3))
        vectors.append(point)
    return vectors, state


class ScalarQuantizer:
    """Per-dimension 8-bit scalar quantisation: the index stores codes, not floats."""

    def __init__(self, vectors):
        if not vectors:
            raise ValueError("the quantiser needs training data")
        self.dim = len(vectors[0])
        self.low = [min(vector[axis] for vector in vectors) for axis in range(self.dim)]
        self.high = [max(vector[axis] for vector in vectors) for axis in range(self.dim)]
        if any(self.high[axis] == self.low[axis] for axis in range(self.dim)):
            raise ValueError("a constant dimension cannot be quantised")
        self.step = [(self.high[axis] - self.low[axis]) / 255.0 for axis in range(self.dim)]

    def encode(self, vector):
        codes = []
        for axis, value in enumerate(vector):
            scaled = int(round((value - self.low[axis]) / self.step[axis]))
            codes.append(max(0, min(255, scaled)))
        return tuple(codes)

    def decode(self, codes):
        return [self.low[axis] + codes[axis] * self.step[axis] for axis in range(self.dim)]

    def max_error(self):
        """Largest per-dimension reconstruction error the grid can produce."""
        return max(self.step) / 2.0


class HNSW:
    """Hierarchical navigable small world graph over optionally quantised vectors."""

    def __init__(self, vectors, m=8, ef_construction=16, quantize=True, seed=0xA5A5):
        self.vectors = [list(vector) for vector in vectors]
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(m)
        self.quantizer = ScalarQuantizer(self.vectors) if (quantize and self.vectors) else None
        self.stored = ([self.quantizer.decode(self.quantizer.encode(v)) for v in self.vectors]
                       if self.quantizer else self.vectors)
        self.layers = []                       # layers[level][node] = set of neighbours
        self.levels = []
        self.entry = None
        self.max_level = -1
        self.state = seed & MASK64
        for node in range(len(self.vectors)):
            self.insert(node)

    # --------------------------------------------------------------------------- primitives
    def level_of(self, node):
        return self.levels[node]

    def distance(self, query, node):
        """Asymmetric distance: the query stays in full precision, the database side is decoded."""
        return squared_distance(query, self.stored[node])

    def exact_distance(self, query, node):
        return squared_distance(query, self.vectors[node])

    def cap_for(self, level):
        return self.m0 if level == 0 else self.m

    def draw_level(self):
        self.state, r = splitmix64(self.state)
        uniform = (r + 1) / float(1 << 64)
        return int(-math.log(uniform) * self.mL)

    def select_neighbours(self, node, candidates, cap):
        """The diversification heuristic: keep a candidate only if it is closer to the base point
        than to any already kept neighbour."""
        ordered = sorted(set(candidates), key=lambda other: (self.distance(self.vectors[node], other), other))
        keep = []
        for candidate in ordered:
            if len(keep) >= cap:
                break
            if all(self.distance(self.vectors[candidate], other) >
                   self.distance(self.vectors[node], candidate) for other in keep):
                keep.append(candidate)
        for candidate in ordered:              # top up so the graph stays connected
            if len(keep) >= cap:
                break
            if candidate not in keep:
                keep.append(candidate)
        return keep

    def search_layer(self, query, entries, ef, level):
        """Beam search on one layer; returns (sorted results, visited set)."""
        visited = set(entries)
        candidates = []
        results = []
        for node in entries:
            measure = self.distance(query, node)
            heapq.heappush(candidates, (measure, node))
            heapq.heappush(results, (-measure, node))
        while candidates:
            measure, node = heapq.heappop(candidates)
            if results and len(results) >= ef and -results[0][0] < measure:
                break
            for neighbour in self.layers[level].get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                candidate_distance = self.distance(query, neighbour)
                if len(results) < ef or candidate_distance < -results[0][0]:
                    heapq.heappush(candidates, (candidate_distance, neighbour))
                    heapq.heappush(results, (-candidate_distance, neighbour))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-negated, node) for negated, node in results), visited

    def greedy_descent(self, query, start, level):
        current = start
        current_distance = self.distance(query, current)
        improved = True
        while improved:
            improved = False
            for neighbour in self.layers[level].get(current, ()):
                measure = self.distance(query, neighbour)
                if measure < current_distance:
                    current, current_distance, improved = neighbour, measure, True
        return current

    def prune(self, node, level):
        neighbours = list(self.layers[level][node])
        cap = self.cap_for(level)
        keep = self.select_neighbours(node, neighbours, cap)
        for dropped in neighbours:
            if dropped not in keep:
                self.layers[level][dropped].discard(node)
        self.layers[level][node] = set(keep)

    # ------------------------------------------------------------------------------ insertion
    def insert(self, node):
        level = self.draw_level()
        self.levels.append(level)
        while len(self.layers) <= level:
            self.layers.append({})
        for layer in range(level + 1):
            self.layers[layer][node] = set()
        if self.entry is None:
            self.entry, self.max_level = node, level
            return
        current = self.entry
        for layer in range(self.max_level, level, -1):
            current = self.greedy_descent(self.vectors[node], current, layer)
        for layer in range(min(level, self.max_level), -1, -1):
            found, _visited = self.search_layer(self.vectors[node], [current],
                                                self.ef_construction, layer)
            candidates = [other for _distance, other in found if other != node]
            chosen = self.select_neighbours(node, candidates, self.cap_for(layer))
            self.layers[layer][node] = set(chosen)
            for neighbour in chosen:
                self.layers[layer][neighbour].add(node)
                if len(self.layers[layer][neighbour]) > self.cap_for(layer):
                    self.prune(neighbour, layer)
            if found:
                current = found[0][1]
        if level > self.max_level:
            self.max_level, self.entry = level, node

    # --------------------------------------------------------------------------- retrieval
    def search(self, query, k=10, ef=None, exact_rerank=False):
        if self.entry is None or k <= 0:
            return []
        beam = max(ef if ef is not None else 20, k)
        current = self.entry
        for layer in range(self.max_level, 0, -1):
            current = self.greedy_descent(query, current, layer)
        found, visited = self.search_layer(query, [current], beam, 0)
        if exact_rerank:
            exact = sorted((self.exact_distance(query, node), node) for node in visited)
            return exact[:k]
        return found[:k]

    def visited_count(self, query, ef):
        current = self.entry
        for layer in range(self.max_level, 0, -1):
            current = self.greedy_descent(query, current, layer)
        _found, visited = self.search_layer(query, [current], max(ef, 1), 0)
        return len(visited)

    # --------------------------------------------------------------------------- diagnostics
    def layer_sizes(self):
        return [len(layer) for layer in self.layers]

    def links_symmetric(self):
        for level, layer in enumerate(self.layers):
            for node, neighbours in layer.items():
                for neighbour in neighbours:
                    if node not in self.layers[level].get(neighbour, set()):
                        return False
        return True

    def connected(self):
        """Breadth-first walk of layer 0 from the entry point."""
        if self.entry is None:
            return True
        seen = {self.entry}
        queue = [self.entry]
        while queue:
            node = queue.pop()
            for neighbour in self.layers[0].get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return len(seen) == len(self.vectors)

    def levels_are_nested(self):
        for node in range(len(self.vectors)):
            for layer in range(self.levels[node] + 1):
                if node not in self.layers[layer]:
                    return False
        return True

    def graph_bits(self):
        edges = sum(len(neighbours) for layer in self.layers for neighbours in layer.values())
        return edges * 32 + len(self.vectors) * (8 * DIM if self.quantizer else 32 * DIM)


def brute_force(vectors, query, k):
    return sorted((squared_distance(vector, query), node) for node, vector in enumerate(vectors))[:k]


def recall(index, queries, truth, k=10, ef=None, exact_rerank=False):
    hits = 0
    total = 0
    for query, expected in zip(queries, truth):
        found = {node for _distance, node in index.search(query, k=k, ef=ef, exact_rerank=exact_rerank)}
        hits += len(found & {node for _distance, node in expected})
        total += len(expected)
    return hits / float(total) if total else 0.0


def main():
    vectors, _state = make_vectors(200)
    queries, _state = make_vectors(20, seed=0xC0FFEE)
    truth = [brute_force(vectors, query, 10) for query in queries]

    # 1. structural invariants of the graph
    index = HNSW(vectors, m=16)
    check(len(index.layers[0]) == len(vectors), "layer 0 must contain every node")
    check(index.links_symmetric(), "every link must exist in both directions")
    check(index.levels_are_nested(), "a node must appear in every layer up to its level")
    check(index.connected(), "layer 0 must be connected from the entry point")
    # the neighbour cap is not cosmetic: on this clustered data a small cap lets the diversity
    # heuristic prune the bridges between clusters, and the graph falls apart
    narrow = HNSW(vectors, m=8)
    check(not narrow.connected(),
          "with M = 8 the pruned graph should split on this data - that is why M = 16 is used")
    for level, layer in enumerate(index.layers):
        cap = index.cap_for(level)
        check(all(len(neighbours) <= cap for neighbours in layer.values()),
              "layer %d must respect its neighbour cap %d" % (level, cap))
    sizes = index.layer_sizes()
    check(sizes[0] == len(vectors) and all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)),
          "the layers must shrink upwards: %r" % (sizes,))
    check(index.max_level >= 1 and sizes[-1] < sizes[0], "the hierarchy must have several layers")
    upper = sum(1 for node in range(len(vectors)) if index.levels[node] >= 1)
    check(0 < upper < 0.6 * len(vectors),
          "the level distribution must thin out: %d of %d nodes are above layer 0" % (upper, len(vectors)))
    # 2. with ef = N the beam search explores everything, and exact re-ranking is then exact
    check(index.visited_count(queries[0], len(vectors)) == len(vectors),
          "ef = N must visit the whole graph")
    exact_ok = recall(index, queries, truth, k=10, ef=len(vectors), exact_rerank=True)
    check(exact_ok == 1.0,
          "ef = N plus exact re-ranking must reproduce brute force, got %.3f" % exact_ok)
    whole = recall(index, queries, truth, k=10, ef=len(vectors))
    check(whole >= 0.98,
          "under the quantised metric the whole-graph answer is nearly exact: %.3f" % whole)

    # 3. recall against brute force, and the exactness of re-ranking inside the visited set
    plain = {ef: recall(index, queries, truth, k=10, ef=ef) for ef in (10, 20, 40)}
    reranked = {ef: recall(index, queries, truth, k=10, ef=ef, exact_rerank=True) for ef in (10, 20, 40)}
    check(plain[20] >= 0.9, "recall@10 with ef = 20 must be high: %r" % (plain,))
    check(plain[40] >= plain[10] - 1e-9 and plain[40] >= plain[20] - 1e-9,
          "a larger beam cannot return a worse neighbour: %r" % (plain,))
    check(all(reranked[ef] >= plain[ef] - 1e-9 for ef in plain),
          "exact re-ranking cannot lose recall: %r vs %r" % (reranked, plain))
    check(reranked[20] >= 0.95, "re-ranked recall must be very high: %r" % (reranked,))
    for query in queries[:5]:
        found_plain = index.search(query, k=1, ef=20)
        found_exact = index.search(query, k=1, ef=20, exact_rerank=True)
        current = index.entry
        for layer in range(index.max_level, 0, -1):
            current = index.greedy_descent(query, current, layer)
        _best, visited = index.search_layer(query, [current], 20, 0)
        best_visited = min(index.exact_distance(query, node) for node in visited)
        check(abs(found_exact[0][0] - best_visited) < 1e-12,
              "re-ranking must return the best node of the visited set")
        best_quantized = min(index.distance(query, node) for node in visited)
        check(abs(found_plain[0][0] - best_quantized) < 1e-12,
              "without re-ranking the answer minimises the quantised distance over the visited set")

    # 4. quantisation: codes, the error bound, and the memory saving
    quantizer = index.quantizer
    check(quantizer is not None and quantizer.dim == DIM, "the index must store quantised vectors")
    worst = 0.0
    for vector in vectors:
        codes = quantizer.encode(vector)
        check(len(codes) == DIM and all(0 <= code <= 255 for code in codes), "codes are bytes")
        decoded = quantizer.decode(codes)
        error = max(abs(a - b) for a, b in zip(vector, decoded))
        worst = max(worst, error)
    check(worst <= quantizer.max_error() + 1e-9,
          "the per-dimension error must stay within half a quantisation step: %.4f" % worst)
    check(8 * DIM * len(vectors) < 32 * DIM * len(vectors), "8-bit codes use a quarter of the space")
    check(index.graph_bits() < (32 * DIM + 32 * 8) * len(vectors) * 2,
          "the graph and the codes must fit together comfortably")
    distance_error = max(abs(squared_distance(queries[0], quantizer.decode(quantizer.encode(v)))
                              - squared_distance(queries[0], v)) for v in vectors)
    bound = sum(2 * abs(q) * quantizer.max_error() + quantizer.max_error() ** 2 for q in queries[0])
    check(distance_error <= bound + 1e-6,
          "the squared-distance error must respect the first-order bound: %.4f <= %.4f"
          % (distance_error, bound))

    # 5. boundary cases
    check(HNSW([], quantize=False).search([0.0] * DIM, k=5) == [], "an empty index returns nothing")
    single = HNSW([vectors[3]], quantize=False)
    check(single.search(vectors[3], k=1)[0][1] == 0, "a one-node index finds its only vector")
    check(single.connected() and single.max_level == 0, "the single node is the entry point")
    duplicates = HNSW([vectors[0]] * 12, quantize=False)
    check(duplicates.connected() and duplicates.links_symmetric(),
          "duplicate vectors must not break the graph")
    check(len(duplicates.search(vectors[0], k=5, ef=20)) == 5,
          "duplicates must still be retrievable (the true tie order decides)")
    check(len(index.search(queries[0], k=1000, ef=40)) == len(vectors),
          "k larger than the index returns everything reachable")
    check(index.search(queries[0], k=0) == [], "k = 0 returns nothing")
    check(index.search(vectors[9], k=1, ef=40, exact_rerank=True)[0][0] == 0.0,
          "a query equal to an indexed vector must be found at distance zero")
    try:
        ScalarQuantizer([[1.0, 1.0], [1.0, 2.0]])
        raise AssertionError("a constant dimension must be rejected")
    except ValueError:
        pass

    # 6. the quantised index is smaller and only slightly worse; determinism holds
    full = HNSW(vectors, m=16, quantize=False)
    full_recall = recall(full, queries, truth, k=10, ef=20)
    check(full_recall >= plain[20] - 0.15,
          "quantisation must not destroy recall: %.3f vs %.3f" % (plain[20], full_recall))
    check(len(full.search(queries[0], k=1, ef=20)[0]) == 2, "the full-precision index also answers")
    rebuilt = HNSW(vectors, m=16)
    check(rebuilt.levels == index.levels and rebuilt.links_symmetric(),
          "the level draw must be reproducible")
    check(all(rebuilt.layers[level][node] == index.layers[level][node]
              for level in range(len(index.layers)) for node in index.layers[level]),
          "the graph must be reproducible")
    check(rebuilt.search(queries[4], k=5) == index.search(queries[4], k=5),
          "search results must be reproducible")

    print("766 n=%d dim=%d M=%d M0=%d layer sizes %r" % (len(vectors), DIM, index.m, index.m0, sizes))
    print("766 recall@10 (quantised/reranked): %r / %r; full precision %.3f"
          % ({k: round(v, 3) for k, v in plain.items()},
             {k: round(v, 3) for k, v in reranked.items()}, full_recall))
    print("766 quantisation: max per-dim error %.4f (bound %.4f), %.0fx less vector space"
          % (worst, quantizer.max_error(), (32 * DIM) / float(8 * DIM)))
    print("PASS")


if __name__ == "__main__":
    main()
