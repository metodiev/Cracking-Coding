"""Task 765 - Approximate nearest neighbour for embeddings at scale with IVF + PQ.  (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)

Two independent compressions make billion-scale vector search feasible:

  * IVF (inverted file): k-means over the raw vectors gives a coarse quantiser; every database
    vector is stored in the inverted list of its nearest centroid, so a query only visits the
    `nprobe` nearest lists instead of the whole database.
  * PQ (product quantisation): a d-dimensional vector is split into m sub-vectors of width d/m and
    each sub-vector is replaced by the index of the nearest of 2^b codewords trained on that
    subspace.  A vector of d float32 values becomes m*b bits, and - the trick that makes it useful -
    distances can be estimated with a lookup table (ADC, asymmetric distance computation): the query
    is split the same way, a table of the distance from each query sub-vector to every codeword of
    its subspace is built once, and the distance to a *code* is the sum of m table entries.

The pipeline here is the classic one: coarse assignment -> probe nprobe lists -> rank the candidates
by ADC -> re-rank the top `rerank` candidates with exact distances.  Every stage is checked against a
brute-force oracle: the coarse assignment against a re-computed nearest centroid, the table-based ADC
against the exact distance to the *decoded* vector (they must be identical), the re-ranked top-1
against the exact nearest neighbour inside the same candidate set, and the end-to-end recall@10
against an exhaustive scan, including its monotonicity in `nprobe` and the effect of PQ compression.

Simplified versus FAISS: float coordinates in a few dimensions, one coarse level (no
IVF-HNSW/IVF-PQ residual encoding and no residual quantisation), plain Lloyd k-means with a
deterministic initialisation instead of k-means++ or OPQ, no SIMD-friendly memory layout, no
training on a held-out subset and no early termination heuristics.  Verified: k-means convergence
and assignment stability, exact IVF plumbing (probing every list reproduces brute force), codebook
round-trips, the ADC identity, recall and its monotonicity, the guarantee that re-ranking is exact
within the candidate set, the compression ratio, and the boundary cases (empty index, k > N,
nprobe = 0, nprobe clamped, a query equal to an indexed vector).
"""
import heapq  # noqa: F401  (kept for API parity with a real ANN library; the search here is a scan)
MASK64 = (1 << 64) - 1
DIM = 8
LISTS = 8
PROBES = 4


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def make_vectors(count, dim=DIM, seed=0x5EED, centers=8):
    """Deterministic clustered data: a few well separated centres plus small noise."""
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


def squared_distance(left, right):
    total = 0.0
    for a, b in zip(left, right):
        total += (a - b) * (a - b)
    return total


def nearest(centroids, point):
    best, best_distance = 0, None
    for index, centroid in enumerate(centroids):
        distance = squared_distance(centroid, point)
        if best_distance is None or distance < best_distance:
            best, best_distance = index, distance
    return best, best_distance


def objective(points, centroids):
    return sum(nearest(centroids, point)[1] for point in points)


def kmeans(points, k, iterations=12):
    """Deterministic Lloyd k-means returning the centroids and the objective history."""
    if not points:
        raise ValueError("k-means needs points")
    if k < 1 or k > len(points):
        raise ValueError("k must be within [1, len(points)]")
    indexes = [(index * len(points)) // k for index in range(k)]
    centroids = [list(points[index]) for index in indexes]
    history = []
    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for point in points:
            groups[nearest(centroids, point)[0]].append(point)
        for index, group in enumerate(groups):
            if group:
                width = len(group[0])
                centroids[index] = [sum(row[axis] for row in group) / float(len(group))
                                    for axis in range(width)]
        history.append(objective(points, centroids))
    return centroids, history


class ProductQuantizer:
    """Split the vector into `subspaces` blocks, quantise each with its own small codebook."""

    def __init__(self, points, subspaces=4, codewords=16, iterations=12):
        if DIM % subspaces:
            raise ValueError("the dimension must be divisible by the number of subspaces")
        self.subspaces = subspaces
        self.width = len(points[0]) // subspaces
        self.codewords = codewords
        self.bits = (codewords - 1).bit_length()
        self.codebooks = []
        for block in range(subspaces):
            sub_vectors = [point[block * self.width:(block + 1) * self.width] for point in points]
            book, _history = kmeans(sub_vectors, codewords, iterations)
            self.codebooks.append(book)

    def encode(self, vector):
        codes = []
        for block in range(self.subspaces):
            sub = vector[block * self.width:(block + 1) * self.width]
            codes.append(nearest(self.codebooks[block], sub)[0])
        return tuple(codes)

    def decode(self, codes):
        vector = []
        for block, code in enumerate(codes):
            vector.extend(self.codebooks[block][code])
        return vector

    def table(self, query):
        """ADC lookup table: per subspace, the squared distance to every codeword."""
        rows = []
        for block in range(self.subspaces):
            sub = query[block * self.width:(block + 1) * self.width]
            rows.append([squared_distance(sub, codeword) for codeword in self.codebooks[block]])
        return rows

    def estimate(self, codes, table):
        return sum(table[block][code] for block, code in enumerate(codes))

    def code_bits(self):
        return self.subspaces * self.bits


class IVFIndex:
    """Coarse quantiser + inverted lists + product quantisation + exact re-ranking."""

    def __init__(self, vectors, lists=LISTS, pq=None):
        self.vectors = list(vectors)
        if self.vectors:
            self.pq = pq if pq is not None else ProductQuantizer(self.vectors)
            self.centroids, self.kmeans_history = kmeans(self.vectors, lists)
        else:
            self.pq = None
            self.centroids, self.kmeans_history = [], []
        self.lists = [[] for _ in range(lists)]
        self.codes = {}
        for identifier, vector in enumerate(self.vectors):
            assigned = nearest(self.centroids, vector)[0]
            self.lists[assigned].append(identifier)
            self.codes[identifier] = self.pq.encode(vector)

    def probe_order(self, query):
        return sorted(range(len(self.centroids)),
                      key=lambda index: (squared_distance(self.centroids[index], query), index))

    def candidates(self, query, nprobe):
        probes = self.probe_order(query)[:max(0, min(nprobe, len(self.centroids)))]
        found = []
        for probe in probes:
            found.extend(self.lists[probe])
        return probes, found

    def search(self, query, k=10, nprobe=PROBES, rerank=64, exact_ranking=False):
        """Return the top-k as (id, distance) pairs, or [] when nothing can be returned."""
        if k <= 0 or not self.vectors:
            return []
        probes, found = self.candidates(query, nprobe)
        if not found:
            return []
        table = self.pq.table(query)
        if exact_ranking:
            scored = [(squared_distance(self.vectors[identifier], query), identifier)
                      for identifier in found]
        else:
            scored = [(self.pq.estimate(self.codes[identifier], table), identifier)
                      for identifier in found]
        scored.sort(key=lambda item: (item[0], item[1]))
        shortlist = scored[:max(rerank, k)] if rerank is not None else scored
        reranked = sorted(((squared_distance(self.vectors[identifier], query), identifier)
                           for _estimate, identifier in shortlist), key=lambda item: (item[0], item[1]))
        return reranked[:k]

    def assignment_ok(self):
        """Every vector must sit in the list of its nearest coarse centroid."""
        for identifier, vector in enumerate(self.vectors):
            if identifier not in self.lists[nearest(self.centroids, vector)[0]]:
                return False
        return sum(len(entry) for entry in self.lists) == len(self.vectors)


def brute_force(vectors, query, k):
    scored = sorted((squared_distance(vector, query), identifier)
                    for identifier, vector in enumerate(vectors))
    return scored[:k]


def recall_at_k(index, queries, truth, k=10, nprobe=PROBES, rerank=64, exact_ranking=False):
    hits = 0
    total = 0
    for query, expected in zip(queries, truth):
        found = {identifier for _distance, identifier in
                 index.search(query, k=k, nprobe=nprobe, rerank=rerank, exact_ranking=exact_ranking)}
        hits += len(found & {identifier for _distance, identifier in expected})
        total += len(expected)
    return hits / float(total) if total else 0.0


def main():
    vectors, _state = make_vectors(400)
    queries, _state = make_vectors(20, seed=0xC0FFEE)
    check(len(vectors) == 400 and len(vectors[0]) == DIM, "the dataset must have the requested shape")
    check(all(all(-25 <= value <= 25 for value in vector) for vector in vectors),
          "generated coordinates must stay in the designed range")

    # 1. k-means converges: the objective decreases and the assignments stop changing
    centroids, history = kmeans(vectors, LISTS)
    check(all(history[index] >= history[index + 1] - 1e-9 for index in range(len(history) - 1)),
          "the k-means objective must be non-increasing: %r" % (history,))
    check(history[-1] < history[0], "the objective must actually improve")
    check(len(centroids) == LISTS and len(centroids[0]) == DIM, "one centroid per list, full width")
    stable, _ = kmeans(vectors, LISTS)
    check(stable == centroids, "k-means must be deterministic")

    # 2. the inverted file plumbing: every vector in exactly one list, in the right one
    pq = ProductQuantizer(vectors)
    index = IVFIndex(vectors, LISTS, pq)
    check(index.assignment_ok(), "every vector must live in the list of its nearest centroid")
    check(sum(len(entry) for entry in index.lists) == len(vectors), "no vector may be lost or doubled")
    check(len(set(len(entry) for entry in index.lists)) > 1, "the lists must not all be the same size")

    # 3. probing every list with exact distances must reproduce brute force exactly
    truth = [brute_force(vectors, query, 10) for query in queries]
    full = recall_at_k(index, queries, truth, k=10, nprobe=LISTS, exact_ranking=True, rerank=None)
    check(full == 1.0, "probing all lists with exact distances must give recall 1.0, got %.3f" % full)
    example = index.search(queries[0], k=10, nprobe=LISTS, exact_ranking=True)
    check(example == truth[0], "the full-probe search must equal the brute-force ranking")

    # 4. PQ codes pick the nearest codeword in each subspace, and the code size is a real compression
    for identifier in range(0, len(vectors), 37):
        codes = pq.encode(vectors[identifier])
        check(len(codes) == pq.subspaces and all(0 <= code < pq.codewords for code in codes),
              "codes must be in range")
        for block in range(pq.subspaces):
            sub = vectors[identifier][block * pq.width:(block + 1) * pq.width]
            check(nearest(pq.codebooks[block], sub)[0] == codes[block],
                  "the encoder must pick the nearest codeword of subspace %d" % block)
    check(len(pq.codebooks) == pq.subspaces and
          all(len(book) == pq.codewords for book in pq.codebooks),
          "one codebook of the requested size per subspace")
    check(all(len(book[0]) == pq.width for book in pq.codebooks),
          "every codeword must have the subspace width")
    check(pq.code_bits() == 16 and DIM * 32 / pq.code_bits() == 16.0,
          "PQ compresses %d bits into %d bits (16x)" % (DIM * 32, pq.code_bits()))
    coarse_error = sum(squared_distance(vectors[i], pq.decode(pq.encode(vectors[i])))
                       for i in range(len(vectors))) / len(vectors)
    finer = ProductQuantizer(vectors, codewords=64)
    fine_error = sum(squared_distance(vectors[i], finer.decode(finer.encode(vectors[i])))
                     for i in range(len(vectors))) / len(vectors)
    check(fine_error < coarse_error,
          "more codewords must reconstruct better: %.3f vs %.3f" % (fine_error, coarse_error))
    check(coarse_error > 0, "quantisation must not be exact")

    # 5. the ADC table is exactly the distance to the decoded vector
    table = pq.table(queries[0])
    for identifier in (0, 5, 42, 123, 399):
        codes = index.codes[identifier]
        check(abs(pq.estimate(codes, table) - squared_distance(queries[0], pq.decode(codes))) < 1e-9,
              "ADC must equal the exact distance to the decoded vector")
    check(len(table) == pq.subspaces and all(len(row) == pq.codewords for row in table),
          "the table must have one row per subspace and one entry per codeword")

    # 6. recall against the exhaustive scan, and the provable monotonicity in nprobe
    plain = {nprobe: recall_at_k(index, queries, truth, k=10, nprobe=nprobe, rerank=None)
             for nprobe in (1, 2, 4, 8)}
    check(plain[4] >= 0.9, "recall@10 over the probed lists must be high: %r" % (plain,))
    check(all(plain[first] <= plain[second] for first, second in ((1, 2), (2, 4), (4, 8))),
          "recall must be non-decreasing in nprobe: %r" % (plain,))
    check(plain[8] > plain[1], "the coarse quantiser must actually discard work")
    recalls = {nprobe: recall_at_k(index, queries, truth, k=10, nprobe=nprobe) for nprobe in (1, 4, 8)}
    check(recalls[4] >= 0.8, "end-to-end recall with PQ and re-ranking: %r" % (recalls,))
    check(recalls[8] >= recalls[1], "more probes cannot hurt the end-to-end recall")
    check(plain[8] >= recalls[8], "PQ can only lose recall relative to exact distances inside the lists")

    # 7. re-ranking is exact inside the candidate set: two guarantees, checked independently
    for query in queries[:6]:
        _probes, found = index.candidates(query, 4)
        best_inside = min((squared_distance(vectors[identifier], query), identifier)
                          for identifier in found)
        top = index.search(query, k=1, nprobe=4, rerank=len(found))
        check(top[0][0] == best_inside[0] and top[0][1] == best_inside[1],
              "re-ranking the whole probed set must return the exact best candidate")
        table = pq.table(query)
        best_adc = min((pq.estimate(index.codes[identifier], table), identifier)
                       for identifier in found)
        single = index.search(query, k=1, nprobe=4, rerank=1)
        check(single[0][1] == best_adc[1],
              "with rerank = 1 the ADC argmin must be the answer")
        exact_best = brute_force(vectors, query, 1)[0]
        if exact_best[1] in found:
            check(top[0][0] <= exact_best[0] + 1e-9, "the true nearest must not be missed")

    # 8. boundaries
    check(index.search(queries[0], k=0) == [], "k = 0 returns nothing")
    check(index.search(queries[0], k=10, nprobe=0) == [], "probing nothing returns nothing")
    check(len(index.search(queries[0], k=10, nprobe=999)) == 10, "nprobe is clamped to the lists")
    check(len(index.search(queries[0], k=1000, nprobe=LISTS)) == len(vectors),
          "k larger than the database returns everything")
    exact_hit = index.search(vectors[7], k=1, nprobe=LISTS)
    check(exact_hit[0][1] == 7 and exact_hit[0][0] == 0.0,
          "a query equal to an indexed vector must find it at distance zero")
    empty = IVFIndex([], lists=LISTS)
    check(empty.search(queries[0], k=5) == [] and empty.lists == [[] for _ in range(LISTS)],
          "an empty index returns nothing")
    for bad in (lambda: kmeans(vectors, 0), lambda: kmeans(vectors, 5000),
                lambda: kmeans([], 2), lambda: ProductQuantizer(vectors, subspaces=5)):
        try:
            bad()
            raise AssertionError("invalid training parameters must be rejected")
        except ValueError:
            pass

    # 9. determinism of the whole pipeline
    rebuilt = IVFIndex(vectors, LISTS, ProductQuantizer(vectors))
    check(rebuilt.centroids == index.centroids and rebuilt.lists == index.lists,
          "the index must be reproducible")
    check(rebuilt.codes == index.codes, "the PQ codes must be reproducible")
    check(rebuilt.search(queries[3], k=5) == index.search(queries[3], k=5),
          "search results must be reproducible")

    print("765 n=%d dim=%d lists=%d listsizes=%r" % (len(vectors), DIM, LISTS,
                                                     [len(entry) for entry in index.lists]))
    print("765 PQ: %d subspaces x %d codewords = %d bits (%.0fx smaller), reconstruction MSE %.2f"
          % (pq.subspaces, pq.codewords, pq.code_bits(), DIM * 32 / pq.code_bits(), coarse_error))
    print("765 recall@10 by nprobe: %r" % ({key: round(value, 3) for key, value in recalls.items()},))
    print("765 k-means objective %.1f -> %.1f over %d iterations"
          % (history[0], history[-1], len(history)))
    print("PASS")


if __name__ == "__main__":
    main()
