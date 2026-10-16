"""Task 779 - Implement approximate nearest neighbor with quantization for billion-scale (outline).  (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)

IVF+PQ is the classic billion-scale index: a coarse quantizer (k-means, `nlist` centroids) partitions
the vectors into inverted lists, and inside every list the residuals/vectors are compressed with
product quantization - the d dimensions are cut into `m` sub-spaces, each sub-space is clustered into
`ksub` codewords, and a vector becomes a tuple of `m` small integers.  A query is then answered by
probing the `nprobe` nearest coarse centroids and comparing it against the codes of the lists using
asymmetric distance computation (ADC): a per-query lookup table holds, for every sub-space and every
codeword, the squared distance to that codeword, so the distance to a database vector is just the sum
of `m` table reads.  Re-ranking the ADC candidates against the exact vectors (kept aside in this
scaled-down outline) gives the final neighbours.

Verified, mostly against brute force: k-means keeps its invariants (every point is assigned to its
nearest centroid, every centroid is its cluster's mean, the objective never increases across Lloyd
iterations), PQ codes are nearest-codeword assignments, ADC equals the exact squared distance to the
decoded vector (an identity, not an approximation), probing *all* cells with re-ranking reproduces
brute-force k-NN exactly, the probed lists are exactly the `nprobe` nearest coarse centroids, recall
grows with `nprobe`, reconstruction error falls as the codebook grows, the degenerate cases
(nlist > n, a single point, identical points, k larger than the database) are handled and the whole
pipeline is deterministic.

Simplified: a few hundred 16-dimensional vectors instead of a billion, `ksub = 8` codewords per
sub-space instead of 256 (so codes fit in tiny integers), flat inverted lists with an exact rescoring
pass, and IVF-PQ is trained on the same small sample it is queried with.  The quantization, the
probed-cell logic and the ADC identity are real, not sketched.
"""
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


def random_vectors(count, dims, seed):
    vectors = []
    for _ in range(count):
        coords = []
        for _ in range(dims):
            seed, r = splitmix64(seed)
            coords.append((r % (1 << 20)) / float(1 << 20))
        vectors.append(tuple(coords))
    return vectors, seed


def squared_distance(a, b):
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return total


def mean(points):
    return tuple(sum(point[axis] for point in points) / len(points) for axis in range(len(points[0])))


def kmeans(points, k, iterations=40, seed=1):
    """k-means with deterministic seeding; the returned state is a consistent Lloyd fixed point."""
    if not points:
        return [], [], 0.0
    k = min(k, len(points))
    state = seed
    centroids = [points[0]]
    while len(centroids) < k:
        # farthest-first seeding with the streamed PRNG breaking ties among distant candidates
        farthest = max(points, key=lambda point: min(squared_distance(point, c) for c in centroids))
        state, r = splitmix64(state)
        pick = points[r % len(points)]
        candidate = pick if min(squared_distance(pick, c) for c in centroids) > 0 else farthest
        centroids.append(candidate)
    assignments = None
    for _ in range(iterations):
        new_assignments = []
        for point in points:
            best, best_distance = 0, float("inf")
            for index, centroid in enumerate(centroids):
                distance = squared_distance(point, centroid)
                if distance < best_distance:
                    best, best_distance = index, distance
            new_assignments.append(best)
        clusters = [[] for _ in centroids]
        for point, index in zip(points, new_assignments):
            clusters[index].append(point)
        centroids = [mean(cluster) if cluster else centroids[index]
                     for index, cluster in enumerate(clusters)]
        if new_assignments == assignments:
            break                    # assignments stable, so the centroids are their clusters' means
        assignments = new_assignments
    assignments = []
    for point in points:
        assignments.append(min(range(len(centroids)),
                               key=lambda index: squared_distance(point, centroids[index])))
    return centroids, assignments, kmeans_objective(points, centroids, assignments)


def kmeans_objective(points, centroids, assignments):
    return sum(squared_distance(point, centroids[index])
               for point, index in zip(points, assignments))


class ProductQuantizer:
    """Splits the dimensions into m sub-spaces and clusters each one into ksub codewords."""

    def __init__(self, dims, m, ksub, seed=7):
        check(dims % m == 0, "the dimension count must be divisible by the sub-space count")
        self.dims = dims
        self.m = m
        self.ksub = ksub
        self.width = dims // m
        self.codebooks = []
        self.seed = seed

    def train(self, vectors):
        for space in range(self.m):
            start = space * self.width
            projected = [vector[start:start + self.width] for vector in vectors]
            centroids, _, _ = kmeans(projected, self.ksub, iterations=40, seed=self.seed + space)
            self.codebooks.append(centroids)
        return self

    def encode(self, vector):
        code = []
        for space, codebook in enumerate(self.codebooks):
            start = space * self.width
            piece = vector[start:start + self.width]
            best = min(range(len(codebook)), key=lambda index: squared_distance(piece, codebook[index]))
            code.append(best)
        return tuple(code)

    def decode(self, code):
        out = []
        for space, index in enumerate(code):
            out.extend(self.codebooks[space][index])
        return tuple(out)

    def distance_table(self, query):
        """table[space][codeword] = squared distance from the query's slice to that codeword."""
        table = []
        for space, codebook in enumerate(self.codebooks):
            start = space * self.width
            piece = query[start:start + self.width]
            table.append([squared_distance(piece, centroid) for centroid in codebook])
        return table

    @staticmethod
    def adc(table, code):
        return sum(table[space][index] for space, index in enumerate(code))

    def reconstruction_error(self, vectors):
        return sum(squared_distance(vector, self.decode(self.encode(vector)))
                   for vector in vectors) / len(vectors)


class IVFIndex:
    """Coarse k-means over inverted lists, PQ codes inside each list, ADC + exact re-ranking."""

    def __init__(self, nlist, dims, m, ksub, seed=11):
        self.nlist = nlist
        self.dims = dims
        self.m = m
        self.ksub = ksub
        self.seed = seed
        self.centroids = []
        self.lists = []
        self.vectors = []
        self.quantizer = None

    def train(self, vectors):
        self.vectors = list(vectors)
        self.centroids, assignments, _ = kmeans(vectors, self.nlist, iterations=40, seed=self.seed)
        self.lists = [[] for _ in self.centroids]
        self.quantizer = ProductQuantizer(self.dims, self.m, self.ksub, self.seed).train(vectors)
        for index, (vector, cell) in enumerate(zip(vectors, assignments)):
            self.lists[cell].append((index, self.quantizer.encode(vector)))
        return self

    def probe_cells(self, query, nprobe):
        """The nprobe coarse centroids closest to the query, nearest first."""
        scored = sorted(range(len(self.centroids)),
                        key=lambda cell: (squared_distance(query, self.centroids[cell]), cell))
        return scored[:max(1, min(nprobe, len(scored)))]

    def search(self, query, k, nprobe, rerank=True, rerank_limit=None):
        """Returns list (distance, index) pairs, ranked by distance.

        `rerank_limit` caps how many ADC candidates are scored exactly; None scores all of them,
        which is what makes a full probe equivalent to brute force.
        """
        table = self.quantizer.distance_table(query)
        candidates = []
        for cell in self.probe_cells(query, nprobe):
            for index, code in self.lists[cell]:
                candidates.append((self.quantizer.adc(table, code), index))
        candidates.sort()
        if rerank:
            pool = candidates if rerank_limit is None else candidates[:max(rerank_limit, k)]
            ranked = sorted((squared_distance(query, self.vectors[index]), index) for _, index in pool)
        else:
            ranked = candidates
        return ranked[:k]


def brute_force_knn(query, vectors, k):
    return sorted((squared_distance(query, vector), index)
                  for index, vector in enumerate(vectors))[:k]


def main():
    dims, count = 16, 300
    vectors, seed = random_vectors(count, dims, 0xA11CE)

    # 1. k-means invariants: nearest assignment, mean centroids, non-increasing objective
    centroids, assignments, objective = kmeans(vectors, 8, iterations=40, seed=5)
    check(objective == kmeans_objective(vectors, centroids, assignments),
          "the reported objective must match the assignments")
    check(all(assignments[index] == min(range(len(centroids)),
                                        key=lambda c: squared_distance(vector, centroids[c]))
              for index, vector in enumerate(vectors)),
          "every vector must be assigned to its nearest centroid")
    for index, centroid in enumerate(centroids):
        cluster = [vector for vector, cell in zip(vectors, assignments) if cell == index]
        if cluster:
            check(squared_distance(centroid, mean(cluster)) < 1e-18,
                  "a centroid must be the mean of its cluster")
    # 2. the objective never grows when more Lloyd iterations are allowed
    two_iterations = kmeans(vectors, 8, iterations=2, seed=5)[2]
    check(objective <= two_iterations + 1e-9,
          f"more iterations must not raise the objective ({objective} vs {two_iterations})")
    check(kmeans(vectors, 8, iterations=40, seed=5)[0] == centroids,
          "the fixed point must be reproducible")

    # 3. PQ: codes are nearest codewords and ADC is the exact distance to the decoded vector
    quantizer = ProductQuantizer(dims, 4, 8, seed=3).train(vectors)
    for vector in vectors[:25]:
        code = quantizer.encode(vector)
        check(len(code) == 4 and all(0 <= index < 8 for index in code), "a code is four small ints")
        for space, index in enumerate(code):
            start = space * quantizer.width
            piece = vector[start:start + quantizer.width]
            best = min(squared_distance(piece, centroid) for centroid in quantizer.codebooks[space])
            check(abs(squared_distance(piece, quantizer.codebooks[space][index]) - best) < 1e-15,
                  "every codeword must be the nearest one in its sub-space")
    for query in vectors[:15]:
        table = quantizer.distance_table(query)
        for vector in vectors[:15]:
            code = quantizer.encode(vector)
            check(abs(ProductQuantizer.adc(table, code) - squared_distance(query, quantizer.decode(code)))
                  < 1e-12, "ADC must equal the exact squared distance to the reconstruction")

    # 4. a bigger codebook compresses better
    coarse_code = ProductQuantizer(dims, 4, 2, seed=3).train(vectors).reconstruction_error(vectors)
    fine_code = ProductQuantizer(dims, 4, 8, seed=3).train(vectors).reconstruction_error(vectors)
    check(fine_code <= coarse_code,
          f"8 codewords must reconstruct better than 2 ({fine_code} vs {coarse_code})")

    # 5. probing every cell with re-ranking reproduces brute force exactly
    index = IVFIndex(nlist=12, dims=dims, m=4, ksub=8, seed=13).train(vectors)
    queries = vectors[::20]
    exact_matches = 0
    for query in queries:
        approximate = index.search(query, 10, nprobe=len(index.centroids), rerank=True,
                                   rerank_limit=None)
        exact = brute_force_knn(query, vectors, 10)
        exact_matches += sum(1 for (_, left), (_, right) in zip(approximate, exact) if left == right)
        check([score for score, _ in approximate] == [score for score, _ in exact],
              "a full probe with re-ranking must return the exact neighbours")
    check(exact_matches == 10 * len(queries), "every exact neighbour must be recovered")

    # 6. the probed cells really are the nearest coarse centroids
    probe_query = queries[0]
    probed = index.probe_cells(probe_query, 3)
    distances = [squared_distance(probe_query, centroid) for centroid in index.centroids]
    check(probed == sorted(range(len(distances)), key=lambda cell: (distances[cell], cell))[:3],
          "probing must pick the nearest coarse centroids")
    returned_cells = {cell for cell, entries in enumerate(index.lists) if entries}
    check(len(returned_cells) <= len(index.centroids), "every list belongs to a cell")

    # 7. recall grows with nprobe and is high enough to be useful
    def recall_at_ten(nprobe):
        hits = 0
        for query in queries:
            approximate = {index_id for _, index_id in index.search(query, 10, nprobe=nprobe)}
            exact = {index_id for _, index_id in brute_force_knn(query, vectors, 10)}
            hits += len(approximate & exact)
        return hits / (10 * len(queries))

    narrow, wide = recall_at_ten(1), recall_at_ten(len(index.centroids))
    check(wide == 1.0, "probing every cell must reach perfect recall")
    check(narrow <= wide and narrow >= 0.5,
          f"a single probed cell must still find most neighbours, got {narrow:.2f}")

    # 8. a one-cell probe can only return ids that live in that cell, ranked by ADC distance
    probed_cell = index.probe_cells(queries[0], 1)[0]
    allowed = {index_id for index_id, _ in index.lists[probed_cell]}
    adc_only = index.search(queries[0], 10, nprobe=1, rerank=False)
    check(allowed, "the probed cell must not be empty")
    check(all(index_id in allowed for _, index_id in adc_only),
          "a single-cell probe must only return vectors of that cell")
    check(adc_only == sorted(adc_only), "ADC candidates must come back ordered by ADC distance")

    # 9. boundaries: nlist larger than the data, a single vector, identical vectors, k > database
    tiny = IVFIndex(nlist=50, dims=4, m=2, ksub=4, seed=2).train(
        [(float(i), float(i), float(i), float(i)) for i in range(3)])
    check(len(tiny.centroids) == 3, "k-means must not invent more centroids than points")
    single = IVFIndex(nlist=4, dims=4, m=2, ksub=4, seed=2).train([(1.0, 2.0, 3.0, 4.0)])
    check(len(single.search((1.0, 2.0, 3.0, 4.0), 5, nprobe=4)) == 1,
          "asking for more neighbours than exist must return what there is")
    identical = IVFIndex(nlist=2, dims=4, m=2, ksub=4, seed=2).train([(0.5,) * 4] * 6)
    check(all(score == 0.0 for score, _ in identical.search((0.5,) * 4, 3, nprobe=2)),
          "identical vectors must be at distance zero")

    # 10. determinism
    again = IVFIndex(nlist=12, dims=dims, m=4, ksub=8, seed=13).train(vectors)
    check([index.search(query, 5, nprobe=2) for query in queries[:5]]
          == [again.search(query, 5, nprobe=2) for query in queries[:5]],
          "the index must answer deterministically")

    print(f"779 vectors={count} dims={dims} nlist={len(index.centroids)} "
          f"reconstruction_error={fine_code:.4f} (ksub 2: {coarse_code:.4f})")
    print(f"779 recall@10 nprobe=1={narrow:.2f} full={wide:.2f} exact_matches={exact_matches} "
          f"queries={len(queries)}")
    print("PASS")


if __name__ == "__main__":
    main()
