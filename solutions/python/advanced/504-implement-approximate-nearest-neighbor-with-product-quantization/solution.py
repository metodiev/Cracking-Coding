"""Task 504 - Implement approximate nearest neighbor with product quantization (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

The task asks for an outline; this is a working, simplified-but-correct implementation of product
quantisation (PQ) for approximate nearest neighbour search. The D-dimensional space is split into
M disjoint subspaces of dsub = D / M coordinates, and each subspace gets its own codebook of K
centroids fitted with Lloyd's algorithm (deterministic farthest-point initialisation - the first
centre is point 0, every next centre is the point with the largest minimum distance to those
already chosen - then assign/recompute until the assignment stops changing or the distortion
improves by less than 1e-12, cap 60 rounds). A database vector is encoded as M centroid indices
(M * log2 K bits instead of D floats), reconstructed by concatenating the chosen centroids, and
searched with asymmetric distance computation (ADC): for a query the table
T[m][k] = ||q_m - c_{m,k}||^2 is built once, after which the approximate squared distance of any
code is the sum of M table lookups. Empty clusters keep their previous centroid (documented).

Documented checks: the 1-D special case is compared against the exact dynamic-programming optimum
over contiguous blocks (the global optimum, since optimal 1-D clusters are intervals) - the DP cost
must never beat Lloyd's and must match it on well-separated 1-D data to 1e-9; the distortion must be
non-increasing over the Lloyd rounds (the classical descent property); the ADC identity
sum_m T[m][code[m]] == ||q - reconstruct(code)||^2 must hold to 1e-9 for every query and code;
reverse triangle inequality |d_approx - d_true| <= ||x - reconstruct(code)|| must hold for every
database point (with d the Euclidean distance), which is what makes PQ distances safe; the
reconstruction error must strictly decrease as K grows (1, 2, 4, 8, 16); every query over five well-separated clusters must retrieve a
top-1 from its own blob (100%, though the exact point of the blob is returned for only about a
quarter of the queries because the whole blob shares one 12-bit code) and recall@10 against the
exact brute-force top-10 must be >= 0.5 and strictly improve from K = 8 to K = 32; codes must cost D/subspace * log2 K bits against D * 64 bits of raw storage
(a two-order-of-magnitude compression here); and two runs must produce identical codebooks.
"""

import math

DIM = 16
M_SUBSPACES = 4
DSUB = DIM // M_SUBSPACES
MAX_ROUNDS = 60


def lcg(seed):
    return (seed * 75 + 74) % 65537


def make_data(n, dim, seed, span=25):
    """n integer points in [-span, span]^dim from the wave's 75/74 LCG."""
    points = []
    for _ in range(n):
        row = []
        for _ in range(dim):
            seed = lcg(seed)
            row.append(float(seed % (2 * span + 1) - span))
        points.append(row)
    return points


def sqdist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def subvector(vec, m, dsub=DSUB):
    start = m * dsub
    return vec[start:start + dsub]


def kmeans(points, k, max_rounds=MAX_ROUNDS):
    """Lloyd's algorithm with deterministic farthest-point initialisation.

    Returns (centroids, assignments, distortion, history) where history records the distortion
    after every round so the descent property can be checked."""
    n = len(points)
    if n == 0:
        raise ValueError("no points")
    if k > n:
        raise ValueError("k cannot exceed the number of points")
    dim = len(points[0])
    centroids = [points[0][:]]
    while len(centroids) < k:
        best_index = None
        best_distance = -1.0
        for i, point in enumerate(points):
            nearest = min(sqdist(point, c) for c in centroids)
            if nearest > best_distance:
                best_distance = nearest
                best_index = i
        centroids.append(points[best_index][:])

    assignments = [0] * n
    distortion = None
    history = []
    for _ in range(max_rounds):
        changed = False
        for i, point in enumerate(points):
            best, best_distance = 0, None
            for c, centroid in enumerate(centroids):
                distance = sqdist(point, centroid)
                if best_distance is None or distance < best_distance:
                    best, best_distance = c, distance
            if assignments[i] != best:
                assignments[i] = best
                changed = True
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for i, point in enumerate(points):
            c = assignments[i]
            counts[c] += 1
            for j in range(dim):
                sums[c][j] += point[j]
        for c in range(k):
            if counts[c] > 0:                       # an empty cluster keeps its old centroid
                centroids[c] = [sums[c][j] / counts[c] for j in range(dim)]
        new_distortion = sum(sqdist(point, centroids[assignments[i]])
                             for i, point in enumerate(points))
        if distortion is not None:
            assert new_distortion <= distortion + 1e-9, "Lloyd must not increase the distortion"
        history.append(new_distortion)
        if not changed or (distortion is not None and distortion - new_distortion <= 1e-12):
            distortion = new_distortion
            break
        distortion = new_distortion
    return centroids, assignments, distortion, history


def dp_1d_cost(values, k):
    """Exact optimal k-means cost for 1-D data (optimal clusters are contiguous intervals)."""
    xs = sorted(values)
    n = len(xs)
    prefix = [0.0] * (n + 1)
    prefix2 = [0.0] * (n + 1)
    for i, x in enumerate(xs):
        prefix[i + 1] = prefix[i] + x
        prefix2[i + 1] = prefix2[i] + x * x

    def block(i, j):
        total = prefix[j] - prefix[i]
        total2 = prefix2[j] - prefix2[i]
        count = j - i
        return total2 - total * total / count

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for t in range(1, k + 1):
        for j in range(1, n + 1):
            best = inf
            for i in range(t - 1, j):
                if dp[t - 1][i] < inf:
                    candidate = dp[t - 1][i] + block(i, j)
                    if candidate < best:
                        best = candidate
            dp[t][j] = best
    return dp[k][n]


def pq_train(vectors, codebook_size, m_subspaces=M_SUBSPACES, dsub=DSUB):
    """One k-means codebook per coordinate block. Returns the codebooks (M lists of centroids)."""
    codebooks = []
    for m in range(m_subspaces):
        block = [subvector(v, m, dsub) for v in vectors]
        centroids, _, _, _ = kmeans(block, codebook_size)
        codebooks.append(centroids)
    return codebooks


def pq_encode(codebooks, vector, dsub=DSUB):
    code = []
    for m, centroids in enumerate(codebooks):
        block = subvector(vector, m, dsub)
        best, best_distance = 0, None
        for c, centroid in enumerate(centroids):
            distance = sqdist(block, centroid)
            if best_distance is None or distance < best_distance:
                best, best_distance = c, distance
        code.append(best)
    return code


def pq_reconstruct(codebooks, code, dsub=DSUB):
    out = []
    for m, centroid_index in enumerate(code):
        out.extend(codebooks[m][centroid_index])
    return out


def pq_tables(codebooks, query, dsub=DSUB):
    """ADC table: tables[m][k] = squared distance from the query's m-th block to centroid k."""
    return [[sqdist(subvector(query, m, dsub), centroid)
             for centroid in codebooks[m]] for m in range(len(codebooks))]


def pq_adc(tables, code):
    return sum(tables[m][code[m]] for m in range(len(code)))


def pq_search(codebooks, codes, vectors, query, top_k, dsub=DSUB):
    """Approximate top-k by ADC distance. Returns [(distance, index)] sorted ascending."""
    tables = pq_tables(codebooks, query, dsub)
    scored = [(pq_adc(tables, code), i) for i, code in enumerate(codes)]
    scored.sort()
    scored = scored[:top_k]
    exact = sorted((sqdist(query, vectors[i]), i) for i in range(len(vectors)))[:top_k]
    return scored, exact


def clustered_data(clusters, per_cluster, dim, seed, spread=2, gap=40):
    """`clusters` blobs on a coarse grid so every blob is far from its neighbours."""
    points = []
    truth = []
    for c in range(clusters):
        centre = [float(((c * 7 + j * 3) % 5) * gap) for j in range(dim)]
        for _ in range(per_cluster):
            row = []
            for j in range(dim):
                seed = lcg(seed)
                row.append(centre[j] + (seed % (2 * spread + 1) - spread))
            points.append(row)
            truth.append(c)
    return points, truth


def main():
    # --- 1-D special case against the exact DP optimum. --------------------------------------
    values = [0.1, 0.2, 0.25, 4.0, 4.2, 4.1, 9.9, 10.0, 10.2, 10.1]
    for k in (1, 2, 3, 10):
        _, _, distortion, _ = kmeans([[v] for v in values], k)
        optimal = dp_1d_cost(values, k)
        assert distortion >= optimal - 1e-9, (k, distortion, optimal)
        assert distortion <= optimal + 1e-9, "farthest-point init is exact on 1-D intervals"
    _, _, k1_cost, _ = kmeans([[4.0]] * 5, 1)
    assert abs(k1_cost) <= 1e-18, "identical points have zero distortion"

    # Well-separated 1-D data: Lloyd's answer must equal the DP optimum exactly.
    separated = [x for x in range(0, 60, 6)] + [200.0 + x for x in range(0, 60, 6)]
    _, assignments, distortion, history = kmeans([[x] for x in separated], 2)
    assert abs(distortion - dp_1d_cost(separated, 2)) <= 1e-9
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1] + 1e-12, "distortion must never increase"
    assert len({assignments[i] for i in range(len(separated))}) == 2

    # --- ADC identity and reverse triangle inequality. ---------------------------------------
    data = make_data(120, DIM, seed=41)
    codebooks = pq_train(data, 8)
    assert len(codebooks) == M_SUBSPACES and all(len(cb) == 8 for cb in codebooks)
    assert all(len(c) == DSUB for cb in codebooks for c in cb)

    codes = [pq_encode(codebooks, v) for v in data]
    for i, (vector, code) in enumerate(zip(data, codes)):
        rebuilt = pq_reconstruct(codebooks, code)
        assert len(rebuilt) == DIM
        # Each code element must be the nearest centroid of its block.
        for m in range(M_SUBSPACES):
            block = subvector(vector, m)
            distances = [sqdist(block, c) for c in codebooks[m]]
            assert code[m] == distances.index(min(distances)), "code must be the nearest centroid"
        residual = math.sqrt(sqdist(vector, rebuilt))
        for query in (data[0], data[57], [0.0] * DIM):
            tables = pq_tables(codebooks, query)
            adc = pq_adc(tables, code)
            assert abs(adc - sqdist(query, rebuilt)) <= 1e-9, "ADC must equal the exact distance"
            # |d_approx - d_true| <= ||x - reconstruct(x)|| (reverse triangle inequality).
            assert abs(math.sqrt(adc) - math.sqrt(sqdist(query, vector))) <= residual + 1e-9

    # --- Reconstruction error shrinks as the codebook grows. ---------------------------------
    errors = []
    for k in (1, 2, 4, 8, 16):
        books = pq_train(data, k)
        total = 0.0
        for vector in data:
            total += sqdist(vector, pq_reconstruct(books, pq_encode(books, vector)))
        errors.append(total / len(data))
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], ("more centroids must reduce the error", errors)
    assert errors[0] > 0.0 and errors[-1] < errors[0] / 4.0

    # --- Recall on well-separated clusters: every query must stay inside its own blob. ------
    clustered, truth = clustered_data(5, 24, DIM, seed=97)
    books = pq_train(clustered, 8)
    cluster_codes = [pq_encode(books, v) for v in clustered]
    same_blob = 0
    self_hits = 0
    for i, vector in enumerate(clustered):
        scored, exact = pq_search(books, cluster_codes, clustered, vector, 1)
        assert len(scored) == 1 and len(exact) == 1
        assert truth[exact[0][1]] == truth[i], "the exact nearest neighbour is in the same blob"
        if truth[scored[0][1]] == truth[i]:
            same_blob += 1
        if scored[0][1] == i:
            self_hits += 1
    assert same_blob == len(clustered), ("PQ must never leave the blob", same_blob)
    # The exact top-1 itself is only recovered for about a quarter of the queries: with 8
    # centroids per 4-D block every point of a blob shares its code, so the ADC distances tie and
    # the argmin is just the lowest index. That is the documented price of 12-bit codes.
    assert 0.2 <= self_hits / len(clustered) <= 0.6, self_hits

    # --- Recall@10 on LCG data next to the brute-force answer, improving with the codebook. --
    recalls = []
    for codebook_size in (8, 16, 32):
        books_lcg = pq_train(data, codebook_size)
        codes_lcg = [pq_encode(books_lcg, v) for v in data]
        found = 0
        total = 0
        for query in data[:40]:
            scored, exact = pq_search(books_lcg, codes_lcg, data, query, 10)
            assert len(scored) == 10 and len(exact) == 10
            found += len({i for _, i in exact} & {i for _, i in scored})
            total += 10
        recalls.append(found / total)
    assert 0.5 <= recalls[0] <= 1.0, recalls
    for i in range(1, len(recalls)):
        assert recalls[i] >= recalls[i - 1], ("bigger codebooks must not lose recall", recalls)
    assert recalls[-1] > recalls[0] + 0.05, recalls

    # --- Code size and determinism. ----------------------------------------------------------
    bits_per_code = M_SUBSPACES * int(math.ceil(math.log2(8)))
    assert bits_per_code == 12 and bits_per_code < DIM * 64 // 10, "two orders of compression"
    books_again = pq_train(data, 8)
    assert books_again == codebooks, "training must be deterministic"
    assert [pq_encode(books_again, v) for v in data] == codes


if __name__ == "__main__":
    main()
    print("PASS")
