"""Task 815 - Implement real-time vector search at billion-vector scale with sharding and reranking.  (Tier: Master, CTCI Beyond - Advanced Data Structures & Algorithms)

A production vector search tier is a warehouse of embeddings split over many shards, each shard
carrying a coarse quantiser (an IVF: k-means centroids plus one inverted list per centroid) so that a
query only touches the few lists whose centroids are nearest, plus a *rerank* stage that rescores the
short candidate list with exact distances.  This implementation builds that whole path in process:

  * `shard_of(vector_id, shard_count)` is a keyed splitmix64 mix of the id, so a shard owns a
    partition of the id space that is deterministic, independent of the data and stable when the
    index grows (a real deployment hashes the id for exactly that reason);
  * `VectorCluster` builds one `IvfShard` per shard with deterministic Lloyd k-means
    (farthest-point initialisation, so there is no random seed to fix);
  * `search` scores the query against every centroid of every shard, scans the `nprobe` nearest lists
    using a cheap 3-bit scalar-quantised proxy, pools the candidates and *reranks the pool with exact
    float distances* - the rerank is what turns the coarse first-stage ordering into the final top-k;
  * `scale_plan` models the storage and the per-query cost of the same layout at any N.  The standard
    rule keeps a list at a constant size: `lists_per_shard ~ sqrt(rows_per_shard)`, and with a fixed
    `nprobe` the modelled candidate scans grow like sqrt(N) instead of N.

So the billion-vector claim is a typed parameter and an analysis on the *same* code path as the small
measured experiment: 600 vectors of dimension 12 are materialised, 10**9 are only planned.

Verified: identical answers across two runs, the shards partitioning the id space exactly (nothing
lost, nothing duplicated) with a bound on the imbalance, exact-distance agreement with a hand-computed
L2 oracle on a tiny case, full recall when every list is probed, recall@10 above 0.9 over 30
perturbed queries against the brute-force oracle, reranking measurably beating the quantised-only
ordering, monotone recall in `nprobe`, the answer always being a subset of the candidate pool,
per-query cost budgets (centroid distances, exact rerank calls, fraction of the corpus touched,
weighted operation count) and the sublinear growth of the modelled cost at 10**9 and 4 * 10**9.
Edge cases: k above the corpus size, duplicate vectors, a query equal to a stored vector, an empty
shard, an empty corpus, a single-vector corpus and a dimension mismatch.

Simplified: 600 vectors of dimension 12 live in one process, so "billion" is a modelled parameter
rather than terabytes of resident floats; distances are exact float arithmetic (no SIMD, no float16,
no product quantisation codes), the shard count is small so a query fans out to every shard instead of
a routing tier, and the latency budget is a deterministic weighted operation count, not wall-clock
time - no clocks, threads, sockets or sleeping anywhere.  At N = 600 with 24 lists per shard the
pruning factor is deliberately modest (probing 6 of 24 lists still touches about a third of the tiny
corpus, and the weighted operation count stays under 75% of brute force); the point of the small run
is the recall/rereank measurement, while the sublinear pruning factor is what `scale_plan` models and
the checks verify at 10**9, where a query touches under 0.1% of the corpus.
"""
import math

MASK64 = (1 << 64) - 1
DIMENSION = 12
SHARDS = 8
NLIST = 24
NPROBE = 6
K = 10
POOL = 128
QUANT_BITS = 3
QUANT_LEVELS = 1 << QUANT_BITS


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def mix(value, salt=0):
    _, out = splitmix64((value * 0x9E3779B97F4A7C15 + salt) & MASK64)
    return out


class Rng:
    """Fixed-seed splitmix64 stream: every run generates the same vectors."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def unit(self):
        """A deterministic value in [-1, 1)."""
        self.state, value = splitmix64(self.state)
        return (value >> 11) / float(1 << 52) - 1.0

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


# ------------------------------------------------------------------ distances
def sq_distance(a, b):
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    total = 0.0
    for left, right in zip(a, b):
        diff = left - right
        total += diff * diff
    return total


def centroid(points):
    if not points:
        raise ValueError("a centroid needs at least one point")
    out = [0.0] * len(points[0])
    for point in points:
        for index, value in enumerate(point):
            out[index] += value
    return [value / len(points) for value in out]


def quantise(vector):
    """3-bit scalar quantisation on the fixed range [-1, 1): the cheap first-stage proxy."""
    out = []
    for value in vector:
        level = int((value + 1.0) * 0.5 * QUANT_LEVELS)
        out.append(max(0, min(QUANT_LEVELS - 1, level)))
    return out


def quantised_distance(left_levels, right_levels):
    total = 0
    for left, right in zip(left_levels, right_levels):
        diff = left - right
        total += diff * diff
    return total


# ------------------------------------------------------------------ sharding
def shard_of(vector_id, shard_count):
    return mix(vector_id, 0x5A17ED) % shard_count


def build_shards(vectors, shard_count):
    """Partition the id space; returns {shard: [ids in ascending order]}."""
    shards = {shard: [] for shard in range(shard_count)}
    for vector_id in sorted(vectors):
        shards[shard_of(vector_id, shard_count)].append(vector_id)
    return shards


def kmeans(points, clusters, iterations):
    """Deterministic Lloyd k-means: farthest-point initialisation, so no seed is needed."""
    if not points:
        return [], []
    clusters = max(1, min(clusters, len(points)))
    centroids = [list(points[0])]
    while len(centroids) < clusters:
        far, far_distance = None, -1.0
        for point in points:
            nearest = min(sq_distance(point, centre) for centre in centroids)
            if nearest > far_distance:
                far, far_distance = point, nearest
        centroids.append(list(far))
    assignment = [0] * len(points)
    for _ in range(iterations):
        for index, point in enumerate(points):
            best, best_distance = 0, None
            for cluster, centre in enumerate(centroids):
                distance = sq_distance(point, centre)
                if best_distance is None or distance < best_distance:
                    best, best_distance = cluster, distance
            assignment[index] = best
        for cluster in range(len(centroids)):
            members = [point for point, owner in zip(points, assignment) if owner == cluster]
            if members:
                centroids[cluster] = centroid(members)
    return centroids, assignment


class IvfShard:
    """One shard: centroids, one inverted list per centroid, and the quantised copies of its rows."""

    def __init__(self, ids, vectors, clusters, iterations):
        self.ids = list(ids)
        self.vectors = vectors
        self.centroids = []
        self.lists = []
        self.levels = {vector_id: quantise(vectors[vector_id]) for vector_id in self.ids}
        if not self.ids:
            return
        points = [vectors[vector_id] for vector_id in self.ids]
        self.centroids, assignment = kmeans(points, clusters, iterations)
        self.lists = [[] for _ in self.centroids]
        for vector_id, owner in zip(self.ids, assignment):
            self.lists[owner].append(vector_id)
        for entry in self.lists:
            entry.sort()

    def coverage(self):
        return sorted(vector_id for entry in self.lists for vector_id in entry)

    def scan(self, query_levels, query, nprobe, counters):
        """Score the centroids, probe the nearest lists and return quantised-ranked candidates."""
        if not self.ids:
            return []
        scored = []
        for index, centre in enumerate(self.centroids):
            counters["centroid_calls"] += 1
            scored.append((sq_distance(query, centre), index))
        scored.sort()
        candidates = []
        for _, index in scored[:nprobe]:
            for vector_id in self.lists[index]:
                counters["quantised_calls"] += 1
                candidates.append((quantised_distance(query_levels, self.levels[vector_id]), vector_id))
        return candidates

    def exact_top(self, query, k, counters):
        scored = []
        for vector_id in self.ids:
            counters["exact_calls"] += 1
            scored.append((sq_distance(query, self.vectors[vector_id]), vector_id))
        scored.sort()
        return [(distance, vector_id) for distance, vector_id in scored[:k]]


def new_counters():
    return {"centroid_calls": 0, "quantised_calls": 0, "exact_calls": 0, "pool_size": 0}


class VectorCluster:
    """The sharded IVF index plus the rerank stage; also holds the brute-force oracle."""

    def __init__(self, vectors, shard_count=SHARDS, clusters=NLIST, iterations=8):
        self.vectors = vectors
        self.shards = build_shards(vectors, shard_count)
        self.index = {shard: IvfShard(ids, vectors, clusters, iterations)
                      for shard, ids in sorted(self.shards.items())}

    def brute_force(self, query, k):
        scored = sorted((sq_distance(query, vector), vector_id)
                        for vector_id, vector in self.vectors.items())
        return [(distance, vector_id) for distance, vector_id in scored[:k]]

    def search(self, query, k=K, nprobe=NPROBE, pool=POOL, counters=None, rerank=True):
        """Two stages: probe every shard with the cheap proxy, then rerank the pool exactly.

        Returns (top-k, pool ids); with rerank=False the first stage's proxy ordering is returned,
        which is exactly the ordering the rerank is supposed to improve on."""
        counters = counters if counters is not None else new_counters()
        levels = quantise(query)
        candidates = []
        for shard in sorted(self.index):
            candidates.extend(self.index[shard].scan(levels, query, nprobe, counters))
        candidates.sort()
        pool_ids = [vector_id for _, vector_id in candidates[:pool]]
        counters["pool_size"] = max(counters["pool_size"], len(pool_ids))
        if not rerank:
            return [(distance, vector_id) for distance, vector_id in candidates[:k]], pool_ids
        ranked = []
        for vector_id in pool_ids:
            counters["exact_calls"] += 1
            ranked.append((sq_distance(query, self.vectors[vector_id]), vector_id))
        ranked.sort()
        return [(distance, vector_id) for distance, vector_id in ranked[:k]], pool_ids

    def proxy_ranking(self, query, k=K, nprobe=NPROBE, pool=POOL):
        """The first stage's own ordering: quantised distances only, no float rerank."""
        ranked, _ = self.search(query, k=k, nprobe=nprobe, pool=pool, counters=new_counters(),
                                rerank=False)
        return [vector_id for _, vector_id in ranked]


# ------------------------------------------------------------------ the scale model
def scale_plan(total_vectors, dimension, shard_count, nprobe=8):
    """Model storage and per-query cost of the same layout at any N, without materialising it.

    Keeping a list at a constant size means lists_per_shard ~ sqrt(rows_per_shard); with a fixed
    nprobe the candidate scans then grow like sqrt(N * shard_count), not like N."""
    rows_per_shard = -(-total_vectors // shard_count)
    lists_per_shard = max(1, math.isqrt(rows_per_shard))
    rows_per_list = -(-rows_per_shard // lists_per_shard)
    probed = min(nprobe, lists_per_shard)
    return {
        "total_vectors": total_vectors,
        "bytes": total_vectors * dimension * 4,
        "shards": shard_count,
        "lists_per_shard": lists_per_shard,
        "rows_per_list": rows_per_list,
        "nprobe": probed,
        "centroid_calls_per_query": shard_count * lists_per_shard,
        "candidate_scans_per_query": shard_count * probed * rows_per_list,
    }


def make_vectors(rng, count, dimension):
    return {vector_id: [rng.unit() for _ in range(dimension)] for vector_id in range(count)}


def recall_at_k(truth_ids, approximate_ids):
    if not truth_ids:
        return 1.0
    return len(set(truth_ids) & set(approximate_ids)) / float(len(truth_ids))


def perturb(rng, vector, amount=0.05):
    return [value + amount * rng.unit() for value in vector]


def main():
    rng = Rng(0x1EC70)
    vectors = make_vectors(rng, 600, DIMENSION)
    cluster = VectorCluster(vectors)
    queries = [perturb(rng, vectors[(offset * 7) % len(vectors)]) for offset in range(30)]

    # 1. sharding partitions the id space exactly, deterministically, with bounded imbalance
    shards = build_shards(vectors, SHARDS)
    flat = sorted(vector_id for ids in shards.values() for vector_id in ids)
    check(flat == sorted(vectors), "the shards must partition the id space exactly")
    check(len(flat) == len(set(flat)), "no vector may live in two shards")
    check(build_shards(vectors, SHARDS) == shards, "shard assignment must be deterministic")
    sizes = [len(ids) for ids in shards.values()]
    check(max(sizes) <= 2 * (len(vectors) // SHARDS), f"shard imbalance {sizes} is too large")
    check(0 <= shard_of(7, SHARDS) < SHARDS and shard_of(7, SHARDS) == shard_of(7, SHARDS),
          "shard_of must be a stable function into the shard range")
    check(len({shard_of(vector_id, SHARDS) for vector_id in range(400)}) == SHARDS,
          "the hash must spread over every shard")

    # 2. the exact distance matches a hand-computed L2 oracle, and quantisation clamps
    check(sq_distance([0.0, 0.0], [1.0, 2.0]) == 5.0,
          "squared L2 must be the sum of squared differences")
    check(sq_distance([1.0], [1.0]) == 0.0, "a point is at distance zero from itself")
    try:
        sq_distance([1.0, 2.0], [1.0])
        check(False, "a dimension mismatch must be rejected")
    except ValueError:
        pass
    check(quantise([-1.0]) == [0] and quantise([0.999]) == [QUANT_LEVELS - 1],
          "scalar quantisation must clamp into the level range")
    check(quantised_distance(quantise([0.0]), quantise([0.0])) == 0,
          "identical levels must be at proxy distance zero")

    # 3. every inverted list is intact: probing everything reproduces the brute-force answer
    for shard, index in sorted(cluster.index.items()):
        check(index.coverage() == sorted(shards[shard]),
              f"shard {shard} must hold every one of its vectors exactly once")
    counters = new_counters()
    query = vectors[3]
    collected = []
    for shard in sorted(cluster.index):
        collected.extend(cluster.index[shard].exact_top(query, 5, counters))
    check([(distance, vector_id) for distance, vector_id in sorted(collected)[:5]] ==
          cluster.brute_force(query, 5),
          "probing every shard exactly must agree with the brute-force oracle")
    widest = max(len(index.lists) for index in cluster.index.values())
    found, _ = cluster.search(query, k=5, nprobe=widest, pool=len(vectors))
    check([vector_id for _, vector_id in found] ==
          [vector_id for _, vector_id in cluster.brute_force(query, 5)],
          "probing every list of every shard must reproduce the exact answer")

    # 4. recall@10 over 30 perturbed queries, with and without the exact rerank
    recalls, proxy_recalls, gains = [], [], 0
    for query in queries:
        truth = [vector_id for _, vector_id in cluster.brute_force(query, K)]
        counters = new_counters()
        found, pool_ids = cluster.search(query, k=K, counters=counters, rerank=True)
        proxy_ids = cluster.proxy_ranking(query, k=K)
        recalls.append(recall_at_k(truth, [vector_id for _, vector_id in found]))
        proxy_recalls.append(recall_at_k(truth, proxy_ids))
        gains += 1 if recalls[-1] > proxy_recalls[-1] else 0
        check(set(vector_id for _, vector_id in found) <= set(pool_ids),
              "the final answer must be a subset of the candidate pool")
        check(counters["pool_size"] <= POOL, "the pool must respect its bound")
    mean_recall = sum(recalls) / len(recalls)
    mean_proxy = sum(proxy_recalls) / len(proxy_recalls)
    check(mean_recall >= 0.9, f"recall@{K} must exceed 0.9, measured {mean_recall:.3f}")
    check(mean_recall >= mean_proxy,
          f"the rerank must not hurt recall: {mean_recall:.3f} < {mean_proxy:.3f}")
    check(gains > 0, "the exact rerank must improve at least one query")

    # 5. recall is monotone in nprobe and reaches 1.0 when everything is probed
    query = queries[0]
    truth = [vector_id for _, vector_id in cluster.brute_force(query, K)]
    previous = -1.0
    for probe in (1, 2, 6, widest):
        found, _ = cluster.search(query, k=K, nprobe=probe, pool=len(vectors))
        current = recall_at_k(truth, [vector_id for _, vector_id in found])
        check(current >= previous, f"recall must not fall when nprobe grows, probe={probe}")
        previous = current
    check(previous == 1.0, "probing every list must reach recall 1.0")

    # 6. cost budgets: a query touches a bounded, deterministic part of the corpus
    counters = new_counters()
    found, _ = cluster.search(queries[1], k=K, counters=counters)
    check(counters["centroid_calls"] == SHARDS * NLIST,
          f"a query must score every centroid once, got {counters['centroid_calls']}")
    check(counters["quantised_calls"] <= len(vectors) // 2,
          f"the first stage must scan under half the corpus, got {counters['quantised_calls']}")
    check(counters["exact_calls"] == counters["pool_size"] <= POOL,
          "the rerank must cost exactly one distance per pooled candidate")
    check(counters["exact_calls"] < 0.25 * len(vectors),
          f"the rerank must touch under 25% of the corpus, got {counters['exact_calls']}")
    cycles = (counters["centroid_calls"] + counters["exact_calls"]) * DIMENSION * 4 + \
        counters["quantised_calls"] * DIMENSION
    brute_cycles = len(vectors) * DIMENSION * 4
    check(cycles < 0.75 * brute_cycles,
          f"the weighted operation count {cycles} must stay under 75% of brute force "
          f"({brute_cycles}) at this deliberately small N")

    # 7. the modelled billion-vector plan is sublinear in N
    small = scale_plan(600, DIMENSION, SHARDS)
    big = scale_plan(10 ** 9, DIMENSION, SHARDS)
    quad = scale_plan(4 * 10 ** 9, DIMENSION, SHARDS)
    check(small["bytes"] == 600 * DIMENSION * 4, "the memory model must be rows times dimensions")
    check(big["bytes"] == 10 ** 9 * DIMENSION * 4, "the 1e9 memory model must be linear in N")
    check(big["lists_per_shard"] ** 2 <= big["total_vectors"] // SHARDS,
          "the sqrt rule must keep the list count near the square root of the shard size")
    growth_n = big["total_vectors"] / small["total_vectors"]
    growth_c = big["candidate_scans_per_query"] / small["candidate_scans_per_query"]
    check(growth_c < growth_n / 100,
          f"the modelled per-query work must grow far slower than N: {growth_c:.1f} vs {growth_n:.0f}")
    check(big["candidate_scans_per_query"] < 0.01 * big["total_vectors"],
          f"a query at 1e9 vectors must scan under 1% of the corpus, modelled "
          f"{big['candidate_scans_per_query']}")
    check(quad["candidate_scans_per_query"] / big["candidate_scans_per_query"] < 2.5,
          "quadrupling N must less than triple the modelled per-query cost")

    # 8. edge cases and determinism
    empty = VectorCluster({}, shard_count=4)
    check(empty.brute_force([0.0] * DIMENSION, 5) == [], "an empty corpus has no neighbours")
    check(empty.search([0.0] * DIMENSION, k=5)[0] == [], "searching an empty corpus is empty")
    check(empty.index[0].coverage() == [], "an empty shard holds nothing")
    single = VectorCluster({99: [0.5] * DIMENSION}, shard_count=4)
    found, _ = single.search([0.6] * DIMENSION, k=3)
    check(len(found) == 1 and found[0][1] == 99, "a single vector must be found with any k")
    duplicate = VectorCluster({1: [0.25] * DIMENSION, 2: [0.25] * DIMENSION, 3: [0.9] * DIMENSION},
                              shard_count=2)
    found, _ = duplicate.search([0.25] * DIMENSION, k=2)
    check([vector_id for _, vector_id in found] == [1, 2],
          "two identical vectors must both surface with distance zero")
    check(found[0][0] == 0.0, "an exact match must have distance zero")
    everything, _ = cluster.search(query, k=10 ** 6, nprobe=widest, pool=len(vectors))
    check(len(everything) == len(vectors), "a k above the corpus size must return the whole corpus")
    first, _ = cluster.search(vectors[5], k=K)
    second, _ = cluster.search(vectors[5], k=K)
    check(first == second, "two identical queries must return identical results")
    check(first[0][1] == 5 and abs(first[0][0]) < 1e-12,
          "a query equal to a stored vector must return that vector first, at distance zero")

    print(f"recall@{K}: two-stage={mean_recall:.3f} proxy-only={mean_proxy:.3f} "
          f"full-probe=1.000 (gains on {gains}/30 queries)")
    print(f"cost/query: centroids={counters['centroid_calls']} scanned="
          f"{counters['quantised_calls']} rerank={counters['exact_calls']} "
          f"of {len(vectors)}; modelled 1e9 scan={big['candidate_scans_per_query']}")
    print("PASS")


if __name__ == "__main__":
    main()
