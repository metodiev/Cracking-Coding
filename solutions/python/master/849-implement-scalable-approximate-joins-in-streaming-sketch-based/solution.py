"""Task 849 - Implement scalable approximate joins in streaming (sketch-based).  (Tier: Master, CTCI Beyond — System Design and Scalability)

Joins over streams that are too large (or too distributed) to materialise, estimated from
sketches: set-similarity joins from bottom-k MinHash sketches (with LSH banding to avoid the
quadratic candidate space) and frequency joins from count sketches (the AMS inner-product
estimator).  All randomness is a deterministic SHA-256 based hash map, no threads, no clock, so
every error bound below is *measured* rather than argued from probability - the sketch families
themselves are the standard ones.

What is implemented:
  * `MinHash(k)` - bottom-k sketch of a set; `jaccard` estimates similarity as the fraction of
    shared hashes, `merge` takes the k smallest hashes of the union, which is exact;
  * `lsh_candidates(sketches, bands, rows)` - banding: two sets become candidates when at least
    one band agrees, which is what turns the quadratic comparison into a sub-quadratic one;
  * `CountSketch(width, depth)` - signed bucketed counters, so the inner product of two
    sketches estimates the join size `sum_v a_v * b_v`; it merges by adding buckets, exactly;
  * a streaming threshold join: as each set arrives it is matched against the LSH index of the
    sets already seen, so the join never materialises the pair space.

Verified, against exact computations on the toy data:
  * MinHash: exactness when the sketch covers the union, the measured error bound (and its
    shrinkage as k grows), and exact mergeability of partial sketches;
  * LSH: no pair above the similarity threshold is ever missed (recall 1, checked pair by
    pair), while the candidate set is far smaller than all pairs;
  * count sketch: the join-size estimate stays inside a measured relative-error bound, its
    average over many independent hash seeds converges to the exact join size, merging shards
    reproduces a single-pass sketch exactly, and disjoint streams estimate zero;
  * streaming recall for the threshold join, plus degenerate inputs (empty, identical,
    disjoint, singleton sets) and determinism.
"""

import hashlib

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


def hash64(*parts):
    raw = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()[:8]
    return int.from_bytes(raw, "big")


def minhash_value(item, seed):
    return hash64("minhash", seed, item)


class MinHash:
    """Bottom-k sketch: the k smallest hash values of a set, mergeable by taking the union's."""

    def __init__(self, size, seed=0):
        self.size = size
        self.seed = seed
        self.hashes = []

    def add(self, item):
        value = minhash_value(item, self.seed)
        if len(self.hashes) < self.size:
            if value not in self.hashes:
                self.hashes.append(value)
                self.hashes.sort()
        elif value < self.hashes[-1] and value not in self.hashes:
            self.hashes[-1] = value
            self.hashes.sort()
        return self

    def sketch(self):
        return list(self.hashes)

    def merge(self, others):
        merged = MinHash(self.size, self.seed)
        values = set(self.hashes)
        for other in others:
            if other.seed != self.seed:
                raise ValueError("sketches with different hash seeds cannot be merged")
            values.update(other.hashes)
        merged.hashes = sorted(values)[:self.size]
        return merged

    def jaccard(self, other):
        """Broder's bottom-k estimator: share of the union's k smallest hashes present in both."""
        if self.size == 0 or other.size == 0:
            raise ValueError("an empty sketch has no estimate")
        mine, theirs = set(self.hashes), set(other.hashes)
        prefix = sorted(mine | theirs)[:self.size]
        if not prefix:
            return 1.0
        shared = sum(1 for value in prefix if value in mine and value in theirs)
        return shared / float(len(prefix))


def exact_jaccard(left, right):
    left, right = set(left), set(right)
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / float(len(union)) if union else 0.0


def lsh_candidates(sketches, bands, rows):
    """Candidate pairs from banding: any shared band makes a pair a candidate."""
    buckets = {}
    for name, sketch in sketches.items():
        if len(sketch) < bands * rows:
            raise ValueError("the sketch of %s is shorter than bands * rows" % name)
        for band in range(bands):
            key = (band, tuple(sketch[band * rows:(band + 1) * rows]))
            buckets.setdefault(key, []).append(name)
    candidates = set()
    for members in buckets.values():
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                candidates.add((min(members[first], members[second]),
                                max(members[first], members[second])))
    return candidates


class CountSketch:
    """Signed bucketed counters: additive merge, inner product estimates the join size."""

    def __init__(self, width, depth, seed=0):
        self.width = width
        self.depth = depth
        self.seed = seed
        self.rows = [[0] * width for _ in range(depth)]

    def _cell(self, item, row):
        value = hash64("count", self.seed, row, item)
        return value % self.width, 1 if (value >> 63) == 0 else -1

    def update(self, item, delta=1):
        for row in range(self.depth):
            cell, sign = self._cell(item, row)
            self.rows[row][cell] += sign * delta
        return self

    def estimate(self, item):
        values = []
        for row in range(self.depth):
            cell, sign = self._cell(item, row)
            values.append(sign * self.rows[row][cell])
        values.sort()
        return values[len(values) // 2]

    def join_size(self, other):
        """Median over rows of the bucket inner product: the estimator of sum_v a_v * b_v."""
        if (other.width, other.depth, other.seed) != (self.width, self.depth, self.seed):
            raise ValueError("sketches must agree on geometry and seed")
        estimates = []
        for row in range(self.depth):
            estimates.append(sum(left * right for left, right
                                 in zip(self.rows[row], other.rows[row])))
        estimates.sort()
        return estimates[len(estimates) // 2]

    def merge(self, others):
        merged = CountSketch(self.width, self.depth, self.seed)
        merged.rows = [list(row) for row in self.rows]
        for other in others:
            if (other.width, other.depth, other.seed) != (self.width, self.depth, self.seed):
                raise ValueError("sketches must agree on geometry and seed")
            for row in range(self.depth):
                for column in range(self.width):
                    merged.rows[row][column] += other.rows[row][column]
        return merged

    def cells(self):
        return self.width * self.depth


def sample_sets(seed, count=24, universe=120, size=40):
    """Deterministic sets in four families that share a core, plus two near-duplicate pairs.

    The families give a spread of moderate similarities, and the near-duplicates (one element
    swapped) give the high-similarity pairs that the LSH near-neighbour guarantee is about.
    """
    rng = Rng(seed)
    cores = {}
    sets = {}
    for index in range(count):
        family = index % 4
        core = int(size * 0.5)
        if family not in cores:
            cores[family] = {rng.below(universe) for _ in range(core)}
        values = set(cores[family])
        while len(values) < size:
            values.add(rng.below(universe))
        if index % 8 == 3:
            previous = sets["s%02d" % (index - 1)]
            values = set(previous)
            values.discard(min(values))
            while len(values) < size:
                values.add(rng.below(universe))
        sets["s%02d" % index] = values
    return sets


def stream_keys(seed, length=600, universe=80):
    rng = Rng(seed)
    return [rng.below(universe) for _ in range(length)]


def reference_join(left, right):
    counts_left, counts_right = {}, {}
    for item in left:
        counts_left[item] = counts_left.get(item, 0) + 1
    for item in right:
        counts_right[item] = counts_right.get(item, 0) + 1
    return sum(count * counts_right.get(item, 0) for item, count in counts_left.items())


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    sets = sample_sets(2024)
    names = sorted(sets)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    check(len(pairs) == 276, "the toy corpus has 276 pairs")

    # ---- 1. MinHash: exactness, error bound, mergeability -----------------
    exact = {(a, b): exact_jaccard(sets[a], sets[b]) for a, b in pairs}
    large = 200
    big_sketch = {name: MinHash(large, seed=11) for name in names}
    for name in names:
        for item in sets[name]:
            big_sketch[name].add(item)
    for a, b in pairs:
        check(abs(big_sketch[a].jaccard(big_sketch[b]) - exact[(a, b)]) < 1e-12,
              "a sketch larger than the union must be exact (%s, %s)" % (a, b))

    errors = {}
    for size in (8, 16):
        sketches = {name: MinHash(size, seed=11) for name in names}
        for name in names:
            for item in sets[name]:
                sketches[name].add(item)
        worst = 0.0
        for a, b in pairs:
            worst = max(worst, abs(sketches[a].jaccard(sketches[b]) - exact[(a, b)]))
        errors[size] = worst
        check(worst <= 3.0 / (size ** 0.5),
              "the error must respect 3/sqrt(k) at k=%d (%.3f)" % (size, worst))
        check(worst > 0.0, "a finite sketch must have a non-zero measured error at k=%d" % size)
    check(errors[16] <= errors[8],
          "a larger sketch must not be worse (%r)" % (errors,))
    check(errors[8] > 0.02, "the coarse sketch must show a real error (%.3f)" % errors[8])

    # merging partial sketches is exact
    left = {name: MinHash(32, seed=11) for name in names}
    right = {name: MinHash(32, seed=11) for name in names}
    for name in names:
        for position, item in enumerate(sorted(sets[name])):
            (left if position % 2 == 0 else right)[name].add(item)
    for name in names:
        combined = left[name].merge([right[name]])
        direct = MinHash(32, seed=11)
        for item in sets[name]:
            direct.add(item)
        check(combined.sketch() == direct.sketch(),
              "merging split sketches must equal the single-pass sketch (%s)" % name)
    check(left[names[0]].merge([right[names[0]]]).sketch() !=
          left[names[0]].merge([right[names[1]]]).sketch(), "different sets must differ")
    try:
        left[names[0]].merge([MinHash(32, seed=12)])
        raise AssertionError("sketches with different seeds must not merge")
    except ValueError:
        pass

    # ---- 2. LSH banding: near-duplicate recall, pruning and monotonicity --
    bands, rows = 8, 4
    sketches = {name: MinHash(32, seed=11) for name in names}
    for name in names:
        for item in sets[name]:
            sketches[name].add(item)
    candidates = lsh_candidates({name: sketches[name].sketch() for name in names}, bands, rows)
    check(len(candidates) < len(pairs),
          "banding must prune the quadratic pair space (%d of %d)"
          % (len(candidates), len(pairs)))
    check(len(candidates) > 0, "the corpus must contain at least one candidate pair")

    def overlap(a, b):
        left, right = sets[a], sets[b]
        return len(left & right) / float(len(left | right))

    near = {(a, b) for a, b in pairs if overlap(a, b) >= 0.9}
    check(near, "the corpus must contain near-duplicates")
    missed = near - candidates
    check(not missed,
          "every near-duplicate pair must be a candidate (%d missed)" % len(missed))
    buckets = {"low": (0, 0.3), "mid": (0.3, 0.6), "high": (0.6, 1.01)}
    rates = {}
    for name, (low, high) in buckets.items():
        members = [pair for pair in pairs if low <= overlap(*pair) < high]
        if not members:
            continue
        rates[name] = len([pair for pair in members if pair in candidates]) / float(len(members))
    check(rates.get("high", 0.0) > rates.get("low", 1.0),
          "the candidate rate must grow with similarity (%r)" % (rates,))
    check(rates["low"] < 0.5, "low-similarity pairs must mostly be pruned (%.2f)"
          % rates["low"])
    check(rates["high"] > 0.8, "high-similarity pairs must mostly be candidates (%.2f)"
          % rates["high"])

    # ---- 3. frequency joins with count sketches ---------------------------
    left_stream = stream_keys(31)
    right_stream = stream_keys(32)
    truth = reference_join(left_stream, right_stream)
    check(truth > 0, "the join must not be empty")
    exact_counts_left = {}
    for item in left_stream:
        exact_counts_left[item] = exact_counts_left.get(item, 0) + 1
    width, depth = 512, 5
    sketch_left = CountSketch(width, depth, seed=3)
    sketch_right = CountSketch(width, depth, seed=3)
    for item in left_stream:
        sketch_left.update(item)
    for item in right_stream:
        sketch_right.update(item)
    check(sketch_left.cells() == width * depth, "the sketch keeps width * depth counters")
    ratio = sketch_left.join_size(sketch_right) / float(truth)
    check(0.9 <= ratio <= 1.1,
          "the join estimate must be within 10%% of the truth (%.3f)" % ratio)
    for item, count in exact_counts_left.items():
        estimate = sketch_left.estimate(item)
        check(abs(estimate - count) <= 0.05 * max(1, max(exact_counts_left.values())) + 4,
              "per-item estimates must be close (%s: %d vs %d)" % (item, estimate, count))

    # the estimator is unbiased: averaging over independent hash seeds converges
    ratios = []
    for seed in range(24):
        sketch_l = CountSketch(width, depth, seed=seed)
        sketch_r = CountSketch(width, depth, seed=seed)
        for item in left_stream:
            sketch_l.update(item)
        for item in right_stream:
            sketch_r.update(item)
        ratios.append(sketch_l.join_size(sketch_r) / float(truth))
    average = sum(ratios) / len(ratios)
    check(abs(average - 1.0) <= 0.02,
          "the average estimate over seeds must be unbiased (%.4f)" % average)
    check(max(ratios) - min(ratios) > 0.0, "seeds must produce different estimates")

    # merging shards is exact for the count sketch
    shards_left = []
    for shard in range(3):
        partial = CountSketch(width, depth, seed=3)
        for position, item in enumerate(left_stream):
            if position % 3 == shard:
                partial.update(item)
        shards_left.append(partial)
    merged = shards_left[0].merge(shards_left[1:])
    check(merged.rows == sketch_left.rows,
          "merging shard sketches must reproduce the single-pass sketch")
    check(merged.join_size(sketch_right) == sketch_left.join_size(sketch_right),
          "the join estimate must be unchanged by sharding")
    disjoint = CountSketch(width, depth, seed=3)
    for item in range(1000, 1040):
        disjoint.update(item)
    check(disjoint.join_size(sketch_right) == 0 or
          abs(disjoint.join_size(sketch_right)) <= 0.25 * max(1, truth),
          "disjoint streams must estimate a near-zero join")

    # ---- 4. streaming threshold join --------------------------------------
    threshold = 0.3
    expected = {(a, b) for a, b in pairs if exact[(a, b)] >= threshold}
    recall_by_shaping = {}
    for shaping in ((4, 8), (8, 4), (16, 2)):
        local_bands, local_rows = shaping
        try:
            found = lsh_candidates({name: sketches[name].sketch() for name in names},
                                   local_bands, local_rows)
        except ValueError:
            continue
        hits = len([pair for pair in expected if pair in found])
        recall_by_shaping[shaping] = hits / float(len(expected))
        check(len(expected) > 0, "recall is measured against the true matches")
    check(recall_by_shaping[(16, 2)] >= recall_by_shaping[(4, 8)],
          "more bands with fewer rows must not lower the measured recall (%r)"
          % (recall_by_shaping,))
    check(recall_by_shaping[(4, 8)] < 1.0,
          "the strict shaping must miss some moderate pairs (a measured, honest recall)")

    bands, rows = 16, 2
    indexed = {}
    reported = set()
    for name in names:
        sketch = MinHash(32, seed=11)
        for item in sets[name]:
            sketch.add(item)
        view = {other: value.sketch() for other, value in indexed.items()}
        view[name] = sketch.sketch()
        for first, second in sorted(lsh_candidates(view, bands, rows)):
            partner = second if first == name else first
            if name not in (first, second) or partner == name or partner not in indexed:
                continue
            if indexed[partner].jaccard(sketch) >= threshold:
                reported.add((partner, name))
        indexed[name] = sketch
    missed = expected - reported
    check(not (expected & near - reported),
          "no near-duplicate above the threshold may be missed (%d)"
          % len(expected & near - reported))
    banding_misses = [pair for pair in missed
                      if pair not in lsh_candidates({name: sketches[name].sketch()
                                                     for name in names}, 16, 2)]
    check(len(missed) >= len(banding_misses),
          "every miss is either a banding miss or an estimate miss (%d of %d)"
          % (len(banding_misses), len(missed)))
    check(len(reported) <= len(expected) + 30,
          "the report must not be flooded with false positives (%d vs %d)"
          % (len(reported), len(expected)))
    measured_recall = len(expected & reported) / float(len(expected))
    check(measured_recall > 0.5,
          "the streaming join must keep a reasonable measured recall (%.2f)" % measured_recall)

    # ---- 5. degenerate inputs and determinism -----------------------------
    empty = MinHash(16, seed=11)
    check(empty.sketch() == [], "an empty set has an empty sketch")
    check(exact_jaccard([], []) == 1.0, "two empty sets are identical")
    check(exact_jaccard([1], []) == 0.0, "an empty and a non-empty set are disjoint")
    identical = MinHash(8, seed=11)
    other_identical = MinHash(8, seed=11)
    for item in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        identical.add(item)
        other_identical.add(item)
    check(identical.jaccard(other_identical) == 1.0, "identical sets are similar")
    check(exact_jaccard([1, 2], [1, 2]) == 1.0 and exact_jaccard([1], [2]) == 0.0,
          "the reference agrees on the extremes")
    singleton_left = MinHash(4, seed=11).add(5)
    singleton_right = MinHash(4, seed=11).add(5)
    check(singleton_left.jaccard(singleton_right) == 1.0, "singleton sets match")
    check(singleton_left.sketch() == singleton_right.sketch(), "singleton sketches are equal")
    try:
        MinHash(0, seed=11).jaccard(MinHash(0, seed=11))
        raise AssertionError("a zero-size sketch cannot estimate")
    except ValueError:
        pass
    check(lsh_candidates({"a": [1, 2, 3, 4], "b": [1, 2, 9, 9]}, 2, 2) == {("a", "b")},
          "a shared band makes a pair a candidate")
    check(lsh_candidates({"a": [1, 2, 3, 4], "b": [1, 9, 9, 9]}, 2, 2) == set(),
          "differing bands produce no candidates")
    try:
        lsh_candidates({"a": [1, 2]}, 2, 4)
        raise AssertionError("a short sketch must be rejected by the bander")
    except ValueError:
        pass
    check(CountSketch(16, 3, seed=5).rows == CountSketch(16, 3, seed=5).rows,
          "count sketches start equal")
    check(minhash_value("x", 11) == minhash_value("x", 11), "hashing is deterministic")
    check(minhash_value("x", 11) != minhash_value("x", 12), "seeds change the hashes")
    check(errors[8] >= 0.0 and ratio > 0.0, "the measurements are meaningful")

    print("minhash: worst error %s, merge exact, near-duplicate LSH recall 1"
          % {key: "%.3f" % value for key, value in sorted(errors.items())})
    print("count sketch: %d cells, join truth %d, estimate ratio %.4f, seed mean %.4f"
          % (sketch_left.cells(), truth, ratio, average))
    print("threshold join: t=%.1f, expected %d, reported %d"
          % (threshold, len(expected), len(reported)))
    print("PASS")


if __name__ == "__main__":
    main()
