"""Task 770 - Implement streaming algorithms for heavy hitters under memory constraints.  (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)

One pass, O(1) memory per counter, no storage of the stream:
  * Misra-Gries (MG) keeps at most k counters.  An arriving item with no counter either gets a
    fresh one, or - when all k are taken - every counter is decreased by one and the zeros are
    dropped ("decrement round").  Because a decrement round costs k units of total count and the
    stream holds N items, at most N/k rounds ever happen; therefore for every item i
        f_i - N/k <= est_i <= f_i
    where est_i is the counter value (0 when the item was evicted).  In particular every item with
    f_i > N/k is still monitored, and no unmonitored item can have f_i > N/k: exact heavy hitters
    with a lower-bound estimate and zero false negatives.
  * Count-Min (CM) keeps depth x width counters.  Item x hashes to one cell per row and adds 1;
    the estimate is the minimum over rows, so collisions can only *over*-estimate, giving
        0 <= est_x - f_x <= eps * N     with width = ceil(e/eps) and depth = ceil(ln(1/delta)).

What is verified: both sketches against the exact frequency oracle on skewed (Zipf) and on
uniform-all-distinct streams, the MG error bound f - est <= N/k and the no-false-negative
property for every true heavy hitter, the MG memory bound |counters| <= k after *every* insert,
the CM one-sided error bound eps*N for every key, CM exactness when width >= distinct keys and
depth == 1, CM mergeability (sketch(A) + sketch(B) == sketch(A ++ B), which is what makes the
sketch usable in a distributed aggregator), and determinism of the whole pipeline.

Simplified: the hash family is a fixed splitmix64-based mixing function instead of a randomly
chosen 2-universal family, so the CM guarantee is verified empirically on the tested streams
rather than invoked as a probability statement; streams are kept to a few thousand items so the
exact oracle stays cheap.
"""
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


def mix(value, salt):
    """Deterministic avalanche hash of a key; same key + salt always lands in the same cell."""
    _, r = splitmix64(((value + 1) * 0x9E3779B97F4A7C15 + salt * 0xC2B2AE3D27D4EB4F) & MASK64)
    return r


def exact_counts(stream):
    counts = {}
    for key in stream:
        counts[key] = counts.get(key, 0) + 1
    return counts


def zipf_stream(length, distinct, alpha, seed):
    """Deterministic Zipf(alpha) stream: weight of rank i is (i + 1) ** -alpha."""
    weights = [(i + 1) ** (-alpha) for i in range(distinct)]
    total = sum(weights)
    thresholds = []
    acc = 0.0
    for w in weights:
        acc += w
        thresholds.append(acc / total)
    stream = []
    for _ in range(length):
        seed, r = splitmix64(seed)
        u = (r % (1 << 30)) / float(1 << 30)
        lo, hi = 0, distinct - 1
        while lo < hi:                       # first rank whose cumulative mass exceeds u
            mid = (lo + hi) // 2
            if thresholds[mid] <= u:
                lo = mid + 1
            else:
                hi = mid
        stream.append(lo)
    return stream, seed


def uniform_stream(length, distinct, seed):
    stream = []
    for _ in range(length):
        seed, r = splitmix64(seed)
        stream.append(r % distinct)
    return stream, seed


class MisraGries:
    """At most k counters; estimates are lower bounds with error <= N / k."""

    def __init__(self, k):
        check(k >= 1, "Misra-Gries needs at least one counter")
        self.k = k
        self.counters = {}
        self.seen = 0
        self.rounds = 0
        self.max_size = 0

    def add(self, key):
        self.seen += 1
        if key in self.counters:
            self.counters[key] += 1
        elif len(self.counters) < self.k:
            self.counters[key] = 1
        else:
            for monitored in list(self.counters):
                self.counters[monitored] -= 1
                if self.counters[monitored] == 0:
                    del self.counters[monitored]
            self.rounds += 1
        if len(self.counters) > self.max_size:
            self.max_size = len(self.counters)
        return self.counters.get(key, 0)

    def estimate(self, key):
        return self.counters.get(key, 0)

    def monitored(self):
        return sorted(self.counters)

    def error_bound(self):
        return self.seen / self.k if self.k else 0.0


class CountMinSketch:
    """depth x width counters; estimate = min over rows, so it can only over-estimate."""

    def __init__(self, width, depth):
        check(width >= 1 and depth >= 1, "a sketch needs a positive width and depth")
        self.width = width
        self.depth = depth
        self.rows = [[0] * width for _ in range(depth)]

    def add(self, key, count=1):
        for row in range(self.depth):
            self.rows[row][mix(key, row + 1) % self.width] += count

    def estimate(self, key):
        return min(self.rows[row][mix(key, row + 1) % self.width] for row in range(self.depth))

    def merge(self, other):
        check(self.width == other.width and self.depth == other.depth,
              "only sketches with identical geometry can be merged")
        merged = CountMinSketch(self.width, self.depth)
        merged.rows = [[a + b for a, b in zip(row, other_row)]
                       for row, other_row in zip(self.rows, other.rows)]
        return merged

    def cells(self):
        return self.width * self.depth


def cms_geometry(eps, delta):
    """width = ceil(e/eps), depth = ceil(ln(1/delta)) - the standard Count-Min choice."""
    import math
    return int(math.ceil(math.e / eps)), int(math.ceil(math.log(1.0 / delta)))


def build_cms(stream, width, depth):
    sketch = CountMinSketch(width, depth)
    for key in stream:
        sketch.add(key)
    return sketch


def main():
    # 1. the exact oracle itself
    counts = exact_counts([7, 7, 3, 7, 1, 3])
    check(counts == {7: 3, 3: 2, 1: 1} and sum(counts.values()) == 6, "the frequency oracle is wrong")

    # 2. boundary: empty stream, single item, single counter
    empty = MisraGries(4)
    check(empty.estimate(1) == 0 and empty.monitored() == [] and empty.error_bound() == 0.0,
          "an empty MG stream must have no counters and no error budget")
    one = MisraGries(1)
    one.add(5)
    check(one.estimate(5) == 1 and one.monitored() == [5], "a single-item stream must be counted")
    one.add(6)
    check(one.monitored() == [] and one.estimate(5) == 0,
          "k=1: a second distinct key forces the only counter to be dropped")
    check(CountMinSketch(4, 3).estimate(99) == 0, "an unused sketch must report zero")

    # 3. generous memory: with k >= distinct keys MG degenerates to exact counting
    stream, _ = uniform_stream(300, 25, 0xA11CE)
    exact = exact_counts(stream)
    generous = MisraGries(len(set(stream)))
    for key in stream:
        generous.add(key)
    check(all(generous.estimate(key) == exact[key] for key in exact),
          "with k >= distinct keys every estimate must be exact")
    check(generous.max_size == len(set(stream)), "the counter budget must actually be used")

    # 4. the MG guarantee on a Zipf stream: heavy hitters survive, error <= N / k
    stream, seed = zipf_stream(3000, 120, 1.0, 0x5EED)
    exact = exact_counts(stream)
    total = len(stream)
    k = 16
    heavy = MisraGries(k)
    size_violations = 0
    for key in stream:
        heavy.add(key)
        if len(heavy.counters) > k:
            size_violations += 1
    check(size_violations == 0, "MG must never hold more than k counters")
    check(heavy.max_size <= k, "the observed peak counter count must respect k")
    bound = heavy.error_bound()
    true_heavy = [key for key, f in exact.items() if f > bound]
    check(len(true_heavy) >= 3, "the test stream must really contain heavy hitters")
    for key in true_heavy:
        check(key in heavy.counters, f"heavy hitter {key} must still be monitored")
        check(heavy.estimate(key) <= exact[key], "MG estimates must be lower bounds")
        check(exact[key] - heavy.estimate(key) <= bound,
              f"MG error for {key} exceeds N/k = {bound}")
    for key in heavy.monitored():
        check(heavy.estimate(key) <= exact[key], "monitored estimates must not exceed the truth")
        check(exact[key] - heavy.estimate(key) <= bound, "monitored error must respect N/k")
    for key, f in exact.items():
        if key not in heavy.counters:
            check(f <= bound, f"unmonitored key {key} has frequency {f} > N/k = {bound}")
    check(len(heavy.monitored()) <= k, "the reported heavy hitters must fit in the budget")
    top_estimate = max(heavy.monitored(), key=lambda key: heavy.estimate(key))
    check(heavy.estimate(top_estimate) == max(heavy.counters.values()),
          "the reported top key must carry the largest counter")
    check(exact[top_estimate] == max(exact.values()), "the reported top key must be the true mode")

    # 5. all-distinct stream: nobody exceeds N/k, and MG returns nothing above the budget
    distinct_stream = list(range(400))
    flat = MisraGries(8)
    for key in distinct_stream:
        flat.add(key)
    flat_bound = flat.error_bound()
    check(all(f == 1 and 1 <= flat_bound for f in exact_counts(distinct_stream).values()),
          "the all-distinct stream must have no heavy hitter")
    check(all(flat.estimate(key) <= 1 for key in distinct_stream), "estimates must stay below 1")
    for key, f in exact_counts(distinct_stream).items():
        if key not in flat.counters:
            check(f <= flat_bound, "no evicted key may have f > N/k")

    # 6. Count-Min on the same Zipf stream: one-sided error bounded by eps * N
    eps, delta = 0.01, 0.05
    width, depth = cms_geometry(eps, delta)
    sketch = build_cms(stream, width, depth)
    check(width == int(2.718281828459045 / eps) + 1 or width >= 271, "the width must follow e/eps")
    check(depth >= 3, "delta = 0.05 needs at least three rows")
    worst_error = 0
    for key, f in exact.items():
        est = sketch.estimate(key)
        check(est >= f, "Count-Min may only over-estimate")
        worst_error = max(worst_error, est - f)
    check(worst_error <= eps * total, f"Count-Min error {worst_error} exceeds eps*N = {eps * total}")
    check(sketch.cells() <= 4000, "the sketch must respect the memory budget")
    unseen = [key for key in range(200, 260) if key not in exact]
    row_ceiling = min(max(row) for row in sketch.rows)   # rigorous cap: est is a min of one cell per row
    for key in unseen:
        check(sketch.estimate(key) <= eps * total,
              "an unseen key may only report collision mass below eps*N")
        check(sketch.estimate(key) <= row_ceiling,
              "an unseen estimate must respect the smallest row maximum")
    collided = [key for key in unseen if sketch.estimate(key) > 0]
    check(len(collided) < len(unseen),
          "most never-seen keys must still estimate to zero on a sparse sketch")
    check(len(collided) > 0, "with 272 cells some collisions are expected on a 3000-item stream")

    # 6b. a deliberately tiny sketch must show real collision error, still inside eps * N
    tight_eps = 0.2
    tight_width, tight_depth = cms_geometry(tight_eps, 0.05)
    tight = build_cms(stream, tight_width, tight_depth)
    tight_worst = max(tight.estimate(key) - f for key, f in exact.items())
    check(tight_width == 14 and tight_depth == 3, "e/eps and ln(1/delta) must give a 14x3 sketch")
    check(tight_worst > 0, "a 42-cell sketch cannot be collision free on 120 distinct keys")
    check(tight_worst <= tight_eps * total,
          f"the tiny sketch error {tight_worst} exceeds eps*N = {tight_eps * total}")
    check(all(tight.estimate(key) >= f for key, f in exact.items()),
          "the tiny sketch must still over-estimate only")

    # 7. exact limit of Count-Min: one row whose hash is injective on the key set
    keys = sorted(exact)
    width_exact = None
    for candidate in range(len(keys), 40 * len(keys)):
        if len({mix(key, 1) % candidate for key in keys}) == len(keys):
            width_exact = candidate
            break
    check(width_exact is not None, "an injective single-row width must exist")
    check(len({mix(key, 1) % width_exact for key in keys}) == len(keys),
          "the chosen width must really be collision free for these keys")
    exact_sketch = build_cms(stream, width_exact, 1)
    check(all(exact_sketch.estimate(key) == f for key, f in exact.items()),
          "a single collision-free row must be exact")

    # 8. mergeability: sum of per-shard sketches equals the sketch of the concatenation
    left, right = stream[:1500], stream[1500:]
    merged = build_cms(left, width, depth).merge(build_cms(right, width, depth))
    check(merged.rows == sketch.rows, "merged Count-Min counters must match the single-pass sketch")
    check(merged.estimate(stream[0]) == sketch.estimate(stream[0]),
          "merged estimates must match the single-pass estimate")
    for key, f in exact.items():
        check(merged.estimate(key) >= f, "merged sketches must keep the one-sided guarantee")

    # 9. MG under merging shards keeps its bound on the union of shard estimates
    shard_a, shard_b = MisraGries(k), MisraGries(k)
    for key in left:
        shard_a.add(key)
    for key in right:
        shard_b.add(key)
    union_seen = shard_a.seen + shard_b.seen
    for key in true_heavy:
        est = shard_a.estimate(key) + shard_b.estimate(key)
        check(est <= exact[key], "summed shard estimates must remain lower bounds")
        check(exact[key] - est <= union_seen / k, "summed shard error must respect N/k of the union")

    # 10. determinism
    repeat = build_cms(stream, width, depth)
    check(repeat.rows == sketch.rows, "Count-Min must be deterministic")
    repeat_mg = MisraGries(k)
    for key in stream:
        repeat_mg.add(key)
    check(repeat_mg.counters == heavy.counters, "Misra-Gries must be deterministic")

    print(f"770 N={total} distinct={len(exact)} heavy={len(true_heavy)} mg_bound={bound:.1f} "
          f"mg_rounds={heavy.rounds}")
    print(f"770 cms {width}x{depth}={sketch.cells()} cells worst_error={worst_error} "
          f"eps*N={eps * total:.0f}")
    print(f"770 tight_cms {tight_width}x{tight_depth} worst_error={tight_worst} "
          f"budget={tight_eps * total:.0f}")
    print("PASS")


if __name__ == "__main__":
    main()
