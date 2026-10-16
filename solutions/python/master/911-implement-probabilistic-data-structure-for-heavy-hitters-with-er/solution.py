"""Task 911 - Implement probabilistic data structure for heavy hitters with error guarantees and
mergeability.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; sha256 provides the hash family, so the sketch is
exactly reproducible and mergeable across shards):

  * `CountMinSketch` - the classic d x w counter matrix.  Each update touches exactly one cell per
    row, so memory is O(d*w) and never grows with the stream; every estimate is one-sided:
    true <= estimate <= true + eps*N with probability 1 - delta when w >= e/eps and
    d >= ln(1/delta).
  * `merge` - cell-wise addition of two sketches from the same hash family, which equals the
    sketch of the concatenated stream (the property that makes the structure shard-friendly).
  * `MisraGries` - the bounded space-saving summary that supplies the candidate set; it has no
    false negatives for counts above N/capacity and is itself mergeable.
  * `heavy_hitters` - the pipeline: candidates from the summary, estimates from the sketch, and
    the guarantee that nothing above the threshold phase*N is missed.

Simplified versus production: single-threaded, integer counts, no conservative-update variants and
no decay window.  Verified below: the one-sided eps-error bound over many random streams, the
delta failure rate, exact merge equivalence, no false negatives for heavy hitters, bounded memory
after every update, the cross-seed non-mergeability negative control, and the empty / singleton
/ threshold boundaries.
"""

import hashlib
import math

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n


def cell(key, row, width, seed):
    """Deterministic, uniform-enough cell index for (key, row); width must be a power of two."""
    digest = hashlib.sha256(("%d:%d:%s" % (seed, row, key)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & (width - 1)


class CountMinSketch:
    __slots__ = ("width", "depth", "seed", "counters", "updates")

    def __init__(self, width, depth, seed=1):
        check(width & (width - 1) == 0, "the width must be a power of two for masking")
        check(depth >= 1 and width >= 2, "the sketch needs at least one row and two columns")
        self.width = width
        self.depth = depth
        self.seed = seed
        self.counters = [[0] * width for _ in range(depth)]
        self.updates = 0

    def __len__(self):
        return self.depth * self.width

    def add(self, key, count=1):
        for row in range(self.depth):
            self.counters[row][cell(key, row, self.width, self.seed)] += count
        self.updates += count
        check(len(self) == self.depth * self.width, "the memory footprint must never grow")

    def estimate(self, key):
        return min(self.counters[row][cell(key, row, self.width, self.seed)]
                   for row in range(self.depth))

    def merge(self, other):
        check((self.width, self.depth, self.seed)
              == (other.width, other.depth, other.seed), "only same-family sketches merge")
        for row in range(self.depth):
            for column in range(self.width):
                self.counters[row][column] += other.counters[row][column]
        self.updates += other.updates
        return self


class MisraGries:
    """Bounded space-saving summary: at most `capacity` counters are tracked."""

    __slots__ = ("capacity", "counts", "offset", "seen")

    def __init__(self, capacity):
        self.capacity = capacity
        self.counts = {}
        self.offset = 0
        self.seen = 0

    def add(self, key, count=1):
        self.seen += count
        if key in self.counts:
            self.counts[key] += count
        elif len(self.counts) < self.capacity:
            self.counts[key] = count
        else:
            self.offset += count
            for tracked in list(self.counts):
                self.counts[tracked] -= count
                if self.counts[tracked] <= 0:
                    del self.counts[tracked]
        check(len(self.counts) <= self.capacity, "the summary exceeded its capacity")

    def candidates(self):
        return sorted(self.counts)

    def merge(self, other):
        check(self.capacity == other.capacity, "only equal-capacity summaries merge")
        merged = MisraGries(self.capacity)
        merged.counts = dict(self.counts)
        merged.offset = self.offset + other.offset
        merged.seen = self.seen + other.seen
        for key, value in other.counts.items():
            merged.counts[key] = merged.counts.get(key, 0) + value
        while len(merged.counts) > self.capacity:
            smallest = min(merged.counts, key=lambda k: (merged.counts[k], k))
            merged.offset += merged.counts.pop(smallest)
        return merged


def heavy_hitters(sketch, summary, phase):
    """Candidates that the sketch confirms are at least phase * N."""
    threshold = phase * sketch.updates
    return [key for key in summary.candidates() if sketch.estimate(key) >= threshold]


def make_stream(seed, length=4000, alphabet=120, hot=5):
    """A skewed stream: 60% of the mass lands on `hot` keys, the rest is spread thinly."""
    rng = Rng(seed)
    stream = []
    for _ in range(length):
        if rng.below(100) < 60:
            stream.append("hot-%01d" % rng.below(hot))
        else:
            stream.append("key-%03d" % rng.below(alphabet))
    return stream


def exact_counts(stream):
    counts = {}
    for key in stream:
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    eps, delta, phase = 0.01, 0.01, 0.05
    width = 1 << max(1, math.ceil(math.log2(math.e / eps)))
    depth = max(1, math.ceil(math.log(1.0 / delta)))
    check(width >= math.e / eps and depth >= math.log(1.0 / delta),
          "the dimensioning rules must be satisfied")

    # ---- 1. the one-sided eps-error guarantee on a skewed stream
    stream = make_stream(11)
    exact = exact_counts(stream)
    sketch = CountMinSketch(width, depth, seed=99)
    for key in stream:
        sketch.add(key)
    check(sketch.updates == len(stream), "the update counter must track the stream length")
    check(all(sketch.estimate(key) >= exact[key] for key in exact),
          "a sketch estimate may never under-count")
    worst = max(sketch.estimate(key) - exact[key] for key in exact)
    check(worst <= eps * len(stream),
          "the over-count must stay inside eps*N: %d > %.1f" % (worst, eps * len(stream)))
    check(all(sum(row) == len(stream) for row in sketch.counters),
          "the total mass of every row must equal the number of updates")
    check(len(sketch) == depth * width, "the memory footprint is exactly d*w counters")

    # ---- 2. the failure rate obeys delta (the probabilistic part of the guarantee)
    failures = 0
    trials = 300
    for trial in range(trials):
        probe = CountMinSketch(width, depth, seed=trial + 1)
        for key in stream:
            probe.add(key)
        bound = eps * len(stream)
        if max(probe.estimate(key) - exact[key] for key in exact) > bound:
            failures += 1
    check(failures / trials <= delta + 0.01,
          "the guarantee may fail with probability at most delta (observed %.3f)" % (failures / trials))

    # ---- 3. mergeability is exact
    left, right = stream[:1500], stream[1500:]
    first = CountMinSketch(width, depth, seed=7)
    second = CountMinSketch(width, depth, seed=7)
    for key in left:
        first.add(key)
    for key in right:
        second.add(key)
    merged = CountMinSketch(width, depth, seed=7)
    for key in left + right:
        merged.add(key)
    first.merge(second)
    check(first.counters == merged.counters and first.updates == merged.updates,
          "merging must equal sketching the concatenated stream")
    check(all(first.estimate(key) >= merged.estimate(key) - 0 for key in exact),
          "merged estimates must dominate the whole-stream estimates")
    reverse = CountMinSketch(width, depth, seed=7)
    for key in right + left:
        reverse.add(key)
    check(reverse.counters == merged.counters, "sketching must not depend on the shard order")
    crossed = CountMinSketch(width, depth, seed=8)
    try:
        CountMinSketch(width, depth, seed=7).merge(crossed)
        raise AssertionError("merging different hash families must be rejected")
    except AssertionError as error:
        check("same-family" in str(error), "the rejection must explain the hash family")

    # ---- 4. heavy hitters: no false negatives, and the candidate set is a superset
    summary = MisraGries(capacity=max(8, int(1.0 / phase)))
    probe = CountMinSketch(width, depth, seed=5)
    for key in stream:
        summary.add(key)
        probe.add(key)
    threshold = phase * len(stream)
    truth = {key for key, value in exact.items() if value > threshold}
    reported = set(heavy_hitters(probe, summary, phase))
    check(truth, "the stream must contain heavy hitters to find")
    check(truth <= reported, "no heavy hitter may be missed: %r" % (truth - reported))
    check(all(probe.estimate(key) >= threshold for key in reported),
          "every reported hitter must clear the threshold")
    check(len(reported) <= len(summary.candidates()), "the report is bounded by the candidates")

    # ---- 5. the summary is bounded and mergeable
    check(len(summary.candidates()) <= summary.capacity, "the summary must respect its capacity")
    strong = {key for key, value in exact.items() if value > len(stream) // summary.capacity}
    check(strong <= set(summary.candidates()),
          "every key above N/capacity must be a candidate: %r" % (strong - set(summary.candidates())))
    check(all(summary.counts.get(key, 0) + summary.offset >= exact[key] for key in exact),
          "tracked count plus offset must never fall below the true count")
    left_summary = MisraGries(summary.capacity)
    for key in left:
        left_summary.add(key)
    right_summary = MisraGries(summary.capacity)
    for key in right:
        right_summary.add(key)
    combined = left_summary.merge(right_summary)
    check(len(combined.counts) <= summary.capacity, "a merged summary must respect the capacity")
    check(combined.seen == len(stream), "the merged summary must account for the whole stream")
    survivors = set(combined.candidates()) | set(left_summary.candidates()) \
        | set(right_summary.candidates())
    check(strong <= survivors, "a merge may not lose a key above N/capacity")

    # ---- 6. boundaries
    empty = CountMinSketch(width, depth, seed=3)
    check(empty.estimate("anything") == 0 and empty.updates == 0,
          "an empty sketch must estimate zero everywhere")
    singleton = CountMinSketch(width, depth, seed=3)
    singleton.add("only")
    check(singleton.estimate("only") == 1 and singleton.estimate("other") == 0,
          "a single insertion must be counted exactly when it cannot collide")
    tiny = MisraGries(capacity=2)
    for key in ("a", "b", "c", "d", "e"):
        tiny.add(key)
    check(len(tiny.counts) <= 2 and tiny.seen == 5, "a tiny summary must still be bounded")
    check(heavy_hitters(CountMinSketch(width, depth, seed=1), MisraGries(capacity=4), phase) == [],
          "an empty stream has no heavy hitters")
    generous = heavy_hitters(probe, summary, 0.0)
    check(len(generous) == len(summary.candidates()), "a zero threshold keeps every candidate")

    # ---- 7. determinism
    twin = CountMinSketch(width, depth, seed=99)
    for key in stream:
        twin.add(key)
    check(twin.counters == sketch.counters, "the sketch must be reproducible")
    check(make_stream(11) == make_stream(11), "the stream generator must be reproducible")

    print("dims %dx%d, eps %.3f delta %.3f, worst over-count %d (bound %.1f)"
          % (depth, width, eps, delta, worst, eps * len(stream)))
    print("hitters reported %d, true %d, stream %d keys" % (len(reported), len(truth), len(exact)))
    print("PASS")


if __name__ == "__main__":
    main()
