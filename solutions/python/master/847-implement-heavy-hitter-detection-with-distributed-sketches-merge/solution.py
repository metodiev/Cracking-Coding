"""Task 847 - Implement heavy-hitter detection with distributed sketches, mergeable.  (Tier: Master, CTCI Beyond — System Design and Scalability)

Heavy hitters (frequent-item detection) over a stream that arrives at several shards, with
summaries that can be *merged*: the point is that combining per-shard sketches must give the
same detection quality as one sketch over the concatenated stream.  Everything is
deterministic - no threads and no wall clock, and the hash family is a fixed SHA-256 based
index map instead of a random one - so the guarantees can be measured exactly.

Two summary families are implemented:

  * `MisraGries` - the deterministic streaming counter set.  It keeps at most k counters and, on
    a collision, decrements every counter; the total weight it ever evicted is tracked, which
    gives the classic sandwich `true - evicted <= value <= true` and `true <= evicted` for any
    item not in the summary.  It is mergeable by summing the counters item-wise and truncating
    back to k, which loses the truncated weight (recorded as `merged_loss`).
  * `CountMin` - sorted-row count-min sketch: `depth` rows of `width` counters, updated at one
    independent index per row and queried by taking the minimum.  It is *exactly* mergeable
    (the merged counters equal the counters of a single sketch over the concatenation) and it
    never underestimates, so it also supports turnstile deletions.

Simplified versus production: the stream and the shards live in one process, hashing is
deterministic rather than seeded-random (so error bounds are measured, not probabilistic), and
the update schedule is a list, not a network.

Verified against exact counting on the whole universe:
  * Misra-Gries: never overestimates, the sandwich invariants hold for every item, every item
    with true count above the classic N/(k+1) threshold is in the summary, deletions are
    refused (the negative control that shows why MG cannot do turnstiles);
  * count-min: never underestimates, its measured maximum overestimate is below the predicted
    eps * N, and merging shard sketches reproduces the single-pass counters exactly;
  * distributed recall: for a phi sweep, every true heavy hitter (count >= phi * N) survives the
    merge with the full error accounting, and the merge is shown to be non-trivial (taking the
    union of per-shard summaries without summing loses hits);
  * turnstiles: deletions are handled by count-min and an item that is added and then removed
    again stops being reported;
  * space, determinism, empty/single/all-distinct streams and the per-shard error accounting.
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


def index_of(item, row, width):
    """Deterministic hash-based index; the honest stand-in for a random hash family."""
    raw = hashlib.sha256(("cm|%d|%s" % (row, item)).encode()).digest()[:8]
    return int.from_bytes(raw, "big") % width


class MisraGries:
    """Bounded counter set with eviction accounting; mergeable by item-wise summation."""

    def __init__(self, counters):
        self.size = counters
        self.counts = {}
        self.evicted = 0
        self.merged_loss = 0
        self.seen = 0

    def update(self, item, delta=1):
        if delta < 0:
            raise ValueError("Misra-Gries cannot handle deletions")
        self.seen += delta
        for _ in range(delta):
            if item in self.counts:
                self.counts[item] += 1
            elif len(self.counts) < self.size:
                self.counts[item] = 1
            else:
                for key in list(self.counts):
                    self.counts[key] -= 1
                    if self.counts[key] <= 0:
                        del self.counts[key]
                self.evicted += 1
        return self

    def value(self, item):
        return self.counts.get(item, 0)

    def heavy_set(self, threshold):
        return sorted(item for item, value in self.counts.items() if value >= threshold)

    def error_bound(self):
        return self.evicted + self.merged_loss

    def merge(self, others, size=None):
        """Sum the counters item-wise, then truncate back to `size` and record the loss."""
        limit = size if size is not None else self.size
        merged = MisraGries(limit)
        merged.seen = self.seen + sum(other.seen for other in others)
        merged.evicted = self.evicted + sum(other.evicted for other in others)
        combined = dict(self.counts)
        for other in others:
            for item, value in other.counts.items():
                combined[item] = combined.get(item, 0) + value
        ordered = sorted(combined.items(), key=lambda entry: (-entry[1], entry[0]))
        merged.counts = dict(ordered[:limit])
        merged.merged_loss = sum(value for _, value in ordered[limit:])
        return merged

    def size_of(self):
        return len(self.counts)


class CountMin:
    """Sorted-row count-min sketch; exactly mergeable and safe under deletions."""

    def __init__(self, width, depth, seed=0):
        self.width = width
        self.depth = depth
        self.seed = seed
        self.rows = [[0] * width for _ in range(depth)]
        self.total = 0
        self.updates = 0

    def update(self, item, delta=1):
        for row in range(self.depth):
            self.rows[row][index_of(item, row + self.seed, self.width)] += delta
        self.total += delta
        self.updates += 1
        return self

    def estimate(self, item):
        return min(self.rows[row][index_of(item, row + self.seed, self.width)]
                   for row in range(self.depth))

    def merge(self, others):
        merged = CountMin(self.width, self.depth, self.seed)
        merged.rows = [list(row) for row in self.rows]
        merged.total = self.total
        merged.updates = self.updates
        for other in others:
            if (other.width, other.depth, other.seed) != (self.width, self.depth, self.seed):
                raise ValueError("sketches with different geometry cannot be merged")
            for row in range(self.depth):
                for column in range(self.width):
                    merged.rows[row][column] += other.rows[row][column]
            merged.total += other.total
            merged.updates += other.updates
        return merged

    def cells(self):
        return sum(len(row) for row in self.rows)


def exact_counts(stream):
    counts = {}
    for item, delta in stream:
        counts[item] = counts.get(item, 0) + delta
    return counts


def skewed_stream(seed, length=4000, universe=200, skew=1.3):
    """A deterministic Zipf stream: item 0 carries about a quarter of the weight, the tail is thin."""
    running = 0.0
    cumulative = []
    for index in range(universe):
        running += 1.0 / ((index + 1) ** skew)
        cumulative.append(running)
    rng = Rng(seed)
    stream = []
    for _ in range(length):
        target = (rng.next() / float(1 << 64)) * running
        low, high = 0, len(cumulative) - 1
        while low < high:
            middle = (low + high) // 2
            if cumulative[middle] < target:
                low = middle + 1
            else:
                high = middle
        stream.append((low, 1))
    return stream


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    stream = skewed_stream(2024)
    exact = exact_counts(stream)
    total = sum(exact.values())
    check(total == len(stream), "every event weighs one unit")
    check(len(exact) > 100, "the stream must exercise a wide universe")

    # ---- 1. Misra-Gries invariants ---------------------------------------
    for size in (4, 16, 64):
        mg = MisraGries(size)
        for item, _ in stream:
            mg.update(item)
        check(mg.seen == total, "the summary counts every event")
        check(mg.size_of() <= size, "the summary never exceeds its budget")
        for item, count in exact.items():
            check(mg.value(item) <= count,
                  "Misra-Gries never overestimates (%s: %d vs %d)" % (item, mg.value(item),
                                                                      count))
            check(count - mg.value(item) <= mg.evicted,
                  "the sandwich invariant must hold for %s" % item)
            if mg.value(item) == 0:
                check(count <= mg.evicted, "an unseen item cannot exceed the evicted weight")
        misses = [item for item, count in exact.items()
                  if count > total / (size + 1) and mg.value(item) == 0]
        check(not misses, "every item above N/(k+1) must be summarised (size %d)" % size)
        phi = 1.0 / (size + 1)
        true_heavy = {item for item, count in exact.items() if count >= phi * total}
        check(true_heavy, "a skewed stream must have heavy hitters at size %d" % size)
        check(true_heavy <= set(mg.counts),
              "every true heavy hitter must be present at size %d" % size)

    # ---- 2. the classic threshold, measured ------------------------------
    for size in (8, 32):
        mg = MisraGries(size)
        for item, _ in stream:
            mg.update(item)
        threshold = total / (size + 1)
        true_heavy = {item for item, count in exact.items() if count > threshold}
        reported = set(mg.heavy_set(1))
        check(true_heavy <= reported,
              "size %d: every item above N/(k+1) must be reported" % size)
        check(len(true_heavy) <= size, "the number of such items is bounded by k")
        phi = 1.0 / (size + 1)
        phi_heavy = {item for item, count in exact.items() if count >= phi * total}
        check(phi_heavy <= reported,
              "size %d: items above phi*N must be reported" % size)

    # ---- 3. deletions are refused (negative control) ---------------------
    try:
        MisraGries(8).update("x", -1)
        raise AssertionError("Misra-Gries must refuse a deletion")
    except ValueError:
        pass

    # ---- 4. count-min guarantees -----------------------------------------
    epsilon = 0.1
    width = int(2.718281828459045 / epsilon) + 1
    sketch = CountMin(width, 4)
    for item, delta in stream:
        sketch.update(item, delta)
    check(sketch.cells() == width * 4, "the sketch keeps exactly width * depth counters")
    check(sketch.total == total, "the sketch tracks the stream weight")
    overestimates = []
    for item, count in exact.items():
        estimate = sketch.estimate(item)
        check(estimate >= count, "count-min never underestimates (%s)" % item)
        overestimates.append(estimate - count)
    measured = max(overestimates)
    check(measured <= epsilon * total,
          "the measured error must respect eps * N (%.1f vs %.1f)" % (measured,
                                                                      epsilon * total))
    check(sum(1 for value in overestimates if value > 0) > 0,
          "collisions must actually happen, or the error check is vacuous")
    check(measured > 0, "a tiny sketch must have a non-zero measured error")

    # ---- 5. exact mergeability of count-min ------------------------------
    shards = [[], [], [], []]
    for position, entry in enumerate(stream):
        shards[position % 4].append(entry)
    sketches = []
    for shard in shards:
        partial = CountMin(width, 4)
        for item, delta in shard:
            partial.update(item, delta)
        sketches.append(partial)
    merged = sketches[0].merge(sketches[1:])
    single = CountMin(width, 4)
    for item, delta in stream:
        single.update(item, delta)
    check(merged.rows == single.rows,
          "merging count-min sketches must equal one sketch over the concatenated stream")
    check(merged.total == single.total == total, "the merged weight must match")
    check(all(merged.estimate(item) == single.estimate(item) for item in exact),
          "every estimate must survive the merge unchanged")

    # ---- 6. distributed heavy hitters with the error accounting ----------
    def phi_sweep(phi):
        size = int(1.0 / phi) + 1
        summaries = []
        for shard in shards:
            partial = MisraGries(size)
            for item, _ in shard:
                partial.update(item)
            summaries.append(partial)
        combined = summaries[0].merge(summaries[1:], size=size)
        bound = combined.error_bound()
        truth = {item for item, count in exact.items() if count >= phi * total}
        reported = {item for item, value in combined.counts.items()
                    if value + bound >= phi * total}
        missed = truth - reported
        return size, bound, reported, missed

    for phi in (0.2, 0.1, 0.05, 0.02, 0.01):
        size, bound, reported, missed = phi_sweep(phi)
        check(not missed,
              "phi=%.2f: heavy hitters must survive the merge (missed %r)" % (phi, missed))
        check(len(reported) < len(exact),
              "the report must be a small subset of the universe at phi=%.2f" % phi)
        check(bound < total * phi * 10,
              "the accumulated error must stay comparable to phi*N at phi=%.2f" % phi)

    # merge error accounting: the merged loss is the truncated weight, never negative
    size = 20
    summaries = []
    for shard in shards:
        partial = MisraGries(size)
        for item, _ in shard:
            partial.update(item)
        summaries.append(partial)
    combined = summaries[0].merge(summaries[1:], size=size)
    check(combined.merged_loss >= 0 and combined.size_of() <= size,
          "the merged summary respects its budget and records its loss")
    check(combined.seen == total, "the merged summary keeps the total weight")
    check(combined.evicted == sum(summary.evicted for summary in summaries),
          "per-shard eviction counts add up")

    # the merge is not trivial: applying the global threshold to each shard on its own, without
    # summing the values, misses every item whose weight is spread across the shards
    def union_only(phi):
        size_inner = int(1.0 / phi) + 1
        reported = set()
        for shard in shards:
            partial = MisraGries(size_inner)
            for item, _ in shard:
                partial.update(item)
            reported.update(item for item, value in partial.counts.items()
                            if value >= phi * total)
        truth = {item for item, count in exact.items() if count >= phi * total}
        return truth - reported

    lost = union_only(0.05)
    check(lost, "not summing the shard values must miss at least one heavy hitter")
    check(len(lost) >= 2, "the naive per-shard threshold must miss several hitters (%d)"
          % len(lost))

    # ---- 7. turnstiles with count-min -------------------------------------
    turnstile = CountMin(width, 4)
    for item, delta in stream:
        turnstile.update(item, delta)
    hottest = max(exact, key=lambda item: exact[item])
    for _ in range(exact[hottest]):
        turnstile.update(hottest, -1)
    check(turnstile.estimate(hottest) >= 0, "estimates never go below the net true count")
    remaining = {(item, count - (exact[hottest] if item == hottest else 0))
                 for item, count in exact.items()}
    for item, count in remaining:
        check(turnstile.estimate(item) >= count,
              "the turnstile sketch must not underestimate %s" % item)
    check(0 <= turnstile.estimate(hottest) <= epsilon * total,
          "a fully deleted item's estimate falls back to the collision level, never below zero")
    check(turnstile.total == total - exact[hottest],
          "the tracked weight follows the deletions exactly")

    # ---- 8. degenerate streams, space and determinism ---------------------
    empty = MisraGries(8)
    check(empty.size_of() == 0 and empty.evicted == 0, "an empty stream leaves an empty summary")
    single = MisraGries(4)
    single.update("only")
    check(single.counts == {"only": 1}, "a single event is summarised exactly")
    distinct = MisraGries(4)
    for index in range(50):
        distinct.update("item-%d" % index)
    check(distinct.size_of() <= 4 and distinct.evicted >= 10,
          "an all-distinct stream evicts the counters and keeps only recent items")
    check(all(value == 1 for value in distinct.counts.values()),
          "no item of an all-distinct stream can be counted twice")
    tiny = CountMin(4, 2)
    tiny.update("a", 3)
    tiny.update("a", -1)
    check(tiny.estimate("a") == 2 and tiny.total == 2, "count-min handles signed updates")
    check(CountMin(4, 2).rows == CountMin(4, 2).rows, "sketches start empty and equal")
    try:
        CountMin(4, 2).merge([CountMin(8, 2)])
        raise AssertionError("merging skatches of different geometry must fail")
    except ValueError:
        pass
    left = MisraGries(10)
    right = MisraGries(10)
    for item, _ in stream[:500]:
        left.update(item)
    for item, _ in stream[500:]:
        right.update(item)
    check(left.merge([right]).counts == left.merge([right]).counts,
          "merging must be reproducible")
    check(len(stream) == 4000 and index_of("x", 0, 97) == index_of("x", 0, 97),
          "the hash index map is deterministic")
    check(index_of("x", 0, 97) != index_of("y", 0, 97) or 97 < 2,
          "different items usually land in different cells")

    print("stream: %d events, %d distinct items, %d units" % (len(stream), len(exact), total))
    print("misra-gries: k=%d keeps %d counters, evicted %d, reported %d hitters"
          % (20, combined.size_of(), combined.evicted, len(combined.counts)))
    print("count-min: %dx%d cells, measured max error %.0f (eps*N = %.0f), merge exact"
          % (width, 4, measured, epsilon * total))
    print("PASS")


if __name__ == "__main__":
    main()
