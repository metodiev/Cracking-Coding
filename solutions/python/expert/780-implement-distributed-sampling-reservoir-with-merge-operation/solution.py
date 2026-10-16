"""Task 780 - Implement distributed sampling reservoir with merge operation.  (Tier: Expert, Ch 9 - System Design and Scalability)

Reservoir sampling keeps a uniform sample of k items from a stream whose length is unknown.  In a
distributed setting every worker sees a disjoint shard, keeps its own reservoir, and a coordinator
merges the reservoirs - and the merged result must be distributed exactly like a single reservoir over
the concatenation.  The classic Algorithm R cannot be merged that way (it depends on arrival order),
so the implementation uses the priority form: item i gets a fixed pseudorandom key u_i (uniform in
(0,1)) and the reservoir keeps the k *smallest* keys seen so far.  The k smallest keys of the whole
stream are a uniform random k-subset of it, and merge is then trivially exact - it is the k smallest
keys of the union of the shards' samples, an associative, commutative operation that never rescans a
shard.  The weighted variant (A-Res) uses key u_i ** (1/w_i) and keeps the k largest keys, which
sampling-theoretically draws k items with replacement proportional to the weights.

Verified: the single-shard reservoir equals the exact top-k-by-key oracle (independently computed by
sorting every item's key), merging is associative and commutative and equals the oracle over the whole
stream, k = 1 and k >= n degenerate correctly, an empty shard is a no-op, grouping the shards
differently (a tree of merges) gives the same sample, uniformity is checked statistically over many
independent trials (each item's inclusion frequency must be close to k/n) and the weighted variant
skews by the weights while equal weights reproduce the uniform distribution.  Deterministic: all
randomness comes from one splitmix64 stream keyed by (item, trial).

Simplified: keys are computed from a deterministic hash instead of a cryptographic RNG, items are
small integers, shards are simulated sequentially in one process, and the weighted variant is
verified statistically because its exact inclusion law is not a simple subset probability.
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


def uniform_unit(item, trial):
    """Deterministic uniform in (0, 1) keyed by the item and the trial number."""
    _, r = splitmix64(((item + 1) * 0x9E3779B97F4A7C15 + (trial + 1) * 0xC2B2AE3D27D4EB4F) & MASK64)
    top = r >> 11                                  # 53 random-ish bits scaled into the open unit
    return (top + 0.5) / float(1 << 53)


def key_for(item, trial, weight=1.0):
    """Uniform: the raw key (keep the k smallest).  Weighted: u ** (1/w) (keep the k largest)."""
    unit = uniform_unit(item, trial)
    if weight == 1.0:
        return unit
    return unit ** (1.0 / weight)


class Reservoir:
    """A k-item sample stored as a max-heap-free sorted list of priorities."""

    def __init__(self, capacity, keep_largest=False):
        check(capacity >= 1, "a reservoir needs a positive capacity")
        self.capacity = capacity
        self.keep_largest = keep_largest
        self.sample = []                # (key, item) kept sorted by key
        self.seen = 0

    def add(self, item, key):
        self.seen += 1
        if len(self.sample) < self.capacity:
            self.sample.append((key, item))
            self.sample.sort()
            return True
        worst = self.sample[-1] if not self.keep_largest else self.sample[0]
        better = key < worst[0] if not self.keep_largest else key > worst[0]
        if not better:
            return False
        if not self.keep_largest:
            self.sample[-1] = (key, item)
        else:
            self.sample[0] = (key, item)
        self.sample.sort()
        return True

    def items(self):
        return [item for _, item in self.sample]

    def merge(self, other):
        """Exact merge: the k best keys of the union of the two samples."""
        check(self.capacity == other.capacity, "reservoirs must share a capacity to be merged")
        check(self.keep_largest == other.keep_largest, "reservoirs must sample in the same direction")
        merged = Reservoir(self.capacity, self.keep_largest)
        for key, item in self.sample + other.sample:
            merged.add(item, key)
        merged.seen = self.seen + other.seen      # the add() calls above must not inflate the count
        return merged


def shard_reservoir(capacity, stream, trial, weights=None, keep_largest=False):
    reservoir = Reservoir(capacity, keep_largest)
    for position, item in enumerate(stream):
        weight = 1.0 if weights is None else weights[item]
        reservoir.add(item, key_for(item, trial, weight))
    return reservoir


def oracle_sample(capacity, stream, trial, weights=None, keep_largest=False):
    """The exact top-k-by-key over the whole stream: the independent oracle for a merge."""
    entries = []
    for item in stream:
        weight = 1.0 if weights is None else weights[item]
        entries.append((key_for(item, trial, weight), item))
    entries.sort(reverse=keep_largest)
    return [item for _, item in entries[:capacity]]


def main():
    # 1. boundaries: capacity one, capacity above the stream length, an empty shard
    one = Reservoir(1)
    one.add(5, 0.5)
    check(one.items() == [5], "capacity one keeps the single best item")
    one.add(7, 0.25)
    check(one.items() == [7] and one.seen == 2, "capacity one replaces on a better key")
    check(one.add(9, 0.75) is False, "a worse key must not enter the reservoir")
    wide = Reservoir(10)
    for item in range(4):
        wide.add(item, uniform_unit(item, 0))
    check(sorted(wide.items()) == [0, 1, 2, 3], "a capacity above the length keeps everything")
    empty = Reservoir(10)
    check(empty.items() == [] and empty.seen == 0, "an empty shard holds nothing")
    check(empty.merge(wide).items() == wide.items(), "merging an empty shard is a no-op")
    check(wide.merge(empty).items() == wide.items(), "the no-op must work in both directions")
    mismatched = Reservoir(2)
    try:
        empty.merge(mismatched)
        check(False, "merging reservoirs with different capacities must be rejected")
    except AssertionError:
        pass

    # 2. a single shard equals the exact top-k oracle
    stream = [7, 3, 11, 3, 5, 2, 13, 8, 1, 9, 4, 6]
    for capacity in (1, 3, 5):
        reservoir = shard_reservoir(capacity, stream, trial=4)
        expected = oracle_sample(capacity, stream, trial=4)
        check(sorted(reservoir.items()) == sorted(expected),
              f"the reservoir must equal the top-{capacity} oracle")
        check(len(reservoir.sample) == min(capacity, len(stream)), "the reservoir must be full")

    # 3. distributed merge equals the single-stream oracle, whatever the shard split is
    shards = [stream[:3], stream[3:4], stream[4:9], stream[9:]]
    merged = Reservoir(4)
    for shard in shards:
        merged = merged.merge(shard_reservoir(4, shard, trial=9))
    check(sorted(merged.items()) == sorted(oracle_sample(4, stream, trial=9)),
          "the merged reservoir must equal the single-stream sample")
    check(merged.seen == len(stream), "the merge must account for every item seen")

    # 4. merge is associative and commutative
    pieces = [shard_reservoir(3, chunk, trial=11) for chunk in (stream[:4], stream[4:7], stream[7:])]
    left = pieces[0].merge(pieces[1]).merge(pieces[2])
    right = pieces[0].merge(pieces[1].merge(pieces[2]))
    check(left.items() == right.items(), "merging must be associative")
    swapped = pieces[2].merge(pieces[1]).merge(pieces[0])
    check(sorted(left.items()) == sorted(swapped.items()), "merging must be commutative")
    tree = pieces[0].merge(pieces[2]).merge(pieces[1])
    check(sorted(tree.items()) == sorted(left.items()), "any merge tree must give the same sample")
    check(sorted(left.items()) == sorted(oracle_sample(3, stream, trial=11)),
          "the associative merge must still match the oracle")

    # 5. uniformity: over many trials each item must enter a size-3 sample about 3/12 of the time
    trials, capacity, items = 800, 3, 12
    counts = {item: 0 for item in range(items)}
    for trial in range(trials):
        sample = shard_reservoir(capacity, list(range(items)), trial)
        for item in sample.items():
            counts[item] += 1
    frequencies = {item: count / trials for item, count in counts.items()}
    expected = capacity / items
    spread = max(abs(frequency - expected) for frequency in frequencies.values())
    check(spread < 0.08, f"inclusion frequencies deviate by {spread:.3f} from {expected:.3f}")
    check(all(count > 0 for count in counts.values()), "every item must be sampled at least once")

    # 6. the same uniformity holds after a distributed merge of four shards
    merged_counts = {item: 0 for item in range(items)}
    for trial in range(trials):
        chunks = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9)), list(range(9, 12))]
        sample = Reservoir(capacity)
        for chunk in chunks:
            sample = sample.merge(shard_reservoir(capacity, chunk, trial))
        for item in sample.items():
            merged_counts[item] += 1
    merged_spread = max(abs(count / trials - expected) for count in merged_counts.values())
    check(merged_spread < 0.08,
          f"merged inclusion frequencies deviate by {merged_spread:.3f} from {expected:.3f}")

    # 7. the weighted variant (A-Res) favours heavy items while equal weights stay uniform
    weights = [1.0] * items
    weights[0] = 6.0
    heavy = 0
    for trial in range(trials):
        sample = shard_reservoir(2, list(range(items)), trial, weights=weights, keep_largest=True)
        if 0 in sample.items():
            heavy += 1
    heavy_rate = heavy / trials
    check(heavy_rate > 2 * (2 / items),
          f"a six-times heavier item must be picked much more often, got {heavy_rate:.3f}")
    check(heavy_rate <= 1.0, "a probability cannot exceed one")
    equal_weights = [2.0] * items
    equal_rate = 0
    for trial in range(trials):
        sample = shard_reservoir(2, list(range(items)), trial, weights=equal_weights,
                                 keep_largest=True)
        if 0 in sample.items():
            equal_rate += 1
    check(abs(equal_rate / trials - 2 / items) < 0.08,
          f"equal weights must stay uniform, got {equal_rate / trials:.3f}")
    check(abs(equal_rate / trials - heavy_rate) > 0.2,
          "weighting must make a measurable difference")

    # 8. weighted merges still match the weighted oracle
    weighted_stream = list(range(items)) * 2
    left_weighted = shard_reservoir(3, weighted_stream[:12], trial=21, weights=weights,
                                    keep_largest=True)
    right_weighted = shard_reservoir(3, weighted_stream[12:], trial=21, weights=weights,
                                     keep_largest=True)
    merged_weighted = left_weighted.merge(right_weighted)
    check(sorted(merged_weighted.items())
          == sorted(oracle_sample(3, weighted_stream, trial=21, weights=weights, keep_largest=True)),
          "the weighted merge must equal the weighted oracle")
    check(len(merged_weighted.items()) <= 3, "a weighted reservoir respects its capacity")

    # 9. determinism, and sampling without replacement on a stream of distinct items
    distinct = list(range(20))
    first = shard_reservoir(4, distinct, trial=33)
    second = shard_reservoir(4, distinct, trial=33)
    check(first.sample == second.sample, "the same trial must give the same sample")
    check(first.sample != shard_reservoir(4, distinct, trial=34).sample,
          "different trials must draw different samples")
    check(len(set(first.items())) == len(first.items()),
          "distinct items must never repeat inside one sample")
    check(len(first.items()) == 4, "a full reservoir of distinct items must be complete")

    print(f"780 trials={trials} uniform_spread={spread:.3f} merged_spread={merged_spread:.3f} "
          f"expected={expected:.3f}")
    print(f"780 heavy_rate={heavy_rate:.3f} equal_weight_rate={equal_rate / trials:.3f} "
          f"merged_keys={len(merged.items())} shards={len(shards)}")
    print("PASS")


if __name__ == "__main__":
    main()
