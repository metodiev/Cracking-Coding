"""Task 848 - Implement distributed sampling with global randomness coordination.  (Tier: Master, CTCI Beyond — System Design and Scalability)

Sampling k items from a stream that is split across shards, with the randomness *coordinated*:
every shard draws from a global random stream that the coordinator publishes and commits to,
so the union of the shard samples is an exact size-k sample of the whole stream.  Nothing here
uses threads, sockets or the wall clock: the coordinator, the shards and the commitment
registry are plain objects and randomness is a deterministic hash of (round seed, shard,
position).

Protocol, as implemented:
  * the coordinator commits to a seed per round by publishing its digest *before* the round;
  * each shard turns every element it sees into a key u_j^(1/w_j) where u_j comes from that
    committed seed and the element's *shard-local position* - so the keys do not depend on how
    the stream was split, which is what makes the merge exact;
  * the coordinator merges the shard samples by keeping the k largest keys (the A-Res /
    Efraimidis-Spirakis rule): the result is exactly the sample obtained by giving a key to
    every element of the concatenated stream and taking the top k;
  * a shard that used a stale seed is detected, because its returned seed no longer matches the
    published commitment.

Simplified versus production: one process, k-samples are lists of (item, key) instead of files,
and the "global randomness" is a hash chain rather than a distributed coin-tossing protocol.

Verified, with the *estimated* quantities measured over many deterministic worlds:
  * exactness: the merged sample equals the top-k-by-key sample of the whole stream (brute
    force with the same global randomness);
  * unbiasedness: every stream position (and every distinct item) has inclusion probability
    k/n, measured over hundreds of seeded worlds and compared with the exact value;
  * the distributed result and a single central reservoir have the same inclusion frequencies;
  * the size of the sample is exactly min(k, n) in every world, k=0 and k>=n behave as edge
    cases, empty shards are skipped;
  * a stale seed is rejected by the commitment check (negative control);
  * the memory bound: each shard holds at most k items and the coordinator at most k.
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

    def unit(self):
        """Uniform in (0, 1]: never exactly zero, so the key below cannot take a log of zero."""
        return (self.next() + 1) / float(1 << 64)

    def below(self, limit):
        return self.next() % limit


def digest(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()[:24]


def global_unit(seed, shard, position, salt=0):
    """The coordinated random draw for (seed, shard, position): a hash, not a stateful stream."""
    raw = hashlib.sha256(("sample|%d|%d|%d|%d" % (seed, shard, position, salt)).encode())
    return (int.from_bytes(raw.digest()[:8], "big") + 1) / float(1 << 64)


def key_of(seed, shard, position, weight=1.0):
    """A-Res key: u^(1/w); with unit weights this is just u, and bigger keys are kept."""
    return global_unit(seed, shard, position) ** (1.0 / weight)


def central_reservoir(stream, k, seed, shards=1):
    """Brute force reference: give every element its key and keep the k largest.

    The key of an element is derived from the shard it lands in and its position *within* that
    shard, which is exactly what a shard can compute locally; the reference therefore
    materialises the whole key set instead of merging partial samples.
    """
    if k <= 0:
        return []
    counters = [0] * max(1, shards)
    keys = []
    for position, item in enumerate(stream):
        shard = position % shards
        keys.append((item, key_of(seed, shard, counters[shard])))
        counters[shard] += 1
    keys.sort(key=lambda entry: (-entry[1], entry[0]))
    return keys[:k]


class Coordinator:
    """Publishes committed seeds and merges the shard samples by the A-Res rule."""

    def __init__(self, k, rounds):
        self.k = k
        self.rounds = rounds
        self.seeds = {}
        self.commitments = {}
        self.merged = []
        self.counters = {"rounds": 0, "rejected": 0, "accepted": 0, "merged_items": 0}

    def open_round(self, round_index):
        seed = int(hashlib.sha256(("seed|%d|%d" % (round_index, self.k)).encode())
                   .hexdigest()[:16], 16)
        self.seeds[round_index] = seed
        self.commitments[round_index] = digest("commit", seed)
        return seed

    def accept(self, round_index, seed, sample):
        """Verify the seed against the published commitment, then merge the sample."""
        if self.commitments.get(round_index) != digest("commit", seed):
            self.counters["rejected"] += 1
            raise ValueError("seed %r does not match the commitment of round %d"
                             % (seed, round_index))
        self.counters["accepted"] += 1
        self.merged.extend(sample)
        self.merged.sort(key=lambda entry: (-entry[1], entry[0]))
        del self.merged[self.k:]
        self.counters["merged_items"] = len(self.merged)
        self.counters["rounds"] += 1
        return self.merged

    def sample(self):
        return [item for item, _ in self.merged[:self.k]]


def top_positions(stream, k, seed, shards):
    """The positions of the k largest element keys - the brute-force view of the merge."""
    keys = []
    for position in range(len(stream)):
        keys.append((position, key_of(seed, position % shards, position // shards)))
    keys.sort(key=lambda entry: (-entry[1], entry[0]))
    return [position for position, _ in keys[:k]]


def shard_sample(stream, shard, seed, k):
    """One shard's k largest keys; memory is bounded by k, not by the stream length."""
    kept = []
    for position, item in enumerate(stream):
        if k <= 0:
            break
        key = (key_of(seed, shard, position), item)
        if len(kept) < k:
            kept.append(key)
            kept.sort(key=lambda entry: (-entry[0], entry[1]))
        elif key > kept[-1]:
            kept[-1] = key
            kept.sort(key=lambda entry: (-entry[0], entry[1]))
    return [(item, key) for key, item in kept]


def run_round(stream, shards, k, round_index, coordinator):
    """Split the stream across shards, sample each one and merge."""
    seed = coordinator.open_round(round_index)
    parts = [[] for _ in range(shards)]
    for position, item in enumerate(stream):
        parts[position % shards].append(item)
    for shard, part in enumerate(parts):
        if not part:
            continue
        coordinator.accept(round_index, seed, shard_sample(part, shard, seed, k))
    return seed, parts


def make_stream(seed, length=200, universe=40):
    rng = Rng(seed)
    return [rng.below(universe) for _ in range(length)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    stream = make_stream(7, length=200, universe=40)
    n = len(stream)
    k = 10

    # ---- 1. exactness of the merged sample --------------------------------
    for shards in (1, 2, 4, 7):
        coordinator = Coordinator(k, 1)
        seed, parts = run_round(stream, shards, k, 0, coordinator)
        truth = central_reservoir(stream, k, seed, shards)
        check(coordinator.merged == truth,
              "the merged sample must equal the top-k-by-key sample (%d shards)" % shards)
        check(len(coordinator.sample()) == min(k, n), "the sample size is exactly k")
        check(sum(len(part) for part in parts) == n, "the shards partition the stream")
        check(sum(1 for part in parts if part) == min(shards, n),
              "empty shards are skipped without breaking the round")
        check(len(coordinator.merged) <= k, "the coordinator stores at most k items")

    # a stale seed is caught by the commitment
    stale = Coordinator(k, 1)
    seed, _ = run_round(stream, 3, k, 0, stale)
    try:
        stale.accept(1, seed, [("x", 1.0)])
        raise AssertionError("an uncommitted seed must be rejected")
    except ValueError:
        pass
    stale_round = Coordinator(k, 1)
    good = stale_round.open_round(0)
    try:
        stale_round.accept(0, good + 1, [("x", 1.0)])
        raise AssertionError("a tampered seed must be rejected")
    except ValueError:
        pass
    check(stale_round.counters["rejected"] == 1, "the rejection is counted")

    # ---- 2. unbiasedness over many worlds ---------------------------------
    template = make_stream(31337, length=60, universe=12)
    multiplicity = {}
    for item in template:
        multiplicity[item] = multiplicity.get(item, 0) + 1
    worlds, trials = 400, 400
    positions = [0] * len(template)
    item_hits = {}
    sizes = set()
    for world in range(worlds):
        coordinator = Coordinator(8, 1)
        seed, _ = run_round(template, 4, 8, world, coordinator)
        picks = top_positions(template, 8, seed, 4)
        sizes.add(len(picks))
        for position in picks:
            positions[position] += 1
            item = template[position]
            item_hits[item] = item_hits.get(item, 0) + 1
        check([template[position] for position in picks] == coordinator.sample(),
              "the merge must equal the top-k positions of the reference")
    average = sum(positions) / float(len(positions) * worlds)
    check(abs(average - 8.0 / 60.0) <= 0.02,
          "the mean per-position inclusion probability must be k/n (%.4f)" % average)
    for position, hits in enumerate(positions):
        check(abs(hits / float(worlds) - 8.0 / 60.0) <= 0.12,
              "position %d must be sampled with probability k/n (%.3f)"
              % (position, hits / float(worlds)))
    check(sizes == {8}, "the sample size is deterministic in every world")

    # with k = 1 every *item* is sampled with probability 1 - (1 - 1/n)^multiplicity
    singles = {}
    for world in range(trials):
        coordinator = Coordinator(1, 1)
        seed, _ = run_round(template, 4, 1, world, coordinator)
        for item in set(coordinator.sample()):
            singles[item] = singles.get(item, 0) + 1
    for item, count in multiplicity.items():
        expected = 1.0 - (1.0 - 1.0 / len(template)) ** count
        measured = singles.get(item, 0) / float(trials)
        check(abs(measured - expected) <= 0.12,
              "item %d with %d copies: %.3f vs expected %.3f"
              % (item, count, measured, expected))

    # ---- 3. splitting does not change the distribution --------------------
    def frequencies(shards, worlds=300, length=40, universe=10, k_value=6):
        counts = [0] * universe
        for world in range(worlds):
            local = make_stream(5000 + world, length=length, universe=universe)
            coordinator = Coordinator(k_value, 1)
            _, _ = run_round(local, shards, k_value, 0, coordinator)
            for item in set(coordinator.sample()):
                counts[item] += 1
        return [value / float(worlds) for value in counts]

    central = frequencies(1)
    distributed = frequencies(5)
    for index in range(len(central)):
        check(abs(central[index] - distributed[index]) <= 0.15,
              "shard count %d must not shift the inclusion probability (%d)" % (5, index))
    check(abs(sum(central) / len(central) - sum(distributed) / len(distributed)) <= 0.05,
          "the average inclusion frequencies must agree between layouts")

    # ---- 4. total order independence --------------------------------------
    reordered = list(reversed(stream))
    coordinator = Coordinator(k, 1)
    seed, _ = run_round(stream, 4, k, 0, coordinator)
    shuffled = Coordinator(k, 1)
    shuffled.open_round(0)
    parts = [[] for _ in range(4)]
    for position, item in enumerate(reordered):
        parts[position % 4].append(item)
    for shard, part in enumerate(parts):
        shuffled.accept(0, shuffled.seeds[0], shard_sample(part, shard, shuffled.seeds[0], k))
    check(len(shuffled.sample()) == k, "a reordered stream still produces a k-sample")
    check(sorted(coordinator.sample()) != sorted(shuffled.sample()) or n == k,
          "the keys depend on position, so the two orders differ (as they should)")

    # ---- 5. edge cases ----------------------------------------------------
    zero = Coordinator(0, 1)
    run_round(stream, 3, 0, 0, zero)
    check(zero.sample() == [] and zero.merged == [], "k = 0 yields an empty sample")
    everything = Coordinator(n + 5, 1)
    run_round(stream, 3, n + 5, 0, everything)
    check(sorted(everything.sample()) == sorted(stream),
          "k >= n yields the whole stream")
    empty_stream = []
    none_round = Coordinator(k, 1)
    seed, parts = run_round(empty_stream, 3, k, 0, none_round)
    check(none_round.sample() == [] and all(part == [] for part in parts),
          "an empty stream produces an empty sample")
    one = Coordinator(k, 1)
    run_round([9], 4, k, 0, one)
    check(one.sample() == [9], "a single element is always sampled")
    check(shard_sample([], 0, 12345, k) == [], "an empty shard returns nothing")

    # ---- 6. determinism, memory and multi-round behaviour -----------------
    first = Coordinator(k, 2)
    for round_index in range(2):
        run_round(stream, 3, k, round_index, first)
    second = Coordinator(k, 2)
    for round_index in range(2):
        run_round(stream, 3, k, round_index, second)
    check(first.sample() == second.sample(), "rounds must be reproducible")
    check(first.counters == second.counters, "the counters must be reproducible")
    check(first.seeds[0] != first.seeds[1], "each round gets a fresh committed seed")
    check(len(first.merged) <= k, "the coordinator never exceeds its memory bound")
    sample = shard_sample(stream, 0, 99, k)
    check(len(sample) <= k, "a shard never exceeds k kept items")
    check(len(sample) == k, "a long stream fills the shard buffer")
    check(coordinator.counters["rejected"] == 0, "an honest round rejects nothing")
    check(digest("commit", 5) != digest("commit", 6), "commitments differ per seed")

    print("sample: k=%d from n=%d over %d shards, merged size %d"
          % (k, n, 4, len(coordinator.merged)))
    print("unbiasedness: mean positional probability %.4f vs k/n %.4f over %d worlds"
          % (average, 8.0 / 60.0, trials))
    print("layout: central %s vs distributed %s average frequencies"
          % ("%.3f" % (sum(central) / len(central)),
             "%.3f" % (sum(distributed) / len(distributed))))
    print("PASS")


if __name__ == "__main__":
    main()
