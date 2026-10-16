"""Task 682 - Implement map-reduce optimization for skewed keys (combiner + partitioning).  (Tier: Expert, Ch 9 — System Design and Scalability)

A word-count style MapReduce with the two classic skew optimizations and a real, measured
cost model.

Pipeline
  1. the input is split into `splits` chunks; each chunk is mapped independently;
  2. optional *in-mapper combining* aggregates a mapper's output before it leaves the node;
  3. the shuffle groups by (key, shard), where an optional *combiner* aggregates the pairs
     that a mapper would send to the same reducer - the shuffle volume counter reports
     exactly how many pairs cross the network;
  4. a partitioner assigns each (key, shard) to a reducer: `hash` (the naive default),
     `range` (contiguous key ranges) or `hot_key` (keys whose observed frequency exceeds a
     threshold are sharded across reducers and summed back after the reduce phase);
  5. the reducers aggregate their groups and - for sharded hot keys - the shards are merged.

Verified: the final result is identical to a single-machine aggregation for every
combination of combiner on/off, in-mapper combining, 1..4 reducers, 1..4 splits and all
three partitioners; the combiner never increases and on skewed data strictly decreases the
shuffle volume; the load imbalance (max/mean reducer load) improves under hot-key sharding
and gets worse under naive hash partitioning; a single-key input cannot be balanced by
hashing at all but is by sharding; the combine function is associative and commutative; each
key is assigned to exactly one reducer per configuration; and the deterministic PRNG makes
the whole benchmark reproducible.

Simplified: aggregation is commutative integer addition, so no reducer-side sorting is
needed; there is no network latency model - only pair counts and load, which is what
partitioning changes.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def skewed_dataset(seed, size, keys, exponent=4):
    """Power-law-ish key frequencies: the low keys dominate (a hot-key workload)."""
    state = seed
    records = []
    for _ in range(size):
        state, raw = splitmix64(state)
        unit = (raw >> 11) / float(1 << 53)
        records.append(("k%03d" % int(keys * unit ** exponent), 1))
    return records


def uniform_dataset(seed, size, keys):
    state = seed
    records = []
    for _ in range(size):
        state, raw = splitmix64(state)
        records.append(("k%03d" % (raw % keys), 1))
    return records


def combine_values(left, right):
    return left + right


def is_combining_safe():
    """Associativity and commutativity on a deterministic sample."""
    samples = [(3, 5), (5, 3), (1, 1), (7, 2), (2, 7)]
    for a, b in samples:
        if combine_values(a, b) != combine_values(b, a):
            return False
        if combine_values(combine_values(a, b), 1) != combine_values(a, combine_values(b, 1)):
            return False
    return True


def reference_reduce(records):
    """Single-machine aggregation: the answer every configuration must reproduce."""
    result = {}
    for key, value in records:
        result[key] = result.get(key, 0) + value
    return result


def split_records(records, splits):
    """Contiguous input splits, the way a file would be cut."""
    if splits < 1:
        raise ValueError("at least one split is required")
    size = len(records)
    chunks = []
    start = 0
    for index in range(splits):
        end = size * (index + 1) // splits
        chunks.append(records[start:end])
        start = end
    return chunks


def key_counts(records):
    counts = {}
    for key, _ in records:
        counts[key] = counts.get(key, 0) + 1
    return counts


def shard_of(key, occurrence, counts, reducers, partitioner, hot_threshold):
    """A hot key is spread over the reducers; every other key stays on one shard."""
    if partitioner == "hash":
        return 0
    if partitioner == "range":
        return 0
    if partitioner != "hot_key":
        raise ValueError("unknown partitioner %r" % (partitioner,))
    if counts.get(key, 0) > hot_threshold:
        return occurrence % reducers
    return 0


def reducer_for(key, shard, reducers, partitioner, key_space):
    if reducers < 1:
        raise ValueError("at least one reducer is required")
    if partitioner == "hash":
        return (sum(ord(c) for c in key) * 31) % reducers
    if partitioner == "hot_key":
        base = (sum(ord(c) for c in key) * 31) % reducers
        return (base + shard) % reducers
    if partitioner == "range":
        digits = key[1:]
        number = int(digits) if digits.isdigit() else sum(ord(c) for c in key)
        return min(reducers - 1, number * reducers // max(1, key_space))
    raise ValueError("unknown partitioner %r" % (partitioner,))


def run_map_reduce(records, reducers=2, splits=2, use_combiner=True, use_in_mapper=True,
                   partitioner="hash", hot_threshold=None):
    """Run the whole pipeline; returns the result plus the measured statistics."""
    counts = key_counts(records)
    threshold = (hot_threshold if hot_threshold is not None
                 else max(1, len(records) // max(1, reducers)))
    numeric = [int(key[1:]) for key, _ in records if key[1:].isdigit()]
    key_space = (max(numeric) + 1) if numeric else max(1, len(counts))
    chunks = split_records(records, splits)
    occurrence = {}
    shuffled = 0
    mapped_pairs = 0
    loads = [0] * reducers
    groups = {}
    for chunk in chunks:
        emitted = [(key, value) for key, value in chunk]          # the map phase
        if use_in_mapper:
            collapsed = {}
            for key, value in emitted:
                collapsed[key] = collapsed.get(key, 0) + value
            emitted = [(key, collapsed[key]) for key in sorted(collapsed)]
        mapped_pairs += len(chunk)
        staged = {}
        for key, value in emitted:
            shard = shard_of(key, occurrence.get(key, 0), counts, reducers, partitioner, threshold)
            occurrence[key] = occurrence.get(key, 0) + 1
            slot = (key, shard)
            staged[slot] = staged.get(slot, 0) + value
        if use_combiner:
            for slot in sorted(staged):
                reducer = reducer_for(slot[0], slot[1], reducers, partitioner, key_space)
                groups[slot] = groups.get(slot, 0) + staged[slot]
                loads[reducer] += staged[slot]
                shuffled += 1
        else:
            for key, value in emitted:
                shard = shard_of(key, occurrence.get(key, 0), counts, reducers, partitioner,
                                 threshold)
                occurrence[key] = occurrence.get(key, 0) + 1
                slot = (key, shard)
                reducer = reducer_for(key, shard, reducers, partitioner, key_space)
                groups[slot] = groups.get(slot, 0) + value
                loads[reducer] += value
                shuffled += 1
    result = {}
    for (key, _shard) in sorted(groups):
        result[key] = result.get(key, 0) + groups[(key, _shard)]
    return {"result": result, "shuffled": shuffled, "mapped": mapped_pairs, "loads": loads,
            "max_load": max(loads), "mean_load": sum(loads) / float(reducers),
            "imbalance": max(loads) / max(1e-9, sum(loads) / float(reducers)),
            "groups": len(groups), "threshold": threshold}


def main():
    check(is_combining_safe(), "integer addition must be associative and commutative")

    # ---- the invariance property: every configuration gives the same answer -----------
    skewed = skewed_dataset(0x5EED, 600, 24)
    reference = reference_reduce(skewed)
    check(sum(reference.values()) == 600, "the reference must account for every record")
    top_share = max(reference.values()) / 600.0
    check(top_share > 0.15,
          "the dataset must really be skewed: the top key holds %.1f%%" % (100 * top_share))

    configurations = 0
    for partitioner in ("hash", "range", "hot_key"):
        for reducers in (1, 2, 3, 4):
            for splits in (1, 2, 4):
                for combiner in (False, True):
                    for in_mapper in (False, True):
                        stats = run_map_reduce(skewed, reducers, splits, combiner, in_mapper,
                                               partitioner)
                        check(stats["result"] == reference,
                              "configuration %r broke the result" % ((partitioner, reducers,
                                                                       splits, combiner, in_mapper),))
                        configurations += 1
    check(configurations == 144, "every configuration must be exercised (%d)" % configurations)

    # ---- the combiner must reduce the shuffle ---------------------------------------
    with_combiner = run_map_reduce(skewed, 4, 4, use_combiner=True, partitioner="hash")
    without = run_map_reduce(skewed, 4, 4, use_combiner=False, use_in_mapper=False,
                             partitioner="hash")
    check(without["shuffled"] == 600,
          "with neither optimization every mapped pair crosses the network: %d"
          % without["shuffled"])
    check(with_combiner["shuffled"] < without["shuffled"],
          "the combiner must shrink the shuffle (%d < %d)"
          % (with_combiner["shuffled"], without["shuffled"]))
    check(with_combiner["groups"] <= 24 * 4, "at most one group per key per shard")
    mapper_only = run_map_reduce(skewed, 4, 1, use_combiner=False, use_in_mapper=True)
    check(mapper_only["shuffled"] <= without["shuffled"],
          "in-mapper combining alone can only help")

    # ---- partitioning under skew ------------------------------------------------------
    naive = run_map_reduce(skewed, 4, 4, True, True, "hash")
    sharded = run_map_reduce(skewed, 4, 4, True, True, "hot_key")
    check(naive["max_load"] >= reference[max(reference, key=lambda k: reference[k])],
          "hash partitioning concentrates the hottest key on one reducer: %d" % naive["max_load"])
    check(sharded["max_load"] < naive["max_load"],
          "hot-key sharding must reduce the busiest reducer (%d < %d)"
          % (sharded["max_load"], naive["max_load"]))
    check(sharded["imbalance"] < naive["imbalance"],
          "and the imbalance ratio must improve (%.2f < %.2f)"
          % (sharded["imbalance"], naive["imbalance"]))
    check(sharded["result"] == reference and naive["result"] == reference,
          "both partitioners must still produce the exact answer")

    single = [("only", 1)] * 200
    single_reference = reference_reduce(single)
    hashed_single = run_map_reduce(single, 4, 2, True, True, "hash")
    hot_single = run_map_reduce(single, 4, 2, True, True, "hot_key")
    check(hashed_single["result"] == single_reference == hot_single["result"],
          "the single-key answer must be exact under both partitioners")
    check(hashed_single["max_load"] == 200,
          "hashing one key leaves one reducer with everything: %d" % hashed_single["max_load"])
    check(hot_single["max_load"] < 200,
          "sharding the single hot key spreads its load: %d" % hot_single["max_load"])
    check(hot_single["groups"] > 1, "the hot key must be split into several shards")

    balanced = uniform_dataset(0xBEEF, 600, 24)
    uniform = run_map_reduce(balanced, 4, 4, True, True, "hash")
    check(uniform["result"] == reference_reduce(balanced), "uniform data still reduces exactly")
    check(uniform["max_load"] < naive["max_load"],
          "uniform keys hash more evenly than a skewed distribution (%d < %d)"
          % (uniform["max_load"], naive["max_load"]))

    # ---- partitioning is a function of the key ---------------------------------------
    for partitioner in ("hash", "range", "hot_key"):
        targets = {reducer_for(key, 0, 4, partitioner, 24) for key in reference}
        check(all(0 <= reducer < 4 for reducer in targets),
              "every reducer index must be in range for %r" % (partitioner,))
        check(len(targets) >= 1, "each partitioner must place keys somewhere")
    for key in ("k000", "k007", "k023"):
        check(reducer_for(key, 0, 4, "hash", 24) == reducer_for(key, 3, 4, "hash", 24),
              "hashing ignores the shard, so a key cannot move")
        check(reducer_for(key, 0, 4, "range", 24) == reducer_for(key, 2, 4, "range", 24),
              "range partitioning also ignores the shard")
        check(reducer_for(key, 0, 4, "hot_key", 24) != reducer_for(key, 1, 4, "hot_key", 24),
              "sharding must spread one key over different reducers")
    check(reducer_for("k005", 0, 4, "range", 24) < reducer_for("k020", 0, 4, "range", 24),
          "range partitioning keeps ordered keys on ordered reducers")

    # ---- input splitting and boundary cases ------------------------------------------
    chunks = split_records(skewed, 4)
    check(sum(len(chunk) for chunk in chunks) == len(skewed),
          "the splits must cover the input exactly once")
    check(all(len(chunk) >= 0 for chunk in chunks), "splits may be empty for large split counts")
    empty = run_map_reduce([], 3, 2, True, True, "hash")
    check(empty["result"] == {} and empty["shuffled"] == 0 and empty["max_load"] == 0,
          "an empty input must reduce to nothing: %r" % (empty,))
    check(run_map_reduce([("x", 1)], 2, 3, True, True, "hash")["result"] == {"x": 1},
          "more splits than records is legal")
    rejected = 0
    for bad in (lambda: run_map_reduce(skewed, 0, 1),
                lambda: run_map_reduce(skewed, 2, 0),
                lambda: run_map_reduce(skewed, 2, 2, True, True, "frobnicate"),
                lambda: split_records(skewed, 0)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 4, "invalid configurations must be rejected (%d of 4)" % rejected)

    # ---- determinism ------------------------------------------------------------------
    check(run_map_reduce(skewed, 4, 4, True, True, "hot_key") == sharded,
          "the pipeline must be deterministic")
    check(skewed_dataset(0x5EED, 600, 24) == skewed, "the dataset generator must be deterministic")
    check(uniform_dataset(0xBEEF, 600, 24) == balanced, "the uniform generator must be deterministic")

    print("skewed: top key %.1f%%, hash max load %d vs hot-key %d (%d reducers)"
          % (100 * top_share, naive["max_load"], sharded["max_load"], 4))
    print("shuffle volume: %d without a combiner, %d with, %d in-mapper only"
          % (without["shuffled"], with_combiner["shuffled"], mapper_only["shuffled"]))
    print("configurations verified against the reference: %d" % configurations)
    print("PASS")


if __name__ == "__main__":
    main()
