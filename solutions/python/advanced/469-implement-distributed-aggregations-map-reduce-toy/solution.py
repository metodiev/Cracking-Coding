"""Task 469 - Implement distributed aggregations map-reduce toy. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Deterministic word-count MapReduce with explicit phases:
  * split   - three input splits of LCG-generated lines over a 40-word vocabulary;
  * map     - each split emits (word, 1) pairs;
  * partition - partition_of(word) = polynomial hash (h = (h*31 + ord(ch)) mod
    1000003) mod R with R = 4 reducers (documented rule);
  * shuffle - pairs are grouped by (partition, word) and sorted by (partition, word);
  * reduce  - each (partition, word) group is summed to one (word, count) pair.

Checks: the reduced counts equal a plain Counter of all words; every word's pairs
land in exactly one partition (the hash of the word); per-partition pair totals are
conserved (map pairs in == reduced counts out); every word is reduced exactly once;
the pipeline is deterministic (two runs give identical results); and the
hand-computed partition of "w00" is 3 (h("w00") = 115895, 115895 mod 4 = 3).
"""

VOCAB = ["w%02d" % i for i in range(40)]
REDUCERS = 4


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def make_splits(seed):
    """Three splits of 20 lines each; lines hold 1..8 vocabulary words."""
    splits = []
    for _ in range(3):
        lines = []
        for _ in range(20):
            seed = lcg_value(seed)
            length = seed % 8 + 1
            line = []
            for _ in range(length):
                seed = lcg_value(seed)
                line.append(VOCAB[seed % len(VOCAB)])
            lines.append(line)
        splits.append(lines)
    return splits


def partition_of(word):
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) % 1000003
    return h % REDUCERS


def map_phase(splits):
    pairs = []
    for split_id, lines in enumerate(splits):
        for word in (w for line in lines for w in line):
            pairs.append((split_id, word, 1))
    return pairs


def partition_phase(pairs):
    buckets = {}
    for split_id, word, count in pairs:
        buckets.setdefault(partition_of(word), []).append((split_id, word, count))
    return buckets


def shuffle_phase(buckets):
    groups = {}
    for part in sorted(buckets):
        for _, word, count in sorted(buckets[part], key=lambda p: (p[1], p[0])):
            groups.setdefault((part, word), []).append(count)
    return groups


def reduce_phase(groups):
    return {(part, word): sum(counts) for (part, word), counts in groups.items()}


def run_pipeline(splits):
    return reduce_phase(shuffle_phase(partition_phase(map_phase(splits))))


def main():
    assert partition_of("w00") == 3  # hand-checked: h("w00") = 115895 -> 3 (mod 4)
    assert partition_of("w") == 3  # hand-checked: h("w") = 119, 119 mod 4 = 3

    splits = make_splits(20260116)
    pairs = map_phase(splits)
    buckets = partition_phase(pairs)
    groups = shuffle_phase(buckets)
    reduced = reduce_phase(groups)

    # Plain-Counter oracle over all words.
    counts = {}
    for line in (line for split in splits for line in split):
        for word in line:
            counts[word] = counts.get(word, 0) + 1
    flat = {}
    for (part, word), total in reduced.items():
        assert word not in flat, word  # every word reduced exactly once
        flat[word] = total
    assert flat == counts

    # Words must be routed to exactly the partition of their hash.
    words_seen = {}
    for (part, word) in reduced:
        words_seen[word] = part
        assert part == partition_of(word)
    for word in counts:
        assert word in words_seen

    # Conservation: map pairs per partition == reduced counts per partition.
    per_partition_in = {p: 0 for p in range(REDUCERS)}
    for _, word, count in pairs:
        per_partition_in[partition_of(word)] += count
    per_partition_out = {p: 0 for p in range(REDUCERS)}
    for (part, _), total in reduced.items():
        per_partition_out[part] += total
    assert per_partition_in == per_partition_out
    assert sum(per_partition_in.values()) == len(pairs)

    # Determinism: same input -> identical pipeline output.
    assert run_pipeline(make_splits(20260116)) == reduced


if __name__ == "__main__":
    main()
    print("PASS")
