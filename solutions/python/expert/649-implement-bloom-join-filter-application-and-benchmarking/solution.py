"""Task 649 - Implement bloom join filter application and benchmarking.  (Tier: Expert, Ch 14 — Databases)

A bloom-filter semijoin reduction ("bloom join").  The build side is hashed into a bit array
of m bits with k probes per key, using double hashing `(h1 + i * h2) mod m` with an odd h2 so
the probe sequence visits distinct slots; the join then probes the filter for every row of
the *other* side and only rows that pass are handed to the ordinary hash join.  A bloom
filter has one-sided error: it can report a false positive, never a false negative, so the
result of a bloom join must be identical to a plain hash join - the checks assert exactly
that, on hand-built data, on random data, on duplicate-heavy build sides, on empty inputs
and even with a deliberately tiny, badly tuned filter.

What the benchmark shows is the trade-off, not a free lunch: the filter must be probed once
per probe row (that cost is always paid), and it only pays off when the build side is
selective, because then the number of rows that reach the hash table - and, in a distributed
plan, the number of rows that have to be shipped - drops sharply.  The measured false
positive rate is compared against the textbook value `(1 - e^{-kn/m})^k`.
"""

import math

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a(text):
    """64-bit FNV-1a: a deterministic string hash with no dependence on Python's hash()."""
    digest = FNV_OFFSET
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * FNV_PRIME) & MASK64
    return digest


class BloomFilter:
    def __init__(self, bits, hashes):
        if bits < 1 or hashes < 1:
            raise AssertionError("a bloom filter needs at least one bit and one hash")
        self.bits = bits
        self.hashes = hashes
        self.words = bytearray((bits + 7) // 8)
        self.keys = 0

    def positions(self, key):
        first = fnv1a("a:" + str(key))
        second = fnv1a("b:" + str(key)) | 1        # odd, so the probe sequence is a permutation
        for step in range(self.hashes):
            yield (first + step * second) % self.bits

    def add(self, key):
        for position in self.positions(key):
            self.words[position >> 3] |= 1 << (position & 7)
        self.keys += 1

    def contains(self, key):
        return all(self.words[position >> 3] >> (position & 7) & 1
                   for position in self.positions(key))

    def bits_set(self):
        return sum(bin(word).count("1") for word in self.words)


def theoretical_false_positive_rate(bits, hashes, keys):
    return (1.0 - math.exp(-hashes * keys / bits)) ** hashes


def optimal_hashes(bits, keys):
    """k = (m/n) ln 2 minimises the false positive probability."""
    return max(1, int(round(bits / max(1, keys) * math.log(2))))


def build_filter(keys, bits, hashes):
    bloom = BloomFilter(bits, hashes)
    for key in keys:
        bloom.add(key)
    return bloom


def hash_table_of(rows, key):
    table = {}
    for row in rows:
        table.setdefault(row[key], []).append(row)
    return table


def plain_hash_join(build, probe, key=1):
    """Reference join; counts the rows that had to be probed and the matched pairs."""
    table = hash_table_of(build, key)
    stats = {"probes": 0, "pairs": 0, "filter_probes": 0, "rejected_by_filter": 0}
    result = []
    for row in probe:
        stats["probes"] += 1
        for match in table.get(row[key], ()):
            stats["pairs"] += 1
            result.append(row + match)
    return result, stats


def bloom_join(build, probe, key=1, bits=None, hashes=None):
    keys = [row[key] for row in build]
    if bits is None:
        bits = max(64, 10 * max(1, len(keys)))
    if hashes is None:
        hashes = optimal_hashes(bits, len(keys))
    bloom = build_filter(keys, bits, hashes)
    table = hash_table_of(build, key)
    stats = {"probes": 0, "pairs": 0, "filter_probes": 0, "rejected_by_filter": 0}
    result = []
    for row in probe:
        stats["filter_probes"] += 1
        if not bloom.contains(row[key]):           # the only place a row is dropped
            stats["rejected_by_filter"] += 1
            continue
        stats["probes"] += 1
        for match in table.get(row[key], ()):
            stats["pairs"] += 1
            result.append(row + match)
    return result, stats, bloom


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def make_rows(count, key_domain, seed, base):
    rows, state = [], seed
    for index in range(count):
        state = lcg(state)
        rows.append((base + index, state % key_domain + 1))
    return rows, state


def main():
    # no false negatives, ever ------------------------------------------------------------
    keys = [index * 7 + 1 for index in range(120)]
    bloom = build_filter(keys, 512, 3)
    check(all(bloom.contains(key) for key in keys),
          "every inserted key is reported present (no false negatives)")
    check(bloom.bits_set() <= 512 and bloom.keys == 120, "the bit array bounds itself")
    check(not bloom.contains(10 ** 9 + 3) or bloom.contains(10 ** 9 + 3),
          "a miss may be reported as present (one-sided error)")

    # measured vs theoretical false positive rate -----------------------------------------
    for bits_per_key in (4, 8, 16):
        bits = bits_per_key * 100
        hashes = optimal_hashes(bits, 100)
        bloom = build_filter(range(100), bits, hashes)
        absent = [100000 + index for index in range(20000)]
        false_positives = sum(1 for key in absent if bloom.contains(key))
        measured = false_positives / len(absent)
        theory = theoretical_false_positive_rate(bits, hashes, 100)
        check(theory / 4 <= measured <= max(4 * theory, 0.005),
              "m/n=%d: measured %.4f vs theoretical %.4f (double hashing is approximate)"
              % (bits_per_key, measured, theory))
        if bits_per_key == 8:
            measured_8 = measured
            theory_8 = theory
    check(0.005 < measured_8 < 0.10, "8 bits per key gives a few percent: %.4f" % measured_8)
    sparse = build_filter(range(100), 8 * 100, 1)
    sparse_absent = [100000 + index for index in range(20000)]
    sparse_rate = sum(1 for key in sparse_absent if sparse.contains(key)) / len(sparse_absent)
    optimal = build_filter(range(100), 8 * 100, optimal_hashes(800, 100))
    optimal_rate = sum(1 for key in sparse_absent if optimal.contains(key)) / len(sparse_absent)
    check(optimal_rate < sparse_rate,
          "the optimal k beats k=1: %.4f < %.4f" % (optimal_rate, sparse_rate))
    dense = build_filter(range(100), 40 * 100, optimal_hashes(4000, 100))
    dense_rate = sum(1 for key in sparse_absent if dense.contains(key)) / len(sparse_absent)
    check(dense_rate < optimal_rate, "more bits per key means fewer positives: %.5f < %.5f"
          % (dense_rate, optimal_rate))
    check(theory_8 == theoretical_false_positive_rate(800, optimal_hashes(800, 100), 100),
          "the rate formula is deterministic")

    # the bloom join returns exactly what the plain hash join returns ---------------------
    build_rows, state = make_rows(400, 500, 777, 0)
    probe_rows, _ = make_rows(600, 500, state, 1000)
    main_plain, main_plain_stats = plain_hash_join(build_rows, probe_rows)
    main_bloomed, main_bloom_stats, main_bloom = bloom_join(build_rows, probe_rows)
    check(main_bloomed == main_plain, "bloom join and hash join results differ (%d vs %d rows)"
          % (len(main_bloomed), len(main_plain)))
    check(main_bloom_stats["pairs"] == main_plain_stats["pairs"], "no match was lost by the filter")
    check(main_bloom_stats["probes"] <= len(probe_rows), "the filter never adds probe rows")
    check(main_bloom_stats["filter_probes"] == len(probe_rows),
          "the filter is probed once per probe row - that is the cost of the reduction")

    for trial in range(12):
        state = lcg(state + trial * 7919)
        trial_build, _ = make_rows(30 + state % 80, 40, state, 0)
        trial_probe, _ = make_rows(50 + state % 120, 40, state >> 3, 1000)
        trial_plain, trial_plain_stats = plain_hash_join(trial_build, trial_probe)
        trial_bloomed, trial_stats, _ = bloom_join(trial_build, trial_probe,
                                                   bits=max(64, (state % 40) * 8),
                                                   hashes=1 + state % 4)
        check(trial_bloomed == trial_plain, "random trial %d lost or invented a row" % trial)
        check(trial_stats["probes"] <= trial_plain_stats["probes"],
              "trial %d probes fewer rows" % trial)

    # duplicates on the build side and empty sides ----------------------------------------
    duplicated = [(index, 7) for index in range(50)] + [(index, 8) for index in range(50)]
    probes = [(100 + index, key) for index, key in enumerate((7, 8, 9, 7, 8))]
    plain, _ = plain_hash_join(duplicated, probes)
    bloomed, dup_stats, _ = bloom_join(duplicated, probes)
    check(bloomed == plain and len(bloomed) == 2 * 50 + 2 * 50,
          "duplicate build keys are filtered and joined correctly")
    check(dup_stats["rejected_by_filter"] == 1, "only the absent key was filtered out")

    empty_build, empty_stats, empty_bloom = bloom_join([], probes)
    check(empty_build == [] and empty_stats["probes"] == 0,
          "an empty build side filters every probe row away")
    check(empty_stats["rejected_by_filter"] == len(probes) and empty_bloom.keys == 0,
          "an empty filter rejects everything and the result is empty, as it must be")
    check(plain_hash_join([], probes)[0] == [], "the reference agrees")
    no_probes, no_stats, _ = bloom_join(build_rows, [])
    check(no_probes == [] and no_stats["filter_probes"] == 0, "an empty probe side costs nothing")

    # a deliberately over-full filter: many false positives, still no wrong answers -------
    tiny, tiny_stats, tiny_bloom = bloom_join(build_rows, probe_rows, bits=64, hashes=1)
    check(tiny == main_plain, "even a 64-bit filter cannot change the join result")
    check(tiny_bloom.keys == len(build_rows), "the tiny filter still absorbed every build key")
    check(tiny_stats["rejected_by_filter"] < len(probe_rows),
          "a saturated filter rejects less: %d" % tiny_stats["rejected_by_filter"])

    # benchmark: a selective build side is where the reduction pays ------------------------
    big_probe, _ = make_rows(20000, 100000, 31337, 0)
    selective_build = [(index, (index % 500) * 199 + 1) for index in range(500)]
    bench_plain, bench_plain_stats = plain_hash_join(selective_build, big_probe)
    bench_result, bench_stats, _ = bloom_join(selective_build, big_probe)
    check(bench_result == bench_plain, "the selective benchmark keeps the same result")
    check(bench_stats["probes"] * 3 < bench_plain_stats["probes"],
          "the filter cut probes from %d to %d"
          % (bench_plain_stats["probes"], bench_stats["probes"]))
    check(bench_stats["filter_probes"] == bench_plain_stats["probes"],
          "each probe row paid one filter probe")
    check(len(bench_plain) == len(set(bench_plain)), "the benchmark has no row duplication")

    # ... but it cannot help when every probe key is present ------------------------------
    full_build = [(index, index % 100000 + 1) for index in range(60000)]
    matched_probe = [(index, index % 60000 + 1) for index in range(20000)]
    _, covered_stats, _ = bloom_join(full_build, matched_probe)
    check(covered_stats["rejected_by_filter"] == 0,
          "a filter covering every probe key rejects nothing")
    check(covered_stats["probes"] == len(matched_probe), "so every probe row still hits the table")

    # determinism and error handling ------------------------------------------------------
    repeat, repeat_stats, _ = bloom_join(build_rows, probe_rows)
    check(repeat == main_bloomed and repeat_stats == main_bloom_stats,
          "the bloom join is deterministic")
    check(len(main_bloom.words) == (main_bloom.bits + 7) // 8,
          "the bit array is sized ceil(m/8): %d bytes for %d bits"
          % (len(main_bloom.words), main_bloom.bits))
    try:
        BloomFilter(0, 3)
    except AssertionError:
        pass
    else:
        raise AssertionError("a zero-bit filter must be rejected")
    print("[649] bloom join: 8 bits/key FP rate %.4f (theory %.4f); selective build cut probes "
          "%d -> %d" % (measured_8, theory_8, bench_plain_stats["probes"], bench_stats["probes"]))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
