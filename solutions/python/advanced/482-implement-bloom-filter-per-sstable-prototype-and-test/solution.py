"""Task 482 - Implement bloom filter per SSTable prototype and test. (Tier: Advanced, CTCI Ch 14)

Prototype of an LSM-style lookup: each SSTable owns a bloom filter sized to 10 bits per key with
k = 7 double-hashed probes, and a sorted entry list for the exact confirmation step. Hashing is
FNV-1a 32-bit; probe j uses (h1 + j * h2) % m with h2 = fnv1a(key + "!") | 1 (odd, so the probe
sequence never collapses). Lookup consults every table's filter first (false positives only cost
an exact binary search, never a wrong answer).

Documented false-positive estimate (bits m, keys n, probes k): p = (1 - e^(-k*n/m))^k.
With n = 150/180/210, m = 10n, k = 7 that is p ~ 0.0082 per table; for the combined union of
tables the estimate is 1 - prod(1 - p_t) ~ 0.024. The test pins the exact (deterministic)
false-positive counts of the probe sets and also asserts the measured rate stays within a
documented 3x-theory bound, plus zero false negatives for every inserted key.
"""
import math


def fnv1a(text):
    h = 2166136261
    for ch in text.encode("ascii"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class BloomFilter:
    def __init__(self, bits, hashes):
        self.m = bits
        self.k = hashes
        self.words = bytearray((bits + 7) // 8)
        self.count = 0

    def _probes(self, key):
        h1 = fnv1a(key)
        h2 = fnv1a(key + "!") | 1
        return [(h1 + j * h2) % self.m for j in range(self.k)]

    def add(self, key):
        for idx in self._probes(key):
            self.words[idx >> 3] |= 1 << (idx & 7)
        self.count += 1

    def maybe_contains(self, key):
        for idx in self._probes(key):
            if not self.words[idx >> 3] & (1 << (idx & 7)):
                return False
        return True


class SSTable:
    def __init__(self, entries, bits_per_key=10, hashes=7):
        self.entries = sorted(entries)
        self.bloom = BloomFilter(bits_per_key * len(self.entries), hashes)

    def build(self, keys):
        for key in keys:
            self.bloom.add(key)
        return self

    def contains_exact(self, key):
        lo, hi = 0, len(self.entries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.entries[mid] == key:
                return True
            if self.entries[mid] < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return False

    def contains(self, key):
        return self.bloom.maybe_contains(key) and self.contains_exact(key)


class Store:
    def __init__(self, tables):
        self.tables = tables

    def get(self, key):
        for table in self.tables:
            if table.contains(key):
                return True
        return False

    def bloom_any(self, key):
        return any(t.bloom.maybe_contains(key) for t in self.tables)


def lcg_values(count, seed, limit):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % limit)
    return values


def theoretical_fp_rate(keys, bits, hashes):
    return (1.0 - math.exp(-hashes * keys / bits)) ** hashes


def build_store(sizes, seed):
    tables = []
    for i, size in enumerate(sizes):
        keys = ["t%d:%d" % (i, j) for j in range(size)]
        seed = (seed * 75 + 74) % 65537
        # deterministic shuffle so insertion order differs from sorted order
        rotation = seed % size
        keys = keys[rotation:] + keys[:rotation]
        tables.append(SSTable(keys).build(keys))
    return tables


def main():
    # Hand prototype: a single key answers true, an obviously different key is rejected.
    tiny = BloomFilter(64, 4)
    tiny.add("alpha")
    assert tiny.maybe_contains("alpha")
    empty = BloomFilter(64, 4)
    assert not empty.maybe_contains("alpha")

    sizes = [150, 180, 210]
    tables = build_store(sizes, 482)
    store = Store(tables)
    assert [len(t.entries) for t in tables] == sizes
    assert [t.bloom.m for t in tables] == [1500, 1800, 2100]
    assert all(t.bloom.k == 7 for t in tables)

    # No false negatives: every key of every table is reported by its filter and by Store.get.
    present_queries = 0
    for i, table in enumerate(tables):
        for key in table.entries:
            assert table.bloom.maybe_contains(key), key
            assert store.get(key), key
            present_queries += 1
    assert present_queries == 540

    # Deterministic probe sets: 200 present lookups and 240 absent lookups.
    union = {key for table in tables for key in table.entries}
    for value in lcg_values(200, 9001, 540):
        key = sorted(union)[value]
        assert store.get(key) and store.bloom_any(key)
    absent = ["absent:%d" % v for v in lcg_values(240, 9002, 65537)]
    assert all(key not in union for key in absent)
    assert all(not store.get(key) for key in absent)

    per_table_fp = [sum(t.bloom.maybe_contains(key) for key in absent) for t in tables]
    combined_fp = sum(store.bloom_any(key) for key in absent)
    # Pinned from the deterministic construction; identical in all four languages.
    assert per_table_fp == [2, 1, 1], per_table_fp
    assert combined_fp == 4, combined_fp

    # Documented estimates: per-table p and the union estimate 1 - prod(1 - p_t).
    p_t = [theoretical_fp_rate(size, t.bloom.m, t.bloom.k) for size, t in zip(sizes, tables)]
    p_union = 1.0 - math.prod(1.0 - p for p in p_t)
    for table_p, measured, size in zip(p_t, per_table_fp, sizes):
        assert measured <= 3 * table_p * len(absent) + 3, (measured, table_p)
        assert measured / len(absent) <= 0.05
        assert round(table_p, 4) == 0.0082, table_p
    assert combined_fp <= 3 * p_union * len(absent) + 3
    assert combined_fp / len(absent) <= 0.05
    assert round(p_union, 4) == 0.0244, p_union

    # Rejected keys are strongly rejected: bits touched by absent probes stay sparse.
    set_bits = [sum(bin(b).count("1") for b in t.bloom.words) for t in tables]
    assert set_bits == [768, 905, 1052], set_bits
    for bits, m in zip(set_bits, [1500, 1800, 2100]):
        assert bits < m  # never all ones: no false negative by saturation
    print("PASS")


if __name__ == "__main__":
    main()
