"""Task 478 - Implement log-structured merge tree (LSM) basic flush/merge simulation. (Tier: Advanced, CTCI Ch 14 — Databases)

LSM engine: every write/delete lands in a memtable keyed by a globally increasing sequence
number (deletes are tombstones), and the memtable is flushed to an immutable sorted SSTable
after a fixed number of operations. When four tables accumulate they are merged into one,
deduplicating keys by highest sequence number and *keeping tombstones* so that deleted keys
never resurrect. Reads take the highest-sequence entry across the memtable and every table;
scans merge all sources the same way and drop tombstones. With 80 ops and a flush every 4
ops this yields exactly 20 flushes, 6 merges and 2 tables at the end (hand-traced); every
get/scan is checked against a last-write-per-key oracle after each operation and after merges.
"""

FLUSH_AFTER = 4
MERGE_FANOUT = 4
KEYS = 16


class SSTable:
    def __init__(self, table_id, entries):
        self.id = table_id
        self.entries = entries      # key -> (seq, value); value None is a tombstone


def merge_tables(tables):
    merged = {}
    for table in tables:
        for key, (seq, value) in table.entries.items():
            if key not in merged or seq > merged[key][0]:
                merged[key] = (seq, value)
    return SSTable(-1, merged)


class LSMStore:
    def __init__(self, flush_after=FLUSH_AFTER):
        self.memtable = {}
        self.sstables = []
        self.seq = 0
        self.ops_since_flush = 0
        self.flush_after = flush_after
        self.flush_count = 0
        self.merge_count = 0
        self.next_table_id = 0

    def write(self, key, value):
        self.seq += 1
        self.memtable[key] = (self.seq, value)
        self.ops_since_flush += 1
        if self.ops_since_flush >= self.flush_after:
            self.flush()

    def delete(self, key):
        self.write(key, None)

    def flush(self):
        if not self.memtable:
            return
        self.next_table_id += 1
        self.sstables.append(SSTable(self.next_table_id, dict(self.memtable)))
        self.memtable = {}
        self.ops_since_flush = 0
        self.flush_count += 1
        while len(self.sstables) >= MERGE_FANOUT:
            merged = merge_tables(self.sstables[:MERGE_FANOUT])
            self.sstables = [merged] + self.sstables[MERGE_FANOUT:]
            self.merge_count += 1

    def _entry_for(self, key):
        best = self.memtable.get(key)
        for table in self.sstables:
            entry = table.entries.get(key)
            if entry is not None and (best is None or entry[0] > best[0]):
                best = entry
        return best

    def get(self, key):
        entry = self._entry_for(key)
        if entry is None:
            return None
        return entry[1]                     # None means deleted

    def scan(self):
        merged = {}
        for table in self.sstables:
            for key, entry in table.entries.items():
                if key not in merged or entry[0] > merged[key][0]:
                    merged[key] = entry
        for key, entry in self.memtable.items():
            if key not in merged or entry[0] > merged[key][0]:
                merged[key] = entry
        return [(key, entry[1]) for key, entry in sorted(merged.items()) if entry[1] is not None]


def lcg(seed):
    return (seed * 75 + 74) % 65537


def gen_ops(count, seed):
    ops = []
    for _ in range(count):
        seed = lcg(seed)
        kind = seed % 5
        seed = lcg(seed)
        key = seed % KEYS
        seed = lcg(seed)
        value = seed % 100
        ops.append((kind, key, value))
    return ops


def run_workload():
    store = LSMStore()
    oracle = {}
    for i, (kind, key, value) in enumerate(gen_ops(80, 31337)):
        if kind == 4:
            store.delete(key)
            oracle[key] = None
        else:
            store.write(key, value)
            oracle[key] = value
        for k in range(KEYS):
            assert store.get(k) == oracle.get(k), (i, k, store.get(k), oracle.get(k))
        expected_scan = sorted((k, v) for k, v in oracle.items() if v is not None)
        assert store.scan() == expected_scan, i
    assert store.flush_count == 20, "one flush per four ops"
    assert store.merge_count == 6, "merge at flushes 4,7,10,13,16,19"
    assert len(store.sstables) == 2, "one merged table plus the last flush"
    assert store.memtable == {}, "memtable drained by the last flush"
    for k in range(KEYS):
        assert store.get(k) == oracle.get(k)
    return store, oracle


def main():
    # Flush policy: the first table appears exactly on the 4th operation.
    s = LSMStore()
    s.write(0, 10)
    s.write(1, 11)
    s.write(2, 12)
    assert len(s.sstables) == 0 and s.get(0) == 10
    s.write(3, 13)
    assert len(s.sstables) == 1 and s.memtable == {} and s.flush_count == 1
    assert s.scan() == [(0, 10), (1, 11), (2, 12), (3, 13)]
    s.delete(1)
    assert s.get(1) is None
    s.write(1, 99)
    assert s.get(1) == 99
    assert s.scan() == [(0, 10), (1, 99), (2, 12), (3, 13)]

    # Tombstones survive flushing and merging: a deleted key stays deleted.
    s2 = LSMStore(flush_after=2)
    s2.write(5, 1)
    s2.write(5, 2)                          # flush: {5: 2}
    s2.delete(5)
    s2.write(6, 3)                          # flush: {5: tombstone, 6: 3}
    assert s2.get(5) is None and s2.get(6) == 3
    s2.sstables = [merge_tables(s2.sstables)]
    assert s2.get(5) is None and s2.scan() == [(6, 3)], "merge must not resurrect the tombstone"
    s2.write(5, 7)
    s2.write(8, 8)                          # flush: {5: 7, 8: 8}
    assert s2.get(5) == 7 and s2.get(8) == 8
    assert s2.scan() == [(5, 7), (6, 3), (8, 8)]

    run_workload()


if __name__ == "__main__":
    main()
    print("PASS")
