"""Task 484 - Implement simple key-value store with memtable + SSTable. (Tier: Advanced, CTCI Ch 14)

LSM-style prototype with documented semantics:
- Writes land in an in-memory memtable (dict key -> value; deletes insert a TOMBSTONE marker).
- When the memtable reaches `flush_threshold` entries it flushes to a new immutable SSTable:
  the records are written sorted by key and appended to the sstable list (oldest first).
- get(key) probes the memtable, then the SSTables from newest to oldest; the first record for the
  key wins. A tombstone means "deleted" (returns None) and must shadow older values in older
  SSTables - that is what makes deletes correct without rewriting data.
- compact() flushes the memtable, merges all SSTables newest-wins into a single one, and drops
  tombstones plus every record they shadow, so the result holds exactly the live keys.

The test asserts get == oracle after every single operation of an LCG-driven 240-op script (puts,
gets, deletes) including tombstone shadowing across SSTables, then checks the post-compaction
layout (no tombstones, one record per live key) and idempotence of a second compaction.
"""

TOMBSTONE = None


class KVStore:
    def __init__(self, flush_threshold=4):
        self.flush_threshold = flush_threshold
        self.memtable = {}
        self.sstables = []
        self.flushes = 0
        self.compactions = 0

    def put(self, key, value):
        self.memtable[key] = value
        self._maybe_flush()

    def delete(self, key):
        self.memtable[key] = TOMBSTONE
        self._maybe_flush()

    def _maybe_flush(self):
        if len(self.memtable) >= self.flush_threshold:
            self.flush()

    def flush(self):
        if not self.memtable:
            return
        records = sorted(self.memtable.items())
        self.sstables.append(records)
        self.memtable = {}
        self.flushes += 1

    def get(self, key):
        if key in self.memtable:
            return self.memtable[key]
        for records in reversed(self.sstables):
            for rkey, value in records:
                if rkey == key:
                    return value
        return None

    def compact(self):
        self.flush()
        newest = {}
        for records in reversed(self.sstables):  # newest wins; oldest applied first
            for key, value in records:
                if key not in newest:
                    newest[key] = value
        live = sorted((key, value) for key, value in newest.items() if value is not TOMBSTONE)
        self.sstables = [live] if live else []
        self.compactions += 1


class Oracle:
    def __init__(self, keys):
        self.live = {key: None for key in keys}

    def put(self, key, value):
        self.live[key] = value

    def delete(self, key):
        self.live.pop(key, None)

    def get(self, key):
        return self.live.get(key)


def find_record(records_list, key):
    for records in records_list:
        for rkey, value in records:
            if rkey == key:
                return value
    return None


def lcg_ops(count, seed, key_count):
    ops = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        kind = seed % 3
        seed = (seed * 75 + 74) % 65537
        key = "k%d" % (seed % key_count)
        seed = (seed * 75 + 74) % 65537
        value = seed % 1000
        ops.append((kind, key, value))
    return ops


def main():
    # Hand case: tombstone shadowing across flushes and compaction.
    store = KVStore(flush_threshold=2)
    store.put("a", 1)
    store.put("b", 2)
    assert store.flushes == 1 and store.memtable == {}
    assert store.get("a") == 1 and store.get("b") == 2
    store.put("a", 5)
    store.flush()
    store.delete("b")
    assert store.get("b") is None
    store.flush()
    assert store.get("a") == 5 and store.get("b") is None
    assert len(store.sstables) == 3
    assert find_record([store.sstables[-1]], "b") is TOMBSTONE
    assert find_record([store.sstables[0]], "b") == 2  # old value still on disk, shadowed
    store.compact()
    assert store.compactions == 1 and len(store.sstables) == 1
    assert store.sstables[0] == [("a", 5)]  # tombstone and shadowed value both dropped
    assert store.get("a") == 5 and store.get("b") is None
    store.compact()
    assert store.sstables[0] == [("a", 5)]  # idempotent

    # Randomized script: get must equal the oracle after every step.
    keys = ["k%d" % i for i in range(12)]
    store = KVStore(flush_threshold=4)
    oracle = Oracle(keys)
    gets = tombstones = 0
    for step, (kind, key, value) in enumerate(lcg_ops(240, 484, 12)):
        if kind == 0:
            store.put(key, value)
            oracle.put(key, value)
        elif kind == 1:
            assert store.get(key) == oracle.get(key), step
            gets += 1
        else:
            store.delete(key)
            oracle.delete(key)
            tombstones += 1
        for probe in keys:
            assert store.get(probe) == oracle.get(probe), (step, probe)
        if step % 60 == 59:
            store.compact()
            for probe in keys:
                assert store.get(probe) == oracle.get(probe), ("post-compact", step, probe)
    assert gets == 84 and tombstones == 71, (gets, tombstones)
    assert store.flushes == 35, store.flushes
    assert store.compactions == 4
    store.compact()
    assert len(store.sstables) == 1
    live = sorted(oracle.live.items())
    assert store.sstables[0] == [(key, value) for key, value in live if value is not None]
    assert all(value is not None for _, value in store.sstables[0])
    for probe in keys:
        assert store.get(probe) == oracle.get(probe), probe
    assert sorted(k for k, _ in store.sstables[0]) == sorted(oracle.live.keys())
    print("PASS")


if __name__ == "__main__":
    main()
