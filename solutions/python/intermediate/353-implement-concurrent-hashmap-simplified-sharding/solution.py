"""Task 353 - Implement concurrent hashmap simplified (sharding). (Tier: Intermediate, CTCI Ch 15)

Sharded map: a fixed number of SHARDS independent dicts, each guarded by its own lock.
Hash policy (documented, identical in all four languages): shard(key) = ((key % S) + S) % S
for S = 4 shards - non-negative keys map to key % 4, and negative keys are normalised into
the same range (e.g. -1 -> shard 3, -4 -> shard 0), so every key belongs to exactly one
shard. The self-test proves: (a) each inserted key sits in exactly its policy shard,
(b) 4 worker threads upserting overlapping keys (same value per key, so the final state is
order-independent) and disjoint ranges all persist with an exact final count, and
(c) the result equals a single-threaded reference map after a removal, key by key.
"""
import threading

SHARDS = 4
THREADS = 4
KEYS = 50


class ShardedMap:
    def __init__(self, num_shards):
        if num_shards <= 0:
            raise ValueError("num_shards must be positive")
        self.shards = [{} for _ in range(num_shards)]
        self.locks = [threading.Lock() for _ in range(num_shards)]

    def shard_index(self, key):
        count = len(self.shards)
        return ((key % count) + count) % count

    def put(self, key, value):
        shard = self.shard_index(key)
        with self.locks[shard]:
            self.shards[shard][key] = value

    def get(self, key):
        shard = self.shard_index(key)
        with self.locks[shard]:
            return self.shards[shard].get(key)

    def remove(self, key):
        shard = self.shard_index(key)
        with self.locks[shard]:
            if key in self.shards[shard]:
                del self.shards[shard][key]
                return True
            return False

    def size(self):
        total = 0
        for shard, lock in zip(self.shards, self.locks):
            with lock:
                total += len(shard)
        return total

    def keys(self):
        result = []
        for shard, lock in zip(self.shards, self.locks):
            with lock:
                result.extend(shard)
        return sorted(result)

    def shards_containing(self, key):
        """Shard indexes whose dict currently holds the key (single-threaded use)."""
        return [index for index, shard in enumerate(self.shards) if key in shard]


def main():
    mapping = ShardedMap(SHARDS)

    # Phase 1 (single-threaded): keys 0..49 land in shards 0..3 with hand-checked counts.
    for key in range(KEYS):
        mapping.put(key, key * 3)
    assert [len(shard) for shard in mapping.shards] == [13, 13, 12, 12], "hand-checked shard counts"

    # Every inserted key is in exactly one shard, and it is the policy shard.
    for key in range(KEYS):
        assert mapping.shards_containing(key) == [mapping.shard_index(key)], \
            "key {} lives in exactly its policy shard".format(key)

    # Negative keys are normalised by the documented policy (hand-checked: -1 -> 3, -4 -> 0).
    assert mapping.shard_index(-1) == 3 and mapping.shard_index(-4) == 0, "negative key policy"
    mapping.put(-1, 999)
    assert mapping.shards_containing(-1) == [3], "negative key lands in one shard"
    assert mapping.get(-1) == 999, "negative key round-trips"
    assert mapping.remove(-1) is True, "negative key removed"
    assert mapping.remove(-1) is False, "second remove of a missing key returns False"
    assert mapping.size() == KEYS, "size restored after the negative-key check"

    # Phase 2 (concurrent): 4 threads upsert keys 0..49 (same value per key in every
    # thread, so any interleaving ends in the same state) and add disjoint keys 100..199.
    def worker(thread_index):
        for key in range(KEYS):
            mapping.put(key, key * 3 + 1000)
        for key in range(100 + thread_index * 25, 100 + (thread_index + 1) * 25):
            mapping.put(key, key + 1)

    workers = [threading.Thread(target=worker, args=(index,), daemon=True)
               for index in range(THREADS)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(10)
        assert not thread.is_alive(), "upsert worker did not finish in time"

    assert mapping.size() == 150, "every concurrent upsert persisted exactly once"
    for key in range(KEYS):
        assert mapping.get(key) == key * 3 + 1000, "overlapping upsert result at {}".format(key)
        assert mapping.shards_containing(key) == [mapping.shard_index(key)]
    for key in range(100, 200):
        assert mapping.get(key) == key + 1, "disjoint upsert result at {}".format(key)
        assert mapping.shards_containing(key) == [mapping.shard_index(key)]

    # Reference map built single-threaded with the same logical operations.
    reference = {}
    for key in range(KEYS):
        reference[key] = key * 3
    for key in range(KEYS):
        reference[key] = key * 3 + 1000
    for thread_index in range(THREADS):
        for key in range(100 + thread_index * 25, 125 + thread_index * 25):
            reference[key] = key + 1

    assert mapping.keys() == sorted(reference.keys()), "same key set as the reference"
    assert mapping.size() == len(reference), "same size as the reference"
    for key in range(-10, 260):
        assert mapping.get(key) == reference.get(key), "reference match at {}".format(key)

    # Removal path, then a full re-comparison against the reference.
    assert mapping.remove(150) is True, "remove existing key"
    assert mapping.get(150) is None, "removed key reads as missing"
    assert mapping.remove(150) is False, "remove missing key"
    del reference[150]
    assert mapping.size() == len(reference)
    for key in range(-10, 260):
        assert mapping.get(key) == reference.get(key), "post-removal match at {}".format(key)


if __name__ == "__main__":
    main()
    print("PASS")
