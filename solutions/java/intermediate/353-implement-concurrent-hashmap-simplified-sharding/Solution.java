/** Task 353 - Implement concurrent hashmap simplified (sharding). (Tier: Intermediate, CTCI Ch 15) */
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

public class Solution {
    static final int SHARDS = 4;
    static final int THREADS = 4;
    static final int KEYS = 50;

    /**
     * Hash policy (documented, identical in all four languages):
     * shard(key) = ((key % S) + S) % S, S = 4 shards. Negative keys are normalised
     * into the same range (-1 -> 3, -4 -> 0), so every key belongs to exactly one shard.
     */
    static class ShardedMap {
        private final List<Map<Integer, Integer>> shards = new ArrayList<>();
        private final List<ReentrantLock> locks = new ArrayList<>();

        ShardedMap(int numShards) {
            if (numShards <= 0) throw new IllegalArgumentException("numShards must be positive");
            for (int i = 0; i < numShards; i++) {
                shards.add(new HashMap<>());
                locks.add(new ReentrantLock());
            }
        }

        int shardIndex(int key) {
            int count = shards.size();
            return ((key % count) + count) % count;
        }

        void put(int key, int value) {
            int index = shardIndex(key);
            locks.get(index).lock();
            try {
                shards.get(index).put(key, value);
            } finally {
                locks.get(index).unlock();
            }
        }

        Integer get(int key) {
            int index = shardIndex(key);
            locks.get(index).lock();
            try {
                return shards.get(index).get(key);
            } finally {
                locks.get(index).unlock();
            }
        }

        boolean remove(int key) {
            int index = shardIndex(key);
            locks.get(index).lock();
            try {
                return shards.get(index).remove(key) != null;
            } finally {
                locks.get(index).unlock();
            }
        }

        int size() {
            int total = 0;
            for (int i = 0; i < shards.size(); i++) {
                locks.get(i).lock();
                try {
                    total += shards.get(i).size();
                } finally {
                    locks.get(i).unlock();
                }
            }
            return total;
        }

        List<Integer> keys() {
            TreeSet<Integer> result = new TreeSet<>();
            for (int i = 0; i < shards.size(); i++) {
                locks.get(i).lock();
                try {
                    result.addAll(shards.get(i).keySet());
                } finally {
                    locks.get(i).unlock();
                }
            }
            return new ArrayList<>(result);
        }

        /** Shard indexes whose map currently holds the key (single-threaded use). */
        List<Integer> shardsContaining(int key) {
            List<Integer> result = new ArrayList<>();
            for (int i = 0; i < shards.size(); i++) if (shards.get(i).containsKey(key)) result.add(i);
            return result;
        }
    }

    public static void main(String[] args) {
        ShardedMap mapping = new ShardedMap(SHARDS);

        // Phase 1 (single-threaded): keys 0..49 land in shards 0..3 with hand-checked counts.
        for (int key = 0; key < KEYS; key++) mapping.put(key, key * 3);
        check(mapping.shards.get(0).size() == 13 && mapping.shards.get(1).size() == 13
              && mapping.shards.get(2).size() == 12 && mapping.shards.get(3).size() == 12,
              "hand-checked shard counts");

        // Every inserted key is in exactly one shard, and it is the policy shard.
        for (int key = 0; key < KEYS; key++) {
            check(mapping.shardsContaining(key).equals(List.of(mapping.shardIndex(key))),
                  "key lives in exactly its policy shard: " + key);
        }

        // Negative keys are normalised by the documented policy (hand-checked: -1 -> 3, -4 -> 0).
        check(mapping.shardIndex(-1) == 3 && mapping.shardIndex(-4) == 0, "negative key policy");
        mapping.put(-1, 999);
        check(mapping.shardsContaining(-1).equals(List.of(3)), "negative key lands in one shard");
        check(mapping.get(-1) == 999, "negative key round-trips");
        check(mapping.remove(-1), "negative key removed");
        check(!mapping.remove(-1), "second remove of a missing key returns false");
        check(mapping.size() == KEYS, "size restored after the negative-key check");

        // Phase 2 (concurrent): 4 threads upsert keys 0..49 (same value per key in every
        // thread, so any interleaving ends in the same state) and add disjoint keys 100..199.
        Thread[] workers = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            final int threadIndex = t;
            workers[t] = new Thread(() -> worker(mapping, threadIndex));
            workers[t].start();
        }
        for (Thread worker : workers) {
            join(worker, 10000);
            check(!worker.isAlive(), "upsert worker did not finish in time");
        }

        check(mapping.size() == 150, "every concurrent upsert persisted exactly once");
        for (int key = 0; key < KEYS; key++) {
            check(mapping.get(key) == key * 3 + 1000, "overlapping upsert result at " + key);
            check(mapping.shardsContaining(key).equals(List.of(mapping.shardIndex(key))),
                  "one shard after upserts: " + key);
        }
        for (int key = 100; key < 200; key++) {
            check(mapping.get(key) == key + 1, "disjoint upsert result at " + key);
            check(mapping.shardsContaining(key).equals(List.of(mapping.shardIndex(key))),
                  "one shard after upserts: " + key);
        }

        // Reference map built single-threaded with the same logical operations.
        Map<Integer, Integer> reference = new HashMap<>();
        for (int key = 0; key < KEYS; key++) reference.put(key, key * 3);
        for (int key = 0; key < KEYS; key++) reference.put(key, key * 3 + 1000);
        for (int t = 0; t < THREADS; t++) {
            for (int key = 100 + t * 25; key < 125 + t * 25; key++) reference.put(key, key + 1);
        }

        check(mapping.keys().equals(new ArrayList<>(new TreeSet<>(reference.keySet()))),
              "same key set as the reference");
        check(mapping.size() == reference.size(), "same size as the reference");
        for (int key = -10; key < 260; key++) {
            check(Objects.equals(mapping.get(key), reference.get(key)), "reference match at " + key);
        }

        // Removal path, then a full re-comparison against the reference.
        check(mapping.remove(150), "remove existing key");
        check(mapping.get(150) == null, "removed key reads as missing");
        check(!mapping.remove(150), "remove missing key");
        reference.remove(150);
        check(mapping.size() == reference.size(), "size after removal");
        for (int key = -10; key < 260; key++) {
            check(Objects.equals(mapping.get(key), reference.get(key)), "post-removal match at " + key);
        }

        System.out.println("PASS");
    }

    static void worker(ShardedMap mapping, int threadIndex) {
        for (int key = 0; key < KEYS; key++) mapping.put(key, key * 3 + 1000);
        for (int key = 100 + threadIndex * 25; key < 100 + (threadIndex + 1) * 25; key++) {
            mapping.put(key, key + 1);
        }
    }

    static void join(Thread thread, long timeoutMillis) {
        try {
            thread.join(timeoutMillis);
        } catch (InterruptedException exc) {
            Thread.currentThread().interrupt();
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
