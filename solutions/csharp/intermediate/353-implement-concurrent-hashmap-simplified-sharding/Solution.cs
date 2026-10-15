// Task 353 - Implement concurrent hashmap simplified (sharding). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    private const int Shards = 4;
    private const int Threads = 4;
    private const int Keys = 50;

    /// <summary>
    /// Hash policy (documented, identical in all four languages): shard(key) =
    /// ((key % S) + S) % S, S = 4 shards. Negative keys are normalised into the same
    /// range (-1 -> 3, -4 -> 0), so every key belongs to exactly one shard.
    /// </summary>
    private class ShardedMap
    {
        private readonly Dictionary<int, int>[] shards;
        private readonly object[] locks;

        public ShardedMap(int numShards)
        {
            if (numShards <= 0) throw new ArgumentException("numShards must be positive");
            shards = new Dictionary<int, int>[numShards];
            locks = new object[numShards];
            for (int i = 0; i < numShards; i++)
            {
                shards[i] = new Dictionary<int, int>();
                locks[i] = new object();
            }
        }

        public int ShardIndex(int key)
        {
            int count = shards.Length;
            return ((key % count) + count) % count;
        }

        public void Put(int key, int value)
        {
            int index = ShardIndex(key);
            lock (locks[index]) { shards[index][key] = value; }
        }

        public int? Get(int key)
        {
            int index = ShardIndex(key);
            lock (locks[index]) { return shards[index].TryGetValue(key, out int value) ? value : (int?)null; }
        }

        public bool Remove(int key)
        {
            int index = ShardIndex(key);
            lock (locks[index]) { return shards[index].Remove(key); }
        }

        public int Size()
        {
            int total = 0;
            for (int i = 0; i < shards.Length; i++)
            {
                lock (locks[i]) { total += shards[i].Count; }
            }
            return total;
        }

        public List<int> Keys()
        {
            var result = new List<int>();
            for (int i = 0; i < shards.Length; i++)
            {
                lock (locks[i]) { result.AddRange(shards[i].Keys); }
            }
            result.Sort();
            return result;
        }

        /// <summary>Shard indexes whose dictionary currently holds the key (single-threaded use).</summary>
        public List<int> ShardsContaining(int key)
        {
            var result = new List<int>();
            for (int i = 0; i < shards.Length; i++) if (shards[i].ContainsKey(key)) result.Add(i);
            return result;
        }

        public int ShardSize(int index) { lock (locks[index]) { return shards[index].Count; } }
    }

    public static void Main()
    {
        var mapping = new ShardedMap(Shards);

        // Phase 1 (single-threaded): keys 0..49 land in shards 0..3 with hand-checked counts.
        for (int key = 0; key < Keys; key++) mapping.Put(key, key * 3);
        Check(mapping.ShardSize(0) == 13 && mapping.ShardSize(1) == 13
              && mapping.ShardSize(2) == 12 && mapping.ShardSize(3) == 12,
              "hand-checked shard counts");

        // Every inserted key is in exactly one shard, and it is the policy shard.
        for (int key = 0; key < Keys; key++)
        {
            Check(ShardsEqual(mapping.ShardsContaining(key), new List<int> { mapping.ShardIndex(key) }),
                  "key lives in exactly its policy shard: " + key);
        }

        // Negative keys are normalised by the documented policy (hand-checked: -1 -> 3, -4 -> 0).
        Check(mapping.ShardIndex(-1) == 3 && mapping.ShardIndex(-4) == 0, "negative key policy");
        mapping.Put(-1, 999);
        Check(ShardsEqual(mapping.ShardsContaining(-1), new List<int> { 3 }), "negative key lands in one shard");
        Check(mapping.Get(-1) == 999, "negative key round-trips");
        Check(mapping.Remove(-1), "negative key removed");
        Check(!mapping.Remove(-1), "second remove of a missing key returns false");
        Check(mapping.Size() == Keys, "size restored after the negative-key check");

        // Phase 2 (concurrent): 4 threads upsert keys 0..49 (same value per key in every
        // thread, so any interleaving ends in the same state) and add disjoint keys 100..199.
        var workers = new Thread[Threads];
        for (int t = 0; t < Threads; t++)
        {
            int threadIndex = t;
            workers[t] = new Thread(() => Worker(mapping, threadIndex));
            workers[t].Start();
        }
        foreach (var worker in workers)
        {
            worker.Join(10000);                    // timeout sanity: must terminate
            Check(!worker.IsAlive, "upsert worker did not finish in time");
        }

        Check(mapping.Size() == 150, "every concurrent upsert persisted exactly once");
        for (int key = 0; key < Keys; key++)
        {
            Check(mapping.Get(key) == key * 3 + 1000, "overlapping upsert result at " + key);
            Check(ShardsEqual(mapping.ShardsContaining(key), new List<int> { mapping.ShardIndex(key) }),
                  "one shard after upserts: " + key);
        }
        for (int key = 100; key < 200; key++)
        {
            Check(mapping.Get(key) == key + 1, "disjoint upsert result at " + key);
            Check(ShardsEqual(mapping.ShardsContaining(key), new List<int> { mapping.ShardIndex(key) }),
                  "one shard after upserts: " + key);
        }

        // Reference map built single-threaded with the same logical operations.
        var reference = new Dictionary<int, int>();
        for (int key = 0; key < Keys; key++) reference[key] = key * 3;
        for (int key = 0; key < Keys; key++) reference[key] = key * 3 + 1000;
        for (int t = 0; t < Threads; t++)
        {
            for (int key = 100 + t * 25; key < 125 + t * 25; key++) reference[key] = key + 1;
        }

        var referenceKeys = new List<int>(reference.Keys);
        referenceKeys.Sort();
        Check(ListsEqual(mapping.Keys(), referenceKeys), "same key set as the reference");
        Check(mapping.Size() == reference.Count, "same size as the reference");
        for (int key = -10; key < 260; key++)
        {
            Check(NullableEquals(mapping.Get(key), reference.TryGetValue(key, out int value) ? value : (int?)null),
                  "reference match at " + key);
        }

        // Removal path, then a full re-comparison against the reference.
        Check(mapping.Remove(150), "remove existing key");
        Check(mapping.Get(150) == null, "removed key reads as missing");
        Check(!mapping.Remove(150), "remove missing key");
        reference.Remove(150);
        Check(mapping.Size() == reference.Count, "size after removal");
        for (int key = -10; key < 260; key++)
        {
            Check(NullableEquals(mapping.Get(key), reference.TryGetValue(key, out int value) ? value : (int?)null),
                  "post-removal match at " + key);
        }

        Console.WriteLine("PASS");
    }

    private static void Worker(ShardedMap mapping, int threadIndex)
    {
        for (int key = 0; key < Keys; key++) mapping.Put(key, key * 3 + 1000);
        for (int key = 100 + threadIndex * 25; key < 100 + (threadIndex + 1) * 25; key++)
        {
            mapping.Put(key, key + 1);
        }
    }

    private static bool ListsEqual(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static bool ShardsEqual(List<int> a, List<int> b) { return ListsEqual(a, b); }

    private static bool NullableEquals(int? a, int? b)
    {
        if (a.HasValue != b.HasValue) return false;
        return !a.HasValue || a.Value == b.Value;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
