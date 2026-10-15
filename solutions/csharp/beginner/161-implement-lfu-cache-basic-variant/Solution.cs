// Task 161 - Implement LFU cache (basic variant). (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class LFUCache
    {
        private readonly int capacity;
        private readonly Dictionary<int, long[]> data = new Dictionary<int, long[]>(); // {value, freq, seq}
        private long tick = 0;

        public LFUCache(int capacity)
        {
            if (capacity <= 0) throw new ArgumentException("capacity must be positive");
            this.capacity = capacity;
        }

        public int Get(int key)
        {
            if (!data.TryGetValue(key, out var entry)) return -1;
            entry[1]++;
            entry[2] = tick++;
            return (int)entry[0];
        }

        public void Put(int key, int value)
        {
            if (data.TryGetValue(key, out var entry))
            {
                entry[0] = value;
                entry[1]++;
                entry[2] = tick++;
                return;
            }
            if (data.Count >= capacity)
            {
                int victim = -1;
                long bestFreq = long.MaxValue, bestSeq = long.MaxValue;
                foreach (var kv in data)
                {
                    long freq = kv.Value[1], seq = kv.Value[2];
                    if (freq < bestFreq || (freq == bestFreq && seq < bestSeq))
                    {
                        bestFreq = freq;
                        bestSeq = seq;
                        victim = kv.Key;
                    }
                }
                data.Remove(victim);
            }
            data[key] = new long[] { value, 1, tick++ };
        }
    }

    public static void Main()
    {
        var cache = new LFUCache(2);
        cache.Put(1, 1);
        cache.Put(2, 2);
        Check(cache.Get(1) == 1, "get 1");
        cache.Put(3, 3);
        Check(cache.Get(2) == -1, "2 evicted (lowest freq)");
        Check(cache.Get(3) == 3, "get 3");
        cache.Put(4, 4);
        Check(cache.Get(1) == -1, "tie on freq -> LRU key 1 evicted");
        Check(cache.Get(3) == 3 && cache.Get(4) == 4, "3 and 4 remain");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
