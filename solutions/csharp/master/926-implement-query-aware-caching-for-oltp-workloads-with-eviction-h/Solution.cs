// Task 926 - Implement query-aware caching for OLTP workloads with eviction hints. (Tier: Master, Beyond — Distributed Systems & Databases)
//
// A small query cache with a capacity, exercised under a skewed OLTP workload. Every query carries a *hint*: a
// reuse class, where hot queries are expected to repeat and cold queries are one-off scans. The LRU baseline evicts
// the least recently used entry; the hint-aware policy refuses to admit a cold miss (so a scan cannot flush the
// working set) and, when it must evict, prefers the cached entry with the worst hint class, breaking ties by
// recency. A recency list is maintained independently of the cache dictionary so the LRU invariant can be checked
// after every operation against an oracle, and writes invalidate the cached value of the written key.
//
// Simplifications: single-process cache, one value per key, no locking, hints are given (not learned), the workload
// is a synthetic two-class skew from a hand-written splitmix64, statistics are counted rather than timed.
// Verified: the cached key set and every eviction match the independently maintained LRU oracle, capacity is never
// exceeded, the hint-aware policy beats LRU on the skewed workload, writes invalidate, a capacity-1 cache behaves as
// a single slot, a zero capacity is rejected, and identical seeds reproduce identical hit counts.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class QueryCache
{
    internal readonly int Capacity;
    internal readonly bool HintAware;
    internal readonly int[] Hint;                                     // per key: 0 hot, 1 cold
    internal readonly Dictionary<int, int> Values = new Dictionary<int, int>();
    internal readonly List<int> Recency = new List<int>();            // least recently used first
    internal int Hits, Misses, Admissions, Rejections;
    internal QueryCache(int capacity, bool hintAware, int[] hint)
    {
        if (capacity <= 0) throw new Exception("cache capacity must be positive");
        Capacity = capacity; HintAware = hintAware; Hint = hint;
    }
    internal int? Get(int key)
    {
        if (!Values.TryGetValue(key, out int value)) { Misses++; return null; }
        Hits++; Touch(key);
        return value;
    }
    internal bool Put(int key, int value)
    {
        if (HintAware && Hint[key] == 1 && !Values.ContainsKey(key)) { Rejections++; Misses++; return false; }  // do not admit a scan
        if (!Values.ContainsKey(key) && Values.Count >= Capacity) Evict();
        Admissions++;
        Values[key] = value;
        Touch(key);
        return true;
    }
    internal void Invalidate(int key)
    {
        if (Values.Remove(key)) Recency.Remove(key);
    }
    private void Touch(int key)
    {
        Recency.Remove(key);
        Recency.Add(key);
    }
    /** Eviction: LRU for the baseline, worst hint class first (recency as the tie-break) for the hint-aware policy. */
    private void Evict()
    {
        int victim = Recency[0];
        if (HintAware)
            foreach (int key in Recency)
                if (Hint[key] > Hint[victim] || (Hint[key] == Hint[victim] && Recency.IndexOf(key) < Recency.IndexOf(victim))) victim = key;
        Values.Remove(victim);
        Recency.Remove(victim);
    }
    internal bool Sanity() => Values.Count <= Capacity && Values.Count == Recency.Count && Recency.Distinct().Count() == Recency.Count;
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 926UL;
    /** Two-class skew: 30% of the accesses hit the small hot set, the rest scan a wide cold set once. */
    private static int NextKey(ref ulong state, int[] hint, int hot, int cold)
    {
        double u = Rng.Signed(ref state);
        if (u < 0.3) return 1 + (int)(Rng.Next(ref state) % (ulong)hot);
        int key = hot + 1 + (int)(Rng.Next(ref state) % (ulong)cold);
        hint[key] = 1;
        return key;
    }
    public static void Main()
    {
        const int hot = 12, cold = 300, capacity = 16;
        var hint = new int[hot + cold + 1];
        // 1. LRU correctness against an independently maintained oracle
        var lru = new QueryCache(capacity, hintAware: false, hint);
        var oracleRecency = new List<int>();
        var oracleValues = new Dictionary<int, int>();
        int oracleMismatches = 0, evictions = 0;
        _state = 926UL;
        for (int step = 0; step < 4000; step++)
        {
            int key = NextKey(ref _state, hint, hot, cold);
            int? hit = lru.Get(key);
            if (hit.HasValue) Check(hit.Value == key * 7 % 1000, "cached value matches the computed value");
            else lru.Put(key, key * 7 % 1000);
            // the oracle mirrors the same access on its own recency list and dictionary
            if (oracleValues.ContainsKey(key)) { oracleRecency.Remove(key); oracleRecency.Add(key); }
            else
            {
                if (oracleValues.Count >= capacity)
                {
                    int victim = oracleRecency[0];
                    oracleValues.Remove(victim); oracleRecency.Remove(victim); evictions++;
                }
                oracleValues[key] = key * 7 % 1000; oracleRecency.Add(key);
            }
            if (!lru.Recency.SequenceEqual(oracleRecency) || lru.Values.Count != oracleValues.Count) oracleMismatches++;
            if (!lru.Sanity()) oracleMismatches++;
        }
        Check(oracleMismatches == 0, "4000 accesses keep the cache identical to the LRU oracle (" + evictions + " oracle evictions)");
        Check(lru.Hits + lru.Misses == 4000, "every access is accounted for as a hit or a miss");

        // 2. the hint-aware policy beats LRU on the same skewed workload
        var aware = new QueryCache(capacity, hintAware: true, hint);
        _state = 926UL;
        int hotAdmissions = 0;
        for (int step = 0; step < 4000; step++)
        {
            int key = NextKey(ref _state, hint, hot, cold);
            int? hit = aware.Get(key);
            if (!hit.HasValue)
            {
                if (aware.Put(key, key * 7 % 1000) && key <= hot) hotAdmissions++;
            }
        }
        Check(aware.Hits > lru.Hits, "hint-aware hits " + aware.Hits + " > LRU hits " + lru.Hits + " on the same workload");
        Check(aware.Rejections > 0, "the hint-aware policy rejected " + aware.Rejections + " cold admissions");
        Check(aware.Sanity(), "the hint-aware cache respects its capacity and bookkeeping");
        Check(hotAdmissions > 0 && aware.Values.Keys.All(k => k <= hot || hint[k] == 0),
            "cached keys are the hot working set, not one-off scans");

        // 3. invalidation on write and a single-slot cache
        var small = new QueryCache(8, hintAware: false, hint);
        small.Put(3, 21);
        small.Get(3);
        int hitsBefore = small.Hits;
        small.Invalidate(3);
        Check(small.Get(3) == null && small.Hits == hitsBefore, "a write invalidates the cached value (the next read misses)");
        var single = new QueryCache(1, hintAware: false, hint);
        single.Put(1, 1); single.Put(2, 2);
        Check(single.Values.Count == 1 && single.Get(1) == null && single.Get(2) == 2,
            "a capacity-1 cache keeps only the most recent key (boundary)");
        bool rejected = false;
        try { new QueryCache(0, hintAware: false, hint); } catch (Exception) { rejected = true; }
        Check(rejected, "a zero capacity is rejected at construction (boundary)");

        // 4. reproducibility
        _state = 926UL;
        var replay = new QueryCache(capacity, hintAware: false, hint);
        for (int step = 0; step < 500; step++)
        {
            int key = NextKey(ref _state, hint, hot, cold);
            if (!replay.Get(key).HasValue) replay.Put(key, key * 7 % 1000);
        }
        _state = 926UL;
        int expectedHits = 0;
        var mirror = new QueryCache(capacity, hintAware: false, hint);
        for (int step = 0; step < 500; step++)
        {
            int key = NextKey(ref _state, hint, hot, cold);
            if (mirror.Get(key).HasValue) expectedHits++;
            else mirror.Put(key, key * 7 % 1000);
        }
        Check(replay.Hits == expectedHits, "the same seed reproduces the same hit count (" + replay.Hits + ")");

        Console.WriteLine("LRU: hits " + lru.Hits + ", misses " + lru.Misses + ", " + evictions + " oracle evictions, 0 mismatches");
        Console.WriteLine("hint-aware: hits " + aware.Hits + " (" + aware.Rejections + " cold admissions rejected, " + hotAdmissions +
            " hot admits); capacity " + capacity + ", hot " + hot + ", cold " + cold + " keys");
        Console.WriteLine("PASS");
    }
}
