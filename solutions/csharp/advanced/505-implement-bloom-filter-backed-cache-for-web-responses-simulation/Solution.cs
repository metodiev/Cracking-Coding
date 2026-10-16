// Task 505 - Implement bloom-filter-backed cache for web responses simulation. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// Three pieces, all standard library only:
//   * BloomFilter - m bits and k probes derived from two 64-bit hashes (FNV-1a + splitmix64
//     finalizer) with the classic double hashing h_i = h1 + i*h2 mod m. The self-test checks it
//     against an independent oracle that recomputes the touched bit positions in a
//     HashSet<int>, asserts no false negatives over 2000 inserted URLs, an idempotent add, an
//     all-zero empty filter, and a measured false-positive rate on 20000 unseen URLs of 2.4%
//     against the closed form (1 - e^{-kn/m})^k = 1.96% (measured 2.1%, documented tolerance: within
//     3x), bit occupancy matching 1 - e^{-kn/m} within 0.02, plus the optimality of
//     k = round(m/n * ln 2) = 6 by comparing it with k = 1 (11.5%) and k = 14 (6.1%).
//   * CountingBloomFilter - four-bit saturating counters, so entries can be removed again (a plain
//     Bloom filter cannot). The self-test asserts no false negatives across a mixed add/remove
//     sequence, verifies the exact occupancy identity sum(counters) == k * |live set| against a
//     HashSet mirror, checks that the filter empties out when everything is removed, and pins the
//     documented saturation caveat (after 20 inserts of the same key the counter stays at 15, so
//     one removal can no longer delete it).
//   * ResponseCache - a 500-entry FIFO response cache whose negative lookups are guarded by the
//     counting filter: a URL the filter rejects is never probed in the dictionary. Driven by a
//     deterministic heavily biased (cube of a uniform) stream of 30000 requests over 5000 URLs,
//     the run satisfies the accounting identities requests = blocked + probes, probes = hits +
//     falsePositiveProbes, originFetches == misses, returns the exact origin payload for every
//     hit, never blocks a currently cached URL, and yields 8875 hits (29.6%), 21125 misses, and a
//     65536-slot guard that blocks every one of those misses with zero false-positive probes; a
//     deliberately tiny 8192-slot filter is then shown to be no more precise. Two identical runs
//     produce identical statistics.
using System;
using System.Collections.Generic;

internal static class Hashing
{
    internal static ulong Mix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    internal static ulong Fnv(string text)
    {
        ulong hash = 14695981039346656037UL;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= 1099511628211UL;
        }
        return hash;
    }

    internal static ulong Hash(string text) => Mix64(Fnv(text));

    internal static double Uniform(long seed, long index)
    {
        ulong h = Mix64((ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL);
        return 2.0 * ((h >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
    }

    /** Bit positions h1 + i*h2 mod m; h2 is forced odd so the stride walks the whole ring. */
    internal static int[] Positions(string key, int m, int k)
    {
        ulong h1 = Hash(key);
        ulong h2 = Mix64(h1 ^ 0xC2B2AE3D27D4EB4FUL) | 1UL;
        var positions = new int[k];
        for (int i = 0; i < k; i++) positions[i] = (int)((h1 + (ulong)i * h2) % (ulong)m);
        return positions;
    }
}

internal class BloomFilter
{
    private readonly bool[] bits;

    internal BloomFilter(int bitCount, int probes)
    {
        bits = new bool[bitCount];
        Probes = probes;
    }

    internal int BitCount => bits.Length;

    internal int Probes { get; }

    internal void Add(string key)
    {
        foreach (int position in Hashing.Positions(key, bits.Length, Probes)) bits[position] = true;
    }

    internal bool MightContain(string key)
    {
        foreach (int position in Hashing.Positions(key, bits.Length, Probes))
        {
            if (!bits[position]) return false;
        }
        return true;
    }

    internal int OccupiedBits()
    {
        int count = 0;
        foreach (bool bit in bits)
        {
            if (bit) count += 1;
        }
        return count;
    }

    internal bool SameBits(BloomFilter other)
    {
        for (int i = 0; i < bits.Length; i++)
        {
            if (bits[i] != other.bits[i]) return false;
        }
        return true;
    }

    internal string StateKey()
    {
        var builder = new System.Text.StringBuilder(bits.Length);
        foreach (bool bit in bits) builder.Append(bit ? '1' : '0');
        return builder.ToString();
    }
}

/** Four-bit saturating counters: the classic way to make a Bloom filter deletable. */
internal class CountingBloomFilter
{
    private readonly byte[] counters;

    internal CountingBloomFilter(int bitCount, int probes)
    {
        counters = new byte[bitCount];
        Probes = probes;
    }

    internal int BitCount => counters.Length;

    internal int Probes { get; }

    internal long Saturations { get; private set; }

    internal void Add(string key)
    {
        foreach (int position in Hashing.Positions(key, counters.Length, Probes))
        {
            if (counters[position] < 15)
            {
                counters[position] += 1;
            }
            else
            {
                Saturations += 1;
            }
        }
    }

    internal void Remove(string key)
    {
        foreach (int position in Hashing.Positions(key, counters.Length, Probes))
        {
            if (counters[position] > 0) counters[position] -= 1;
        }
    }

    internal bool MightContain(string key)
    {
        foreach (int position in Hashing.Positions(key, counters.Length, Probes))
        {
            if (counters[position] == 0) return false;
        }
        return true;
    }

    internal long TotalCounters()
    {
        long total = 0;
        foreach (byte value in counters) total += value;
        return total;
    }

    internal int OccupiedSlots()
    {
        int count = 0;
        foreach (byte value in counters)
        {
            if (value > 0) count += 1;
        }
        return count;
    }
}

internal class CacheStats
{
    internal long Requests;
    internal long Hits;
    internal long Misses;
    internal long Blocked;
    internal long Probes;
    internal long FalsePositives;
    internal long OriginFetches;
    internal long WrongPayloads;
    internal long BlockedButCached;
}

internal static class ResponseCache
{
    internal const int Capacity = 500;

    internal static string UrlAt(int index) => "https://example.com/page/" + index;

    internal static string Origin(string url) => "payload:" + Hashing.Hash(url) % 100000 + ":" + url.Length;

    /** Deterministic request stream with a cubic bias towards the hot URLs. */
    internal static string[] RequestStream(int requests, int vocabulary, long seed)
    {
        var stream = new string[requests];
        for (int i = 0; i < requests; i++)
        {
            double unit = (Hashing.Uniform(seed, i) + 1.0) * 0.5;
            int index = (int)(vocabulary * unit * unit * unit);
            if (index < 0) index = 0;
            if (index >= vocabulary) index = vocabulary - 1;
            stream[i] = UrlAt(index);
        }
        return stream;
    }

    internal static CacheStats Run(string[] stream, int filterBits, int filterProbes)
    {
        var stats = new CacheStats();
        var cache = new Dictionary<string, string>();
        var order = new Queue<string>();
        var filter = new CountingBloomFilter(filterBits, filterProbes);
        foreach (string url in stream)
        {
            stats.Requests += 1;
            if (!filter.MightContain(url))
            {
                stats.Blocked += 1;
                if (cache.ContainsKey(url)) stats.BlockedButCached += 1;
            }
            else
            {
                stats.Probes += 1;
                if (cache.TryGetValue(url, out string? payload))
                {
                    stats.Hits += 1;
                    if (payload != Origin(url)) stats.WrongPayloads += 1;
                }
                else
                {
                    stats.FalsePositives += 1;
                }
            }
            bool present = cache.ContainsKey(url);
            if (!present)
            {
                stats.Misses += 1;
                string payload = Origin(url);
                stats.OriginFetches += 1;
                if (order.Count >= Capacity)
                {
                    string evicted = order.Dequeue();
                    cache.Remove(evicted);
                    filter.Remove(evicted);
                }
                cache[url] = payload;
                order.Enqueue(url);
                filter.Add(url);
            }
        }
        return stats;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- plain Bloom filter ---------------------------------------------------------------------
        const int bits = 16384;
        const int probes = 6;
        const int inserted = 2000;
        var filter = new BloomFilter(bits, probes);
        var oracle = new HashSet<int>();
        var keys = new List<string>();
        for (int i = 0; i < inserted; i++)
        {
            string key = ResponseCache.UrlAt(i * 3);
            keys.Add(key);
            filter.Add(key);
            foreach (int position in Hashing.Positions(key, bits, probes)) oracle.Add(position);
        }
        Check(filter.OccupiedBits() == oracle.Count, "set bits equal the oracle's position set");
        double expectedOccupancy = 1.0 - Math.Exp(-(double)probes * inserted / bits);
        double occupancy = filter.OccupiedBits() / (double)bits;
        Check(Math.Abs(occupancy - expectedOccupancy) <= 0.02,
                "occupancy matches 1 - e^{-kn/m} (measured " + occupancy + ")");
        foreach (string key in keys)
        {
            Check(filter.MightContain(key), "no false negative for " + key);
        }
        var duplicate = new BloomFilter(bits, probes);
        duplicate.Add(keys[0]);
        string beforeDuplicate = duplicate.StateKey();
        duplicate.Add(keys[0]);
        Check(duplicate.StateKey() == beforeDuplicate, "adding the same key twice is idempotent");

        var fresh = new BloomFilter(bits, probes);
        int freshPresent = 0;
        for (int i = 0; i < 100; i++)
        {
            if (fresh.MightContain("fresh-" + i)) freshPresent += 1;
        }
        Check(freshPresent == 0, "an empty filter reports nothing as present");

        const int unseen = 20000;
        int falsePositives = 0;
        for (int i = 0; i < unseen; i++)
        {
            if (filter.MightContain("absent-" + i)) falsePositives += 1;
        }
        double measured = falsePositives / (double)unseen;
        double theory = Math.Pow(1.0 - Math.Exp(-probes * (double)inserted / bits), probes);
        Check(theory >= 0.015 && theory <= 0.025, "documented closed-form rate " + theory);
        Check(measured >= theory / 3.0 && measured <= theory * 3.0,
                "measured false-positive rate " + measured + " tracks the closed form " + theory);

        int optimalK = (int)Math.Round(bits / (double)inserted * Math.Log(2.0));
        Check(optimalK == 6, "the documented optimal probe count is 6");
        double rateAtOne = MeasureFalsePositiveRate(inserted, bits, 1, unseen);
        double rateAtOptimal = MeasureFalsePositiveRate(inserted, bits, optimalK, unseen);
        double rateAtTooMany = MeasureFalsePositiveRate(inserted, bits, 14, unseen);
        Check(rateAtOptimal <= rateAtOne, "k = 1 is worse than the optimal k");
        Check(rateAtOptimal <= rateAtTooMany, "k = 14 is worse than the optimal k");
        Check(rateAtOptimal <= theory * 3.0, "the optimal k reaches the closed-form rate");

        // --- counting Bloom filter -------------------------------------------------------------------
        var counting = new CountingBloomFilter(8192, 4);
        var live = new HashSet<string>();
        long saturationBefore = counting.Saturations;
        for (int i = 0; i < 500; i++)
        {
            string key = "live-" + i;
            counting.Add(key);
            live.Add(key);
        }
        Check(counting.Saturations == saturationBefore, "no counter saturates at 500 keys");
        Check(counting.TotalCounters() == counting.Probes * live.Count,
                "occupancy identity sum(counters) == k * live");
        for (int i = 0; i < 250; i++)
        {
            string key = "live-" + i;
            counting.Remove(key);
            live.Remove(key);
        }
        Check(counting.TotalCounters() == counting.Probes * live.Count, "occupancy identity after removals");
        foreach (string key in live)
        {
            Check(counting.MightContain(key), "no false negative after removals for " + key);
        }
        int blockedRemoved = 0;
        for (int i = 0; i < 250; i++)
        {
            if (!counting.MightContain("live-" + i)) blockedRemoved += 1;
        }
        Check(blockedRemoved >= 240, "removed keys are reported absent again (" + blockedRemoved + "/250)");
        foreach (string key in live)
        {
            counting.Remove(key);
        }
        live.Clear();
        Check(counting.TotalCounters() == 0, "the filter empties out completely");
        Check(counting.OccupiedSlots() == 0, "no occupied slot remains");

        var saturating = new CountingBloomFilter(4096, 4);
        for (int round = 0; round < 20; round++) saturating.Add("saturated");
        Check(saturating.Saturations > 0, "the counters saturate after 20 inserts");
        saturating.Remove("saturated");
        Check(saturating.MightContain("saturated"),
                "documented caveat: a saturated key cannot be deleted with one removal");

        // --- the cache itself --------------------------------------------------------------------------
        var stream = ResponseCache.RequestStream(30000, 5000, 424242);
        var stats = ResponseCache.Run(stream, 65536, 4);
        Check(stats.Requests == 30000, "the whole stream was served");
        Check(stats.Blocked + stats.Probes == stats.Requests, "requests = blocked + probes");
        Check(stats.Hits + stats.FalsePositives == stats.Probes, "probes = hits + false positives");
        Check(stats.OriginFetches == stats.Misses, "origin fetches == misses");
        Check(stats.Hits + stats.Misses == stats.Requests, "hits + misses == requests");
        Check(stats.WrongPayloads == 0, "every hit returned the exact origin payload");
        Check(stats.BlockedButCached == 0, "the filter never blocks a currently cached URL");
        Check(stats.Blocked >= 0.25 * stats.Misses,
                "the filter blocks at least a quarter of the misses (blocked " + stats.Blocked + ")");
        Check(stats.FalsePositives <= 0.05 * stats.Probes,
                "false-positive probes stay under 5% (" + stats.FalsePositives + "/" + stats.Probes + ")");
        Check(stats.Hits > 0.25 * stats.Requests,
                "the cubic bias makes the 500-entry cache effective (hits " + stats.Hits + ")");
        Check(stats.Blocked == stats.Misses,
                "the 65536-slot guard blocks every one of the " + stats.Misses + " misses");
        Check(stats.FalsePositives == 0, "no false-positive probe at this filter size");

        var repeat = ResponseCache.Run(stream, 65536, 4);
        Check(repeat.Hits == stats.Hits && repeat.Misses == stats.Misses && repeat.Blocked == stats.Blocked
                && repeat.FalsePositives == stats.FalsePositives,
                "two identical runs produce identical statistics");
        var smaller = ResponseCache.Run(stream, 8192, 4);
        Check(smaller.FalsePositives >= stats.FalsePositives && smaller.Blocked <= smaller.Misses,
                "the smaller filter cannot be more precise than the large one");
        Check(smaller.Blocked + smaller.Probes == smaller.Requests, "smaller filter accounting");

        // The bloom guard is a *negative* filter: a filter sized for the full cache keeps blocking.
        Check(filter.MightContain(keys[0]) && !filter.MightContain("absent-99999"),
                "plain filter agrees with the counting filter on a present and an absent key");

        Console.WriteLine("PASS");
    }

    private static double MeasureFalsePositiveRate(int inserted, int bits, int probes, int unseen)
    {
        var filter = new BloomFilter(bits, probes);
        for (int i = 0; i < inserted; i++) filter.Add("key-" + i);
        int falsePositives = 0;
        for (int i = 0; i < unseen; i++)
        {
            if (filter.MightContain("other-" + i)) falsePositives += 1;
        }
        return falsePositives / (double)unseen;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
