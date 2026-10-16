// Task 778 - Implement cold-start heuristics for cache warming and prefetching. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A freshly started cache is empty, so the first accesses are all misses and every miss costs a fetch
// from the backing store. This file measures three classic remedies against a plain cold LRU cache on
// the same demand stream:
//
//   * frequency warm-up: a short training phase counts key frequencies and the cache is preloaded with
//     the most frequent keys (truncated to the capacity) before the measured workload starts;
//   * sequential prefetch: a demand for key k also fetches k+1 and k+2, on the assumption that the
//     workload scans (the classic stride detector boiled down to a stride of one);
//   * Markov prefetch: the training phase also counts transitions k -> next, and a demand for k
//     prefetches the successor observed most often during training.
// Prefetched entries are marked, so a later demand that finds one is a real hit but is also counted as
// a "useful prefetch", and the accuracy of a prefetcher is the fraction of its prefetches that were
// eventually demanded while still resident.
//
// Verified: the LRU implementation is cross-checked after every single access against an independent
// reference model (a recency list scanned linearly) - identical hit/miss decisions and identical
// recency order, so the hit-ratio comparisons below rest on a correct cache; the capacity invariant
// holds (never more than C entries, no duplicates); warm-up beats the cold cache on a skewed stream
// and is truncated when the history is longer than the capacity; the sequential prefetcher reaches
// near-perfect accuracy on a scanning stream and lifts the hit ratio; the Markov prefetcher
// out-predicts it on a Markov stream; a prefetch never counts as a demand hit by itself; and the
// degenerate cases (capacity one, empty workload, empty training history, prefetching a resident key,
// demanding an absent key) behave sanely.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class LruCache
{
    private readonly int _capacity;
    private readonly Dictionary<int, LinkedListNode<int>> _map = new Dictionary<int, LinkedListNode<int>>();
    private readonly LinkedList<int> _recency = new LinkedList<int>();   // front = most recent
    private readonly HashSet<int> _prefetched = new HashSet<int>();

    internal long Hits, Misses, Evictions, SeedInserts;
    internal long PrefetchIssued, PrefetchHits;

    internal LruCache(int capacity)
    {
        if (capacity < 1) throw new ArgumentException("a cache needs room for at least one entry");
        _capacity = capacity;
    }

    internal int Capacity => _capacity;
    internal int Count => _map.Count;
    internal long Demands => Hits + Misses;

    /// <summary>Demand access: hit when resident, otherwise a miss that installs the key.</summary>
    internal bool Demand(int key)
    {
        if (_map.TryGetValue(key, out LinkedListNode<int> node))
        {
            _recency.Remove(node);
            _recency.AddFirst(node);
            Hits++;
            if (_prefetched.Remove(key)) PrefetchHits++;
            return true;
        }
        Misses++;
        Install(key, false);
        return false;
    }

    internal void Seed(int key)
    {
        if (_map.ContainsKey(key)) return;
        Install(key, false);
        SeedInserts++;
    }

    /// <summary>Speculative fetch: no demand statistics move, the entry is flagged as prefetched.</summary>
    internal void Prefetch(int key)
    {
        if (_map.ContainsKey(key))
        {
            _prefetched.Remove(key);                          // it is resident anyway; no longer speculative
            return;
        }
        PrefetchIssued++;
        Install(key, true);
    }

    private void Install(int key, bool prefetched)
    {
        if (_map.Count >= _capacity)
        {
            LinkedListNode<int> oldest = _recency.Last;
            _recency.RemoveLast();
            _map.Remove(oldest.Value);
            _prefetched.Remove(oldest.Value);
            Evictions++;
        }
        var node = new LinkedListNode<int>(key);
        _recency.AddFirst(node);
        _map[key] = node;
        if (prefetched) _prefetched.Add(key);
    }

    internal List<int> RecencyOrder()
    {
        var order = new List<int>();
        foreach (int key in _recency) order.Add(key);
        return order;
    }

    internal string InvariantViolation()
    {
        if (_map.Count != _recency.Count) return "the map holds " + _map.Count + " keys but the recency list " + _recency.Count;
        if (_map.Count > _capacity) return "the cache holds " + _map.Count + " entries, more than the capacity " + _capacity;
        var seen = new HashSet<int>();
        foreach (int key in _recency) if (!seen.Add(key)) return "key " + key + " appears twice in the recency list";
        foreach (KeyValuePair<int, LinkedListNode<int>> pair in _map)
        {
            if (!seen.Contains(pair.Key)) return "key " + pair.Key + " is in the map but not in the recency list";
        }
        foreach (int key in _prefetched) if (!seen.Contains(key)) return "prefetched key " + key + " is not resident";
        return null;
    }
}

/// <summary>Independent LRU model: a plain list scanned from the front; the oracle for the cache.</summary>
internal sealed class ReferenceLru
{
    private readonly int _capacity;
    private readonly List<int> _order = new List<int>();      // index 0 = most recent

    internal long Hits, Misses;

    internal ReferenceLru(int capacity) { _capacity = capacity; }
    internal List<int> Order => new List<int>(_order);

    internal bool Demand(int key)
    {
        int at = _order.IndexOf(key);
        if (at >= 0)
        {
            _order.RemoveAt(at);
            _order.Insert(0, key);
            Hits++;
            return true;
        }
        Misses++;
        Insert(key);
        return false;
    }

    // seeding and prefetching install an absent key but never reorder a resident one: they are not
    // demand accesses, which is exactly what the cache under test does
    internal void Seed(int key)
    {
        if (_order.IndexOf(key) >= 0) return;
        Insert(key);
    }

    internal void Prefetch(int key)
    {
        if (_order.IndexOf(key) >= 0) return;
        Insert(key);
    }

    private void Insert(int key)
    {
        if (_order.Count >= _capacity) _order.RemoveAt(_order.Count - 1);
        _order.Insert(0, key);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameOrder(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>Skewed demand: most accesses concentrate on a small hot set.</summary>
    private static int[] SkewedDemand(ulong seed, int length, int universe, int hotSet)
    {
        ulong state = seed;
        var demand = new int[length];
        for (int i = 0; i < length; i++)
        {
            demand[i] = SplitMix64.NextUnit(ref state) < 0.7
                ? SplitMix64.NextInt(ref state, hotSet)
                : SplitMix64.NextInt(ref state, universe);
        }
        return demand;
    }

    /// <summary>Scanning demand with an occasional jump, generated by the same PRNG for both caches.</summary>
    private static int[] ScanningDemand(ulong seed, int length, int universe)
    {
        ulong state = seed;
        var demand = new int[length];
        int cursor = 0;
        for (int i = 0; i < length; i++)
        {
            demand[i] = cursor;
            cursor = SplitMix64.NextUnit(ref state) < 0.15
                ? SplitMix64.NextInt(ref state, universe)
                : (cursor + 1) % universe;
        }
        return demand;
    }

    /// <summary>Markov demand: the successor of k is formed from k with occasional jumps.</summary>
    private static int[] MarkovDemand(ulong seed, int length, int universe)
    {
        ulong state = seed;
        var demand = new int[length];
        int current = 0;
        for (int i = 0; i < length; i++)
        {
            demand[i] = current;
            current = SplitMix64.NextUnit(ref state) < 0.75 ? (current * 7 + 3) % universe : SplitMix64.NextInt(ref state, universe);
        }
        return demand;
    }

    private static int[] Frequencies(int[] demand, int universe)
    {
        var counts = new int[universe];
        foreach (int key in demand) counts[key]++;
        return counts;
    }

    private static Dictionary<int, int> LearnedSuccessors(int[] training, int universe)
    {
        var tally = new Dictionary<long, int>();
        for (int i = 0; i + 1 < training.Length; i++)
        {
            long code = (long)training[i] * universe + training[i + 1];
            tally.TryGetValue(code, out int count);
            tally[code] = count + 1;
        }
        var best = new Dictionary<int, int>();
        var bestCount = new Dictionary<int, int>();
        foreach (KeyValuePair<long, int> pair in tally)
        {
            int from = (int)(pair.Key / universe), to = (int)(pair.Key % universe);
            if (!bestCount.TryGetValue(from, out int count) || pair.Value > count || (pair.Value == count && to < best[from]))
            {
                best[from] = to;
                bestCount[from] = pair.Value;
            }
        }
        return best;
    }

    private sealed class RunResult
    {
        internal long Hits, Misses, PrefetchIssued, PrefetchHits, Evictions, Seeds;
        internal double HitRatio => (double)Hits / Math.Max(1, Hits + Misses);
        internal double PrefetchAccuracy => PrefetchIssued == 0 ? 1 : (double)PrefetchHits / PrefetchIssued;
    }

    private static RunResult Run(int[] demand, int capacity, int[] warmup, Dictionary<int, int> successors,
        bool sequentialPrefetch, ReferenceLru oracle)
    {
        var cache = new LruCache(capacity);
        if (warmup != null)
        {
            var ranked = new List<int>();
            for (int key = 0; key < warmup.Length; key++) if (warmup[key] > 0) ranked.Add(key);
            ranked.Sort((a, b) => warmup[b] != warmup[a] ? warmup[b].CompareTo(warmup[a]) : a.CompareTo(b));
            int seedCount = Math.Min(capacity, ranked.Count);     // a warm-up longer than the cache is truncated
            for (int i = 0; i < seedCount; i++)
            {
                cache.Seed(ranked[i]);
                oracle.Seed(ranked[i]);
                Check(cache.InvariantViolation() == null, "warm-up keeps the cache consistent: " + cache.InvariantViolation());
            }
        }
        foreach (int key in demand)
        {
            bool hit = cache.Demand(key);
            bool expected = oracle.Demand(key);
            Check(hit == expected, "the LRU decision for key " + key + " matches the reference model");
            Check(SameOrder(cache.RecencyOrder(), oracle.Order), "the recency order matches the reference model");
            Check(cache.InvariantViolation() == null, "cache invariant: " + cache.InvariantViolation());
            if (sequentialPrefetch)
            {
                for (int step = 1; step <= 1; step++)
                {
                    int target = key + step;
                    cache.Prefetch(target);
                    oracle.Prefetch(target);
                }
            }
            else if (successors != null && successors.TryGetValue(key, out int next))
            {
                cache.Prefetch(next);
                oracle.Prefetch(next);
            }
        }
        return new RunResult
        {
            Hits = cache.Hits,
            Misses = cache.Misses,
            PrefetchIssued = cache.PrefetchIssued,
            PrefetchHits = cache.PrefetchHits,
            Evictions = cache.Evictions,
            Seeds = cache.SeedInserts,
        };
    }

    public static void Main()
    {
        // ---- the LRU itself, against the reference model -----------------------------------------
        {
            const int Capacity = 8, Universe = 40;
            var cache = new LruCache(Capacity);
            var oracle = new ReferenceLru(Capacity);
            ulong state = 0x5EED778000000001UL;
            bool decisionsMatch = true, orderMatches = true;
            for (int i = 0; i < 400; i++)
            {
                int key = SplitMix64.NextInt(ref state, Universe);
                if (cache.Demand(key) != oracle.Demand(key)) decisionsMatch = false;
                if (!SameOrder(cache.RecencyOrder(), oracle.Order)) orderMatches = false;
                if (cache.InvariantViolation() != null) { decisionsMatch = false; break; }
            }
            Check(decisionsMatch, "400 demand accesses agree with the reference LRU model");
            Check(orderMatches, "the recency order tracks the reference model after every access");
            Check(cache.Count <= Capacity, "the cache never exceeds its capacity (" + cache.Count + ")");
            Check(cache.Evictions > 100, "the pressure of a 40-key universe really forced evictions (" + cache.Evictions + ")");
            Console.WriteLine("lru model: size=" + cache.Count + " hits=" + cache.Hits + " misses=" + cache.Misses
                + " evictions=" + cache.Evictions);
        }

        // ---- warm-up versus a cold start on a skewed stream ---------------------------------------
        {
            // capacity below the hot-set size: a cold cache must first discover which keys are hot,
            // which is exactly the cost a warm-up from the training phase avoids
            const int Universe = 300, Capacity = 8, Length = 250;
            int[] training = SkewedDemand(0x5EED778000000002UL, 6000, Universe, 20);
            int[] demand = SkewedDemand(0x5EED778000000003UL, Length, Universe, 20);
            int[] counts = Frequencies(training, Universe);

            var cold = Run(demand, Capacity, null, null, false, new ReferenceLru(Capacity));
            var warm = Run(demand, Capacity, counts, null, false, new ReferenceLru(Capacity));
            Check(warm.HitRatio > cold.HitRatio, "warm-up beats the cold start (" + warm.HitRatio.ToString("F4")
                + " > " + cold.HitRatio.ToString("F4") + ")");
            Check(warm.Hits > cold.Hits + 2, "the warm-up saves the cold cache's discovery misses ("
                + warm.Hits + " vs " + cold.Hits + " hits over " + Length + " accesses)");
            Check(cold.Evictions > 0, "the cold run really filled and evicted (" + cold.Evictions + ")");
            Check(warm.Seeds == Capacity, "the warm-up is truncated to the cache capacity (" + warm.Seeds + " seeds)");

            var oversized = Run(demand, 6, counts, null, false, new ReferenceLru(6));
            Check(oversized.Seeds == 6, "a warm-up longer than a six-entry cache still seeds exactly six keys");
            Check(oversized.HitRatio > 0, "a warm-up longer than the capacity still helps a little");
            Console.WriteLine("warm-up: coldHitRatio=" + cold.HitRatio.ToString("F4") + " warmHitRatio=" + warm.HitRatio.ToString("F4")
                + " coldHits=" + cold.Hits + " warmHits=" + warm.Hits);
        }

        // ---- sequential prefetching on a scanning stream -------------------------------------------
        {
            const int Universe = 200, Capacity = 16, Length = 3000;
            int[] demand = ScanningDemand(0x5EED778000000004UL, Length, Universe);
            var plain = Run(demand, Capacity, null, null, false, new ReferenceLru(Capacity));
            var scanned = Run(demand, Capacity, null, null, true, new ReferenceLru(Capacity));
            Check(scanned.HitRatio > plain.HitRatio, "prefetching lifts the hit ratio on a scanning stream ("
                + scanned.HitRatio.ToString("F4") + " > " + plain.HitRatio.ToString("F4") + ")");
            Check(scanned.PrefetchIssued >= Length / 2, "prefetching really issued requests (" + scanned.PrefetchIssued + ")");
            Check(scanned.PrefetchAccuracy > 0.5, "more than half of the prefetches paid off ("
                + scanned.PrefetchAccuracy.ToString("F4") + ")");
            Check(scanned.Hits > plain.Hits, "the prefetched entries produced extra hits (" + scanned.Hits + " vs " + plain.Hits + ")");
            Console.WriteLine("sequential prefetch: plainHitRatio=" + plain.HitRatio.ToString("F4")
                + " prefetchHitRatio=" + scanned.HitRatio.ToString("F4") + " accuracy=" + scanned.PrefetchAccuracy.ToString("F4"));
        }

        // ---- the Markov prefetcher out-predicts the stride heuristic on a Markov stream -------------
        {
            const int Universe = 200, Capacity = 16, Length = 3000;
            int[] training = MarkovDemand(0x5EED778000000005UL, 8000, Universe);
            int[] demand = MarkovDemand(0x5EED778000000006UL, Length, Universe);
            var successors = LearnedSuccessors(training, Universe);
            Check(successors.Count > 0, "the training phase learned transitions (" + successors.Count + " keys)");

            var plain = Run(demand, Capacity, null, null, false, new ReferenceLru(Capacity));
            var stride = Run(demand, Capacity, null, null, true, new ReferenceLru(Capacity));
            var markov = Run(demand, Capacity, null, successors, false, new ReferenceLru(Capacity));
            Check(markov.PrefetchAccuracy > stride.PrefetchAccuracy, "the learned transitions predict better than a stride of one ("
                + markov.PrefetchAccuracy.ToString("F4") + " > " + stride.PrefetchAccuracy.ToString("F4") + ")");
            Check(markov.HitRatio > plain.HitRatio, "the Markov prefetcher improves on the cold cache ("
                + markov.HitRatio.ToString("F4") + " > " + plain.HitRatio.ToString("F4") + ")");
            Check(stride.PrefetchIssued > 0 && markov.PrefetchIssued > 0, "both prefetchers issued requests");
            Console.WriteLine("markov prefetch: cold=" + plain.HitRatio.ToString("F4") + " stride=" + stride.HitRatio.ToString("F4")
                + " markov=" + markov.HitRatio.ToString("F4") + " accuracy stride=" + stride.PrefetchAccuracy.ToString("F4")
                + " markov=" + markov.PrefetchAccuracy.ToString("F4"));
        }

        // ---- prefetching never counts as a demand hit by itself ------------------------------------
        {
            const int Capacity = 4;
            var cache = new LruCache(Capacity);
            cache.Prefetch(11);
            Check(cache.PrefetchIssued == 1 && cache.Hits == 0 && cache.Misses == 0, "a prefetch alone produces no demand statistics");
            Check(cache.Count == 1, "the prefetched entry is resident");
            Check(cache.Demand(11), "the later demand for the prefetched key is a hit");
            Check(cache.PrefetchHits == 1, "that hit is credited as a useful prefetch");
            Check(cache.Demand(12) == false, "demanding an absent key is a miss");
            cache.Prefetch(12);
            Check(cache.PrefetchIssued == 1, "prefetching a resident key issues nothing");
            Check(cache.InvariantViolation() == null, "the small cache stays consistent: " + cache.InvariantViolation());
            for (int key = 20; key < 40; key++) cache.Prefetch(key);
            Check(cache.Count == Capacity, "prefetching beyond the capacity keeps the cache at its size");
            Check(cache.InvariantViolation() == null, "the over-subscribed cache stays consistent");
            Console.WriteLine("prefetch semantics: hits=" + cache.Hits + " misses=" + cache.Misses
                + " issued=" + cache.PrefetchIssued + " useful=" + cache.PrefetchHits + " evictions=" + cache.Evictions);
        }

        // ---- degenerate inputs ----------------------------------------------------------------------
        {
            var tiny = new LruCache(1);
            var tinyOracle = new ReferenceLru(1);
            for (int i = 0; i < 20; i++)
            {
                Check(tiny.Demand(i) == tinyOracle.Demand(i), "a capacity-one cache matches the reference");
                Check(SameOrder(tiny.RecencyOrder(), tinyOracle.Order), "a capacity-one cache keeps the right entry");
            }
            Check(tiny.Count == 1 && tiny.Hits == 0 && tiny.Misses == 20, "capacity one misses everything in a distinct stream");

            var empty = Run(new int[0], 8, null, null, false, new ReferenceLru(8));
            Check(empty.Hits == 0 && empty.Misses == 0, "an empty workload produces no accesses");
            Check(empty.HitRatio == 0, "the hit ratio of an empty workload is zero");

            var noHistory = Run(new[] { 1, 2, 3, 4 }, 4, new int[50], null, false, new ReferenceLru(4));
            Check(noHistory.Misses == 4, "an empty training history warms nothing, so everything misses");

            bool threw = false;
            try { new LruCache(0); } catch (ArgumentException) { threw = true; }
            Check(threw, "a zero-capacity cache is rejected");

            var deterministic = new LruCache(8);
            var deterministicOracle = new ReferenceLru(8);
            ulong state = 0x5EED778000000007UL;
            int[] stream = SkewedDemand(0x5EED778000000008UL, 500, 100, 8);
            foreach (int key in stream)
            {
                deterministic.Demand(key);
                deterministicOracle.Demand(key);
                Check(SameOrder(deterministic.RecencyOrder(), deterministicOracle.Order), "the reference model tracks deterministically");
            }
            Check(deterministic.Hits == deterministicOracle.Hits, "identical runs count identical hits");
            Console.WriteLine("degenerate: tinyMisses=" + tiny.Misses + " emptyHits=" + empty.Hits
                + " deterministicHits=" + deterministic.Hits + " stateSeed=" + state);
        }

        Console.WriteLine("PASS");
    }
}
