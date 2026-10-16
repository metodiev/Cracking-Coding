// Task 770 - Implement streaming algorithms for heavy hitters under memory constraints. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
//
// A heavy hitter is a stream item whose frequency exceeds a fraction of the stream length. Three
// sublinear-space counters are implemented and each is checked against the exact frequency table
// computed from the very same stream (the independent oracle):
//
//   * MisraGries  - c counters, no sketch of the items; on overflow every counter is decremented and
//                   emptied slots are forgotten. Guarantees (i) every item with f > n/(c+1) is
//                   monitored and (ii) for monitored items 0 <= f - f_hat <= n/(c+1).
//   * SpaceSaving - c counters that are never empty; a new item steals the slot of the current
//                   minimum and starts counting one above it. Guarantees f_hat >= f and
//                   f_hat - f <= n/c, and min counter <= n/c at all times.
//   * CountMinSketch - d rows of w counters with conservative updates, so the estimate is a min over
//                   rows and always >= f. With w = e/eps the classic (eps, delta = e^-d) bound
//                   |f_hat - f| <= eps*n holds with probability 1 - delta; since d rows use the same
//                   hash family, the d = 5 estimate is pointwise <= the d = 1 estimate, which is
//                   checked as an invariant.
//
// Documented simplicity: one pass, one machine, whole-stream (tumbling) frequencies - no sliding
// windows, no decaying counters, no mergeable/distributed sketch summary, and item identities are
// integers hashed with a hand-written splitmix64 mixer instead of a universal hash family. Verified:
// the monotone error guarantees above against exact counts on a skewed 2000-item stream, the
// (eps, delta) failure rate of the sketch measured over 300 independent hash seeds, the error scale
// against the predicted n/w collision mass, plus the degenerate streams (empty, one repeated key,
// all distinct keys, a single counter).
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
}

internal static class StreamGen
{
    /// <summary>Power-of-uniform key draw: the density piles up on the low keys, so real heavy hitters exist.</summary>
    internal static int[] Skewed(ulong seed, int length, int domain, double exponent)
    {
        ulong state = seed;
        var stream = new int[length];
        for (int i = 0; i < length; i++)
        {
            ulong raw = SplitMix64.Next(ref state);
            double unit = (raw >> 11) * (1.0 / 9007199254740992.0);
            int key = (int)(domain * Math.Pow(unit, exponent));
            stream[i] = key >= domain ? domain - 1 : key;
        }
        return stream;
    }

    internal static int[] ExactCounts(int[] stream, int domain)
    {
        var counts = new int[domain];
        foreach (int key in stream) counts[key]++;
        return counts;
    }
}

/// <summary>Misra-Gries: c counters, decrement-all on overflow.</summary>
internal sealed class MisraGries
{
    private const int Empty = int.MinValue;
    private readonly int[] _keys;
    private readonly long[] _counts;
    private int _used;

    internal MisraGries(int counters)
    {
        if (counters < 1) throw new ArgumentException("at least one counter is required");
        _keys = new int[counters];
        _counts = new long[counters];
        for (int i = 0; i < counters; i++) _keys[i] = Empty;
    }

    internal int Slots => _keys.Length;
    internal int Monitored => _used;
    internal long Decrements;

    internal void Add(int key)
    {
        for (int i = 0; i < _used; i++)
        {
            if (_keys[i] == key) { _counts[i]++; return; }
        }
        if (_used < _keys.Length) { _keys[_used] = key; _counts[_used] = 1; _used++; return; }
        int survivors = 0;
        for (int i = 0; i < _used; i++)
        {
            _counts[i]--;
            Decrements++;
            if (_counts[i] == 0) continue;
            _keys[survivors] = _keys[i];
            _counts[survivors] = _counts[i];
            survivors++;
        }
        for (int i = survivors; i < _used; i++) _keys[i] = Empty;
        _used = survivors;
    }

    internal long Estimate(int key)
    {
        for (int i = 0; i < _used; i++) if (_keys[i] == key) return _counts[i];
        return 0;
    }

    internal List<int> MonitoredKeys()
    {
        var keys = new List<int>();
        for (int i = 0; i < _used; i++) keys.Add(_keys[i]);
        keys.Sort();
        return keys;
    }

    internal long CounterSum()
    {
        long sum = 0;
        for (int i = 0; i < _used; i++) sum += _counts[i];
        return sum;
    }
}

/// <summary>Space-Saving: c always-full counters; a newcomer takes over the minimum slot.</summary>
internal sealed class SpaceSaving
{
    private readonly int[] _keys;
    private readonly long[] _counts;
    private int _used;
    internal long Replacements;

    internal SpaceSaving(int counters)
    {
        if (counters < 1) throw new ArgumentException("at least one counter is required");
        _keys = new int[counters];
        _counts = new long[counters];
    }

    internal int Slots => _keys.Length;
    internal int Tracked => _used;

    internal void Add(int key)
    {
        for (int i = 0; i < _used; i++)
        {
            if (_keys[i] == key) { _counts[i]++; return; }
        }
        if (_used < _keys.Length) { _keys[_used] = key; _counts[_used] = 1; _used++; return; }
        int slot = 0;
        for (int i = 1; i < _used; i++) if (_counts[i] < _counts[slot]) slot = i;
        _counts[slot]++;
        _keys[slot] = key;
        Replacements++;
    }

    internal long Estimate(int key)
    {
        for (int i = 0; i < _used; i++) if (_keys[i] == key) return _counts[i];
        return 0;
    }

    internal long MinCount()
    {
        long min = long.MaxValue;
        for (int i = 0; i < _used; i++) if (_counts[i] < min) min = _counts[i];
        return _used == 0 ? 0 : min;
    }

    internal long CounterSum()
    {
        long sum = 0;
        for (int i = 0; i < _used; i++) sum += _counts[i];
        return sum;
    }

    internal List<int> TrackedKeys()
    {
        var keys = new List<int>();
        for (int i = 0; i < _used; i++) keys.Add(_keys[i]);
        keys.Sort();
        return keys;
    }
}

/// <summary>Count-Min sketch with conservative updates (estimate is a min over rows).</summary>
internal sealed class CountMinSketch
{
    private readonly int _width;
    private readonly int _depth;
    private readonly int[] _table;
    private readonly ulong _seed;

    internal CountMinSketch(int width, int depth, ulong seed)
    {
        if (width < 1 || depth < 1) throw new ArgumentException("width and depth must be positive");
        _width = width;
        _depth = depth;
        _seed = seed;
        _table = new int[width * depth];
    }

    internal int Width => _width;
    internal int Depth => _depth;

    /// <summary>Column that a key hashes to inside a row; exposed so the file can recompute the
    /// collision mass that bounds the sketch error.</summary>
    internal int Column(int key, int row) => Slot(key, row) % _width;

    private int Slot(int key, int row)
    {
        // row 0 of two sketches sharing a seed is the same hash, which is what makes the
        // "deeper sketch is never worse" invariant below hold pointwise.
        ulong z = (ulong)(uint)key * 0x9E3779B97F4A7C15UL + _seed + (ulong)row * 0xD1B54A32D192ED03UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return row * _width + (int)(z % (ulong)_width);
    }

    internal long Estimate(int key)
    {
        long best = long.MaxValue;
        for (int row = 0; row < _depth; row++)
        {
            long value = _table[Slot(key, row)];
            if (value < best) best = value;
        }
        return best;
    }

    /// <summary>Conservative update: only the rows that currently sit at the minimum are bumped.</summary>
    internal void Add(int key)
    {
        long min = Estimate(key);
        for (int row = 0; row < _depth; row++)
        {
            int slot = Slot(key, row);
            if (_table[slot] == min) _table[slot]++;
        }
    }

    internal long TotalCounters()
    {
        long sum = 0;
        foreach (int value in _table) sum += value;
        return sum;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Contains(List<int> sorted, int key) => sorted.BinarySearch(key) >= 0;

    public static void Main()
    {
        const int Length = 2000, Domain = 500, Counters = 20;
        int[] stream = StreamGen.Skewed(0x5EED5EED0BADF00DUL, Length, Domain, 4.0);
        int[] exact = StreamGen.ExactCounts(stream, Domain);

        // ---- Misra-Gries -----------------------------------------------------------------------
        {
            var mg = new MisraGries(Counters);
            foreach (int key in stream) mg.Add(key);
            double threshold = (double)Length / (Counters + 1);

            bool contained = true, errorBounded = true, positive = true;
            int trueHeavy = 0;
            var monitored = mg.MonitoredKeys();
            int worst = 0;
            for (int key = 0; key < Domain; key++)
            {
                if (exact[key] > threshold)
                {
                    trueHeavy++;
                    if (!Contains(monitored, key)) contained = false;
                }
                long estimate = mg.Estimate(key);
                if (estimate > exact[key]) positive = false;
                if (exact[key] - estimate > threshold) errorBounded = false;
                if (exact[key] - estimate > worst) worst = (int)(exact[key] - estimate);
            }
            Check(mg.Monitored <= Counters, "Misra-Gries never uses more than c = " + Counters + " counters");
            Check(trueHeavy > 0, "the skewed stream really contains f > n/(c+1) heavy hitters (" + trueHeavy + ")");
            Check(contained, "every true heavy hitter is monitored by Misra-Gries");
            Check(errorBounded, "Misra-Gries underestimates by at most n/(c+1) = " + threshold.ToString("F1") + " (worst " + worst + ")");
            Check(positive, "Misra-Gries never overestimates (counters only ever count real occurrences)");
            Check(mg.CounterSum() <= Length, "the retained counters never exceed the stream length");
            Console.WriteLine("MisraGries c=" + Counters + " monitored=" + mg.Monitored + " trueHeavy=" + trueHeavy
                + " worstUnderestimate=" + worst + " decrements=" + mg.Decrements + " counterSum=" + mg.CounterSum());
        }

        // ---- Space-Saving ------------------------------------------------------------------------
        {
            var ss = new SpaceSaving(Counters);
            foreach (int key in stream) ss.Add(key);
            double threshold = (double)Length / Counters;

            var tracked = ss.TrackedKeys();
            bool contained = true, overBounded = true, heavyNotUnder = true;
            int trueHeavy = 0;
            long worst = 0;
            for (int key = 0; key < Domain; key++)
            {
                long estimate = ss.Estimate(key);
                if (exact[key] > threshold)
                {
                    trueHeavy++;
                    if (!Contains(tracked, key)) contained = false;
                    // a heavy hitter is never the minimum slot, so its counter only ever grows
                    if (estimate < exact[key]) heavyNotUnder = false;
                }
                if (estimate - exact[key] > threshold) overBounded = false;
                if (estimate - exact[key] > worst) worst = estimate - exact[key];
            }
            Check(ss.Tracked <= Counters, "Space-Saving fills at most c slots");
            Check(contained, "every item with f > n/c is tracked by Space-Saving (" + trueHeavy + " such items)");
            Check(heavyNotUnder, "a tracked heavy hitter is never underestimated");
            Check(overBounded, "Space-Saving overestimates by at most n/c = " + threshold.ToString("F1") + " (worst " + worst + ")");
            Check(ss.MinCount() <= threshold, "the minimum counter never exceeds n/c: " + ss.MinCount() + " <= " + threshold.ToString("F1"));
            Check(ss.CounterSum() == Length, "the counters of Space-Saving account for every stream item");
            Check(ss.Replacements > 0, "the fixed number of counters really forced replacements (" + ss.Replacements + ")");
            Console.WriteLine("SpaceSaving c=" + Counters + " tracked=" + ss.Tracked + " trueHeavy=" + trueHeavy
                + " worstOverestimate=" + worst + " minCounter=" + ss.MinCount() + " replacements=" + ss.Replacements);
        }

        // ---- Count-Min sketch: invariants and the (eps, delta) promise ---------------------------
        {
            const ulong Seed = 0xA5A5A5A5DEADBEEFUL;
            var shallow = new CountMinSketch(64, 1, Seed);
            var deep = new CountMinSketch(64, 5, Seed);
            foreach (int key in stream) { shallow.Add(key); deep.Add(key); }

            // Independently recomputed collision mass: with one row the counter of a key is exactly
            // the total frequency of every key sharing its column, so error == collision mass.
            long CollisionMass(CountMinSketch sketch, int key, int row)
            {
                long mass = 0;
                for (int other = 0; other < Domain; other++)
                {
                    if (other == key) continue;
                    if (sketch.Column(other, row) == sketch.Column(key, row)) mass += exact[other];
                }
                return mass;
            }

            bool neverUnder = true, deeperNotWorse = true, exactSingleRow = true, withinCollisionMass = true;
            long shallowErr = 0, deepErr = 0;
            for (int key = 0; key < Domain; key++)
            {
                long s = shallow.Estimate(key), d = deep.Estimate(key);
                if (s < exact[key] || d < exact[key]) neverUnder = false;
                if (d > s) deeperNotWorse = false;
                if (s - exact[key] != CollisionMass(shallow, key, 0)) exactSingleRow = false;
                long bestRowMass = long.MaxValue;
                for (int row = 0; row < deep.Depth; row++)
                {
                    long mass = CollisionMass(deep, key, row);
                    if (mass < bestRowMass) bestRowMass = mass;
                }
                if (d - exact[key] > bestRowMass) withinCollisionMass = false;
                shallowErr += s - exact[key];
                deepErr += d - exact[key];
            }
            Check(neverUnder, "conservative updates keep both sketches above the true counts");
            Check(deeperNotWorse, "a five-row sketch is pointwise no worse than the one-row sketch with the same seed");
            Check(exactSingleRow, "with one row the error equals the exact collision mass of that column");
            Check(withinCollisionMass, "the multi-row error stays below the smallest per-row collision mass");
            Check(deepErr < shallowErr, "taking the minimum over five rows shrinks the total collision error (" + deepErr + " < " + shallowErr + ")");
            Check(deepErr > 0, "the sketch really does collide (total error " + deepErr + " over " + Domain + " keys)");

            // Statistical check of the (eps, delta) promise: with w = e/eps and d rows the union
            // bound over all Domain queried keys is at most Domain * e^-d, so the fraction of seeds
            // that expose any over-large error must stay near that number.
            const int Seeds = 300, Width = 100, Depth = 10;
            double eps = Math.E / Width;
            double delta = Math.Pow(Math.E, -Depth);
            double unionBound = Domain * delta;
            int violatingSeeds = 0, seedsWithError = 0;
            for (int s = 0; s < Seeds; s++)
            {
                var cms = new CountMinSketch(Width, Depth, 0x123456789ABCDEF0UL + (ulong)s * 0x9E3779B97F4A7C15UL);
                foreach (int key in stream) cms.Add(key);
                long maxError = 0;
                for (int key = 0; key < Domain; key++)
                {
                    long error = cms.Estimate(key) - exact[key];
                    if (error > maxError) maxError = error;
                }
                if (maxError > 0) seedsWithError++;
                if (maxError > eps * Length) violatingSeeds++;
            }
            double failureRate = (double)violatingSeeds / Seeds;
            Check(seedsWithError > (Seeds * 9) / 10, "nearly every seed produces a non-exact sketch (" + seedsWithError + "/" + Seeds + ")");
            Check(failureRate <= 0.05, "seeds exposing an error above eps*n are rare: " + violatingSeeds + "/" + Seeds
                + " (union bound " + unionBound.ToString("F4") + ", eps*n = " + (eps * Length).ToString("F1") + ")");
            Console.WriteLine("CMS w=" + Width + " d=" + Depth + " eps=" + eps.ToString("F4") + " delta=" + delta.ToString("F4")
                + " seedsAboveBound=" + violatingSeeds + "/" + Seeds + " unionBound=" + unionBound.ToString("F4"));

            // width matters: a much narrower sketch must be visibly worse
            double narrowError = 0, wideError = 0;
            for (int s = 0; s < 50; s++)
            {
                ulong seed = 0xFEEDFACECAFEBEEFUL + (ulong)s * 0x9E3779B97F4A7C15UL;
                var narrow = new CountMinSketch(8, 3, seed);
                var wide = new CountMinSketch(128, 3, seed);
                foreach (int key in stream) { narrow.Add(key); wide.Add(key); }
                for (int key = 0; key < Domain; key++)
                {
                    narrowError += narrow.Estimate(key) - exact[key];
                    wideError += wide.Estimate(key) - exact[key];
                }
            }
            Check(narrowError > 2 * wideError, "a 16x narrower sketch carries far more collision error ("
                + narrowError.ToString("F0") + " vs " + wideError.ToString("F0") + ")");
        }

        // ---- degenerate streams -------------------------------------------------------------------
        {
            var empty = new MisraGries(4);
            var emptySs = new SpaceSaving(4);
            var emptyCms = new CountMinSketch(16, 3, 7UL);
            Check(empty.Monitored == 0 && emptySs.Tracked == 0, "an empty stream leaves every counter empty");
            Check(empty.Estimate(3) == 0 && emptySs.Estimate(3) == 0 && emptyCms.Estimate(3) == 0,
                "an empty stream reports zero for every query");
            Check(emptySs.MinCount() == 0, "an empty Space-Saving table has no minimum to report");

            var repeated = new MisraGries(1);
            var repeatedSs = new SpaceSaving(1);
            for (int i = 0; i < 500; i++) { repeated.Add(9); repeatedSs.Add(9); }
            Check(repeated.Estimate(9) == 500, "a single counter tracks a single repeated key exactly");
            Check(repeated.Monitored == 1 && repeatedSs.Tracked == 1, "one counter cannot track two keys");
            Check(repeatedSs.Estimate(9) == 500, "Space-Saving with one counter is exact on a single key");

            var distinct = new int[Domain];
            for (int i = 0; i < Domain; i++) distinct[i] = i;
            var distinctMg = new MisraGries(Counters);
            var distinctSs = new SpaceSaving(Counters);
            foreach (int key in distinct) { distinctMg.Add(key); distinctSs.Add(key); }
            bool estimatesWithinOne = true, neverAboveExact = true, neverBelowExact = true;
            for (int key = 0; key < Domain; key++)
            {
                long mgEstimate = distinctMg.Estimate(key);
                long ssEstimate = distinctSs.Estimate(key);
                if (mgEstimate > 1) estimatesWithinOne = false;       // each key occurs once
                if (mgEstimate > 1) neverAboveExact = false;
                if (ssEstimate < 1 && Contains(distinctSs.TrackedKeys(), key)) neverBelowExact = false;
            }
            Check(distinctMg.Monitored <= Counters && distinctSs.Tracked <= Counters, "the memory bound holds on the all-distinct stream");
            Check(estimatesWithinOne, "with every frequency equal to one Misra-Gries can only report 0 or 1");
            Check(neverAboveExact, "Misra-Gries is exact-capped on the all-distinct stream");
            Check(neverBelowExact, "Space-Saving never returns 0 for a key it still tracks");
            Check(distinctSs.CounterSum() == Domain, "Space-Saving accounts for all " + Domain + " distinct items");
            Check(distinctSs.MinCount() <= Domain / Counters + 1, "the minimum counter stays near n/c on the all-distinct stream");

            bool threw = false;
            try { new MisraGries(0); } catch (ArgumentException) { threw = true; }
            Check(threw, "a zero-counter Misra-Gries is rejected");
            bool threwCms = false;
            try { new CountMinSketch(0, 1, 1UL); } catch (ArgumentException) { threwCms = true; }
            Check(threwCms, "a zero-width sketch is rejected");
        }

        // ---- determinism --------------------------------------------------------------------------
        {
            var first = new MisraGries(Counters);
            foreach (int key in stream) first.Add(key);
            var second = new MisraGries(Counters);
            foreach (int key in stream) second.Add(key);
            var a = first.MonitoredKeys();
            var b = second.MonitoredKeys();
            bool same = a.Count == b.Count;
            for (int i = 0; same && i < a.Count; i++) if (a[i] != b[i]) same = false;
            Check(same, "two identical runs of the same stream monitor the same keys");

            var cmsA = new CountMinSketch(32, 4, 99UL);
            var cmsB = new CountMinSketch(32, 4, 99UL);
            foreach (int key in stream) { cmsA.Add(key); cmsB.Add(key); }
            bool estimatesMatch = true;
            for (int key = 0; key < Domain; key++) if (cmsA.Estimate(key) != cmsB.Estimate(key)) estimatesMatch = false;
            Check(estimatesMatch, "the sketch estimates are reproducible for a fixed seed");
            Console.WriteLine("stream length=" + Length + " domain=" + Domain + " counters=" + Counters);
        }

        Console.WriteLine("PASS");
    }
}
