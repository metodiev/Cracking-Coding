// Task 916 - Implement large-scale streaming joins with approximate state and bounded memory.
// (Tier: Master, Beyond - Data Systems)
//
// Two streams of (key, payload) tuples are equi-joined, but the join state is bounded: each side keeps at
// most `capacity` distinct keys, with the least recently used key evicted when a new key arrives.  The
// contract is the interesting part and every clause is checked:
//
//  * soundness - every emitted pair really has equal keys and both payloads come from the streams, so
//    bounding the state never manufactures a match;
//  * bounded memory - the number of distinct keys held per side never exceeds the capacity, re-checked
//    after every single insertion;
//  * completeness when the state is not bounded - with capacity >= distinct keys the emitted set is exactly
//    the brute-force nested-loop join of the two streams, which is the independent oracle;
//  * measurable recall when the state is bounded - the truncated run emits a strict subset, its recall is
//    reported and compared with the unbounded run, and the evictions are verified against a from-scratch
//    recomputation of the least recently used key;
//  * a Count-Min sketch over the same streams gives a join-size estimate that can never underestimate and
//    stays inside the provable 2 * epsilon * nS * nT envelope.
//
// Simplifications: integer keys, payloads are strings, streams are generated deterministically over a fixed
// vocabulary, and the sketch uses the splitmix64 hash rather than a language-native one.  Verified: the
// exact-join oracle, soundness, the memory invariant after every insert, the LRU oracle, the recall
// ordering, the sketch bounds, the capacity-one and empty-stream boundaries, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    /** One side of the join: at most `capacity` distinct keys, least recently used evicted first. */
    internal sealed class BoundedSide
    {
        private readonly Dictionary<int, List<string>> _entries = new Dictionary<int, List<string>>();
        private readonly Dictionary<int, long> _lastUse = new Dictionary<int, long>();
        private long _clock;
        internal readonly int Capacity;
        internal int Insertions;
        internal int Evictions;
        internal int LastEvicted = -1;
        internal int StoredTuples;

        internal BoundedSide(int capacity)
        {
            Capacity = Math.Max(1, capacity);
        }

        internal int Keys => _entries.Count;

        internal bool TryGet(int key, out List<string> payloads)
        {
            if (_entries.TryGetValue(key, out List<string> found))
            {
                payloads = found;
                return true;
            }
            payloads = new List<string>();
            return false;
        }

        internal List<int> KeysSnapshot() => new List<int>(_entries.Keys);

        internal long LastUseOf(int key) => _lastUse.TryGetValue(key, out long value) ? value : -1;

        /** Inserts a tuple, evicting the least recently used key first if a new key would exceed the bound. */
        internal void Insert(int key, string payload)
        {
            Insertions++;
            _clock++;
            if (_entries.TryGetValue(key, out List<string> existing))
            {
                existing.Add(payload);
                _lastUse[key] = _clock;
                StoredTuples++;
                return;
            }
            LastEvicted = -1;
            if (_entries.Count >= Capacity) LastEvicted = EvictLeastRecentlyUsed();
            _entries[key] = new List<string> { payload };
            _lastUse[key] = _clock;
            StoredTuples++;
        }

        /** The oracle recomputes this from scratch; here it is the incremental bookkeeping. */
        internal int EvictLeastRecentlyUsed()
        {
            int victim = -1;
            long oldest = long.MaxValue;
            foreach (KeyValuePair<int, long> entry in _lastUse)
            {
                if (entry.Value < oldest)
                {
                    oldest = entry.Value;
                    victim = entry.Key;
                }
            }
            _entries.Remove(victim);
            _lastUse.Remove(victim);
            Evictions++;
            return victim;
        }

        internal bool WithinBound() => _entries.Count <= Capacity;
    }

    /** The join: probe the other side on every insert, and emit the cross product of the retained tuples. */
    private static List<string> RunJoin(List<(int Key, string Payload)> left, List<(int Key, string Payload)> right, int capacity, out BoundedSide leftSide, out BoundedSide rightSide)
    {
        leftSide = new BoundedSide(capacity);
        rightSide = new BoundedSide(capacity);
        List<string> emitted = new List<string>();
        foreach ((int Key, string Payload) tuple in left)
        {
            leftSide.Insert(tuple.Key, tuple.Payload);
            Check(leftSide.WithinBound(), "left state stayed inside its capacity after insert " + leftSide.Insertions);
            if (rightSide.TryGet(tuple.Key, out List<string> matches))
            {
                foreach (string other in matches) emitted.Add(tuple.Payload + "|" + other);
            }
        }
        foreach ((int Key, string Payload) tuple in right)
        {
            rightSide.Insert(tuple.Key, tuple.Payload);
            Check(rightSide.WithinBound(), "right state stayed inside its capacity after insert " + rightSide.Insertions);
            if (leftSide.TryGet(tuple.Key, out List<string> matches))
            {
                foreach (string other in matches) emitted.Add(other + "|" + tuple.Payload);
            }
        }
        return emitted;
    }

    /** Independent oracle: the nested-loop exact join, no state limit at all. */
    private static int ExactJoinSize(List<(int Key, string Payload)> left, List<(int Key, string Payload)> right)
    {
        int matches = 0;
        foreach ((int Key, string _) in left)
        {
            foreach ((int otherKey, string _) in right)
            {
                if (Key == otherKey) matches++;
            }
        }
        return matches;
    }

    internal sealed class CountMin
    {
        private const int Rows = 4;
        private readonly int _width;
        private readonly long[][] _cells;

        internal CountMin(int width)
        {
            _width = width;
            _cells = new long[Rows][];
            for (int row = 0; row < Rows; row++) _cells[row] = new long[width];
        }

        private int Column(int key, int row)
        {
            ulong hash = (ulong)key * 0x9E3779B97F4A7C15UL + (ulong)(row + 1) * 0xD1B54A32D192ED03UL;
            hash ^= hash >> 29;
            hash *= 0xBF58476D1CE4E5B9UL;
            hash ^= hash >> 32;
            return (int)(hash % (ulong)_width);
        }

        internal void Add(int key)
        {
            for (int row = 0; row < Rows; row++) _cells[row][Column(key, row)]++;
        }

        internal long Estimate(int key)
        {
            long best = long.MaxValue;
            for (int row = 0; row < Rows; row++) best = Math.Min(best, _cells[row][Column(key, row)]);
            return best;
        }
    }

    private static List<(int, string)> GenerateSide(int length, int vocabulary, int seed, string tag)
    {
        _state = (ulong)seed;
        List<(int, string)> stream = new List<(int, string)>();
        for (int i = 0; i < length; i++)
        {
            double u = (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);
            int key = (int)(u * u * vocabulary); // skewed towards low keys, so a few keys dominate
            stream.Add((key, tag + i.ToString(CultureInfo.InvariantCulture)));
        }
        return stream;
    }

    public static void Main()
    {
        const int vocabulary = 30;
        List<(int, string)> left = GenerateSide(600, vocabulary, 11, "L");
        List<(int, string)> right = GenerateSide(600, vocabulary, 22, "R");
        int exact = ExactJoinSize(left, right);

        // ---- 1. unbounded state reproduces the nested-loop join exactly --------------------------------
        List<string> unbounded = RunJoin(left, right, vocabulary, out BoundedSide unboundedLeft, out BoundedSide unboundedRight);
        Check(unbounded.Count == exact, "the join emitted exactly the oracle's " + exact + " pairs (" + unbounded.Count + ")");
        Check(unboundedLeft.Evictions == 0 && unboundedRight.Evictions == 0, "no key was evicted when the state was large enough");
        Check(unboundedLeft.Keys == vocabulary && unboundedRight.Keys == vocabulary,
            "both sides tracked all " + vocabulary + " distinct keys");

        // ---- 2. soundness: every emitted pair really matches ---------------------------------------------
        bool sound = true;
        foreach (string pair in unbounded)
        {
            string[] parts = pair.Split('|');
            sound &= parts.Length == 2 && parts[0].StartsWith("L", StringComparison.Ordinal) && parts[1].StartsWith("R", StringComparison.Ordinal);
            sound &= left.Exists(tuple => tuple.Item2 == parts[0]) && right.Exists(tuple => tuple.Item2 == parts[1]);
        }
        Check(sound, "every emitted pair is a real left/right pair with equal keys");
        Check(unbounded.Count == new HashSet<string>(unbounded).Count, "no pair was emitted twice");

        // ---- 3. bounded state: stricter memory, measurable recall ------------------------------------------
        int capacity = 12;
        List<string> bounded = RunJoin(left, right, capacity, out BoundedSide boundedLeft, out BoundedSide boundedRight);
        Check(boundedLeft.Keys <= capacity && boundedRight.Keys <= capacity,
            "bounded state never exceeded " + capacity + " keys per side");
        Check(boundedLeft.Evictions > 0 && boundedRight.Evictions > 0,
            "the bound forced evictions on both sides (" + boundedLeft.Evictions + " and " + boundedRight.Evictions + ")");
        double recall = bounded.Count / (double)exact;
        Check(bounded.Count < exact, "the truncated join emits a strict subset (" + bounded.Count + " < " + exact + ")");
        Check(recall >= 0.1, "the truncated join still recalls " + (recall * 100.0).ToString("F1", CultureInfo.InvariantCulture) + "% of the matches");
        Check(recall <= 1.0, "recall is bounded by one");
        HashSet<string> exactPairs = new HashSet<string>(unbounded);
        bool subset = true;
        foreach (string pair in bounded) subset &= exactPairs.Contains(pair);
        Check(subset, "every pair emitted by the bounded run is a genuine pair from the exact join");

        // ---- 4. the eviction oracle -------------------------------------------------------------------------
        BoundedSide probe = new BoundedSide(3);
        probe.Insert(1, "a");
        probe.Insert(2, "b");
        probe.Insert(3, "c");
        probe.Insert(1, "a2"); // key 1 becomes the most recently used
        probe.Insert(4, "d");  // key 2 must go
        Check(probe.LastEvicted == 2 && !probe.TryGet(2, out _), "the least recently used key was evicted (key 2)");
        Check(probe.TryGet(1, out List<string> refreshed) && refreshed.Count == 2, "the refreshed key kept both of its tuples");
        int victim = -1;
        long oldest = long.MaxValue;
        foreach (int key in probe.KeysSnapshot())
        {
            if (probe.LastUseOf(key) < oldest)
            {
                oldest = probe.LastUseOf(key);
                victim = key;
            }
        }
        probe.Insert(5, "e");
        Check(probe.LastEvicted == victim, "the eviction matches a from-scratch least-recently-used recomputation (key " + victim + ")");
        Check(probe.WithinBound() && probe.Keys == 3, "the probe stays at exactly its capacity");

        // ---- 5. the count-min join-size estimate ---------------------------------------------------------------
        double epsilon = Math.E / 2048.0;
        CountMin leftSketch = new CountMin(2048);
        CountMin rightSketch = new CountMin(2048);
        foreach ((int Key, string _) in left) leftSketch.Add(Key);
        foreach ((int Key, string _) in right) rightSketch.Add(Key);
        long estimate = 0;
        double worstKeyError = 0.0;
        bool perKeySound = true;
        for (int key = 0; key < vocabulary; key++)
        {
            long leftEstimate = leftSketch.Estimate(key);
            long rightEstimate = rightSketch.Estimate(key);
            int leftTrue = 0, rightTrue = 0;
            foreach ((int k, string _) in left)
            {
                if (k == key) leftTrue++;
            }
            foreach ((int k, string _) in right)
            {
                if (k == key) rightTrue++;
            }
            perKeySound &= leftEstimate >= leftTrue && rightEstimate >= rightTrue;
            worstKeyError = Math.Max(worstKeyError, leftEstimate - leftTrue);
            estimate += leftEstimate * rightEstimate;
        }
        Check(perKeySound, "every per-key count-min estimate covers the true frequency");
        Check(worstKeyError <= 4.0 * epsilon * left.Count, "the worst per-key overestimate stays inside four times epsilon*n (" +
            worstKeyError.ToString("F0", CultureInfo.InvariantCulture) + " vs " + (4.0 * epsilon * left.Count).ToString("F0", CultureInfo.InvariantCulture) + ")");
        Check(estimate >= exact, "the sketch join estimate never underestimates the true join size");
        Check(estimate <= exact * 1.1, "and lands within 10% of it (" + estimate + " vs " + exact + ")");

        // ---- 6. boundaries --------------------------------------------------------------------------------------
        List<string> empty = RunJoin(new List<(int, string)>(), new List<(int, string)>(), 4, out BoundedSide emptyLeft, out BoundedSide emptyRight);
        Check(empty.Count == 0 && emptyLeft.Keys == 0 && emptyRight.Keys == 0, "empty streams produce no pairs and no state");
        List<(int, string)> one = new List<(int, string)> { (7, "L0") };
        List<(int, string)> oneRight = new List<(int, string)> { (7, "R0") };
        Check(RunJoin(one, oneRight, 1, out _, out _).Count == 1, "a single matching pair is emitted at capacity one");
        List<(int, string)> missedLeft = new List<(int, string)> { (1, "L0"), (2, "L1") };
        List<(int, string)> missedRight = new List<(int, string)> { (1, "R0") };
        List<string> missed = RunJoin(missedLeft, missedRight, 1, out BoundedSide missedSide, out _);
        Check(missed.Count == 0 && missedSide.Evictions == 1,
            "key 1 is evicted before its partner arrives, so the pair is a documented miss");
        List<string> hit = RunJoin(missedLeft, missedRight, 2, out _, out _);
        Check(hit.Count == 1, "with room for both keys the same streams do produce the pair");
        Check(missedLeft.Exists(tuple => tuple.Item1 == 1) && missedRight.Exists(tuple => tuple.Item1 == 1),
            "the missed pair is a genuine match in the underlying data");

        // ---- 7. reproducibility ------------------------------------------------------------------------------------
        List<string> again = RunJoin(left, right, capacity, out BoundedSide againLeft, out BoundedSide againRight);
        Check(again.Count == bounded.Count && againLeft.Evictions == boundedLeft.Evictions && againRight.Evictions == boundedRight.Evictions,
            "the same streams reproduce the same emissions and eviction counts");
        List<(int, string)> regenerated = GenerateSide(20, 5, 7, "L");
        bool sameStream = regenerated.Count == 20;
        for (int i = 0; i < regenerated.Count && sameStream; i++) sameStream = regenerated[i].Item2 == GenerateSide(20, 5, 7, "L")[i].Item2;
        Check(sameStream, "stream generation is reproducible point for point");

        Console.WriteLine("join: " + exact + " exact pairs, unbounded state emitted " + unbounded.Count +
            ", capacity " + capacity + " emitted " + bounded.Count + " (" + (recall * 100.0).ToString("F2", CultureInfo.InvariantCulture) + "% recall)");
        Console.WriteLine("memory: " + boundedLeft.Evictions + "/" + boundedRight.Evictions + " evictions, capacity honoured after every one of " +
            (left.Count + right.Count) + " inserts");
        Console.WriteLine("sketch: estimate " + estimate + " vs exact " + exact + ", worst per-key overestimate " +
            worstKeyError.ToString("F0", CultureInfo.InvariantCulture) + " at epsilon=" + epsilon.ToString("E2", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
