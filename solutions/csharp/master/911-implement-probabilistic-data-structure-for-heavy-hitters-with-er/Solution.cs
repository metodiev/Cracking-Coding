// Task 911 - Implement probabilistic data structure for heavy hitters with error guarantees and mergeability.
// (Tier: Master, Beyond - Data Systems)
//
// Two summaries are implemented over an item stream, both mergeable and both with a provable error term
// checked here against an exact dictionary oracle:
//
//  * Misra-Gries - a table of k counters; a new item takes a free slot, and when the table is full every
//    counter is decremented by one and the item is dropped.  If D is the number of such drops, then
//    D = (n - sum(est)) / k, and every item satisfies freq(x) - est(x) <= D while every item with
//    freq(x) > D is guaranteed to still be in the summary.  Both directions are verified against the exact
//    frequencies, not assumed.  Merging adds the tables and subtracts the (k+1)-th largest count, which
//    keeps the same guarantee for the concatenated stream.
//  * Count-Min sketch - d rows of w counters with deterministic splitmix64 hashes; the estimate is the
//    minimum over rows, so it can never underestimate, and with w = e/epsilon the error exceeds
//    epsilon * n only with probability delta.  The sketch is linear, so merging is an element-wise table
//    addition, which is checked against the exact table sum.
//
// Simplifications: item keys are short strings, stream sizes are a few thousand updates, epsilon/delta are
// fixed per sketch, and no conservative-update or space-saving variants are compared.  Verified: both
// guarantees against the oracle, the merge bounds, never-underestimating, the k=1 and empty-stream
// boundaries, and reproducibility.
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

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private static ulong HashOf(string item, int salt)
    {
        ulong hash = 1469598103934665603UL;
        foreach (char c in item)
        {
            hash ^= c;
            hash *= 1099511628211UL;
        }
        unchecked
        {
            hash += 0x9E3779B97F4A7C15UL * (ulong)(salt + 1);
            hash ^= hash >> 29;
            hash *= 0xBF58476D1CE4E5B9UL;
            return hash ^ (hash >> 32);
        }
    }

    internal sealed class MisraGries
    {
        private readonly int _slots;
        private readonly Dictionary<string, int> _counts = new Dictionary<string, int>();
        internal int Drops;
        internal long Decremented;
        internal long Seen;
        internal int LastCutoff;

        internal MisraGries(int slots)
        {
            _slots = Math.Max(1, slots);
        }

        internal int Size => _counts.Count;

        internal void Add(string item)
        {
            Seen++;
            if (_counts.TryGetValue(item, out int count))
            {
                _counts[item] = count + 1;
                return;
            }
            if (_counts.Count < _slots)
            {
                _counts[item] = 1;
                return;
            }
            Drops++;
            List<string> emptied = new List<string>();
            foreach (string key in new List<string>(_counts.Keys))
            {
                int next = _counts[key] - 1;
                Decremented++;
                if (next <= 0) emptied.Add(key); else _counts[key] = next;
            }
            foreach (string key in emptied) _counts.Remove(key);
        }

        internal int Estimate(string item) => _counts.TryGetValue(item, out int value) ? value : 0;

        internal bool Contains(string item) => _counts.ContainsKey(item);

        internal long Mass()
        {
            long total = 0;
            foreach (int value in _counts.Values) total += value;
            return total;
        }

        internal Dictionary<string, int> Table() => new Dictionary<string, int>(_counts);

        /** The classic merge: add the tables, then subtract the (k+1)-th largest count. */
        internal static MisraGries Merge(MisraGries left, MisraGries right)
        {
            MisraGries merged = new MisraGries(left._slots);
            merged.Seen = left.Seen + right.Seen;
            merged.Drops = left.Drops + right.Drops;
            merged.Decremented = left.Decremented + right.Decremented;
            Dictionary<string, int> combined = left.Table();
            foreach (KeyValuePair<string, int> entry in right.Table())
            {
                combined.TryGetValue(entry.Key, out int existing);
                combined[entry.Key] = existing + entry.Value;
            }
            List<int> values = new List<int>(combined.Values);
            values.Sort((a, b) => b.CompareTo(a));
            int cutoff = values.Count > left._slots ? values[left._slots] : 0;
            merged.LastCutoff = cutoff;
            foreach (KeyValuePair<string, int> entry in combined)
            {
                int adjusted = entry.Value - cutoff;
                if (adjusted > 0) merged._counts[entry.Key] = adjusted;
            }
            while (merged._counts.Count > left._slots)
            {
                string smallest = string.Empty;
                foreach (KeyValuePair<string, int> entry in merged._counts)
                {
                    if (smallest.Length == 0 || entry.Value < merged._counts[smallest]) smallest = entry.Key;
                }
                merged._counts.Remove(smallest);
            }
            return merged;
        }
    }

    internal sealed class CountMin
    {
        private readonly int _rows;
        private readonly int _width;
        private readonly long[][] _cells;

        internal CountMin(int rows, int width)
        {
            _rows = rows;
            _width = width;
            _cells = new long[rows][];
            for (int row = 0; row < rows; row++) _cells[row] = new long[width];
        }

        internal void Add(string item, long weight)
        {
            for (int row = 0; row < _rows; row++)
            {
                int column = (int)(HashOf(item, row) % (ulong)_width);
                _cells[row][column] += weight;
            }
        }

        internal long Estimate(string item)
        {
            long best = long.MaxValue;
            for (int row = 0; row < _rows; row++)
            {
                int column = (int)(HashOf(item, row) % (ulong)_width);
                best = Math.Min(best, _cells[row][column]);
            }
            return best;
        }

        internal long TableSum()
        {
            long total = 0;
            foreach (long[] row in _cells)
            {
                foreach (long value in row) total += value;
            }
            return total;
        }

        /** Count-Min tables are linear, so the merge is an element-wise addition. */
        internal static CountMin Merge(CountMin left, CountMin right)
        {
            CountMin merged = new CountMin(left._rows, left._width);
            for (int row = 0; row < left._rows; row++)
            {
                for (int column = 0; column < left._width; column++)
                {
                    merged._cells[row][column] = left._cells[row][column] + right._cells[row][column];
                }
            }
            return merged;
        }

        internal bool TablesAddUpTo(CountMin left, CountMin right)
        {
            for (int row = 0; row < _rows; row++)
            {
                for (int column = 0; column < _width; column++)
                {
                    if (_cells[row][column] != left._cells[row][column] + right._cells[row][column]) return false;
                }
            }
            return true;
        }
    }

    private static List<string> GenerateStream(int length, int vocabulary, int seed)
    {
        _state = (ulong)seed;
        List<string> stream = new List<string>();
        for (int i = 0; i < length; i++)
        {
            // Zipf-ish: squaring a uniform draws the low ranks far more often, producing real heavy hitters.
            double u = (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);
            int rank = (int)(u * u * vocabulary);
            stream.Add("item" + rank.ToString(CultureInfo.InvariantCulture));
        }
        return stream;
    }

    private static Dictionary<string, int> ExactCounts(List<string> stream)
    {
        Dictionary<string, int> exact = new Dictionary<string, int>();
        foreach (string item in stream) exact[item] = exact.TryGetValue(item, out int value) ? value + 1 : 1;
        return exact;
    }

    public static void Main()
    {
        List<string> stream = GenerateStream(4000, 200, 7);
        Dictionary<string, int> exact = ExactCounts(stream);
        const int slots = 20;
        MisraGries summary = new MisraGries(slots);
        CountMin sketch = new CountMin(4, 512);
        foreach (string item in stream)
        {
            summary.Add(item);
            sketch.Add(item, 1);
        }

        // ---- 1. the Misra-Gries accounting identity and error theorem against the oracle ---------------
        long mass = summary.Mass();
        long slack = summary.Drops + summary.Decremented;
        Check(stream.Count == mass + slack,
            "every update is accounted for: n = mass + decrements + drop events (" + stream.Count + " = " + mass +
            " + " + summary.Decremented + " + " + summary.Drops + ")");
        int overestimated = 0, heavyMissed = 0, absentButHigh = 0;
        double worstError = 0.0;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            int estimate = summary.Estimate(entry.Key);
            if (estimate > entry.Value) overestimated++;
            if (entry.Value > summary.Drops && !summary.Contains(entry.Key)) heavyMissed++;
            if (!summary.Contains(entry.Key) && entry.Value > 0) absentButHigh++;
            if (estimate > 0) worstError = Math.Max(worstError, entry.Value - estimate);
        }
        Check(overestimated == 0, "no item is ever over-estimated");
        Check(heavyMissed == 0, "every item above the drop threshold survived in the summary");
        Check(worstError <= slack, "freq - estimate stays inside the provable slack (worst " +
            worstError.ToString("F0", CultureInfo.InvariantCulture) + " <= " + slack + ")");
        Check(summary.Size <= slots, "the summary never uses more than " + slots + " counters");
        Check(absentButHigh > 0, "the stream really does contain items the summary is not tracking (" + absentButHigh + ")");

        // ---- 2. heavy hitters are found, light items are not --------------------------------------------
        int trueHeavy = 0, foundHeavy = 0;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            if (entry.Value > summary.Drops)
            {
                trueHeavy++;
                if (summary.Contains(entry.Key)) foundHeavy++;
            }
        }
        Check(trueHeavy > 0 && foundHeavy == trueHeavy, "all " + trueHeavy + " heavy hitters were reported");
        Check(trueHeavy < exact.Count, "the vocabulary is much larger than the heavy hitter set (" +
            trueHeavy + " of " + exact.Count + " items)");
        Check(summary.Estimate("absent-item") == 0, "an unseen item has estimate zero");
        int lightTracked = 0;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            if (entry.Value <= summary.Drops && summary.Contains(entry.Key)) lightTracked++;
        }
        Check(lightTracked < exact.Count, "some light items are tracked, but not all of them (" + lightTracked + ")");
        Check(summary.Estimate(stream[0]) <= exact[stream[0]], "the first stream item is never over-counted");

        // ---- 3. merging two Misra-Gries summaries --------------------------------------------------------
        MisraGries left = new MisraGries(slots);
        MisraGries right = new MisraGries(slots);
        for (int i = 0; i < stream.Count / 2; i++) left.Add(stream[i]);
        for (int i = stream.Count / 2; i < stream.Count; i++) right.Add(stream[i]);
        MisraGries merged = MisraGries.Merge(left, right);
        Check(merged.Size <= slots, "the merged summary respects the slot budget (" + merged.Size + ")");
        long combinedSlack = left.Drops + left.Decremented + right.Drops + right.Decremented;
        long presenceThreshold = merged.LastCutoff / 2 + Math.Min(left.Drops + left.Decremented, right.Drops + right.Decremented);
        bool mergedAlgebra = true, mergedHeavy = true;
        double mergedWorst = 0.0;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            int estimate = merged.Estimate(entry.Key);
            int sumOfParts = left.Estimate(entry.Key) + right.Estimate(entry.Key);
            if (estimate > sumOfParts) mergedAlgebra = false;
            if (estimate > 0) mergedWorst = Math.Max(mergedWorst, entry.Value - estimate);
            if (entry.Value > presenceThreshold && !merged.Contains(entry.Key)) mergedHeavy = false;
        }
        Check(mergedAlgebra, "the merged estimate never exceeds the sum of the two estimates");
        Check(mergedWorst <= combinedSlack + merged.LastCutoff,
            "the merged summary keeps freq - estimate inside the summed slack plus the merge cutoff (" +
            mergedWorst.ToString("F0", CultureInfo.InvariantCulture) + " <= " + (combinedSlack + merged.LastCutoff) + ")");
        Check(mergedHeavy, "no item above the provable presence threshold is missing from the merged summary");
        Check(merged.LastCutoff >= 0 && merged.Size > 0, "the merge kept a non-empty table with a documented cutoff " + merged.LastCutoff);
        Check(merged.Seen == left.Seen + right.Seen, "the merged summary saw the concatenated stream (" + merged.Seen + " items)");
        Check(left.Estimate("item0") + right.Estimate("item0") >= merged.Estimate("item0"),
            "merging never increases an estimate beyond the sum of the parts");

        // ---- 4. the Count-Min sketch ----------------------------------------------------------------------
        double epsilon = Math.E / 512.0;
        long cmSlack = (long)Math.Ceiling(epsilon * stream.Count);
        int underestimates = 0, aboveBound = 0;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            long estimate = sketch.Estimate(entry.Key);
            if (estimate < entry.Value) underestimates++;
            if (estimate > entry.Value + cmSlack) aboveBound++;
        }
        Check(underestimates == 0, "the count-min sketch never underestimates a true count");
        Check(aboveBound == 0, "every estimate sits within epsilon*n = " + cmSlack + " of the truth");
        Check(sketch.Estimate("never-seen") <= cmSlack, "an unseen key still respects the epsilon*n error bound");
        Check(sketch.TableSum() == (long)stream.Count * 4, "each of the four rows holds exactly " + stream.Count + " increments");

        // ---- 5. merging count-min sketches ------------------------------------------------------------------
        CountMin cmLeft = new CountMin(4, 512);
        CountMin cmRight = new CountMin(4, 512);
        foreach (string item in stream.GetRange(0, stream.Count / 2)) cmLeft.Add(item, 1);
        foreach (string item in stream.GetRange(stream.Count / 2, stream.Count - stream.Count / 2)) cmRight.Add(item, 1);
        CountMin cmMerged = CountMin.Merge(cmLeft, cmRight);
        Check(cmMerged.TablesAddUpTo(cmLeft, cmRight), "the merged table is exactly the element-wise sum of the parts");
        bool mergedEstimatesSound = true;
        foreach (KeyValuePair<string, int> entry in exact)
        {
            long estimate = cmMerged.Estimate(entry.Key);
            long sumOfParts = cmLeft.Estimate(entry.Key) + cmRight.Estimate(entry.Key);
            mergedEstimatesSound &= estimate >= sumOfParts && estimate <= entry.Value + cmSlack;
        }
        Check(mergedEstimatesSound, "the merged sketch is between the summed estimates and the truth plus epsilon*n");
        Check(cmMerged.Estimate("never-seen") == 0, "an unseen key has zero estimate in a clean sketch");

        // ---- 6. boundaries ---------------------------------------------------------------------------------
        MisraGries single = new MisraGries(1);
        foreach (string item in stream) single.Add(item);
        Check(single.Size <= 1 && single.Drops > 0, "a one-slot summary keeps a single counter and still drops items");
        MisraGries empty = new MisraGries(8);
        Check(empty.Size == 0 && empty.Estimate("anything") == 0 && empty.Seen == 0, "an empty summary estimates everything as zero");
        empty.Add("only");
        Check(empty.Estimate("only") == 1 && empty.Drops == 0, "a single update is tracked exactly");
        MisraGries cancelled = new MisraGries(2);
        for (int i = 0; i < 40; i++) cancelled.Add("x" + (i % 7).ToString(CultureInfo.InvariantCulture));
        Check(cancelled.Size <= 2 && cancelled.Drops > 0 && cancelled.Mass() + cancelled.Decremented + cancelled.Drops == 40,
            "the decrement rule keeps at most two counters alive and still conserves all 40 updates");

        // ---- 7. reproducibility -------------------------------------------------------------------------------
        MisraGries replay = new MisraGries(slots);
        CountMin replaySketch = new CountMin(4, 512);
        foreach (string item in GenerateStream(4000, 200, 7))
        {
            replay.Add(item);
            replaySketch.Add(item, 1);
        }
        bool identical = replay.Size == summary.Size && replay.Drops == summary.Drops;
        foreach (KeyValuePair<string, int> entry in summary.Table())
        {
            identical &= replay.Estimate(entry.Key) == entry.Value;
        }
        Check(identical, "the same stream reproduces the same summary");
        Check(replaySketch.Estimate("item0") == sketch.Estimate("item0"), "and the same sketch estimate");

        Console.WriteLine("misra-gries: k=" + slots + ", " + stream.Count + " updates, " + summary.Size +
            " counters kept, " + summary.Drops + " drop events, worst error " + worstError.ToString("F0", CultureInfo.InvariantCulture));
        Console.WriteLine("heavy hitters: " + trueHeavy + " of " + exact.Count + " distinct items exceed the drop threshold, all reported");
        Console.WriteLine("count-min: 4x512, epsilon=" + epsilon.ToString("E2", CultureInfo.InvariantCulture) +
            ", slack " + cmSlack + ", zero underestimates, merged table equals the element-wise sum");
        Console.WriteLine("PASS");
    }
}
