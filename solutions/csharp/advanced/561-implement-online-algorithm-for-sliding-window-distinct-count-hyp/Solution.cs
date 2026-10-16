// Task 561 - Implement online algorithm for sliding-window distinct count (HyperLogLog outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Distinct counting in a sliding window, implemented rather than sketched. Two structures are provided.
// (1) An exact online window: events arrive in non-decreasing time, and a head-indexed queue plus a
// per-item multiplicity map retires everything at or before (now - window), so the count is the number
// of live keys in O(1) amortised per event; a live-set recomputation and a from-scratch window scan are
// kept as references. (2) HyperLogLog with p index bits (m = 2^p registers): the 64-bit hash of
// splitmix64 supplies the bucket index from its low p bits and the rank as the position of the leading
// one within the remaining 64-p bits, each register keeps the maximum rank seen, merging two sketches is
// a register-wise maximum (so merge is exact, not approximate), and the estimate is the standard
// alpha_m * m^2 / sum 2^{-M_j} with the linear-counting correction m ln(m/zeros) for small ranges.
// For the sliding window the sketch is kept per time bucket of size window/8 and the query merges the
// buckets that overlap the window; that over-counts, because a bucket can stick out of the window, and
// the file measures the resulting error instead of pretending it is unbiased. The documented standard
// error of HyperLogLog is 1.04/sqrt(m), which the accuracy checks are calibrated against with a factor
// of three. Random streams use splitmix64 (documented 64-bit generator) - the repo's 16-bit LCG would
// repeat far too soon for 20 000 distinct items.
//
// Documented checks: over a 2000-event stream (universe 40, window 200) the exact online count matches a
// live-set recomputation at every arrival instant (every 100th also checked against a from-scratch
// window scan); inserting a new item raises the count by exactly one and re-inserting a live item leaves
// it unchanged; retirement brings it back
// down; an empty window and a single live item behave as expected; HyperLogLog estimates 20 000 distinct
// items to within three standard errors for m = 256, 1024 and 4096 and is monotone as items are added;
// it returns exactly 0 for an empty sketch and about 1 for a one-item sketch; the register-wise merge of
// two sketches is bit-identical to the sketch of the union, so the merged estimate matches the direct one;
// and the bucketed sliding-window estimate stays within 25 % of the exact count (it over-counts, never
// collapsing), with repeated runs identical.
using System;
using System.Collections.Generic;
using System.Numerics;

internal static class SplitMix64
{
    /** splitmix64 (Steele/Lea/Flood): 64-bit state advanced by a golden-ratio increment. */
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int Below(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    /** One-shot mixing of an item id into a 64-bit hash. */
    internal static ulong Mix(ulong x)
    {
        return Next(ref x);
    }
}

internal sealed class HyperLogLog
{
    internal readonly int Precision;
    internal readonly int Registers;
    internal readonly int[] Counters;

    internal HyperLogLog(int precision)
    {
        Precision = precision;
        Registers = 1 << precision;
        Counters = new int[Registers];
    }

    internal void Add(ulong hash)
    {
        int index = (int)(hash & (ulong)(Registers - 1));
        ulong rest = hash >> Precision;
        int rank = rest == 0 ? 64 - Precision + 1 : BitOperations.LeadingZeroCount(rest) - Precision + 1;
        if (rank > Counters[index]) Counters[index] = rank;
    }

    internal void Merge(HyperLogLog other)
    {
        if (other.Registers != Registers) throw new Exception("sketch sizes differ");
        for (int i = 0; i < Registers; i++)
        {
            if (other.Counters[i] > Counters[i]) Counters[i] = other.Counters[i];
        }
    }

    internal int ZeroCount()
    {
        int zeros = 0;
        foreach (int value in Counters)
        {
            if (value == 0) zeros++;
        }
        return zeros;
    }

    internal double Estimate()
    {
        double alpha = Registers == 16 ? 0.673 : Registers == 32 ? 0.697
                : Registers == 64 ? 0.709 : 0.7213 / (1.0 + 1.079 / Registers);
        double sum = 0.0;
        foreach (int value in Counters) sum += Math.Pow(2.0, -value);
        double raw = alpha * Registers * Registers / sum;
        int zeros = ZeroCount();
        if (raw <= 2.5 * Registers && zeros > 0)
        {
            return Registers * Math.Log((double)Registers / zeros);
        }
        return raw;
    }

    internal bool RegistersEqual(HyperLogLog other)
    {
        if (other.Registers != Registers) return false;
        for (int i = 0; i < Registers; i++)
        {
            if (other.Counters[i] != Counters[i]) return false;
        }
        return true;
    }

    internal double StandardError()
    {
        return 1.04 / Math.Sqrt(Registers);
    }
}

internal sealed class ExactWindowDistinct
{
    private readonly int window;
    private readonly List<int> times = new List<int>();
    private readonly List<int> items = new List<int>();
    private readonly Dictionary<int, int> multiplicities = new Dictionary<int, int>();
    private int head;

    internal ExactWindowDistinct(int window)
    {
        this.window = window;
    }

    internal int Count
    {
        get { return multiplicities.Count; }
    }

    internal void Add(int time, int item)
    {
        while (head < times.Count && times[head] <= time - window)
        {
            int old = items[head];
            int remaining = multiplicities[old] - 1;
            if (remaining == 0) multiplicities.Remove(old);
            else multiplicities[old] = remaining;
            head++;
        }
        times.Add(time);
        items.Add(item);
        multiplicities.TryGetValue(item, out int current);
        multiplicities[item] = current + 1;
    }
}

internal sealed class BucketedWindowHll
{
    private readonly int window;
    private readonly int bucketSize;
    private readonly int precision;
    private readonly List<int> bucketIds = new List<int>();
    private readonly List<HyperLogLog> sketches = new List<HyperLogLog>();
    private int currentBucket = int.MinValue;

    internal BucketedWindowHll(int window, int bucketSize, int precision)
    {
        this.window = window;
        this.bucketSize = bucketSize;
        this.precision = precision;
    }

    internal void Add(int time, int item)
    {
        int bucket = time / bucketSize;
        if (bucket != currentBucket)
        {
            bucketIds.Add(bucket);
            sketches.Add(new HyperLogLog(precision));
            currentBucket = bucket;
        }
        sketches[sketches.Count - 1].Add(SplitMix64.Mix((ulong)item * 0x9E3779B97F4A7C15UL + 1UL));
    }

    internal double Query(int time)
    {
        int lowest = (time - window) / bucketSize;
        int highest = time / bucketSize;
        var merged = new HyperLogLog(precision);
        for (int i = 0; i < bucketIds.Count; i++)
        {
            if (bucketIds[i] < lowest || bucketIds[i] > highest) continue;
            merged.Merge(sketches[i]);
        }
        return merged.Estimate();
    }
}

public static class Solution
{
    public static void Main()
    {
        const int Events = 2000;
        const int Universe = 40;
        const int Window = 200;
        ulong rng = 561001UL;
        var times = new int[Events];
        var items = new int[Events];
        var exact = new ExactWindowDistinct(Window);
        int newLive = 0;
        for (int i = 0; i < Events; i++)
        {
            times[i] = i * 7 + SplitMix64.Below(ref rng, 5);
            items[i] = SplitMix64.Below(ref rng, Universe);
            HashSet<int> liveBefore = LiveSet(times, items, i, times[i], Window);
            exact.Add(times[i], items[i]);
            int expected = liveBefore.Count + (liveBefore.Contains(items[i]) ? 0 : 1);
            Check(exact.Count == expected, "the online count is wrong at event " + i + ": got "
                    + exact.Count + ", expected " + expected);
            if (!liveBefore.Contains(items[i])) newLive++;
            if (i % 100 == 0)
            {
                Check(exact.Count == BruteForceWindow(times, items, i, Window),
                        "the online window count differs from the brute-force recomputation at event " + i);
            }
        }
        Check(newLive > 100, "many events introduced a new live item (" + newLive + ")");
        Check(exact.Count == BruteForceWindow(times, items, Events - 1, Window),
                "the final count matches the brute force");

        // Retirement: a single item that ages out.
        var single = new ExactWindowDistinct(10);
        single.Add(0, 5);
        Check(single.Count == 1, "one live item is counted");
        single.Add(5, 7);
        Check(single.Count == 2, "two live items are counted");
        single.Add(5, 7);
        Check(single.Count == 2, "a duplicate does not change the count");
        single.Add(20, 9);
        Check(single.Count == 1, "the older items retired and only the new one remains");
        var empty = new ExactWindowDistinct(3);
        Check(empty.Count == 0, "an empty window counts zero");
        empty.Add(0, 1);
        empty.Add(10, 2);
        Check(empty.Count == 1, "the window boundary retires the first item");

        // HyperLogLog accuracy on a 20000-item stream.
        foreach (int precision in new[] { 8, 10, 12 })
        {
            ulong stream = 561100UL + (ulong)precision;
            var sketch = new HyperLogLog(precision);
            double previous = sketch.Estimate();
            Check(sketch.Estimate() == 0.0, "an empty sketch estimates zero");
            for (int item = 1; item <= 20000; item++)
            {
                sketch.Add(SplitMix64.Mix((ulong)item));
                if (item % 2000 == 0)
                {
                    double now = sketch.Estimate();
                    Check(now >= previous - 1e-9, "the estimate is monotone as items are added");
                    previous = now;
                }
            }
            double estimate = sketch.Estimate();
            double relativeError = Math.Abs(estimate - 20000.0) / 20000.0;
            double bound = 3.0 * sketch.StandardError();
            Check(relativeError <= bound, "m = " + sketch.Registers + " estimate " + estimate
                    + " is off by " + relativeError + ", bound " + bound);
            Check(Math.Abs(previous - estimate) <= 1e-9, "the monotone chain ends at the final estimate");
        }

        // Merge is register-exact.
        var leftSketch = new HyperLogLog(10);
        var rightSketch = new HyperLogLog(10);
        var unionSketch = new HyperLogLog(10);
        for (int item = 1; item <= 500; item++) leftSketch.Add(SplitMix64.Mix((ulong)item));
        for (int item = 400; item <= 900; item++)
        {
            rightSketch.Add(SplitMix64.Mix((ulong)item));
            unionSketch.Add(SplitMix64.Mix((ulong)item));
        }
        for (int item = 1; item <= 500; item++) unionSketch.Add(SplitMix64.Mix((ulong)item));
        var mergedSketch = new HyperLogLog(10);
        mergedSketch.Merge(leftSketch);
        mergedSketch.Merge(rightSketch);
        Check(mergedSketch.RegistersEqual(unionSketch), "merging twice equals the sketch of the union");
        Check(Math.Abs(mergedSketch.Estimate() - unionSketch.Estimate()) <= 1e-12,
                "the merged estimate equals the direct estimate");
        Check(Math.Abs(mergedSketch.Estimate() - 900.0) / 900.0 <= 3.0 * mergedSketch.StandardError(),
                "the union of 900 distinct items is estimated within three standard errors, got "
                + mergedSketch.Estimate());

        var oneItem = new HyperLogLog(10);
        oneItem.Add(SplitMix64.Mix(42UL));
        Check(Math.Abs(oneItem.Estimate() - 1.0) <= 0.05, "one item estimates about one, got "
                + oneItem.Estimate());

        // Bucketed sliding window: measured error against the exact online structure.
        const int BucketWindow = 128;
        var exactWindow = new ExactWindowDistinct(BucketWindow);
        var bucketed = new BucketedWindowHll(BucketWindow, BucketWindow / 8, 10);
        double worstRelative = 0.0;
        int checks = 0;
        ulong streamRng = 561200UL;
        for (int i = 0; i < 3000; i++)
        {
            int time = i * 2 + SplitMix64.Below(ref streamRng, 3);
            int item = SplitMix64.Below(ref streamRng, 400);
            exactWindow.Add(time, item);
            bucketed.Add(time, item);
            if (i % 25 == 0 && exactWindow.Count > 0)
            {
                double estimate = bucketed.Query(time);
                double relative = Math.Abs(estimate - exactWindow.Count) / exactWindow.Count;
                if (relative > worstRelative) worstRelative = relative;
                Check(relative <= 0.25, "bucketed window estimate off by " + relative + " at i = " + i
                        + " (exact " + exactWindow.Count + ", estimate " + estimate + ")");
                checks++;
            }
        }
        Check(checks >= 100, "the window query was checked at least 100 times, got " + checks);
        Check(worstRelative <= 0.25, "the worst relative window error stays within 25 %, got "
                + worstRelative);

        // Determinism.
        var repeat = new HyperLogLog(10);
        for (int item = 1; item <= 20000; item++) repeat.Add(SplitMix64.Mix((ulong)item));
        var again = new HyperLogLog(10);
        for (int item = 1; item <= 20000; item++) again.Add(SplitMix64.Mix((ulong)item));
        Check(repeat.RegistersEqual(again), "repeated hashing gives identical sketches");
        var repeatExact = new ExactWindowDistinct(Window);
        for (int i = 0; i < Events; i++) repeatExact.Add(times[i], items[i]);
        Check(repeatExact.Count == exact.Count, "the online window structure is reproducible");

        Console.WriteLine("PASS");
    }

    private static HashSet<int> LiveSet(int[] times, int[] items, int upToIndex, int now, int window)
    {
        var seen = new HashSet<int>();
        for (int j = 0; j < upToIndex; j++)
        {
            if (times[j] > now - window) seen.Add(items[j]);
        }
        return seen;
    }

    private static int BruteForceWindow(int[] times, int[] items, int upTo, int window)
    {
        var seen = new HashSet<int>();
        for (int i = upTo; i >= 0; i--)
        {
            if (times[i] <= times[upTo] - window) break;
            seen.Add(items[i]);
        }
        return seen.Count;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
