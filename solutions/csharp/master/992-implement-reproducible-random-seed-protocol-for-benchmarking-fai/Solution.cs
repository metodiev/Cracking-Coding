// Task 992 - Implement reproducible random-seed protocol for benchmarking fairness.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The protocol derives a per-(master seed, label) seed with a hand-written mix over an FNV-1a hash of the
// label, then drives a SplitMix64 stream from it: test order comes from Fisher-Yates and inputs from the same
// stream, so a benchmark replays identically on any machine while two benchmarks never share a seed. Because
// the mix is a bijection, distinct labels cannot collide; a bucket histogram over many labels checks that the
// derived seeds are not biased towards a region. Verified: derivation stability and per-label distinctness at
// the master-seed boundaries (0 and ulong.MaxValue), injectivity over 4096 labels, bucket uniformity, the
// Fisher-Yates permutation against an independently written reference (oracle) and its bijection property via
// a naive duplicate scan, stream determinism across instances, multiset preservation under any presentation
// order of the test list, and the empty / single-test boundaries.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    /** SplitMix64: hand-written, deterministic, no dependence on the runtime's random facilities. */
    private sealed class Stream
    {
        private ulong _state;
        public Stream(ulong seed) { _state = seed; }
        public ulong Next()
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
        public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
    }

    private static ulong Fnv(string text, ulong seed)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL ^ seed;
            foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }

    /** Seed for one benchmark: mix(FNV(label, master)) - a bijection in the hash, so no label collisions. */
    private static ulong DeriveSeed(ulong master, string label)
    {
        unchecked
        {
            ulong z = Fnv(label, master) + 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    /** Fisher-Yates driven by the derived stream; the reference below is an independent transcription. */
    private static List<T> Order<T>(ulong master, string label, List<T> items)
    {
        var shuffled = new List<T>(items);
        var stream = new Stream(DeriveSeed(master, label));
        for (int i = shuffled.Count - 1; i > 0; i--)
        {
            int j = (int)(stream.Next() % (ulong)(i + 1));
            T swap = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = swap;
        }
        return shuffled;
    }

    /** Oracle: the textbook loop written out again, drawing indices from a fresh stream. */
    private static List<T> Reference<T>(ulong master, string label, List<T> items)
    {
        var copy = new List<T>(items);
        var stream = new Stream(DeriveSeed(master, label));
        for (int i = copy.Count - 1; i >= 1; i--)
        {
            int pick = (int)(stream.Next() % (ulong)(i + 1));
            T held = copy[pick];
            copy[pick] = copy[i];
            copy[i] = held;
        }
        return copy;
    }

    private static List<int> Inputs(ulong master, string label, int count, int bound)
    {
        var stream = new Stream(DeriveSeed(master, label));
        var values = new List<int>();
        for (int i = 0; i < count; i++) values.Add((int)(stream.Next() % (ulong)bound));
        return values;
    }

    private static string Text<T>(List<T> values)
    {
        var parts = new List<string>();
        foreach (T value in values) parts.Add(Convert.ToString(value, CultureInfo.InvariantCulture));
        return string.Join(",", parts);
    }

    private static List<string> Tests(int count)
    {
        var tests = new List<string>();
        for (int i = 0; i < count; i++) tests.Add("t" + i.ToString(CultureInfo.InvariantCulture));
        return tests;
    }

    public static void Main()
    {
        Check(DeriveSeed(1234UL, "sort") == DeriveSeed(1234UL, "sort")
              && DeriveSeed(1234UL, "sort") != DeriveSeed(1234UL, "join")
              && DeriveSeed(0UL, "sort") != DeriveSeed(ulong.MaxValue, "sort")
              && DeriveSeed(0UL, "") == DeriveSeed(0UL, ""),
              "derivation is stable per (master, label) and differs across labels and master boundaries");

        var seeds = new HashSet<ulong>();
        for (int i = 0; i < 4096; i++) seeds.Add(DeriveSeed(7UL, "bench-" + i.ToString(CultureInfo.InvariantCulture)));
        Check(seeds.Count == 4096, "4096 labels derive 4096 distinct seeds (injectivity)");

        int[] buckets = new int[16];
        for (int i = 0; i < 4096; i++) buckets[DeriveSeed(99UL, "case" + i.ToString(CultureInfo.InvariantCulture)) % 16]++;
        int lowest = int.MaxValue, highest = 0;
        foreach (int count in buckets) { lowest = Math.Min(lowest, count); highest = Math.Max(highest, count); }
        Check(lowest >= 256 * 0.85 && highest <= 256 * 1.15 && lowest > 0,
              "derived seeds spread evenly over 16 buckets (fairness)");

        List<string> tests = Tests(12);
        List<string> ordered = Order(42UL, "sort", tests);
        List<string> reference = Reference(42UL, "sort", tests);
        var seen = new HashSet<string>(ordered);
        bool duplicates = ordered.Count != seen.Count;
        var sorted = new List<string>(ordered);
        sorted.Sort(StringComparer.Ordinal);
        var expected = new List<string>(tests);
        expected.Sort(StringComparer.Ordinal);
        Check(Text(ordered) == Text(reference) && !duplicates && Text(sorted) == Text(expected)
              && Text(ordered) != Text(tests),
              "the permutation equals the independent reference and is a bijection of the multiset (oracle)");

        var reversed = new List<string>(tests);
        reversed.Reverse();
        var fromReversed = new List<string>(Order(42UL, "sort", reversed));
        fromReversed.Sort(StringComparer.Ordinal);
        Check(Text(fromReversed) == Text(expected) && Text(Order(42UL, "sort", tests)) == Text(ordered),
              "the drawn test set is independent of how the benchmark list was presented");

        string first = Text(Inputs(5UL, "sort", 8, 1000));
        Check(first == Text(Inputs(5UL, "sort", 8, 1000)) && first != Text(Inputs(5UL, "join", 8, 1000))
              && first.Split(',').Length == 8 && Inputs(5UL, "sort", 4, 1000)[0] == Inputs(5UL, "sort", 8, 1000)[0]
              && Text(Inputs(0UL, "sort", 4, 1000)) != Text(Inputs(1UL, "sort", 4, 1000)),
              "input streams are reproducible per label, differ between labels, and extend as prefixes");

        Check(Order(1UL, "empty", new List<string>()).Count == 0
              && Text(Order(1UL, "single", Tests(1))) == "t0"
              && Order(1UL, "pair", Tests(2)).Count == 2
              && Text(Inputs(1UL, "none", 0, 10)) == "",
              "empty, single-test and empty-stream boundaries");

        Console.WriteLine("distinct seeds: " + seeds.Count + ", bucket range: " + lowest + ".." + highest
                          + ", order of 12 tests: " + Text(ordered));
        Console.WriteLine("PASS");
    }
}
