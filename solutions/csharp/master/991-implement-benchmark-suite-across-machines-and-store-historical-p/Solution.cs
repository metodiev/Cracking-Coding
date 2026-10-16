// Task 991 - Implement benchmark suite across machines and store historical performance.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The suite runs every benchmark on every machine at a fixed size, takes a handful of samples and appends a
// history entry per (machine, benchmark) pair carrying median, median absolute deviation and p95. Machines
// differ by a speed factor and each sample gets a deterministic +-4% jitter from a hand-written splitmix64
// stream, so the history is reproducible. A calibration benchmark yields a per-machine speed index; dividing
// measured medians by it makes machines comparable. A verdict compares the latest run against the pair's
// baseline: it needs at least three runs of at least three samples, and reports a regression only when the
// ratio exceeds 20% *and* the movement clears three times the dispersion, which keeps noise-only reruns quiet.
// Verified: median/MAD against a naive insertion-sort oracle at the even-count boundary, byte-identical
// history for repeated runs, the 1.8x machine ratio, cross-machine comparability after normalisation, a
// detected 40% slowdown and a 2x speedup, zero verdicts over twelve noise-only reruns, the insufficient-data
// boundary, the append-only latest-entry invariant and an empty history.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public static class Solution
{
    private enum Shape { Linear, Quadratic, NLogN }

    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }
    private static ulong Mix(ulong z)
    {
        unchecked
        {
            z += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            return (z ^ (z >> 27)) * 0x94D049BB133111EBUL ^ (z >> 31);
        }
    }
    private static ulong Fnv(string text)
    {
        unchecked
        {
            ulong h = 14695981039346656037UL;
            foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h *= 1099511628211UL; }
            return h;
        }
    }
    private static double Cost(Shape shape, int n) =>
        shape == Shape.Linear ? n : shape == Shape.Quadratic ? (double)n * n : n * Math.Log2(n);

    private static long Median(long[] values)
    {
        var copy = new List<long>(values);
        copy.Sort();
        int mid = copy.Count / 2;
        return copy.Count % 2 == 1 ? copy[mid] : (copy[mid - 1] + copy[mid]) / 2;
    }
    private static long Mad(long[] values)
    {
        long median = Median(values);
        var deviations = new List<long>();
        foreach (long value in values) deviations.Add(Math.Abs(value - median));
        deviations.Sort();
        int mid = deviations.Count / 2;
        return deviations.Count % 2 == 1 ? deviations[mid] : (deviations[mid - 1] + deviations[mid]) / 2;
    }
    private static long Percentile(long[] values, double quantile)
    {
        var copy = new List<long>(values);
        copy.Sort();
        int index = (int)(quantile * (copy.Count - 1) + 0.5);
        return copy[Math.Max(0, Math.Min(copy.Count - 1, index))];
    }

    private sealed class Entry
    {
        public readonly int RunIndex, Samples;
        public readonly string Machine, Benchmark;
        public readonly long Median, Mad, P95;
        public Entry(int runIndex, string machine, string benchmark, int samples, long median, long mad, long p95)
        { RunIndex = runIndex; Machine = machine; Benchmark = benchmark; Samples = samples; Median = median; Mad = mad; P95 = p95; }
        public string Line() => RunIndex + "|" + Machine + "|" + Benchmark + "|" + Samples + "|" + Median + "|" + Mad + "|" + P95;
    }

    private sealed class History
    {
        private readonly List<Entry> _entries = new List<Entry>();
        public IReadOnlyList<Entry> Entries => _entries;
        public void Append(Entry entry) { _entries.Add(entry); }
        public List<Entry> Series(string machine, string benchmark)
        {
            var series = new List<Entry>();
            foreach (Entry entry in _entries)
                if (entry.Machine == machine && entry.Benchmark == benchmark) series.Add(entry);
            return series;
        }
        public Entry Latest(string machine, string benchmark)
        {
            List<Entry> series = Series(machine, benchmark);
            return series.Count == 0 ? null : series[series.Count - 1];
        }
        public string Digest()
        {
            var text = new StringBuilder();
            foreach (Entry entry in _entries) text.Append(entry.Line()).Append('\n');
            return Fnv(text.ToString()).ToString("x16", CultureInfo.InvariantCulture);
        }
    }

    private static long[] Sample(double cost, double speed, int samples, ulong seed)
    {
        var values = new long[samples];
        ulong state = seed;
        for (int i = 0; i < samples; i++)
        {
            state = Mix(state);
            double jitter = 1.0 + ((state >> 11) * (1.0 / 9007199254740992.0) - 0.5) * 0.08;
            values[i] = (long)(cost / speed * jitter);
        }
        return values;
    }

    private static void RunSuite(History history, List<(string Name, double Speed)> machines, List<(string Name, Shape Shape)> benchmarks,
                                 int n, int samples, ulong seed, int runIndex)
    {
        ulong state = seed;
        foreach ((string machine, double speed) in machines)
            foreach ((string benchmark, Shape shape) in benchmarks)
            {
                long[] values = Sample(Cost(shape, n), speed, samples, state);
                state = Mix(state + (ulong)values[0]);
                history.Append(new Entry(runIndex, machine, benchmark, samples, Median(values), Mad(values), Percentile(values, 0.95)));
            }
    }

    private static string Verdict(History history, string machine, string benchmark)
    {
        List<Entry> series = history.Series(machine, benchmark);
        if (series.Count < 3) return "insufficient-data";
        Entry baseline = series[0], latest = series[series.Count - 1];
        if (baseline.Samples < 3 || latest.Samples < 3) return "insufficient-data";
        double ratio = (double)latest.Median / baseline.Median;
        long dispersion = 3 * Math.Max(baseline.Mad, latest.Mad);
        if (ratio > 1.20 && latest.Median - baseline.Median > dispersion) return "regression";
        if (ratio < 0.85 && baseline.Median - latest.Median > dispersion) return "improvement";
        return "stable";
    }

    /** Speed index of a machine relative to the reference, measured on the calibration benchmark. */
    private static double SpeedIndex(History history, string reference, string machine) =>
        (double)history.Latest(machine, "calib").Median / history.Latest(reference, "calib").Median;

    private static List<(string, double)> Machines(double factor) => new List<(string, double)>
    {
        ("runner-a", 1.0), ("runner-b", 0.6), ("runner-c", 1.8 / factor),
    };
    private static List<(string, Shape)> Benchmarks() => new List<(string, Shape)>
    {
        ("calib", Shape.Linear), ("sort", Shape.NLogN), ("join", Shape.Quadratic),
    };
    private static int Pairs(bool stableOnly, History history)

    {
        int count = 0;
        foreach ((string machine, double _) in Machines(1.0))
            foreach ((string benchmark, Shape _) in Benchmarks())
                if (!stableOnly || Verdict(history, machine, benchmark) == "stable") count++;
        return count;
    }

    public static void Main()
    {
        long[] odd = { 30, 10, 40, 20, 50 }, even = { 40, 10, 30, 20 };
        var oracle = new List<long>(even);
        for (int i = 1; i < oracle.Count; i++)
        {
            long key = oracle[i];
            int j = i - 1;
            while (j >= 0 && oracle[j] > key) { oracle[j + 1] = oracle[j]; j--; }
            oracle[j + 1] = key;
        }
        long[] single = { 4200 };
        Check(Median(odd) == 30 && Median(even) == 25 && (oracle[1] + oracle[2]) / 2 == 25 && Mad(even) == 10 && Mad(odd) == 10
              && Percentile(single, 0.95) == 4200 && Percentile(odd, 0.95) == 50,
              "median and MAD match a naive insertion-sort oracle at the even-count boundary");

        var history = new History();
        var repeat = new History();
        RunSuite(history, Machines(1.0), Benchmarks(), 2000, 5, 1234UL, 0);
        RunSuite(repeat, Machines(1.0), Benchmarks(), 2000, 5, 1234UL, 0);
        Check(history.Digest() == repeat.Digest() && history.Entries.Count == 9
              && history.Latest("runner-a", "sort").P95 >= history.Latest("runner-a", "sort").Median,
              "identical runs produce an identical history");

        double ratio = (double)history.Latest("runner-a", "calib").Median / history.Latest("runner-c", "calib").Median;
        double lowest = double.MaxValue, highest = 0.0;
        foreach ((string machine, double _) in Machines(1.0))
        {
            double normalised = history.Latest(machine, "sort").Median / SpeedIndex(history, "runner-a", machine);
            lowest = Math.Min(lowest, normalised);
            highest = Math.Max(highest, normalised);
        }
        Check(ratio > 1.6 && ratio < 2.0 && highest / lowest < 1.15,
              "a 1.8x faster machine measures close to 1.8x faster and normalised medians agree across machines");

        var verdicts = new History();
        RunSuite(verdicts, Machines(1.0), Benchmarks(), 2000, 5, 77UL, 0);
        RunSuite(verdicts, Machines(1.0), Benchmarks(), 2000, 5, 78UL, 1);
        RunSuite(verdicts, Machines(1.4), Benchmarks(), 2000, 5, 79UL, 2);
        Check(Verdict(verdicts, "runner-c", "sort") == "regression" && Verdict(verdicts, "runner-a", "sort") == "stable"
              && Verdict(verdicts, "runner-b", "join") == "stable",
              "an injected 40% slowdown is reported as a regression while untouched pairs stay stable");
        var speedups = new History();
        for (int run = 0; run < 3; run++) RunSuite(speedups, Machines(1.0), Benchmarks(), 2000, 5, 100UL + (ulong)run, run);
        RunSuite(speedups, Machines(0.5), Benchmarks(), 2000, 5, 200UL, 3);
        Check(Verdict(speedups, "runner-c", "sort") == "improvement" && Verdict(speedups, "runner-a", "sort") == "stable",
              "a 2x speedup is reported as an improvement");

        var noisy = new History();
        for (int run = 0; run < 12; run++) RunSuite(noisy, Machines(1.0), Benchmarks(), 2000, 5, 5000UL + (ulong)run * 7919UL, run);
        Check(Pairs(true, noisy) == 9, "twelve noise-only reruns produce no verdict at all");

        var thin = new History();
        var solo = new List<(string, double)> { ("solo", 1.0) };
        var calib = new List<(string, Shape)> { ("calib", Shape.Linear) };
        RunSuite(thin, solo, calib, 1000, 2, 3UL, 0);
        RunSuite(thin, solo, calib, 1000, 2, 4UL, 1);
        bool fewRuns = Verdict(thin, "solo", "calib") == "insufficient-data";
        RunSuite(thin, solo, calib, 1000, 2, 5UL, 2);
        var fresh = new History();
        bool appendOnly = true;
        foreach ((string machine, double _) in Machines(1.0))
            foreach ((string benchmark, Shape _) in Benchmarks())
            {
                List<Entry> series = verdicts.Series(machine, benchmark);
                if (series.Count != 3 || verdicts.Latest(machine, benchmark) != series[series.Count - 1]) appendOnly = false;
            }
        Check(fewRuns && Verdict(thin, "solo", "calib") == "insufficient-data" && appendOnly
              && fresh.Entries.Count == 0 && fresh.Latest("runner-a", "sort") == null
              && Verdict(fresh, "runner-a", "sort") == "insufficient-data",
              "insufficient data is refused, history is append-only, and an empty history has no verdict");

        Console.WriteLine("history entries: " + verdicts.Entries.Count + ", digest: " + verdicts.Digest()
                          + ", calib median: " + history.Latest("runner-a", "calib").Median + " ns");
        Console.WriteLine("PASS");
    }
}
