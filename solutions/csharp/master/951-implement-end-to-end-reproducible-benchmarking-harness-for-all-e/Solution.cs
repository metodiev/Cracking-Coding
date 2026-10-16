// Task 951 - Implement end-to-end reproducible benchmarking harness for all exercises in repo.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The repository is modelled as a list of exercises, each pairing a deterministic operation counter
// with a per-operation cost in microseconds.  A run performs `warmup` discarded repetitions followed
// by `repetitions` measured ones; every sample mixes the analytic operation count with a jitter drawn
// from a hand-written splitmix64 stream, so timings are reproducible byte-for-byte for a fixed seed
// yet still noisy.  Reproducibility is pinned by an FNV-1a fingerprint over the rendered samples, and
// regressions are found by comparing a run against a stored baseline with a relative tolerance.
//
// Verified: the closed-form operation count of the sum exercise against an iterative oracle, the
// jitter envelope of every sample, fingerprint stability and seed sensitivity, warm-up exclusion,
// identical-run and 1.5x-slowdown comparisons, plus the empty and single-repetition boundaries.
// Simplification: the clock is analytic (no wall time, no subprocesses) so the harness itself stays
// deterministic and testable; the measured surface is the scheduling/reporting logic, not the OS.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

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

    private static double NextUnit() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static double Jitter() => 0.85 + 0.30 * NextUnit();

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Exercise
    {
        public readonly string Name;
        public readonly int N;
        public readonly double MicrosPerOp;
        public readonly Func<int, long> Ops;

        public Exercise(string name, int n, double microsPerOp, Func<int, long> ops)
        {
            Name = name;
            N = n;
            MicrosPerOp = microsPerOp;
            Ops = ops;
        }
    }

    private sealed class Row
    {
        public readonly string Name;
        public readonly long Ops;
        public readonly List<double> Samples = new List<double>();
        public double Median;
        public double Min;

        public Row(string name, long ops)
        {
            Name = name;
            Ops = ops;
        }
    }

    private sealed class Report
    {
        public readonly List<Row> Rows = new List<Row>();
    }

    private static double Percentile(List<double> sorted, double q)
    {
        if (sorted.Count == 0) return 0.0;
        int index = (int)(q * (sorted.Count - 1) + 0.5);
        return sorted[index];
    }

    private static Report RunSuite(ulong seed, List<Exercise> exercises, int warmup, int reps)
    {
        _state = seed;
        var report = new Report();
        foreach (Exercise ex in exercises)
        {
            long ops = ex.Ops(ex.N);
            var row = new Row(ex.Name, ops);
            for (int i = 0; i < warmup; i++) Jitter(); // discarded draws keep the stream position honest
            for (int i = 0; i < reps; i++) row.Samples.Add(ops * ex.MicrosPerOp * Jitter());
            var sorted = new List<double>(row.Samples);
            sorted.Sort();
            row.Median = Percentile(sorted, 0.5);
            row.Min = sorted.Count > 0 ? sorted[0] : 0.0;
            report.Rows.Add(row);
        }
        return report;
    }

    private static ulong Fingerprint(Report report)
    {
        var sb = new StringBuilder();
        foreach (Row row in report.Rows)
        {
            sb.Append(row.Name).Append(':').Append(row.Ops.ToString(CultureInfo.InvariantCulture)).Append('[');
            foreach (double sample in row.Samples) sb.Append(sample.ToString("F6", CultureInfo.InvariantCulture)).Append(',');
            sb.Append("]|");
        }
        unchecked
        {
            ulong hash = 14695981039346656037UL;
            foreach (char ch in sb.ToString())
            {
                hash ^= ch;
                hash *= 1099511628211UL;
            }
            return hash;
        }
    }

    /** Compares a candidate run against a stored baseline; returns the sorted list of findings. */
    private static List<string> Compare(Report baseline, Report candidate, double tolerance)
    {
        var issues = new List<string>();
        var byName = new Dictionary<string, Row>(StringComparer.Ordinal);
        foreach (Row row in baseline.Rows) byName[row.Name] = row;
        foreach (Row row in candidate.Rows)
        {
            if (!byName.TryGetValue(row.Name, out Row stored))
            {
                issues.Add("new exercise: " + row.Name);
                continue;
            }
            if (stored.Median > 0 && row.Median > stored.Median * (1.0 + tolerance))
                issues.Add("regression " + row.Name + " " + stored.Median.ToString("F3", CultureInfo.InvariantCulture) +
                    " -> " + row.Median.ToString("F3", CultureInfo.InvariantCulture) + " (+" +
                    ((row.Median / stored.Median - 1.0) * 100.0).ToString("F1", CultureInfo.InvariantCulture) + "%)");
        }
        foreach (Row row in baseline.Rows)
        {
            if (!candidate.Rows.Exists(r => r.Name == row.Name)) issues.Add("missing exercise: " + row.Name);
        }
        issues.Sort(StringComparer.Ordinal);
        return issues;
    }

    public static void Main()
    {
        Exercise sum = new Exercise("sum-1..n", 1000, 0.02, n => (long)n * (n + 1) / 2);

        // Independent oracle: the closed form the harness benchmarks against a hand-written loop.
        bool closedFormOk = true;
        for (int n = 1; n <= 64; n++)
        {
            long iterative = 0;
            for (int i = 1; i <= n; i++) iterative += i;
            closedFormOk &= sum.Ops(n) == iterative;
        }
        Check(closedFormOk, "the closed-form operation count matches the iterative oracle for n=1..64");

        var suite = new List<Exercise>
        {
            sum,
            new Exercise("sort-nlogn", 4096, 0.05, n => (long)(n * Math.Log2(n))),
            new Exercise("matmul", 64, 0.01, n => (long)n * n * n),
            new Exercise("binary-search", 1 << 20, 0.12, n => (long)Math.Log2(n) + 1),
        };

        Report first = RunSuite(20240901UL, suite, 3, 9);
        Report second = RunSuite(20240901UL, suite, 3, 9);
        Report otherSeed = RunSuite(424242UL, suite, 3, 9);

        Check(Fingerprint(first) == Fingerprint(second), "the same seed reproduces the same suite fingerprint");
        Check(Fingerprint(first) != Fingerprint(otherSeed), "a different seed yields a different fingerprint");

        bool modelOk = true;
        foreach (Row row in first.Rows)
        {
            Exercise ex = suite.Find(e => e.Name == row.Name)!;
            double lo = row.Ops * ex.MicrosPerOp * 0.85 - 1e-9;
            double hi = row.Ops * ex.MicrosPerOp * 1.15 + 1e-9;
            foreach (double sample in row.Samples) modelOk &= sample >= lo && sample <= hi;
            modelOk &= row.Min <= row.Median + 1e-12;
        }
        Check(modelOk, "every sample stays inside the +-15% jitter envelope of ops x cost and min <= median");

        bool rawStable = first.Rows.Count == second.Rows.Count;
        for (int i = 0; rawStable && i < first.Rows.Count; i++)
        {
            Row a = first.Rows[i];
            Row b = second.Rows[i];
            rawStable &= a.Name == b.Name && a.Samples.Count == b.Samples.Count;
            for (int j = 0; rawStable && j < a.Samples.Count; j++) rawStable &= a.Samples[j] == b.Samples[j];
        }
        Check(rawStable, "the raw measured sample sequence is identical between two runs at the same seed");

        Report noWarmup = RunSuite(20240901UL, suite, 0, 9);
        Check(Fingerprint(noWarmup) != Fingerprint(first), "warm-up repetitions consume the jitter stream, so their exclusion shifts the samples");
        bool repsOk = noWarmup.Rows.Count == suite.Count;
        foreach (Row row in noWarmup.Rows) repsOk &= row.Samples.Count == 9;
        Check(repsOk, "warm-up repetitions are never recorded among the measured samples");

        Check(Compare(first, second, 0.05).Count == 0, "an identical re-run reports no regression against the baseline");

        var slower = new List<Exercise>();
        foreach (Exercise ex in suite) slower.Add(new Exercise(ex.Name, ex.N, ex.MicrosPerOp * 1.5, ex.Ops));
        List<string> regressions = Compare(first, RunSuite(20240901UL, slower, 3, 9), 0.05);
        Check(regressions.Count == suite.Count, "a 50% slowdown is reported for every exercise (" + regressions.Count + " findings)");
        Check(regressions[0].StartsWith("regression ", StringComparison.Ordinal), "regression findings are labelled and sorted");

        Report empty = RunSuite(7UL, new List<Exercise>(), 2, 4);
        Check(empty.Rows.Count == 0 && Fingerprint(empty) == 14695981039346656037UL,
            "the empty suite has no rows and the FNV offset-basis fingerprint");

        Report single = RunSuite(11UL, new List<Exercise> { new Exercise("one", 5, 1.0, n => n) }, 0, 1);
        Check(single.Rows.Count == 1 && single.Rows[0].Samples.Count == 1 &&
            Math.Abs(single.Rows[0].Median - single.Rows[0].Samples[0]) < 1e-12,
            "a single exercise with one repetition has median == the only sample");

        Console.WriteLine("harness: " + suite.Count + " exercises x 9 repetitions x 3 warm-up; fingerprint " +
            Fingerprint(first).ToString("x16", CultureInfo.InvariantCulture));
        Console.WriteLine("model: samples stay inside ops x cost x [0.85, 1.15]; empty-suite fingerprint is the FNV offset basis");
        Console.WriteLine("regression: " + regressions.Count + "/" + suite.Count + " exercises flagged at 1.5x cost with a 5% tolerance");
        Console.WriteLine("PASS");
    }
}
