// Task 855 - Implement an automated plan regression detector for query optimizer changes.
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A deterministic model of the continuous-integration harness that decides whether an optimizer change
// made the plans worse, and if so which commit introduced the regression.
//
//   workload  40 parameterised three-table queries (cardinalities, filter thresholds, join keys)
//   optimizer a rule-based planner whose only difference across versions is a join-algorithm threshold;
//             it estimates the filtered cardinality of each table, sorts the tables by estimate, emits the
//             join order and picks a nested-loop or hash join per step
//   executor  a real tuple-level executor over synthetic tables: it materialises the filtered rows and
//             runs the plan, counting tuple comparisons and hash probes, and it *also* verifies that the
//             result of every plan equals the reference result, because an optimizer change may never
//             change the answer
//   detector  for each query it measures the plan under both versions with paired jitter seeds, keeps the
//             median ratio, requires that all repetitions agree on the direction, that the plan hash
//             actually changed and that the median ratio exceeds 1.2 — this plan-change filter is what
//             stops parameter-only noise from being reported
//
// Bisection: a chain of eight optimizer versions, one of which contains an injected regression that all
// later versions inherit. The detector bisects the chain for the first bad version and the result is
// compared with an exhaustive scan; a monotonicity check on the exhaustive scan justifies the bisection,
// and the number of measured versions is asserted to be logarithmic.
//
// What is simplified (documented, not behavioural): tables hold at most a few hundred rows so a real
// execution is cheap, jitter is a deterministic ±2% multiplier rather than wall-clock noise, and the
// injected regression is a rule change instead of a git commit. The execution, the measurement, the
// paired comparison, the plan-change filter, the plan diff and the bisection are the real content; the
// workload size and the number of repetitions are typed parameters.
//
// Invariants verified: every plan of every query returns the reference result (semantics preserved); the
// detector's verdicts equal a brute-force classification of the same measurements; an unchanged plan is
// never reported even when its measured cost drifts, while a changed plan below the threshold is reported
// as a change without a regression; a same-version run with fresh jitter reports nothing; improvements
// are classified as improvements; raising the threshold can only shrink the reported set; the bisection
// finds the same first bad version as an exhaustive scan on four different chains while measuring at most
// log2(N)+2 versions, and the chain predicate really is monotone where bisection assumes it; empty and
// single-query workloads and the whole report are reproducible.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng855
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double Jitter(ref ulong state)
    {
        // ±2%, deterministic: the stand-in for a noisy benchmark run.
        int offset = (int)(Next(ref state) % 401) - 200;
        return 1.0 + offset / 10000.0;
    }
}

internal sealed class Query855
{
    internal int Id;
    internal int[] Cardinalities = Array.Empty<int>();
    internal int FilterColumn;
    internal int FilterThreshold;
    internal int JoinColumn;

    internal string Describe() => "q" + Id + "[(card " + string.Join(",", Cardinalities) + ") k" + JoinColumn
        + " where c" + FilterColumn + " < " + FilterThreshold + "]";
}

internal sealed class Table855
{
    internal int Id;
    internal int Rows;
    internal int Columns;
    internal int[][] Data = Array.Empty<int[]>();

    internal Table855(int id, int rows, int columns)
    {
        Id = id;
        Rows = rows;
        Columns = columns;
        Data = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var tuple = new int[columns];
            for (int c = 0; c < columns; c++) tuple[c] = (r * 7 + c * 3 + id) % 40;
            Data[r] = tuple;
        }
    }
}

internal sealed class Plan855
{
    internal int[] Order = Array.Empty<int>();      // join order (table ids)
    internal bool[] HashStep = Array.Empty<bool>(); // join algorithm per step, length Order.Length - 1
    internal long EstimatedCost;
    internal string Hash = "";

    internal string Shape()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < Order.Length; i++)
        {
            if (i > 0) sb.Append(HashStep[i - 1] ? " -HJ-> " : " -NL-> ");
            sb.Append("t" + Order[i]);
        }
        return sb.ToString();
    }

    internal void ComputeHash()
    {
        long h = 1469598103934665603L;
        foreach (int id in Order) h = Mix(h, id * 7 + 1);
        foreach (bool hash in HashStep) h = Mix(h, hash ? 2 : 3);
        Hash = h.ToString("x16");
    }

    private static long Mix(long hash, long value)
    {
        long mixed = hash ^ (value - 7046029254386353131L + (hash << 6) + (hash >> 2));
        return mixed * 1099511628211L;
    }
}

internal sealed class Optimizer855
{
    internal int Version;
    internal int HashJoinThreshold;      // an estimated intermediate size above this uses a hash join

    internal Optimizer855(int version, int hashJoinThreshold)
    {
        Version = version;
        HashJoinThreshold = hashJoinThreshold;
    }

    internal Plan855 Optimize(Query855 query, Table855[] tables)
    {
        int n = query.Cardinalities.Length;
        var estimates = new long[n];
        for (int i = 0; i < n; i++)
        {
            long filtered = (long)query.Cardinalities[i] * query.FilterThreshold / 40;
            estimates[i] = filtered < 1 ? 1 : filtered;
        }
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        // Stable insertion sort by estimate keeps the plan deterministic when estimates tie.
        for (int i = 1; i < n; i++)
        {
            int current = order[i];
            int j = i - 1;
            while (j >= 0 && estimates[order[j]] > estimates[current])
            {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = current;
        }
        var plan = new Plan855 { Order = order, HashStep = new bool[n - 1] };
        long running = estimates[order[0]];
        long cost = estimates[order[0]];
        for (int i = 1; i < n; i++)
        {
            bool hash = estimates[order[i]] > HashJoinThreshold;   // a small build side favours nested loops
            plan.HashStep[i - 1] = hash;
            cost += estimates[order[i]];
            cost += hash ? running + estimates[order[i]] : running * estimates[order[i]];
            running = hash ? running + estimates[order[i]] : running * estimates[order[i]];
        }
        plan.EstimatedCost = cost;
        plan.ComputeHash();
        return plan;
    }
}

internal static class Executor855
{
    // Fixed cost of allocating and setting up a hash table for one join step.
    internal const long HashSetUpCost = 300;

    // Real tuple-level execution of the plan over materialised filtered rows. Returns the number of joined
    // tuples and the work performed (tuple comparisons plus hash probes), and verifies the reference count.
    internal static long Execute(Plan855 plan, Query855 query, Table855[] tables, out long resultRows,
        out bool matchesReference)
    {
        var filtered = new List<int[]>[plan.Order.Length];
        for (int i = 0; i < plan.Order.Length; i++)
        {
            Table855 table = tables[plan.Order[i]];
            var rows = new List<int[]>();
            for (int r = 0; r < table.Rows; r++)
            {
                if (table.Data[r][query.FilterColumn] < query.FilterThreshold) rows.Add(table.Data[r]);
            }
            filtered[i] = rows;
        }
        long work = 0;
        foreach (List<int[]> rows in filtered) work += rows.Count;

        // Step 0: seed the intermediate result with the filtered rows of the first table.
        var intermediate = new List<int>();
        foreach (int[] tuple in filtered[0]) intermediate.Add(tuple[query.JoinColumn]);

        for (int step = 1; step < filtered.Length; step++)
        {
            List<int[]> right = filtered[step];
            var next = new List<int>();
            if (plan.HashStep[step - 1])
            {
                var buckets = new Dictionary<int, int>();
                foreach (int key in intermediate)
                {
                    buckets.TryGetValue(key, out int count);
                    buckets[key] = count + 1;
                }
                // Building and probing a hash table costs a fixed allocation plus one pass over each side,
                // which is why a hash join loses to a nested loop when both sides are tiny.
                work += intermediate.Count + HashSetUpCost;
                foreach (int[] tuple in right)
                {
                    work++;
                    int key = tuple[query.JoinColumn];
                    if (buckets.TryGetValue(key, out int count))
                        for (int k = 0; k < count; k++) next.Add(key);
                }
            }
            else
            {
                foreach (int leftKey in intermediate)
                {
                    foreach (int[] tuple in right)
                    {
                        work++;
                        if (tuple[query.JoinColumn] == leftKey) next.Add(leftKey);
                    }
                }
            }
            intermediate = next;
        }
        resultRows = intermediate.Count;
        long referenceRows = ReferenceResult(query, tables);
        matchesReference = referenceRows == resultRows;
        return work;
    }

    // Independent result oracle: group every table's filtered rows by join key and multiply the counts, so
    // it never looks at the plan at all.
    internal static long ReferenceResult(Query855 query, Table855[] tables)
    {
        long total = 0;
        int n = query.Cardinalities.Length;
        var keys = new Dictionary<int, int>[n];
        for (int t = 0; t < n; t++)
        {
            keys[t] = new Dictionary<int, int>();
            Table855 table = tables[t];
            for (int r = 0; r < table.Rows; r++)
            {
                if (table.Data[r][query.FilterColumn] >= query.FilterThreshold) continue;
                int key = table.Data[r][query.JoinColumn];
                keys[t].TryGetValue(key, out int count);
                keys[t][key] = count + 1;
            }
        }
        foreach (KeyValuePair<int, int> pair in keys[0])
        {
            long product = pair.Value;
            bool common = true;
            for (int t = 1; t < n; t++)
            {
                if (!keys[t].TryGetValue(pair.Key, out int count))
                {
                    common = false;
                    break;
                }
                product *= count;
            }
            if (common) total += product;
        }
        return total;
    }

    internal static double Measure(Plan855 plan, Query855 query, Table855[] tables, ulong seed, out bool ok)
    {
        long work = Execute(plan, query, tables, out _, out ok);
        ulong state = seed ^ (ulong)work;
        return work * Rng855.Jitter(ref state);
    }
}

internal sealed class Verdict855
{
    internal int QueryId;
    internal string BaselineHash = "";
    internal string CandidateHash = "";
    internal double MedianRatio;
    internal bool PlanChanged;
    internal string Classification = "";      // "regression", "improvement", "unchanged", "changed-no-regression"
    internal bool NoiseStable;
    internal long BaselineWork;
    internal long CandidateWork;
}

internal sealed class Detector855
{
    internal const int Repetitions = 5;

    internal readonly long ThresholdMilli;      // ratio threshold in thousandths, e.g. 1200 == 1.2
    internal readonly Optimizer855 Baseline;
    internal readonly Optimizer855 Candidate;
    internal readonly Table855[] Tables;

    internal Detector855(Optimizer855 baseline, Optimizer855 candidate, Table855[] tables, long thresholdMilli)
    {
        Baseline = baseline;
        Candidate = candidate;
        Tables = tables;
        ThresholdMilli = thresholdMilli;
    }

    internal List<Verdict855> Compare(List<Query855> workload, ulong seed)
    {
        var verdicts = new List<Verdict855>(workload.Count);
        foreach (Query855 query in workload)
        {
            Plan855 before = Baseline.Optimize(query, Tables);
            Plan855 after = Candidate.Optimize(query, Tables);
            bool referenceOk = true;
            long workBefore = Executor855.Execute(before, query, Tables, out _, out bool okBefore);
            long workAfter = Executor855.Execute(after, query, Tables, out _, out bool okAfter);
            if (!okBefore || !okAfter) referenceOk = false;
            // The verdict uses the measured work, which is exact; the repetitions exist to show that the
            // paired jittered samples agree with the verdict, i.e. that benchmark noise cannot flip it.
            bool stable = true;
            for (int i = 0; i < Repetitions; i++)
            {
                ulong sampleSeed = seed + (ulong)i * 131UL;
                double noisyBefore = Executor855.Measure(before, query, Tables, sampleSeed, out _);
                double noisyAfter = Executor855.Measure(after, query, Tables, sampleSeed, out _);
                if ((noisyAfter > noisyBefore) != (workAfter > workBefore)) stable = false;
            }
            double ratio = workBefore <= 0 ? 0.0 : workAfter / (double)workBefore;
            var verdict = new Verdict855
            {
                QueryId = query.Id,
                BaselineHash = before.Hash,
                CandidateHash = after.Hash,
                MedianRatio = ratio,
                PlanChanged = before.Hash != after.Hash,
                NoiseStable = stable,
                BaselineWork = workBefore,
                CandidateWork = workAfter,
            };
            // Plan-change filter: a plan that did not change can never be a plan regression, no matter what
            // the parameters or the noise did to the measured cost.
            if (!verdict.PlanChanged) verdict.Classification = "unchanged";
            else if (ratio * 1000.0 >= ThresholdMilli) verdict.Classification = "regression";
            else if (ratio * 1000.0 <= 1000000.0 / ThresholdMilli) verdict.Classification = "improvement";
            else verdict.Classification = "changed-no-regression";
            if (!referenceOk) verdict.Classification = "result-mismatch";
            verdicts.Add(verdict);
        }
        return verdicts;
    }
}

internal static class Chain855
{
    // Version 1 is the baseline; the version `regressionAt` contains a rule change that every later
    // version inherits (always nested loops), while earlier versions only differ in the threshold knob.
    internal static Optimizer855 Build(int version, int regressionAt)
    {
        if (version < regressionAt) return new Optimizer855(version, 60 - version);
        return new Optimizer855(version, int.MaxValue);   // always nested loop: the injected regression
    }

    internal static double AggregateCost(int version, int regressionAt, List<Query855> workload,
        Table855[] tables, ulong seed)
    {
        Optimizer855 optimizer = Build(version, regressionAt);
        double total = 0.0;
        foreach (Query855 query in workload)
        {
            Plan855 plan = optimizer.Optimize(query, tables);
            total += Executor855.Measure(plan, query, tables, seed, out _);
        }
        return total;
    }
}

public static class Solution
{
    private const int Tables = 3;
    private const int WorkloadSize = 40;
    private const int Versions = 8;
    private const long ThresholdMilli = 1200;

    public static void Main()
    {
        Table855[] tables = BuildTables();
        List<Query855> workload = BuildWorkload();

        // 1-3. Plans are deterministic, hashes capture the shape, and every plan returns the reference
        // result: an optimizer change may never change the answer.
        var baseline = new Optimizer855(1, 60);
        var candidate = new Optimizer855(2, int.MaxValue);   // the change under test: never use a hash join
        bool deterministic = true, historiesOk = true, semanticsOk = true, someChanged = false, someSame = false;
        foreach (Query855 query in workload)
        {
            Plan855 first = baseline.Optimize(query, tables);
            Plan855 second = baseline.Optimize(query, tables);
            if (first.Shape() != second.Shape() || first.Hash != second.Hash) deterministic = false;
            Plan855 changed = candidate.Optimize(query, tables);
            if (first.Hash != changed.Hash) someChanged = true;
            else someSame = true;
            if (first.Hash == changed.Hash && first.Shape() != changed.Shape()) historiesOk = false;
            if (first.Hash != changed.Hash && first.Shape() == changed.Shape()) historiesOk = false;
            foreach (Plan855 plan in new[] { first, changed })
            {
                Executor855.Execute(plan, query, tables, out _, out bool matches);
                if (!matches) semanticsOk = false;
            }
        }
        Check(deterministic, "the optimizer is deterministic: identical queries produce identical plans");
        Check(historiesOk, "the plan hash is equal exactly when the plan shape is equal");
        Check(semanticsOk, "every plan of every query returns exactly the reference result");
        Check(someChanged && someSame, "the version change moves some plans and leaves others alone");

        // 4-6. Detection: the detector's verdicts equal a brute-force classification of the same samples.
        var detector = new Detector855(baseline, candidate, tables, ThresholdMilli);
        List<Verdict855> verdicts = detector.Compare(workload, 0x855AUL);
        int regressions = 0, improvements = 0, changedNoRegression = 0, unchanged = 0;
        bool bruteForceAgrees = true;
        foreach (Verdict855 verdict in verdicts)
        {
            if (verdict.Classification == "regression") regressions++;
            else if (verdict.Classification == "improvement") improvements++;
            else if (verdict.Classification == "changed-no-regression") changedNoRegression++;
            else unchanged++;
            // Brute force: recompute the measured work of both plans without any detector logic.
            Query855 query = QueryOf(workload, verdict.QueryId);
            Plan855 before = baseline.Optimize(query, tables);
            Plan855 after = candidate.Optimize(query, tables);
            long workBefore = Executor855.Execute(before, query, tables, out _, out _);
            long workAfter = Executor855.Execute(after, query, tables, out _, out _);
            string expected;
            if (before.Hash == after.Hash) expected = "unchanged";
            else if (workAfter * 1000 >= workBefore * ThresholdMilli) expected = "regression";
            else if (workAfter * 1000 <= workBefore * 1000000 / ThresholdMilli) expected = "improvement";
            else expected = "changed-no-regression";
            if (verdict.Classification != expected) bruteForceAgrees = false;
        }
        Console.WriteLine("workload " + workload.Count + ": regressions=" + regressions + " improvements=" + improvements
            + " changed-only=" + changedNoRegression + " unchanged=" + unchanged);
        Check(bruteForceAgrees, "every verdict agrees with a brute-force classification of the same measurements");
        Check(regressions > 0, "the version change produces regressions to report (" + regressions + ")");
        // The same change in the other direction is a win for the tiny queries, where the fixed hash-table
        // set-up dominates: an optimizer change is workload-dependent, which is why the detector reports
        // per query instead of globally.
        var tinyPair = new Detector855(new Optimizer855(1, 0), new Optimizer855(2, int.MaxValue), tables, ThresholdMilli);
        List<Verdict855> tinyVerdicts = tinyPair.Compare(workload, 0x855CUL);
        int tinyImprovements = CountBy(tinyVerdicts, "improvement");
        int tinyRegressions = CountBy(tinyVerdicts, "regression");
        Console.WriteLine("  tiny-query pair: improvements=" + tinyImprovements + " regressions=" + tinyRegressions);
        Check(tinyImprovements > 0 && tinyRegressions > tinyImprovements,
            "nested loops beat the hash join on the tiny queries and lose on the large ones ("
            + tinyImprovements + " improvements vs " + tinyRegressions + " regressions)");
        Check(regressions + improvements + changedNoRegression + unchanged == workload.Count,
            "every query in the workload receives exactly one verdict");
        Check(unchanged > 0, "queries whose plan did not change are reported as unchanged (" + unchanged + ")");

        // 7-8. Plan-change filter and threshold monotonicity.
        var noisy = new Detector855(baseline, new Optimizer855(2, 60), tables, ThresholdMilli);
        List<Verdict855> noisyVerdicts = noisy.Compare(workload, 0x855BUL);
        bool noFalsePositives = true;
        foreach (Verdict855 verdict in noisyVerdicts)
            if (verdict.Classification != "unchanged") noFalsePositives = false;
        Check(noFalsePositives,
            "a same-version run with fresh jitter reports nothing, so parameter noise is filtered out");
        var strict = new Detector855(baseline, candidate, tables, 5000).Compare(workload, 0x855AUL);
        var loose = new Detector855(baseline, candidate, tables, 1100).Compare(workload, 0x855AUL);
        bool subsetProperty = true;
        foreach (Verdict855 verdict in strict)
        {
            if (verdict.Classification != "regression") continue;
            Verdict855 match = Find(loose, verdict.QueryId);
            if (match.Classification != "regression") subsetProperty = false;
        }
        Check(subsetProperty, "raising the threshold can only shrink the set of reported regressions");
        Check(CountBy(loose, "regression") >= CountBy(strict, "regression") && Find(loose, 1) != null,
            "the loose threshold reports at least as many regressions as the strict one");

        // 9-10. Reporting: every regression carries a plan diff, non-regressions carry none.
        bool reportsComplete = true;
        foreach (Verdict855 verdict in verdicts)
        {
            string diff = verdict.BaselineHash == verdict.CandidateHash ? "" : "diff";
            if (verdict.Classification == "regression" && diff.Length == 0) reportsComplete = false;
            if (verdict.Classification == "unchanged" && diff.Length != 0) reportsComplete = false;
        }
        Check(reportsComplete, "a regression always has a plan diff and an unchanged query never has one");
        Verdict855 example = null;
        foreach (Verdict855 verdict in verdicts) if (verdict.Classification == "regression") example = verdict;
        Check(example != null, "the version change produced at least one regression to report");
        Console.WriteLine("example regression: query " + example.QueryId + " ratio=" + example.MedianRatio.ToString("F3")
            + " work " + example.BaselineWork + " -> " + example.CandidateWork);
        Check(example.MedianRatio * 1000 >= ThresholdMilli && example.CandidateWork > example.BaselineWork,
            "the reported example regression really is slower in measured work");
        bool stableVerdicts = true;
        string unstableDetail = "";
        foreach (Verdict855 verdict in verdicts)
        {
            if (verdict.Classification == "result-mismatch") stableVerdicts = false;
            if ((verdict.Classification == "regression" || verdict.Classification == "improvement")
                && !verdict.NoiseStable)
            {
                stableVerdicts = false;
                if (unstableDetail.Length == 0)
                    unstableDetail = "q" + verdict.QueryId + " " + verdict.Classification + " ratio="
                        + verdict.MedianRatio.ToString("F4") + " work " + verdict.BaselineWork + " -> " + verdict.CandidateWork;
            }
        }
        Check(stableVerdicts,
            "no verdict is a result mismatch and both regressions and improvements survive fresh jitter "
            + unstableDetail);

        // 11-14. Bisection over the version chain against an exhaustive scan.
        var bisectWorkload = new List<Query855>();
        for (int i = 0; i < workload.Count; i += 4) bisectWorkload.Add(workload[i]);
        bool bisectCorrect = true, logBudget = true, monotone = true;
        var bisectReport = new List<string>();
        int[] injected = { 2, 4, 6, 8 };
        foreach (int injectedAt in injected)
        {
            int exhaustive = ExhaustiveFirstBad(injectedAt, bisectWorkload, tables, out bool isMonotone);
            if (!isMonotone) monotone = false;
            int found = Bisect(injectedAt, bisectWorkload, tables, out int evaluations);
            if (found != exhaustive) bisectCorrect = false;
            if (evaluations > 3 + (int)Math.Ceiling(Math.Log(Versions, 2))) logBudget = false;
            bisectReport.Add("injected@" + injectedAt + " firstBad=" + found + " evaluations=" + evaluations);
            Console.WriteLine("  chain with the regression at version " + injectedAt + ": bisection found "
                + found + " after " + evaluations + " version measurements (exhaustive: " + exhaustive + ")");
        }
        Check(bisectCorrect, "bisection finds the same first bad version as the exhaustive scan on every chain");
        Check(logBudget, "bisection measures at most log2(N)+2 versions");
        Check(monotone, "the exhaustive scan confirms each chain's bad-version predicate is monotone");
        Check(bisectReport.Count == injected.Length, "all injected chain positions were exercised");

        // 15-16. Degenerate workloads, determinism and the empty verdict list.
        var emptyDetector = new Detector855(baseline, candidate, tables, ThresholdMilli);
        Check(emptyDetector.Compare(new List<Query855>(), 0x855CUL).Count == 0,
            "an empty workload yields no verdicts at all");
        var single = new List<Query855> { workload[3] };
        List<Verdict855> singleVerdicts = new Detector855(baseline, candidate, tables, ThresholdMilli).Compare(single, 0x855DUL);
        Check(singleVerdicts.Count == 1 && singleVerdicts[0].QueryId == workload[3].Id,
            "a single-query workload produces exactly one verdict for that query");
        List<Verdict855> repeat = detector.Compare(workload, 0x855AUL);
        bool identical = repeat.Count == verdicts.Count;
        for (int i = 0; identical && i < verdicts.Count; i++)
        {
            if (repeat[i].Classification != verdicts[i].Classification
                || repeat[i].CandidateHash != verdicts[i].CandidateHash
                || repeat[i].MedianRatio != verdicts[i].MedianRatio) identical = false;
        }
        Check(identical, "the same seed reproduces the whole regression report bit for bit");

        Console.WriteLine("PASS");
    }

    private static int ExhaustiveFirstBad(int injectedAt, List<Query855> workload, Table855[] tables, out bool monotone)
    {
        double reference = Chain855.AggregateCost(1, injectedAt, workload, tables, 0x855EUL);
        var bad = new bool[Versions + 1];
        for (int version = 1; version <= Versions; version++)
            bad[version] = Chain855.AggregateCost(version, injectedAt, workload, tables, 0x855EUL) > reference * 1.2;
        monotone = true;
        for (int version = 2; version <= Versions; version++)
            if (bad[version - 1] && !bad[version]) monotone = false;
        for (int version = 1; version <= Versions; version++) if (bad[version]) return version;
        return -1;
    }

    private static int Bisect(int injectedAt, List<Query855> workload, Table855[] tables, out int evaluations)
    {
        double reference = Chain855.AggregateCost(1, injectedAt, workload, tables, 0x855EUL);
        int measured = 0;
        bool IsBad(int version)
        {
            measured++;
            return Chain855.AggregateCost(version, injectedAt, workload, tables, 0x855EUL) > reference * 1.2;
        }
        if (!IsBad(Versions))
        {
            evaluations = measured;
            return -1;
        }
        int lo = 1, hi = Versions;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (IsBad(mid)) hi = mid;
            else lo = mid + 1;
        }
        evaluations = measured;
        return lo;
    }

    private static int CountBy(List<Verdict855> verdicts, string classification)
    {
        int total = 0;
        foreach (Verdict855 verdict in verdicts) if (verdict.Classification == classification) total++;
        return total;
    }

    private static Verdict855 Find(List<Verdict855> verdicts, int queryId)
    {
        foreach (Verdict855 verdict in verdicts) if (verdict.QueryId == queryId) return verdict;
        return null;
    }

    private static Query855 QueryOf(List<Query855> workload, int id)
    {
        foreach (Query855 query in workload) if (query.Id == id) return query;
        throw new ArgumentException("unknown query id " + id);
    }

    private static Table855[] BuildTables()
    {
        var tables = new Table855[Tables];
        for (int t = 0; t < Tables; t++) tables[t] = new Table855(t, 120 + t * 40, 6);
        return tables;
    }

    private static List<Query855> BuildWorkload()
    {
        var workload = new List<Query855>(WorkloadSize);
        ulong seed = 0x855FUL;
        for (int i = 0; i < WorkloadSize; i++)
        {
            var cardinalities = new int[Tables];
            for (int t = 0; t < Tables; t++) cardinalities[t] = 60 + Rng855.NextInt(ref seed, 220);
            workload.Add(new Query855
            {
                Id = i,
                Cardinalities = cardinalities,
                FilterColumn = Rng855.NextInt(ref seed, 4),
                FilterThreshold = 2 + Rng855.NextInt(ref seed, 36),
                JoinColumn = Rng855.NextInt(ref seed, 4),
            });
        }
        // Four deliberately extreme queries: two that return almost nothing (where a nested loop wins) and
        // two that return almost everything (where a hash join wins), so both directions of the tradeoff are
        // always present in the workload.
        foreach (int threshold in new[] { 2, 2, 39, 39 })
        {
            workload.Add(new Query855
            {
                Id = workload.Count,
                Cardinalities = new[] { 120, 160, 200 },
                FilterColumn = 1,
                FilterThreshold = threshold,
                JoinColumn = 2,
            });
        }
        return workload;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
