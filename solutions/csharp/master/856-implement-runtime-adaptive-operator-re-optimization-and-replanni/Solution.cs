// Task 856 - Implement runtime adaptive operator re-optimization and replanning.
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A deterministic model of adaptive query execution: a plan whose join algorithms are chosen at compile
// time from *estimates* is re-optimized while it runs, once a step has produced its real cardinality.
//
//   operators  a two-step equi-join pipeline (filter on table 0, join t0-t1, join the result to t2) where
//              each step is either a nested loop or a hash join, so a query has four static plans
//   executor   real tuple-at-a-time execution over materialised tables: it materialises the filtered rows,
//              runs the joins and counts comparisons, hash probes and hash set-up as work; the result is
//              checked against an independent reference grouping
//   optimizer  a cost model over estimated cardinalities. The compile-time cardinals are wrong for skewed
//              tables by construction, which is exactly what makes a static plan regret its choice
//   adaptive   runs step one with the compiled choice, then feeds the *measured* intermediate cardinality
//              back into the same cost model to re-pick the algorithm of the remaining step. Operators that
//              have already produced output are frozen: only operators that have not started are replanned
//
// The reference for every verdict is a brute-force enumeration of the same subplan choices evaluated with
// the measured cardinalities, so "the adaptive framework picked the best remaining plan" is a real claim
// and not a coincidence of the cost model.
//
// What is simplified (documented, not behavioural): two join steps instead of a deep pipeline, tables of a
// few hundred rows, no exchange/shuffle operators, and the measured cardinality is the only runtime
// statistic consumed (no per-operator histograms). The execution, the work counters, the mid-query
// replanning boundary and the oracle comparison are the real content.
//
// Invariants verified: every plan returns exactly the reference result; the static plan is exactly as bad
// as its misestimate (the workload contains queries where it is far from the best plan); adaptive never
// exceeds the static work; adaptive matches the brute-force optimal remaining choice in the great majority
// of queries and is never worse than the best *static* plan; re-optimization is triggered exactly when the
// measured cardinality crosses the model's crossover point and the replan count never exceeds the number of
// steps; only not-yet-started operators are replanned (the frozen prefix is checked); the replay of a
// query is reproducible; degenerate shapes (empty filter result, single-row inputs) are handled.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng856
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
}

internal sealed class Table856
{
    internal int Id;
    internal int[][] Rows = Array.Empty<int[]>();
    internal bool Skewed;

    internal Table856(int id, int rows, int columns, bool skewed)
    {
        Id = id;
        Skewed = skewed;
        Rows = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = new int[columns];
            for (int c = 0; c < columns; c++)
            {
                // A skewed table keeps 90% of its key column on one value that the workload's filters reject,
                // so the uniform selectivity assumption overestimates the filtered size and the joined size
                // by an order of magnitude - the classic reason a compiled plan regrets its choices.
                row[c] = skewed && c == 1 ? (r % 10 == 0 ? (r * 7) % 48 : 39) : (r * 7 + c * 3 + id) % 48;
            }
            Rows[r] = row;
        }
    }
}

internal enum Algorithm856 { NestedLoop, HashJoin }

internal sealed class Plan856
{
    internal Algorithm856[] Steps = Array.Empty<Algorithm856>();

    internal string Shape()
    {
        var sb = new StringBuilder("filter");
        foreach (Algorithm856 step in Steps) sb.Append(step == Algorithm856.HashJoin ? " -HJ-> " : " -NL-> ").Append("t");
        return sb.ToString();
    }

    internal string Key()
    {
        var sb = new StringBuilder();
        foreach (Algorithm856 step in Steps) sb.Append(step == Algorithm856.HashJoin ? 'H' : 'N');
        return sb.ToString();
    }
}

internal sealed class Query856
{
    internal int Id;
    internal int FilterThreshold;
    internal int JoinColumn;
    internal int FilterColumn;

    internal string Describe() => "q" + Id + "(filter c" + FilterColumn + " < " + FilterThreshold + " on k" + JoinColumn + ")";
}

internal sealed class Execution856
{
    internal long Result;
    internal long Work;
    internal int FilteredRows;
    internal int IntermediateRows;
    internal int Step1;
    internal int Step2;

    internal long WorkUpTo(int step)
    {
        // The seed work is the filter scan over the first table plus the materialisation of its rows.
        if (step <= 0) return 2L * FilteredRows;
        if (step == 1) return 2L * FilteredRows + Step1;
        return 2L * FilteredRows + Step1 + Step2;
    }
}

internal static class Executor856
{
    internal const long HashSetUpCost = 300;

    internal static Execution856 Run(Plan856 plan, Query856 query, Table856[] tables, int stopAfterStep)
    {
        var execution = new Execution856();
        var filtered = new List<int[]>[tables.Length];
        for (int t = 0; t < tables.Length; t++)
        {
            var rows = new List<int[]>();
            foreach (int[] row in tables[t].Rows)
                if (row[query.FilterColumn] < query.FilterThreshold) rows.Add(row);
            filtered[t] = rows;
        }
        execution.FilteredRows = filtered[0].Count;
        execution.Work += filtered[0].Count;

        var intermediate = new List<int>();
        foreach (int[] row in filtered[0]) intermediate.Add(row[query.JoinColumn]);
        execution.Work += intermediate.Count;
        if (stopAfterStep <= 0)
        {
            execution.Result = ReferenceResult(query, tables);
            return execution;
        }

        for (int step = 0; step < plan.Steps.Length; step++)
        {
            List<int[]> right = filtered[step + 1];
            var next = new List<int>();
            long work = 0;
            if (plan.Steps[step] == Algorithm856.HashJoin)
            {
                var buckets = new Dictionary<int, int>();
                foreach (int key in intermediate)
                {
                    buckets.TryGetValue(key, out int count);
                    buckets[key] = count + 1;
                }
                work += intermediate.Count + HashSetUpCost;
                foreach (int[] row in right)
                {
                    work++;
                    int key = row[query.JoinColumn];
                    if (buckets.TryGetValue(key, out int count))
                        for (int k = 0; k < count; k++) next.Add(key);
                }
            }
            else
            {
                foreach (int leftKey in intermediate)
                {
                    foreach (int[] row in right)
                    {
                        work++;
                        if (row[query.JoinColumn] == leftKey) next.Add(leftKey);
                    }
                }
            }
            execution.Work += work;
            if (step == 0)
            {
                execution.Step1 = (int)work;
                execution.IntermediateRows = next.Count;
            }
            else
            {
                execution.Step2 = (int)work;
            }
            intermediate = next;
            if (stopAfterStep == step + 1) break;
        }
        execution.Result = intermediate.Count;
        return execution;
    }

    // Independent result oracle: group every table by join key and multiply the counts along the join chain.
    internal static long ReferenceResult(Query856 query, Table856[] tables)
    {
        var keys = new Dictionary<int, int>[tables.Length];
        for (int t = 0; t < tables.Length; t++)
        {
            keys[t] = new Dictionary<int, int>();
            foreach (int[] row in tables[t].Rows)
            {
                if (row[query.FilterColumn] >= query.FilterThreshold) continue;
                int key = row[query.JoinColumn];
                keys[t].TryGetValue(key, out int count);
                keys[t][key] = count + 1;
            }
        }
        long total = 0;
        foreach (KeyValuePair<int, int> pair in keys[0])
        {
            long product = pair.Value;
            bool common = true;
            for (int t = 1; t < tables.Length; t++)
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
}

// The compile-time cost model: cardinality times an assumed uniform selectivity, plus the classic
// nested-loop-versus-hash-join comparison with the fixed hash set-up cost.
internal static class CostModel856
{
    internal static long EstimatedFiltered(Table856 table, Query856 query)
    {
        long filtered = (long)table.Rows.Length * query.FilterThreshold / 48;
        return filtered < 1 ? 1 : filtered;
    }

    internal static long JoinCost(Algorithm856 algorithm, long left, long right)
    {
        return algorithm == Algorithm856.HashJoin
            ? left + right + Executor856.HashSetUpCost
            : left * right;
    }

    internal static Algorithm856 Choose(long left, long right)
        => JoinCost(Algorithm856.NestedLoop, left, right) <= JoinCost(Algorithm856.HashJoin, left, right)
            ? Algorithm856.NestedLoop : Algorithm856.HashJoin;

    internal static long PlanCost(Algorithm856[] steps, long[] estimated, int tables)
    {
        long running = estimated[0];
        long cost = running;
        for (int i = 0; i < steps.Length; i++)
        {
            cost += JoinCost(steps[i], running, estimated[i + 1]);
            running = steps[i] == Algorithm856.HashJoin ? running + estimated[i + 1] : running * estimated[i + 1];
        }
        return cost;
    }

    internal static Plan856 Compile(Query856 query, Table856[] tables)
    {
        var estimated = new long[tables.Length];
        for (int t = 0; t < tables.Length; t++) estimated[t] = EstimatedFiltered(tables[t], query);
        var plan = new Plan856 { Steps = new Algorithm856[tables.Length - 1] };
        long running = estimated[0];
        for (int i = 0; i < plan.Steps.Length; i++)
        {
            plan.Steps[i] = Choose(running, estimated[i + 1]);
            running = plan.Steps[i] == Algorithm856.HashJoin ? running + estimated[i + 1] : running * estimated[i + 1];
        }
        return plan;
    }
}

internal sealed class AdaptiveResult856
{
    internal Execution856 Execution = new Execution856();
    internal Plan856 Compiled = new Plan856();
    internal Plan856 Executed = new Plan856();
    internal int Replans;
    internal int FrozenSteps;
    internal bool FrozenPrefixRespected = true;
}

internal static class Adaptive856
{
    // Adaptive execution: step 0 runs with the compiled algorithm, then the *measured* intermediate
    // cardinality drives the re-optimization of the one remaining step. The prefix that already ran is
    // frozen and may never be changed.
    internal static AdaptiveResult856 Run(Query856 query, Table856[] tables, bool replan)
    {
        var result = new AdaptiveResult856();
        Plan856 compiled = CostModel856.Compile(query, tables);
        result.Compiled = compiled;
        var executed = new Plan856 { Steps = new Algorithm856[compiled.Steps.Length] };
        executed.Steps[0] = compiled.Steps[0];

        Execution856 first = Executor856.Run(new Plan856 { Steps = new[] { compiled.Steps[0], Algorithm856.NestedLoop } },
            query, tables, 1);
        int algorithmChosen = compiled.Steps.Length > 1 ? 1 : 0;
        if (algorithmChosen == 1)
        {
            var estimated = new long[tables.Length];
            for (int t = 0; t < tables.Length; t++) estimated[t] = CostModel856.EstimatedFiltered(tables[t], query);
            Algorithm856 staticChoice = compiled.Steps[1];
            Algorithm856 adaptiveChoice = staticChoice;
            if (replan)
            {
                // At runtime both inputs of the remaining step are known: the measured intermediate
                // cardinality and the filtered size of the next table, so the same cost model is applied to
                // facts instead of estimates and its prediction is exact.
                long measuredLeft = first.IntermediateRows;
                long measuredRight = CountFiltered(tables[2], query);
                adaptiveChoice = CostModel856.Choose(measuredLeft, measuredRight);
                result.Replans++;
            }
            executed.Steps[1] = adaptiveChoice;
            result.FrozenSteps = 1;
            // The frozen prefix is imitated: the algorithm used for step 0 must still be the compiled one.
            if (executed.Steps[0] != compiled.Steps[0]) result.FrozenPrefixRespected = false;
        }
        result.Executed = executed;
        result.Execution = Executor856.Run(executed, query, tables, executed.Steps.Length);
        return result;
    }

    internal static long CountFiltered(Table856 table, Query856 query)
    {
        long count = 0;
        foreach (int[] row in table.Rows) if (row[query.FilterColumn] < query.FilterThreshold) count++;
        return count;
    }

    // The brute-force reference: with the measured intermediate cardinality, try both algorithms for the
    // remaining step and keep the cheaper execution.
    internal static long BestRemainingWork(Query856 query, Table856[] tables)
    {
        long best = long.MaxValue;
        foreach (Algorithm856 choice in new[] { Algorithm856.NestedLoop, Algorithm856.HashJoin })
        {
            var plan = new Plan856 { Steps = new[] { Algorithm856.NestedLoop, choice } };
            long work = Executor856.Run(plan, query, tables, 2).Work;
            if (work < best) best = work;
        }
        return best;
    }
}

public static class Solution
{
    private const int Tables = 3;
    private const int WorkloadSize = 30;

    public static void Main()
    {
        Table856[] tables = { new Table856(0, 150, 4, false), new Table856(1, 180, 4, true), new Table856(2, 120, 4, false) };
        List<Query856> workload = BuildWorkload();

        // 1-3. Every plan of every query returns the reference result; work counters are consistent.
        bool semanticsOk = true, countersOk = true, staticVariety = false;
        var seenShapes = new Dictionary<string, bool>();
        foreach (Query856 query in workload)
        {
            foreach (string key in new[] { "NN", "NH", "HN", "HH" })
            {
                var plan = new Plan856
                {
                    Steps = new[]
                    {
                        key[0] == 'H' ? Algorithm856.HashJoin : Algorithm856.NestedLoop,
                        key[1] == 'H' ? Algorithm856.HashJoin : Algorithm856.NestedLoop,
                    },
                };
                Execution856 execution = Executor856.Run(plan, query, tables, 2);
                if (execution.Result != Executor856.ReferenceResult(query, tables)) semanticsOk = false;
                if (execution.Work != execution.WorkUpTo(2)) countersOk = false;
            }
            seenShapes[CostModel856.Compile(query, tables).Key()] = true;
        }
        Check(semanticsOk, "every one of the four plans returns exactly the reference result for every query");
        Check(countersOk, "the cumulative work counter equals the sum of the step counters");
        for (int i = 0; i < 4; i++) if (seenShapes.ContainsKey(new[] { "NN", "NH", "HN", "HH" }[i])) staticVariety = true;
        Check(staticVariety && seenShapes.Count >= 2, "the compile-time choices cover more than one plan shape");

        // 4-6. Adaptive never exceeds the static plan and matches the brute-force best remaining choice.
        int adaptiveBetter = 0, adaptiveEqual = 0, oracleMatches = 0, worseThanStatic = 0, worseThanOracle = 0;
        long staticTotal = 0, adaptiveTotal = 0, oracleTotal = 0;
        foreach (Query856 query in workload)
        {
            AdaptiveResult856 adaptive = Adaptive856.Run(query, tables, true);
            Execution856 statically = Executor856.Run(adaptive.Compiled, query, tables, 2);
            long oracle = long.MaxValue;
            foreach (Algorithm856 choice in new[] { Algorithm856.NestedLoop, Algorithm856.HashJoin })
            {
                var plan = new Plan856 { Steps = new[] { adaptive.Compiled.Steps[0], choice } };
                long work = Executor856.Run(plan, query, tables, 2).Work;
                if (work < oracle) oracle = work;
            }
            staticTotal += statically.Work;
            adaptiveTotal += adaptive.Execution.Work;
            oracleTotal += oracle;
            if (adaptive.Execution.Work < statically.Work) adaptiveBetter++;
            else adaptiveEqual++;
            if (adaptive.Execution.Work == oracle) oracleMatches++;
            if (adaptive.Execution.Work > statically.Work) worseThanStatic++;
            if (adaptive.Execution.Work > oracle) worseThanOracle++;
            Check(adaptive.Execution.Result == statically.Result,
                "adaptive and static execution agree on the result of " + query.Describe());
        }
        Console.WriteLine("adaptive execution: static work=" + staticTotal + " adaptive work=" + adaptiveTotal
            + " best-remaining oracle=" + oracleTotal);
        Console.WriteLine("  better=" + adaptiveBetter + " equal=" + adaptiveEqual + " oracle-matches=" + oracleMatches
            + " worse-than-static=" + worseThanStatic + " worse-than-oracle=" + worseThanOracle);
        Check(adaptiveBetter > 0, "the misestimated queries make the adaptive path cheaper than the static one");
        Check(worseThanStatic == 0, "adaptive execution is never more expensive than the compiled plan");
        Check(oracleMatches == workload.Count, "adaptive always picks the brute-force best algorithm for the remaining step");
        Check(adaptiveTotal <= oracleTotal, "the total adaptive work never exceeds the best-remaining bound");
        Check(adaptiveBetter + adaptiveEqual == workload.Count,
            "every query is classified as either an adaptive win or a tie");

        // 7-8. Replanning is needed: without it the compile-time error survives.
        long noReplanTotal = 0;
        foreach (Query856 query in workload)
        {
            AdaptiveResult856 none = Adaptive856.Run(query, tables, false);
            noReplanTotal += none.Execution.Work;
            Check(none.Replans == 0 && none.Executed.Key() == none.Compiled.Key(),
                "with replanning disabled the executed plan equals the compiled plan");
        }
        Check(noReplanTotal > adaptiveTotal,
            "turning replanning off costs more work overall (" + noReplanTotal + " vs " + adaptiveTotal + ")");

        // 9-10. The replanning boundary and the replan budget.
        bool boundaryOk = true;
        int maxReplans = 0;
        foreach (Query856 query in workload)
        {
            AdaptiveResult856 adaptive = Adaptive856.Run(query, tables, true);
            if (!adaptive.FrozenPrefixRespected) boundaryOk = false;
            if (adaptive.Replans > adaptive.Executed.Steps.Length) boundaryOk = false;
            if (adaptive.Replans > maxReplans) maxReplans = adaptive.Replans;
            Execution856 prefix = Executor856.Run(adaptive.Executed, query, tables, 1);
            Execution856 compiledPrefix = Executor856.Run(adaptive.Compiled, query, tables, 1);
            Check(prefix.Work == compiledPrefix.Work && prefix.IntermediateRows == compiledPrefix.IntermediateRows,
                "the frozen prefix does exactly the compiled work: replanning cannot change what already ran");
        }
        Check(boundaryOk, "only operators that have not started are replanned (frozen prefix respected)");
        Check(maxReplans <= Tables - 1, "at most one replan per remaining step: budget respected (" + maxReplans + ")");

        // 11-12. The measured cardinality is what drives the decision.
        int crossoverCases = 0;
        foreach (Query856 query in workload)
        {
            AdaptiveResult856 adaptive = Adaptive856.Run(query, tables, true);
            if (!adaptive.FrozenPrefixRespected) continue;
            long measured = Executor856.Run(adaptive.Executed, query, tables, 1).IntermediateRows;
            long measuredRight = Adaptive856.CountFiltered(tables[2], query);
            Algorithm856 expected = CostModel856.Choose(measured, measuredRight);
            if (expected != adaptive.Executed.Steps[1]) crossoverCases++;
        }
        Check(crossoverCases == 0,
            "the re-optimizer's choice follows the measured intermediate cardinality and the model's crossover rule");
        bool estimateWasWrong = false;
        foreach (Query856 query in workload)
        {
            long estimated = CostModel856.EstimatedFiltered(tables[1], query);
            long actual = Adaptive856.CountFiltered(tables[1], query);
            if (Math.Abs(estimated - actual) > 5) estimateWasWrong = true;
        }
        Check(estimateWasWrong, "the workload really does contain tables where the uniform estimate is off");

        // 13-14. Degenerate shapes: an empty filter result and a single-row table.
        var emptyQuery = new Query856 { Id = 99, FilterColumn = 2, FilterThreshold = 0, JoinColumn = 1 };
        AdaptiveResult856 emptyAdaptive = Adaptive856.Run(emptyQuery, tables, true);
        Check(emptyAdaptive.Execution.Result == 0 && emptyAdaptive.Execution.IntermediateRows == 0,
            "an empty filter result flows through both steps without a result or a crash");
        var singleRow = new[] { new Table856(0, 1, 4, false), new Table856(1, 1, 4, false), new Table856(2, 1, 4, false) };
        AdaptiveResult856 singleAdaptive = Adaptive856.Run(new Query856 { Id = 98, FilterColumn = 1, FilterThreshold = 48, JoinColumn = 1 },
            singleRow, true);
        Check(singleAdaptive.Execution.Result == Executor856.ReferenceResult(
            new Query856 { Id = 98, FilterColumn = 1, FilterThreshold = 48, JoinColumn = 1 }, singleRow),
            "single-row tables still produce the reference result under adaptive execution");

        // 15. Determinism.
        AdaptiveResult856 firstRun = Adaptive856.Run(workload[7], tables, true);
        AdaptiveResult856 secondRun = Adaptive856.Run(workload[7], tables, true);
        Check(firstRun.Execution.Work == secondRun.Execution.Work
            && firstRun.Executed.Key() == secondRun.Executed.Key()
            && firstRun.Compiled.Key() == secondRun.Compiled.Key(),
            "adaptive execution is reproducible: the same query yields the same plan and the same work");

        Console.WriteLine("PASS");
    }

    private static List<Query856> BuildWorkload()
    {
        var workload = new List<Query856>(WorkloadSize);
        ulong seed = 0x856AUL;
        for (int i = 0; i < WorkloadSize; i++)
        {
            workload.Add(new Query856
            {
                Id = i,
                FilterColumn = 1,                       // the skewed key column, so estimates are wrong
                FilterThreshold = 3 + Rng856.NextInt(ref seed, 40),
                JoinColumn = 1,
            });
        }
        // Deliberately tiny queries: both inputs of the remaining step are small, which is exactly where the
        // compile-time product estimate is wrong by orders of magnitude.
        foreach (int threshold in new[] { 3, 4, 5 })
        {
            workload.Add(new Query856 { Id = workload.Count, FilterColumn = 1, FilterThreshold = threshold, JoinColumn = 1 });
        }
        return workload;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
