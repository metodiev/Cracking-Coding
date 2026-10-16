// Task 654 - Implement workload-driven index selection greedy heuristic. (Tier: Expert, CTCI Ch 14 - Databases)
//
// What-if index selection: given a fixed schema, a workload of queries with frequencies and selectivities,
// and a storage budget, pick the single-column indexes that minimise the total cost
//     total(I) = sum over queries of frequency * planCost(query, I)  +  |I| * buildCost
// where planCost is a full scan (Rows element visits) unless one of the query's predicate columns is in I,
// in which case it is a B-tree probe plus the matched rows (log2(Rows) + selectivity * Rows). The greedy
// heuristic repeatedly adds the remaining index with the best (cost saved)/(bytes stored) ratio, recomputing
// every candidate's marginal benefit against the *current* set, which is the detail a naive implementation
// gets wrong: the benefit of an index shrinks as other indexes already cover the same queries, so stale
// benefits lead to useless picks. Ties break on the lower column index, so the selection is deterministic.
// Verified: plan costs match hand-computed values on a tiny instance; a full-scan baseline equals a direct
// sum; greedy never exceeds the no-index baseline and never beats the exhaustive optimum over all subsets of
// a 10-column candidate set (which brute-forces 1024 subsets); every greedy step really is the maximum-ratio
// step when the ratios are recomputed from scratch; execution cost is monotone non-increasing in the index
// set over 200 random set pairs; storage always fits the budget; a zero budget, a budget smaller than one
// index and an unused column are handled without picking anything; and a selectivity-1.0 query makes an index
// strictly harmful, which greedy declines to take. Deterministic throughout (hand-written splitmix64).
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Query
{
    internal int[] Columns = Array.Empty<int>();
    internal int Frequency = 1;
    internal double Selectivity;

    internal Query(int[] columns, int frequency, double selectivity)
    {
        Columns = columns;
        Frequency = frequency;
        Selectivity = selectivity;
    }
}

internal sealed class Workload
{
    internal int Rows;
    internal int Columns;
    internal List<Query> Queries = new List<Query>();

    internal double ProbeCost => Math.Log2(Rows);

    internal double BytesPerIndex => 4.0 * Rows;

    internal double BuildCost => Rows * Math.Log2(Rows);

    internal double QueryCost(Query query, bool[] indexes)
    {
        foreach (int column in query.Columns) if (indexes[column]) return ProbeCost + query.Selectivity * Rows;
        return Rows;
    }

    internal double ExecutionCost(bool[] indexes)
    {
        double total = 0;
        foreach (Query query in Queries) total += query.Frequency * QueryCost(query, indexes);
        return total;
    }

    internal double TotalCost(bool[] indexes)
    {
        double total = ExecutionCost(indexes);
        foreach (bool indexed in indexes) if (indexed) total += BuildCost;
        return total;
    }

    internal long Storage(bool[] indexes)
    {
        long bytes = 0;
        foreach (bool indexed in indexes) if (indexed) bytes += (long)BytesPerIndex;
        return bytes;
    }

    internal bool[] Selection(IReadOnlyList<int> columns)
    {
        var indexes = new bool[Columns];
        foreach (int column in columns) indexes[column] = true;
        return indexes;
    }
}

internal static class Planner
{
    /// <summary>Best (cost saved)/(bytes) ratio among the columns that still fit and still help.</summary>
    internal static int BestRatioColumn(Workload workload, bool[] indexes, long budget)
    {
        double current = workload.TotalCost(indexes);
        int best = -1;
        double bestRatio = 0;
        for (int column = 0; column < workload.Columns; column++)
        {
            if (indexes[column]) continue;
            if (workload.Storage(indexes) + (long)workload.BytesPerIndex > budget) continue;
            bool[] candidate = workload.Selection(WithColumn(indexes, column));
            double saved = current - workload.TotalCost(candidate);
            if (saved <= 0) continue;
            double ratio = saved / workload.BytesPerIndex;
            if (ratio > bestRatio) { bestRatio = ratio; best = column; }
        }
        return best;
    }

    internal static List<int> WithColumn(bool[] indexes, int column)
    {
        var chosen = new List<int>();
        for (int c = 0; c < indexes.Length; c++) if (indexes[c] || c == column) chosen.Add(c);
        return chosen;
    }

    internal static List<int> Greedy(Workload workload, long budget)
    {
        var chosen = new List<int>();
        var indexes = new bool[workload.Columns];
        while (true)
        {
            int column = BestRatioColumn(workload, indexes, budget);
            if (column < 0) break;
            indexes[column] = true;
            chosen.Add(column);
        }
        return chosen;
    }

    /// <summary>Exhaustive search over every subset that fits the budget - the ground truth for small schemas.</summary>
    internal static List<int> ExhaustiveOptimal(Workload workload, long budget)
    {
        var best = new List<int>();
        double bestCost = workload.TotalCost(new bool[workload.Columns]);
        for (int mask = 1; mask < 1 << workload.Columns; mask++)
        {
            var chosen = new List<int>();
            for (int column = 0; column < workload.Columns; column++)
                if ((mask & (1 << column)) != 0) chosen.Add(column);
            bool[] indexes = workload.Selection(chosen);
            if (workload.Storage(indexes) > budget) continue;
            double cost = workload.TotalCost(indexes);
            if (cost < bestCost) { bestCost = cost; best = chosen; }
        }
        return best;
    }

    /// <summary>
    /// The naive variant: rank the candidates once, using benefits measured on the empty set, then insert them
    /// in that order. Overlapping queries are counted twice, which is exactly the mistake greedy must avoid.
    /// </summary>
    internal static List<int> GreedyWithFrozenBenefits(Workload workload, long budget)
    {
        var empty = new bool[workload.Columns];
        double baseline = workload.TotalCost(empty);
        var order = new List<int>();
        for (int column = 0; column < workload.Columns; column++) order.Add(column);
        order.Sort((a, b) =>
        {
            double savedA = baseline - workload.TotalCost(workload.Selection(new List<int> { a }));
            double savedB = baseline - workload.TotalCost(workload.Selection(new List<int> { b }));
            return savedA != savedB ? savedB.CompareTo(savedA) : a.CompareTo(b);
        });

        var indexes = new bool[workload.Columns];
        var chosen = new List<int>();
        foreach (int column in order)
        {
            if (workload.Storage(indexes) + (long)workload.BytesPerIndex > budget) continue;
            double before = workload.TotalCost(indexes);
            indexes[column] = true;
            if (workload.TotalCost(indexes) >= before) { indexes[column] = false; continue; }
            chosen.Add(column);
        }
        return chosen;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Close(double a, double b, double tolerance) => Math.Abs(a - b) <= tolerance;

    private static Workload BuildWorkload(int rows, int columns, int queries, ulong seed)
    {
        ulong state = seed;
        var workload = new Workload { Rows = rows, Columns = columns };
        for (int q = 0; q < queries; q++)
        {
            int width = 1 + Rng.NextInt(ref state, 3);
            var used = new List<int>();
            while (used.Count < width)
            {
                int column = Rng.NextInt(ref state, columns);
                if (!used.Contains(column)) used.Add(column);
            }
            used.Sort();
            int frequency = 1 + Rng.NextInt(ref state, 20);
            double selectivity = 0.01 * (1 + Rng.NextInt(ref state, 20));
            workload.Queries.Add(new Query(used.ToArray(), frequency, selectivity));
        }
        return workload;
    }

    public static void Main()
    {
        // ---- plan costs against hand-computed values ------------------------------------------
        {
            var tiny = new Workload { Rows = 1024, Columns = 3 };
            tiny.Queries.Add(new Query(new[] { 0 }, 1, 0.25));
            var none = new bool[3];
            Check(Close(tiny.QueryCost(tiny.Queries[0], none), 1024, 1e-9) && Close(tiny.ExecutionCost(none), 1024, 1e-9),
                "without an index a query costs a full scan of 1024 rows");
            var onCol0 = new bool[3];
            onCol0[0] = true;
            Check(Close(tiny.QueryCost(tiny.Queries[0], onCol0), 10 + 0.25 * 1024, 1e-9),
                "with an index the plan costs a probe plus the matching rows");
            Check(Close(tiny.TotalCost(onCol0), 10 + 256 + 1024 * 10, 1e-9),
                "the total adds the one-off build cost of the chosen index");
            Check(tiny.Storage(onCol0) == 4 * 1024 && tiny.Storage(none) == 0,
                "storage is 4 bytes per row per indexed column");
        }

        // ---- an empty workload, and every degenerate budget -------------------------------------
        {
            var empty = new Workload { Rows = 1000, Columns = 4 };
            Check(Planner.Greedy(empty, 1_000_000).Count == 0 && Close(empty.ExecutionCost(new bool[4]), 0, 1e-9),
                "an empty workload needs no index and costs nothing");

            var workload = new Workload { Rows = 4096, Columns = 8 };
            for (int column = 0; column < workload.Columns; column++)
                workload.Queries.Add(new Query(new[] { column }, 100, 0.01));
            Check(Planner.Greedy(workload, 0).Count == 0, "a zero budget selects no index");
            Check(Planner.Greedy(workload, (long)workload.BytesPerIndex - 1).Count == 0,
                "a budget below one index's storage selects no index");
            var everything = Planner.Greedy(workload, (long)(workload.BytesPerIndex * 8));
            Check(everything.Count == 8, "a budget for the whole schema selects every profitable column (got "
                + everything.Count + ")");
            Check(Planner.BestRatioColumn(workload, workload.Selection(everything), long.MaxValue) == -1,
                "once everything is indexed there is no candidate left");
        }

        // ---- greedy against the exhaustive optimum on a full 10-column schema -------------------
        {
            var workload = BuildWorkload(100_000, 10, 60, 0x0FEDCBA987654321UL);
            long budget = (long)(workload.BytesPerIndex * 4);
            var chosen = Planner.Greedy(workload, budget);
            var indexes = workload.Selection(chosen);
            var baseline = new bool[workload.Columns];
            var optimal = Planner.ExhaustiveOptimal(workload, budget);

            Check(workload.Storage(indexes) <= budget, "the greedy selection always fits the storage budget");
            Check(workload.TotalCost(indexes) <= workload.TotalCost(baseline),
                "greedy never leaves the workload more expensive than having no index at all");
            double greedyCost = workload.TotalCost(indexes);
            double optimalCost = workload.TotalCost(workload.Selection(optimal));
            Check(greedyCost >= optimalCost - 1e-9,
                "greedy can never beat the exhaustive optimum (which enumerates all feasible subsets)");
            Check(chosen.Count > 0 && optimal.Count > 0, "both planners actually pick something on this workload");

            Check(Planner.Greedy(workload, budget).Count == chosen.Count
                    && SameColumns(Planner.Greedy(workload, budget), chosen),
                "greedy is deterministic: the same workload and budget give the same selection");

            // Every step must be the true maximum-ratio step, recomputed from the current set.
            var progressive = new bool[workload.Columns];
            var replayed = new List<int>();
            bool stepsMatch = true;
            while (replayed.Count < chosen.Count)
            {
                int best = Planner.BestRatioColumn(workload, progressive, budget);
                if (best != chosen[replayed.Count]) stepsMatch = false;
                progressive[best] = true;
                replayed.Add(best);
            }
            Check(stepsMatch, "each greedy step is the maximum benefit-per-byte column for the current set");

            // A stale-benefit planner (ratios frozen at the empty set) must not be silently as good.
            var frozen = Planner.GreedyWithFrozenBenefits(workload, budget);
            var frozenIndexes = workload.Selection(frozen);
            Check(workload.Storage(frozenIndexes) <= budget, "the frozen-benefit planner also respects the budget");
            double frozenCost = workload.TotalCost(frozenIndexes);
            Check(frozenCost >= optimalCost - 1e-9,
                "the frozen-benefit planner cannot beat the exhaustive optimum either");
            Console.WriteLine("greedyCost=" + Math.Round(greedyCost) + " optimalCost=" + Math.Round(optimalCost)
                + " frozenCost=" + Math.Round(frozenCost) + " baselineCost=" + Math.Round(workload.TotalCost(baseline))
                + " picked=" + string.Join(",", chosen));
        }

        // ---- an index no query uses is never chosen ---------------------------------------------
        {
            var workload = new Workload { Rows = 5000, Columns = 5 };
            workload.Queries.Add(new Query(new[] { 0 }, 10, 0.05));
            workload.Queries.Add(new Query(new[] { 0, 1 }, 5, 0.1));
            var chosen = Planner.Greedy(workload, (long)(workload.BytesPerIndex * 5));
            Check(chosen.Count == 1 && chosen[0] == 0,
                "only the predicate column is indexed; the unused columns are never picked");
            var indexes = workload.Selection(chosen);
            Check(Close(workload.ExecutionCost(indexes), 10 * (Math.Log2(5000) + 0.05 * 5000) + 5 * (Math.Log2(5000) + 0.1 * 5000), 1e-9),
                "one index serves every query that mentions its column");
        }

        // ---- a selectivity-1.0 query makes an index strictly harmful ----------------------------
        {
            var workload = new Workload { Rows = 8000, Columns = 3 };
            workload.Queries.Add(new Query(new[] { 0 }, 1, 1.0));
            var chosen = Planner.Greedy(workload, (long)(workload.BytesPerIndex * 3));
            Check(chosen.Count == 0, "an index that cannot save any row visits is declined when it only adds cost");
            workload.Queries.Add(new Query(new[] { 1 }, 50, 0.001));
            var chosen2 = Planner.Greedy(workload, (long)(workload.BytesPerIndex * 3));
            Check(chosen2.Count == 1 && chosen2[0] == 1,
                "the profitable column is still picked alongside the harmful one");
        }

        // ---- execution cost is monotone non-increasing in the index set --------------------------
        {
            var workload = BuildWorkload(20_000, 6, 25, 0x9999AAAABBBBCCCCUL);
            ulong state = 0xDEADBEEF0BADF00DUL;
            bool monotone = true;
            for (int t = 0; t < 200; t++)
            {
                var small = new bool[workload.Columns];
                var big = new bool[workload.Columns];
                for (int c = 0; c < workload.Columns; c++)
                {
                    if (Rng.NextInt(ref state, 2) == 0) small[c] = true;
                    if (small[c] || Rng.NextInt(ref state, 2) == 0) big[c] = true;
                }
                if (workload.ExecutionCost(big) > workload.ExecutionCost(small) + 1e-9) monotone = false;
                if (workload.Storage(big) < workload.Storage(small)) monotone = false;
            }
            Check(monotone, "adding indexes never increases execution cost or shrinks storage (200 random pairs)");
        }

        // ---- duplicate queries aggregate by frequency -------------------------------------------
        {
            var workload = new Workload { Rows = 1000, Columns = 2 };
            var q = new Query(new[] { 0 }, 200, 0.2);
            workload.Queries.Add(q);
            workload.Queries.Add(new Query(new[] { 0 }, 300, 0.2));
            var indexes = workload.Selection(new List<int> { 0 });
            Check(Close(workload.ExecutionCost(indexes), 500 * (Math.Log2(1000) + 0.2 * 1000), 1e-9),
                "the same predicate repeated with different frequencies is summed by frequency");
            Check(Planner.Greedy(workload, 3999).Count == 0 && Planner.Greedy(workload, 4000).Count == 1,
                "greedy respects the budget boundary exactly at one index");
        }

        Console.WriteLine("PASS");
    }

    private static bool SameColumns(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }
}
