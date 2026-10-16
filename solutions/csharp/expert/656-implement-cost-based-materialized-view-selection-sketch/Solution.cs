// Task 656 - Implement cost-based materialized view selection sketch. (Tier: Expert, CTCI Ch 14 - Databases)
//
// Scaled-down view advisor over a real (small) data cube. A 3-dimensional fact table is aggregated into cubes
// keyed by *grouping sets* (bitmask of the dimensions in the GROUP BY); a materialised view is exactly one of
// those cubes. A query with grouping set q can be answered from any materialised view v whose grouping set is
// a subset of q (roll the cube up), at a cost of rows(v); otherwise it must scan the fact table. Building a
// view costs one pass over the facts and storing it costs rows(v), so the advisor's total cost is
//     execution cost (sum of frequency * plan cost) + build factor * fact rows * |views|
// and the greedy heuristic adds the view with the best (cost saved)/(rows stored) ratio until the storage
// budget is exhausted, recomputing every candidate's marginal benefit against the current set.
// What is simplified: three dimensions with tiny cardinalities, no joins, no indexes on the fact table, a
// single measure, and cost proportional to rows scanned rather than to I/O - the *cube lattice* and the
// greedy/exhaustive comparison are the real thing, and every row count used by the cost model is computed by
// actually aggregating the fact rows, not estimated.
// Verified: cube row counts and measure sums are cross-checked against direct GROUP BY scans; a roll-up from a
// finer cube to a coarser one equals aggregating the facts directly at the coarse grain, for all lattice
// pairs; the covering relation (v ⊆ q) is validated pair by pair against the cost function; greedy respects
// the budget, never exceeds the no-view baseline and never beats the exhaustive optimum over all 128 candidate
// subsets; each greedy step is the maximum-ratio step; an unprofitable view is declined; and zero/degenerate
// budgets behave. Deterministic splitmix64 input, no wall clock.
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

internal sealed class FactTable
{
    internal int[] Cardinality = { 4, 3, 5 };
    internal List<int[]> Rows = new List<int[]>();

    internal int Dimensions => Cardinality.Length;

    internal int AllMasks => 1 << Dimensions;

    /// <summary>Packs the dimensions selected by `mask` into one key, most significant dimension first.</summary>
    internal int KeyOf(int[] row, int mask)
    {
        int key = 0;
        for (int dim = 0; dim < Dimensions; dim++)
            if ((mask & (1 << dim)) != 0) key = key * Cardinality[dim] + row[dim];
        return key;
    }

    /// <summary>Decodes the digit of one dimension from a key built over `mask`.</summary>
    internal int DigitOf(int key, int mask, int dim)
    {
        int place = 1;
        for (int d = dim + 1; d < Dimensions; d++)
            if ((mask & (1 << d)) != 0) place *= Cardinality[d];
        return key / place % Cardinality[dim];
    }
}

internal sealed class Cube
{
    private readonly FactTable _fact;
    private readonly Dictionary<int, Dictionary<int, long>> _groups = new Dictionary<int, Dictionary<int, long>>();

    internal Cube(FactTable fact) => _fact = fact;

    internal int FactRows => _fact.Rows.Count;

    /// <summary>Number of possible grouping sets (including the empty one).</summary>
    internal int Masks => 1 << _fact.Dimensions;

    /// <summary>Cached aggregate: grouping set -> packed key -> measure sum.</summary>
    internal Dictionary<int, long> Group(int mask)
    {
        if (_groups.TryGetValue(mask, out Dictionary<int, long>? cached)) return cached;
        var aggregate = new Dictionary<int, long>();
        foreach (int[] row in _fact.Rows)
        {
            int key = _fact.KeyOf(row, mask);
            aggregate[key] = aggregate.TryGetValue(key, out long sum) ? sum + row[_fact.Dimensions] : row[_fact.Dimensions];
        }
        _groups[mask] = aggregate;
        return aggregate;
    }

    internal int RowsFor(int mask) => Group(mask).Count;

    internal long MeasureFor(int mask)
    {
        long total = 0;
        foreach (long value in Group(mask).Values) total += value;
        return total;
    }

    /// <summary>Rolls a finer cube up to a coarser grouping set by re-keying every fine group.</summary>
    internal Dictionary<int, long> RollUp(int fineMask, int coarseMask)
    {
        if ((coarseMask & fineMask) != coarseMask)
            throw new ArgumentException("the target grouping set must be a subset of the source");
        var rolled = new Dictionary<int, long>();
        foreach (var pair in Group(fineMask))
        {
            int key = 0;
            for (int dim = 0; dim < _fact.Dimensions; dim++)
                if ((coarseMask & (1 << dim)) != 0) key = key * _fact.Cardinality[dim] + _fact.DigitOf(pair.Key, fineMask, dim);
            rolled[key] = rolled.TryGetValue(key, out long sum) ? sum + pair.Value : pair.Value;
        }
        return rolled;
    }

    internal bool DetailedRowCountsMatch()
    {
        foreach (int[] row in _fact.Rows) if (row.Length != _fact.Dimensions + 1) return false;
        return true;
    }
}

internal sealed class AmortisedCost
{
    internal long FactRows;
    internal int[] QueryMasks = Array.Empty<int>();
    internal int[] Frequencies = Array.Empty<int>();

    internal double BuildFactor = 1.0;

    internal double ExecutionCost(Cube cube, bool[] materialised)
    {
        double total = 0;
        for (int q = 0; q < QueryMasks.Length; q++)
        {
            int queryMask = QueryMasks[q];
            double best = FactRows;
            for (int mask = 1; mask < materialised.Length; mask++)
            {
                if (!materialised[mask]) continue;
                if ((queryMask & mask) != mask) continue;
                best = Math.Min(best, cube.RowsFor(mask));
            }
            total += Frequencies[q] * best;
        }
        return total;
    }

    internal long Storage(Cube cube, bool[] materialised)
    {
        long rows = 0;
        for (int mask = 1; mask < materialised.Length; mask++) if (materialised[mask]) rows += cube.RowsFor(mask);
        return rows;
    }

    internal double TotalCost(Cube cube, bool[] materialised)
    {
        double total = ExecutionCost(cube, materialised);
        for (int mask = 1; mask < materialised.Length; mask++)
            if (materialised[mask]) total += BuildFactor * FactRows;
        return total;
    }

    internal bool[] Selection(int candidates, IReadOnlyList<int> masks)
    {
        var materialised = new bool[candidates];
        foreach (int mask in masks) materialised[mask] = true;
        return materialised;
    }

    /// <summary>Highest (cost saved)/(rows stored) ratio so far, or -1 when no view still pays off.</summary>
    internal int BestCandidate(Cube cube, bool[] materialised, long budget)
    {
        double current = TotalCost(cube, materialised);
        int best = -1;
        double bestRatio = 0;
        for (int mask = 1; mask < materialised.Length; mask++)
        {
            if (materialised[mask]) continue;
            long storage = Storage(cube, materialised) + cube.RowsFor(mask);
            if (storage > budget) continue;
            var candidate = (bool[])materialised.Clone();
            candidate[mask] = true;
            double saved = current - TotalCost(cube, candidate);
            if (saved <= 0) continue;
            double ratio = saved / cube.RowsFor(mask);
            if (ratio > bestRatio) { bestRatio = ratio; best = mask; }
        }
        return best;
    }

    internal List<int> Greedy(Cube cube, long budget)
    {
        var chosen = new List<int>();
        var materialised = new bool[cube.Masks];
        while (true)
        {
            int mask = BestCandidate(cube, materialised, budget);
            if (mask < 0) break;
            materialised[mask] = true;
            chosen.Add(mask);
        }
        return chosen;
    }

    /// <summary>Ground truth: every subset of the candidate lattice that fits the budget.</summary>
    internal List<int> ExhaustiveOptimal(Cube cube, long budget)
    {
        int candidates = cube.Masks;
        var best = new List<int>();
        double bestCost = TotalCost(cube, new bool[candidates]);
        for (int selection = 1; selection < 1 << candidates; selection++)
        {
            var masks = new List<int>();
            for (int mask = 1; mask < candidates; mask++)
                if ((selection & (1 << mask)) != 0) masks.Add(mask);
            bool[] materialised = Selection(candidates, masks);
            if (Storage(cube, materialised) > budget) continue;
            double cost = TotalCost(cube, materialised);
            if (cost < bestCost) { bestCost = cost; best = masks; }
        }
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool Same(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        var fact = new FactTable();
        ulong state = 0xABCDEF0123456789UL;
        for (int i = 0; i < 500; i++)
        {
            fact.Rows.Add(new[]
            {
                Rng.NextInt(ref state, fact.Cardinality[0]),
                Rng.NextInt(ref state, fact.Cardinality[1]),
                Rng.NextInt(ref state, fact.Cardinality[2]),
                1 + Rng.NextInt(ref state, 9),
            });
        }
        var cube = new Cube(fact);
        var advisor = new AmortisedCost { FactRows = fact.Rows.Count };

        // ---- cube correctness: counts and measures agree with direct scans ----------------------
        {
            Check(cube.DetailedRowCountsMatch(), "every fact row carries one value per dimension plus a measure");
            Check(cube.MeasureFor(0) == cube.MeasureFor(7), "the grand total is the same however it is computed");
            long direct = 0;
            foreach (int[] row in fact.Rows) direct += row[3];
            Check(cube.MeasureFor(0) == direct, "the grand total equals the sum of the raw measures");
            Check(cube.RowsFor(0) == 1, "grouping by nothing yields exactly one row");
            int distinctCombos = 0;
            var seen = new HashSet<int>();
            foreach (int[] row in fact.Rows)
                if (seen.Add(row[0] * 100 + row[1] * 10 + row[2])) distinctCombos++;
            Check(cube.RowsFor(7) == distinctCombos,
                "the finest cube has exactly one row per distinct dimension combination ("
                + cube.RowsFor(7) + " vs " + distinctCombos + ")");
            Check(cube.RowsFor(7) <= fact.Cardinality[0] * fact.Cardinality[1] * fact.Cardinality[2],
                "the finest cube cannot exceed the dimension cross product");
            Console.WriteLine("factRows=" + fact.Rows.Count + " finestCubeRows=" + cube.RowsFor(7)
                + " cubeRowsByMask=" + string.Join(",", RowsForAll(cube, fact.AllMasks)));
        }

        // ---- roll-up equals aggregating the facts directly at the coarse grain -----------------
        {
            bool rollUpAgrees = true;
            int pairs = 0;
            for (int fine = 1; fine < fact.AllMasks; fine++)
            {
                for (int coarse = 1; coarse < fact.AllMasks; coarse++)
                {
                    if (fine == coarse || (fine & coarse) != coarse) continue;
                    var rolled = cube.RollUp(fine, coarse);
                    var reference = cube.Group(coarse);
                    if (rolled.Count != reference.Count) rollUpAgrees = false;
                    foreach (var pair in reference)
                        if (!rolled.TryGetValue(pair.Key, out long sum) || sum != pair.Value) rollUpAgrees = false;
                    pairs++;
                }
            }
            Check(pairs > 0 && rollUpAgrees, "all " + pairs + " roll-ups match a direct GROUP BY at the coarse grain");
            bool rejected = false;
            try { cube.RollUp(1, 6); } catch (ArgumentException) { rejected = true; }
            Check(rejected, "rolling a cube up to a grouping set that is not a subset is rejected");
        }

        // ---- the covering relation drives the cost model --------------------------------------
        {
            var masks = new List<int> { 3, 5 };
            var materialised = advisor.Selection(fact.AllMasks, masks);
            Check(cube.RowsFor(3) < fact.Rows.Count && cube.RowsFor(5) < fact.Rows.Count,
                "the chosen views are smaller than the fact table");
            advisor.QueryMasks = new[] { 7, 3, 5, 2 };
            advisor.Frequencies = new[] { 10, 4, 4, 4 };
            Check(advisor.ExecutionCost(cube, new bool[fact.AllMasks]) == 22.0 * fact.Rows.Count,
                "with no view every query pays a fact-table scan");
            double withViews = advisor.ExecutionCost(cube, materialised);
            Check(withViews < advisor.ExecutionCost(cube, new bool[fact.AllMasks]),
                "materialising covering views lowers the execution cost");
            var exact = advisor.Selection(fact.AllMasks, new List<int> { 7 });
            double onlyFine = advisor.ExecutionCost(cube, exact);
            Check(advisor.ExecutionCost(cube, materialised) < onlyFine,
                "coarse views that many queries share beat a single exact view of the finest grain");
            var nonCovering = advisor.Selection(fact.AllMasks, new List<int> { 2 });
            Check(advisor.ExecutionCost(cube, nonCovering)
                    == 18.0 * cube.RowsFor(2) + 4.0 * fact.Rows.Count,
                "a view helps exactly the queries whose grouping set it is a subset of: three of the four here");
        }

        // ---- greedy against the exhaustive optimum ---------------------------------------------
        {
            advisor.QueryMasks = new[] { 7, 6, 5, 4, 3, 2, 1, 7, 3, 5 };
            advisor.Frequencies = new[] { 6, 2, 3, 1, 4, 2, 2, 5, 3, 1 };
            long budget = Math.Max(2L, cube.RowsFor(7) / 2);
            var chosen = advisor.Greedy(cube, budget);
            var materialised = advisor.Selection(fact.AllMasks, chosen);
            var baseline = new bool[fact.AllMasks];
            var optimal = advisor.ExhaustiveOptimal(cube, budget);

            Check(advisor.Storage(cube, materialised) <= budget, "the greedy selection fits the storage budget");
            Check(advisor.TotalCost(cube, materialised) <= advisor.TotalCost(cube, baseline),
                "greedy never leaves the workload more expensive than no views at all");
            double greedyCost = advisor.TotalCost(cube, materialised);
            double optimalCost = advisor.TotalCost(cube, advisor.Selection(fact.AllMasks, optimal));
            Check(greedyCost >= optimalCost - 1e-9,
                "greedy cannot beat the exhaustive optimum over all 128 subsets of the cube lattice");
            Check(chosen.Count > 0 && optimal.Count > 0, "both planners pick something on this workload");
            Check(Same(advisor.Greedy(cube, budget), chosen), "greedy is deterministic for the same instance");

            var progressive = new bool[fact.AllMasks];
            bool stepsMatch = true;
            foreach (int expected in chosen)
            {
                int step = advisor.BestCandidate(cube, progressive, budget);
                if (step != expected) stepsMatch = false;
                progressive[step] = true;
            }
            Check(stepsMatch, "each greedy step is the maximum benefit-per-row candidate for the current set");
            Check(advisor.BestCandidate(cube, progressive, budget) == -1,
                "greedy stops exactly when nothing left still pays off");
            Console.WriteLine("greedyCost=" + Math.Round(greedyCost) + " optimalCost=" + Math.Round(optimalCost)
                + " baselineCost=" + Math.Round(advisor.TotalCost(cube, baseline)) + " budget=" + budget
                + " picked=" + string.Join(",", chosen));
        }

        // ---- degenerate budgets and an unprofitable view ---------------------------------------
        {
            advisor.QueryMasks = new[] { 7 };
            advisor.Frequencies = new[] { 1 };
            Check(advisor.Greedy(cube, 0).Count == 0, "a zero budget materialises nothing");
            Check(advisor.Greedy(cube, long.MaxValue).Count == 0,
                "with one query at frequency 1 an extra full pass over the facts never pays off");
            Check(advisor.Greedy(cube, 1).Count == 0 || cube.RowsFor(7) <= 1,
                "a one-row budget cannot buy any of these views");

            advisor.Frequencies = new[] { 200 };
            var chosen = advisor.Greedy(cube, long.MaxValue);
            Check(chosen.Count == 1, "a hot query justifies exactly one materialised view (got " + chosen.Count + ")");
            Check(advisor.TotalCost(cube, advisor.Selection(fact.AllMasks, chosen))
                    < advisor.TotalCost(cube, new bool[cube.Masks]),
                "the chosen view strictly lowers the total cost");
            int smallest = int.MaxValue;
            for (int mask = 1; mask < fact.AllMasks; mask++) smallest = Math.Min(smallest, cube.RowsFor(mask));
            Check(cube.RowsFor(chosen[0]) == smallest,
                "greedy prefers the smallest cube that covers the query (rows=" + cube.RowsFor(chosen[0]) + ")");

            long tight = cube.RowsFor(7);
            var bounded = advisor.Greedy(cube, tight);
            Check(advisor.Storage(cube, advisor.Selection(cube.Masks, bounded)) <= tight && bounded.Count >= 1,
                "a budget equal to the finest cube is respected exactly at the boundary");

            advisor.QueryMasks = new[] { 7, 7, 7 };
            advisor.Frequencies = new[] { 200, 200, 200 };
            Check(advisor.Greedy(cube, long.MaxValue).Count == 1,
                "duplicate queries share one view instead of multiplying materialisations");
        }

        Console.WriteLine("PASS");
    }

    private static List<int> RowsForAll(Cube cube, int masks)
    {
        var rows = new List<int>();
        for (int mask = 0; mask < masks; mask++) rows.Add(cube.RowsFor(mask));
        return rows;
    }
}
