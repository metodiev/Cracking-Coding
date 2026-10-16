// Task 641 - Implement advanced query planner cost model stub and plan enumeration. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A deliberately small but honest planner: a catalogue of table/column statistics, a textbook
// selectivity estimator, a cost model with named constants, and an exhaustive enumerator that
// scores every access path x join method x join order combination and returns the cheapest plan.
//
// What is a stub (documented, not hidden): the histograms are just (row count, distinct count,
// min, max), so selectivity assumes a uniform distribution; conjunctive predicates are multiplied
// (independence) even when they reference correlated columns; the cost constants are fixed rather
// than calibrated per host; there is no memo/interesting-orders machinery and no parallel or bushy
// plans; and the queries are two-table selects with a single equi-join predicate.
//
// Verified: every formula is pinned against hand-computed numbers (e.g. 1000 rows / 100 pages with
// a 50-distinct key gives selectivity 0.02, 20 matching rows and an index scan cost of exactly
// 17.2), the planner's choice equals the brute-force argmin over the full enumerated space, the
// enumeration count is exactly |paths(A)| * |paths(B)| * methods * orders, and the classic
// crossover behaviour holds: a selective predicate prefers the index, a non-selective one the
// sequential scan.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class ColumnStats
{
    public readonly string Name;
    public readonly int Distinct;
    public readonly int Min;
    public readonly int Max;

    public ColumnStats(string name, int distinct, int min, int max)
    {
        Name = name;
        Distinct = distinct;
        Min = min;
        Max = max;
    }
}

internal sealed class TableStats
{
    public readonly string Name;
    public readonly int Rows;
    public readonly int Pages;
    public readonly Dictionary<string, ColumnStats> Columns = new Dictionary<string, ColumnStats>(StringComparer.Ordinal);
    public readonly string IndexedColumn;

    public TableStats(string name, int rows, int pages, string indexedColumn, params ColumnStats[] columns)
    {
        Name = name;
        Rows = rows;
        Pages = pages;
        IndexedColumn = indexedColumn;
        foreach (ColumnStats column in columns) Columns[column.Name] = column;
    }

    public int TuplesPerPage => Math.Max(1, Rows / Math.Max(1, Pages));

    public bool HasIndexOn(string column) => IndexedColumn.Length != 0 && string.Equals(IndexedColumn, column, StringComparison.Ordinal);
}

internal sealed class Predicate
{
    public enum Kind { Equal, Between, And }

    public readonly Kind Type;
    public readonly string Column;
    public readonly int Low;
    public readonly int High;
    public readonly Predicate? Left;
    public readonly Predicate? Right;

    private Predicate(Kind type, string column, int low, int high, Predicate? left, Predicate? right)
    {
        Type = type;
        Column = column;
        Low = low;
        High = high;
        Left = left;
        Right = right;
    }

    public static Predicate Eq(string column, int value) => new Predicate(Kind.Equal, column, value, value, null, null);
    public static Predicate Range(string column, int low, int high) => new Predicate(Kind.Between, column, low, high, null, null);
    public static Predicate And(Predicate left, Predicate right) => new Predicate(Kind.And, "", 0, 0, left, right);

    // Leaves, in the order they were written, so the access path can drive from a single column.
    public IEnumerable<Predicate> Leaves()
    {
        if (Type == Kind.And)
        {
            foreach (Predicate leaf in Left!.Leaves()) yield return leaf;
            foreach (Predicate leaf in Right!.Leaves()) yield return leaf;
        }
        else
        {
            yield return this;
        }
    }

    public override string ToString() => Type switch
    {
        Kind.Equal => Column + "=" + Low,
        Kind.Between => Column + " between " + Low + " and " + High,
        _ => "(" + Left + " and " + Right + ")",
    };
}

internal static class Selectivity
{
    public static double Estimate(Predicate predicate, TableStats table)
    {
        switch (predicate.Type)
        {
            case Predicate.Kind.Equal:
            {
                ColumnStats column = Lookup(predicate.Column, table);
                return 1.0 / column.Distinct;
            }
            case Predicate.Kind.Between:
            {
                ColumnStats column = Lookup(predicate.Column, table);
                double span = column.Max - column.Min + 1.0;
                double width = Math.Min(predicate.High, column.Max) - Math.Max(predicate.Low, column.Min) + 1.0;
                if (width <= 0) return 1.0 / column.Distinct;
                return Math.Min(1.0, width / span);
            }
            default:
                return Estimate(predicate.Left!, table) * Estimate(predicate.Right!, table);
        }
    }

    public static int Rows(Predicate predicate, TableStats table)
        => Math.Max(1, (int)Math.Round(table.Rows * Estimate(predicate, table)));

    private static ColumnStats Lookup(string column, TableStats table)
    {
        if (!table.Columns.TryGetValue(column, out ColumnStats stats))
        {
            throw new Exception("no statistics for column " + column + " of table " + table.Name);
        }
        return stats;
    }
}

internal sealed class CostModel
{
    public const double SeqPageCost = 1.0;
    public const double RandomPageCost = 4.0;
    public const double CpuTupleCost = 0.01;
    public const double IndexProbeCost = 0.05;
    public const double HashBuildCost = 0.02;
    public const double HashProbeCost = 0.01;
    public const double SortCpuCost = 0.03;

    public const int IndexTreePages = 2;

    // Full heap scan: every page read sequentially, every row touched by the filter.
    public double SeqScan(TableStats table) => table.Pages * SeqPageCost + table.Rows * CpuTupleCost;

    // Index scan: tree pages plus one leaf-page fetch per tuples-per-page worth of matches, then a
    // probe and a projection step per matching row.
    public double IndexScan(TableStats table, int matches)
    {
        double pages = IndexTreePages + Math.Ceiling(matches / (double)table.TuplesPerPage);
        return pages * RandomPageCost + matches * (IndexProbeCost + CpuTupleCost);
    }

    // Index-only scan: the same random leaf fetches, but no heap tuple is touched.
    public double IndexOnlyScan(TableStats table, int matches)
    {
        double pages = IndexTreePages + Math.Ceiling(matches / (double)table.TuplesPerPage);
        return pages * RandomPageCost + matches * IndexProbeCost;
    }

    public double NestedLoop(double outerRows, double innerRows) => outerRows * innerRows * CpuTupleCost;

    public double HashJoin(double outerRows, double innerRows, double outputRows)
        => innerRows * HashBuildCost + outerRows * HashProbeCost + outputRows * CpuTupleCost;

    public double SortMerge(double outerRows, double innerRows, double outputRows)
        => Sort(outerRows) + Sort(innerRows) + (outerRows + innerRows) * CpuTupleCost + outputRows * CpuTupleCost;

    public double Sort(double rows) => rows * Math.Log2(Math.Max(2.0, rows)) * SortCpuCost;
}

internal sealed class Query
{
    public readonly TableStats Outer;
    public readonly TableStats Inner;
    public readonly Predicate OuterFilter;
    public readonly Predicate InnerFilter;
    public readonly string OuterJoinColumn;
    public readonly string InnerJoinColumn;

    public Query(TableStats outer, Predicate outerFilter, string outerJoinColumn,
        TableStats inner, Predicate innerFilter, string innerJoinColumn)
    {
        Outer = outer;
        OuterFilter = outerFilter;
        OuterJoinColumn = outerJoinColumn;
        Inner = inner;
        InnerFilter = innerFilter;
        InnerJoinColumn = innerJoinColumn;
    }
}

internal sealed class Plan
{
    public readonly string Description;
    public readonly double Cost;
    public readonly double Rows;

    public Plan(string description, double cost, double rows)
    {
        Description = description;
        Cost = cost;
        Rows = rows;
    }
}

internal sealed class Planner
{
    private readonly CostModel _model = new CostModel();

    public List<Plan> EnumerateAll(Query query)
    {
        var plans = new List<Plan>();
        foreach (Plan outer in AccessPaths(query.Outer, query.OuterFilter))
        {
            foreach (Plan inner in AccessPaths(query.Inner, query.InnerFilter))
            {
                foreach (string method in new[] { "nested-loop", "hash-join", "sort-merge" })
                {
                    plans.Add(Join(query, outer, inner, method));
                    plans.Add(Join(query, inner, outer, method));
                }
            }
        }
        return plans;
    }

    public Plan Best(Query query) => EnumerateAll(query).OrderBy(p => p.Cost).ThenBy(p => p.Description, StringComparer.Ordinal).First();

    public int AccessPathCount(Predicate filter, TableStats table) => AccessPaths(table, filter).Count;

    // Access paths for one relation: a sequential scan plus, for every equality/range leaf on an
    // indexed column, an index scan and an index-only scan.
    public List<Plan> AccessPaths(TableStats table, Predicate filter)
    {
        var paths = new List<Plan>();
        int rows = Selectivity.Rows(filter, table);
        paths.Add(new Plan("seq(" + table.Name + ")", _model.SeqScan(table), rows));

        foreach (Predicate leaf in filter.Leaves())
        {
            if (!table.HasIndexOn(leaf.Column)) continue;
            string label = leaf.ToString();
            paths.Add(new Plan("index(" + table.Name + "." + label + ")", _model.IndexScan(table, rows), rows));
            paths.Add(new Plan("index-only(" + table.Name + "." + label + ")", _model.IndexOnlyScan(table, rows), rows));
        }
        return paths;
    }

    public Plan Join(Query query, Plan left, Plan right, string method)
    {
        double rows = JoinRows(query, left.Rows, right.Rows);
        double joinCost = method switch
        {
            "nested-loop" => _model.NestedLoop(left.Rows, right.Rows),
            "hash-join" => _model.HashJoin(left.Rows, right.Rows, rows),
            _ => _model.SortMerge(left.Rows, right.Rows, rows),
        };
        return new Plan(left.Description + " " + method + " " + right.Description, left.Cost + right.Cost + joinCost, rows);
    }

    // Textbook join cardinality: |A| * |B| / max(ndv(A.k), ndv(B.k)); never below 1 row.
    public double JoinRows(Query query, double leftRows, double rightRows)
    {
        double leftDistinct = Distinct(query.Outer, query.OuterJoinColumn);
        double rightDistinct = Distinct(query.Inner, query.InnerJoinColumn);
        double selectivity = 1.0 / Math.Max(leftDistinct, rightDistinct);
        return Math.Max(1.0, leftRows * rightRows * selectivity);
    }

    private static double Distinct(TableStats table, string column)
    {
        if (!table.Columns.TryGetValue(column, out ColumnStats stats))
        {
            throw new Exception("no statistics for join column " + column);
        }
        return stats.Distinct;
    }
}

public static class Solution
{
    private static string F(double value) => value.ToString("0.####", CultureInfo.InvariantCulture);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void Close(double actual, double expected, string label)
    {
        Check(Math.Abs(actual - expected) < 1e-9, label + ": expected " + F(expected) + " got " + F(actual));
    }

    private static TableStats Orders() => new TableStats("orders", 1000, 100, "cust",
        new ColumnStats("cust", 50, 1, 50),
        new ColumnStats("amount", 500, 1, 500),
        new ColumnStats("day", 100, 0, 99));

    private static TableStats Customers() => new TableStats("customers", 200, 20, "",
        new ColumnStats("id", 200, 1, 200),
        new ColumnStats("region", 4, 1, 4));

    private static Query Sample() => new Query(Orders(), Predicate.Eq("cust", 7), "cust",
        Customers(), Predicate.Range("id", 1, 200), "id");

    private static void SelectivityFormulas()
    {
        TableStats orders = Orders();
        Close(Selectivity.Estimate(Predicate.Eq("cust", 7), orders), 0.02, "equality selectivity 1/ndv");
        Close(Selectivity.Estimate(Predicate.Range("day", 10, 19), orders), 0.1, "range selectivity width/span");
        Close(Selectivity.Estimate(Predicate.Range("day", 0, 200), orders), 1.0, "range clamped to 1");
        Close(Selectivity.Estimate(Predicate.Range("day", 500, 600), orders), 1.0 / 100, "range outside the domain falls back to 1/ndv");

        Predicate both = Predicate.And(Predicate.Eq("cust", 7), Predicate.Range("day", 10, 19));
        Close(Selectivity.Estimate(both, orders), 0.002, "conjunction multiplies selectivities");
        Check(Selectivity.Estimate(both, orders) <= Selectivity.Estimate(Predicate.Eq("cust", 7), orders), "conjunction is no wider than its leaves");

        Check(Selectivity.Rows(Predicate.Eq("cust", 7), orders) == 20, "row estimate 1000 * 0.02 = 20");
        Check(Selectivity.Rows(Predicate.Range("day", 500, 600), orders) == 10, "row estimate never drops below one per distinct value");

        bool threw = false;
        try { Selectivity.Estimate(Predicate.Eq("nope", 1), orders); }
        catch (Exception) { threw = true; }
        Check(threw, "a predicate on a column without statistics is rejected");
        Console.WriteLine("selectivity: eq=0.02 range=0.1 conjunction=0.002");
    }

    private static void HandComputedCosts()
    {
        TableStats orders = Orders();
        var model = new CostModel();
        Close(model.SeqScan(orders), 100 * 1.0 + 1000 * 0.01, "seq scan = pages + rows*cpu");

        int matches = Selectivity.Rows(Predicate.Eq("cust", 7), orders);
        Check(matches == 20, "20 matching rows for cust=7");
        // 2 tree pages + ceil(20 / 10 tuples-per-page) = 4 random pages, then probe + cpu per row.
        Close(model.IndexScan(orders, matches), 4 * 4.0 + 20 * (0.05 + 0.01), "index scan = 17.2");
        Close(model.IndexOnlyScan(orders, matches), 4 * 4.0 + 20 * 0.05, "index-only scan = 17.0");
        Check(model.IndexOnlyScan(orders, matches) < model.IndexScan(orders, matches), "index-only is cheaper than a heap fetch");
        Console.WriteLine("hand-computed cost model: seq=" + F(model.SeqScan(orders))
            + " index=" + F(model.IndexScan(orders, matches)) + " index-only=" + F(model.IndexOnlyScan(orders, matches)));
    }

    private static void AccessPathCrossover()
    {
        TableStats orders = Orders();
        TableStats tiny = new TableStats("tiny", 1000, 100, "cust", new ColumnStats("cust", 1000, 1, 1000));
        var planner = new Planner();

        Plan best = planner.AccessPaths(orders, Predicate.Eq("cust", 7))
            .OrderBy(p => p.Cost).ThenBy(p => p.Description, StringComparer.Ordinal).First();
        Check(best.Description == "index-only(orders.cust=7)", "selective predicate picks the covering index, got " + best.Description);

        Plan all = planner.AccessPaths(tiny, Predicate.Range("cust", 1, 1000))
            .OrderBy(p => p.Cost).ThenBy(p => p.Description, StringComparer.Ordinal).First();
        Check(all.Description == "seq(tiny)", "non-selective predicate picks the sequential scan, got " + all.Description);

        Plan noIndex = planner.AccessPaths(Customers(), Predicate.Eq("region", 2)).OrderBy(p => p.Cost).First();
        Check(noIndex.Description.StartsWith("seq", StringComparison.Ordinal), "a table without a usable index has only scan paths");
    }

    private static void EnumerationAndArgmin()
    {
        Query query = Sample();
        var planner = new Planner();
        List<Plan> plans = planner.EnumerateAll(query);

        int pathsOuter = planner.AccessPathCount(query.OuterFilter, query.Outer);
        int pathsInner = planner.AccessPathCount(query.InnerFilter, query.Inner);
        Check(pathsOuter == 3, "orders has seq + index + index-only = 3 paths, got " + pathsOuter);
        Check(pathsInner == 1, "customers has only the sequential scan, got " + pathsInner);
        Check(plans.Count == pathsOuter * pathsInner * 3 * 2, "enumeration covers every path x method x order combination");
        Check(plans.All(p => p.Cost > 0), "every enumerated plan has a positive cost");

        Plan best = planner.Best(query);
        double brute = plans.Min(p => p.Cost);
        Close(best.Cost, brute, "the planner returns the brute-force minimum cost");
        Check(plans.Count(p => Math.Abs(p.Cost - brute) < 1e-9) >= 1, "at least one plan achieves the minimum");
        Close(best.Rows, 20.0, "join cardinality 20 * 200 / max(ndv 50, ndv 200) = 20");
        bool cheated = plans.Any(p => p.Cost < brute - 1e-9);
        Check(!cheated, "no enumerated plan beats the reported optimum");
        Console.WriteLine("enumeration: " + plans.Count + " plans, best=" + best.Description + " cost=" + F(best.Cost));
    }

    private static void CostMonotonicity()
    {
        var planner = new Planner();
        var model = new CostModel();
        double previous = -1;
        for (int rows = 100; rows <= 2000; rows += 100)
        {
            var table = new TableStats("t" + rows, rows, rows / 10, "",
                new ColumnStats("k", Math.Max(1, rows / 10), 1, rows));
            double cost = model.SeqScan(table);
            Check(cost > previous, "seq scan cost grows with the table (rows=" + rows + ")");
            previous = cost;
            Plan path = planner.AccessPaths(table, Predicate.Eq("k", 1)).OrderBy(p => p.Cost).First();
            Check(path.Rows <= rows, "estimated rows never exceed the table cardinality");
        }
        Check(previous > 0, "monotonicity loop really ran");
    }

    public static void Main()
    {
        SelectivityFormulas();
        HandComputedCosts();
        AccessPathCrossover();
        EnumerationAndArgmin();
        CostMonotonicity();
        Console.WriteLine("PASS");
    }
}
