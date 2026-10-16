// Task 655 - Implement query rewriter for simple view unfolding. (Tier: Expert, CTCI Ch 14 - Databases)
//
// A miniature relational rewriter: expressions (column references, constants, + - *), comparison predicates
// and single-table queries with a conjunctive WHERE and an expression projection. A *view* is a query over
// one base table that names its output columns; unfolding replaces every reference to a view column in the
// user's query with the view's defining expression, merges the view's own predicates into the query's WHERE
// (so nothing can be filtered away by accident) and retargets the query at the base table. The rewrite also
// constant-folds, and it validates that the resulting plan only mentions base-table columns - a dangling
// reference is an error, never a silently wrong plan.
// Correctness is established semantically, not syntactically: the file contains a tiny evaluator that can run
// a query either against the view (materialise the view, then evaluate the outer query over those rows) or
// against the unfolded plan, and the two result multisets - rendered as strings and sorted - must be equal.
// That comparison is the cross-check for a generated workload of 60 random queries. Verified as well:
// rewriting a projection that computes a+b becomes a base predicate a+b > threshold; a view predicate and a
// query predicate end up conjunctive; constant folding changes no result; the rewritten plan's column set is
// a subset of the base schema; a query with no WHERE keeps the view's predicates; empty results on
// unsatisfiable predicates; and a non-simple view (aggregation or DISTINCT) is rejected instead of being
// unfolded incorrectly. Deterministic: all data comes from a hand-written splitmix64 stream.
using System;
using System.Collections.Generic;
using System.Text;

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

/// <summary>Scalar expression: a column reference, a constant, or a binary + - *.</summary>
internal sealed class Expr
{
    internal string? Column;
    internal int Constant;
    internal char Op;
    internal Expr? Left;
    internal Expr? Right;

    internal static Expr Col(string name) => new Expr { Column = name };

    internal static Expr Const(int value) => new Expr { Constant = value, Column = null, Left = null };

    internal static Expr Binary(char op, Expr left, Expr right) => new Expr { Op = op, Left = left, Right = right };

    internal int Eval(Func<string, int> lookup) => Column is not null
        ? lookup(Column)
        : Left is null ? Constant
        : Op switch
        {
            '+' => Left.Eval(lookup) + Right!.Eval(lookup),
            '-' => Left.Eval(lookup) - Right!.Eval(lookup),
            _ => Left.Eval(lookup) * Right!.Eval(lookup),
        };

    internal void CollectColumns(ISet<string> into)
    {
        if (Column is not null) { into.Add(Column); return; }
        Left?.CollectColumns(into);
        Right?.CollectColumns(into);
    }

    internal Expr Substitute(IReadOnlyDictionary<string, Expr> replacements)
    {
        if (Column is not null) return replacements.TryGetValue(Column, out Expr? replacement) ? replacement : this;
        if (Left is null) return this;
        return Binary(Op, Left.Substitute(replacements), Right!.Substitute(replacements));
    }

    /// <summary>Folds constant arithmetic; a folded node is a constant, never a column.</summary>
    internal Expr Fold()
    {
        if (Left is null || Right is null) return this;
        Expr left = Left.Fold();
        Expr right = Right.Fold();
        if (left.Left is null && left.Column is null && right.Left is null && right.Column is null)
            return Const(Binary(Op, left, right).Eval(_ => 0));
        return Binary(Op, left, right);
    }

    public override string ToString() => Column ?? (Left is null
        ? Constant.ToString()
        : "(" + Left + " " + Op + " " + Right + ")");
}

internal sealed class Predicate
{
    internal Expr Left = Expr.Const(0);
    internal Expr Right = Expr.Const(0);
    internal string Op = "=";

    internal static Predicate Of(Expr left, string op, Expr right) => new Predicate { Left = left, Op = op, Right = right };

    internal bool Test(Func<string, int> lookup)
    {
        int a = Left.Eval(lookup);
        int b = Right.Eval(lookup);
        return Op switch
        {
            "=" => a == b,
            "<>" => a != b,
            "<" => a < b,
            "<=" => a <= b,
            ">" => a > b,
            _ => a >= b,
        };
    }

    internal void CollectColumns(ISet<string> into)
    {
        Left.CollectColumns(into);
        Right.CollectColumns(into);
    }

    internal Predicate Substitute(IReadOnlyDictionary<string, Expr> replacements) =>
        Of(Left.Substitute(replacements), Op, Right.Substitute(replacements)).Fold();

    internal Predicate Fold() => Of(Left.Fold(), Op, Right.Fold());

    public override string ToString() => Left + " " + Op + " " + Right;
}

internal sealed class Query
{
    internal string Source = string.Empty;
    internal List<string> Aliases = new List<string>();
    internal List<Expr> Projection = new List<Expr>();
    internal List<Predicate> Where = new List<Predicate>();
}

/// <summary>A saved query over one base table; `Simple` is false when it aggregates or deduplicates.</summary>
internal sealed class View
{
    internal string Name = string.Empty;
    internal string BaseTable = string.Empty;
    internal bool Simple = true;
    internal List<string> Columns = new List<string>();
    internal List<Expr> Definitions = new List<Expr>();
    internal List<Predicate> Where = new List<Predicate>();

    internal Query AsQuery()
    {
        var query = new Query { Source = BaseTable };
        query.Aliases.AddRange(Columns);
        query.Projection.AddRange(Definitions);
        query.Where.AddRange(Where);
        return query;
    }
}

internal sealed class Dataset
{
    internal List<string> Columns = new List<string>();
    internal List<int[]> Rows = new List<int[]>();

    internal int IndexOf(string column)
    {
        int index = Columns.IndexOf(column);
        if (index < 0) throw new ArgumentException("unknown column '" + column + "'");
        return index;
    }

    internal int Value(string column, int[] row) => row[IndexOf(column)];
}

internal static class Rewriter
{
    /// <summary>Unfolds a simple view into the user's query, retargeted at the base table.</summary>
    internal static Query Unfold(View view, Query query)
    {
        if (!view.Simple)
            throw new InvalidOperationException("view '" + view.Name + "' aggregates or deduplicates; it is not simple");
        if (view.Columns.Count != view.Definitions.Count)
            throw new InvalidOperationException("view '" + view.Name + "' is malformed");

        var replacements = new Dictionary<string, Expr>();
        for (int i = 0; i < view.Columns.Count; i++) replacements[view.Columns[i]] = view.Definitions[i];

        var unfolded = new Query { Source = view.BaseTable };
        unfolded.Aliases.AddRange(query.Aliases);
        foreach (Expr expression in query.Projection) unfolded.Projection.Add(expression.Substitute(replacements).Fold());
        foreach (Predicate predicate in view.Where) unfolded.Where.Add(predicate.Fold());
        foreach (Predicate predicate in query.Where) unfolded.Where.Add(predicate.Substitute(replacements).Fold());
        return unfolded;
    }

    /// <summary>Columns the plan mentions; the caller checks these exist in the base schema.</summary>
    internal static SortedSet<string> ReferencedColumns(Query query)
    {
        var columns = new SortedSet<string>();
        foreach (Expr expression in query.Projection) expression.CollectColumns(columns);
        foreach (Predicate predicate in query.Where) predicate.CollectColumns(columns);
        return columns;
    }

    /// <summary>Throws when the plan mentions a column the source table does not have.</summary>
    internal static void Validate(Query query, IReadOnlyList<string> sourceColumns)
    {
        var available = new HashSet<string>(sourceColumns);
        foreach (string column in ReferencedColumns(query))
            if (!available.Contains(column))
                throw new ArgumentException("plan references missing column '" + column + "'");
    }

    internal static string Describe(Query query) =>
        "select " + string.Join(", ", query.Projection) + " from " + query.Source
        + (query.Where.Count == 0 ? "" : " where " + string.Join(" and ", query.Where));
}

internal static class Evaluator
{
    internal static List<string> Run(Query query, IReadOnlyDictionary<string, Dataset> tables,
                                    IReadOnlyDictionary<string, View> views)
    {
        Dataset source;
        if (tables.TryGetValue(query.Source, out Dataset? table)) source = table;
        else source = Materialize(views[query.Source], tables);

        var results = new List<string>();
        foreach (int[] row in source.Rows)
        {
            bool keep = true;
            foreach (Predicate predicate in query.Where)
                if (!predicate.Test(column => source.Value(column, row))) { keep = false; break; }
            if (!keep) continue;
            var parts = new List<string>();
            for (int i = 0; i < query.Projection.Count; i++)
                parts.Add(query.Aliases[i] + "=" + query.Projection[i].Eval(column => source.Value(column, row)));
            results.Add(string.Join(",", parts));
        }
        results.Sort(StringComparer.Ordinal);
        return results;
    }

    /// <summary>Runs the view definition over its base table, producing the view's materialised rows.</summary>
    internal static Dataset Materialize(View view, IReadOnlyDictionary<string, Dataset> tables)
    {
        Dataset baseTable = tables[view.BaseTable];
        var materialised = new Dataset();
        materialised.Columns.AddRange(view.Columns);
        foreach (int[] row in baseTable.Rows)
        {
            bool keep = true;
            foreach (Predicate predicate in view.Where)
                if (!predicate.Test(column => baseTable.Value(column, row))) { keep = false; break; }
            if (!keep) continue;
            var projected = new int[view.Definitions.Count];
            for (int i = 0; i < projected.Length; i++)
                projected[i] = view.Definitions[i].Eval(column => baseTable.Value(column, row));
            materialised.Rows.Add(projected);
        }
        return materialised;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    private static bool SameResults(List<string> a, List<string> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static Query Q(string source, IEnumerable<string> aliases, IEnumerable<Expr> projection,
                           IEnumerable<Predicate> where)
    {
        var query = new Query { Source = source };
        query.Aliases.AddRange(aliases);
        query.Projection.AddRange(projection);
        query.Where.AddRange(where);
        return query;
    }

    public static void Main()
    {
        var sales = new Dataset();
        sales.Columns.AddRange(new[] { "a", "b", "c" });
        int[][] sampleRows =
        {
            new[] { 3, 4, 1 }, new[] { 10, 1, 2 }, new[] { 0, 0, 0 }, new[] { 7, 7, 7 }, new[] { 2, 9, 4 },
        };
        sales.Rows.AddRange(sampleRows);

        var tallies = new View
        {
            Name = "tallies",
            BaseTable = "sales",
            Columns = { "x", "y", "total", "scaled" },
            Definitions =
            {
                Expr.Col("a"), Expr.Col("b"), Expr.Binary('+', Expr.Col("a"), Expr.Col("b")),
                Expr.Binary('*', Expr.Col("b"), Expr.Col("c")),
            },
            Where = { Predicate.Of(Expr.Col("c"), "<>", Expr.Const(0)) },
        };
        var tables = new Dictionary<string, Dataset> { { "sales", sales } };
        var views = new Dictionary<string, View> { { "tallies", tallies } };

        // ---- the projection is replaced by the view's defining expression ----------------------
        {
            var query = Q("tallies", new[] { "total" }, new[] { Expr.Col("total") },
                new[] { Predicate.Of(Expr.Col("total"), ">", Expr.Const(5)) });
            Query unfolded = Rewriter.Unfold(tallies, query);
            Check(unfolded.Source == "sales", "the unfolded plan reads the base table, not the view");
            Check(Rewriter.Describe(unfolded) == "select (a + b) from sales where c <> 0 and (a + b) > 5",
                "the computed column becomes the base expression and the view predicate is merged: "
                + Rewriter.Describe(unfolded));
            Check(unfolded.Where.Count == 2, "the view predicate and the query predicate are conjunctive");
            Check(SameResults(Evaluator.Run(query, tables, views), Evaluator.Run(unfolded, tables, views)),
                "the unfolded plan returns the same rows as the view-based query");
            Check(Evaluator.Run(unfolded, tables, views).Count == 4,
                "the result set is the four surviving rows (the c = 0 row is filtered by the view)");
        }

        // ---- a base column referenced directly, and a query with no WHERE -----------------------
        {
            var query = Q("tallies", new[] { "x", "scaled" }, new[] { Expr.Col("x"), Expr.Col("scaled") },
                new List<Predicate>());
            Query unfolded = Rewriter.Unfold(tallies, query);
            Check(Rewriter.Describe(unfolded) == "select a, (b * c) from sales where c <> 0",
                "a passthrough column stays a column and the view's predicate is carried over");
            Check(SameResults(Evaluator.Run(query, tables, views), Evaluator.Run(unfolded, tables, views)),
                "a query without WHERE still gets the view's filter");
            Check(Evaluator.Run(unfolded, tables, views).Count == 4, "the c = 0 row is excluded by the view");
        }

        // ---- constant folding and the column-set invariant -------------------------------------
        {
            var folded = Expr.Binary('+', Expr.Const(2), Expr.Binary('*', Expr.Const(3), Expr.Const(4))).Fold();
            Check(folded.ToString() == "14", "constant subexpressions fold to a literal: " + folded);
            var mixed = Expr.Binary('+', Expr.Col("a"), Expr.Binary('*', Expr.Const(2), Expr.Const(5))).Fold();
            Check(mixed.ToString() == "(a + 10)", "folding leaves the column part untouched: " + mixed);

            var query = Q("tallies", new[] { "z" },
                new[] { Expr.Binary('+', Expr.Col("x"), Expr.Binary('*', Expr.Const(2), Expr.Const(5))) },
                new[] { Predicate.Of(Expr.Binary('-', Expr.Col("y"), Expr.Const(3)), ">=", Expr.Const(0)) });
            Query unfolded = Rewriter.Unfold(tallies, query);
            Check(Rewriter.Describe(unfolded).Contains("(a + 10)") && Rewriter.Describe(unfolded).Contains("(b - 3) >= 0"),
                "the rewrite folds constants in both the projection and the predicate");
            Check(SameResults(Evaluator.Run(query, tables, views), Evaluator.Run(unfolded, tables, views)),
                "folding does not change the result");
            var columns = Rewriter.ReferencedColumns(unfolded);
            Check(columns.Count == 3 && columns.Contains("a") && columns.Contains("b") && columns.Contains("c"),
                "the unfolded plan mentions exactly the base columns a, b, c: " + string.Join(",", columns));
            Rewriter.Validate(unfolded, sales.Columns);
        }

        // ---- a plan that quotes a column the base table does not have is rejected ---------------
        {
            var bogus = Q("tallies", new[] { "q" }, new[] { Expr.Col("q") }, new List<Predicate>());
            Query bogusPlan = Rewriter.Unfold(tallies, bogus);
            Check(Rewriter.ReferencedColumns(bogusPlan).Contains("q"),
                "an unknown view column simply does not survive the substitution");
            bool rejected = false;
            try { Rewriter.Validate(bogusPlan, sales.Columns); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "an unfolded plan referencing a column that is not in the schema is rejected");
            var knownOnly = Q("tallies", new[] { "x" }, new[] { Expr.Col("x") }, new List<Predicate>());
            Query knownPlan = Rewriter.Unfold(tallies, knownOnly);
            bool accepted = true;
            try { Rewriter.Validate(knownPlan, sales.Columns); }
            catch (ArgumentException) { accepted = false; }
            Check(accepted && Rewriter.ReferencedColumns(knownPlan).Count == 2
                    && Rewriter.ReferencedColumns(knownPlan).Contains("a")
                    && Rewriter.ReferencedColumns(knownPlan).Contains("c"),
                "a well-formed plan passes validation and mentions only the base columns it needs");
        }

        // ---- unsatisfiable predicates and non-simple views -------------------------------------
        {
            var query = Q("tallies", new[] { "x" }, new[] { Expr.Col("x") },
                new[] { Predicate.Of(Expr.Col("x"), ">", Expr.Const(100)), Predicate.Of(Expr.Col("x"), "<", Expr.Const(5)) });
            Query unfolded = Rewriter.Unfold(tallies, query);
            Check(Evaluator.Run(unfolded, tables, views).Count == 0
                    && SameResults(Evaluator.Run(query, tables, views), Evaluator.Run(unfolded, tables, views)),
                "a contradictory WHERE returns nothing through either path");

            var aggregated = new View
            {
                Name = "agg",
                BaseTable = "sales",
                Simple = false,
                Columns = { "sum_a" },
                Definitions = { Expr.Col("a") },
            };
            bool rejected = false;
            try { Rewriter.Unfold(aggregated, query); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected, "an aggregating view is rejected instead of being unfolded incorrectly");
            var malformed = new View { Name = "bad", BaseTable = "sales", Columns = { "p", "q" }, Definitions = { Expr.Col("a") } };
            rejected = false;
            try { Rewriter.Unfold(malformed, query); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected, "a view whose column count does not match its definitions is rejected");
        }

        // ---- 60 generated queries: unfolded plan must equal the view-based evaluation ------------
        {
            ulong state = 0x7777CCCC1111DDDDUL;
            var big = new Dataset();
            big.Columns.AddRange(new[] { "a", "b", "c" });
            for (int i = 0; i < 60; i++)
                big.Rows.Add(new[] { Rng.NextInt(ref state, 12), Rng.NextInt(ref state, 8), Rng.NextInt(ref state, 5) });
            var bigTables = new Dictionary<string, Dataset> { { "sales", big } };

            bool equivalent = true, valid = true, predicatesMerged = true;
            string? failure = null;
            int rowsSeen = 0;
            for (int t = 0; t < 60; t++)
            {
                var view = new View { Name = "v" + t, BaseTable = "sales" };
                view.Columns.AddRange(new[] { "v0", "v1", "v2" });
                view.Definitions.Add(Expr.Col("a"));
                view.Definitions.Add(Expr.Binary('+', Expr.Col("a"), Expr.Col("c")));
                view.Definitions.Add(Expr.Binary('*', Expr.Col("b"), Expr.Const(2)));
                if (Rng.NextInt(ref state, 2) == 0)
                    view.Where.Add(Predicate.Of(Expr.Col("b"), "<", Expr.Const(1 + Rng.NextInt(ref state, 7))));
                var localViews = new Dictionary<string, View> { { view.Name, view } };

                var aliases = new List<string>();
                var projection = new List<Expr>();
                int width = 1 + Rng.NextInt(ref state, 3);
                for (int i = 0; i < width; i++)
                {
                    int which = Rng.NextInt(ref state, 3);
                    aliases.Add("o" + i);
                    projection.Add(which == 0 ? Expr.Col("v0")
                        : which == 1 ? Expr.Col("v1") : Expr.Binary('+', Expr.Col("v2"), Expr.Const(i)));
                }
                var where = new List<Predicate>();
                int predicates = Rng.NextInt(ref state, 3);
                for (int i = 0; i < predicates; i++)
                {
                    string column = Rng.NextInt(ref state, 3) switch { 0 => "v0", 1 => "v1", _ => "v2" };
                    where.Add(Predicate.Of(Expr.Col(column), Rng.NextInt(ref state, 2) == 0 ? ">=" : "<=",
                        Expr.Const(Rng.NextInt(ref state, 20))));
                }

                var query = Q(view.Name, aliases, projection, where);
                Query unfolded = Rewriter.Unfold(view, query);
                var expected = Evaluator.Run(query, bigTables, localViews);
                var actual = Evaluator.Run(unfolded, bigTables, localViews);
                rowsSeen += actual.Count;
                if (!SameResults(expected, actual)) { equivalent = false; failure ??= "case " + t; }
                if (unfolded.Source != "sales" || unfolded.Where.Count != view.Where.Count + where.Count)
                    predicatesMerged = false;
                try { Rewriter.Validate(unfolded, big.Columns); }
                catch (ArgumentException) { valid = false; failure ??= "invalid plan " + t; }
            }
            Check(equivalent, "all 60 random queries give identical results before and after unfolding: "
                + (failure ?? "ok"));
            Check(valid, "every unfolded plan only references base-table columns");
            Check(predicatesMerged, "every unfolded plan carries exactly the union of view and query predicates");
            Check(rowsSeen > 0, "the generated workload actually produced rows");
            Console.WriteLine("generatedQueries=60 rowsReturned=" + rowsSeen + " baseRows=" + big.Rows.Count);
        }

        Console.WriteLine("PASS");
    }
}
