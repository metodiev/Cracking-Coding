// Task 942 - Implement hot-path optimization analyzer that suggests inline and caching opportunities.
// (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
//
// An offline analyzer over a dynamic call graph: nodes are functions (bytecode size, per-call body
// cost, purity, recursion, observed distinct argument tuples) and edges are call sites with their
// dynamic counts, so the dynamic call count of a callee is the sum over the edges that point at it
// and the static call-site count is the number of such edges. Two passes propose transformations:
//   * inline - a non-recursive callee that is small, reached from exactly one call site and hot
//     enough that the saved call overhead outweighs the code growth it costs;
//   * memoize - a pure callee that is hot, repeatedly called with the same few argument tuples and
//     expensive enough that a cache lookup is cheaper than recomputing the body.
// Every proposal carries a predicted saving and the list is ranked by it.
//
// The predictions are checked against a simulator rather than trusted: the analyzer's total saving
// must equal the cost difference between simulating the program as-is and simulating it with all
// suggested transformations applied (inline removes the call overhead, memoize replaces repeated
// body executions by lookups). The proposal sets are also cross-checked with a brute-force filter,
// the negative cases (recursive or multi-site for inline, impure or high-cardinality or
// cheaper-than-a-lookup for memoize, and the code-growth gate) are asserted to be rejected, no
// function is proposed for both transformations, and the ranked list is verified to be sorted with
// a total that is the sum of its parts. Empty graphs and all-recursive graphs are boundary cases.
// Simplified: the graph is a static snapshot with integer cost units, no real bytecode, no
// deoptimization or inlining depth limits and no feedback loop - the criteria, the benefit model
// and its simulation-based validation are the real content.
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class HotFunction
{
    internal readonly string Name;
    internal readonly long Size;
    internal readonly long BodyCost;
    internal readonly bool Pure;
    internal readonly bool Recursive;
    internal readonly long DistinctArguments;

    internal HotFunction(string name, long size, long bodyCost, bool pure, bool recursive, long distinctArguments)
    {
        Name = name;
        Size = size;
        BodyCost = bodyCost;
        Pure = pure;
        Recursive = recursive;
        DistinctArguments = distinctArguments;
    }
}

internal sealed class CallEdge
{
    internal readonly string Caller;
    internal readonly string Callee;
    internal readonly long DynamicCount;

    internal CallEdge(string caller, string callee, long dynamicCount)
    {
        Caller = caller;
        Callee = callee;
        DynamicCount = dynamicCount;
    }
}

internal sealed class CallGraph
{
    internal readonly List<HotFunction> Functions = new List<HotFunction>();
    internal readonly List<CallEdge> Edges = new List<CallEdge>();

    internal HotFunction? Find(string name) => Functions.FirstOrDefault(f => f.Name == name);
    internal long CallSites(string callee) => Edges.Count(e => e.Callee == callee);
    internal long CallCount(string callee) => Edges.Where(e => e.Callee == callee).Sum(e => e.DynamicCount);
}

internal enum SuggestionKind { Inline, Memoize }

internal sealed class Suggestion
{
    internal readonly SuggestionKind Kind;
    internal readonly string Function;
    internal readonly long EstimatedSaving;
    internal readonly string Reason;

    internal Suggestion(SuggestionKind kind, string function, long estimatedSaving, string reason)
    {
        Kind = kind;
        Function = function;
        EstimatedSaving = estimatedSaving;
        Reason = reason;
    }
}

internal sealed class HotPathAnalyzer
{
    internal readonly long SizeLimit;
    internal readonly long HotThreshold;
    internal readonly long OverheadPerCall;
    internal readonly long GrowthWeight;
    internal readonly long CacheLookupCost;
    internal readonly long ReuseRatio;

    internal readonly List<Suggestion> Suggestions = new List<Suggestion>();
    internal long TotalSaving;

    internal HotPathAnalyzer(long sizeLimit, long hotThreshold, long overheadPerCall, long growthWeight,
        long cacheLookupCost, long reuseRatio)
    {
        SizeLimit = sizeLimit;
        HotThreshold = hotThreshold;
        OverheadPerCall = overheadPerCall;
        GrowthWeight = growthWeight;
        CacheLookupCost = cacheLookupCost;
        ReuseRatio = reuseRatio;
    }

    internal void Analyze(CallGraph graph)
    {
        foreach (HotFunction function in graph.Functions)
        {
            long calls = graph.CallCount(function.Name);
            long sites = graph.CallSites(function.Name);
            if (!function.Recursive && sites == 1 && function.Size <= SizeLimit)
            {
                long saving = calls * OverheadPerCall;
                if (calls >= HotThreshold && saving > function.Size * GrowthWeight)
                {
                    Suggestions.Add(new Suggestion(SuggestionKind.Inline, function.Name, saving,
                        "single call site, " + function.Size + " B, " + calls + " dynamic calls"));
                    continue;  // one transformation per function: inlining already removes the call
                }
            }
            if (function.Pure && calls >= HotThreshold && function.DistinctArguments * ReuseRatio <= calls)
            {
                // A memoized call still pays for the lookup, so the body is skipped only on the
                // (calls - distinct) repeated invocations while the lookup is paid on every call.
                long saving = (calls - function.DistinctArguments) * function.BodyCost - calls * CacheLookupCost;
                if (saving > 0)
                {
                    Suggestions.Add(new Suggestion(SuggestionKind.Memoize, function.Name, saving,
                        function.DistinctArguments + " distinct arguments over " + calls + " calls"));
                }
            }
        }
        Suggestions.Sort((a, b) =>
        {
            int bySaving = b.EstimatedSaving.CompareTo(a.EstimatedSaving);
            return bySaving != 0 ? bySaving : string.CompareOrdinal(a.Function, b.Function);
        });
        TotalSaving = Suggestions.Sum(s => s.EstimatedSaving);
    }

    internal HashSet<string> InlinedNames() =>
        new HashSet<string>(Suggestions.Where(s => s.Kind == SuggestionKind.Inline).Select(s => s.Function));
    internal HashSet<string> MemoizedNames() =>
        new HashSet<string>(Suggestions.Where(s => s.Kind == SuggestionKind.Memoize).Select(s => s.Function));

    /** Independent oracle: total cost of running the graph, optionally with transformations applied. */
    internal long Simulate(CallGraph graph, HashSet<string> inlined, HashSet<string> memoized)
    {
        long total = 0;
        foreach (HotFunction function in graph.Functions)
        {
            long calls = graph.CallCount(function.Name);
            long executions = memoized.Contains(function.Name) ? function.DistinctArguments : calls;
            total += executions * function.BodyCost;
            total += calls * OverheadPerCall;                       // the call itself still happens
            if (memoized.Contains(function.Name)) total += calls * CacheLookupCost;
            if (inlined.Contains(function.Name)) total -= calls * OverheadPerCall;  // ...unless inlined
        }
        return total;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static CallGraph ShopGraph()
    {
        var graph = new CallGraph();
        graph.Functions.AddRange(new[]
        {
            new HotFunction("main", 300, 50, false, false, 1),
            new HotFunction("handleRequest", 200, 40, false, false, 1),
            new HotFunction("parseInt", 30, 12, true, false, 40000),
            new HotFunction("hashKey", 20, 8, false, false, 100),
            new HotFunction("lookupConfig", 35, 25, true, false, 20000),
            new HotFunction("formatLog", 200, 30, true, false, 50),
            new HotFunction("renderTemplate", 5000, 200, true, false, 1000),
            new HotFunction("fib", 10, 40, true, true, 900),
            new HotFunction("fatInline", 3000, 300, false, false, 1),
            new HotFunction("tinyPure", 15, 2, true, false, 100),
        });
        graph.Edges.AddRange(new[]
        {
            new CallEdge("main", "handleRequest", 1000),
            new CallEdge("handleRequest", "parseInt", 50000),
            new CallEdge("handleRequest", "hashKey", 60000),
            new CallEdge("main", "hashKey", 20000),
            new CallEdge("handleRequest", "lookupConfig", 30000),
            new CallEdge("handleRequest", "formatLog", 60000),
            new CallEdge("handleRequest", "renderTemplate", 10000),
            new CallEdge("handleRequest", "fib", 20000),
            new CallEdge("main", "fatInline", 4000),
            new CallEdge("main", "tinyPure", 8000),
        });
        return graph;
    }

    private static HotPathAnalyzer Analyzer() => new HotPathAnalyzer(40, 500, 5, 10, 3, 4);

    private static HashSet<string> BruteForceInline(CallGraph graph, HotPathAnalyzer a) =>
        new HashSet<string>(graph.Functions.Where(f => !f.Recursive && graph.CallSites(f.Name) == 1
            && f.Size <= a.SizeLimit && graph.CallCount(f.Name) >= a.HotThreshold
            && graph.CallCount(f.Name) * a.OverheadPerCall > f.Size * a.GrowthWeight).Select(f => f.Name));

    private static HashSet<string> BruteForceMemoize(CallGraph graph, HotPathAnalyzer a, HashSet<string> inlined) =>
        new HashSet<string>(graph.Functions.Where(f =>
        {
            long calls = graph.CallCount(f.Name);
            long saving = (calls - f.DistinctArguments) * f.BodyCost - calls * a.CacheLookupCost;
            return f.Pure && calls >= a.HotThreshold && f.DistinctArguments * a.ReuseRatio <= calls
                   && saving > 0 && !inlined.Contains(f.Name);
        }).Select(f => f.Name));

    public static void Main()
    {
        CallGraph graph = ShopGraph();
        HotPathAnalyzer analyzer = Analyzer();
        analyzer.Analyze(graph);
        HashSet<string> inlined = analyzer.InlinedNames();
        HashSet<string> memoized = analyzer.MemoizedNames();

        // 1. The proposal sets match a brute-force filter over the call graph.
        Check(inlined.SetEquals(BruteForceInline(graph, analyzer)) && inlined.Count > 0,
            "inline proposals match the brute-force filter: " + string.Join(",", inlined.OrderBy(n => n)));
        Check(memoized.SetEquals(BruteForceMemoize(graph, analyzer, inlined)),
            "memoize proposals match the brute-force filter: " + string.Join(",", memoized.OrderBy(n => n)));

        // 2-3. Negative cases: recursion, multiple call sites, purity and the economic gates.
        Check(!inlined.Contains("fib") && !inlined.Contains("hashKey") && !inlined.Contains("fatInline"),
            "recursive, multi-site and code-growth-dominated callees are never inlined");
        Check(!memoized.Contains("hashKey") && !memoized.Contains("parseInt") && !memoized.Contains("tinyPure"),
            "impure, high-cardinality and lookup-dominated callees are never memoized");
        Check(graph.CallCount("tinyPure") * analyzer.CacheLookupCost
              >= (graph.CallCount("tinyPure") - graph.Find("tinyPure")!.DistinctArguments) * graph.Find("tinyPure")!.BodyCost,
            "the rejected cheap callee fails exactly the reuse-pays-for-the-lookup gate");
        Check(graph.CallCount("fatInline") * analyzer.OverheadPerCall <= graph.Find("fatInline")!.Size * analyzer.GrowthWeight,
            "the rejected fat callee fails exactly the code-growth gate");
        Check(memoized.Contains("fib") && graph.Find("fib")!.Recursive,
            "a recursive pure function may still be memoized, which inlining cannot achieve");

        // 4. Simulation oracle: the predicted saving equals the measured cost delta.
        long before = analyzer.Simulate(graph, new HashSet<string>(), new HashSet<string>());
        long after = analyzer.Simulate(graph, inlined, memoized);
        Check(before - after == analyzer.TotalSaving,
            "the predicted saving " + analyzer.TotalSaving + " equals the simulated cost delta " + (before - after));

        // 5. Per-proposal savings recomputed from the call counts and costs.
        bool perItem = analyzer.Suggestions.All(s => s.Kind == SuggestionKind.Inline
            ? s.EstimatedSaving == graph.CallCount(s.Function) * analyzer.OverheadPerCall
            : s.EstimatedSaving == (graph.CallCount(s.Function) - graph.Find(s.Function)!.DistinctArguments)
                * graph.Find(s.Function)!.BodyCost - graph.CallCount(s.Function) * analyzer.CacheLookupCost);
        Check(perItem && analyzer.Suggestions.All(s => s.EstimatedSaving > 0),
            "every proposal's saving is recomputed exactly and is positive");

        // 6. Ranking and no double counting.
        bool sorted = analyzer.Suggestions.Zip(analyzer.Suggestions.Skip(1),
            (a, b) => a.EstimatedSaving >= b.EstimatedSaving).All(x => x);
        Check(sorted && analyzer.TotalSaving == analyzer.Suggestions.Sum(s => s.EstimatedSaving)
              && analyzer.Suggestions[0].EstimatedSaving == analyzer.Suggestions.Max(s => s.EstimatedSaving),
            "proposals are ranked by descending saving and the total is the sum of its parts");

        // 7. Policy invariant: one transformation per function, so the two sets are disjoint.
        Check(!inlined.Overlaps(memoized) && inlined.Count + memoized.Count == analyzer.Suggestions.Count,
            "no function is proposed for both transformations");

        // 8. Boundary: an empty graph, and a graph where nothing is actionable.
        var emptyAnalyzer = Analyzer();
        emptyAnalyzer.Analyze(new CallGraph());
        Check(emptyAnalyzer.Suggestions.Count == 0 && emptyAnalyzer.TotalSaving == 0,
            "an empty call graph yields no proposals and a zero total");
        var hopeless = new CallGraph();
        hopeless.Functions.AddRange(new[]
        {
            new HotFunction("a", 10, 5, false, true, 10000),
            new HotFunction("b", 5000, 60, false, false, 10000),
        });
        hopeless.Edges.AddRange(new[] { new CallEdge("a", "b", 9000), new CallEdge("b", "a", 9000) });
        var hopelessAnalyzer = Analyzer();
        hopelessAnalyzer.Analyze(hopeless);
        Check(hopelessAnalyzer.Suggestions.Count == 0
              && hopelessAnalyzer.Simulate(hopeless, new HashSet<string>(), new HashSet<string>()) > 0,
            "a graph of recursive and impure functions yields nothing actionable");

        // 9-10. Determinism and the reported plan.
        var repeat = Analyzer();
        repeat.Analyze(ShopGraph());
        Check(repeat.Suggestions.Select(s => s.Function + ":" + s.EstimatedSaving)
              .SequenceEqual(analyzer.Suggestions.Select(s => s.Function + ":" + s.EstimatedSaving)),
            "re-analysing the same graph reproduces the ranked plan exactly");

        Console.WriteLine("  plan: " + analyzer.Suggestions.Count + " proposals saving " + analyzer.TotalSaving
            + " cost units of " + before + " (top: " + analyzer.Suggestions[0].Kind + " " + analyzer.Suggestions[0].Function
            + " = " + analyzer.Suggestions[0].EstimatedSaving + ")");
        Console.WriteLine("PASS");
    }
}
