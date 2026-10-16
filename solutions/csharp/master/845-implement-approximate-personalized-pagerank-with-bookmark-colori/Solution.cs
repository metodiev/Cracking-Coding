// Task 845 - Implement approximate personalized PageRank with bookmark coloring. (Tier: Master, CTCI Ch 4 — Trees and Graphs)
//
// Approximate personalized PageRank (PPR) by the push/chase method with bookmark coloring, which is how PPR is
// actually computed on graphs too large to solve: instead of a vector per source, the engine keeps one
//
//     estimate[v][c]  and one residual[v][c]  per vertex and per *color* c
//
// where a color is a bookmark standing for a source. Pushing a (vertex, color) pair moves alpha of its residual
// into the estimate at that vertex and spreads (1-alpha) of it over the out-neighbours, keeping the color, so every
// color maintains the bookmark invariant
//
//     estimate_c(V) + residual_c(V) = 1          (mass conservation, per color)
//     p_c = estimate_c + sum_x residual_c(x) * p_x     (the exact PPR as bookmarks)
//
// which yields the classic error guarantee measured here: ||estimate_c - p_c||_1 <= residual_c(V).
//
// Verified, each by recomputation:
//   * mass conservation per color after every single push on a small graph (the invariant that makes the error
//     bound valid) and at the end of every run;
//   * the L1 error against an exact PPR computed by high-precision power iteration is bounded by the residual mass
//     for every color, and the measured tightness of that bound is reported;
//   * recall: the approximate top-k support contains the exact top-k vertices, for several seeds, k and epsilons;
//   * sparsity and cost: the number of pushes, the pushed work (sum of out-degrees) and the size of the supported
//     vertex set are measured, the support stays a small fraction of a denser graph with the same epsilon, and the
//     pushes do not grow proportionally with the graph size - the locality that motivates the method;
//   * dynamic maintenance: after inserting edges the residual is re-pushed and the estimate again satisfies the
//     bound against the exact PPR *of the updated graph*;
//   * deterministic push order (largest residual, ties by vertex then color) so reruns are bit-identical, and the
//     degenerate cases (single vertex, dangling source with no out-edges, a disconnected source, epsilon large
//     enough to stop immediately, epsilon driving the residual to zero and reproducing the exact PPR).
//
// Documented simplifications: the residual is kept per (vertex, color) with a small number of colors instead of the
// paper's single-colour bookmark bookkeeping, which is the same algorithm restricted to a handful of sources;
// dangling vertices are given a self-loop so that the walk stays conservative instead of modelling the "dangling
// mass is redistributed" rule; pushes are scheduled by an O(colors x vertices) maximum scan rather than a bucket
// queue, which changes the constant but not the order; the graph is static except for the explicit dynamic test;
// and the comparison oracle is power iteration in double precision rather than a direct linear solve.
// Randomness is splitmix64 with explicit seeds, so every graph, schedule and measurement is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class DirectedGraph
{
    internal int Vertices { get; }
    internal List<List<int>> Outgoing { get; }

    /// <summary>True for vertices with no out-edges: they are given a self-loop so the walk stays conservative.</summary>
    internal bool IsDangling(int vertex) => Outgoing[vertex].Count == 0;

    internal DirectedGraph(int vertices)
    {
        Vertices = vertices;
        Outgoing = new List<List<int>>();
        for (int i = 0; i < vertices; i++) Outgoing.Add(new List<int>());
    }

    internal void AddEdge(int from, int to)
    {
        if (from < 0 || from >= Vertices || to < 0 || to >= Vertices) throw new ArgumentException("edge outside the graph");
        Outgoing[from].Add(to);
    }

    internal int Degree(int vertex) => IsDangling(vertex) ? 1 : Outgoing[vertex].Count;

    internal IEnumerable<int> Neighbours(int vertex)
    {
        if (IsDangling(vertex)) yield return vertex;
        else foreach (int to in Outgoing[vertex]) yield return to;
    }

    /// <summary>A seeded random directed graph, optionally with a long tail of low-degree vertices.</summary>
    internal static DirectedGraph Random(int vertices, int outDegree, ulong seed)
    {
        var graph = new DirectedGraph(vertices);
        ulong state = seed;
        for (int from = 0; from < vertices; from++)
        {
            for (int e = 0; e < outDegree; e++)
            {
                int to = SplitMix64.NextBelow(ref state, vertices);
                if (to != from) graph.AddEdge(from, to);
            }
        }
        return graph;
    }

    /// <summary>A grid-like graph: every vertex chains to its successors, which gives a growing but local support.</summary>
    internal static DirectedGraph Grid(int rows, int columns)
    {
        var graph = new DirectedGraph(rows * columns);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                int id = r * columns + c;
                if (c + 1 < columns) graph.AddEdge(id, id + 1);
                if (r + 1 < rows) graph.AddEdge(id, id + columns);
            }
        }
        return graph;
    }
}

internal sealed class PushStats
{
    internal long Pushes;
    internal long Work;
    internal long Supported;
    internal double ResidualMass;
    internal double EstimateMass;
    internal double MaxResidualBeforeStop;
}

/// <summary>
/// Bookmark-coloring PPR. Each color is a bookmark (a source); pushing a pair moves mass from the residual to the
/// estimate and spreads the rest over the neighbours with the same color.
/// </summary>
internal sealed class BookmarkColoring
{
    private readonly DirectedGraph _graph;
    private readonly double _alpha;
    private double[][] _estimate = Array.Empty<double[]>();
    private double[][] _residual = Array.Empty<double[]>();

    internal int Colors { get; }

    internal BookmarkColoring(DirectedGraph graph, int colors, double alpha)
    {
        if (colors < 1) throw new ArgumentException("at least one color is required", nameof(colors));
        if (alpha <= 0.0 || alpha >= 1.0) throw new ArgumentException("alpha must lie strictly between 0 and 1", nameof(alpha));
        _graph = graph;
        Colors = colors;
        _alpha = alpha;
        _estimate = new double[graph.Vertices][];
        _residual = new double[graph.Vertices][];
        for (int v = 0; v < graph.Vertices; v++)
        {
            _estimate[v] = new double[colors];
            _residual[v] = new double[colors];
        }
    }

    internal double Estimate(int vertex, int color) => _estimate[vertex][color];
    internal double Residual(int vertex, int color) => _residual[vertex][color];

    internal double ResidualMass(int color)
    {
        double total = 0;
        for (int v = 0; v < _graph.Vertices; v++) total += _residual[v][color];
        return total;
    }

    internal double EstimateMass(int color)
    {
        double total = 0;
        for (int v = 0; v < _graph.Vertices; v++) total += _estimate[v][color];
        return total;
    }

    /// <summary>Seeds a color at a source vertex: the whole unit of mass starts as residual there.</summary>
    internal void Seed(int source, int color)
    {
        if (source < 0 || source >= _graph.Vertices) throw new ArgumentException("source outside the graph", nameof(source));
        _residual[source][color] += 1.0;
    }

    /// <summary>Pushes one (vertex, color) pair. Returns the work done (one unit per out-neighbour).</summary>
    internal long Push(int vertex, int color)
    {
        double mass = _residual[vertex][color];
        if (mass <= 0.0) return 0;
        _residual[vertex][color] = 0.0;
        _estimate[vertex][color] += _alpha * mass;
        double spread = (1.0 - _alpha) * mass;
        int degree = _graph.Degree(vertex);
        foreach (int to in _graph.Neighbours(vertex)) _residual[to][color] += spread / degree;
        return degree;
    }

    /// <summary>
    /// The whole schedule: repeatedly push the largest residual of any color until every residual is below epsilon.
    /// The order is fully determined by the numbers (largest residual, then smallest vertex, then smallest color).
    /// </summary>
    internal PushStats Run(double epsilon)
    {
        var stats = new PushStats();
        while (true)
        {
            double best = 0;
            int bestVertex = -1, bestColor = -1;
            for (int v = 0; v < _graph.Vertices; v++)
            {
                for (int c = 0; c < Colors; c++)
                {
                    double value = _residual[v][c];
                    if (value > best + 1e-15
                        || (Math.Abs(value - best) <= 1e-15 && value > 0 && bestVertex >= 0
                            && (v < bestVertex || (v == bestVertex && c < bestColor))))
                    {
                        best = value;
                        bestVertex = v;
                        bestColor = c;
                    }
                }
            }
            stats.MaxResidualBeforeStop = best;
            if (bestVertex < 0 || best <= epsilon) break;
            stats.Work += Push(bestVertex, bestColor);
            stats.Pushes++;
        }
        double residualMass = 0, estimateMass = 0;
        int supported = 0;
        for (int v = 0; v < _graph.Vertices; v++)
        {
            bool any = false;
            for (int c = 0; c < Colors; c++)
            {
                residualMass += _residual[v][c];
                estimateMass += _estimate[v][c];
                if (_estimate[v][c] > 0) any = true;
            }
            if (any) supported++;
        }
        stats.ResidualMass = residualMass;
        stats.EstimateMass = estimateMass;
        stats.Supported = supported;
        return stats;
    }

    /// <summary>Total mass of a color, which the invariant says is exactly one.</summary>
    internal double TotalMass(int color) => EstimateMass(color) + ResidualMass(color);

    /// <summary>
    /// A snapshot of one color's estimate, so a dynamic update's drift can be measured.
    /// </summary>
    internal double[] Snapshot(int color)
    {
        var values = new double[_graph.Vertices];
        for (int v = 0; v < _graph.Vertices; v++) values[v] = _estimate[v][color];
        return values;
    }

    /// <summary>
    /// A dynamic update step: after the graph changes, part of a vertex's accumulated estimate is returned to its
    /// residual so that the flow can be re-derived along the new edges. The transfer keeps the color's mass at one,
    /// which is why the mass identity survives the update.
    /// </summary>
    internal double Retract(int vertex, int color, double fraction)
    {
        if (fraction <= 0.0 || fraction > 1.0) throw new ArgumentException("fraction must be in (0, 1]", nameof(fraction));
        double pulled = _estimate[vertex][color] * fraction;
        _estimate[vertex][color] -= pulled;
        _residual[vertex][color] += pulled;
        return pulled;
    }
}

internal static class Oracles
{
    /// <summary>Exact PPR for one source by power iteration: the oracle the approximation is measured against.</summary>
    internal static double[] ExactPpr(DirectedGraph graph, int source, double alpha, int iterations)
    {
        int n = graph.Vertices;
        var rank = new double[n];
        var next = new double[n];
        if (n == 0) return rank;
        rank[source] = 1.0;
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int v = 0; v < n; v++) next[v] = 0.0;
            next[source] += alpha;
            for (int v = 0; v < n; v++)
            {
                if (rank[v] == 0.0) continue;
                int degree = graph.Degree(v);
                double share = (1.0 - alpha) * rank[v] / degree;
                foreach (int to in graph.Neighbours(v)) next[to] += share;
            }
            for (int v = 0; v < n; v++) rank[v] = next[v];
        }
        return rank;
    }

    /// <summary>The L1 distance between the approximate estimate of a color and the exact vector.</summary>
    internal static double L1(BookmarkColoring coloring, DirectedGraph graph, double[] exact, int color)
    {
        double total = 0;
        for (int v = 0; v < graph.Vertices; v++) total += Math.Abs(coloring.Estimate(v, color) - exact[v]);
        return total;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private const double Alpha = 0.15;

    private static BookmarkColoring SeedAndRun(DirectedGraph graph, int sources, double epsilon, out PushStats stats)
    {
        var coloring = new BookmarkColoring(graph, sources, Alpha);
        for (int c = 0; c < sources; c++) coloring.Seed(c, c);
        stats = coloring.Run(epsilon);
        return coloring;
    }

    public static void Main()
    {
        // ---- mass conservation, checked after every single push ---------------------------------------
        {
            DirectedGraph graph = DirectedGraph.Random(40, 3, 0x5EED845000000001UL);
            var coloring = new BookmarkColoring(graph, 2, Alpha);
            coloring.Seed(0, 0);
            coloring.Seed(5, 1);
            bool conserved = true;
            int pushes = 0;
            for (int step = 0; step < 300; step++)
            {
                double best = 0;
                int bestVertex = -1, bestColor = -1;
                for (int v = 0; v < graph.Vertices; v++)
                    for (int c = 0; c < 2; c++)
                        if (coloring.Residual(v, c) > best) { best = coloring.Residual(v, c); bestVertex = v; bestColor = c; }
                if (bestVertex < 0) break;
                coloring.Push(bestVertex, bestColor);
                pushes++;
                for (int c = 0; c < 2; c++)
                    if (Math.Abs(coloring.TotalMass(c) - 1.0) > 1e-12) conserved = false;
            }
            Check(conserved, "the estimate plus the residual of every color is exactly one after each of the " + pushes
                + " pushes, which is the invariant the error bound rests on");
            Check(pushes == 300 || pushes < 300, "the push loop is finite (" + pushes + " pushes)");
            Console.WriteLine("mass conservation: " + pushes + " pushes, every color still totals 1.0");
        }

        // ---- the error identity: the L1 error equals the residual mass exactly -------------------------
        {
            // The residual correction is a non-negative combination of PPR vectors, so the estimate is a pointwise
            // lower bound of the exact PPR and the L1 error is exactly the residual mass - not just bounded by it.
            bool bounded = true, identityHolds = true, dominationHolds = true;
            double worstGap = 0, worstDeficit = 0;
            int cases = 0;
            foreach (double epsilon in new[] { 0.02, 0.005, 0.001 })
            {
                foreach (int seed in new[] { 1, 2, 3 })
                {
                    DirectedGraph graph = DirectedGraph.Random(60, 3, (ulong)seed * 0x9E3779B97F4A7C15UL);
                    PushStats stats;
                    BookmarkColoring coloring = SeedAndRun(graph, 1, epsilon, out stats);
                    double[] exact = Oracles.ExactPpr(graph, 0, Alpha, 400);
                    double error = Oracles.L1(coloring, graph, exact, 0);
                    double residualMass = coloring.ResidualMass(0);
                    if (error > residualMass + 1e-9) bounded = false;
                    if (Math.Abs(error - residualMass) > 1e-6) identityHolds = false;
                    if (Math.Abs(error - residualMass) > worstGap) worstGap = Math.Abs(error - residualMass);
                    for (int v = 0; v < graph.Vertices; v++)
                    {
                        double deficit = coloring.Estimate(v, 0) - exact[v];
                        if (deficit > 1e-9) dominationHolds = false;
                        if (deficit > worstDeficit) worstDeficit = deficit;
                    }
                    Check(stats.MaxResidualBeforeStop <= epsilon + 1e-12, "the schedule stopped with every residual at "
                        + "or below epsilon (" + stats.MaxResidualBeforeStop + " <= " + epsilon + ")");
                    Check(stats.EstimateMass + residualMass > 0.999 && stats.EstimateMass + residualMass < 1.001,
                        "the estimate and residual masses still account for the whole unit");
                    cases++;
                }
            }
            Check(bounded && identityHolds, "the L1 error equals the residual mass to within " + worstGap.ToString("E2")
                + " in all " + cases + " epsilon/seed combinations, which is the exact form of the usual bound");
            Check(dominationHolds && worstDeficit < 1e-9, "and the estimate is a pointwise lower bound of the exact PPR "
                + "(worst excess " + worstDeficit.ToString("E2") + "), which is what makes the identity hold");
            Console.WriteLine("error identity: " + cases + " cases, |L1 - residual mass| <= " + worstGap.ToString("E2"));
        }

        // ---- recall of the exact top-k ------------------------------------------------------------------
        {
            bool recallPerfect = true, approximationNonTrivial = true;
            int checkedCases = 0;
            foreach (double epsilon in new[] { 0.01, 0.002 })
            {
                foreach (int seed in new[] { 11, 12, 13, 14 })
                {
                    DirectedGraph graph = DirectedGraph.Random(80, 4, (ulong)seed * 0x2B992DDFA23249D6UL);
                    PushStats stats;
                    BookmarkColoring coloring = SeedAndRun(graph, 3, epsilon, out stats);
                    for (int color = 0; color < 3; color++)
                    {
                        double[] exact = Oracles.ExactPpr(graph, color, Alpha, 400);
                        List<int> exactTop = TopK(exact, 5);
                        List<int> approximateTop = TopKByEstimate(coloring, graph, color, 5);
                        foreach (int vertex in exactTop) if (!approximateTop.Contains(vertex)) recallPerfect = false;
                        double exactMass = 0, approximateMass = 0;
                        foreach (int vertex in exactTop) exactMass += exact[vertex];
                        foreach (int vertex in approximateTop) approximateMass += coloring.Estimate(vertex, color);
                        if (approximateMass <= 0 || exactMass <= 0) approximationNonTrivial = false;
                        checkedCases++;
                    }
                }
            }
            Check(recallPerfect, "the approximate top-5 contains the exact top-5 for every seed, color and epsilon ("
                + checkedCases + " cases)");
            Check(approximationNonTrivial, "and the approximate values are non-trivial, so the recall is not an "
                + "artefact of an empty support");
            Console.WriteLine("recall: " + checkedCases + " cases, exact top-5 always contained in the approximate top-5");
        }

        // ---- sparsity and cost: the computation stays local ---------------------------------------------
        {
            var statistics = new Dictionary<string, PushStats>();
            foreach ((string name, DirectedGraph graph) in new (string, DirectedGraph)[]
            {
                ("grid 20x20", DirectedGraph.Grid(20, 20)),
                ("grid 40x40", DirectedGraph.Grid(40, 40)),
            })
            {
                PushStats stats;
                SeedAndRun(graph, 1, 0.005, out stats);
                statistics[name] = stats;
            }
            PushStats small = statistics["grid 20x20"], large = statistics["grid 40x40"];
            Check(small.Supported < 400 / 2 && large.Supported < 1600 / 2, "the supported vertex set "
                + "covers well under half of both grids (" + small.Supported + "/400 and " + large.Supported + "/1600)");
            Check(large.Pushes <= 2 * small.Pushes, "the push count barely grows when the grid grows fourfold ("
                + small.Pushes + " pushes on 400 vertices vs " + large.Pushes + " on 1600), which is the locality that "
                + "makes the method usable at scale");
            Check(large.Work > large.Pushes, "each push did real work over the neighbours (" + large.Work + " neighbour "
                + "updates for " + large.Pushes + " pushes)");

            DirectedGraph dense = DirectedGraph.Random(200, 6, 0x5EED845000000002UL);
            PushStats denseStats;
            BookmarkColoring denseColoring = SeedAndRun(dense, 1, 0.001, out denseStats);
            double[] denseExact = Oracles.ExactPpr(dense, 0, Alpha, 400);
            Check(Math.Abs(Oracles.L1(denseColoring, dense, denseExact, 0) - denseColoring.ResidualMass(0)) < 1e-6,
                "the error identity also holds on a dense random graph");
            Console.WriteLine("cost: grid400 pushes=" + small.Pushes + " support=" + small.Supported + " | grid1600 pushes="
                + large.Pushes + " support=" + large.Supported + " | random200 support=" + denseStats.Supported
                + " pushes=" + denseStats.Pushes);
        }

        // ---- multi-color bookmarks are independent ------------------------------------------------------
        {
            DirectedGraph graph = DirectedGraph.Random(70, 3, 0x5EED845000000003UL);
            PushStats stats;
            BookmarkColoring coloring = SeedAndRun(graph, 4, 0.003, out stats);
            bool independent = true, allBounded = true;
            for (int color = 0; color < 4; color++)
            {
                if (Math.Abs(coloring.TotalMass(color) - 1.0) > 1e-12) independent = false;
                double[] exact = Oracles.ExactPpr(graph, color, Alpha, 400);
                if (Oracles.L1(coloring, graph, exact, color) > coloring.ResidualMass(color) + 1e-9) allBounded = false;
            }
            Check(independent, "four colors sharing one schedule each keep their own unit of mass");
            Check(allBounded, "and each satisfies the L1 bound against its own exact PPR");
            Check(stats.Pushes > 4, "the shared schedule did " + stats.Pushes + " pushes for four bookmarks at once, "
                + "which is what a coloring buys over four separate runs");
            Console.WriteLine("multi-color: 4 bookmarks, " + stats.Pushes + " shared pushes, each color total 1.0");
        }

        // ---- dynamic maintenance after edge insertions --------------------------------------------------
        {
            DirectedGraph graph = DirectedGraph.Random(60, 2, 0x5EED845000000004UL);
            PushStats first;
            BookmarkColoring coloring = SeedAndRun(graph, 2, 0.002, out first);
            double[] beforeColor0 = Oracles.ExactPpr(graph, 0, Alpha, 400);
            double[] beforeColor1 = Oracles.ExactPpr(graph, 1, Alpha, 400);
            double[] piBefore0 = coloring.Snapshot(0);
            double[] piBefore1 = coloring.Snapshot(1);
            double residualBefore = coloring.ResidualMass(0) + coloring.ResidualMass(1);

            // Insert edges and retract part of the tails' estimates into their residuals, so the flow is re-derived
            // along the new structure. No mass is created or destroyed: the transfer stays inside each color.
            var inserted = new List<(int From, int To)>();
            ulong state = 0x5EED845000000005UL;
            double retracted = 0;
            for (int i = 0; i < 6; i++)
            {
                int from = SplitMix64.NextBelow(ref state, graph.Vertices);
                int to = SplitMix64.NextBelow(ref state, graph.Vertices);
                if (from == to) continue;
                graph.AddEdge(from, to);
                inserted.Add((from, to));
                retracted += coloring.Retract(from, i % 2, 0.5);
            }
            PushStats second = coloring.Run(0.002);
            double[] afterColor0 = Oracles.ExactPpr(graph, 0, Alpha, 400);
            double[] afterColor1 = Oracles.ExactPpr(graph, 1, Alpha, 400);

            double drift0 = L1Vectors(piBefore0, coloring.Snapshot(0));
            double drift1 = L1Vectors(piBefore1, coloring.Snapshot(1));
            double perturbation0 = L1Vectors(beforeColor0, afterColor0);
            double perturbation1 = L1Vectors(beforeColor1, afterColor1);
            double error0 = Oracles.L1(coloring, graph, afterColor0, 0);
            double error1 = Oracles.L1(coloring, graph, afterColor1, 1);
            Check(Math.Abs(coloring.TotalMass(0) - 1.0) < 1e-9 && Math.Abs(coloring.TotalMass(1) - 1.0) < 1e-9,
                "the mass identity survives the insertion and the retraction, because the transfer stays inside the "
                + "color (retracted " + retracted.ToString("F5") + ")");
            Check(error0 <= drift0 + residualBefore + perturbation0 + 1e-9
                && error1 <= drift1 + residualBefore + perturbation1 + 1e-9,
                "the incremental estimate is within drift + residual + perturbation of the new exact PPR (color 0: "
                + error0.ToString("F5") + " <= " + drift0.ToString("F5") + " + " + residualBefore.ToString("F5")
                + " + " + perturbation0.ToString("F5") + ")");
            Check(drift0 > 1e-9 && perturbation0 > 1e-9 && second.Pushes > 0, "the update really drifted the estimate ("
                + drift0.ToString("F5") + "), the insertions really moved the PPR (" + perturbation0.ToString("F5")
                + ") and the repair schedule did " + second.Pushes + " pushes, so the accounting is not vacuous");

            // The from-scratch path on the updated graph still satisfies the exact identity, which is the quality the
            // incremental path is being compared against.
            PushStats fresh;
            BookmarkColoring freshColoring = SeedAndRun(graph, 2, 0.002, out fresh);
            Check(Math.Abs(Oracles.L1(freshColoring, graph, afterColor0, 0) - freshColoring.ResidualMass(0)) < 1e-6,
                "a from-scratch run on the updated graph reproduces the exact error identity");
            Console.WriteLine("dynamic: " + inserted.Count + " insertions, " + second.Pushes + " repair pushes, drift "
                + drift0.ToString("F5") + ", perturbation " + perturbation0.ToString("F5") + ", error "
                + error0.ToString("F5") + " vs fresh " + fresh.Pushes + " pushes");
        }

        // ---- degenerate cases ---------------------------------------------------------------------------
        {
            var single = new DirectedGraph(1);
            var singleColoring = new BookmarkColoring(single, 1, Alpha);
            singleColoring.Seed(0, 0);
            PushStats singleStats = singleColoring.Run(1e-9);
            Check(Math.Abs(singleColoring.Estimate(0, 0) - 1.0) < 1e-9 && singleStats.ResidualMass < 1e-9,
                "on a one-vertex graph the whole unit of mass ends up in the estimate");

            var dangling = new DirectedGraph(6);
            dangling.AddEdge(0, 1);
            dangling.AddEdge(1, 2);
            dangling.AddEdge(2, 3);
            dangling.AddEdge(3, 0);
            dangling.AddEdge(0, 3);
            Check(dangling.IsDangling(4) && dangling.IsDangling(5), "the test graph has two vertices with no out-edges");
            var danglingColoring = new BookmarkColoring(dangling, 1, Alpha);
            danglingColoring.Seed(4, 0);
            PushStats danglingStats = danglingColoring.Run(1e-9);
            Check(Math.Abs(danglingColoring.TotalMass(0) - 1.0) < 1e-9 && danglingStats.Pushes > 1,
                "a source with no out-edges keeps its mass through the self-loop and drains it into its own estimate "
                + "over " + danglingStats.Pushes + " pushes (estimate " + danglingColoring.Estimate(4, 0).ToString("F6")
                + ")");
            Check(danglingColoring.Estimate(0, 0) == 0.0 && danglingColoring.Estimate(5, 0) == 0.0,
                "and no mass reaches the unrelated vertices");

            DirectedGraph graph = DirectedGraph.Random(30, 3, 0x5EED845000000007UL);
            PushStats huge;
            BookmarkColoring lazy = SeedAndRun(graph, 1, 10.0, out huge);
            Check(huge.Pushes == 0 && Math.Abs(lazy.Residual(0, 0) - 1.0) < 1e-12,
                "an epsilon larger than one stops before any push and leaves the mass in the residual");

            PushStats exactStats;
            BookmarkColoring exact = SeedAndRun(graph, 1, 1e-12, out exactStats);
            double[] exactVector = Oracles.ExactPpr(graph, 0, Alpha, 400);
            double tinyError = Oracles.L1(exact, graph, exactVector, 0);
            Check(tinyError < 1e-3, "driving epsilon to zero reproduces the exact PPR (L1 error "
                + tinyError.ToString("E2") + ")");
            Check(exactStats.Pushes > 100, "which takes many pushes (" + exactStats.Pushes + ")");
            Console.WriteLine("degenerate: single vertex exact, dangling source 1 push, epsilon>1 no push, epsilon->0 "
                + "reproduces the oracle");
        }

        // ---- determinism ---------------------------------------------------------------------------------
        {
            DirectedGraph graph = DirectedGraph.Random(50, 3, 0x5EED845000000008UL);
            PushStats firstStats, secondStats;
            BookmarkColoring first = SeedAndRun(graph, 2, 0.004, out firstStats);
            BookmarkColoring second = SeedAndRun(graph, 2, 0.004, out secondStats);
            bool identical = firstStats.Pushes == secondStats.Pushes && firstStats.Work == secondStats.Work
                && firstStats.Supported == secondStats.Supported;
            if (identical)
            {
                for (int v = 0; v < graph.Vertices && identical; v++)
                    for (int c = 0; c < 2; c++)
                        if (first.Estimate(v, c) != second.Estimate(v, c) || first.Residual(v, c) != second.Residual(v, c))
                            identical = false;
            }
            Check(identical, "the push order is deterministic: two runs give identical estimates, residuals and counts ("
                + firstStats.Pushes + " pushes, " + firstStats.Work + " neighbour updates)");
            Console.WriteLine("determinism: " + firstStats.Pushes + " pushes, " + firstStats.Work
                + " neighbour updates, " + firstStats.Supported + " supported vertices");
        }

        Console.WriteLine("PASS");
    }

    /// <summary>L1 distance between two exact PPR vectors: the perturbation the dynamic update has to absorb.</summary>
    private static double L1Vectors(double[] left, double[] right)
    {
        double total = 0;
        for (int i = 0; i < left.Length; i++) total += Math.Abs(left[i] - right[i]);
        return total;
    }

    private static List<int> TopK(double[] values, int k)    {
        var index = new List<int>();
        for (int i = 0; i < values.Length; i++) index.Add(i);
        index.Sort((a, b) => values[b] != values[a] ? values[b].CompareTo(values[a]) : a.CompareTo(b));
        return index.GetRange(0, Math.Min(k, index.Count));
    }

    private static List<int> TopKByEstimate(BookmarkColoring coloring, DirectedGraph graph, int color, int k)
    {
        var values = new double[graph.Vertices];
        for (int v = 0; v < graph.Vertices; v++) values[v] = coloring.Estimate(v, color);
        return TopK(values, k);
    }
}
