// Task 707 - Implement multi-commodity flow approximate algorithm outline.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// Multi-commodity flow asks for the largest concurrent fraction lambda such that every commodity i
// can push lambda * demand_i units from s_i to t_i while no edge carries more than its capacity.
// Unlike single-commodity flow this LP has no integrality and no pure combinatorial algorithm, so
// this file implements the multiplicative-weights (Garg-Koennemann / Young) outline: keep a
// *length* l_e = exp(eps * load_e / cap_e) / cap_e per edge, repeatedly route one small increment of
// each commodity along its shortest *residual* path under those lengths (saturated edges are dropped
// from the residual graph), and grow the length of every edge the increment uses. Congested edges become expensive, so later increments are pushed onto the paths
// that still have slack; that is what drives the routing towards a balanced, near-optimal
// concurrent flow.
//
// Documented simplification (the task asks for an outline): the classical algorithm adapts its
// increment sizes and stops when the potential stops decreasing, which certifies a (1-eps) ratio;
// here the schedule is a fixed number of equal increments and the guarantee is *not* certified.
// What is verified instead is everything that can be checked independently: the routing is always
// feasible (hard capacities, per-commodity demand caps, path-based conservation), it never exceeds
// a brute-force sparsest-cut upper bound, it matches two hand-computed optima exactly, and for a
// single commodity - where the optimum equals the minimum cut - it converges to the min cut within
// a documented tolerance.
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

internal sealed class MultiCommodityInstance
{
    internal readonly int Vertices;
    private readonly List<int>[] _out;
    private readonly List<int> _edgeTail = new List<int>();
    private readonly List<int> _edgeHead = new List<int>();
    private readonly List<double> _edgeCapacity = new List<double>();
    private readonly List<int[]> _commodities = new List<int[]>();     // {source, sink, demand}

    internal MultiCommodityInstance(int vertices)
    {
        Vertices = vertices;
        _out = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) _out[i] = new List<int>();
    }

    internal int EdgeCount => _edgeTail.Count;
    internal int Tail(int e) => _edgeTail[e];
    internal int Head(int e) => _edgeHead[e];
    internal double Capacity(int e) => _edgeCapacity[e];
    internal List<int> Outgoing(int v) => _out[v];
    internal IReadOnlyList<int[]> Commodities => _commodities;

    internal int AddEdge(int tail, int head, double capacity)
    {
        if (tail == head) throw new ArgumentException("self loops carry no useful flow");
        if (capacity <= 0) throw new ArgumentException("capacities must be positive");
        int id = _edgeTail.Count;
        _edgeTail.Add(tail);
        _edgeHead.Add(head);
        _edgeCapacity.Add(capacity);
        _out[tail].Add(id);
        return id;
    }

    internal void AddCommodity(int source, int sink, double demand)
    {
        if (source < 0 || source >= Vertices || sink < 0 || sink >= Vertices)
            throw new ArgumentOutOfRangeException(nameof(source), "commodity endpoint outside the graph");
        if (source == sink) throw new ArgumentException("a commodity needs two distinct endpoints");
        if (demand < 0) throw new ArgumentOutOfRangeException(nameof(demand), "demands must be non-negative");
        _commodities.Add(new[] { source, sink, (int)demand });
    }
}

internal sealed class MultiCommodityRouting
{
    private readonly MultiCommodityInstance _g;
    private readonly double[] _load;
    private readonly double[] _routed;
    private double _exponent = 0.05;
    internal long ShortestPathCalls;

    internal MultiCommodityRouting(MultiCommodityInstance g)
    {
        _g = g;
        _load = new double[g.EdgeCount];
        _routed = new double[g.Commodities.Count];
    }

    internal double[] Loads => _load;
    internal double[] Routed => _routed;

    /// <summary>Largest concurrent fraction the incremental multiplicative-weights router reaches.</summary>
    internal double Concurrent(int iterations, double eps)
    {
        _exponent = eps;
        for (int i = 0; i < _load.Length; i++) _load[i] = 0;
        for (int i = 0; i < _routed.Length; i++) _routed[i] = 0;
        if (_g.Commodities.Count == 0) return 1.0;                    // nothing to route is trivially routable

        for (int step = 0; step < iterations; step++)
        {
            for (int c = 0; c < _g.Commodities.Count; c++)
            {
                int[] commodity = _g.Commodities[c];
                double remaining = commodity[2] - _routed[c];
                if (remaining <= 0) continue;
                double increment = Math.Min(remaining, commodity[2] / (double)iterations);
                if (increment <= 0) continue;
                List<int>? path = ShortestPath(commodity[0], commodity[1]);
                if (path == null) continue;                            // unreachable: nothing to push
                double slack = increment;
                foreach (int e in path) slack = Math.Min(slack, _g.Capacity(e) - _load[e]);
                if (slack <= 0) continue;                              // this path is already saturated
                foreach (int e in path) _load[e] += slack;             // Length() turns the load into a penalty
                _routed[c] += slack;
            }
        }

        double lambda = double.MaxValue;
        for (int c = 0; c < _g.Commodities.Count; c++)
        {
            double demand = _g.Commodities[c][2];
            double fraction = demand == 0 ? 1.0 : _routed[c] / demand;
            if (fraction < lambda) lambda = fraction;
        }
        return Math.Min(1.0, lambda);
    }

    private double Length(int e) =>
        _load[e] <= 0 ? 1.0 / _g.Capacity(e) : Math.Exp(_exponent * _load[e] / _g.Capacity(e)) / _g.Capacity(e);

    /// <summary>Shortest residual path: saturated edges are removed, congested ones are penalised.</summary>
    private List<int>? ShortestPath(int source, int sink)
    {
        ShortestPathCalls++;
        int n = _g.Vertices;
        var dist = new double[n];
        var viaEdge = new int[n];
        var done = new bool[n];
        for (int i = 0; i < n; i++) { dist[i] = double.PositiveInfinity; viaEdge[i] = -1; }
        dist[source] = 0;
        for (int step = 0; step < n; step++)
        {
            int best = -1;
            for (int v = 0; v < n; v++) if (!done[v] && (best == -1 || dist[v] < dist[best])) best = v;
            if (best == -1 || double.IsPositiveInfinity(dist[best])) break;
            done[best] = true;
            foreach (int e in _g.Outgoing(best))
            {
                if (_g.Capacity(e) - _load[e] <= 1e-12) continue;      // saturated: not usable in the residual
                int head = _g.Head(e);
                double candidate = dist[best] + Length(e);
                if (candidate < dist[head]) { dist[head] = candidate; viaEdge[head] = e; }
            }
        }
        if (double.IsPositiveInfinity(dist[sink])) return null;
        var path = new List<int>();
        for (int cur = sink; cur != source;)
        {
            int e = viaEdge[cur];
            if (e < 0) return null;
            path.Add(e);
            cur = _g.Tail(e);
        }
        path.Reverse();
        return path;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>
    /// Cut bound on the concurrent fraction: for any vertex set S the net flow crossing S is
    /// lambda * (demand separated with its source in S) - lambda * (demand separated the other way),
    /// and that net can never exceed the capacity of the edges leaving S.
    /// </summary>
    private static double CutBound(MultiCommodityInstance g)
    {
        int n = g.Vertices;
        double best = double.MaxValue;
        for (int mask = 1; mask < (1 << n); mask++)
        {
            double outbound = 0, inbound = 0;
            foreach (int[] c in g.Commodities)
            {
                bool sourceIn = (mask & (1 << c[0])) != 0;
                bool sinkIn = (mask & (1 << c[1])) != 0;
                if (sourceIn && !sinkIn) outbound += c[2];
                else if (!sourceIn && sinkIn) inbound += c[2];
            }
            if (outbound <= inbound) continue;
            double cut = 0;
            for (int e = 0; e < g.EdgeCount; e++)
            {
                bool tailIn = (mask & (1 << g.Tail(e))) != 0;
                bool headIn = (mask & (1 << g.Head(e))) != 0;
                if (tailIn && !headIn) cut += g.Capacity(e);
            }
            best = Math.Min(best, cut / (outbound - inbound));
        }
        return best == double.MaxValue ? 1.0 : Math.Min(1.0, best);
    }

    /// <summary>Single-commodity optimum: the minimum s-t cut, i.e. the max flow (max-flow/min-cut).</summary>
    private static double MinCut(MultiCommodityInstance g, int source, int sink)
    {
        int n = g.Vertices;
        double best = double.MaxValue;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            if ((mask & (1 << source)) == 0 || (mask & (1 << sink)) != 0) continue;
            double cut = 0;
            for (int e = 0; e < g.EdgeCount; e++)
            {
                bool tailIn = (mask & (1 << g.Tail(e))) != 0;
                bool headIn = (mask & (1 << g.Head(e))) != 0;
                if (tailIn && !headIn) cut += g.Capacity(e);
            }
            best = Math.Min(best, cut);
        }
        return best;
    }

    public static void Main()
    {
        // ---- a shared bottleneck with a hand-computed optimum ----------------------------------
        {
            // Both commodities must cross edge 1->3, which carries 6 units.
            var g = new MultiCommodityInstance(4);
            g.AddEdge(0, 1, 5);
            g.AddEdge(2, 1, 5);
            int bottleneck = g.AddEdge(1, 3, 6);
            g.AddCommodity(0, 3, 5);
            g.AddCommodity(2, 3, 5);
            var router = new MultiCommodityRouting(g);

            double lambda = router.Concurrent(200, 0.05);
            Check(Math.Abs(lambda - 0.6) < 0.02,
                "the shared bottleneck forces lambda = 0.6 (got " + lambda.ToString("F4") + ")");
            Check(router.Loads[bottleneck] <= 6 + 1e-9, "the bottleneck edge never exceeds its capacity");
            Check(router.Routed[0] <= 5 + 1e-9 && router.Routed[1] <= 5 + 1e-9,
                "neither commodity exceeds its own demand");
            Console.WriteLine("sharedBottleneck lambda=" + lambda.ToString("F4")
                + " bottleneckLoad=" + router.Loads[bottleneck].ToString("F4"));
        }

        // ---- two commodities on disjoint paths are fully routable ------------------------------
        {
            var g = new MultiCommodityInstance(4);
            g.AddEdge(0, 1, 3);
            g.AddEdge(1, 3, 3);
            g.AddEdge(2, 3, 4);
            g.AddCommodity(0, 3, 3);
            g.AddCommodity(2, 3, 4);
            var router = new MultiCommodityRouting(g);

            double lambda = router.Concurrent(120, 0.05);
            Check(lambda > 0.999, "disjoint paths with exact capacity give lambda = 1 (got " + lambda.ToString("F4") + ")");
            Check(CutBound(g) >= 1.0 - 1e-9, "the cut bound confirms that 1 is attainable");
            Console.WriteLine("disjointLambda=" + lambda.ToString("F4"));
        }

        // ---- a single commodity must approach the min cut --------------------------------------
        {
            // Two parallel routes: 0->1->3 (3 + 3) and 0->2->3 (2 + 2), plus a thin 1->2 shortcut.
            var g = new MultiCommodityInstance(4);
            g.AddEdge(0, 1, 3);
            g.AddEdge(1, 3, 3);
            g.AddEdge(0, 2, 2);
            g.AddEdge(2, 3, 2);
            g.AddEdge(1, 2, 1);
            g.AddCommodity(0, 3, 20);
            double cut = MinCut(g, 0, 3);
            Check(Math.Abs(cut - 5) < 1e-9, "the minimum cut of the fixture is 5 (got " + cut + ")");
            var router = new MultiCommodityRouting(g);

            double lambda = router.Concurrent(400, 0.02);
            double achieved = lambda * 20;
            Check(achieved <= cut + 1e-6, "a single commodity can never exceed the min cut");
            Check(achieved >= 0.97 * cut,
                "the multiplicative-weights router converges to the min cut (" + achieved.ToString("F3")
                + " out of " + cut + ")");
            Console.WriteLine("singleCommodityAchieved=" + achieved.ToString("F3") + " minCut=" + cut);
        }

        // ---- randomized instances: feasibility and the cut bound -------------------------------
        {
            const int Instances = 25;
            ulong state = 0xF00DFACEUL;
            bool feasible = true, belowBound = true, monotone = true;
            string firstFailure = "none";
            double worstGap = 0;

            for (int t = 0; t < Instances; t++)
            {
                int n = 5 + Rng.NextInt(ref state, 3);
                var g = new MultiCommodityInstance(n);
                int edges = 7 + Rng.NextInt(ref state, 5);
                var seen = new HashSet<(int, int)>();
                for (int e = 0; e < edges; e++)
                {
                    int tail = Rng.NextInt(ref state, n);
                    int head = Rng.NextInt(ref state, n);
                    if (tail == head || !seen.Add((tail, head))) { e--; continue; }
                    g.AddEdge(tail, head, 1 + Rng.NextInt(ref state, 6));
                }
                int commodities = 1 + Rng.NextInt(ref state, 3);
                for (int c = 0; c < commodities; c++)
                {
                    int s = Rng.NextInt(ref state, n);
                    int target = Rng.NextInt(ref state, n);
                    if (s == target) target = (target + 1) % n;
                    g.AddCommodity(s, target, 1 + Rng.NextInt(ref state, 8));
                }

                var router = new MultiCommodityRouting(g);

                double lambda = router.Concurrent(200, 0.05);
                double bound = CutBound(g);
                for (int e = 0; e < g.EdgeCount; e++)
                {
                    if (router.Loads[e] > g.Capacity(e) + 1e-9) { feasible = false; firstFailure = "instance " + t + " overloaded edge " + e; }
                    if (router.Loads[e] < -1e-12) { feasible = false; firstFailure = "instance " + t + " negative load"; }
                }
                for (int c = 0; c < g.Commodities.Count; c++)
                {
                    if (router.Routed[c] > g.Commodities[c][2] + 1e-9)
                    {
                        feasible = false;
                        firstFailure = "instance " + t + " over-delivered commodity " + c;
                    }
                }
                if (lambda > bound + 1e-9) { belowBound = false; firstFailure = "instance " + t + " beat the cut bound"; }
                double again = new MultiCommodityRouting(g).Concurrent(60, 0.05);
                double more = router.Concurrent(400, 0.05);
                if (more + 1e-6 < again) { monotone = false; firstFailure = "instance " + t + " lost value with more iterations"; }
                if (bound > 0) worstGap = Math.Max(worstGap, (bound - lambda) / bound);
            }
            Check(feasible, "every routing respects all capacities and demands (" + firstFailure + ")");
            Check(belowBound, "no routing exceeds the sparsest-cut upper bound (" + firstFailure + ")");
            Check(monotone, "more iterations never reduce the concurrent fraction (" + firstFailure + ")");
            Console.WriteLine("randomInstances=" + Instances + " worstRelativeGapToCutBound=" + worstGap.ToString("F3"));
        }

        // ---- boundary and rejected input ------------------------------------------------------
        {
            var empty = new MultiCommodityInstance(3);
            empty.AddEdge(0, 1, 5);
            var none = new MultiCommodityRouting(empty);

            Check(none.Concurrent(20, 0.05) == 1.0, "with no commodities the concurrent fraction is 1 by convention");

            var zeroDemand = new MultiCommodityInstance(3);
            zeroDemand.AddEdge(0, 2, 4);
            zeroDemand.AddCommodity(0, 2, 0);
            var zeroRouter = new MultiCommodityRouting(zeroDemand);

            Check(zeroRouter.Concurrent(20, 0.05) == 1.0, "a zero demand is trivially satisfied");

            var unreachable = new MultiCommodityInstance(4);
            unreachable.AddEdge(0, 1, 9);
            unreachable.AddCommodity(0, 3, 5);
            var stuck = new MultiCommodityRouting(unreachable);

            Check(stuck.Concurrent(40, 0.05) == 0.0, "an unreachable commodity forces the concurrent fraction to zero");
            Check(stuck.Loads[0] == 0, "nothing is routed when there is no path");

            bool thrown = false;
            try { unreachable.AddEdge(1, 1, 3); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a self loop is rejected");
            thrown = false;
            try { unreachable.AddEdge(1, 2, 0); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a zero capacity is rejected");
            thrown = false;
            try { unreachable.AddCommodity(0, 9, 1); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a commodity endpoint outside the graph is rejected");
            thrown = false;
            try { unreachable.AddCommodity(0, 0, 1); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a commodity from a vertex to itself is rejected");

            var deterministic = new MultiCommodityInstance(4);
            deterministic.AddEdge(0, 1, 4);
            deterministic.AddEdge(1, 3, 4);
            deterministic.AddEdge(0, 2, 2);
            deterministic.AddEdge(2, 3, 2);
            deterministic.AddCommodity(0, 3, 6);
            var first = new MultiCommodityRouting(deterministic);

            var second = new MultiCommodityRouting(deterministic);

            double a = first.Concurrent(150, 0.03);
            double b = second.Concurrent(150, 0.03);
            Check(a == b, "the router is deterministic");
            Check(first.ShortestPathCalls > 0, "the router really computes shortest paths");
            Console.WriteLine("deterministicLambda=" + a.ToString("F4") + " shortestPathCalls=" + first.ShortestPathCalls);
        }

        Console.WriteLine("PASS");
    }
}
