// Task 706 - Implement max flow with lower/upper bounds and circulation feasibility.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// Every edge carries an interval [lo, hi] instead of a bare capacity, so a flow must respect the
// lower bound *and* conservation at every inner vertex. The standard transform turns that into a
// plain max flow: pre-send lo on every edge, which leaves a surplus balance[v] = sum_out(lo) -
// sum_in(lo) at each vertex; then route the repair flow on the residual capacities hi - lo, with a
// super sink draining the vertices of positive balance and a super source feeding the negative ones.
// The instance is feasible exactly when the super source side saturates (all repairs delivered).
// For the s-t variant a feedback edge t -> s of unbounded capacity is added, lower bounds are
// satisfied on the whole network, its flow is the current value of the s-t flow, and afterwards
// disabling that edge and running one more max flow on the residual graph maximises the value
// inside the lower-bounded polytope (the "feasible flow then augment" recipe). Dinic supplies the
// max flow engine (BFS levels + blocking flow with a current-arc pointer).
//
// The feasible flow must send a non-negative amount from s to t - that is the convention behind the
// t -> s feedback edge, whose flow *is* the value of the repaired flow. (An instance whose lower
// bounds force net flow into the source therefore counts as infeasible, and the second oracle below
// applies the same rule.)
//
// Verified below by three independent means: every returned flow is re-checked for bounds and
// conservation (and for the declared value), tiny instances are solved by exhaustively enumerating
// every integral flow vector in the box, and zero-lower-bound instances are compared with the
// brute-force minimum cut over all vertex subsets (max-flow/min-cut). Malformed input, infeasible
// lower bounds, degenerate edges and unreachable sinks get their own checks.
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

/// <summary>Dinic max flow on integer capacities, with access to the flow carried by an edge.</summary>
internal sealed class Dinic
{
    private readonly List<int>[] _incident;
    private readonly List<int> _to = new List<int>();
    private readonly List<long> _initial = new List<long>();
    private readonly List<long> _residual = new List<long>();
    private readonly int[] _level;
    private readonly int[] _iter;

    internal Dinic(int vertices)
    {
        _incident = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) _incident[i] = new List<int>();
        _level = new int[vertices];
        _iter = new int[vertices];
    }

    internal int AddEdge(int u, int v, long capacity)
    {
        if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        int id = _to.Count;
        _to.Add(v); _initial.Add(capacity); _residual.Add(capacity); _incident[u].Add(id);
        _to.Add(u); _initial.Add(0); _residual.Add(0); _incident[v].Add(id + 1);
        return id;
    }

    internal long FlowOn(int id) => _initial[id] - _residual[id];
    internal long ResidualOf(int id) => _residual[id];

    /// <summary>Removes both directions of an edge from the graph (used to retire the feedback edge).</summary>
    internal void DisableEdge(int id)
    {
        _residual[id] = 0;
        _residual[id ^ 1] = 0;
    }

    internal long MaxFlow(int s, int t)
    {
        long total = 0;
        while (BuildLevels(s, t))
        {
            for (int i = 0; i < _iter.Length; i++) _iter[i] = 0;
            long pushed;
            while ((pushed = Augment(s, t, long.MaxValue)) > 0) total += pushed;
        }
        return total;
    }

    private bool BuildLevels(int s, int t)
    {
        for (int i = 0; i < _level.Length; i++) _level[i] = -1;
        _level[s] = 0;
        var queue = new Queue<int>();
        queue.Enqueue(s);
        while (queue.Count > 0)
        {
            int u = queue.Dequeue();
            foreach (int id in _incident[u])
            {
                if (_residual[id] <= 0) continue;
                int v = _to[id];
                if (_level[v] >= 0) continue;
                _level[v] = _level[u] + 1;
                queue.Enqueue(v);
            }
        }
        return _level[t] >= 0;
    }

    private long Augment(int u, int t, long limit)
    {
        if (u == t) return limit;
        for (; _iter[u] < _incident[u].Count; _iter[u]++)
        {
            int id = _incident[u][_iter[u]];
            int v = _to[id];
            if (_residual[id] <= 0 || _level[v] != _level[u] + 1) continue;
            long pushed = Augment(v, t, Math.Min(limit, _residual[id]));
            if (pushed > 0)
            {
                _residual[id] -= pushed;
                _residual[id ^ 1] += pushed;
                return pushed;
            }
        }
        return 0;
    }
}

internal static class BoundedFlow
{
    /// <summary>edges[i] = {u, v, lower, upper}.</summary>
    internal static bool Solve(int vertices, long[][] edges, int s, int t, out long value, out long[] flow)
    {
        int m = edges.Length;
        var balance = new long[vertices];            // surplus created by pre-sending the lower bounds
        for (int i = 0; i < m; i++)
        {
            ValidateEdge(edges[i], vertices);
            balance[(int)edges[i][0]] -= edges[i][2];
            balance[(int)edges[i][1]] += edges[i][2];
        }
        if (s < 0 || s >= vertices || t < 0 || t >= vertices) throw new ArgumentOutOfRangeException(nameof(s));
        if (s == t) throw new ArgumentException("the source and the sink must differ");

        var g = new Dinic(vertices + 2);
        int superSource = vertices, superSink = vertices + 1;
        var ids = new int[m];
        long inf = 1;
        for (int i = 0; i < m; i++) inf += edges[i][3];
        for (int i = 0; i < m; i++) ids[i] = g.AddEdge((int)edges[i][0], (int)edges[i][1], edges[i][3] - edges[i][2]);
        int feedback = g.AddEdge(t, s, inf);

        // balance[v] = sum of the lower bounds entering v minus leaving v. A vertex with a positive
        // balance still needs extra inflow in the repair flow, so the super source feeds it; a
        // negative balance is a surplus that the super sink drains.
        long required = 0;
        var superEdges = new List<int>();
        for (int v = 0; v < vertices; v++)
        {
            if (balance[v] > 0)
            {
                superEdges.Add(g.AddEdge(superSource, v, balance[v]));    // feed the deficit
                required += balance[v];
            }
            else if (balance[v] < 0)
            {
                superEdges.Add(g.AddEdge(v, superSink, -balance[v]));     // drain the surplus
            }
        }
        flow = new long[m];
        if (g.MaxFlow(superSource, superSink) != required) { value = 0; return false; }

        // The repair network has done its job: retire the feedback and super edges so that the
        // remaining residual graph only contains moves between genuine flows of the instance.
        g.DisableEdge(feedback);
        foreach (int id in superEdges) g.DisableEdge(id);
        g.MaxFlow(s, t);                                  // then push the value as far as it goes
        for (int i = 0; i < m; i++) flow[i] = edges[i][2] + g.FlowOn(ids[i]);
        value = 0;
        for (int i = 0; i < m; i++)
        {
            if (edges[i][0] == s) value += flow[i];
            else if (edges[i][1] == s) value -= flow[i];
        }
        return true;
    }

    /// <summary>A circulation: every edge has a lower and an upper bound and every vertex conserves.</summary>
    internal static bool Circulation(int vertices, long[][] edges, out long[] flow)
    {
        int m = edges.Length;
        var balance = new long[vertices];
        for (int i = 0; i < m; i++)
        {
            ValidateEdge(edges[i], vertices);
            balance[(int)edges[i][0]] -= edges[i][2];
            balance[(int)edges[i][1]] += edges[i][2];
        }
        var g = new Dinic(vertices + 2);
        int superSource = vertices, superSink = vertices + 1;
        var ids = new int[m];
        long required = 0;
        for (int i = 0; i < m; i++) ids[i] = g.AddEdge((int)edges[i][0], (int)edges[i][1], edges[i][3] - edges[i][2]);
        for (int v = 0; v < vertices; v++)
        {
            if (balance[v] > 0) { g.AddEdge(superSource, v, balance[v]); required += balance[v]; }
            else if (balance[v] < 0) g.AddEdge(v, superSink, -balance[v]);
        }
        flow = new long[m];
        if (g.MaxFlow(superSource, superSink) != required) return false;
        for (int i = 0; i < m; i++) flow[i] = edges[i][2] + g.FlowOn(ids[i]);
        return true;
    }

    private static void ValidateEdge(long[] edge, int vertices)
    {
        if (edge.Length != 4) throw new ArgumentException("an edge is described by four numbers");
        if (edge[0] < 0 || edge[0] >= vertices || edge[1] < 0 || edge[1] >= vertices)
            throw new ArgumentOutOfRangeException(nameof(edge), "edge endpoint outside the graph");
        if (edge[0] == edge[1]) throw new ArgumentException("self loops are not supported");
        if (edge[2] < 0 || edge[3] < edge[2]) throw new ArgumentException("bounds must satisfy 0 <= lower <= upper");
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static long[][] Edges(params long[][] edges) => edges;

    private static long[] E(int u, int v, long lo, long hi) => new[] { (long)u, (long)v, lo, hi };

    private static string Describe(long[][] edges)
    {
        var parts = new List<string>();
        foreach (long[] e in edges) parts.Add("(" + e[0] + "->" + e[1] + " [" + e[2] + "," + e[3] + "])");
        return string.Join(" ", parts);
    }

    /// <summary>Verifies bounds and conservation of a circulation; returns null when it is valid.</summary>
    private static string? CirculationViolation(int vertices, long[][] edges, long[] flow)
    {
        for (int i = 0; i < edges.Length; i++)
        {
            if (flow[i] < edges[i][2] || flow[i] > edges[i][3])
                return "edge " + i + " carries " + flow[i] + " outside [" + edges[i][2] + "," + edges[i][3] + "]";
        }
        var net = new long[vertices];
        for (int i = 0; i < edges.Length; i++)
        {
            net[(int)edges[i][0]] -= flow[i];
            net[(int)edges[i][1]] += flow[i];
        }
        for (int v = 0; v < vertices; v++) if (net[v] != 0) return "vertex " + v + " leaks " + net[v];
        return null;
    }

    /// <summary>Verifies bounds and conservation; returns null when the flow is valid.</summary>
    private static string? FlowViolation(int vertices, long[][] edges, long[] flow, int s, int t, long value)
    {
        for (int i = 0; i < edges.Length; i++)
        {
            if (flow[i] < edges[i][2] || flow[i] > edges[i][3])
                return "edge " + i + " carries " + flow[i] + " outside [" + edges[i][2] + "," + edges[i][3] + "]";
        }
        var net = new long[vertices];
        for (int i = 0; i < edges.Length; i++)
        {
            net[(int)edges[i][0]] -= flow[i];
            net[(int)edges[i][1]] += flow[i];
        }
        for (int v = 0; v < vertices; v++)
        {
            if (v == s || v == t) continue;
            if (net[v] != 0) return "vertex " + v + " leaks " + net[v];
        }
        if (net[t] != value) return "declared value " + value + " but the sink receives " + net[t];
        if (net[s] != -value) return "declared value " + value + " but the source loses " + -net[s];
        return null;
    }

    /// <summary>Exhaustive solver for tiny instances: every integral flow vector in the box.</summary>
    private static long BruteForceBest(int vertices, long[][] edges, int s, int t)
    {
        var flow = new long[edges.Length];
        long best = long.MinValue;
        Enumerate(0);
        return best;

        void Enumerate(int index)
        {
            if (index == edges.Length)
            {
                var net = new long[vertices];
                for (int i = 0; i < edges.Length; i++)
                {
                    net[(int)edges[i][0]] -= flow[i];
                    net[(int)edges[i][1]] += flow[i];
                }
                for (int v = 0; v < vertices; v++)
                {
                    if (v == s || v == t) continue;
                    if (net[v] != 0) return;
                }
                if (net[t] < 0) return;         // the value must be non-negative: this is a flow s -> t
                if (net[t] > best) best = net[t];
                return;
            }
            for (long f = edges[index][2]; f <= edges[index][3]; f++)
            {
                flow[index] = f;
                Enumerate(index + 1);
            }
        }
    }

    /// <summary>Brute-force minimum s-t cut, used as an independent oracle for zero bounds.</summary>
    private static long MinCut(int vertices, long[][] edges, int s, int t)
    {
        long best = long.MaxValue;
        for (int mask = 0; mask < (1 << vertices); mask++)
        {
            if ((mask & (1 << s)) == 0 || (mask & (1 << t)) != 0) continue;
            long cut = 0;
            for (int i = 0; i < edges.Length; i++)
            {
                bool inU = (mask & (1 << (int)edges[i][0])) != 0;
                bool inV = (mask & (1 << (int)edges[i][1])) != 0;
                if (inU && !inV) cut += edges[i][3];
            }
            if (cut < best) best = cut;
        }
        return best == long.MaxValue ? 0 : best;
    }

    public static void Main()
    {
        // ---- a single lower-bounded edge ------------------------------------------------------
        {
            var edges = Edges(E(0, 1, 1, 5));
            Check(BoundedFlow.Solve(2, edges, 0, 1, out long value, out long[] flow), "one bounded edge is feasible");
            Check(value == 5, "the maximised value is the upper bound 5 (" + value + ")");
            Check(FlowViolation(2, edges, flow, 0, 1, value) == null,
                "the flow respects bounds, conservation and the declared value");
            Check(flow[0] == 5, "the single edge carries the whole value");
        }

        // ---- the lower bounds force a minimum value -------------------------------------------
        {
            // 0 -> 1 must carry at least 4 and 1 -> 2 at most 7, so the value is pushed to 7.
            var edges = Edges(E(0, 1, 4, 9), E(1, 2, 0, 7));
            Check(BoundedFlow.Solve(3, edges, 0, 2, out long value, out long[] flow), "the chain is feasible");
            Check(value == 7, "the bottleneck caps the maximised value at 7 (" + value + ")");
            Check(flow[0] == 7 && flow[1] == 7, "both edges carry 7 after maximisation");
            Check(FlowViolation(3, edges, flow, 0, 2, value) == null, "the chain flow is valid");

            var infeasible = Edges(E(0, 1, 5, 5), E(1, 2, 0, 3));
            Check(!BoundedFlow.Solve(3, infeasible, 0, 2, out _, out _),
                "a lower bound that cannot be drained is reported as infeasible");
            var pinned = Edges(E(0, 1, 3, 3), E(1, 2, 3, 3));
            Check(BoundedFlow.Solve(3, pinned, 0, 2, out long pinnedValue, out long[] pinnedFlow) && pinnedValue == 3,
                "ties between lower and upper bounds pin the flow exactly");
            Check(pinnedFlow[0] == 3 && pinnedFlow[1] == 3, "a fixed edge carries its one allowed value");
        }

        // ---- circulations ---------------------------------------------------------------------
        {
            var triangle = Edges(E(0, 1, 2, 4), E(1, 2, 2, 4), E(2, 0, 2, 4));
            Check(BoundedFlow.Circulation(3, triangle, out long[] circulation), "the lower-bounded triangle admits a circulation");
            Check(circulation[0] == 2 && circulation[1] == 2 && circulation[2] == 2,
                "the minimal circulation sends 2 around the loop");
            Check(CirculationViolation(3, triangle, circulation) == null,
                "the circulation conserves at every vertex");

            var broken = Edges(E(0, 1, 3, 3), E(1, 0, 0, 1));
            Check(!BoundedFlow.Circulation(2, broken, out _), "a circulation that cannot conserve is rejected");

            var empty = Array.Empty<long[]>();
            Check(BoundedFlow.Circulation(4, empty, out long[] none) && none.Length == 0, "a graph without edges has a trivial circulation");

            var selfConsistent = Edges(E(0, 1, 1, 2), E(1, 2, 1, 2), E(2, 0, 1, 2), E(0, 2, 0, 1));
            Check(BoundedFlow.Circulation(3, selfConsistent, out long[] withSpare), "an extra edge keeps the circulation feasible");
            Check(CirculationViolation(3, selfConsistent, withSpare) == null,
                "the circulation with an extra edge is still conserved");

            var supplies = Edges(E(0, 1, 4, 6), E(1, 2, 0, 6), E(2, 0, 3, 6));
            Check(BoundedFlow.Circulation(3, supplies, out long[] varied), "a circulation can route different amounts per edge");
            Check(CirculationViolation(3, supplies, varied) == null, "that circulation is conserved");
            Check(varied[0] == 4 && varied[1] == 4 && varied[2] == 4,
                "the tightest lower bound fixes the whole cycle at 4");
        }

        // ---- tiny instances versus exhaustive enumeration -------------------------------------
        {
            const int Instances = 60;
            ulong state = 0xB0BAF00DUL;
            bool agree = true, feasibleFlows = true;
            string firstFailure = "none";
            int feasible = 0, infeasible = 0;
            for (int t = 0; t < Instances; t++)
            {
                const int Vertices = 4;
                int m = 3 + Rng.NextInt(ref state, 3);
                var edges = new long[m][];
                for (int i = 0; i < m; i++)
                {
                    int u = Rng.NextInt(ref state, Vertices);
                    int v = Rng.NextInt(ref state, Vertices);
                    if (u == v) v = (v + 1) % Vertices;
                    long lo = Rng.NextInt(ref state, 3);
                    long hi = lo + Rng.NextInt(ref state, 3);
                    edges[i] = E(u, v, lo, hi);
                }
                bool ok = BoundedFlow.Solve(Vertices, edges, 0, Vertices - 1, out long value, out long[] flow);
                long brute = BruteForceBest(Vertices, edges, 0, Vertices - 1);
                if (!ok)
                {
                    infeasible++;
                    if (brute != long.MinValue)
                    {
                        agree = false;
                        firstFailure = "instance " + t + " wrongly infeasible: " + Describe(edges) + " brute=" + brute;
                    }
                    continue;
                }
                feasible++;
                if (brute == long.MinValue) { agree = false; firstFailure = "instance " + t + " wrongly feasible"; }
                else if (value != brute)
                {
                    agree = false;
                    firstFailure = "instance " + t + " value " + value + " but exhaustive search says " + brute;
                }
                string? violation = FlowViolation(Vertices, edges, flow, 0, Vertices - 1, value);
                if (violation != null) { feasibleFlows = false; firstFailure = "instance " + t + ": " + violation; }
            }
            Check(agree, "every tiny instance matches exhaustive enumeration (" + firstFailure + ")");
            Check(feasibleFlows, "every produced flow is within bounds and conserves (" + firstFailure + ")");
            Check(feasible >= 5 && infeasible >= 5,
                "the workload exercises both outcomes: " + feasible + " feasible, " + infeasible + " infeasible");
            Console.WriteLine("tinyInstances=" + Instances + " feasible=" + feasible + " infeasible=" + infeasible);
        }

        // ---- zero lower bounds versus the brute-force minimum cut -----------------------------
        {
            const int Vertices = 7, Instances = 30;
            ulong state = 0xABCDEF12UL;
            bool agree = true;
            string firstFailure = "none";
            for (int t = 0; t < Instances; t++)
            {
                int m = 6 + Rng.NextInt(ref state, 5);
                var edges = new long[m][];
                for (int i = 0; i < m; i++)
                {
                    int u = Rng.NextInt(ref state, Vertices);
                    int v = Rng.NextInt(ref state, Vertices);
                    if (u == v) v = (v + 1) % Vertices;
                    edges[i] = E(u, v, 0, 1 + Rng.NextInt(ref state, 6));
                }
                bool ok = BoundedFlow.Solve(Vertices, edges, 0, Vertices - 1, out long value, out long[] flow);
                Check(ok, "with zero lower bounds the instance is always feasible");
                long cut = MinCut(Vertices, edges, 0, Vertices - 1);
                if (value != cut)
                {
                    agree = false;
                    if (firstFailure == "none") firstFailure = "instance " + t + ": flow " + value + " vs min cut " + cut;
                }
                string? violation = FlowViolation(Vertices, edges, flow, 0, Vertices - 1, value);
                if (violation != null && firstFailure == "none") { agree = false; firstFailure = violation; }
            }
            Check(agree, "max flow equals the brute-force min cut on all " + Instances + " instances (" + firstFailure + ")");
            Console.WriteLine("minCutInstances=" + Instances);
        }

        // ---- rejected input and boundary behaviour --------------------------------------------
        {
            var fine = Edges(E(0, 1, 0, 4));
            bool thrown = false;
            try { BoundedFlow.Solve(2, fine, 0, 0, out _, out _); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a source equal to the sink is rejected");
            thrown = false;
            try { BoundedFlow.Solve(2, fine, 0, 5, out _, out _); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a sink outside the graph is rejected");
            thrown = false;
            try { BoundedFlow.Solve(2, Edges(E(0, 1, 4, 2)), 0, 1, out _, out _); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "an upper bound below the lower bound is rejected");
            thrown = false;
            try { BoundedFlow.Solve(2, Edges(E(1, 2, 0, 1)), 0, 1, out _, out _); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an edge endpoint outside the graph is rejected");
            thrown = false;
            try { BoundedFlow.Circulation(2, Edges(E(0, 1, -1, 2)), out _); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a negative lower bound is rejected");

            var unreachable = Edges(E(0, 1, 0, 9), E(2, 3, 0, 9));
            Check(BoundedFlow.Solve(4, unreachable, 0, 3, out long zero, out long[] zeroFlow) && zero == 0,
                "with no path from source to sink the value is zero");
            Check(FlowViolation(4, unreachable, zeroFlow, 0, 3, zero) == null, "the zero flow is still legal");

            var zeroEdge = Edges(E(0, 1, 0, 0), E(1, 2, 0, 3));
            Check(BoundedFlow.Solve(3, zeroEdge, 0, 2, out long blocked, out long[] blockedFlow) && blocked == 0,
                "a zero-capacity edge blocks the flow completely");
            Check(FlowViolation(3, zeroEdge, blockedFlow, 0, 2, blocked) == null, "the blocked flow is legal");

            var parallel = Edges(E(0, 1, 1, 2), E(0, 1, 2, 5), E(1, 2, 0, 5));
            Check(BoundedFlow.Solve(3, parallel, 0, 2, out long both, out long[] bothFlow) && both == 5,
                "parallel edges add their capacities (" + both + ")");
            Check(FlowViolation(3, parallel, bothFlow, 0, 2, both) == null, "the parallel-edge flow is legal");

            var deterministic = Edges(E(0, 1, 2, 6), E(1, 2, 1, 4), E(0, 2, 1, 3));
            Check(BoundedFlow.Solve(3, deterministic, 0, 2, out long first, out long[] firstFlow), "the fixture is feasible");
            Check(BoundedFlow.Solve(3, deterministic, 0, 2, out long second, out long[] secondFlow), "the fixture stays feasible");
            bool same = first == second && firstFlow.Length == secondFlow.Length;
            for (int i = 0; i < firstFlow.Length && same; i++) if (firstFlow[i] != secondFlow[i]) same = false;
            Check(same, "the solver is deterministic");
            Console.WriteLine("deterministicValue=" + first + " parallelEdges=" + both);
        }

        Console.WriteLine("PASS");
    }
}
