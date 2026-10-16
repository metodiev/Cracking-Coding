// Task 705 - Implement Steiner forest approximations large graphs.
// (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
//
// A Steiner forest instance is a weighted graph plus a list of terminal pairs ("demands"); the goal
// is the cheapest edge set in which every pair is connected - the demands may share a common trunk
// or be served by separate trees, so the optimum is *not* a single Steiner tree. Steiner forest is
// NP-hard, so this file implements the classical greedy shortest-path heuristic (Takahashi &
// Matsuyama generalised to demands): process the demands in order; whenever a pair is not yet
// connected by the chosen edges, run Dijkstra where edges already chosen cost zero and everything
// else costs its weight, and add that path. Reusing chosen edges for free is what produces sharing:
// the second demand of a "shared trunk" instance only pays for the last mile.
//
// Verified below against two independent references: (1) exhaustive enumeration of all edge subsets
// on small random instances gives the exact Steiner forest optimum, so the heuristic must never beat
// it and must stay within the documented factor of it, and (2) a "no sharing" reference that sums
// the demands' individual shortest paths, which the heuristic can never exceed because each residual
// path costs no more than the shortest path in the original graph. Feasibility (every demand pair
// connected by chosen edges), the zero-cost boundary cases, unreachable and malformed demands, and
// the sharing gain on a hand-built instance are all checked too.
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

internal sealed class WeightedGraph
{
    internal readonly int Vertices;
    private readonly List<int>[] _incident;
    private readonly List<int> _edgeU = new List<int>(), _edgeV = new List<int>(), _edgeW = new List<int>();

    internal WeightedGraph(int vertices)
    {
        Vertices = vertices;
        _incident = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) _incident[i] = new List<int>();
    }

    internal int EdgeCount => _edgeU.Count;
    internal int EndU(int edge) => _edgeU[edge];
    internal int EndV(int edge) => _edgeV[edge];
    internal int Weight(int edge) => _edgeW[edge];
    internal List<int> Incident(int vertex) => _incident[vertex];

    internal int AddEdge(int u, int v, int w)
    {
        if (u == v) throw new ArgumentException("self loops are meaningless for Steiner instances");
        if (w < 1) throw new ArgumentException("edge weights must be positive");
        int id = _edgeU.Count;
        _edgeU.Add(u);
        _edgeV.Add(v);
        _edgeW.Add(w);
        _incident[u].Add(id);
        _incident[v].Add(id);
        return id;
    }
}

internal sealed class Dsu
{
    private readonly int[] _parent;
    private readonly int[] _size;

    internal Dsu(int count)
    {
        _parent = new int[count];
        _size = new int[count];
        for (int i = 0; i < count; i++) { _parent[i] = i; _size[i] = 1; }
        Components = count;
    }

    internal int Components { get; private set; }

    internal int Find(int x)
    {
        while (_parent[x] != x) { _parent[x] = _parent[_parent[x]]; x = _parent[x]; }
        return x;
    }

    internal bool Union(int a, int b)
    {
        int ra = Find(a), rb = Find(b);
        if (ra == rb) return false;
        if (_size[ra] < _size[rb]) { int swap = ra; ra = rb; rb = swap; }
        _parent[rb] = ra;
        _size[ra] += _size[rb];
        Components--;
        return true;
    }

    internal bool Connected(int a, int b) => Find(a) == Find(b);
}

internal static class Steiner
{
    /// <summary>
    /// Dijkstra on residual costs (0 for an already chosen edge), returning the edge ids of the path
    /// from source to target, or null when the target is unreachable.
    /// </summary>
    private static List<int>? ShortestPath(WeightedGraph g, int source, int target, bool[] used)
    {
        int n = g.Vertices;
        var dist = new long[n];
        var viaEdge = new int[n];
        var done = new bool[n];
        for (int i = 0; i < n; i++) { dist[i] = long.MaxValue; viaEdge[i] = -1; }
        dist[source] = 0;
        for (int step = 0; step < n; step++)
        {
            int best = -1;
            for (int v = 0; v < n; v++) if (!done[v] && (best == -1 || dist[v] < dist[best])) best = v;
            if (best == -1 || dist[best] == long.MaxValue) break;
            done[best] = true;
            foreach (int e in g.Incident(best))
            {
                int other = g.EndU(e) == best ? g.EndV(e) : g.EndU(e);
                long cost = used[e] ? 0 : g.Weight(e);
                if (dist[best] + cost < dist[other]) { dist[other] = dist[best] + cost; viaEdge[other] = e; }
            }
        }
        if (dist[target] == long.MaxValue) return null;
        var path = new List<int>();
        for (int cur = target; cur != source; )
        {
            int e = viaEdge[cur];
            path.Add(e);
            cur = g.EndU(e) == cur ? g.EndV(e) : g.EndU(e);
        }
        path.Reverse();
        return path;
    }

    /// <summary>The greedy Steiner forest: each unconnected demand adds its residual shortest path.</summary>
    internal static long Forest(WeightedGraph g, IReadOnlyList<(int u, int v)> demands, out bool[] used)
    {
        used = new bool[g.EdgeCount];
        var dsu = new Dsu(g.Vertices);
        long cost = 0;
        foreach (var (u, v) in demands)
        {
            if (u < 0 || u >= g.Vertices || v < 0 || v >= g.Vertices)
                throw new ArgumentOutOfRangeException(nameof(demands), "demand mentions a vertex outside the graph");
            if (dsu.Connected(u, v)) continue;
            List<int>? path = ShortestPath(g, u, v, used);
            if (path == null) throw new InvalidOperationException("demand " + u + "-" + v + " cannot be satisfied");
            foreach (int e in path)
            {
                if (!used[e]) { used[e] = true; cost += g.Weight(e); }
                dsu.Union(g.EndU(e), g.EndV(e));
            }
        }
        return cost;
    }

    /// <summary>Checks that every demand is connected by the chosen edges.</summary>
    internal static bool Feasible(WeightedGraph g, bool[] used, IReadOnlyList<(int u, int v)> demands)
    {
        var dsu = new Dsu(g.Vertices);
        for (int e = 0; e < g.EdgeCount; e++) if (used[e]) dsu.Union(g.EndU(e), g.EndV(e));
        foreach (var (u, v) in demands) if (!dsu.Connected(u, v)) return false;
        return true;
    }

    /// <summary>Reference: sum of the demands' shortest paths when nothing is shared.</summary>
    internal static long IndependentShortestPaths(WeightedGraph g, IReadOnlyList<(int u, int v)> demands)
    {
        long total = 0;
        foreach (var (u, v) in demands)
        {
            var none = new bool[g.EdgeCount];
            List<int>? path = ShortestPath(g, u, v, none);
            foreach (int e in path!) total += g.Weight(e);
        }
        return total;
    }

    /// <summary>Exact optimum by enumerating every edge subset (only for tiny instances).</summary>
    internal static long ExactForest(WeightedGraph g, IReadOnlyList<(int u, int v)> demands)
    {
        int edges = g.EdgeCount;
        long best = long.MaxValue;
        for (int mask = 0; mask < (1 << edges); mask++)
        {
            long cost = 0;
            for (int e = 0; e < edges; e++) if ((mask & (1 << e)) != 0) cost += g.Weight(e);
            if (cost >= best) continue;
            var used = new bool[edges];
            for (int e = 0; e < edges; e++) used[e] = (mask & (1 << e)) != 0;
            if (Feasible(g, used, demands)) best = cost;
        }
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // ---- a hand-built sharing instance ----------------------------------------------------
        {
            // Terminals 3, 4 and 5 hang off the shared trunk 1-2; vertex 0 is a decoy joined by an
            // expensive edge to 4. Both demands must cross the trunk, so the second one only pays
            // for its own last mile.
            var g = new WeightedGraph(6);
            int toSpur3 = g.AddEdge(1, 3, 3);
            int trunk = g.AddEdge(1, 2, 4);
            int toSpur4 = g.AddEdge(2, 4, 3);
            int toSpur5 = g.AddEdge(2, 5, 9);
            int decoy0 = g.AddEdge(0, 1, 4);
            int expensive = g.AddEdge(0, 4, 30);
            var demands = new List<(int u, int v)> { (3, 4), (3, 5) };
            long cost = Steiner.Forest(g, demands, out bool[] used);
            Check(Steiner.Feasible(g, used, demands), "the hand fixture forest connects both demands");
            Check(cost == 3 + 4 + 3 + 9, "the greedy heuristic pays 19 for the shared trunk (got " + cost + ")");
            Check(used[trunk], "the trunk edge is shared by both demands");
            Check(used[toSpur3] && used[toSpur4] && used[toSpur5], "all three spurs are used");
            Check(!used[decoy0] && !used[expensive], "unnecessary edges stay unselected");

            long noSharing = Steiner.IndependentShortestPaths(g, demands);
            Check(noSharing == (3 + 4 + 3) + (3 + 4 + 9),
                "the independent reference is 26 (got " + noSharing + ")");
            long exact = Steiner.ExactForest(g, demands);
            Check(exact == 19, "the exact optimum for the fixture is 19 (got " + exact + ")");
            Check(cost == exact, "the heuristic is optimal on the hand-built sharing instance");
            Check(cost < noSharing, "sharing beats serving each demand with its own shortest path ("
                + cost + " < " + noSharing + ", gain " + (noSharing - cost) + ")");
            Console.WriteLine("fixtureCost=" + cost + " exact=" + exact + " independentSum=" + noSharing
                + " sharingGain=" + (noSharing - cost));
        }

        // ---- small random instances against exhaustive subset search --------------------------
        {
            const int Instances = 40;
            ulong state = 0x51A1B0C4UL;
            bool feasible = true, neverBeatsOptimum = true, withinFactor = true, withinNoSharing = true;
            string firstFailure = "none";
            double worstRatio = 1.0;

            for (int t = 0; t < Instances; t++)
            {
                int vertices = 5 + Rng.NextInt(ref state, 3);
                var g = new WeightedGraph(vertices);
                int edges = 3 + Rng.NextInt(ref state, 6);
                var present = new HashSet<(int, int)>();
                for (int e = 0; e < edges; e++)
                {
                    int u = Rng.NextInt(ref state, vertices);
                    int v = Rng.NextInt(ref state, vertices);
                    if (u == v) { e--; continue; }
                    var key = (Math.Min(u, v), Math.Max(u, v));
                    if (!present.Add(key)) { e--; continue; }
                    g.AddEdge(u, v, 1 + Rng.NextInt(ref state, 9));
                }
                var demands = new List<(int u, int v)>();
                int demandCount = 1 + Rng.NextInt(ref state, 3);
                for (int d = 0; d < demandCount; d++)
                {
                    int u = Rng.NextInt(ref state, vertices);
                    int v = Rng.NextInt(ref state, vertices);
                    demands.Add((u, v));
                }

                long exact = Steiner.ExactForest(g, demands);
                if (exact == long.MaxValue) continue;                  // demands unreachable: skipped by design
                bool[] chosen;
                long cost;
                try { cost = Steiner.Forest(g, demands, out chosen); }
                catch (InvalidOperationException) { continue; }
                if (!Steiner.Feasible(g, chosen, demands)) { feasible = false; firstFailure = "instance " + t + " infeasible"; }
                if (cost < exact) { neverBeatsOptimum = false; firstFailure = "instance " + t + " beat the optimum"; }
                if (cost > 2 * exact) { withinFactor = false; firstFailure = "instance " + t + " exceeded 2x optimum"; }
                if (cost > Steiner.IndependentShortestPaths(g, demands))
                {
                    withinNoSharing = false;
                    firstFailure = "instance " + t + " cost more than the no-sharing reference";
                }
                if (exact > 0) worstRatio = Math.Max(worstRatio, (double)cost / exact);
            }
            Check(feasible, "every heuristic forest satisfies all demands (" + firstFailure + ")");
            Check(neverBeatsOptimum, "the heuristic never beats the exact optimum (" + firstFailure + ")");
            Check(withinFactor, "the heuristic stays within twice the optimum on these instances (" + firstFailure + ")");
            Check(withinNoSharing, "the heuristic never costs more than serving demands separately");
            Console.WriteLine("tinyInstances=" + Instances + " worstRatioToOptimum=" + worstRatio.ToString("F3"));
        }

        // ---- a larger instance where only the references are available ------------------------
        {
            const int Vertices = 150;
            var g = new WeightedGraph(Vertices);
            ulong state = 0xC0FFEE77UL;
            var present = new HashSet<(int, int)>();
            for (int i = 1; i < Vertices; i++)                       // a spanning path keeps everything reachable
            {
                int u = i - 1;
                g.AddEdge(u, i, 1 + Rng.NextInt(ref state, 9));
                present.Add((u, i));
            }
            for (int extra = 0; extra < 260; extra++)
            {
                int u = Rng.NextInt(ref state, Vertices);
                int v = Rng.NextInt(ref state, Vertices);
                if (u == v) continue;
                var key = (Math.Min(u, v), Math.Max(u, v));
                if (!present.Add(key)) continue;
                g.AddEdge(u, v, 10 + Rng.NextInt(ref state, 40));
            }
            var demands = new List<(int u, int v)>();
            for (int d = 0; d < 12; d++)
            {
                int u = Rng.NextInt(ref state, Vertices);
                int v = Rng.NextInt(ref state, Vertices);
                demands.Add((u, v));
            }
            long cost = Steiner.Forest(g, demands, out bool[] used);
            Check(Steiner.Feasible(g, used, demands), "all 12 demands are connected in the 150-vertex graph");
            long noSharing = Steiner.IndependentShortestPaths(g, demands);
            Check(cost <= noSharing, "the forest is at least as cheap as 12 independent shortest paths ("
                + cost + " <= " + noSharing + ")");
            int selected = 0;
            long recomputed = 0;
            for (int e = 0; e < g.EdgeCount; e++) if (used[e]) { selected++; recomputed += g.Weight(e); }
            Check(recomputed == cost, "the reported forest cost equals the sum of the chosen edge weights");
            Check(selected > 0 && selected < g.EdgeCount, "the chosen edges are a proper subset ("
                + selected + " of " + g.EdgeCount + ")");
            Check(cost > 0, "the instance is not trivial");

            long costAgain = Steiner.Forest(g, demands, out bool[] usedAgain);
            bool sameUsed = true;
            for (int e = 0; e < g.EdgeCount; e++) if (used[e] != usedAgain[e]) sameUsed = false;
            Check(costAgain == cost && sameUsed, "the heuristic is deterministic");
            Console.WriteLine("vertices=" + Vertices + " edges=" + g.EdgeCount + " demands=" + demands.Count
                + " forestCost=" + cost + " independentSum=" + noSharing + " selectedEdges=" + selected);
        }

        // ---- boundary and rejected inputs -----------------------------------------------------
        {
            var g = new WeightedGraph(4);
            g.AddEdge(0, 1, 5);
            g.AddEdge(1, 2, 5);
            var isolatedDemand = new List<(int u, int v)> { (0, 0) };
            Check(Steiner.Forest(g, isolatedDemand, out bool[] noneUsed) == 0, "a demand of a vertex with itself costs nothing");
            Check(Steiner.Feasible(g, noneUsed, isolatedDemand), "a degenerate demand is trivially satisfied");
            Check(Steiner.Forest(g, new List<(int, int)>(), out _) == 0, "no demands means an empty forest");

            var unreachable = new List<(int u, int v)> { (0, 3) };
            bool thrown = false;
            try { Steiner.Forest(g, unreachable, out _); }
            catch (InvalidOperationException) { thrown = true; }
            Check(thrown, "an unreachable demand is reported instead of returning a wrong cost");

            thrown = false;
            try { Steiner.Forest(g, new List<(int, int)> { (0, 7) }, out _); }
            catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a demand outside the graph is rejected");

            thrown = false;
            try { g.AddEdge(2, 2, 1); }
            catch (ArgumentException) { thrown = true; }
            Check(thrown, "a self loop is rejected");
            thrown = false;
            try { g.AddEdge(2, 3, 0); }
            catch (ArgumentException) { thrown = true; }
            Check(thrown, "a zero-weight edge is rejected");

            // one demand that ties with its alternative route: the heuristic must pick one of them
            var tie = new WeightedGraph(3);
            int direct = tie.AddEdge(0, 2, 6);
            tie.AddEdge(0, 1, 3);
            tie.AddEdge(1, 2, 3);
            long tieCost = Steiner.Forest(tie, new List<(int, int)> { (0, 2) }, out bool[] tieUsed);
            Check(tieCost == 6, "a single demand costs the shortest path (" + tieCost + ")");
            Check(Steiner.Feasible(tie, tieUsed, new List<(int, int)> { (0, 2) }),
                "the chosen edges really connect the demand");
            int tieEdges = 0;
            for (int e = 0; e < tie.EdgeCount; e++) if (tieUsed[e]) tieEdges++;
            Check(tieEdges == 1 || tieEdges == 2, "the tie is settled with one of the two cost-6 routes");
        }

        Console.WriteLine("PASS");
    }
}
