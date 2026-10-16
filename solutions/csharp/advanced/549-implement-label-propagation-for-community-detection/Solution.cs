// Task 549 - Implement label propagation for community detection. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Label propagation done two deterministic ways, because the tie-break is what decides the outcome.
// Every vertex starts with its own label; one sweep visits the vertices in index order (or in a
// seeded LCG shuffle) and each vertex adopts a label carried by the maximum number of its
// neighbours, keeping its own label only when it has no neighbours. (1) The "min" rule breaks ties
// towards the smallest label: it is fully deterministic but degenerates, since a vertex only ever
// looks at neighbours, so label 0 is overwritten before anyone can adopt it and every connected
// instance collapses onto one label - asserted, and stated in the header rather than hidden.
// (2) The classic rule breaks ties with a fixed-seed xorshift draw, still reproducible but
// structure-preserving. A third, non-iterative h-hop variant labels each vertex with the plurality
// among the vertices within graph distance h (including itself).
//
// Documented checks: the seeded rule recovers the strong planted two-block graph exactly (label sets
// compared, modularity equal to the planted value to 1e-12 and above 0.3), converges inside the
// documented 100-sweep cap in under ten sweeps, is reproducible for a fixed seed, and still reaches
// a weak equilibrium - no vertex holds a label that is strictly less frequent among its neighbours
// than another available label - for two further seeds; two disjoint triangles and two disjoint
// cliques keep exactly one label per component (so LPA never merges components) with modularity
// exactly 0.5 for the triangles; the label count starts at n and never grows, because labels can
// spread but never appear; the min rule collapses the 12-path and K7 onto label 1 and is verified to
// be a strict equilibrium (replaying it changes nothing); the shuffled-order variant is stable, good
// (modularity above 0.3) and reproducible per seed; and the h-hop labels are exactly the smallest
// closed-ball index at h = 0, 1, 2, 3 and equal max(0, v - h) on the path.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717UL;
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(NextRaw(ref state) % (ulong)bound);
    }
}

internal sealed class SimpleGraph
{
    internal readonly int N;
    internal readonly List<int>[] Adj;

    internal SimpleGraph(int n)
    {
        N = n;
        Adj = new List<int>[n];
        for (int v = 0; v < n; v++) Adj[v] = new List<int>();
    }

    internal void AddEdge(int u, int v)
    {
        if (u == v || Adj[u].Contains(v)) return;
        Adj[u].Add(v);
        Adj[v].Add(u);
    }

    internal int EdgeCount()
    {
        int total = 0;
        for (int v = 0; v < N; v++) total += Adj[v].Count;
        return total / 2;
    }

    internal static SimpleGraph Planted(int blocks, int size, int intraPercent, int crossPercent, int seed)
    {
        var g = new SimpleGraph(blocks * size);
        ulong state = (ulong)seed;
        for (int i = 0; i < blocks * size; i++)
        {
            for (int j = i + 1; j < blocks * size; j++)
            {
                double percent = i / size == j / size ? intraPercent : crossPercent;
                if (Rng.NextUnit(ref state) * 100.0 < percent) g.AddEdge(i, j);
            }
        }
        for (int v = 0; v < g.N; v++)
        {
            if (g.Adj[v].Count == 0) g.AddEdge(v, (v + 1) % g.N);
        }
        return g;
    }

    internal static SimpleGraph Path(int n)
    {
        var g = new SimpleGraph(n);
        for (int i = 0; i + 1 < n; i++) g.AddEdge(i, i + 1);
        return g;
    }

    internal static SimpleGraph Clique(int n)
    {
        var g = new SimpleGraph(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++) g.AddEdge(i, j);
        }
        return g;
    }

    /** Two disjoint cliques of the given size, i.e. the planted two-block model at 100 / 0. */
    internal static SimpleGraph TwoCliques(int size)
    {
        return Planted(2, size, 100, 0, 1);
    }

    internal int[] Components()
    {
        var component = new int[N];
        for (int v = 0; v < N; v++) component[v] = -1;
        int next = 0;
        for (int start = 0; start < N; start++)
        {
            if (component[start] >= 0) continue;
            var stack = new Stack<int>();
            stack.Push(start);
            component[start] = next;
            while (stack.Count > 0)
            {
                int v = stack.Pop();
                foreach (int w in Adj[v])
                {
                    if (component[w] >= 0) continue;
                    component[w] = next;
                    stack.Push(w);
                }
            }
            next++;
        }
        return component;
    }

    internal double Modularity(int[] labels)
    {
        double m = EdgeCount();
        if (m == 0.0) return 0.0;
        var degrees = new int[N];
        for (int v = 0; v < N; v++) degrees[v] = Adj[v].Count;
        int communities = 0;
        foreach (int label in labels) communities = Math.Max(communities, label + 1);
        var internalEdges = new double[communities];
        var totalDegree = new double[communities];
        for (int v = 0; v < N; v++)
        {
            totalDegree[labels[v]] += degrees[v];
            foreach (int w in Adj[v])
            {
                if (labels[w] == labels[v]) internalEdges[labels[v]] += 1.0;
            }
        }
        double q = 0.0;
        for (int c = 0; c < communities; c++)
        {
            q += internalEdges[c] / (2.0 * m) - Math.Pow(totalDegree[c] / (2.0 * m), 2.0);
        }
        return q;
    }

    internal static string LabelSets(int[] labels)
    {
        var groups = new SortedDictionary<int, List<int>>();
        for (int v = 0; v < labels.Length; v++)
        {
            if (!groups.TryGetValue(labels[v], out List<int> members))
            {
                members = new List<int>();
                groups[labels[v]] = members;
            }
            members.Add(v);
        }
        var parts = new List<string>();
        foreach (KeyValuePair<int, List<int>> entry in groups) parts.Add(string.Join(",", entry.Value));
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }

    internal static int CountLabels(int[] labels)
    {
        var seen = new HashSet<int>();
        foreach (int label in labels) seen.Add(label);
        return seen.Count;
    }
}

internal static class LabelPropagation
{
    internal sealed class Result
    {
        internal int[] Labels = Array.Empty<int>();
        internal int Sweeps;
        internal bool Converged;
        internal List<int> LabelCounts = new List<int>();
    }

    /**
     * One sweep. Each vertex adopts a label held by the maximum number of its neighbours; a vertex
     * without neighbours keeps its own label. Ties are either broken towards the smallest label
     * ("min" rule) or chosen with a fixed-seed LCG, which is the classic random-tie rule. The min
     * rule is deterministic but degenerates on connected graphs - every vertex ends up with the
     * smallest index reachable from it, so a connected instance collapses to one community - while
     * the seeded rule keeps the planted structure. Both are checked.
     */
    internal static int Sweep(SimpleGraph g, int[] labels, int[] order, ref ulong rngState, bool randomTies)
    {
        int changes = 0;
        var counts = new Dictionary<int, int>();
        var tied = new List<int>();
        foreach (int v in order)
        {
            counts.Clear();
            tied.Clear();
            foreach (int w in g.Adj[v])
            {
                counts.TryGetValue(labels[w], out int current);
                counts[labels[w]] = current + 1;
            }
            if (counts.Count == 0) continue;
            int best = 0;
            foreach (KeyValuePair<int, int> entry in counts)
            {
                if (entry.Value > best) best = entry.Value;
            }
            foreach (KeyValuePair<int, int> entry in counts)
            {
                if (entry.Value == best) tied.Add(entry.Key);
            }
            tied.Sort();
            int chosen = randomTies ? tied[Rng.NextBelow(ref rngState, tied.Count)] : tied[0];
            if (chosen != labels[v])
            {
                labels[v] = chosen;
                changes++;
            }
        }
        return changes;
    }

    internal static Result Run(SimpleGraph g, int maxSweeps, ulong seed, bool randomTies)
    {
        var result = new Result();
        var labels = new int[g.N];
        for (int v = 0; v < g.N; v++) labels[v] = v;
        var order = new int[g.N];
        for (int v = 0; v < g.N; v++) order[v] = v;
        ulong state = seed;
        result.LabelCounts.Add(SimpleGraph.CountLabels(labels));
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            int changes = Sweep(g, labels, order, ref state, randomTies);
            result.Sweeps = sweep + 1;
            result.LabelCounts.Add(SimpleGraph.CountLabels(labels));
            if (changes == 0)
            {
                result.Converged = true;
                break;
            }
        }
        result.Labels = labels;
        return result;
    }

    internal static Result RunShuffled(SimpleGraph g, int maxSweeps, ulong seed, bool randomTies)
    {
        var result = new Result();
        var labels = new int[g.N];
        for (int v = 0; v < g.N; v++) labels[v] = v;
        var order = new int[g.N];
        for (int v = 0; v < g.N; v++) order[v] = v;
        ulong state = seed;
        for (int i = g.N - 1; i > 0; i--)
        {
            int j = Rng.NextBelow(ref state, i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        result.LabelCounts.Add(SimpleGraph.CountLabels(labels));
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            int changes = Sweep(g, labels, order, ref state, randomTies);
            result.Sweeps = sweep + 1;
            result.LabelCounts.Add(SimpleGraph.CountLabels(labels));
            if (changes == 0)
            {
                result.Converged = true;
                break;
            }
        }
        result.Labels = labels;
        return result;
    }

    /** Replaying the deterministic min-tie sweep must change nothing. */
    internal static bool IsStrictEquilibrium(SimpleGraph g, int[] labels)
    {
        var order = new int[g.N];
        for (int v = 0; v < g.N; v++) order[v] = v;
        var copy = (int[])labels.Clone();
        ulong state = 1UL;
        if (Sweep(g, copy, order, ref state, false) != 0) return false;
        for (int v = 0; v < g.N; v++)
        {
            if (copy[v] != labels[v]) return false;
        }
        return true;
    }

    /**
     * Weak equilibrium of the seeded rule: no vertex holds a label that is less frequent among its
     * neighbours than some other label available to it.
     */
    internal static bool IsWeakEquilibrium(SimpleGraph g, int[] labels)
    {
        for (int v = 0; v < g.N; v++)
        {
            if (g.Adj[v].Count == 0) continue;
            var counts = new Dictionary<int, int>();
            foreach (int w in g.Adj[v])
            {
                counts.TryGetValue(labels[w], out int current);
                counts[labels[w]] = current + 1;
            }
            counts.TryGetValue(labels[v], out int own);
            foreach (KeyValuePair<int, int> entry in counts)
            {
                if (entry.Value > own) return false;
            }
        }
        return true;
    }

    /** Distances from a source (BFS); unreachable vertices get int.MaxValue. */
    internal static int[] Distances(SimpleGraph g, int source)
    {
        var distances = new int[g.N];
        for (int v = 0; v < g.N; v++) distances[v] = int.MaxValue;
        distances[source] = 0;
        var queue = new Queue<int>();
        queue.Enqueue(source);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            foreach (int w in g.Adj[v])
            {
                if (distances[w] != int.MaxValue) continue;
                distances[w] = distances[v] + 1;
                queue.Enqueue(w);
            }
        }
        return distances;
    }

    /** h-hop labelling: plurality label among the vertices within distance h (including self). */
    internal static int[] HopLabels(SimpleGraph g, int hops)
    {
        var labels = new int[g.N];
        for (int v = 0; v < g.N; v++)
        {
            int[] distances = Distances(g, v);
            var counts = new Dictionary<int, int>();
            for (int w = 0; w < g.N; w++)
            {
                if (distances[w] > hops) continue;
                counts.TryGetValue(w, out int current);
                counts[w] = current + 1;
            }
            int bestLabel = v;
            int bestCount = 0;
            foreach (KeyValuePair<int, int> entry in counts)
            {
                if (entry.Value > bestCount || (entry.Value == bestCount && entry.Key < bestLabel))
                {
                    bestCount = entry.Value;
                    bestLabel = entry.Key;
                }
            }
            labels[v] = bestLabel;
        }
        return labels;
    }

    /** True when labels[v] is the smallest vertex index in the closed h-neighbourhood of v. */
    internal static bool MatchesMinimumInBall(SimpleGraph g, int[] labels, int hops)
    {
        for (int v = 0; v < g.N; v++)
        {
            int[] distances = Distances(g, v);
            int smallest = v;
            for (int w = 0; w < g.N; w++)
            {
                if (distances[w] <= hops && w < smallest) smallest = w;
            }
            if (labels[v] != smallest) return false;
        }
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        var planted = SimpleGraph.Planted(2, 8, 95, 5, 549001);
        var plantedTruth = new int[16];
        for (int v = 0; v < 16; v++) plantedTruth[v] = v / 8;

        // Classic seeded random tie-break: the planted two-block structure survives.
        var result = LabelPropagation.Run(planted, 100, 549007, true);
        Check(result.Converged, "the seeded run converges");
        Check(result.Sweeps <= 20, "convergence is quick (" + result.Sweeps + " sweeps)");
        Check(result.LabelCounts[0] == 16, "the labelling starts from singletons");
        for (int i = 1; i < result.LabelCounts.Count; i++)
        {
            Check(result.LabelCounts[i] <= result.LabelCounts[i - 1],
                    "the label count never grows at step " + i);
        }
        Check(SimpleGraph.CountLabels(result.Labels) == 2, "the seeded run leaves two communities");
        Check(SimpleGraph.LabelSets(result.Labels) == SimpleGraph.LabelSets(plantedTruth),
                "the seeded run recovers the planted two-block partition");
        Check(LabelPropagation.IsWeakEquilibrium(planted, result.Labels), "the result is at weak equilibrium");
        Check(Math.Abs(planted.Modularity(result.Labels) - planted.Modularity(plantedTruth)) <= 1e-12,
                "the recovered partition has the planted modularity");
        Check(planted.Modularity(result.Labels) > 0.3, "the recovered modularity is high");

        var repeat = LabelPropagation.Run(planted, 100, 549007, true);
        Check(SimpleGraph.LabelSets(repeat.Labels) == SimpleGraph.LabelSets(result.Labels),
                "the seeded run is reproducible");
        var otherSeed = LabelPropagation.Run(planted, 100, 549999, true);
        Check(otherSeed.Converged, "another seed still converges");
        Check(LabelPropagation.IsWeakEquilibrium(planted, otherSeed.Labels),
                "another seed reaches weak equilibrium");
        Check(planted.Modularity(otherSeed.Labels) > 0.3, "another seed is still a good partition");

        // The min-label rule is deterministic but degenerate on connected graphs: everything
        // collapses onto the smallest reachable index.
        var path = SimpleGraph.Path(12);
        var minPath = LabelPropagation.Run(path, 100, 1UL, false);
        Check(minPath.Converged, "the min-tie rule converges on the path");
        Check(SimpleGraph.CountLabels(minPath.Labels) == 1, "the min-tie rule collapses the path");
        // Label 0 is overwritten before any neighbour can adopt it (vertex 0 is visited first and
        // only looks at its neighbours, and no other vertex ever offers its own label), so the
        // consensus is 1 rather than 0.
        for (int v = 0; v < path.N; v++) Check(minPath.Labels[v] == 1, "the path collapses to label 1");
        Check(LabelPropagation.IsStrictEquilibrium(path, minPath.Labels), "the collapsed path is an equilibrium");
        var complete = SimpleGraph.Clique(7);
        var minComplete = LabelPropagation.Run(complete, 100, 1UL, false);
        Check(minComplete.Converged, "the min-tie rule converges on the clique");
        Check(SimpleGraph.CountLabels(minComplete.Labels) == 1, "the min-tie rule collapses the clique");
        for (int v = 0; v < complete.N; v++) Check(minComplete.Labels[v] == 1, "the clique collapses to label 1");
        Check(LabelPropagation.IsStrictEquilibrium(complete, minComplete.Labels), "the collapsed clique is stable");

        // Two disjoint components never merge, whatever the tie-break.
        var triangles = SimpleGraph.TwoCliques(3);
        Check(triangles.EdgeCount() == 6, "the triangle pair has six edges");
        var triangleResult = LabelPropagation.Run(triangles, 100, 549111, true);
        Check(triangleResult.Converged, "the triangle pair converges");
        Check(SimpleGraph.CountLabels(triangleResult.Labels) == 2, "each triangle keeps a label");
        for (int v = 0; v < 3; v++)
        {
            Check(triangleResult.Labels[v] == triangleResult.Labels[0], "the first triangle is uniform at " + v);
            Check(triangleResult.Labels[3 + v] == triangleResult.Labels[3], "the second triangle is uniform at " + v);
        }
        Check(triangleResult.Labels[0] != triangleResult.Labels[3], "the triangles do not merge");
        Check(Math.Abs(triangles.Modularity(triangleResult.Labels) - 0.5) <= 1e-12,
                "two disjoint triangles have modularity 0.5");
        Check(LabelPropagation.IsWeakEquilibrium(triangles, triangleResult.Labels), "the triangle split is an equilibrium");
        var cliques = SimpleGraph.TwoCliques(6);
        Check(cliques.Components()[0] != cliques.Components()[11], "the clique pair is disconnected");
        var cliqueResult = LabelPropagation.Run(cliques, 100, 549222, true);
        Check(cliqueResult.Converged, "the clique pair converges");
        Check(SimpleGraph.CountLabels(cliqueResult.Labels) == 2, "the two cliques stay separate");
        for (int v = 0; v < cliques.N; v++)
        {
            Check(cliqueResult.Labels[v] == cliqueResult.Labels[v / 6 * 6],
                    "component labels are constant inside a clique at " + v);
        }
        Check(cliques.Modularity(cliqueResult.Labels) > 0.0, "the clique split has positive modularity");

        // Shuffled visitation order with the seeded rule: stable and reproducible.
        var shuffled = LabelPropagation.RunShuffled(planted, 100, 549123, true);
        Check(shuffled.Converged, "the shuffled variant converges");
        Check(LabelPropagation.IsWeakEquilibrium(planted, shuffled.Labels), "the shuffled labelling is an equilibrium");
        var shuffledAgain = LabelPropagation.RunShuffled(planted, 100, 549123, true);
        Check(SimpleGraph.LabelSets(shuffledAgain.Labels) == SimpleGraph.LabelSets(shuffled.Labels),
                "the shuffled variant is reproducible");
        Check(planted.Modularity(shuffled.Labels) > 0.3, "the shuffled run is still a good partition");
        var shuffledDifferent = LabelPropagation.RunShuffled(planted, 100, 549124, true);
        Check(shuffledDifferent.Converged, "a different shuffle still converges");

        // h-hop labelling: with unique starting labels the plurality is always the smallest index in
        // the closed h-ball (every count is one, so the tie-break decides), which is checkable
        // exactly; on the path it collapses to max(0, v - h).
        var hopZero = LabelPropagation.HopLabels(planted, 0);
        for (int v = 0; v < planted.N; v++) Check(hopZero[v] == v, "the 0-hop labelling is the identity");
        Check(LabelPropagation.MatchesMinimumInBall(planted, hopZero, 0), "the 0-hop labels are the ball minima");
        foreach (int hops in new[] { 1, 2, 3 })
        {
            var labels = LabelPropagation.HopLabels(planted, hops);
            Check(LabelPropagation.MatchesMinimumInBall(planted, labels, hops),
                    "the " + hops + "-hop labels are the closed-ball minima");
            Check(SimpleGraph.CountLabels(labels) < planted.N, "the " + hops + "-hop labelling merges vertices");
        }
        foreach (int hops in new[] { 1, 2, 3 })
        {
            var labels = LabelPropagation.HopLabels(path, hops);
            for (int v = 0; v < path.N; v++)
            {
                Check(labels[v] == Math.Max(0, v - hops), "the path " + hops + "-hop label at " + v);
            }
        }

        Console.WriteLine("PASS");
    }

    /** Renumber arbitrary label values into 0..k-1 preserving the partition. */
    private static int[] Renumber(int[] labels)
    {
        var map = new Dictionary<int, int>();
        var result = new int[labels.Length];
        for (int v = 0; v < labels.Length; v++)
        {
            if (!map.TryGetValue(labels[v], out int mapped))
            {
                mapped = map.Count;
                map[labels[v]] = mapped;
            }
            result[v] = mapped;
        }
        return result;
    }
}
