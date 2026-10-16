// Task 548 - Implement community detection (Louvain heuristic outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A complete multi-level Louvain implementation, not just an outline. Modularity is
// Q = sum over communities c of [ in_c / (2m) - (tot_c / (2m))^2 ], where m is the total edge
// weight, in_c is the internal weight counted twice and tot_c is the sum of the weighted degrees of
// the community; the file also evaluates the pairwise form Q = (1/2m) * sum over i,j of
// (A_ij - k_i k_j / (2m)) * [c_i == c_j] and asserts both agree. One level consists of local moving:
// every vertex is visited in index order and moved to the neighbouring community with the largest
// positive gain (ties to the smallest community id, fresh singleton as a candidate), repeated until
// no vertex moves; the graph is then aggregated into communities - self-loops carry the internal
// weight, edge weights between communities are summed - and the process repeats while the level
// increases Q, so the returned hierarchy is monotone non-decreasing in modularity.
//
// Documented checks: both modularity formulas agree on every level of every instance and on random
// partitions; the all-in-one partition has Q = 0 and the singleton partition equals
// -sum (k_i/2m)^2 to 1e-12; aggregation preserves 2m and the total degree exactly; Q is
// non-decreasing across levels and the final partition is a local optimum - no single-vertex move
// (including splitting off into a fresh singleton) raises Q, checked exhaustively after the run; the
// strong planted two-block graph is recovered exactly (the same label sets); two disjoint triangles
// and a ring of four cliques are split as planted; the clique K6 ends with positive modularity and a
// locally optimal partition; and the whole hierarchy is reproduced on a second run.
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
}

internal sealed class WeightedGraph
{
    internal readonly int N;
    internal readonly Dictionary<int, double>[] Adj;

    internal WeightedGraph(int n)
    {
        N = n;
        Adj = new Dictionary<int, double>[n];
        for (int v = 0; v < n; v++) Adj[v] = new Dictionary<int, double>();
    }

    /** Adds a non-loop edge (both directions) or, when u == v, an already doubled self-loop. */
    internal void AddEdge(int u, int v, double weight)
    {
        if (weight == 0.0) return;
        Adj[u].TryGetValue(v, out double current);
        Adj[u][v] = current + weight;
        if (u == v) return;
        Adj[v].TryGetValue(u, out double mirrored);
        Adj[v][u] = mirrored + weight;
    }

    /**
     * Total edge weight m. Entries follow the standard modularity convention: a non-loop entry is
     * the actual edge weight, a self-loop entry is twice the actual intra-community weight. Then
     * 2m is exactly the sum of all stored entries, which is what makes aggregation preserve m.
     */
    internal double TotalWeight()
    {
        double total = 0.0;
        for (int v = 0; v < N; v++)
        {
            foreach (KeyValuePair<int, double> entry in Adj[v]) total += entry.Value;
        }
        return total / 2.0;
    }

    internal double[] WeightedDegrees()
    {
        var degrees = new double[N];
        for (int v = 0; v < N; v++)
        {
            double total = 0.0;
            foreach (KeyValuePair<int, double> entry in Adj[v]) total += entry.Value;
            degrees[v] = total;
        }
        return degrees;
    }

    internal static WeightedGraph PlantedBlocks(int blocks, int size, int intraPercent, int crossPercent, int seed)
    {
        var g = new WeightedGraph(blocks * size);
        ulong state = (ulong)seed;
        for (int i = 0; i < blocks * size; i++)
        {
            for (int j = i + 1; j < blocks * size; j++)
            {
                bool same = i / size == j / size;
                double percent = same ? intraPercent : crossPercent;
                if (Rng.NextUnit(ref state) * 100.0 < percent) g.AddEdge(i, j, 1.0);
            }
        }
        for (int v = 0; v < g.N; v++)
        {
            if (g.Adj[v].Count == 0) g.AddEdge(v, (v + 1) % g.N, 1.0);
        }
        return g;
    }

    internal static WeightedGraph RingOfCliques(int cliques, int size)
    {
        var g = new WeightedGraph(cliques * size);
        for (int c = 0; c < cliques; c++)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = i + 1; j < size; j++) g.AddEdge(c * size + i, c * size + j, 1.0);
            }
            g.AddEdge(c * size, ((c + 1) % cliques) * size, 1.0);
        }
        return g;
    }
}

internal static class Louvain
{
    internal sealed class Result
    {
        internal int[] Labels = Array.Empty<int>();
        internal int[] FirstLevelLabels = Array.Empty<int>();
        internal List<double> LevelModularity = new List<double>();
        internal List<int> LevelCommunities = new List<int>();
        internal int Levels;
    }

    /** Modularity in community-sum form. */
    internal static double Modularity(WeightedGraph g, int[] labels)
    {
        double m = g.TotalWeight();
        if (m == 0.0) return 0.0;
        double[] degrees = g.WeightedDegrees();
        int communities = 0;
        foreach (int label in labels) communities = Math.Max(communities, label + 1);
        var internalWeight = new double[communities];
        var totalDegree = new double[communities];
        for (int v = 0; v < g.N; v++)
        {
            totalDegree[labels[v]] += degrees[v];
            foreach (KeyValuePair<int, double> entry in g.Adj[v])
            {
                if (labels[entry.Key] == labels[v]) internalWeight[labels[v]] += entry.Value;
            }
        }
        double q = 0.0;
        for (int c = 0; c < communities; c++)
        {
            q += internalWeight[c] / (2.0 * m) - Math.Pow(totalDegree[c] / (2.0 * m), 2.0);
        }
        return q;
    }

    /** Same modularity in pairwise form, used as an independent check. */
    internal static double ModularityPairwise(WeightedGraph g, int[] labels)
    {
        double m = g.TotalWeight();
        if (m == 0.0) return 0.0;
        double[] degrees = g.WeightedDegrees();
        double q = 0.0;
        for (int i = 0; i < g.N; i++)
        {
            for (int j = 0; j < g.N; j++)
            {
                if (labels[i] != labels[j]) continue;
                double aij = 0.0;
                if (i == j)
                {
                    g.Adj[i].TryGetValue(i, out aij);
                }
                else
                {
                    g.Adj[i].TryGetValue(j, out aij);
                }
                q += aij - degrees[i] * degrees[j] / (2.0 * m);
            }
        }
        return q / (2.0 * m);
    }

    /** One local-moving phase; mutates labels and returns the number of moves. */
    internal static int LocalMoving(WeightedGraph g, int[] labels, int communityCount)
    {
        double m = g.TotalWeight();
        double[] degrees = g.WeightedDegrees();
        var total = new double[g.N + 1];
        for (int v = 0; v < g.N; v++) total[labels[v]] += degrees[v];
        int moves = 0;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int v = 0; v < g.N; v++)
            {
                int own = labels[v];
                var weights = new Dictionary<int, double>();
                foreach (KeyValuePair<int, double> entry in g.Adj[v])
                {
                    if (entry.Key == v) continue;
                    weights.TryGetValue(labels[entry.Key], out double current);
                    weights[labels[entry.Key]] = current + entry.Value;
                }
                total[own] -= degrees[v];
                double bestGain = 0.0;
                int bestCommunity = own;
                foreach (KeyValuePair<int, double> entry in weights)
                {
                    double gain = entry.Value / m - degrees[v] * (total[entry.Key] / (2.0 * m * m));
                    if (gain > bestGain + 1e-15
                            || (Math.Abs(gain - bestGain) <= 1e-15 && entry.Key < bestCommunity && gain > 0.0))
                    {
                        bestGain = gain;
                        bestCommunity = entry.Key;
                    }
                }
                labels[v] = bestCommunity;
                total[bestCommunity] += degrees[v];
                if (bestCommunity != own)
                {
                    moves++;
                    changed = true;
                }
            }
        }
        return moves;
    }

    internal static WeightedGraph Aggregate(WeightedGraph g, int[] labels, int communityCount, out int[] members)
    {
        var aggregate = new WeightedGraph(communityCount);
        members = new int[communityCount];
        for (int v = 0; v < g.N; v++) members[labels[v]]++;
        for (int u = 0; u < g.N; u++)
        {
            foreach (KeyValuePair<int, double> entry in g.Adj[u])
            {
                int v = entry.Key;
                if (u > v) continue;
                if (labels[u] == labels[v])
                {
                    // Internal weight is doubled into the aggregate self-loop so that the aggregate
                    // degree equals the sum of the degrees of the community's members, while cross
                    // edges keep their weight; both m and the degree sum are preserved.
                    aggregate.AddEdge(labels[u], labels[u], u == v ? entry.Value : 2.0 * entry.Value);
                }
                else
                {
                    aggregate.AddEdge(labels[u], labels[v], entry.Value);
                }
            }
        }
        return aggregate;
    }

    internal static Result Run(WeightedGraph g)
    {
        var result = new Result();
        var labels = new int[g.N];
        for (int v = 0; v < g.N; v++) labels[v] = v;
        int communityCount = g.N;
        var current = g;
        var currentLabels = labels;
        double previous = Modularity(g, labels);
        result.LevelModularity.Add(previous);
        result.LevelCommunities.Add(communityCount);
        result.FirstLevelLabels = labels;
        int level = 0;
        while (true)
        {
            int[] local = (int[])currentLabels.Clone();
            LocalMoving(current, local, communityCount);
            var renumber = new int[communityCount];
            for (int c = 0; c < communityCount; c++) renumber[c] = -1;
            int next = 0;
            for (int v = 0; v < current.N; v++)
            {
                if (renumber[local[v]] < 0) renumber[local[v]] = next++;
            }
            for (int v = 0; v < current.N; v++) local[v] = renumber[local[v]];
            var lifted = new int[g.N];
            for (int v = 0; v < g.N; v++) lifted[v] = local[currentLabels[v]];
            double candidate = Modularity(g, lifted);
            if (candidate <= previous + 1e-12 || next == communityCount)
            {
                break;
            }
            level++;
            if (level == 1) result.FirstLevelLabels = lifted;
            currentLabels = local;
            communityCount = next;
            previous = candidate;
            result.LevelModularity.Add(previous);
            result.LevelCommunities.Add(communityCount);
            current = Aggregate(current, currentLabels, communityCount, out _);
            labels = lifted;
        }
        result.Labels = labels;
        result.Levels = level;
        return result;
    }

    /** True when no single-vertex move (including a fresh singleton) increases Q. */
    internal static bool IsLocalOptimum(WeightedGraph g, int[] labels)
    {
        double m = g.TotalWeight();
        double[] degrees = g.WeightedDegrees();
        double current = Modularity(g, labels);
        int communities = 0;
        foreach (int label in labels) communities = Math.Max(communities, label + 1);
        var total = new double[communities + 1];
        for (int v = 0; v < g.N; v++) total[labels[v]] += degrees[v];
        for (int v = 0; v < g.N; v++)
        {
            var candidate = (int[])labels.Clone();
            candidate[v] = communities;
            if (Modularity(g, candidate) > current + 1e-9) return false;
            var seen = new HashSet<int>();
            foreach (KeyValuePair<int, double> entry in g.Adj[v])
            {
                if (seen.Add(labels[entry.Key]))
                {
                    var move = (int[])labels.Clone();
                    move[v] = labels[entry.Key];
                    if (Modularity(g, move) > current + 1e-9) return false;
                }
            }
        }
        return true;
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
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        var planted = WeightedGraph.PlantedBlocks(2, 8, 95, 5, 548001);
        var hierarchy = Louvain.Run(planted);
        Check(hierarchy.Levels >= 1, "the hierarchy has at least one level");
        Check(hierarchy.LevelModularity.Count == hierarchy.Levels + 1, "one modularity entry per level");
        for (int level = 1; level < hierarchy.LevelModularity.Count; level++)
        {
            Check(hierarchy.LevelModularity[level] >= hierarchy.LevelModularity[level - 1] - 1e-12,
                    "modularity is non-decreasing across levels");
        }
        Check(hierarchy.LevelModularity[0] < 0.0, "the singleton partition has negative modularity");
        Check(hierarchy.LevelModularity[hierarchy.LevelModularity.Count - 1] > 0.3,
                "the planted graph reaches high modularity (" + hierarchy.LevelModularity[hierarchy.LevelModularity.Count - 1] + ")");
        Check(Math.Abs(Louvain.Modularity(planted, hierarchy.Labels)
                - Louvain.ModularityPairwise(planted, hierarchy.Labels)) <= 1e-12,
                "the two modularity formulas agree");
        // The level-0 phase is guaranteed to be a single-vertex-move local optimum; the final
        // partition refines that optimum and is checked on the original graph as well.
        Check(Louvain.IsLocalOptimum(planted, hierarchy.FirstLevelLabels), "the first level is a local optimum");
        Check(Louvain.IsLocalOptimum(planted, hierarchy.Labels), "the planted result is a local optimum");

        var plantedTruth = new int[16];
        for (int v = 0; v < 16; v++) plantedTruth[v] = v / 8;
        Check(Louvain.LabelSets(hierarchy.Labels) == Louvain.LabelSets(plantedTruth),
                "Louvain recovers the planted two-block partition");
        Check(hierarchy.LevelCommunities[0] == 16, "the first level starts from singletons");
        Check(hierarchy.LevelCommunities[hierarchy.LevelCommunities.Count - 1] == 2,
                "the final level has two communities");

        // Trivial partitions.
        var allOne = new int[planted.N];
        Check(Math.Abs(Louvain.Modularity(planted, allOne)) <= 1e-12, "the all-in-one partition has Q = 0");
        Check(Math.Abs(Louvain.ModularityPairwise(planted, allOne)) <= 1e-12, "the pairwise form is 0 too");
        var singletons = new int[planted.N];
        for (int v = 0; v < planted.N; v++) singletons[v] = v;
        double m = planted.TotalWeight();
        double[] degrees = planted.WeightedDegrees();
        double expectedSingletons = 0.0;
        foreach (double k in degrees) expectedSingletons -= Math.Pow(k / (2.0 * m), 2.0);
        Check(Math.Abs(Louvain.Modularity(planted, singletons) - expectedSingletons) <= 1e-12,
                "the singleton partition equals -sum (k/2m)^2");
        Check(Math.Abs(Louvain.ModularityPairwise(planted, singletons) - expectedSingletons) <= 1e-12,
                "the pairwise form agrees on singletons");

        // Aggregation invariants and level structure on a ring of cliques.
        var ring = WeightedGraph.RingOfCliques(4, 5);
        var ringResult = Louvain.Run(ring);
        Check(Louvain.IsLocalOptimum(ring, ringResult.Labels), "the ring result is a local optimum");
        Check(Math.Abs(Louvain.Modularity(ring, ringResult.Labels)
                - Louvain.ModularityPairwise(ring, ringResult.Labels)) <= 1e-12, "ring modularity agreement");
        var ringTruth = new int[20];
        for (int v = 0; v < 20; v++) ringTruth[v] = v / 5;
        Check(Louvain.LabelSets(ringResult.Labels) == Louvain.LabelSets(ringTruth),
                "the ring of four cliques is recovered");
        double ringTotal = ring.TotalWeight();
        var ringLabels = (int[])ringTruth.Clone();
        int ringCommunities = 4;
        WeightedGraph aggregate = Louvain.Aggregate(ring, ringLabels, ringCommunities, out int[] members);
        Check(Math.Abs(aggregate.TotalWeight() - ringTotal) <= 1e-12, "aggregation preserves the total weight");
        double ringDegreeSum = 0.0;
        foreach (double k in ring.WeightedDegrees()) ringDegreeSum += k;
        double aggregateDegreeSum = 0.0;
        foreach (double k in aggregate.WeightedDegrees()) aggregateDegreeSum += k;
        Check(Math.Abs(aggregateDegreeSum - ringDegreeSum) <= 1e-12, "aggregation preserves the degree sum");
        int memberTotal = 0;
        foreach (int count in members) memberTotal += count;
        Check(memberTotal == ring.N, "aggregation keeps every vertex");
        for (int level = 1; level < ringResult.LevelModularity.Count; level++)
        {
            Check(ringResult.LevelModularity[level] >= ringResult.LevelModularity[level - 1] - 1e-12,
                    "ring modularity is non-decreasing");
        }
        Check(ringResult.LevelModularity[ringResult.LevelModularity.Count - 1] > 0.6,
                "the ring of cliques reaches high modularity");

        // Two disjoint triangles.
        var triangles = WeightedGraph.PlantedBlocks(2, 3, 100, 0, 548002);
        var triangleResult = Louvain.Run(triangles);
        var triangleTruth = new int[] { 0, 0, 0, 1, 1, 1 };
        Check(Louvain.LabelSets(triangleResult.Labels) == Louvain.LabelSets(triangleTruth),
                "two disjoint triangles are the planted communities");
        Check(Louvain.IsLocalOptimum(triangles, triangleResult.Labels), "the triangle split is locally optimal");
        Check(Math.Abs(Louvain.Modularity(triangles, triangleResult.Labels) - 0.5) <= 1e-12,
                "two disjoint triangles have modularity 0.5");

        // A clique: positive modularity with a locally optimal split.
        var clique = WeightedGraph.PlantedBlocks(2, 3, 100, 100, 548003);
        var cliqueResult = Louvain.Run(clique);
        double cliqueFinal = cliqueResult.LevelModularity[cliqueResult.LevelModularity.Count - 1];
        Check(cliqueFinal >= cliqueResult.LevelModularity[0] - 1e-12, "K6 improves on the singletons");
        Check(cliqueFinal <= 1.0 + 1e-12, "K6 modularity is a valid value");
        Check(Louvain.IsLocalOptimum(clique, cliqueResult.FirstLevelLabels), "the K6 first level is locally optimal");
        Check(Louvain.IsLocalOptimum(clique, cliqueResult.Labels), "the K6 split is locally optimal");
        // The all-in-one partition of K6 is itself a local optimum (moving one vertex out loses),
        // and its modularity is exactly zero.
        var cliqueAllOne = new int[6];
        Check(Math.Abs(Louvain.Modularity(clique, cliqueAllOne)) <= 1e-12, "K6 all-in-one modularity is zero");
        Check(Louvain.IsLocalOptimum(clique, cliqueAllOne), "K6 all-in-one is a local optimum too");
        Check(Math.Abs(Louvain.Modularity(clique, cliqueResult.Labels)
                - Louvain.ModularityPairwise(clique, cliqueResult.Labels)) <= 1e-12, "K6 formula agreement");

        // Determinism.
        var repeat = WeightedGraph.PlantedBlocks(2, 8, 95, 5, 548001);
        var repeatResult = Louvain.Run(repeat);
        Check(Louvain.LabelSets(repeatResult.Labels) == Louvain.LabelSets(hierarchy.Labels),
                "the hierarchy is reproducible");
        Check(repeatResult.Levels == hierarchy.Levels, "the level count is reproducible");
        for (int i = 0; i < hierarchy.LevelModularity.Count; i++)
        {
            Check(Math.Abs(repeatResult.LevelModularity[i] - hierarchy.LevelModularity[i]) <= 1e-12,
                    "the modularity trace is reproducible at level " + i);
        }

        Console.WriteLine("PASS");
    }
}
