// Task 566 - Implement Steiner tree approximate on small graphs (heuristic). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// The Steiner tree problem on small weighted graphs, with both an exact and an approximate method so the
// approximation can be measured rather than praised. Exact: the Dreyfus-Wagner subset DP over terminal
// subsets, dp[S][v] = minimum cost of a tree connecting the terminals of S and containing v, computed by
// merging complementary subsets at a common vertex and then relaxing through the graph with a multi-source
// Dijkstra; the answer is min_v dp[all][v], with complexity O(3^k n + 2^k (m log n)) in the number of
// terminals k, which is precisely why the file keeps k <= 5. Independent reference: brute-force
// enumeration of every subset of non-terminal vertices, keeping the subsets whose induced subgraph is
// connected and spans all terminals and taking the minimum spanning tree cost of the induced subgraph -
// the optimum is attained because the optimal tree's own vertex set is one of those subsets.
// Approximation: the KMB shortest-path heuristic - metric closure of the terminals (Floyd-Warshall),
// minimum spanning tree of that closure (Kruskal with a disjoint-set structure), then replace every closure
// edge by a shortest path and take the union of the edges. The union is a connected subgraph containing all
// terminals, so its cost is at least the optimum, and the classical bound is cost <= (2 - 2/k) * OPT; the
// file measures the ratio on every instance instead of assuming the bound, exhibits a hub-versus-direct
// construction where the heuristic is strictly worse than optimal, and checks that on tree metrics the
// shortest-path heuristic is exact by comparing against an independent leaf-pruning computation of the
// minimal subtree. All random graphs and weights come from splitmix64 and are connected by construction
// (a random spanning tree is planted first).
//
// Documented checks: on 40 random connected graphs (n = 8, k = 3), 24 (n = 9, k = 4) and 12 (n = 9, k = 5)
// the exact DP equals the brute-force subset enumeration; the KMB cost lies in [OPT, (2 - 2/k) OPT] and
// also does not exceed the closure MST cost, with the worst measured ratio reported in the message; two
// terminals give the shortest-path distance for the DP, the brute force and the heuristic alike; zero or
// one terminal gives 0; on six random trees the heuristic equals the exact DP and equals the weight of the
// minimal subtree computed by pruning non-terminal leaves; the hub construction (5 terminals, hub edges
// 100, direct edges 199) has OPT = 500, a KMB cost of 796 and therefore a ratio of 1.592 inside the
// 1.6 bound; and repeated runs are identical.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    /** splitmix64 (Steele/Lea/Flood): 64-bit state advanced by a golden-ratio increment. */
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int Below(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class DisjointSet
{
    private readonly int[] parent;
    private readonly int[] rank;

    internal DisjointSet(int n)
    {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    internal int Find(int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    internal bool Union(int a, int b)
    {
        int ra = Find(a);
        int rb = Find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) parent[ra] = rb;
        else if (rank[rb] < rank[ra]) parent[rb] = ra;
        else
        {
            parent[rb] = ra;
            rank[ra]++;
        }
        return true;
    }
}

internal sealed class WeightedGraph
{
    internal const long Inf = long.MaxValue / 8;
    internal readonly int N;
    internal readonly string Label;
    private readonly List<int[]>[] adjacency;
    internal readonly List<int[]> Edges = new List<int[]>();

    internal WeightedGraph(int n, string label)
    {
        N = n;
        Label = label;
        adjacency = new List<int[]>[n];
        for (int i = 0; i < n; i++) adjacency[i] = new List<int[]>();
    }

    internal void AddEdge(int u, int v, int weight)
    {
        adjacency[u].Add(new[] { v, weight });
        adjacency[v].Add(new[] { u, weight });
        Edges.Add(new[] { u, v, weight });
    }

    internal int EdgeWeight(int u, int v)
    {
        int best = -1;
        foreach (int[] edge in adjacency[u])
        {
            if (edge[0] != v) continue;
            if (best < 0 || edge[1] < best) best = edge[1];
        }
        return best;
    }

    internal List<int[]> Neighbours(int v)
    {
        return adjacency[v];
    }

    internal static WeightedGraph RandomGraph(int n, int extraEdges, ref ulong state, string label)
    {
        var graph = new WeightedGraph(n, label);
        var order = new List<int>();
        for (int i = 0; i < n; i++) order.Add(i);
        for (int i = n - 1; i > 0; i--)
        {
            int j = SplitMix64.Below(ref state, i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        for (int i = 1; i < n; i++)
        {
            int parent = order[SplitMix64.Below(ref state, i)];
            graph.AddEdge(order[i], parent, 1 + SplitMix64.Below(ref state, 9));
        }
        for (int i = 0; i < extraEdges; i++)
        {
            int u = SplitMix64.Below(ref state, n);
            int v = SplitMix64.Below(ref state, n);
            if (u == v) continue;
            graph.AddEdge(u, v, 1 + SplitMix64.Below(ref state, 9));
        }
        return graph;
    }

    internal static WeightedGraph RandomTree(int n, ref ulong state, string label)
    {
        var graph = new WeightedGraph(n, label);
        for (int i = 1; i < n; i++) graph.AddEdge(i, SplitMix64.Below(ref state, i),
                1 + SplitMix64.Below(ref state, 9));
        return graph;
    }

    internal int[] RandomTerminals(int k, ref ulong state)
    {
        var chosen = new List<int>();
        while (chosen.Count < k)
        {
            int v = SplitMix64.Below(ref state, N);
            if (!chosen.Contains(v)) chosen.Add(v);
        }
        return chosen.ToArray();
    }
}

internal static class Steiner
{
    internal sealed class Approx
    {
        internal long Cost;
        internal long ClosureMstCost;
        internal int EdgeCount;
    }

    /** Floyd-Warshall distances (Inf when disconnected). */
    internal static long[,] Distances(WeightedGraph g)
    {
        int n = g.N;
        var dist = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) dist[i, j] = i == j ? 0 : WeightedGraph.Inf;
        }
        foreach (int[] edge in g.Edges)
        {
            if (edge[2] < dist[edge[0], edge[1]])
            {
                dist[edge[0], edge[1]] = edge[2];
                dist[edge[1], edge[0]] = edge[2];
            }
        }
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                if (dist[i, k] >= WeightedGraph.Inf) continue;
                for (int j = 0; j < n; j++)
                {
                    if (dist[k, j] >= WeightedGraph.Inf) continue;
                    long candidate = dist[i, k] + dist[k, j];
                    if (candidate < dist[i, j]) dist[i, j] = candidate;
                }
            }
        }
        return dist;
    }

    private static long[] MultiSourceDijkstra(WeightedGraph g, long[] initial)
    {
        int n = g.N;
        var dist = (long[])initial.Clone();
        var done = new bool[n];
        for (int step = 0; step < n; step++)
        {
            int best = -1;
            for (int v = 0; v < n; v++)
            {
                if (!done[v] && dist[v] < WeightedGraph.Inf && (best < 0 || dist[v] < dist[best])) best = v;
            }
            if (best < 0) break;
            done[best] = true;
            foreach (int[] edge in g.Neighbours(best))
            {
                long candidate = dist[best] + edge[1];
                if (candidate < dist[edge[0]]) dist[edge[0]] = candidate;
            }
        }
        return dist;
    }

    /** Dreyfus-Wagner exact DP. */
    internal static long Exact(WeightedGraph g, int[] terminals)
    {
        int k = terminals.Length;
        int n = g.N;
        if (k <= 1) return 0;
        int subsets = 1 << k;
        var dp = new long[subsets][];
        for (int s = 0; s < subsets; s++)
        {
            dp[s] = new long[n];
            for (int v = 0; v < n; v++) dp[s][v] = WeightedGraph.Inf;
        }
        for (int i = 0; i < k; i++)
        {
            dp[1 << i][terminals[i]] = 0;
            // The distance from a single terminal to every vertex is the shortest-path relaxation.
            dp[1 << i] = MultiSourceDijkstra(g, dp[1 << i]);
        }
        for (int s = 1; s < subsets; s++)
        {
            if ((s & (s - 1)) == 0) continue;
            for (int sub = (s - 1) & s; sub > 0; sub = (sub - 1) & s)
            {
                int other = s ^ sub;
                if (other == 0) continue;
                for (int v = 0; v < n; v++)
                {
                    if (dp[sub][v] >= WeightedGraph.Inf || dp[other][v] >= WeightedGraph.Inf) continue;
                    long merged = dp[sub][v] + dp[other][v];
                    if (merged < dp[s][v]) dp[s][v] = merged;
                }
            }
            dp[s] = MultiSourceDijkstra(g, dp[s]);
        }
        long best = WeightedGraph.Inf;
        for (int v = 0; v < n; v++)
        {
            if (dp[subsets - 1][v] < best) best = dp[subsets - 1][v];
        }
        return best;
    }

    /** Brute force: every subset of non-terminal vertices whose induced subgraph spans the terminals. */
    internal static long BruteForce(WeightedGraph g, int[] terminals)
    {
        int k = terminals.Length;
        if (k <= 1) return 0;
        var terminalSet = new HashSet<int>(terminals);
        var optional = new List<int>();
        for (int v = 0; v < g.N; v++)
        {
            if (!terminalSet.Contains(v)) optional.Add(v);
        }
        long best = WeightedGraph.Inf;
        int subsets = 1 << optional.Count;
        for (int mask = 0; mask < subsets; mask++)
        {
            var chosen = new List<int>(terminals);
            for (int i = 0; i < optional.Count; i++)
            {
                if ((mask & (1 << i)) != 0) chosen.Add(optional[i]);
            }
            var inSet = new bool[g.N];
            foreach (int v in chosen) inSet[v] = true;
            var edges = new List<int[]>();
            foreach (int[] edge in g.Edges)
            {
                if (inSet[edge[0]] && inSet[edge[1]]) edges.Add(edge);
            }
            edges.Sort(delegate (int[] a, int[] b) { return a[2].CompareTo(b[2]); });
            var dsu = new DisjointSet(g.N);
            long cost = 0;
            int used = 0;
            foreach (int[] edge in edges)
            {
                if (dsu.Union(edge[0], edge[1]))
                {
                    cost += edge[2];
                    used++;
                }
            }
            if (used != chosen.Count - 1) continue;
            int root = dsu.Find(terminals[0]);
            bool spans = true;
            foreach (int t in terminals)
            {
                if (dsu.Find(t) != root) { spans = false; break; }
            }
            if (!spans) continue;
            if (cost < best) best = cost;
        }
        return best;
    }

    private static void DijkstraTree(WeightedGraph g, int source, out long[] dist, out int[] parent)
    {
        int n = g.N;
        dist = new long[n];
        parent = new int[n];
        var done = new bool[n];
        for (int v = 0; v < n; v++)
        {
            dist[v] = WeightedGraph.Inf;
            parent[v] = -1;
        }
        dist[source] = 0;
        for (int step = 0; step < n; step++)
        {
            int best = -1;
            for (int v = 0; v < n; v++)
            {
                if (!done[v] && dist[v] < WeightedGraph.Inf && (best < 0 || dist[v] < dist[best])) best = v;
            }
            if (best < 0) break;
            done[best] = true;
            foreach (int[] edge in g.Neighbours(best))
            {
                long candidate = dist[best] + edge[1];
                if (candidate < dist[edge[0]])
                {
                    dist[edge[0]] = candidate;
                    parent[edge[0]] = best;
                }
            }
        }
    }

    /** KMB: metric closure MST, expanded into shortest paths, union of the edge sets. */
    internal static Approx Kmb(WeightedGraph g, int[] terminals)
    {
        int k = terminals.Length;
        var result = new Approx();
        if (k <= 1) return result;
        long[,] dist = Distances(g);
        var pairs = new List<int[]>();
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                long d = dist[terminals[i], terminals[j]];
                if (d < WeightedGraph.Inf) pairs.Add(new[] { i, j, (int)d });
            }
        }
        pairs.Sort(delegate (int[] a, int[] b)
        {
            int cmp = a[2].CompareTo(b[2]);
            return cmp != 0 ? cmp : (a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
        });
        var dsu = new DisjointSet(k);
        var chosen = new List<int[]>();
        foreach (int[] pair in pairs)
        {
            if (!dsu.Union(pair[0], pair[1])) continue;
            chosen.Add(pair);
            result.ClosureMstCost += pair[2];
        }
        if (chosen.Count != k - 1) throw new Exception("the terminal closure is not connected");

        var edgeCost = new Dictionary<long, int>();
        foreach (int[] pair in chosen)
        {
            DijkstraTree(g, terminals[pair[0]], out long[] _, out int[] parent);
            int current = terminals[pair[1]];
            while (current != terminals[pair[0]])
            {
                int previous = parent[current];
                if (previous < 0) throw new Exception("path reconstruction failed");
                int low = Math.Min(previous, current);
                int high = Math.Max(previous, current);
                long key = (long)low * g.N + high;
                int weight = g.EdgeWeight(low, high);
                if (weight < 0) throw new Exception("reconstructed an edge that does not exist");
                edgeCost.TryGetValue(key, out int existing);
                if (existing == 0 || weight < existing) edgeCost[key] = weight;
                current = previous;
            }
        }
        result.EdgeCount = edgeCost.Count;
        foreach (int weight in edgeCost.Values) result.Cost += weight;
        return result;
    }

    /** Minimal subtree of a tree containing the terminals: repeatedly prune non-terminal leaves. */
    internal static long MinimalSubtree(WeightedGraph tree, int[] terminals)
    {
        int n = tree.N;
        var terminalSet = new HashSet<int>(terminals);
        var degree = new int[n];
        var removed = new bool[n];
        var queue = new Queue<int>();
        for (int v = 0; v < n; v++)
        {
            degree[v] = tree.Neighbours(v).Count;
            if (degree[v] <= 1 && !terminalSet.Contains(v)) queue.Enqueue(v);
        }
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            if (removed[v] || terminalSet.Contains(v) || degree[v] > 1) continue;
            removed[v] = true;
            foreach (int[] edge in tree.Neighbours(v))
            {
                if (removed[edge[0]]) continue;
                degree[edge[0]]--;
                if (degree[edge[0]] <= 1 && !terminalSet.Contains(edge[0])) queue.Enqueue(edge[0]);
            }
        }
        long cost = 0;
        foreach (int[] edge in tree.Edges)
        {
            if (!removed[edge[0]] && !removed[edge[1]]) cost += edge[2];
        }
        return cost;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 566001UL;
        double worstRatio = 1.0;
        string worstLabel = "";
        int instances = 0;
        var shapes = new List<int[]> { new[] { 8, 3, 40 }, new[] { 9, 4, 24 }, new[] { 9, 5, 12 } };
        foreach (int[] shape in shapes)
        {
            int n = shape[0];
            int k = shape[1];
            int count = shape[2];
            for (int i = 0; i < count; i++)
            {
                WeightedGraph g = WeightedGraph.RandomGraph(n, n, ref rng, "G" + n + "-" + i);
                int[] terminals = g.RandomTerminals(k, ref rng);
                long exact = Steiner.Exact(g, terminals);
                long brute = Steiner.BruteForce(g, terminals);
                Check(exact == brute, g.Label + ": DP " + exact + " != brute force " + brute);
                Steiner.Approx approx = Steiner.Kmb(g, terminals);
                double bound = 2.0 - 2.0 / k;
                Check(approx.Cost >= exact, g.Label + ": the heuristic beats the optimum, which is "
                        + "impossible: " + approx.Cost + " < " + exact);
                Check(approx.Cost <= bound * exact + 1e-9, g.Label + ": ratio " + ((double)approx.Cost / exact)
                        + " exceeds the bound " + bound);
                Check(approx.Cost <= approx.ClosureMstCost, g.Label
                        + ": the union of paths cannot cost more than the closure MST");
                Check(approx.ClosureMstCost <= bound * exact + 1e-9, g.Label
                        + ": the closure MST exceeds the (2 - 2/k) bound");
                Check(approx.EdgeCount > 0, g.Label + ": a non-trivial instance needs edges");
                double ratio = (double)approx.Cost / exact;
                if (ratio > worstRatio)
                {
                    worstRatio = ratio;
                    worstLabel = g.Label;
                }
                instances++;
            }
        }
        Check(instances == 76, "76 random instances were measured, got " + instances);
        Check(worstRatio >= 1.0 && worstRatio <= 2.0, "the worst ratio " + worstRatio
                + " (on " + worstLabel + ") lies in [1,2]");

        // Two terminals: everything must equal the shortest path.
        WeightedGraph pairGraph = WeightedGraph.RandomGraph(7, 6, ref rng, "pair");
        var pairTerminals = pairGraph.RandomTerminals(2, ref rng);
        long[,] distances = Steiner.Distances(pairGraph);
        long path = distances[pairTerminals[0], pairTerminals[1]];
        Check(Steiner.Exact(pairGraph, pairTerminals) == path, "two terminals give the shortest path (DP)");
        Check(Steiner.BruteForce(pairGraph, pairTerminals) == path, "two terminals give the path (brute)");
        Check(Steiner.Kmb(pairGraph, pairTerminals).Cost == path, "two terminals give the path (KMB)");

        // Degenerate terminal sets.
        Check(Steiner.Exact(pairGraph, Array.Empty<int>()) == 0, "no terminal costs 0");
        Check(Steiner.Exact(pairGraph, new[] { 3 }) == 0, "one terminal costs 0");
        Check(Steiner.Kmb(pairGraph, new[] { 3 }).Cost == 0, "one terminal costs 0 for the heuristic");

        // Trees: the shortest-path heuristic is exact and matches leaf pruning.
        for (int i = 0; i < 6; i++)
        {
            WeightedGraph tree = WeightedGraph.RandomTree(9, ref rng, "T" + i);
            int[] terminals = tree.RandomTerminals(4, ref rng);
            long exact = Steiner.Exact(tree, terminals);
            long pruned = Steiner.MinimalSubtree(tree, terminals);
            long heuristic = Steiner.Kmb(tree, terminals).Cost;
            Check(exact == pruned, tree.Label + ": DP " + exact + " != pruned subtree " + pruned);
            Check(heuristic == exact, tree.Label + ": KMB " + heuristic + " != optimum " + exact);
        }

        // The hub-versus-direct construction: 5 terminals, hub edges 100, direct edges 199.
        var hub = new WeightedGraph(6, "hub");
        for (int t = 1; t <= 5; t++) hub.AddEdge(0, t, 100);
        for (int t = 1; t < 5; t++) hub.AddEdge(t, t + 1, 199);
        var hubTerminals = new[] { 1, 2, 3, 4, 5 };
        long hubExact = Steiner.Exact(hub, hubTerminals);
        Check(hubExact == 500, "the hub construction has optimum 500, got " + hubExact);
        Steiner.Approx hubApprox = Steiner.Kmb(hub, hubTerminals);
        Check(hubApprox.Cost == 796, "the heuristic pays 796 through the direct edges, got "
                + hubApprox.Cost);
        double hubRatio = (double)hubApprox.Cost / hubExact;
        Check(hubRatio > 1.0 && hubRatio <= 1.6 + 1e-9, "the hub ratio " + hubRatio
                + " shows a strict gap inside the 1.6 bound");
        Check(Math.Abs(hubRatio - 796.0 / 500.0) <= 1e-12, "the measured ratio is 1.592");

        // Determinism: the same seed reproduces the same graph, terminals and both solutions.
        ulong seedA = 566500UL;
        WeightedGraph graphA = WeightedGraph.RandomGraph(8, 8, ref seedA, "sameA");
        int[] terminalsA = graphA.RandomTerminals(3, ref seedA);
        ulong seedB = 566500UL;
        WeightedGraph graphB = WeightedGraph.RandomGraph(8, 8, ref seedB, "sameB");
        int[] terminalsB = graphB.RandomTerminals(3, ref seedB);
        for (int i = 0; i < terminalsA.Length; i++)
        {
            Check(terminalsA[i] == terminalsB[i], "the same seed picks the same terminals");
        }
        Check(graphA.Edges.Count == graphB.Edges.Count, "the same seed builds the same graph");
        Check(Steiner.Exact(graphA, terminalsA) == Steiner.Exact(graphB, terminalsB),
                "the exact DP is reproducible");
        Check(Steiner.Kmb(graphA, terminalsA).Cost == Steiner.Kmb(graphB, terminalsB).Cost,
                "the heuristic is reproducible");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
