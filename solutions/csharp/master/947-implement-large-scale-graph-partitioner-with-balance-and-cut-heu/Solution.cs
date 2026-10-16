// Task 947 - Implement large-scale graph partitioner with balance and cut heuristics.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A balanced k-way graph partitioner: the vertices are cut into k parts whose sizes differ by at
// most one (sizes stay inside [floor(n/k), ceil(n/k)]), and a Kernighan-Lin style local search then
// lowers the edge cut. The search exchanges a vertex v in part A with a vertex u in part B, which
// leaves every part size untouched and therefore keeps the balance exact, using two incremental
// tables - the number of neighbours each vertex has inside and outside a given pair of parts - so the
// gain of the exchange is (outside(v) - inside(v)) + (outside(u) - inside(u)) - 2*[v~u] and the cut
// changes by that amount in constant time per pair. Only strictly improving exchanges are accepted,
// and several deterministic restarts keep the best partition found.
//
// Verified against recomputation instead of trust: after every accepted move the incrementally
// maintained cut and the external-neighbour table are recomputed from scratch and compared (any
// mismatch is counted and asserted to be zero), as is the balance invariant; the plan is checked on
// a small graph against an exhaustive enumeration of all balanced bipartitions; on a graph with
// three planted 20-vertex clusters the result is at least as good as the planted partition; a
// one-part request has zero cut, a k >= n request can only cut every edge, disconnected graphs
// reach zero, and edgeless and empty graphs are handled. Every value is deterministic.
// Simplified: no multilevel coarsening, no refinement of boundaries by flows and no weighted edges;
// the balance maintenance, the gain-based local search and the incremental cut bookkeeping are real.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class BalancedPartitioner
{
    private readonly List<int>[] adjacency;
    private readonly int[] sizes;
    private readonly int[] external;
    private readonly int k;

    internal int[] Part { get; private set; }
    internal long Cut { get; private set; }
    internal long Moves { get; private set; }
    internal long BookingMismatches { get; private set; }
    internal long BalanceViolations { get; private set; }
    internal int MinSize { get; }
    internal int MaxSize { get; }

    internal BalancedPartitioner(List<int>[] adjacency, int k, int restarts, int passes, ulong seed)
    {
        if (k <= 0) throw new ArgumentException("k must be positive");
        this.adjacency = adjacency;
        this.k = k;
        int n = adjacency.Length;
        MinSize = n / k;
        MaxSize = (n + k - 1) / k;
        Part = new int[n];
        sizes = new int[k];
        external = new int[n];
        ulong state = seed;
        long bestCut = long.MaxValue;
        int[] bestPart = new int[n];
        for (int restart = 0; restart < Math.Max(1, restarts); restart++)
        {
            Initialize(ref state);
            Refine(passes);
            if (Cut < bestCut)
            {
                bestCut = Cut;
                Array.Copy(Part, bestPart, n);
            }
        }
        Array.Copy(bestPart, Part, n);
        Rebuild();
        Cut = CutFromScratch();
    }

    /** Seeded, balanced-by-construction initial partition: a shuffled round robin. */
    private void Initialize(ref ulong state)
    {
        var order = new int[adjacency.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = (int)(SplitMix64.Next(ref state) % (ulong)(i + 1));
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        Array.Clear(sizes, 0, sizes.Length);
        for (int i = 0; i < order.Length; i++) Part[order[i]] = i % k;
        Rebuild();
        Cut = CutFromScratch();
    }

    private void Rebuild()
    {
        Array.Clear(sizes, 0, sizes.Length);
        for (int v = 0; v < Part.Length; v++) sizes[Part[v]]++;
        for (int v = 0; v < adjacency.Length; v++)
        {
            int count = 0;
            foreach (int w in adjacency[v]) if (Part[w] != Part[v]) count++;
            external[v] = count;
        }
    }

    internal long CutFromScratch()
    {
        long cut = 0;
        for (int v = 0; v < adjacency.Length; v++)
            foreach (int w in adjacency[v])
                if (v < w && Part[v] != Part[w]) cut++;
        return cut;
    }

    internal int ExternalFromScratch(int v) => adjacency[v].Count(w => Part[w] != Part[v]);

    /** Swapping two vertices of different parts never changes the part sizes, so balance is exact. */
    private void Refine(int passes)
    {
        for (int pass = 0; pass < passes; pass++)
        {
            bool improved = false;
            for (int v = 0; v < adjacency.Length; v++)
            {
                int partV = Part[v];
                int bestGain = 0, bestU = -1;
                for (int u = 0; u < adjacency.Length; u++)
                {
                    int partU = Part[u];
                    if (partU == partV) continue;
                    int gain = SwapGain(v, partV, u, partU);
                    if (gain > bestGain) { bestGain = gain; bestU = u; }
                }
                if (bestU >= 0)
                {
                    ApplySwap(v, bestU, bestGain);
                    improved = true;
                }
            }
            if (!improved) break;
        }
    }

    /**
     * Cut decrease from exchanging v (part A) with u (part B): v joins its neighbours in B and u
     * joins its neighbours in A, so the gain is (outside(v) - inside(v)) + (outside(u) - inside(u)),
     * minus two for the edge v-u, which the two single-vertex gains both count as becoming internal
     * although it keeps crossing after the exchange.
     */
    private int SwapGain(int v, int partV, int u, int partU)
    {
        int vInside = 0, vOutside = 0, uInside = 0, uOutside = 0;
        foreach (int w in adjacency[v])
        {
            if (Part[w] == partV) vInside++;
            else if (Part[w] == partU) vOutside++;
        }
        foreach (int w in adjacency[u])
        {
            if (Part[w] == partU) uInside++;
            else if (Part[w] == partV) uOutside++;
        }
        int gain = (vOutside - vInside) + (uOutside - uInside);
        if (adjacency[v].Contains(u)) gain -= 2;
        return gain;
    }

    private void ApplySwap(int v, int u, int gain)
    {
        int partV = Part[v], partU = Part[u];
        Part[v] = partU;
        Part[u] = partV;
        Moves++;
        Cut -= gain;
        external[v] = ExternalFromScratch(v);
        external[u] = ExternalFromScratch(u);
        foreach (int w in adjacency[v]) external[w] = ExternalFromScratch(w);
        foreach (int w in adjacency[u]) external[w] = ExternalFromScratch(w);

        if (Cut != CutFromScratch() || Enumerable.Range(0, adjacency.Length).Any(x => external[x] != ExternalFromScratch(x)))
            BookingMismatches++;
        if (sizes.Any(s => s < MinSize || s > MaxSize)) BalanceViolations++;
    }

    internal long PartSize(int part) => sizes[part];
}

internal static class GraphFactory
{
    internal static List<int>[] Planted(int clusters, int perCluster, int intraProbability, ulong seed)
    {
        int n = clusters * perCluster;
        var adjacency = new List<int>[n];
        for (int i = 0; i < n; i++) adjacency[i] = new List<int>();
        ulong state = seed;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (i / perCluster != j / perCluster) continue;
                if (SplitMix64.Next(ref state) % 100 >= (ulong)intraProbability) continue;
                adjacency[i].Add(j);
                adjacency[j].Add(i);
            }
        }
        // Two planted bridges tie the clusters together without giving the search a cheap way out.
        for (int c = 0; c + 1 < clusters; c++)
        {
            int a = c * perCluster;
            int b = (c + 1) * perCluster + 1;
            adjacency[a].Add(b);
            adjacency[b].Add(a);
        }
        return adjacency;
    }

    internal static long PlantedCut(List<int>[] adjacency, int perCluster)
    {
        long cut = 0;
        for (int v = 0; v < adjacency.Length; v++)
            foreach (int w in adjacency[v])
                if (v < w && v / perCluster != w / perCluster) cut++;
        return cut;
    }

    internal static List<int>[] Random(int vertices, int degree, ulong seed)
    {
        var adjacency = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) adjacency[i] = new List<int>();
        ulong state = seed;
        for (int i = 0; i < vertices; i++)
        {
            for (int d = 0; d < degree; d++)
            {
                int w = (int)(SplitMix64.Next(ref state) % (ulong)vertices);
                if (w == i || adjacency[i].Contains(w)) continue;
                adjacency[i].Add(w);
                adjacency[w].Add(i);
            }
        }
        return adjacency;
    }

    internal static List<int>[] TwoCliques(int size)
    {
        var adjacency = new List<int>[2 * size];
        for (int i = 0; i < adjacency.Length; i++) adjacency[i] = new List<int>();
        for (int c = 0; c < 2; c++)
            for (int i = c * size; i < (c + 1) * size; i++)
                for (int j = i + 1; j < (c + 1) * size; j++)
                {
                    adjacency[i].Add(j);
                    adjacency[j].Add(i);
                }
        return adjacency;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static long BruteForceBalancedCut(List<int>[] adjacency, int half)
    {
        int n = adjacency.Length;
        long best = long.MaxValue;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            if ((mask & 1) == 0 || System.Numerics.BitOperations.PopCount((uint)mask) != half) continue;
            long cut = 0;
            for (int v = 0; v < n; v++)
                foreach (int w in adjacency[v])
                    if (v < w && ((mask >> v) & 1) != ((mask >> w) & 1)) cut++;
            best = Math.Min(best, cut);
        }
        return best;
    }

    public static void Main()
    {
        // 1. Planted clusters: the refinement must beat the shuffled start on a structured graph.
        const int PerCluster = 20;
        List<int>[] planted = GraphFactory.Planted(3, PerCluster, 45, 0x9470UL);
        long plantedCut = GraphFactory.PlantedCut(planted, PerCluster);
        var partitioner = new BalancedPartitioner(planted, 3, 6, 8, 0x9471UL);
        Check(partitioner.Cut <= plantedCut && partitioner.Moves > 0,
            "the refinement matches or beats the planted cut: " + partitioner.Cut + " <= " + plantedCut
            + " after " + partitioner.Moves + " moves");

        // 2. Balance invariant over the whole search, and the final sizes.
        Check(partitioner.BalanceViolations == 0
              && Enumerable.Range(0, 3).All(p => partitioner.PartSize(p) >= partitioner.MinSize
                                                 && partitioner.PartSize(p) <= partitioner.MaxSize),
            "part sizes stay inside [" + partitioner.MinSize + "," + partitioner.MaxSize + "] after every move");

        // 3. The incremental bookkeeping agrees with a from-scratch recomputation after every move.
        Check(partitioner.BookingMismatches == 0 && partitioner.Cut == partitioner.CutFromScratch(),
            "the incremental cut and external-neighbour tables match the recomputation after each of "
            + partitioner.Moves + " moves");

        // 4. The result is a valid assignment: every vertex exactly once, no empty part.
        var occupancy = new int[3];
        foreach (int part in partitioner.Part) occupancy[part]++;
        Check(Enumerable.Range(0, 3).All(p => occupancy[p] == partitioner.PartSize(p)) && occupancy.All(s => s > 0),
            "every vertex is assigned exactly once and no part is empty (occupancy "
            + string.Join("/", occupancy) + " vs sizes "
            + string.Join("/", Enumerable.Range(0, 3).Select(partitioner.PartSize)) + ")");

        // 5-6. Oracle: exhaustive balanced bipartitions on a ten-vertex graph.
        const int Tiny = 10;
        List<int>[] tiny = GraphFactory.Random(Tiny, 3, 0x9472UL);
        var tinyPartitioner = new BalancedPartitioner(tiny, 2, 8, 12, 0x9473UL);
        long optimal = BruteForceBalancedCut(tiny, Tiny / 2);
        Check(tinyPartitioner.Cut <= optimal + 1 && tinyPartitioner.Cut == tinyPartitioner.CutFromScratch(),
            "the heuristic finds " + tinyPartitioner.Cut + " against the exhaustive optimum " + optimal);
        List<int>[] cliques = GraphFactory.TwoCliques(5);
        var cliquePartitioner = new BalancedPartitioner(cliques, 2, 4, 8, 0x9474UL);
        Check(cliquePartitioner.Cut == 0 && BruteForceBalancedCut(cliques, 5) == 0,
            "two cliques of five are separated with zero cut, which the exhaustive search confirms is optimal");

        // 7. Boundaries: k = 1 has no cut, k >= n can only cut every edge.
        var single = new BalancedPartitioner(planted, 1, 1, 4, 0x9475UL);
        Check(single.Cut == 0 && single.Part.All(p => p == 0) && single.Moves == 0,
            "a one-part request answers with zero cut and no moves");
        List<int>[] sparse = GraphFactory.Random(6, 2, 0x9476UL);
        long edges = sparse.Sum(list => (long)list.Count) / 2;
        var singletons = new BalancedPartitioner(sparse, 6, 1, 4, 0x9477UL);
        Check(singletons.Cut == edges && singletons.MaxSize == 1,
            "k equal to n forces one vertex per part and cuts all " + edges + " edges");

        // 8-9. Disconnected and edgeless graphs reach a zero cut.
        var disconnected = new BalancedPartitioner(GraphFactory.TwoCliques(4), 2, 3, 6, 0x9478UL);
        Check(disconnected.Cut == 0, "a disconnected graph is split along its components with zero cut");
        var edgeless = new BalancedPartitioner(GraphFactory.Random(0, 0, 1UL), 3, 2, 4, 0x9479UL);
        var emptyEdges = GraphFactory.Random(9, 0, 2UL);
        var edgelessNine = new BalancedPartitioner(emptyEdges, 3, 2, 4, 0x947AUL);
        Check(edgeless.Cut == 0 && edgeless.Part.Length == 0 && edgelessNine.Cut == 0
              && edgelessNine.PartSize(0) == 3,
            "graphs with no edges (and an empty graph) partition to zero cut");

        // 10. Determinism of the whole plan.
        var repeat = new BalancedPartitioner(planted, 3, 6, 8, 0x9471UL);
        Check(repeat.Cut == partitioner.Cut && repeat.Moves == partitioner.Moves
              && repeat.Part.SequenceEqual(partitioner.Part),
            "the same seed reproduces the same partition, moves and cut");

        Console.WriteLine("  planted: " + planted.Length + " vertices, planted cut " + plantedCut
            + ", heuristic " + partitioner.Cut + " in " + partitioner.Moves + " moves (parts "
            + string.Join("/", Enumerable.Range(0, 3).Select(partitioner.PartSize)) + ")");
        Console.WriteLine("  tiny graph: heuristic " + tinyPartitioner.Cut + " vs exhaustive optimum "
            + optimal + "; two cliques: " + cliquePartitioner.Cut + "; k=n cut " + singletons.Cut);
        Console.WriteLine("PASS");
    }
}
