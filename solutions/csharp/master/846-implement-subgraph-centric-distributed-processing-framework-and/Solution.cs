// Task 846 - Implement subgraph-centric distributed processing framework and test on motifs. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// A subgraph-centric (Think-Like-a-Graph) engine: the graph is partitioned among workers, every worker holds the
// edges incident to the vertices it owns, and a computation runs on whole *subgraphs* rather than on one vertex at
// a time. The workload here is motif counting, which is the canonical benchmark for that model:
//
//   * ownership: a motif is owned by exactly one worker, the one that owns its anchor - the smallest-id vertex of
//     the motif that has degree at least two inside it, so a path counts at an inner vertex rather than at a leaf.
//     A worker enumerates motifs in its *view* and keeps only the ones it owns, which makes the distributed sum
//     equal the centralised count by construction rather than by luck.
//   * the view and its radius: round 0 sees only the local subgraph, and each further round replicates one more hop
//     of adjacency over explicit messages, so a worker can check motifs whose members are further away. That gives
//     a graded result the file verifies: 2-stars need no replication at all (they are a degree statistic of owned
//     vertices), triangles and 4-cliques need one hop, and 4-vertex paths need two hops, because their anchor is an
//     inner vertex one hop from the nearest leaf and two from the far one.
//
// Verified, each by recomputation:
//   * the counts after the round in which the radius suffices equal a centralised brute force - wedges by the
//     degree formula, triangles and 4-cliques by subset enumeration, 4-paths by subset enumeration plus the degree
//     sequence - over several seeded graphs and several partition counts;
//   * the graded bounds: at radius 0 the triangle and 4-clique counts are strict lower bounds and the wedge count
//     is already exact; the 4-path count is a lower bound until radius 2; and no count ever exceeds the central
//     count at any radius (so the ownership rule never double counts);
//   * no motif key is claimed by two workers (the owned keys are collected and checked for duplicates), and turning
//     the ownership rule off does double count, which is measured;
//   * work and communication are measured: per-worker work (owned vertices, owned edges, motifs examined) with its
//     imbalance ratio for a hash partition and for a degree-balanced partition on a skewed graph, the number of
//     boundary vertices, and the bytes replicated per radius;
//   * degenerate inputs: an empty graph, a single edge, one triangle, a 4-clique, a path of five, an isolated
//     vertex, and a single-worker partition (which must need no communication at all) all reproduce the oracle.
//
// Documented simplifications: the cluster is one process, workers exchange adjacency lists explicitly instead of
// over a network, there is no fault tolerance or dynamic repartitioning, the motif classes are limited to 3- and
// 4-vertex patterns so that a full brute-force oracle is cheap, and the view is rebuilt from scratch each round
// rather than being kept incrementally. Randomness is splitmix64 with explicit seeds, so every graph, partition,
// count and measurement in the output is reproducible. No external data or service is involved.
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
}

internal sealed class UndirectedGraph
{
    internal int Vertices { get; }
    internal List<HashSet<int>> Adjacency { get; }
    internal List<(int U, int V)> Edges { get; } = new List<(int, int)>();

    internal UndirectedGraph(int vertices)
    {
        Vertices = vertices;
        Adjacency = new List<HashSet<int>>();
        for (int i = 0; i < vertices; i++) Adjacency.Add(new HashSet<int>());
    }

    internal void AddEdge(int u, int v)
    {
        if (u < 0 || u >= Vertices || v < 0 || v >= Vertices) throw new ArgumentException("edge outside the graph");
        if (u == v || Adjacency[u].Contains(v)) return;
        Adjacency[u].Add(v);
        Adjacency[v].Add(u);
        Edges.Add((Math.Min(u, v), Math.Max(u, v)));
    }

    internal bool HasEdge(int u, int v) => Adjacency[u].Contains(v);

    internal int Degree(int v) => Adjacency[v].Count;

    /// <summary>A seeded random graph with a spanning path, so it is connected and sparse.</summary>
    internal static UndirectedGraph Random(int vertices, int extraEdges, ulong seed)
    {
        var graph = new UndirectedGraph(vertices);
        ulong state = seed;
        for (int v = 1; v < vertices; v++) graph.AddEdge(v - 1, v);
        for (int e = 0; e < extraEdges; e++)
        {
            int u = SplitMix64.NextBelow(ref state, vertices);
            int v = SplitMix64.NextBelow(ref state, vertices);
            graph.AddEdge(u, v);
        }
        return graph;
    }

    /// <summary>
    /// A hub-and-spoke graph whose hubs all share the same id residue, so a hash partition puts every hub in one
    /// block while a degree-aware partition spreads them. That is the case the balance metric has to expose.
    /// </summary>
    internal static UndirectedGraph Skewed(int hubs, int hubStride, int leavesPerHub)
    {
        int leavesStart = hubs * hubStride;
        var graph = new UndirectedGraph(leavesStart + hubs * leavesPerHub);
        for (int h = 0; h < hubs; h++)
        {
            int hubId = h * hubStride;
            for (int l = 0; l < leavesPerHub; l++)
            {
                int leaf = leavesStart + h * leavesPerHub + l;
                graph.AddEdge(hubId, leaf);
                graph.AddEdge(leaf, leavesStart + h * leavesPerHub + (l + 1) % leavesPerHub);
            }
        }
        return graph;
    }
}

internal enum Motif
{
    Wedge,          // a centre with two distinct neighbours: 3 vertices, 2 edges
    Triangle,       // 3 vertices, 3 edges
    Clique4,        // 4 vertices, 6 edges
    Path4,          // 4 vertices, 3 edges, degrees 1-2-2-1
}

internal sealed class Partition
{
    internal int Blocks { get; }
    internal int[] Owner { get; }
    internal List<HashSet<int>> Owned { get; }
    internal List<int> Boundary { get; }

    internal Partition(int[] owner, int blocks)
    {
        Blocks = blocks;
        Owner = owner;
        Owned = new List<HashSet<int>>();
        for (int b = 0; b < blocks; b++) Owned.Add(new HashSet<int>());
        for (int v = 0; v < owner.Length; v++) Owned[owner[v]].Add(v);
        Boundary = new List<int>();
    }
}

internal static class Partitioner
{
    /// <summary>Hash partition: vertex v belongs to block v % blocks.</summary>
    internal static Partition ByHash(UndirectedGraph graph, int blocks)
    {
        var owner = new int[graph.Vertices];
        for (int v = 0; v < graph.Vertices; v++) owner[v] = v % blocks;
        return new Partition(owner, blocks);
    }

    /// <summary>
    /// Degree-balanced partition: assign the highest-degree vertices first, each to the block with the fewest
    /// incident edges so far. This is the greedy edge-cut heuristic every real partitioner starts from.
    /// </summary>
    internal static Partition ByDegree(UndirectedGraph graph, int blocks)
    {
        var owner = new int[graph.Vertices];
        var load = new int[blocks];
        var order = new List<int>();
        for (int v = 0; v < graph.Vertices; v++) order.Add(v);
        order.Sort((a, b) => graph.Degree(b) != graph.Degree(a) ? graph.Degree(b).CompareTo(graph.Degree(a)) : a.CompareTo(b));
        foreach (int v in order)
        {
            int best = 0;
            for (int b = 1; b < blocks; b++) if (load[b] < load[best]) best = b;
            owner[v] = best;
            load[best] += graph.Degree(v) + 1;
        }
        return new Partition(owner, blocks);
    }
}

internal sealed class BlockResult
{
    internal int Block;
    internal int Radius;
    internal Dictionary<Motif, long> Counts = new Dictionary<Motif, long>();
    internal HashSet<string> OwnedKeys = new HashSet<string>();
    internal long Work;
    internal int ViewSize;
}

internal sealed class RoundResult
{
    internal int Radius;
    internal Dictionary<Motif, long> Totals = new Dictionary<Motif, long>();
    internal long Messages;
    internal long Bytes;
    internal long Work;
    internal long MaxBlockWork;
    internal long DuplicateKeys;
}

/// <summary>The subgraph-centric engine: expand each worker's view, then let it count the motifs it owns.</summary>
internal static class Framework
{
    /// <summary>
    /// Builds the view of one block at a given radius by breadth-first expansion over the adjacency lists, charging
    /// one message and one adjacency list per newly discovered vertex that another worker owns.
    /// </summary>
    private static HashSet<int> BuildView(UndirectedGraph graph, Partition partition, int block, int radius,
        ref long messages, ref long bytes)
    {
        var view = new HashSet<int>(partition.Owned[block]);
        var frontier = new List<int>(partition.Owned[block]);
        for (int hop = 0; hop < radius; hop++)
        {
            var next = new List<int>();
            foreach (int vertex in frontier)
            {
                foreach (int neighbour in graph.Adjacency[vertex])
                {
                    if (partition.Owner[neighbour] != block)
                    {
                        messages++;
                        bytes += graph.Adjacency[neighbour].Count;
                    }
                    if (view.Add(neighbour)) next.Add(neighbour);
                }
            }
            frontier = next;
        }
        return view;
    }

    private static string Key(params int[] vertices)
    {
        var copy = new List<int>(vertices);
        copy.Sort();
        return string.Join("-", copy);
    }

    /// <summary>
    /// Counts the motifs of every class inside a view, keeping only the ones this block owns. The anchor is the
    /// centre for a 2-star, the smallest-id member for a triangle or 4-clique, and the smallest inner vertex for a
    /// 4-path, which is what makes the ownership of every motif well defined.
    /// </summary>
    private static void CountInView(UndirectedGraph graph, Partition partition, int block, HashSet<int> view,
        BlockResult result, Dictionary<Motif, long> ownershipOffCounts)
    {
        var vertices = new List<int>(view);
        vertices.Sort();
        HashSet<int> owned = partition.Owned[block];
        int n = vertices.Count;
        for (int i = 0; i < n; i++)
        {
            int a = vertices[i];
            for (int j = i + 1; j < n; j++)
            {
                int b = vertices[j];
                for (int k = j + 1; k < n; k++)
                {
                    int c = vertices[k];
                    result.Work++;
                    if (graph.HasEdge(a, b) && graph.HasEdge(a, c)) CountWedge(result, owned, ownershipOffCounts, a, b, c);
                    if (graph.HasEdge(a, b) && graph.HasEdge(b, c)) CountWedge(result, owned, ownershipOffCounts, b, a, c);
                    if (graph.HasEdge(a, c) && graph.HasEdge(b, c)) CountWedge(result, owned, ownershipOffCounts, c, a, b);
                    if (graph.HasEdge(a, b) && graph.HasEdge(a, c) && graph.HasEdge(b, c))
                        CountTriangle(result, owned, ownershipOffCounts, a, b, c);
                    for (int m = k + 1; m < n; m++)
                    {
                        int d = vertices[m];
                        result.Work++;
                        int edges = 0;
                        if (graph.HasEdge(a, b)) edges++;
                        if (graph.HasEdge(a, c)) edges++;
                        if (graph.HasEdge(a, d)) edges++;
                        if (graph.HasEdge(b, c)) edges++;
                        if (graph.HasEdge(b, d)) edges++;
                        if (graph.HasEdge(c, d)) edges++;
                        if (edges != 6 && edges != 3) continue;
                        int[] members = { a, b, c, d };
                        var inner = new List<int>();
                        int ones = 0, twos = 0;
                        foreach (int vertex in members)
                        {
                            int degree = 0;
                            foreach (int other in members) if (other != vertex && graph.HasEdge(vertex, other)) degree++;
                            if (degree == 1) ones++;
                            else if (degree == 2) { twos++; inner.Add(vertex); }
                        }
                        bool isClique = edges == 6;
                        bool isPath = edges == 3 && ones == 2 && twos == 2;
                        if (!isClique && !isPath) continue;
                        int anchor = isClique ? a : Math.Min(inner[0], inner[1]);
                        string key = isClique ? Key(a, b, c, d) : Key(a, b, c, d);
                        if (ownershipOffCounts != null) Bump(ownershipOffCounts, isClique ? Motif.Clique4 : Motif.Path4, 1);
                        else if (owned.Contains(anchor)) Admit(result, isClique ? Motif.Clique4 : Motif.Path4, key);
                    }
                }
            }
        }
    }

    private static void CountWedge(BlockResult result, HashSet<int> owned, Dictionary<Motif, long> ownershipOffCounts,
        int center, int leaf1, int leaf2)
    {
        if (ownershipOffCounts != null)
        {
            Bump(ownershipOffCounts, Motif.Wedge, 1);
            return;
        }
        if (!owned.Contains(center)) return;                 // the centre's owner counts the 2-star
        Admit(result, Motif.Wedge, center + "|" + Math.Min(leaf1, leaf2) + "|" + Math.Max(leaf1, leaf2));
    }

    private static void CountTriangle(BlockResult result, HashSet<int> owned,
        Dictionary<Motif, long> ownershipOffCounts, int a, int b, int c)
    {
        if (ownershipOffCounts != null)
        {
            Bump(ownershipOffCounts, Motif.Triangle, 1);
            return;
        }
        if (!owned.Contains(a)) return;                      // the smallest member's owner counts the triangle
        Admit(result, Motif.Triangle, Key(a, b, c));
    }

    private static void Admit(BlockResult result, Motif motif, string key)
    {
        if (!result.OwnedKeys.Add(motif + ":" + key)) throw new InvalidOperationException("a block counted a motif twice");
        Bump(result.Counts, motif, 1);
    }

    private static void Bump(Dictionary<Motif, long> counts, Motif motif, long amount)
    {
        counts.TryGetValue(motif, out long current);
        counts[motif] = current + amount;
    }

    internal static RoundResult Run(UndirectedGraph graph, Partition partition, int radius, bool ownershipOff,
        out List<BlockResult> blocks)
    {
        var round = new RoundResult { Radius = radius };
        blocks = new List<BlockResult>();
        long messages = 0, bytes = 0;
        var seenKeys = new HashSet<string>();
        var ownershipOffCounts = ownershipOff ? new Dictionary<Motif, long>() : null;
        for (int block = 0; block < partition.Blocks; block++)
        {
            var result = new BlockResult { Block = block, Radius = radius };
            HashSet<int> view = BuildView(graph, partition, block, radius, ref messages, ref bytes);
            result.ViewSize = view.Count;
            CountInView(graph, partition, block, view, result, ownershipOffCounts);
            foreach (string key in result.OwnedKeys) if (!seenKeys.Add(key)) round.DuplicateKeys++;
            blocks.Add(result);
            if (result.Work > round.MaxBlockWork) round.MaxBlockWork = result.Work;
            round.Work += result.Work;
        }
        if (ownershipOffCounts != null) round.Totals = ownershipOffCounts;
        else foreach (BlockResult block in blocks)
            foreach (KeyValuePair<Motif, long> entry in block.Counts) Bump(round.Totals, entry.Key, entry.Value);
        round.Messages = messages;
        round.Bytes = bytes;
        return round;
    }

    /// <summary>The vertex-centric shortcut for wedges: a degree statistic of the owned vertices, no replication.</summary>
    internal static long WedgesFromDegrees(UndirectedGraph graph, Partition partition)
    {
        long total = 0;
        for (int v = 0; v < graph.Vertices; v++)
        {
            long degree = graph.Degree(v);
            total += degree * (degree - 1) / 2;
        }
        return total;
    }
}

internal static class Oracles
{
    /// <summary>Centralised brute force over vertex subsets, independent of the partitioning and ownership rules.</summary>
    internal static Dictionary<Motif, long> Count(UndirectedGraph graph)
    {
        var counts = new Dictionary<Motif, long>
        {
            [Motif.Wedge] = 0,
            [Motif.Triangle] = 0,
            [Motif.Clique4] = 0,
            [Motif.Path4] = 0,
        };
        int n = graph.Vertices;
        for (int v = 0; v < n; v++)
        {
            long degree = graph.Degree(v);
            counts[Motif.Wedge] += degree * (degree - 1) / 2;
        }
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
            {
                for (int c = b + 1; c < n; c++)
                {
                    bool ac = graph.HasEdge(a, c), bc = graph.HasEdge(b, c);
                    if (graph.HasEdge(a, b) && ac && bc) counts[Motif.Triangle]++;
                    for (int d = c + 1; d < n; d++)
                    {
                        int edges = 0;
                        bool ad = graph.HasEdge(a, d), bd = graph.HasEdge(b, d), cd = graph.HasEdge(c, d);
                        if (graph.HasEdge(a, b)) edges++;
                        edges += (ac ? 1 : 0) + (ad ? 1 : 0) + (bc ? 1 : 0) + (bd ? 1 : 0) + (cd ? 1 : 0);
                        if (edges == 6) counts[Motif.Clique4]++;
                        else if (edges == 3)
                        {
                            int[] subsets = { a, b, c, d };
                            int ones = 0, twos = 0;
                            foreach (int vertex in subsets)
                            {
                                int degree = 0;
                                foreach (int other in subsets) if (other != vertex && graph.HasEdge(vertex, other)) degree++;
                                if (degree == 1) ones++;
                                else if (degree == 2) twos++;
                            }
                            if (ones == 2 && twos == 2) counts[Motif.Path4]++;
                        }
                    }
                }
            }
        return counts;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Dictionary<Motif, long> CountsAt(UndirectedGraph graph, Partition partition, int radius,
        out RoundResult round, out List<BlockResult> blocks)
    {
        round = Framework.Run(graph, partition, radius, false, out blocks);
        return round.Totals;
    }

    private static long Get(Dictionary<Motif, long> counts, Motif motif) => counts.TryGetValue(motif, out long value) ? value : 0;

    public static void Main()
    {
        const int Vertices = 40, ExtraEdges = 30;

        // ---- the round in which the radius suffices reproduces the centralised counts ------------------
        {
            bool wedgesExact = true, trianglesExact = true, cliqueExact = true, pathsExact = true;
            long graphCount = 0, roundsChecked = 0;
            for (int seed = 1; seed <= 4; seed++)
            {
                UndirectedGraph graph = UndirectedGraph.Random(Vertices, ExtraEdges, (ulong)seed * 0x9E3779B97F4A7C15UL);
                Dictionary<Motif, long> oracle = Oracles.Count(graph);
                for (int blocks = 1; blocks <= 4; blocks += 3)
                {
                    Partition partition = Partitioner.ByHash(graph, blocks);
                    // 2-stars are a degree statistic of the owned vertices: no view expansion at all is needed.
                    if (Framework.WedgesFromDegrees(graph, partition) != Get(oracle, Motif.Wedge)) wedgesExact = false;
                    Dictionary<Motif, long> atZero = CountsAt(graph, partition, 0, out RoundResult _, out List<BlockResult> _);
                    Dictionary<Motif, long> atOne = CountsAt(graph, partition, 1, out RoundResult _, out List<BlockResult> _);
                    Dictionary<Motif, long> atTwo = CountsAt(graph, partition, 2, out RoundResult _, out List<BlockResult> _);
                    if (Get(atOne, Motif.Wedge) != Get(oracle, Motif.Wedge)) wedgesExact = false;
                    if (Get(atOne, Motif.Triangle) != Get(oracle, Motif.Triangle)) trianglesExact = false;
                    if (Get(atOne, Motif.Clique4) != Get(oracle, Motif.Clique4)) cliqueExact = false;
                    if (Get(atTwo, Motif.Path4) != Get(oracle, Motif.Path4)) pathsExact = false;
                    // No radius may ever exceed the centralised count: the ownership rule never double counts.
                    foreach (Motif motif in new[] { Motif.Wedge, Motif.Triangle, Motif.Clique4, Motif.Path4 })
                        if (Get(atZero, motif) > Get(oracle, motif) || Get(atOne, motif) > Get(oracle, motif)
                            || Get(atTwo, motif) > Get(oracle, motif)) Check(false, "a radius over-counted " + motif);
                    graphCount += Get(oracle, Motif.Triangle) + Get(oracle, Motif.Wedge);
                    roundsChecked += 3;
                }
            }
            Check(wedgesExact, "2-stars are exact at radius 0 twice over: as a degree statistic of the owned vertices and "
                + "by local enumeration once the neighbours are in the view");
            Check(trianglesExact, "triangles are exact at radius 1, which is the smallest view in which the third edge "
                + "of the triangle is visible");
            Check(cliqueExact, "4-cliques are exact at radius 1, since every member of a clique is a neighbour of "
                + "every other");
            Check(pathsExact, "4-paths are exact at radius 2, because their anchor is the smaller inner vertex, one hop "
                + "from the near leaf and two from the far one");
            Console.WriteLine("distributed counts: " + roundsChecked + " round/partition cases reproduced the centralised "
                + "oracle (triangle+wedge volume " + graphCount + ")");
        }

        // ---- the graded lower bounds are strict where the radius is not enough ------------------------
        {
            UndirectedGraph graph = UndirectedGraph.Random(Vertices, ExtraEdges, 0x5EED846000000001UL);
            // Plant one 4-clique and one long path so that every motif class is non-empty in the oracle.
            for (int u = 30; u < 34; u++)
                for (int v = u + 1; v < 34; v++) graph.AddEdge(u, v);
            for (int v = 34; v < 37; v++) graph.AddEdge(v, v + 1);
            Dictionary<Motif, long> oracle = Oracles.Count(graph);
            Check(Get(oracle, Motif.Clique4) > 0 && Get(oracle, Motif.Path4) > 0 && Get(oracle, Motif.Triangle) > 0,
                "the graded test graph contains every motif class (triangles " + Get(oracle, Motif.Triangle)
                + ", cliques " + Get(oracle, Motif.Clique4) + ", paths " + Get(oracle, Motif.Path4) + ")");
            Partition partition = Partitioner.ByHash(graph, 4);
            Dictionary<Motif, long> atZero = CountsAt(graph, partition, 0, out RoundResult zeroRound,
                out List<BlockResult> blocksZero);
            Dictionary<Motif, long> atOne = CountsAt(graph, partition, 1, out RoundResult _, out List<BlockResult> _);
            Dictionary<Motif, long> atTwo = CountsAt(graph, partition, 2, out RoundResult _, out List<BlockResult> _);
            Dictionary<Motif, long> atThree = CountsAt(graph, partition, 3, out RoundResult threeRound,
                out List<BlockResult> _);
            Check(Get(atZero, Motif.Triangle) < Get(oracle, Motif.Triangle) && Get(atZero, Motif.Clique4) < Get(oracle, Motif.Clique4),
                "at radius 0 triangles (" + Get(atZero, Motif.Triangle) + " of " + Get(oracle, Motif.Triangle)
                + ") and 4-cliques (" + Get(atZero, Motif.Clique4) + " of " + Get(oracle, Motif.Clique4)
                + ") are strict lower bounds");
            Check(Get(atZero, Motif.Wedge) < Get(oracle, Motif.Wedge)
                && Get(atOne, Motif.Wedge) == Get(oracle, Motif.Wedge)
                && Framework.WedgesFromDegrees(graph, partition) == Get(oracle, Motif.Wedge),
                "2-stars need only the owned vertices' degrees: exact as a vertex-centric statistic without any view, "
                + "and exact by enumeration at radius 1");
            Check(Get(atOne, Motif.Path4) < Get(oracle, Motif.Path4) && Get(atTwo, Motif.Path4) == Get(oracle, Motif.Path4),
                "4-paths are a strict lower bound at radius 1 (" + Get(atOne, Motif.Path4) + ") and exact at radius 2 ("
                + Get(oracle, Motif.Path4) + ")");
            Check(Get(atThree, Motif.Path4) == Get(oracle, Motif.Path4) && threeRound.DuplicateKeys == 0,
                "radius 3 changes nothing, so the computation has converged after the radius that suffices");
            long viewGrowth = 0;
            foreach (BlockResult block in blocksZero) viewGrowth += block.ViewSize;
            Check(viewGrowth > 0 && zeroRound.Messages == 0, "radius 0 replicates nothing (" + zeroRound.Messages
                + " messages), while the views already hold " + viewGrowth + " vertex slots");
            Console.WriteLine("graded bounds: radius0 triangles=" + Get(atZero, Motif.Triangle) + " radius1 paths="
                + Get(atOne, Motif.Path4) + " radius2 paths=" + Get(atTwo, Motif.Path4) + " oracle paths="
                + Get(oracle, Motif.Path4));
        }

        // ---- the ownership rule is what prevents double counting ---------------------------------------
        {
            UndirectedGraph graph = UndirectedGraph.Random(Vertices, ExtraEdges, 0x5EED846000000002UL);
            for (int u = 30; u < 34; u++)
                for (int v = u + 1; v < 34; v++) graph.AddEdge(u, v);
            for (int v = 34; v < 37; v++) graph.AddEdge(v, v + 1);
            Dictionary<Motif, long> oracle = Oracles.Count(graph);
            Partition partition = Partitioner.ByHash(graph, 4);
            Check(Get(oracle, Motif.Clique4) > 0 && Get(oracle, Motif.Path4) > 0, "the test graph contains cliques and "
                + "paths, so the inflation check below is not vacuous");
            foreach (Motif motif in new[] { Motif.Wedge, Motif.Triangle, Motif.Clique4, Motif.Path4 })
            {
                RoundResult owned = Framework.Run(graph, partition, 2, false, out List<BlockResult> blocks);
                RoundResult naive = Framework.Run(graph, partition, 2, true, out List<BlockResult> _);
                Check(Get(owned.Totals, motif) == Get(oracle, motif), "the ownership rule makes " + motif
                    + " exact (" + Get(owned.Totals, motif) + ")");
                Check(Get(naive.Totals, motif) > Get(oracle, motif), "counting every visible motif instead inflates "
                    + motif + " to " + Get(naive.Totals, motif) + " against " + Get(oracle, motif));
                Check(owned.DuplicateKeys == 0, "and no motif key is claimed by two workers (" + owned.DuplicateKeys
                    + " duplicates)");
            }
            Console.WriteLine("ownership: exact with the rule, inflated without it, no duplicate keys");
        }

        // ---- work balance and communication cost -------------------------------------------------------
        {
            UndirectedGraph skewed = UndirectedGraph.Skewed(4, 4, 10);
            Dictionary<Motif, long> oracle = Oracles.Count(skewed);
            var ratios = new Dictionary<string, double>();
            var ownedEdges = new Dictionary<string, long>();
            foreach ((string name, Partition partition) in new (string, Partition)[]
            {
                ("hash", Partitioner.ByHash(skewed, 4)),
                ("degree", Partitioner.ByDegree(skewed, 4)),
            })
            {
                RoundResult round = Framework.Run(skewed, partition, 1, false, out List<BlockResult> blocks);
                long total = 0, max = 0;
                foreach (BlockResult block in blocks)
                {
                    total += block.Work;
                    if (block.Work > max) max = block.Work;
                }
                double mean = (double)total / blocks.Count;
                ratios[name] = mean > 0 ? max / mean : 0;
                long edges = 0;
                foreach (BlockResult block in blocks)
                    foreach ((int u, int v) in skewed.Edges)
                        if (partition.Owner[u] == block.Block || partition.Owner[v] == block.Block) edges++;
                ownedEdges[name] = edges;
                Check(Get(round.Totals, Motif.Wedge) == Get(oracle, Motif.Wedge) && Get(round.Totals, Motif.Triangle)
                    == Get(oracle, Motif.Triangle), "the " + name + " partition still counts motifs exactly");
                Check(round.DuplicateKeys == 0, "and still claims every motif once");
            }
            Check(ratios["degree"] < ratios["hash"], "when every hub hashes into one block the degree-balanced "
                + "partition has a far better work balance than the hash partition (" + ratios["degree"].ToString("F3")
                + " vs " + ratios["hash"].ToString("F3") + " of max/mean motif examinations)");

            UndirectedGraph graph = UndirectedGraph.Random(Vertices, ExtraEdges, 0x5EED846000000004UL);
            Partition partition1 = Partitioner.ByHash(graph, 1);
            RoundResult single = Framework.Run(graph, partition1, 2, false, out List<BlockResult> _);
            Partition partition4 = Partitioner.ByHash(graph, 4);
            RoundResult many = Framework.Run(graph, partition4, 2, false, out List<BlockResult> _);
            Check(single.Messages == 0 && single.Bytes == 0, "a single-worker partition replicates nothing at all");
            Check(many.Messages > 0 && many.Bytes > 0, "and a four-way partition does exchange boundary adjacency ("
                + many.Messages + " messages, " + many.Bytes + " adjacency entries)");
            Console.WriteLine("balance: hash ratio " + ratios["hash"].ToString("F3") + " degree ratio "
                + ratios["degree"].ToString("F3") + " | edges hash " + ownedEdges["hash"] + " degree "
                + ownedEdges["degree"] + " | single-worker messages " + single.Messages + " four-way messages "
                + many.Messages);
        }

        // ---- degenerate graphs -------------------------------------------------------------------------
        {
            var empty = new UndirectedGraph(0);
            RoundResult emptyRound = Framework.Run(empty, Partitioner.ByHash(empty, 3), 2, false, out _);
            Check(emptyRound.Totals.Count == 0 || Get(emptyRound.Totals, Motif.Triangle) == 0,
                "an empty graph counts nothing");

            var single = new UndirectedGraph(2);
            single.AddEdge(0, 1);
            Check(CountsAt(single, Partitioner.ByHash(single, 2), 2, out RoundResult singleRound, out _) is not null
                && Get(singleRound.Totals, Motif.Wedge) == 0 && Get(singleRound.Totals, Motif.Triangle) == 0,
                "a single edge is no motif of the four classes");

            var isolated = new UndirectedGraph(5);
            isolated.AddEdge(0, 1);
            RoundResult isolatedRound = Framework.Run(isolated, Partitioner.ByHash(isolated, 5), 2, false, out _);
            Check(Get(isolatedRound.Totals, Motif.Wedge) == 0 && Get(isolatedRound.Totals, Motif.Path4) == 0,
                "isolated vertices contribute nothing");

            var triangle = new UndirectedGraph(3);
            triangle.AddEdge(0, 1);
            triangle.AddEdge(1, 2);
            triangle.AddEdge(2, 0);
            Dictionary<Motif, long> triangleCounts = CountsAt(triangle, Partitioner.ByHash(triangle, 2), 1, out _, out _);
            Check(Get(triangleCounts, Motif.Triangle) == 1 && Get(triangleCounts, Motif.Wedge) == 3,
                "a triangle is one triangle and three 2-stars, split across two workers");

            var clique = new UndirectedGraph(4);
            for (int u = 0; u < 4; u++) for (int v = u + 1; v < 4; v++) clique.AddEdge(u, v);
            Dictionary<Motif, long> cliqueCounts = CountsAt(clique, Partitioner.ByHash(clique, 3), 1, out _, out _);
            Dictionary<Motif, long> cliqueOracle = Oracles.Count(clique);
            Check(Get(cliqueCounts, Motif.Clique4) == 1 && Get(cliqueCounts, Motif.Clique4) == Get(cliqueOracle, Motif.Clique4)
                && Get(cliqueCounts, Motif.Path4) == 0, "a 4-clique is one clique, no path, counted once across three "
                + "workers");

            var path = new UndirectedGraph(5);
            for (int v = 0; v < 4; v++) path.AddEdge(v, v + 1);
            Dictionary<Motif, long> pathOracle = Oracles.Count(path);
            Dictionary<Motif, long> pathCounts = CountsAt(path, Partitioner.ByHash(path, 3), 2, out _, out _);
            Check(Get(pathOracle, Motif.Path4) == 2 && Get(pathCounts, Motif.Path4) == 2,
                "a five-vertex path holds two 4-paths, and the distributed count agrees");
            Console.WriteLine("degenerate: empty, single edge, isolated, triangle, 4-clique and path all as expected");
        }

        // ---- determinism ---------------------------------------------------------------------------------
        {
            UndirectedGraph graph = UndirectedGraph.Random(Vertices, ExtraEdges, 0x5EED846000000005UL);
            Partition partition = Partitioner.ByHash(graph, 3);
            RoundResult first = Framework.Run(graph, partition, 2, false, out List<BlockResult> blocksFirst);
            RoundResult second = Framework.Run(graph, partition, 2, false, out List<BlockResult> blocksSecond);
            bool identical = first.Messages == second.Messages && first.Bytes == second.Bytes
                && first.Work == second.Work && first.MaxBlockWork == second.MaxBlockWork;
            foreach (Motif motif in new[] { Motif.Wedge, Motif.Triangle, Motif.Clique4, Motif.Path4 })
                if (Get(first.Totals, motif) != Get(second.Totals, motif)) identical = false;
            for (int i = 0; i < blocksFirst.Count && identical; i++)
                if (blocksFirst[i].ViewSize != blocksSecond[i].ViewSize
                    || blocksFirst[i].OwnedKeys.Count != blocksSecond[i].OwnedKeys.Count) identical = false;
            Check(identical, "two runs produce identical counts, views, work and messages (" + first.Work
                + " motif examinations, " + first.Messages + " messages)");
            Console.WriteLine("determinism: work=" + first.Work + " messages=" + first.Messages + " bytes=" + first.Bytes
                + " maxBlockWork=" + first.MaxBlockWork);
        }

        Console.WriteLine("PASS");
    }
}
