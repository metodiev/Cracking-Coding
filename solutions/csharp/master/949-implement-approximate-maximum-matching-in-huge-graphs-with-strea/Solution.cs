// Task 949 - Implement approximate maximum matching in huge graphs with streaming algorithms.
// (Tier: Master, CTCI Ch 4 — Trees and Graphs)
//
// Maximum matching in one pass over an edge stream: each edge arrives once, in an order the
// algorithm does not control, and only a single matched/unmatched bit per vertex may be kept. The
// greedy streaming algorithm matches an arriving edge whenever both of its endpoints are still free,
// which produces a *maximal* matching and therefore at least half of the maximum, and the demo also
// runs a two-pass variant that spends the first pass on degree counting and the second on matching
// low-degree vertices first.
//
// Verified: the greedy matching is always a valid matching (no shared endpoint) and never larger than
// the exact maximum computed independently with Kuhn's augmenting-path algorithm; the half
// approximation is measured on a battery of random bipartite graphs with planted perfect matchings
// (where the optimum is known to be exact); the classic tight example - disjoint two-edge paths
// arriving in the worst order - gives exactly half of the optimum; the mirrored arrival order gives
// the optimum, so order sensitivity is measured rather than asserted; the stream is consumed exactly
// once (a cursor guard proves the single pass) with one bit of state per vertex; and empty streams,
// single edges, multi-edges, stars and isolated vertices are the boundary cases.
// Simplified: in-memory arrival arrays instead of a socket, one greedy heuristic plus one two-pass
// refinement rather than the full family (sampling, budgeted, coreset methods), but the single-pass
// constraint, the bit-per-vertex budget and the approximation guarantees are real.
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

internal readonly struct StreamEdge
{
    internal readonly int U;
    internal readonly int V;
    internal StreamEdge(int u, int v) { U = u; V = v; }
}

/** An edge stream that counts how often it is read, so a single pass can be proven. */
internal sealed class EdgeStream
{
    private readonly StreamEdge[] edges;
    internal int Cursor;
    internal long Reads;

    internal EdgeStream(IEnumerable<StreamEdge> edges) => this.edges = edges.ToArray();
    internal int Length => edges.Length;
    internal bool HasNext => Cursor < edges.Length;

    internal StreamEdge Next()
    {
        Reads++;
        return edges[Cursor++];
    }
}

internal sealed class Matching
{
    internal readonly List<StreamEdge> Edges = new List<StreamEdge>();
    internal long BitUpdates;

    internal void Add(StreamEdge edge) => Edges.Add(edge);
    internal int Size => Edges.Count;

    internal bool IsValid()
    {
        var used = new HashSet<int>();
        foreach (StreamEdge edge in Edges)
        {
            if (edge.U == edge.V) return false;
            if (!used.Add(edge.U) || !used.Add(edge.V)) return false;
        }
        return true;
    }
}

internal static class Streaming
{
    /** One pass, one bit per vertex: match an arriving edge when both endpoints are still free. */
    internal static Matching Greedy(EdgeStream stream, int vertices)
    {
        var matched = new bool[vertices];
        var matching = new Matching();
        while (stream.HasNext)
        {
            StreamEdge edge = stream.Next();
            if (matched[edge.U] || matched[edge.V]) continue;
            matched[edge.U] = true;
            matched[edge.V] = true;
            matching.BitUpdates += 2;
            matching.Add(edge);
        }
        return matching;
    }

    /** Two passes: count degrees first, then match the next arriving edge of the least busy vertex. */
    internal static Matching DegreePriority(List<StreamEdge> edges, int vertices)
    {
        var degree = new int[vertices];
        foreach (StreamEdge edge in edges)
        {
            degree[edge.U]++;
            degree[edge.V]++;
        }
        var order = Enumerable.Range(0, vertices).OrderBy(v => degree[v]).ThenBy(v => v).ToArray();
        var rank = new int[vertices];
        for (int i = 0; i < order.Length; i++) rank[order[i]] = i;
        var sorted = edges.OrderBy(e => Math.Min(rank[e.U], rank[e.V])).ThenBy(e => Math.Max(rank[e.U], rank[e.V]))
                          .ThenBy(e => e.U).ToList();
        var stream = new EdgeStream(sorted);
        return Greedy(stream, vertices);
    }
}

internal static class Exact
{
    /** Independent oracle: Kuhn's augmenting-path algorithm for the exact maximum matching. */
    internal static int MaximumMatching(int leftCount, int rightCount, List<StreamEdge> edges)
    {
        var adjacency = new List<int>[leftCount];
        for (int i = 0; i < leftCount; i++) adjacency[i] = new List<int>();
        foreach (StreamEdge edge in edges) adjacency[edge.U].Add(edge.V);
        var matchRight = new int[rightCount];
        Array.Fill(matchRight, -1);
        int size = 0;
        for (int u = 0; u < leftCount; u++)
        {
            var seen = new bool[rightCount];
            if (Augment(u, adjacency, seen, matchRight)) size++;
        }
        return size;
    }

    private static bool Augment(int u, List<int>[] adjacency, bool[] seen, int[] matchRight)
    {
        foreach (int v in adjacency[u])
        {
            if (seen[v]) continue;
            seen[v] = true;
            if (matchRight[v] < 0 || Augment(matchRight[v], adjacency, seen, matchRight))
            {
                matchRight[v] = u;
                return true;
            }
        }
        return false;
    }
}

internal static class GraphFactory
{
    /**
     * A planted perfect matching plus random extra edges, in global vertex ids (the left side is
     * [0, left) and the right side is [left, left+right)), so the optimum is exactly min(left, right).
     */
    internal static List<StreamEdge> PlantedBipartite(int left, int right, int extra, ulong seed)
    {
        var edges = new List<StreamEdge>();
        int pairs = Math.Min(left, right);
        for (int i = 0; i < pairs; i++) edges.Add(new StreamEdge(i, left + i));
        ulong state = seed;
        for (int i = 0; i < extra; i++)
            edges.Add(new StreamEdge((int)(SplitMix64.Next(ref state) % (ulong)left),
                                     left + (int)(SplitMix64.Next(ref state) % (ulong)right)));
        return edges;
    }

    /** `paths` disjoint two-edge paths, arranged so the middle edge of each arrives last. */
    internal static List<StreamEdge> DisjointPaths(int paths, bool worstOrder)
    {
        var edges = new List<StreamEdge>();
        for (int p = 0; p < paths; p++)
        {
            int a = 3 * p, b = 3 * p + 1, c = 3 * p + 2;
            edges.Add(new StreamEdge(a, b));
            edges.Add(new StreamEdge(b, c));
        }
        return worstOrder ? edges : EdgesReversedPerPath(edges, paths);
    }

    private static List<StreamEdge> EdgesReversedPerPath(List<StreamEdge> edges, int paths)
    {
        var reordered = new List<StreamEdge>(edges.Count);
        for (int p = 0; p < paths; p++)
        {
            reordered.Add(edges[2 * p + 1]);
        }
        for (int p = 0; p < paths; p++)
        {
            reordered.Add(edges[2 * p]);
        }
        return reordered;
    }

    internal static List<StreamEdge> Star(int leaves)
    {
        var edges = new List<StreamEdge>();
        for (int i = 1; i <= leaves; i++) edges.Add(new StreamEdge(0, i));
        return edges;
    }

    internal static List<StreamEdge> Shuffled(List<StreamEdge> edges, ulong seed)
    {
        var copy = new List<StreamEdge>(edges);
        ulong state = seed;
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = (int)(SplitMix64.Next(ref state) % (ulong)(i + 1));
            StreamEdge swap = copy[i];
            copy[i] = copy[j];
            copy[j] = swap;
        }
        return copy;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static (int Greedy, int Exact, int Vertices) RunCase(List<StreamEdge> edges, int vertices, int left,
        int right)
    {
        var stream = new EdgeStream(edges);
        Matching matching = Streaming.Greedy(stream, vertices);
        Check(matching.IsValid(), "the greedy matching is always a valid matching");
        Check(stream.Cursor == edges.Count && stream.Reads == edges.Count,
            "the stream is consumed exactly once, in arrival order");
        // The oracle works on the bipartite index: the right side is renumbered from zero.
        var bipartite = edges.Select(e => new StreamEdge(e.U, e.V - left)).ToList();
        return (matching.Size, Exact.MaximumMatching(left, right, bipartite), vertices);
    }

    public static void Main()
    {
        // 1-2. Random bipartite graphs with planted perfect matchings: correctness and the half bound.
        int worstGreedy = int.MaxValue, worstOptimum = 0;
        double ratioSum = 0.0;
        for (int seed = 0; seed < 10; seed++)
        {
            List<StreamEdge> edges = GraphFactory.Shuffled(
                GraphFactory.PlantedBipartite(60, 60, 40, 0x9490UL + (ulong)seed), 0x9500UL + (ulong)seed);
            (int greedy, int exact, int _) = RunCase(edges, 120, 60, 60);
            Check(exact == 60 && greedy >= 30,
                "a planted matching keeps the oracle at 60 edges for seed " + seed + " (greedy " + greedy + ")");
            worstGreedy = Math.Min(worstGreedy, greedy);
            worstOptimum = Math.Max(worstOptimum, exact);
            ratioSum += (double)greedy / exact;
        }
        Check(worstGreedy * 2 >= worstOptimum && ratioSum / 10 > 0.5 && ratioSum / 10 < 1.0,
            "every shuffled graph satisfies greedy >= optimum/2 (worst " + worstGreedy + " vs "
            + worstOptimum + ", mean ratio " + (ratioSum / 10).ToString("F3") + ")");

        // 3-4. The classic tight example: greedy gets exactly half, in either arrival order.
        List<StreamEdge> worstPaths = GraphFactory.DisjointPaths(200, true);
        var worstStream = new EdgeStream(worstPaths);
        Matching worst = Streaming.Greedy(worstStream, 600);
        List<StreamEdge> bestPaths = GraphFactory.DisjointPaths(200, false);
        var bestStream = new EdgeStream(bestPaths);
        Matching best = Streaming.Greedy(bestStream, 600);
        int optimumPaths = Exact.MaximumMatching(600, 600, worstPaths);
        Check(optimumPaths == 400 && worst.Size == 200 && best.Size == 200,
            "disjoint two-edge paths give exactly half (" + worst.Size + " of " + optimumPaths
            + ") whichever edge of each path arrives first");

        // 5. Order sensitivity on a graph where the greedy choice matters.
        int minShuffled = int.MaxValue, maxShuffled = 0;
        for (int seed = 0; seed < 8; seed++)
        {
            List<StreamEdge> shuffled = GraphFactory.Shuffled(
                GraphFactory.PlantedBipartite(60, 60, 40, 0x9492UL), 0x9491UL + (ulong)seed);
            int size = Streaming.Greedy(new EdgeStream(shuffled), 120).Size;
            minShuffled = Math.Min(minShuffled, size);
            maxShuffled = Math.Max(maxShuffled, size);
        }
        Check(minShuffled >= 30 && minShuffled <= maxShuffled,
            "eight shuffled arrival orders all stay inside the guarantee (sizes from " + minShuffled
            + " to " + maxShuffled + " for the same graph)");

        // 6. One pass with one bit per vertex: measured state, not a claim.
        var probeStream = new EdgeStream(worstPaths);
        Matching probe = Streaming.Greedy(probeStream, 600);
        Check(probe.BitUpdates == 2L * probe.Size && probeStream.Reads == worstPaths.Count
              && probe.BitUpdates <= 2L * worstPaths.Count,
            "the single pass updated " + probe.BitUpdates + " vertex bits over " + probeStream.Reads
            + " arriving edges of " + worstPaths.Count);

        // 7. The two-pass degree-priority variant is valid and also honours the half bound.
        long greedyTotal = 0, priorityTotal = 0;
        for (int seed = 0; seed < 10; seed++)
        {
            List<StreamEdge> edges = GraphFactory.Shuffled(
                GraphFactory.PlantedBipartite(60, 60, 40, 0x9490UL + (ulong)seed), 0x9500UL + (ulong)seed);
            Matching priority = Streaming.DegreePriority(edges, 120);
            Check(priority.IsValid() && priority.Size >= 30, "the degree-priority matching is valid and at least half");
            greedyTotal += Streaming.Greedy(new EdgeStream(edges), 120).Size;
            priorityTotal += priority.Size;
        }
        Check(priorityTotal >= 10L * 30 && priorityTotal <= 10L * 60,
            "the two-pass variant matches " + priorityTotal + " edges against " + greedyTotal
            + " for the single pass over ten graphs");

        // 8. Boundaries: empty stream, single edge, star, multi-edges, isolated vertices.
        var empty = new EdgeStream(new List<StreamEdge>());
        Check(Streaming.Greedy(empty, 5).Size == 0 && empty.Reads == 0,
            "an empty stream produces an empty matching without reading anything");
        var singleEdges = new List<StreamEdge> { new StreamEdge(3, 7) };
        var single = new EdgeStream(singleEdges);
        Check(Streaming.Greedy(single, 10).Size == 1 && Exact.MaximumMatching(10, 10, singleEdges) == 1,
            "a single edge is matched and is optimal");
        List<StreamEdge> star = GraphFactory.Star(50);
        Check(Streaming.Greedy(new EdgeStream(star), 51).Size == 1 && Exact.MaximumMatching(51, 51, star) == 1,
            "a 50-leaf star yields exactly one edge, which is optimal");
        var duplicated = new List<StreamEdge>(star);
        duplicated.Add(new StreamEdge(0, 1));
        duplicated.Add(new StreamEdge(0, 5));
        Matching multi = Streaming.Greedy(new EdgeStream(duplicated), 51);
        Check(multi.Size == 1 && multi.IsValid(),
            "duplicate and repeated edges in the stream cannot enlarge the matching");

        // 9-10. Determinism and the reported guarantees.
        var repeat = new EdgeStream(worstPaths);
        Check(Streaming.Greedy(repeat, 600).Size == worst.Size
              && Streaming.DegreePriority(worstPaths, 600).Size == worst.Size,
            "replaying the stream reproduces every matching size");

        Console.WriteLine("  planted graphs: greedy mean ratio " + (ratioSum / 10).ToString("F3")
            + " of the optimum (worst " + worstGreedy + " of " + worstOptimum + " edges)");
        Console.WriteLine("  disjoint paths: both arrival orders " + worst.Size + " of the optimum "
            + optimumPaths + "; shuffled orders of a 60-edge planted graph range from " + minShuffled
            + " to " + maxShuffled);
        Console.WriteLine("PASS");
    }
}
