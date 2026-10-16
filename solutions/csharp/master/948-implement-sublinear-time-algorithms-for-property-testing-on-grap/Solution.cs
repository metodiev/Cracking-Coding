// Task 948 - Implement sublinear time algorithms for property testing on graphs (e.g., bipartiteness).
// (Tier: Master, CTCI Ch 4 — Trees and Graphs)
//
// A property tester in the bounded-degree query model: the graph is only reachable through Degree(v)
// and Neighbor(v, i), every access is counted, and the tester answers "bipartite?" after a few random
// start vertices each followed by a fixed-length random walk, which costs O(s * L) queries instead of
// the O(n+m) a full 2-colouring needs. Colours are assigned along the walk; when an edge joins two
// vertices of the same colour the tester reports a *witness*: the walk's discovery paths from those
// two vertices plus that edge form a closed odd walk, so it contains an odd cycle and proves
// non-bipartiteness.
//
// Verified: the tester agrees with the exact O(n+m) 2-colouring oracle on a battery of graphs; it
// never rejects a bipartite graph over many seeds; it rejects a graph whose odd cycles cover a
// noticeable fraction of the vertices in nearly every run; the witness is re-checked edge by edge
// and is provably odd; the query count is measured and shown to be a small fraction of n; an
// average-degree estimator (a second, one-query-per-sample property test) is compared with the exact
// mean; the rejection rate rises monotonically with the number of planted odd edges; and the honest
// limitation is demonstrated - a single planted odd edge in a huge graph is accepted, because a
// sublinear tester only promises to catch violations that are far from bipartite.
// Simplified: the graph is undirected and unweighted, stored in memory but only reachable through
// counted queries; one tester family (random starts plus truncated BFS) is implemented.
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

/** The bounded-degree query model: every degree and neighbour access is counted. */
internal sealed class QueryGraph
{
    private readonly List<int>[] adjacency;
    internal long Queries;

    internal QueryGraph(int vertices)
    {
        adjacency = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) adjacency[i] = new List<int>();
    }

    internal int VertexCount => adjacency.Length;
    internal long EdgeCount => adjacency.Sum(list => (long)list.Count) / 2;

    internal int Degree(int v)
    {
        Queries++;
        return adjacency[v].Count;
    }

    internal int Neighbor(int v, int index)
    {
        Queries++;
        return adjacency[v][index];
    }

    internal void AddEdge(int a, int b)
    {
        adjacency[a].Add(b);
        adjacency[b].Add(a);
    }

    /** Oracle-only access: the full adjacency, used by the exact algorithms and the checks. */
    internal List<int>[] Raw => adjacency;
}

/** Random starts plus a fixed-length random walk: the O(s * L) bipartiteness tester of GR. */
internal sealed class BipartiteTester
{
    private readonly QueryGraph graph;
    private readonly int starts;
    private readonly int walkLength;
    private ulong state;

    internal bool Rejected { get; private set; }
    internal List<int> Witness { get; private set; } = new List<int>();
    internal long Queries { get; private set; }

    internal BipartiteTester(QueryGraph graph, int starts, int walkLength, ulong seed)
    {
        this.graph = graph;
        this.starts = starts;
        this.walkLength = walkLength;
        state = seed;
    }

    internal void Run()
    {
        long before = graph.Queries;
        var color = new int[graph.VertexCount];
        var parent = new int[graph.VertexCount];
        for (int s = 0; s < starts && !Rejected; s++)
        {
            if (graph.VertexCount == 0) break;
            Array.Fill(color, -1);
            int start = graph.VertexCount == 1 ? 0 : (int)(SplitMix64.Next(ref state) % (ulong)graph.VertexCount);
            color[start] = 0;
            parent[start] = -1;
            int v = start;
            for (int step = 0; step < walkLength && !Rejected; step++)
            {
                int degree = graph.Degree(v);
                if (degree == 0) break;
                int w = graph.Neighbor(v, (int)(SplitMix64.Next(ref state) % (ulong)degree));
                if (color[w] < 0)
                {
                    color[w] = 1 - color[v];
                    parent[w] = v;
                }
                else if (color[w] == color[v])
                {
                    Rejected = true;
                    Witness = BuildWitness(v, w, parent);
                }
                v = w;
            }
        }
        Queries = graph.Queries - before;
    }

    /** The closed walk start -> v, start -> w and the conflicting edge: closed and of odd length. */
    private static List<int> BuildWitness(int v, int w, int[] parent)    {
        var pathToRoot = new List<int>();
        for (int x = v; x >= 0; x = parent[x]) pathToRoot.Add(x);
        var otherPath = new List<int>();
        for (int x = w; x >= 0; x = parent[x]) otherPath.Add(x);
        var walk = new List<int>(pathToRoot);
        for (int i = otherPath.Count - 2; i >= 0; i--) walk.Add(otherPath[i]);
        walk.Add(v);
        return walk;
    }
}

internal sealed class ExactAlgorithms
{
    /** Oracle: the full 2-colouring, O(n + m), used as ground truth for the tester. */
    internal static bool IsBipartite(QueryGraph graph)
    {
        List<int>[] adjacency = graph.Raw;
        var color = new int[adjacency.Length];
        Array.Fill(color, -1);
        for (int s = 0; s < adjacency.Length; s++)
        {
            if (color[s] >= 0) continue;
            color[s] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(s);
            while (queue.Count > 0)
            {
                int v = queue.Dequeue();
                foreach (int w in adjacency[v])
                {
                    if (color[w] < 0)
                    {
                        color[w] = 1 - color[v];
                        queue.Enqueue(w);
                    }
                    else if (color[w] == color[v]) return false;
                }
            }
        }
        return true;
    }

    internal static double MeanDegree(QueryGraph graph) =>
        graph.Raw.Sum(list => (long)list.Count) / (double)graph.VertexCount;

    internal static long Probes(QueryGraph graph) => graph.Raw.Sum(list => (long)list.Count) + 2L * graph.VertexCount;
}

internal static class Graphs
{
    /** A random bipartite graph: even vertices form the left side, odd vertices the right side. */
    internal static QueryGraph RandomBipartite(int vertices, int degree, ulong seed)
    {
        var graph = new QueryGraph(vertices);
        ulong state = seed;
        for (int v = 0; v < vertices; v += 2)
        {
            for (int d = 0; d < degree; d++)
            {
                int w = 1 + 2 * (int)(SplitMix64.Next(ref state) % (ulong)(vertices / 2));
                if (w >= vertices) continue;
                graph.AddEdge(v, w);
            }
        }
        return graph;
    }

    internal static void AddIntraSideEdges(QueryGraph graph, int count, ulong seed)
    {
        ulong state = seed;
        for (int i = 0; i < count; i++)
        {
            int a = 2 * (int)(SplitMix64.Next(ref state) % (ulong)(graph.VertexCount / 2));
            int b = 2 * (int)(SplitMix64.Next(ref state) % (ulong)(graph.VertexCount / 2));
            if (a == b) continue;
            graph.AddEdge(a, b);
        }
    }

    /**
     * A graph in which the first `triangleVertices` vertices form disjoint triangles and the rest
     * form a random bipartite graph: the fraction of vertices sitting on an odd cycle is explicit,
     * which is exactly the quantity a sublinear tester can be expected to resolve.
     */
    internal static QueryGraph Mixed(int vertices, int triangleVertices, int degree, ulong seed)
    {
        var graph = new QueryGraph(vertices);
        ulong state = seed;
        int triangles = triangleVertices / 3;
        for (int t = 0; t < triangles; t++)
        {
            int a = 3 * t, b = 3 * t + 1, c = 3 * t + 2;
            graph.AddEdge(a, b);
            graph.AddEdge(b, c);
            graph.AddEdge(c, a);
        }
        int first = triangles * 3;
        int side = Math.Max(1, (vertices - first) / 2);
        for (int v = first; v < vertices; v += 2)
        {
            for (int d = 0; d < degree; d++)
            {
                int w = first + 1 + 2 * (int)(SplitMix64.Next(ref state) % (ulong)side);
                if (w >= vertices) continue;
                graph.AddEdge(v, w);
            }
        }
        return graph;
    }

    internal static QueryGraph Cycle(int length)
    {
        var graph = new QueryGraph(length);
        for (int i = 0; i < length; i++) graph.AddEdge(i, (i + 1) % length);
        return graph;
    }

    internal static QueryGraph Triangle()
    {
        var graph = new QueryGraph(3);
        graph.AddEdge(0, 1);
        graph.AddEdge(1, 2);
        graph.AddEdge(2, 0);
        return graph;
    }

    internal static QueryGraph Path(int length)
    {
        var graph = new QueryGraph(length);
        for (int i = 0; i + 1 < length; i++) graph.AddEdge(i, i + 1);
        return graph;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static double RejectionRate(QueryGraph graph, int starts, int walkLength, int seeds)
    {
        int rejections = 0;
        for (int seed = 0; seed < seeds; seed++)
        {
            var tester = new BipartiteTester(graph, starts, walkLength, 0x9480UL + (ulong)seed);
            tester.Run();
            if (tester.Rejected) rejections++;
        }
        return (double)rejections / seeds;
    }

    public static void Main()
    {
        const int Vertices = 200000;
        const int Starts = 12;
        const int Walk = 40;

        QueryGraph bipartite = Graphs.RandomBipartite(Vertices, 3, 0x9481UL);
        QueryGraph farFromBipartite = Graphs.Mixed(Vertices, Vertices, 3, 0x9482UL);

        // 1. Oracle agreement: the exact 2-colouring and the tester tell the same story.
        Check(ExactAlgorithms.IsBipartite(bipartite) && !ExactAlgorithms.IsBipartite(farFromBipartite),
            "the exact O(n+m) 2-colouring classifies both large graphs as expected");
        double bipartiteRejections = RejectionRate(bipartite, Starts, Walk, 32);
        double farRejections = RejectionRate(farFromBipartite, Starts, Walk, 32);
        Check(bipartiteRejections == 0.0 && farRejections >= 31.0 / 32.0,
            "the tester never rejects the bipartite graph and rejects the graph whose vertices all lie "
            + "on odd cycles in " + farRejections.ToString("P0") + " of runs");

        // 2. The witness of a rejection is a closed odd walk made of real edges.
        var witnessTester = new BipartiteTester(farFromBipartite, Starts, Walk, 0x9483UL);
        witnessTester.Run();
        List<int> walk = witnessTester.Witness;
        bool closed = walk.Count > 3 && walk[0] == walk[walk.Count - 1];
        bool odd = (walk.Count - 1) % 2 == 1;  // the walk lists one more vertex than it has steps
        bool edgesExist = true;
        for (int i = 0; i + 1 < walk.Count; i++)
            edgesExist &= farFromBipartite.Raw[walk[i]].Contains(walk[i + 1]);
        Check(witnessTester.Rejected && closed && odd && edgesExist,
            "the rejection carries a closed odd walk of " + (walk.Count - 1) + " steps as a witness");

        // 3. Query accounting: measured cost and strict sublinearity.
        var counter = new BipartiteTester(farFromBipartite, Starts, Walk, 0x9483UL);
        counter.Run();
        long exactProbes = ExactAlgorithms.Probes(farFromBipartite);
        Check(counter.Queries < Vertices / 4 && counter.Queries * 10 < exactProbes && counter.Queries > 0,
            "one run costs " + counter.Queries + " queries against " + exactProbes
            + " probes for the exact algorithm on " + Vertices + " vertices");

        // 4. Small-graph boundaries against the oracle.
        Check(!ExactAlgorithms.IsBipartite(Graphs.Triangle()) && RejectionRate(Graphs.Triangle(), 8, 16, 4) == 1.0
              && RejectionRate(Graphs.Path(4), 8, 16, 4) == 0.0 && RejectionRate(Graphs.Cycle(6), 8, 16, 4) == 0.0
              && RejectionRate(Graphs.Cycle(7), 16, 64, 8) == 1.0,
            "triangles and odd cycles are rejected while paths and even cycles are accepted");
        var single = new QueryGraph(1);
        Check(ExactAlgorithms.IsBipartite(single) && RejectionRate(single, 4, 16, 4) == 0.0,
            "a single vertex is bipartite and the tester touches nothing it cannot see");
        var empty = new QueryGraph(0);
        var emptyTester = new BipartiteTester(empty, 4, 4, 1UL);
        emptyTester.Run();
        Check(ExactAlgorithms.IsBipartite(empty) && !emptyTester.Rejected && emptyTester.Queries == 0,
            "an empty graph needs no exploration and no traversal");

        // 5. The rejection rate tracks the fraction of vertices lying on an odd cycle.
        double[] rates = new double[4];
        int[] triangleVertices = { 0, 400, 4000, 20000 };
        for (int i = 0; i < triangleVertices.Length; i++)
        {
            var variant = Graphs.Mixed(20000, triangleVertices[i], 3, 0x9484UL);
            rates[i] = RejectionRate(variant, Starts, Walk, 24);
        }
        bool monotone = true;
        for (int i = 0; i + 1 < rates.Length; i++) monotone &= rates[i + 1] >= rates[i];
        Check(monotone && rates[0] == 0.0 && rates[rates.Length - 1] > 0.9,
            "rejection rate rises with the share of vertices on odd cycles: "
            + string.Join(" -> ", rates.Select(r => r.ToString("P0"))) + " for those shares");

        // 6. The honest limitation: a single odd edge in a huge graph stays invisible.
        QueryGraph nearBipartite = Graphs.RandomBipartite(Vertices, 3, 0x9481UL);
        Graphs.AddIntraSideEdges(nearBipartite, 1, 0x9486UL);
        Check(!ExactAlgorithms.IsBipartite(nearBipartite) && RejectionRate(nearBipartite, Starts, Walk, 32) <= 0.1,
            "one planted odd edge makes the graph non-bipartite yet the tester accepts it: only "
            + "violations covering a large fraction of the vertices are promised to be caught");

        // 7-8. Second property test: estimation of the average degree with one query per sample.
        double exactMean = ExactAlgorithms.MeanDegree(bipartite);
        double best = 0.0, worst = double.MaxValue, total = 0.0;
        for (int seed = 0; seed < 20; seed++)
        {
            long before = bipartite.Queries;
            double mean = 0.0;
            int samples = 200;
            ulong state = 0x9487UL + (ulong)seed;
            for (int i = 0; i < samples; i++)
                mean += bipartite.Degree((int)(SplitMix64.Next(ref state) % (ulong)Vertices));
            mean /= samples;
            long spent = bipartite.Queries - before;
            best = Math.Max(best, spent);
            worst = Math.Min(worst, Math.Abs(mean - exactMean) / exactMean);
            total += mean;
        }
        Check(best == 200 && total / 20 <= exactMean * 1.05 && total / 20 >= exactMean * 0.95 && worst < 0.2,
            "sampling 200 vertices estimates the mean degree " + exactMean.ToString("F3") + " as "
            + (total / 20).ToString("F3") + " with exactly 200 degree queries");
        var zeroSample = new GraphSampler(bipartite);
        Check(Throws(() => zeroSample.MeanDegree(0, 1UL)), "an estimator with zero samples is rejected");

        // 9-10. Determinism of verdicts, witnesses and query counts.
        var first = new BipartiteTester(farFromBipartite, Starts, Walk, 0x9488UL);
        first.Run();
        var second = new BipartiteTester(farFromBipartite, Starts, Walk, 0x9488UL);
        second.Run();
        var cleanRepeat = new BipartiteTester(bipartite, Starts, Walk, 0x9489UL);
        cleanRepeat.Run();
        Check(first.Rejected == second.Rejected && first.Queries == second.Queries
              && first.Witness.SequenceEqual(second.Witness) && !cleanRepeat.Rejected,
            "the same seed reproduces the verdict, the witness and the query count");

        Console.WriteLine("  graph: " + Vertices + " vertices, " + bipartite.EdgeCount + " bipartite edges; tester "
            + Starts + " walks of " + Walk + " steps uses " + counter.Queries + " queries of " + exactProbes
            + " exact probes");
        Console.WriteLine("  rejections: bipartite " + bipartiteRejections.ToString("P0") + ", far "
            + farRejections.ToString("P0") + ", one planted odd edge "
            + RejectionRate(nearBipartite, Starts, Walk, 32).ToString("P0"));
        Console.WriteLine("PASS");
    }

    private sealed class GraphSampler
    {
        private readonly QueryGraph graph;
        internal GraphSampler(QueryGraph graph) => this.graph = graph;

        internal double MeanDegree(int samples, ulong seed)
        {
            if (samples <= 0) throw new ArgumentException("at least one sample is required");
            ulong state = seed;
            long sum = 0;
            for (int i = 0; i < samples; i++)
                sum += graph.Degree((int)(SplitMix64.Next(ref state) % (ulong)graph.VertexCount));
            return (double)sum / samples;
        }
    }
}
