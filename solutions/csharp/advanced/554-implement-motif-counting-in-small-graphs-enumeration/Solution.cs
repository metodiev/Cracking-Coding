// Task 554 - Implement motif counting in small graphs (enumeration). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Exhaustive induced-subgraph (graphlet) enumeration for small graphs. Every k-subset of the vertex
// set is enumerated, its induced adjacency packed into the k(k-1)/2 low bits of an int and minimised
// over all k! relabelings, so the canonical code identifies its isomorphism class exactly; the code
// is both the classifier and the histogram key. Because the same classifier is applied to every graph
// on k labeled vertices, the file also derives the number of motif classes itself instead of quoting
// them. As an independent second method, for a fixed motif the file enumerates *ordered* vertex
// tuples whose label-wise induced adjacency equals the motif and divides by the motif's automorphism
// count |Aut| (also computed by brute force over the k! relabelings), which must equal the subset
// count - two structurally different procedures agreeing is the main correctness argument here. No
// randomness is involved anywhere; the only graphs used are named explicit graphs (cliques, cycles,
// complete bipartite stars, the 3-cube and the Petersen graph) plus splitmix64-seeded Erdos-Renyi
// graphs, used only where a "typical" instance is wanted.
//
// Documented checks: the classifier finds 1, 2, 4 and 11 isomorphism classes on 1, 2, 3 and 4
// vertices (A000088); the eleven named 4-vertex motifs (edgeless, K2+2 isolated, 2K2, P3+isolated,
// K1,3, C4, P4, triangle+isolated, paw, diamond, K4) have eleven distinct canonical codes and are
// exactly the class set; on eight test graphs (K6, C5, C6, K1,5, the 3-cube, the Petersen graph, and
// two splitmix64 Erdos-Renyi graphs) the 4-vertex histogram sums to C(n,4) and the 3-vertex histogram
// to C(n,3), and for all eleven motifs and all eight graphs the subset count equals tuple count /
// |Aut| (11 x 8 = 88 independent comparisons); the closed forms hold (K_n has C(n,4) K4s and nothing
// else; K1,n has C(n,3) stars K1,3 and C(n,4) independent 4-sets and nothing else; the five 4-subsets
// of C5 are all P4s); the triangle count agrees with the edge-wedge formula (1/3) * sum over edges of
// |N(u) n N(v)| and with the wedge identity sum_v C(deg v,2) = 3T + (#induced P3); the Petersen graph
// and C5 are triangle-free, so their four triangle-bearing 4-vertex counts are 0; maximum independent
// sets computed by brute force match the known values alpha(C5) = 2, alpha(3-cube) = 4,
// alpha(Petersen) = 4; and for n < 4 every count is 0.
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

internal sealed class SmallGraph
{
    internal readonly int N;
    internal readonly bool[,] Adj;
    internal readonly string Label;

    internal SmallGraph(int n, string label)
    {
        N = n;
        Label = label;
        Adj = new bool[n, n];
    }

    internal void AddEdge(int u, int v)
    {
        Adj[u, v] = true;
        Adj[v, u] = true;
    }

    internal int Degree(int v)
    {
        int d = 0;
        for (int w = 0; w < N; w++)
        {
            if (Adj[v, w]) d++;
        }
        return d;
    }

    internal int EdgeCount()
    {
        int e = 0;
        for (int u = 0; u < N; u++)
        {
            for (int v = u + 1; v < N; v++)
            {
                if (Adj[u, v]) e++;
            }
        }
        return e;
    }

    internal static SmallGraph FromEdges(int n, string label, int[] pairs)
    {
        var g = new SmallGraph(n, label);
        for (int i = 0; i + 1 < pairs.Length; i += 2) g.AddEdge(pairs[i], pairs[i + 1]);
        return g;
    }

    internal static SmallGraph Clique(int n)
    {
        var g = new SmallGraph(n, "K" + n);
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++) g.AddEdge(u, v);
        }
        return g;
    }

    internal static SmallGraph Cycle(int n)
    {
        var g = new SmallGraph(n, "C" + n);
        for (int i = 0; i < n; i++) g.AddEdge(i, (i + 1) % n);
        return g;
    }

    internal static SmallGraph Star(int leaves)
    {
        var g = new SmallGraph(leaves + 1, "K1," + leaves);
        for (int i = 1; i <= leaves; i++) g.AddEdge(0, i);
        return g;
    }

    internal static SmallGraph Cube()
    {
        var g = new SmallGraph(8, "Q3");
        for (int v = 0; v < 8; v++)
        {
            for (int b = 0; b < 3; b++) g.AddEdge(v, v ^ (1 << b));
        }
        return g;
    }

    internal static SmallGraph Petersen()
    {
        var g = new SmallGraph(10, "Petersen");
        for (int i = 0; i < 5; i++)
        {
            g.AddEdge(i, (i + 1) % 5);
            g.AddEdge(i, i + 5);
            g.AddEdge(i + 5, ((i + 2) % 5) + 5);
        }
        return g;
    }

    internal static SmallGraph RandomGraph(int n, int probabilityPpm, ref ulong state)
    {
        var g = new SmallGraph(n, "G(" + n + ",rand)");
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (SplitMix64.Below(ref state, 1000000) < probabilityPpm) g.AddEdge(u, v);
            }
        }
        return g;
    }
}

internal static class MotifTable
{
    private static readonly Dictionary<int, int[][]> PermCache = new Dictionary<int, int[][]>();

    internal static int[][] Permutations(int k)
    {
        if (PermCache.TryGetValue(k, out int[][]? cached)) return cached;
        var sink = new List<int[]>();
        Build(k, 0, new int[k], new bool[k], sink);
        int[][] result = sink.ToArray();
        PermCache[k] = result;
        return result;
    }

    private static void Build(int k, int depth, int[] current, bool[] used, List<int[]> sink)
    {
        if (depth == k)
        {
            sink.Add((int[])current.Clone());
            return;
        }
        for (int i = 0; i < k; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            Build(k, depth + 1, current, used, sink);
            used[i] = false;
        }
    }

    internal static int CanonicalCode(bool[,] m, int k)
    {
        int best = int.MaxValue;
        foreach (int[] perm in Permutations(k))
        {
            int code = 0;
            for (int i = 0; i < k; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    if (m[perm[i], perm[j]]) code |= 1 << (i * k + j);
                }
            }
            if (code < best) best = code;
        }
        return best;
    }

    /** All graphs on k labeled vertices grouped into isomorphism classes; returns one representative per class. */
    internal static List<bool[,]> Representatives(int k)
    {
        int pairs = k * (k - 1) / 2;
        var byCode = new Dictionary<int, bool[,]>();
        for (int mask = 0; mask < (1 << pairs); mask++)
        {
            var m = new bool[k, k];
            int bit = 0;
            for (int i = 0; i < k; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    if ((mask & (1 << bit)) != 0)
                    {
                        m[i, j] = true;
                        m[j, i] = true;
                    }
                    bit++;
                }
            }
            int code = CanonicalCode(m, k);
            if (!byCode.ContainsKey(code)) byCode[code] = m;
        }
        return new List<bool[,]>(byCode.Values);
    }

    internal static int[][] Automorphisms(bool[,] m, int k)
    {
        var sink = new List<int[]>();
        foreach (int[] perm in Permutations(k))
        {
            bool ok = true;
            for (int i = 0; i < k && ok; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    if (m[i, j] != m[perm[i], perm[j]]) { ok = false; break; }
                }
            }
            if (ok) sink.Add(perm);
        }
        return sink.ToArray();
    }

    /** Histogram of induced k-subgraph classes over all C(n,k) subsets. */
    internal static Dictionary<int, long> EnumerateInduced(SmallGraph g, int k)
    {
        var histogram = new Dictionary<int, long>();
        var subset = new int[k];
        Enumerate(g, k, 0, 0, subset, delegate (int[] s)
        {
            int code = CanonicalCode(SubMatrix(g, s), k);
            histogram.TryGetValue(code, out long c);
            histogram[code] = c + 1;
        });
        return histogram;
    }

    internal static void Enumerate(SmallGraph g, int k, int start, int depth, int[] subset,
            Action<int[]> sink)
    {
        if (depth == k)
        {
            sink(subset);
            return;
        }
        for (int v = start; v < g.N; v++)
        {
            subset[depth] = v;
            Enumerate(g, k, v + 1, depth + 1, subset, sink);
        }
    }

    internal static bool[,] SubMatrix(SmallGraph g, int[] subset)
    {
        int k = subset.Length;
        var m = new bool[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                if (g.Adj[subset[i], subset[j]])
                {
                    m[i, j] = true;
                    m[j, i] = true;
                }
            }
        }
        return m;
    }

    /** Ordered tuples whose label-wise induced adjacency equals the motif, before dividing by |Aut|. */
    internal static long RawTupleCount(SmallGraph g, bool[,] motif, int k)
    {
        var used = new bool[g.N];
        var tuple = new int[k];
        return Count(g, motif, k, 0, tuple, used);
    }

    private static long Count(SmallGraph g, bool[,] motif, int k, int depth, int[] tuple, bool[] used)
    {
        if (depth == k)
        {
            for (int i = 0; i < k; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    if (g.Adj[tuple[i], tuple[j]] != motif[i, j]) return 0;
                }
            }
            return 1;
        }
        long total = 0;
        for (int v = 0; v < g.N; v++)
        {
            if (used[v]) continue;
            tuple[depth] = v;
            used[v] = true;
            total += Count(g, motif, k, depth + 1, tuple, used);
            used[v] = false;
        }
        return total;
    }

    internal static long Binomial(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        long result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    /** Brute force over all 2^n vertex subsets. */
    internal static int MaxIndependentSet(SmallGraph g)
    {
        int best = 0;
        for (int mask = 0; mask < (1 << g.N); mask++)
        {
            int size = 0;
            bool independent = true;
            for (int u = 0; u < g.N && independent; u++)
            {
                if ((mask & (1 << u)) == 0) continue;
                size++;
                for (int v = u + 1; v < g.N; v++)
                {
                    if ((mask & (1 << v)) != 0 && g.Adj[u, v]) { independent = false; break; }
                }
            }
            if (independent && size > best) best = size;
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        // The classifier derives the class counts by itself: 1, 2, 4, 11 for k = 1..4.
        int[] expectedClasses = { 1, 1, 2, 4, 11 };
        for (int k = 1; k <= 4; k++)
        {
            int got = MotifTable.Representatives(k).Count;
            Check(got == expectedClasses[k], "k=" + k + " has " + expectedClasses[k]
                    + " motif classes, got " + got);
        }

        // Named 4-vertex motifs; the eleven codes must be distinct and cover every class.
        var named = new Dictionary<string, bool[,]>
        {
            { "edgeless4", Matrix(4, new int[0]) },
            { "K2+2iso", Matrix(4, new[] { 0, 1 }) },
            { "2K2", Matrix(4, new[] { 0, 1, 2, 3 }) },
            { "P3+iso", Matrix(4, new[] { 0, 1, 1, 2 }) },
            { "K1,3", Matrix(4, new[] { 0, 1, 0, 2, 0, 3 }) },
            { "P4", Matrix(4, new[] { 0, 1, 1, 2, 2, 3 }) },
            { "C4", Matrix(4, new[] { 0, 1, 1, 2, 2, 3, 3, 0 }) },
            { "tri+iso", Matrix(4, new[] { 0, 1, 1, 2, 0, 2 }) },
            { "paw", Matrix(4, new[] { 0, 1, 1, 2, 0, 2, 0, 3 }) },
            { "diamond", Matrix(4, new[] { 0, 2, 0, 3, 1, 2, 1, 3, 2, 3 }) },
            { "K4", Matrix(4, new[] { 0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3 }) },
        };
        Check(named.Count == 11, "eleven named 4-vertex motifs");
        var codes = new Dictionary<string, int>();
        foreach (KeyValuePair<string, bool[,]> entry in named)
        {
            codes[entry.Key] = MotifTable.CanonicalCode(entry.Value, 4);
        }
        var codeSet = new HashSet<int>(codes.Values);
        Check(codeSet.Count == 11, "the eleven named motifs have distinct canonical codes");
        var classCodes = new HashSet<int>();
        foreach (bool[,] rep in MotifTable.Representatives(4)) classCodes.Add(MotifTable.CanonicalCode(rep, 4));
        Check(codeSet.SetEquals(classCodes), "the named motifs are exactly the 11 classes");

        var named3 = new Dictionary<string, int>
        {
            { "edgeless3", MotifTable.CanonicalCode(Matrix(3, new int[0]), 3) },
            { "K2+iso", MotifTable.CanonicalCode(Matrix(3, new[] { 0, 1 }), 3) },
            { "P3", MotifTable.CanonicalCode(Matrix(3, new[] { 0, 1, 1, 2 }), 3) },
            { "K3", MotifTable.CanonicalCode(Matrix(3, new[] { 0, 1, 1, 2, 0, 2 }), 3) },
        };
        Check(new HashSet<int>(named3.Values).Count == 4, "the 3-vertex motifs are exactly the 4 classes");
        Check(codes["K1,3"] == MotifTable.CanonicalCode(Matrix(4, new[] { 0, 1, 0, 2, 0, 3 }), 4),
                "K1,3 canonical code is stable");

        ulong rng = 554001UL;
        var graphs = new List<SmallGraph>
        {
            SmallGraph.Clique(6),
            SmallGraph.Cycle(5),
            SmallGraph.Cycle(6),
            SmallGraph.Star(5),
            SmallGraph.Cube(),
            SmallGraph.Petersen(),
            SmallGraph.RandomGraph(7, 400000, ref rng),
            SmallGraph.RandomGraph(8, 500000, ref rng),
        };

        int tupleComparisons = 0;
        foreach (SmallGraph g in graphs)
        {
            Dictionary<int, long> hist4 = MotifTable.EnumerateInduced(g, 4);
            long total4 = 0;
            foreach (long v in hist4.Values) total4 += v;
            Check(total4 == MotifTable.Binomial(g.N, 4), g.Label + ": 4-vertex counts sum to C(n,4) = "
                    + MotifTable.Binomial(g.N, 4) + ", got " + total4);
            Check(hist4.Count <= 11, g.Label + " uses at most the 11 classes");

            Dictionary<int, long> hist3 = MotifTable.EnumerateInduced(g, 3);
            long total3 = 0;
            foreach (long v in hist3.Values) total3 += v;
            Check(total3 == MotifTable.Binomial(g.N, 3), g.Label + ": 3-vertex counts sum to C(n,3)");

            foreach (KeyValuePair<string, bool[,]> entry in named)
            {
                int code = codes[entry.Key];
                long expected = hist4.TryGetValue(code, out long c) ? c : 0;
                int[][] auts = MotifTable.Automorphisms(entry.Value, 4);
                Check(auts.Length >= 1, "every motif has at least the identity automorphism");
                long tuples = MotifTable.RawTupleCount(g, entry.Value, 4);
                Check(tuples % auts.Length == 0, "tuple count is divisible by |Aut| for " + entry.Key);
                long viaTuples = tuples / auts.Length;
                Check(viaTuples == expected, g.Label + " " + entry.Key + ": subset count " + expected
                        + " != tuple count / |Aut| = " + viaTuples);
                tupleComparisons++;
            }

            // Triangle count by three independent routes.
            long triangles = hist3.TryGetValue(named3["K3"], out long t) ? t : 0;
            long inducedP3 = hist3.TryGetValue(named3["P3"], out long p) ? p : 0;
            long wedges = 0;
            for (int v = 0; v < g.N; v++)
            {
                int d = g.Degree(v);
                wedges += (long)d * (d - 1) / 2;
            }
            Check(wedges == 3 * triangles + inducedP3, g.Label
                    + ": sum_v C(deg v,2) = 3T + #P3, got " + wedges + " vs " + (3 * triangles + inducedP3));

            long edgeTriangles = 0;
            for (int u = 0; u < g.N; u++)
            {
                for (int v = u + 1; v < g.N; v++)
                {
                    if (!g.Adj[u, v]) continue;
                    for (int w = 0; w < g.N; w++)
                    {
                        if (g.Adj[u, w] && g.Adj[v, w]) edgeTriangles++;
                    }
                }
            }
            Check(edgeTriangles == 3 * triangles, g.Label
                    + ": (1/3) * sum over edges |N(u) n N(v)| equals the triangle count");
        }
        Check(tupleComparisons == 88, "88 subset/tuple comparisons (11 motifs x 8 graphs), got "
                + tupleComparisons);

        // Closed forms.
        Check(MotifTable.EnumerateInduced(SmallGraph.Clique(6), 4)[codes["K4"]] == 15,
                "K6 has C(6,4) = 15 K4s");
        Check(MotifTable.EnumerateInduced(SmallGraph.Clique(6), 4).Count == 1, "K6 has only one motif type");
        var star5 = SmallGraph.Star(5);
        Dictionary<int, long> starHist = MotifTable.EnumerateInduced(star5, 4);
        Check(starHist.Count == 2, "K1,5 has exactly two 4-vertex motif types");
        Check(starHist[codes["K1,3"]] == 10, "K1,5 has C(5,3) = 10 stars K1,3");
        Check(starHist[codes["edgeless4"]] == 5, "K1,5 has C(5,4) = 5 independent 4-sets");
        Check(MotifTable.EnumerateInduced(SmallGraph.Cycle(5), 4)[codes["P4"]] == 5,
                "the five 4-subsets of C5 are P4s");
        Check(MotifTable.EnumerateInduced(SmallGraph.Cycle(6), 4)[codes["P4"]] == 6,
                "C6 has 6 consecutive P4s; the remaining 9 subsets are checked against tuple counting");

        // Triangle-free graphs: the four triangle-bearing 4-vertex motifs must vanish.
        string[] triangleMotifs = { "tri+iso", "paw", "diamond", "K4" };
        foreach (SmallGraph g in new[] { SmallGraph.Cycle(5), SmallGraph.Petersen(), SmallGraph.Cube() })
        {
            Dictionary<int, long> hist = MotifTable.EnumerateInduced(g, 4);
            foreach (string name in triangleMotifs)
            {
                Check(!hist.ContainsKey(codes[name]) || hist[codes[name]] == 0,
                        g.Label + " is triangle-free, so " + name + " must not occur");
            }
        }

        // Known independence numbers.
        Check(MotifTable.MaxIndependentSet(SmallGraph.Cycle(5)) == 2, "alpha(C5) = 2");
        Check(MotifTable.MaxIndependentSet(SmallGraph.Cube()) == 4, "alpha(Q3) = 4");
        Check(MotifTable.MaxIndependentSet(SmallGraph.Petersen()) == 4, "alpha(Petersen) = 4");
        Check(MotifTable.EnumerateInduced(SmallGraph.Cube(), 4)[codes["edgeless4"]] >= 1,
                "the two bipartition halves of Q3 give independent 4-sets");

        // Boundary: fewer than k vertices.
        foreach (int n in new[] { 0, 1, 2, 3 })
        {
            Check(MotifTable.EnumerateInduced(SmallGraph.Clique(n), 4).Count == 0,
                    "K" + n + " has no 4-vertex induced subgraph");
            Check(MotifTable.Binomial(n, 4) == 0, "C(" + n + ",4) = 0");
        }

        Console.WriteLine("PASS");
    }

    private static bool[,] Matrix(int k, int[] pairs)
    {
        var m = new bool[k, k];
        for (int i = 0; i + 1 < pairs.Length; i += 2)
        {
            m[pairs[i], pairs[i + 1]] = true;
            m[pairs[i + 1], pairs[i]] = true;
        }
        return m;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
