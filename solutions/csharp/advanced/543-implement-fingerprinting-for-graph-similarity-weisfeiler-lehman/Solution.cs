// Task 543 - Implement fingerprinting for graph similarity (Weisfeiler-Lehman sketch). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Weisfeiler-Lehman colour refinement used as a graph fingerprint. Round 0 colours a vertex by its
// degree; every later round replaces the colour by the canonical string "old|sorted neighbour
// colours", so the colour of a vertex is a complete description of its rooted ball and the colour
// multiset of a round is a relabelling-invariant sketch. A fingerprint is the sequence of these
// multisets (rendered as sorted "count*colour" entries), and the WL kernel is the histogram
// intersection summed over the rounds, normalised by the self intersections - so the similarity of
// two graphs is 1 exactly when their fingerprints coincide. The file also contains a backtracking
// isomorphism test (degree filter plus adjacency consistency) as ground truth, because colour
// refinement is a necessary but not sufficient invariant.
//
// Documented checks: over 12 LCG graphs (n = 8) plus a relabelled copy of each, fingerprints and
// WL similarity 1 are invariant and the colours of round r are always prefixes of round r+1 (the
// refinement property); the number of distinct colours never decreases across rounds; every
// isomorphic pair in the mixed shape set gets similarity 1 and every pair whose fingerprints differ
// is confirmed non-isomorphic by brute force; the sketch separates a pair that degree sequences
// cannot - the 5-vertex square-with-a-pendant-edge and the triangle-with-a-pendant-path both have
// degree multiset (3,2,2,2,1) yet differ in round 1 (one has a 4-cycle, the other a triangle);
// and two documented failures are asserted rather than hidden: C6 versus two disjoint triangles,
// and K3,3 versus the triangular prism, are non-isomorphic 2- and 3-regular pairs whose
// fingerprints coincide at every round, because colour refinement cannot count cycles.
using System;
using System.Collections.Generic;
using System.Text;

internal sealed class Graph
{
    internal readonly int N;
    internal readonly List<int>[] Adj;

    internal Graph(int n)
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

    internal static Graph Cycle(int n)
    {
        var g = new Graph(n);
        for (int i = 0; i < n; i++) g.AddEdge(i, (i + 1) % n);
        return g;
    }

    internal static Graph Path(int n)
    {
        var g = new Graph(n);
        for (int i = 0; i + 1 < n; i++) g.AddEdge(i, i + 1);
        return g;
    }

    internal static Graph Complete(int n)
    {
        var g = new Graph(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++) g.AddEdge(i, j);
        }
        return g;
    }

    internal static Graph CompleteBipartite(int a, int b)
    {
        var g = new Graph(a + b);
        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < b; j++) g.AddEdge(i, a + j);
        }
        return g;
    }

    internal static Graph DisjointUnion(Graph first, Graph second)
    {
        var g = new Graph(first.N + second.N);
        for (int v = 0; v < first.N; v++)
        {
            foreach (int w in first.Adj[v])
            {
                if (v < w) g.AddEdge(v, w);
            }
        }
        for (int v = 0; v < second.N; v++)
        {
            foreach (int w in second.Adj[v])
            {
                if (v < w) g.AddEdge(first.N + v, first.N + w);
            }
        }
        return g;
    }

    internal static Graph Lcg(int n, int extraEdges, ref int state)
    {
        var g = new Graph(n);
        for (int i = 0; i + 1 < n; i++)
        {
            state = (state * 75 + 74) % 65537;
            g.AddEdge(i, i + 1);
            if (state % 2 == 0) g.AddEdge(i, (i + 1 + state % (n - i - 1)) % n);
        }
        for (int k = 0; k < extraEdges; k++)
        {
            state = (state * 75 + 74) % 65537;
            int u = state % n;
            state = (state * 75 + 74) % 65537;
            int v = state % n;
            g.AddEdge(u, v);
        }
        return g;
    }

    internal Graph Relabel(int[] permutation)
    {
        var g = new Graph(N);
        for (int v = 0; v < N; v++)
        {
            foreach (int w in Adj[v])
            {
                if (v < w) g.AddEdge(permutation[v], permutation[w]);
            }
        }
        return g;
    }

    internal bool IsomorphicBrute(Graph other)
    {
        if (other.N != N) return false;
        if (EdgeCount() != other.EdgeCount()) return false;
        var mine = Degrees();
        var theirs = other.Degrees();
        for (int i = 0; i < N; i++)
        {
            if (mine[i] != theirs[i]) return false;
        }
        var map = new int[N];
        var used = new bool[N];
        for (int v = 0; v < N; v++) map[v] = -1;
        return Match(0, map, used, other);
    }

    internal int[] Degrees()
    {
        var degrees = new int[N];
        for (int v = 0; v < N; v++) degrees[v] = Adj[v].Count;
        Array.Sort(degrees);
        return degrees;
    }

    private bool Match(int index, int[] map, bool[] used, Graph other)
    {
        if (index == N) return true;
        for (int candidate = 0; candidate < N; candidate++)
        {
            if (used[candidate]) continue;
            if (Adj[index].Count != other.Adj[candidate].Count) continue;
            bool ok = true;
            foreach (int w in Adj[index])
            {
                if (w >= index) continue;
                if (!other.Adj[candidate].Contains(map[w]))
                {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            map[index] = candidate;
            used[candidate] = true;
            if (Match(index + 1, map, used, other)) return true;
            used[candidate] = false;
            map[index] = -1;
        }
        return false;
    }
}

internal static class Wl
{
    /** colours[r][v] is the canonical colour of v after r refinement rounds. */
    internal static List<string[]> Colours(Graph g, int rounds)
    {
        var levels = new List<string[]>();
        var current = new string[g.N];
        for (int v = 0; v < g.N; v++) current[v] = "d" + g.Adj[v].Count;
        levels.Add((string[])current.Clone());
        for (int r = 1; r <= rounds; r++)
        {
            var next = new string[g.N];
            for (int v = 0; v < g.N; v++)
            {
                var neighbourColours = new List<string>();
                foreach (int w in g.Adj[v]) neighbourColours.Add(current[w]);
                neighbourColours.Sort(StringComparer.Ordinal);
                var sb = new StringBuilder(current[v]).Append('|');
                foreach (string colour in neighbourColours) sb.Append(colour).Append(',');
                next[v] = sb.ToString();
            }
            levels.Add(next);
            current = next;
        }
        return levels;
    }

    internal static List<SortedDictionary<string, int>> Histograms(Graph g, int rounds)
    {
        var result = new List<SortedDictionary<string, int>>();
        foreach (string[] level in Colours(g, rounds))
        {
            var histogram = new SortedDictionary<string, int>(StringComparer.Ordinal);
            foreach (string colour in level)
            {
                histogram.TryGetValue(colour, out int count);
                histogram[colour] = count + 1;
            }
            result.Add(histogram);
        }
        return result;
    }

    internal static string Fingerprint(Graph g, int rounds)
    {
        var parts = new List<string>();
        foreach (SortedDictionary<string, int> histogram in Histograms(g, rounds))
        {
            var sb = new StringBuilder();
            foreach (KeyValuePair<string, int> entry in histogram)
            {
                sb.Append(entry.Value).Append('*').Append(entry.Key).Append(';');
            }
            parts.Add(sb.ToString());
        }
        return string.Join("|", parts);
    }

    internal static int KernelRaw(Graph a, Graph b, int rounds)
    {
        var ha = Histograms(a, rounds);
        var hb = Histograms(b, rounds);
        int total = 0;
        for (int r = 0; r < ha.Count; r++)
        {
            foreach (KeyValuePair<string, int> entry in ha[r])
            {
                if (hb[r].TryGetValue(entry.Key, out int other)) total += Math.Min(entry.Value, other);
            }
        }
        return total;
    }

    internal static double KernelSimilarity(Graph a, Graph b, int rounds)
    {
        double raw = KernelRaw(a, b, rounds);
        return raw / Math.Sqrt((double)KernelRaw(a, a, rounds) * KernelRaw(b, b, rounds));
    }

    internal static int DistinctColourCounts(Graph g, int rounds, out List<int> counts)
    {
        counts = new List<int>();
        foreach (SortedDictionary<string, int> histogram in Histograms(g, rounds)) counts.Add(histogram.Count);
        return counts.Count;
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
        const int rounds = 4;
        int state = 543001;
        var graphs = new List<Graph>();
        var relabelled = new List<Graph>();
        for (int i = 0; i < 12; i++)
        {
            Graph g = Graph.Lcg(8, 6, ref state);
            graphs.Add(g);
            var permutation = new int[8];
            for (int v = 0; v < 8; v++) permutation[v] = v;
            for (int v = 7; v > 0; v--)
            {
                state = (state * 75 + 74) % 65537;
                int j = state % (v + 1);
                int tmp = permutation[v];
                permutation[v] = permutation[j];
                permutation[j] = tmp;
            }
            relabelled.Add(g.Relabel(permutation));
        }

        for (int i = 0; i < graphs.Count; i++)
        {
            Check(Wl.Fingerprint(graphs[i], rounds) == Wl.Fingerprint(relabelled[i], rounds),
                    "relabelling preserves the fingerprint at index " + i);
            Check(Math.Abs(Wl.KernelSimilarity(graphs[i], relabelled[i], rounds) - 1.0) <= 1e-12,
                    "isomorphic graphs have WL similarity 1 at index " + i);
            Check(graphs[i].IsomorphicBrute(relabelled[i]), "brute force confirms isomorphism at index " + i);
            Check(graphs[i].EdgeCount() >= 8, "the LCG graph is non-trivial at index " + i);

            List<string[]> levels = Wl.Colours(graphs[i], rounds);
            for (int r = 1; r < levels.Count; r++)
            {
                for (int v = 0; v < graphs[i].N; v++)
                {
                    Check(levels[r][v].StartsWith(levels[r - 1][v], StringComparison.Ordinal),
                            "round " + r + " refines round " + (r - 1) + " at vertex " + v);
                }
            }
            Wl.DistinctColourCounts(graphs[i], rounds, out List<int> counts);
            for (int r = 1; r < counts.Count; r++)
            {
                Check(counts[r] >= counts[r - 1], "colour count is non-decreasing at round " + r);
                Check(counts[r] <= graphs[i].N, "colour count stays below n at round " + r);
            }
            Check(Wl.KernelSimilarity(graphs[i], graphs[i], rounds) == 1.0, "self similarity is exactly 1");
        }

        var shapes = new List<Graph>
        {
            Graph.Cycle(6), Graph.DisjointUnion(Graph.Cycle(3), Graph.Cycle(3)),
            Graph.CompleteBipartite(3, 3), Prism(), Graph.Complete(3), Graph.Path(3),
            Graph.Complete(4), Graph.Path(4), Graph.Cycle(4), SquareWithPendant(), TriangleWithPath(),
        };

        int wtlFailures = 0;
        int separated = 0;
        for (int i = 0; i < shapes.Count; i++)
        {
            for (int j = 0; j < shapes.Count; j++)
            {
                if (i == j) continue;
                bool iso = shapes[i].IsomorphicBrute(shapes[j]);
                bool sameFingerprint = Wl.Fingerprint(shapes[i], rounds) == Wl.Fingerprint(shapes[j], rounds);
                if (iso)
                {
                    Check(sameFingerprint, "isomorphic shapes must share a fingerprint");
                    Check(Math.Abs(Wl.KernelSimilarity(shapes[i], shapes[j], rounds) - 1.0) <= 1e-12,
                            "isomorphic shapes have similarity 1");
                }
                else if (!sameFingerprint)
                {
                    Check(!iso, "a differing fingerprint proves non-isomorphism");
                    Check(Wl.KernelSimilarity(shapes[i], shapes[j], rounds) < 1.0,
                            "a separated pair has similarity below 1");
                    separated++;
                }
                else
                {
                    wtlFailures++;
                }
            }
        }
        Check(separated > 0, "colour refinement separates some non-isomorphic pairs");
        Check(wtlFailures >= 4, "the two regular counterexamples are observed (" + wtlFailures + " ordered pairs)");

        // The documented limitations, asserted explicitly.
        Check(Graph.Cycle(6).IsomorphicBrute(Graph.DisjointUnion(Graph.Cycle(3), Graph.Cycle(3))) == false,
                "C6 and 2 x C3 are non-isomorphic");
        Check(Wl.Fingerprint(Graph.Cycle(6), rounds)
                == Wl.Fingerprint(Graph.DisjointUnion(Graph.Cycle(3), Graph.Cycle(3)), rounds),
                "colour refinement cannot separate C6 from 2 x C3");
        Check(Graph.CompleteBipartite(3, 3).IsomorphicBrute(Prism()) == false, "K3,3 and the prism differ");
        Check(Wl.Fingerprint(Graph.CompleteBipartite(3, 3), rounds) == Wl.Fingerprint(Prism(), rounds),
                "colour refinement cannot separate K3,3 from the prism");

        // A pair with a shared degree sequence that the sketch does separate.
        Graph squarePendant = SquareWithPendant();
        Graph trianglePath = TriangleWithPath();
        Check(string.Join(",", squarePendant.Degrees()) == string.Join(",", trianglePath.Degrees()),
                "the two 5-vertex examples share a degree sequence");
        Check(squarePendant.IsomorphicBrute(trianglePath) == false, "the two 5-vertex examples differ");
        Check(Wl.Fingerprint(squarePendant, rounds) != Wl.Fingerprint(trianglePath, rounds),
                "colour refinement separates the square-pendant from the triangle-path");
        Check(Wl.Fingerprint(squarePendant, 1) != Wl.Fingerprint(trianglePath, 1),
                "the separation is visible from round 1 on");
        Check(Graph.Complete(3).IsomorphicBrute(Graph.Path(3)) == false, "K3 and P3 differ");
        Check(Wl.Fingerprint(Graph.Complete(3), rounds) != Wl.Fingerprint(Graph.Path(3), rounds),
                "K3 and P3 have different fingerprints already at round 0");

        // Determinism and symmetry.
        Check(Wl.Fingerprint(SquareWithPendant(), rounds) == Wl.Fingerprint(SquareWithPendant(), rounds),
                "fingerprints are reproducible");
        for (int i = 0; i < shapes.Count; i++)
        {
            for (int j = 0; j < shapes.Count; j++)
            {
                double sim = Wl.KernelSimilarity(shapes[i], shapes[j], rounds);
                Check(sim >= 0.0 && sim <= 1.0 + 1e-12, "similarity stays in [0,1]");
                Check(Math.Abs(sim - Wl.KernelSimilarity(shapes[j], shapes[i], rounds)) <= 1e-12,
                        "similarity is symmetric");
            }
        }

        Console.WriteLine("PASS");
    }

    /** Square a-b-c-d-a with a pendant vertex attached to a. */
    private static Graph SquareWithPendant()
    {
        var g = new Graph(5);
        g.AddEdge(0, 1);
        g.AddEdge(1, 2);
        g.AddEdge(2, 3);
        g.AddEdge(3, 0);
        g.AddEdge(0, 4);
        return g;
    }

    /** Triangle a-b-c plus the path c-d-e. */
    private static Graph TriangleWithPath()
    {
        var g = new Graph(5);
        g.AddEdge(0, 1);
        g.AddEdge(1, 2);
        g.AddEdge(2, 0);
        g.AddEdge(2, 3);
        g.AddEdge(3, 4);
        return g;
    }

    /** Triangular prism: two triangles joined by a perfect matching. */
    private static Graph Prism()
    {
        var g = new Graph(6);
        g.AddEdge(0, 1);
        g.AddEdge(1, 2);
        g.AddEdge(2, 0);
        g.AddEdge(3, 4);
        g.AddEdge(4, 5);
        g.AddEdge(5, 3);
        g.AddEdge(0, 3);
        g.AddEdge(1, 4);
        g.AddEdge(2, 5);
        return g;
    }
}
