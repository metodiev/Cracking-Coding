// Task 553 - Implement induced subgraph frequency sampling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Induced subgraph frequency by uniform sampling, for graphlet/motif studies. A k-subset is drawn by
// partial Fisher-Yates on the vertex pool, so every one of the C(n,k) subsets has probability exactly
// 1/C(n,k); draws are independent (with replacement across draws), hence the estimator "fraction of
// sampled subsets whose induced subgraph is isomorphic to the pattern" is unbiased with standard error
// sqrt(p(1-p)/m) - quoted as a plain binomial standard error because consecutive draws are
// independent, not a without-replacement sample. Two subsets are compared by canonical form: the
// induced adjacency is packed into the k(k-1)/2 low bits of an int and minimised over all k!
// relabelings, which classifies k <= 5 vertex patterns exactly. The exact reference is a full
// enumeration of all C(n,k) subsets with the same classifier, and the motif-significance part scores a
// real graph's triangle count against an Erdos-Renyi null model with the *same edge density* built
// from splitmix64 (a documented 64-bit generator - the repo's 16-bit LCG has too short a cycle for
// hundreds of independent null graphs).
//
// Documented checks: the classifier finds 4 isomorphism classes on 3 vertices and 11 on 4 vertices
// (A000088: 1, 2, 4, 11); on C5 the 10 three-subsets split exactly 5 induced P3 / 5 single-edge /
// 0 triangle, matching hand counting; a 10 000-draw sample of the 2-subsets of 5 vertices is uniform
// (every one of the 10 subsets appears, chi-square asserted below 40 for 9 degrees of freedom, and
// every count lies within 4 sigma of 1000); uniform draws reproduce the exact frequency of
// triangles and of induced P3 on a 10-vertex splitmix64 graph to within 4 standard errors for
// k = 3 and k = 4, with the error shrinking when the sample grows tenfold; degenerate frequencies are
// exact (a triangle pattern in K5 gives p = 1 and an edgeless pattern in K5 gives p = 0, no sampling
// noise at all); repeated runs with the same seed are identical; and on K5 plus 7 isolated vertices
// (12 vertices, density 20/132) the triangle Z-score against 400 density-matched G(12, p) null graphs
// is above 5, while the same statistic for the triangle-free 12-cycle is below 1.
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

internal sealed class SamplerGraph
{
    internal readonly int N;
    internal readonly bool[,] Adj;

    internal SamplerGraph(int n)
    {
        N = n;
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

    internal static SamplerGraph Clique(int k)
    {
        var g = new SamplerGraph(k);
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++) g.AddEdge(i, j);
        }
        return g;
    }

    internal static SamplerGraph Cycle(int k)
    {
        var g = new SamplerGraph(k);
        for (int i = 0; i < k; i++) g.AddEdge(i, (i + 1) % k);
        return g;
    }

    internal SamplerGraph PlusIsolated(int extra)
    {
        var g = new SamplerGraph(N + extra);
        for (int u = 0; u < N; u++)
        {
            for (int v = 0; v < N; v++)
            {
                if (Adj[u, v]) g.AddEdge(u, v);
            }
        }
        return g;
    }

    /** Deterministic Erdos-Renyi graph with edge probability probabilityPpm / 1000000. */
    internal static SamplerGraph RandomGraph(int n, int probabilityPpm, ref ulong state)
    {
        var g = new SamplerGraph(n);
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

internal static class Motifs
{
    private static readonly Dictionary<int, int[][]> Cache = new Dictionary<int, int[][]>();

    internal static int[][] Permutations(int k)
    {
        if (Cache.TryGetValue(k, out int[][]? cached)) return cached;
        var sink = new List<int[]>();
        Build(k, 0, new int[k], new bool[k], sink);
        int[][] result = sink.ToArray();
        Cache[k] = result;
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

    /** Canonical form of the subgraph of g induced on `subset` (length <= 5). */
    internal static int CanonicalCode(SamplerGraph g, int[] subset)
    {
        int k = subset.Length;
        int best = int.MaxValue;
        foreach (int[] perm in Permutations(k))
        {
            int code = 0;
            for (int i = 0; i < k; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    if (g.Adj[subset[perm[i]], subset[perm[j]]]) code |= 1 << (i * k + j);
                }
            }
            if (code < best) best = code;
        }
        return best;
    }

    internal static int CanonicalCodeOfMatrix(bool[,] m, int k)
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

    internal static long Binomial(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        long result = 1;
        for (int i = 0; i < k; i++)
        {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    /** Frequency of a canonical code among all C(n,k) induced subsets (the exact reference). */
    internal static long ExactCount(SamplerGraph g, int k, int targetCode)
    {
        long hits = 0;
        var subset = new int[k];
        EnumerateSubsets(g, k, 0, 0, subset, delegate (int[] s)
        {
            if (CanonicalCode(g, s) == targetCode) hits++;
        });
        return hits;
    }

    internal static void EnumerateSubsets(SamplerGraph g, int k, int start, int depth, int[] subset,
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
            EnumerateSubsets(g, k, v + 1, depth + 1, subset, sink);
        }
    }

    internal static SamplerGraph Prototype3(int edges)
    {
        var g = new SamplerGraph(3);
        if (edges >= 1) g.AddEdge(0, 1);
        if (edges >= 2) g.AddEdge(1, 2);
        if (edges >= 3) g.AddEdge(0, 2);
        return g;
    }
}

internal static class Sampling
{
    internal sealed class Estimate
    {
        internal long Hits;
        internal long Samples;
        internal double P;
        internal double StandardError;
    }

    /** Uniform k-subset by partial Fisher-Yates; every C(n,k) subset is equally likely. */
    internal static int[] SampleSubset(int n, int k, ref ulong state)
    {
        var pool = new int[n];
        for (int i = 0; i < n; i++) pool[i] = i;
        for (int i = 0; i < k; i++)
        {
            int j = i + SplitMix64.Below(ref state, n - i);
            int tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
        }
        var subset = new int[k];
        Array.Copy(pool, subset, k);
        Array.Sort(subset);
        return subset;
    }

    internal static Estimate Frequency(SamplerGraph g, int k, int targetCode, long samples,
            ref ulong state)
    {
        long hits = 0;
        for (long i = 0; i < samples; i++)
        {
            int[] subset = SampleSubset(g.N, k, ref state);
            if (Motifs.CanonicalCode(g, subset) == targetCode) hits++;
        }
        double p = (double)hits / samples;
        return new Estimate
        {
            Hits = hits,
            Samples = samples,
            P = p,
            StandardError = Math.Sqrt(p * (1.0 - p) / samples),
        };
    }
}

public static class Solution
{
    public static void Main()
    {
        // The classifier: 4 classes on 3 vertices, 11 on 4 (A000088: 1, 2, 4, 11).
        Check(ClassCount(3) == 4, "3-vertex motifs form 4 classes, got " + ClassCount(3));
        Check(ClassCount(4) == 11, "4-vertex motifs form 11 classes, got " + ClassCount(4));

        int edgeless3 = Motifs.CanonicalCodeOfMatrix(new bool[3, 3], 3);
        int p3 = Motifs.CanonicalCodeOfMatrix(Prototype(3, 2), 3);
        int triangle3 = Motifs.CanonicalCodeOfMatrix(Prototype(3, 3), 3);
        var c5 = SamplerGraph.Cycle(5);

        // Hand counting on C5: 5 induced P3, 5 single edges, no triangle, 10 subsets in total.
        Check(Motifs.ExactCount(c5, 3, p3) == 5, "C5 has 5 induced P3");
        Check(Motifs.ExactCount(c5, 3, Motifs.CanonicalCodeOfMatrix(Prototype(3, 1), 3)) == 5,
                "C5 has 5 single-edge triples");
        Check(Motifs.ExactCount(c5, 3, triangle3) == 0, "C5 has no triangle");
        Check(5 + 5 + 0 == Motifs.Binomial(5, 3), "the three counts exhaust C(5,3) = 10");
        Check(Motifs.ExactCount(c5, 3, edgeless3) == 0, "C5 has no independent triple");
        Check(Motifs.ExactCount(c5, 4, PathCode(4)) == 5,
                "the five 4-subsets of C5 all induce P4");

        // Uniformity of the sampler over all 10 two-subsets of 5 vertices.
        ulong state = 553001UL;
        long draws = 10000;
        var counts = new long[10];
        var index = new Dictionary<string, int>();
        for (int a = 0; a < 5; a++)
        {
            for (int b = a + 1; b < 5; b++) index[a + "," + b] = index.Count;
        }
        for (long i = 0; i < draws; i++)
        {
            int[] s = Sampling.SampleSubset(5, 2, ref state);
            Check(s.Length == 2 && s[0] < s[1], "a sample has two distinct sorted vertices");
            counts[index[s[0] + "," + s[1]]]++;
        }
        double chiSquare = 0.0;
        double expected = (double)draws / 10;
        foreach (long c in counts)
        {
            chiSquare += (c - expected) * (c - expected) / expected;
            Check(c > 0, "every one of the 10 subsets appears in 10000 draws");
            Check(Math.Abs(c - expected) <= 4.0 * Math.Sqrt(expected * 0.9),
                    "each subset count is within 4 sigma of 1000, got " + c);
        }
        Check(chiSquare < 40.0, "chi-square " + chiSquare + " is below the 40 threshold for 9 d.o.f.");

        // Estimator versus the exact enumeration on a 10-vertex splitmix64 graph.
        ulong graphState = 553100UL;
        var g10 = SamplerGraph.RandomGraph(10, 450000, ref graphState);
        ulong sampleState = 553200UL;
        int[] patterns3 = { edgeless3, Motifs.CanonicalCodeOfMatrix(Prototype(3, 1), 3), p3, triangle3 };
        long exact3 = 0;
        foreach (int code in patterns3) exact3 += Motifs.ExactCount(g10, 3, code);
        Check(exact3 == Motifs.Binomial(10, 3), "the four 3-vertex classes partition C(10,3) = 120");
        long subsetCount3 = Motifs.Binomial(10, 3);
        foreach (int code in patterns3)
        {
            double truth = (double)Motifs.ExactCount(g10, 3, code) / subsetCount3;
            ulong before = sampleState;
            Sampling.Estimate small = Sampling.Frequency(g10, 3, code, 4000, ref sampleState);
            Check(Math.Abs(small.P - truth) <= 4.0 * small.StandardError + 1e-12,
                    "k=3 estimate within 4 standard errors for code " + code + ": " + small.P
                    + " vs " + truth);
            ulong repeatState = before;
            Sampling.Estimate again = Sampling.Frequency(g10, 3, code, 4000, ref repeatState);
            Check(again.Hits == small.Hits && again.P == small.P, "repeated sampling is identical");
        }

        int paw = Motifs.CanonicalCodeOfMatrix(Prototype(4, 4), 4);
        int c4 = Motifs.CanonicalCodeOfMatrix(Prototype(4, 4, true), 4);
        foreach (int code in new[] { paw, c4, Motifs.CanonicalCodeOfMatrix(Prototype(4, 0), 4) })
        {
            double truth = (double)Motifs.ExactCount(g10, 4, code) / Motifs.Binomial(10, 4);
            ulong s1 = 553300UL;
            Sampling.Estimate coarse = Sampling.Frequency(g10, 4, code, 500, ref s1);
            ulong s2 = 553300UL;
            Sampling.Estimate fine = Sampling.Frequency(g10, 4, code, 5000, ref s2);
            Check(Math.Abs(coarse.P - truth) <= 4.0 * coarse.StandardError + 1e-12,
                    "k=4 coarse estimate within 4 standard errors, code " + code);
            Check(Math.Abs(fine.P - truth) <= 3.0 * fine.StandardError + 1e-12,
                    "k=4 fine estimate within 3 standard errors, code " + code);
            Check(fine.StandardError <= coarse.StandardError, "a larger sample has a smaller standard error");
        }

        // Degenerate frequencies are hit exactly: p = 1 and p = 0 with no sampling noise.
        var k5 = SamplerGraph.Clique(5);
        ulong s3 = 553400UL;
        Sampling.Estimate always = Sampling.Frequency(k5, 3, triangle3, 500, ref s3);
        Check(always.Hits == 500 && Math.Abs(always.P - 1.0) <= 1e-12 && always.StandardError == 0.0,
                "every 3-subset of K5 is a triangle");
        ulong s4 = 553401UL;
        Sampling.Estimate never = Sampling.Frequency(k5, 3, edgeless3, 500, ref s4);
        Check(never.Hits == 0 && never.P == 0.0, "no 3-subset of K5 is edgeless");

        // Motif significance against a density-matched Erdos-Renyi null model.
        var real = SamplerGraph.Clique(5).PlusIsolated(7);
        Check(real.N == 12 && real.EdgeCount() == 10, "the real graph is K5 plus 7 isolated vertices");
        int densityPpm = (int)Math.Round(real.EdgeCount() * 1000000.0 /
                (real.N * (real.N - 1) / 2.0));
        Check(densityPpm == 151515 || densityPpm == 151516, "density 20/132 ppm = " + densityPpm);
        double realTriangles = Motifs.ExactCount(real, 3, triangle3);
        Check(realTriangles == 10, "K5 contributes exactly C(5,3) = 10 triangles");
        ulong nullState = 553500UL;
        int nullGraphs = 400;
        double sum = 0.0;
        double sumSquares = 0.0;
        for (int i = 0; i < nullGraphs; i++)
        {
            SamplerGraph dummy = SamplerGraph.RandomGraph(12, densityPpm, ref nullState);
            double t = Motifs.ExactCount(dummy, 3, triangle3);
            sum += t;
            sumSquares += t * t;
        }
        double mean = sum / nullGraphs;
        double variance = sumSquares / nullGraphs - mean * mean;
        Check(variance > 0.0, "the null model has a positive triangle-count variance");
        double sd = Math.Sqrt(variance);
        double zReal = (realTriangles - mean) / sd;
        double analytic = Motifs.Binomial(12, 3) * Math.Pow(densityPpm / 1000000.0, 3.0);
        Check(Math.Abs(mean - analytic) < 3.0 * sd / Math.Sqrt(nullGraphs),
                "the sampled null mean " + mean + " matches C(12,3)p^3 = " + analytic);
        Check(zReal > 5.0, "the triangle Z-score of K5+7isolated is above 5, got " + zReal);

        SamplerGraph dummyCycle = SamplerGraph.Cycle(12);
        int cyclePpm = (int)Math.Round(dummyCycle.EdgeCount() * 1000000.0 / 66.0);
        ulong cycleNullState = 553600UL;
        double cycleSum = 0.0;
        double cycleSquares = 0.0;
        for (int i = 0; i < nullGraphs; i++)
        {
            SamplerGraph dummy = SamplerGraph.RandomGraph(12, cyclePpm, ref cycleNullState);
            double t = Motifs.ExactCount(dummy, 3, triangle3);
            cycleSum += t;
            cycleSquares += t * t;
        }
        double cycleMean = cycleSum / nullGraphs;
        double cycleSd = Math.Sqrt(cycleSquares / nullGraphs - cycleMean * cycleMean);
        Check(Motifs.ExactCount(dummyCycle, 3, triangle3) == 0, "C12 is triangle-free");
        double zCycle = (0.0 - cycleMean) / cycleSd;
        Check(zCycle < 1.0, "the triangle-free cycle has a Z-score below 1, got " + zCycle);
        Check(zCycle < zReal, "the planted K5 is more triangle-significant than the cycle");

        Console.WriteLine("PASS");
    }

    private static bool[,] Prototype(int k, int edges, bool asCycle = false)
    {
        var m = new bool[k, k];
        if (asCycle)
        {
            for (int i = 0; i < k; i++)
            {
                m[i, (i + 1) % k] = true;
                m[(i + 1) % k, i] = true;
            }
            return m;
        }
        int placed = 0;
        for (int i = 0; i < k && placed < edges; i++)
        {
            for (int j = i + 1; j < k && placed < edges; j++)
            {
                m[i, j] = true;
                m[j, i] = true;
                placed++;
            }
        }
        return m;
    }

    private static int PathCode(int k)
    {
        var m = new bool[k, k];
        for (int i = 0; i + 1 < k; i++)
        {
            m[i, i + 1] = true;
            m[i + 1, i] = true;
        }
        return Motifs.CanonicalCodeOfMatrix(m, k);
    }

    private static int ClassCount(int k)
    {
        int pairs = k * (k - 1) / 2;
        var seen = new HashSet<int>();
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
            seen.Add(Motifs.CanonicalCodeOfMatrix(m, k));
        }
        return seen.Count;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
