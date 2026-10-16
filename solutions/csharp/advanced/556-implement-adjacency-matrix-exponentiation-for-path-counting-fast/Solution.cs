// Task 556 - Implement adjacency matrix exponentiation for path counting (fast exponent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Walk counting by fast (binary) exponentiation of the adjacency matrix. Over the ordinary (+,*)
// semiring the (i,j) entry of A^k counts walks of length exactly k from i to j - vertices may repeat,
// unlike the DAG case - so the total number of walks is the sum of all entries of A^k, and the same
// matrix mod a prime counts walks when the numbers get large. Squaring the base and multiplying the
// accumulator whenever the current bit of the exponent is set uses at most 2*floor(log2 k) + 1 matrix
// multiplications, and the file counts them to substantiate the "fast" in the task name rather than
// asserting it. Three independent ways of counting walks are kept: brute-force DFS enumeration of
// every walk (the reference, allowing revisits), a layer-by-layer DP over walk length, and the matrix
// power computed in two ways (binary exponentiation and repeated multiplication). Large-integer
// verification uses System.Numerics.BigInteger, and K_n has a closed form from its spectrum
// {n-1, -1, ..., -1}: (A^k)_ii = ((n-1)^k + (n-1)(-1)^k)/n and (A^k)_ij = ((n-1)^k - (-1)^k)/n, while
// the total number of walks is n(n-1)^k (n starting vertices, n-1 choices at each step).
// Random graphs come from splitmix64 (documented 64-bit generator).
//
// Documented checks: for five graphs on 5-8 vertices and every k = 0..6 the total walk counts from
// brute-force DFS, the layer DP and binary matrix exponentiation all agree (5 graphs x 7 lengths x 3
// counts = 105 three-way comparisons, plus an entrywise DP-vs-matrix-power comparison at every k); the
// K_n closed form is satisfied exactly for n = 5 and n = 6 and k = 0..8 (checked as integer identities
// n * (A^k)_ij = ...); trace(A^3) = 6 * number of triangles on the test graphs; the semigroup law
// A^(a+b) = A^a A^b holds for a, b <= 4; the modular power equals BigInteger-exact power reduced mod
// 1e9+7 for k up to 12; A^0 is the identity and an edgeless graph has no walk of positive length;
// binary exponentiation uses no more than 2*floor(log2 k) + 1 multiplications (measured, k <= 64);
// and repeated runs are identical.
using System;
using System.Collections.Generic;
using System.Numerics;

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

internal sealed class Graph
{
    internal readonly int N;
    internal readonly bool[,] Adj;
    internal readonly string Label;

    internal Graph(int n, string label)
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

    internal int TriangleCount()
    {
        int t = 0;
        for (int a = 0; a < N; a++)
        {
            for (int b = a + 1; b < N; b++)
            {
                for (int c = b + 1; c < N; c++)
                {
                    if (Adj[a, b] && Adj[b, c] && Adj[a, c]) t++;
                }
            }
        }
        return t;
    }

    internal static Graph Clique(int n)
    {
        var g = new Graph(n, "K" + n);
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++) g.AddEdge(u, v);
        }
        return g;
    }

    internal static Graph Cycle(int n)
    {
        var g = new Graph(n, "C" + n);
        for (int i = 0; i < n; i++) g.AddEdge(i, (i + 1) % n);
        return g;
    }

    internal static Graph Path(int n)
    {
        var g = new Graph(n, "P" + n);
        for (int i = 0; i + 1 < n; i++) g.AddEdge(i, i + 1);
        return g;
    }

    internal static Graph RandomGraph(int n, int probabilityPpm, ref ulong state)
    {
        var g = new Graph(n, "G(" + n + ",rand)");
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (SplitMix64.Below(ref state, 1000000) < probabilityPpm) g.AddEdge(u, v);
            }
        }
        return g;
    }

    internal long[,] AsMatrix()
    {
        var m = new long[N, N];
        for (int u = 0; u < N; u++)
        {
            for (int v = 0; v < N; v++) m[u, v] = Adj[u, v] ? 1L : 0L;
        }
        return m;
    }
}

internal static class WalkCount
{
    internal static long Multiplications;
    internal const long Modulus = 1000000007L;

    internal static long[,] Multiply(long[,] a, long[,] b, long mod)
    {
        int n = a.GetLength(0);
        var c = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int t = 0; t < n; t++)
            {
                long left = a[i, t];
                if (left == 0) continue;
                for (int j = 0; j < n; j++)
                {
                    long right = b[t, j];
                    if (right == 0) continue;
                    c[i, j] = mod > 0 ? (c[i, j] + left * right) % mod : c[i, j] + left * right;
                }
            }
        }
        return c;
    }

    internal static long[,] Power(Graph g, int exponent, long mod, bool countMultiplications)
    {
        int n = g.N;
        var result = new long[n, n];
        for (int i = 0; i < n; i++) result[i, i] = 1;
        long[,] basis = g.AsMatrix();
        if (mod > 0)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) basis[i, j] %= mod;
            }
        }
        int remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0)
            {
                result = Multiply(result, basis, mod);
                if (countMultiplications) Multiplications++;
            }
            remaining >>= 1;
            if (remaining > 0)
            {
                basis = Multiply(basis, basis, mod);
                if (countMultiplications) Multiplications++;
            }
        }
        return result;
    }

    internal static long TotalWalks(long[,] m)
    {
        long total = 0;
        for (int i = 0; i < m.GetLength(0); i++)
        {
            for (int j = 0; j < m.GetLength(1); j++) total += m[i, j];
        }
        return total;
    }

    internal static long TotalWalkCount(Graph g, int exponent)
    {
        return TotalWalks(Power(g, exponent, 0L, false));
    }

    /** Layer-by-layer DP: w_t[i,j] = number of walks of t edges from i to j. */
    internal static long[,] DpWalkMatrix(Graph g, int exponent)
    {
        int n = g.N;
        var current = new long[n, n];
        for (int i = 0; i < n; i++) current[i, i] = 1;
        for (int t = 0; t < exponent; t++)
        {
            var next = new long[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    if (current[i, k] == 0) continue;
                    for (int j = 0; j < n; j++)
                    {
                        if (g.Adj[k, j]) next[i, j] += current[i, k];
                    }
                }
            }
            current = next;
        }
        return current;
    }

    /** Brute-force enumeration of every walk of exactly `edges` edges (revisits allowed). */
    internal static long BruteWalkCount(Graph g, int edges)
    {
        long total = 0;
        for (int start = 0; start < g.N; start++) total += Dfs(g, start, edges);
        return total;
    }

    private static long Dfs(Graph g, int at, int remaining)
    {
        if (remaining == 0) return 1;
        long total = 0;
        for (int v = 0; v < g.N; v++)
        {
            if (g.Adj[at, v]) total += Dfs(g, v, remaining - 1);
        }
        return total;
    }

    internal static BigInteger[,] ExactPower(BigInteger[,] a, int exponent)
    {
        int n = a.GetLength(0);
        var result = new BigInteger[n, n];
        for (int i = 0; i < n; i++) result[i, i] = BigInteger.One;
        BigInteger[,] basis = (BigInteger[,])a.Clone();
        int remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0) result = ExactMultiply(result, basis);
            remaining >>= 1;
            if (remaining > 0) basis = ExactMultiply(basis, basis);
        }
        return result;
    }

    private static BigInteger[,] ExactMultiply(BigInteger[,] a, BigInteger[,] b)
    {
        int n = a.GetLength(0);
        var c = new BigInteger[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int t = 0; t < n; t++)
            {
                if (a[i, t].IsZero) continue;
                for (int j = 0; j < n; j++) c[i, j] += a[i, t] * b[t, j];
            }
        }
        return c;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 556001UL;
        var graphs = new List<Graph>
        {
            Graph.Clique(5),
            Graph.Cycle(6),
            Graph.Path(8),
            Graph.RandomGraph(7, 450000, ref rng),
            Graph.RandomGraph(6, 250000, ref rng),
        };

        int comparisons = 0;
        foreach (Graph g in graphs)
        {
            for (int k = 0; k <= 6; k++)
            {
                long brute = WalkCount.BruteWalkCount(g, k);
                long dp = WalkCount.TotalWalks(WalkCount.DpWalkMatrix(g, k));
                long fast = WalkCount.TotalWalkCount(g, k);
                Check(brute == dp, g.Label + ": DFS " + brute + " != DP " + dp + " at k = " + k);
                Check(dp == fast, g.Label + ": DP " + dp + " != matrix power " + fast + " at k = " + k);
                long[,] byDp = WalkCount.DpWalkMatrix(g, k);
                long[,] byPower = WalkCount.Power(g, k, 0L, false);
                for (int i = 0; i < g.N; i++)
                {
                    for (int j = 0; j < g.N; j++)
                    {
                        Check(byDp[i, j] == byPower[i, j], g.Label + ": entry " + i + "," + j
                                + " differs at k = " + k);
                    }
                }
                comparisons += 3;
            }
            Check(WalkCount.TotalWalkCount(g, 0) == g.N, g.Label + ": A^0 is the identity");
        }
        Check(comparisons == 105, "105 three-way comparisons, got " + comparisons);

        // Closed form for K_n from the eigenvalues n-1 and -1.
        foreach (int n in new[] { 5, 6 })
        {
            var kn = Graph.Clique(n);
            for (int k = 0; k <= 8; k++)
            {
                long[,] m = WalkCount.Power(kn, k, 0L, false);
                long power = 1;
                for (int e = 0; e < k; e++) power *= n - 1;
                long alternating = ((k % 2) == 0) ? 1 : -1;
                Check(n * m[0, 0] == power + (n - 1) * alternating, "K" + n + " diagonal closed form at k = "
                        + k + ": " + (n * m[0, 0]) + " vs " + (power + (n - 1) * alternating));
                Check(n * m[0, 1] == power - alternating, "K" + n + " off-diagonal closed form at k = " + k);
                Check(WalkCount.TotalWalkCount(kn, k) == n * power,
                        "K" + n + " total walk count is n(n-1)^k at k = " + k + ", got "
                        + WalkCount.TotalWalkCount(kn, k));
            }
        }

        // trace(A^3) = 6 * triangles.
        foreach (Graph g in graphs)
        {
            long[,] cube = WalkCount.Power(g, 3, 0L, false);
            long trace = 0;
            for (int i = 0; i < g.N; i++) trace += cube[i, i];
            Check(trace == 6L * g.TriangleCount(), g.Label + ": trace(A^3) = 6 * triangles, got " + trace
                    + " vs " + (6L * g.TriangleCount()));
        }

        // Semigroup law and modular agreement with exact BigInteger arithmetic.
        var rand = graphs[3];
        for (int a = 0; a <= 4; a++)
        {
            for (int b = 0; b <= 4; b++)
            {
                long[,] left = WalkCount.Power(rand, a + b, 0L, false);
                long[,] right = WalkCount.Multiply(WalkCount.Power(rand, a, 0L, false),
                        WalkCount.Power(rand, b, 0L, false), 0L);
                for (int i = 0; i < rand.N; i++)
                {
                    for (int j = 0; j < rand.N; j++)
                    {
                        Check(left[i, j] == right[i, j], "A^(a+b) != A^a A^b at a=" + a + " b=" + b
                                + " cell " + i + "," + j);
                    }
                }
            }
        }

        BigInteger[,] exact = new BigInteger[rand.N, rand.N];
        for (int i = 0; i < rand.N; i++)
        {
            for (int j = 0; j < rand.N; j++) exact[i, j] = rand.Adj[i, j] ? BigInteger.One : BigInteger.Zero;
        }
        for (int k = 0; k <= 12; k++)
        {
            BigInteger[,] big = WalkCount.ExactPower(exact, k);
            long[,] modular = WalkCount.Power(rand, k, WalkCount.Modulus, false);
            for (int i = 0; i < rand.N; i++)
            {
                for (int j = 0; j < rand.N; j++)
                {
                    Check(modular[i, j] == (long)(big[i, j] % WalkCount.Modulus),
                            "modular power != exact mod 1e9+7 at k = " + k + " cell " + i + "," + j);
                }
            }
        }

        // The "fast" claim: at most 2*floor(log2 k) + 1 multiplications.
        foreach (int exponent in new[] { 1, 2, 3, 7, 16, 31, 64 })
        {
            WalkCount.Multiplications = 0;
            WalkCount.Power(rand, exponent, 0L, true);
            int log2 = 0;
            while ((1 << (log2 + 1)) <= exponent) log2++;
            Check(WalkCount.Multiplications <= 2 * log2 + 1, "exponent " + exponent + " needs "
                    + WalkCount.Multiplications + " multiplications, budget " + (2 * log2 + 1));
        }
        WalkCount.Multiplications = 0;
        WalkCount.Power(rand, 64, 0L, true);
        long fastMults = WalkCount.Multiplications;
        WalkCount.Multiplications = 0;
        WalkCount.Power(rand, 13, 0L, true);
        Check(fastMults <= 2 * 6 + 1 && fastMults > 0, "k = 64 measured " + fastMults + " multiplications");

        // Boundary cases.
        var edgeless = new Graph(4, "empty4");
        Check(WalkCount.TotalWalkCount(edgeless, 0) == 4 && WalkCount.TotalWalkCount(edgeless, 1) == 0
                && WalkCount.TotalWalkCount(edgeless, 9) == 0,
                "an edgeless graph has no walk of positive length");

        // Determinism.
        ulong sameRng = 556001UL;
        var repeat = Graph.RandomGraph(7, 450000, ref sameRng);
        for (int k = 0; k <= 6; k++)
        {
            Check(WalkCount.TotalWalkCount(repeat, k) == WalkCount.TotalWalkCount(graphs[3], k),
                    "the same seed reproduces the graph and counts at k = " + k);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
