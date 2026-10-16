// Task 555 - Implement dynamic programming for counting paths of length k in DAG (matrix exponentiation). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Counting walks of exactly k edges in a directed acyclic graph by dynamic programming over a
// topological order: with f_0[v] = 1 for every v (the empty walk at v) and
// f_t[v] = sum over in-neighbours u of f_{t-1}[u], the answer is sum over v of f_k[v]; the whole
// profile is O(k * (n + m)). Because a DAG has no directed cycle, every walk is automatically a
// simple path, so the same numbers count simple paths of k edges - a fact this file checks against a
// DFS that enumerates simple paths explicitly rather than assuming it. Two independent
// implementations are kept: the DP, and binary exponentiation of the adjacency matrix over the
// ordinary (+,*) semiring, whose (i,j) entry after k multiplications counts walks of length k from i
// to j. Random DAGs use splitmix64 (documented 64-bit generator) and are produced in topological
// order (edges only from the lower to the higher label), so the order is known by construction.
//
// Documented checks: on the transitive tournament T_n (edges i -> j for i < j) the closed form for
// walks of k edges is C(n, k+1) - an increasing (k+1)-tuple of labels - which the DP reproduces for
// n = 7 and every k = 0..6, and summing over k gives 2^7 - 1 = 127; on random DAGs on 9 vertices the
// DP agrees with both the DFS path enumeration and matrix exponentiation for every k = 0..8
// (3 methods x 9 lengths); k = 0 gives n (one empty walk per vertex) and k = 1 gives the edge count;
// the longest path computed by its own DP is an upper bound for the lengths with a non-zero count and
// the first length beyond it has count 0; walks equal simple paths on the tested DAGs (checked against
// the enumerator, not assumed); a single vertex has count 1 at k = 0 and 0 at k = 1; and repeated runs
// give identical profiles.
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

internal sealed class Dag
{
    internal readonly int N;
    internal readonly bool[,] Adj;

    internal Dag(int n)
    {
        N = n;
        Adj = new bool[n, n];
    }

    internal void AddEdge(int u, int v)
    {
        Adj[u, v] = true;
    }

    internal int EdgeCount()
    {
        int e = 0;
        for (int u = 0; u < N; u++)
        {
            for (int v = 0; v < N; v++)
            {
                if (Adj[u, v]) e++;
            }
        }
        return e;
    }

    /** Transitive tournament: i -> j exactly when i < j. */
    internal static Dag TransitiveTournament(int n)
    {
        var g = new Dag(n);
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++) g.AddEdge(u, v);
        }
        return g;
    }

    /** Edges only from lower to higher labels, so 0..n-1 is a topological order. */
    internal static Dag RandomDag(int n, int probabilityPpm, ref ulong state)
    {
        var g = new Dag(n);
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

internal static class PathCount
{
    /** f_t[v] = number of walks of t edges ending at v; topological order is 0..n-1. */
    internal static long[] Profile(Dag g, int maxK)
    {
        var result = new long[maxK + 1];
        var current = new long[g.N];
        for (int v = 0; v < g.N; v++) current[v] = 1;
        long total = 0;
        for (int v = 0; v < g.N; v++) total += current[v];
        result[0] = total;
        for (int t = 1; t <= maxK; t++)
        {
            var next = new long[g.N];
            for (int u = 0; u < g.N; u++)
            {
                if (current[u] == 0) continue;
                for (int v = u + 1; v < g.N; v++)
                {
                    if (g.Adj[u, v]) next[v] += current[u];
                }
            }
            total = 0;
            for (int v = 0; v < g.N; v++) total += next[v];
            result[t] = total;
            current = next;
        }
        return result;
    }

    internal static long Count(Dag g, int k)
    {
        return Profile(g, k)[k];
    }

    internal static long[,] Multiply(Dag g, long[,] a, long[,] b)
    {
        int n = g.N;
        var c = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int t = 0; t < n; t++)
            {
                if (a[i, t] == 0) continue;
                for (int j = 0; j < n; j++)
                {
                    if (b[t, j] != 0) c[i, j] += a[i, t] * b[t, j];
                }
            }
        }
        return c;
    }

    internal static long[,] Power(Dag g, int k)
    {
        int n = g.N;
        var result = new long[n, n];
        for (int i = 0; i < n; i++) result[i, i] = 1;
        var basis = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) basis[i, j] = g.Adj[i, j] ? 1 : 0;
        }
        int exponent = k;
        while (exponent > 0)
        {
            if ((exponent & 1) != 0) result = Multiply(g, result, basis);
            exponent >>= 1;
            if (exponent > 0) basis = Multiply(g, basis, basis);
        }
        return result;
    }

    internal static long MatrixPowerTotal(Dag g, int k)
    {
        long[,] m = Power(g, k);
        long total = 0;
        for (int i = 0; i < g.N; i++)
        {
            for (int j = 0; j < g.N; j++) total += m[i, j];
        }
        return total;
    }

    /** Enumerates simple paths of exactly k edges by DFS (independent of the DP). */
    internal static long SimplePaths(Dag g, int k)
    {
        long total = 0;
        var visited = new bool[g.N];
        for (int start = 0; start < g.N; start++)
        {
            visited[start] = true;
            total += Walk(g, start, k, visited);
            visited[start] = false;
        }
        return total;
    }

    private static long Walk(Dag g, int at, int remaining, bool[] visited)
    {
        if (remaining == 0) return 1;
        long total = 0;
        for (int v = 0; v < g.N; v++)
        {
            if (!g.Adj[at, v] || visited[v]) continue;
            visited[v] = true;
            total += Walk(g, v, remaining - 1, visited);
            visited[v] = false;
        }
        return total;
    }

    /** Longest path in edges, by the standard DP over the topological order. */
    internal static int LongestPathLength(Dag g)
    {
        var longest = new int[g.N];
        int best = 0;
        for (int u = 0; u < g.N; u++)
        {
            for (int v = u + 1; v < g.N; v++)
            {
                if (g.Adj[u, v] && longest[u] + 1 > longest[v]) longest[v] = longest[u] + 1;
            }
            if (longest[u] > best) best = longest[u];
        }
        return best;
    }

    internal static long Binomial(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        long result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        // Closed form on the transitive tournament: increasing (k+1)-tuples of 7 labels.
        var t7 = Dag.TransitiveTournament(7);
        long[] profile = PathCount.Profile(t7, 7);
        long sum = 0;
        for (int k = 0; k <= 6; k++)
        {
            Check(profile[k] == PathCount.Binomial(7, k + 1), "T7 walk count for k=" + k + " is C(7,k+1)="
                    + PathCount.Binomial(7, k + 1) + ", got " + profile[k]);
            sum += profile[k];
        }
        Check(profile[7] == 0, "T7 has no walk of 7 edges");
        Check(sum == 127, "the path profile of T7 sums to 2^7 - 1 = 127, got " + sum);
        Check(PathCount.LongestPathLength(t7) == 6, "the longest path of T7 has 6 edges");

        // Random DAGs: DP vs DFS path enumeration vs matrix exponentiation.
        ulong rng = 555001UL;
        var dags = new List<Dag>();
        for (int i = 0; i < 4; i++) dags.Add(Dag.RandomDag(9, 300000 + i * 150000, ref rng));
        dags.Add(Dag.TransitiveTournament(9));
        int comparisons = 0;
        foreach (Dag g in dags)
        {
            long[] dp = PathCount.Profile(g, g.N);
            Check(dp[0] == g.N, "k = 0 gives one empty walk per vertex");
            Check(dp[1] == g.EdgeCount(), "k = 1 gives the edge count");
            int longest = PathCount.LongestPathLength(g);
            for (int k = 0; k <= g.N; k++)
            {
                long matrix = PathCount.MatrixPowerTotal(g, k);
                long dfs = PathCount.SimplePaths(g, k);
                Check(dp[k] == matrix, "DP != matrix exponentiation at k = " + k + ": " + dp[k]
                        + " vs " + matrix);
                Check(dp[k] == dfs, "DP != DFS simple paths at k = " + k + ": " + dp[k] + " vs " + dfs);
                if (k > longest) Check(dp[k] == 0, "no walk longer than the longest path (k = " + k + ")");
                else Check(dp[k] > 0, "some walk exists for k = " + k + " <= longest = " + longest);
                comparisons += 3;
            }
        }
        Check(comparisons == 150, "3 methods x 10 lengths x 5 DAGs = 150 comparisons, got " + comparisons);

        // Repeated multiplication must equal binary exponentiation too.
        var small = Dag.RandomDag(6, 450000, ref rng);
        long[,] powered = PathCount.Power(small, 1);
        long[,] naive = PathCount.Power(small, 1);
        for (int exponent = 1; exponent <= 6; exponent++)
        {
            if (exponent > 1)
            {
                powered = PathCount.Multiply(small, powered, naive);
            }
            long[,] fast = PathCount.Power(small, exponent);
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    Check(powered[i, j] == fast[i, j], "repeated multiplication != fast power at exponent "
                            + exponent + " cell " + i + "," + j);
                }
            }
        }

        // Boundary cases.
        var single = new Dag(1);
        Check(PathCount.Count(single, 0) == 1 && PathCount.Count(single, 1) == 0,
                "a single vertex has one empty walk and no edge");
        var empty3 = new Dag(3);
        Check(PathCount.Count(empty3, 0) == 3 && PathCount.Count(empty3, 1) == 0
                && PathCount.Count(empty3, 5) == 0, "an edgeless DAG only has empty walks");

        // Determinism.
        long[] repeat = PathCount.Profile(t7, 7);
        for (int k = 0; k <= 7; k++) Check(repeat[k] == profile[k], "profile is reproducible at k = " + k);
        ulong sameRng = 555001UL;
        var repeatDag = Dag.RandomDag(9, 300000, ref sameRng);
        Check(PathCount.Count(repeatDag, 4) == PathCount.Count(dags[0], 4),
                "the same seed reproduces the same DAG and count");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
