// Task 557 - Implement algebraic path finding with min-plus matrix multiply (tropical semiring). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Shortest-path combinatorics in the tropical (min-plus) semiring, where "addition" is min, "multiplication"
// is +, the additive identity is +infinity and the multiplicative identity is I_0 (0 on the diagonal,
// infinity elsewhere). Matrix powers then have a precise algebraic meaning: A^(k) with the tropical
// product is the matrix of minimum-weight walks with *exactly* k edges, the truncated sum
// I_0 (+) A (+) A^(2) (+) ... (+) A^(n-1) is the all-pairs shortest path matrix (every shortest walk can
// be taken to be a simple path, so n-1 edges suffice), and the same computation is textbook
// Floyd-Warshall - two algorithms of very different complexity but identical output, which the file
// checks cell by cell. Infinity handling uses the sentinel long.MaxValue/4, and products skip infinite
// entries so no overflow can occur. Negative edge weights are allowed (the tests include a weighted DAG
// and a graph with a negative cycle, where the closure diagonal goes negative and the "distance" is
// honestly reported as -infinity rather than a number). A documented caveat is included: exactly-k
// powers are *not* shortest-path distances - on a 3-vertex path with a single edge, A^(2)[0][1] is
// infinity while the shortest-path matrix has 1 - so the truncation to k <= n-1 is what makes the
// closure correct, not the power on its own. Random weights come from splitmix64 (documented 64-bit
// generator) instead of the repo's 16-bit LCG.
//
// Documented checks: for a 6-vertex random weighted digraph and every k = 0..6 the exactly-k tropical
// power equals both a layer DP and a brute-force minimisation over all walks (3 x 7 x 36 cells);
// the truncated closure equals Floyd-Warshall and equals all-pairs Dijkstra on the non-negative
// instance (cell by cell, infinity included); the semiring laws hold on matrices containing infinities
// (associativity of the product, both distributive laws over min, I_0 as the identity, idempotence
// a (+) a = a, monotonicity of the product); on a 5-vertex weighted DAG the closed distances equal a
// brute-force minimum over all simple paths; a negative cycle makes the closure diagonal negative,
// which is asserted explicitly as the honest "no finite distance" signal; the hand-computed 4-vertex
// example gives the expected distances and the parent pointers reconstruct a walk whose weight matches
// the claimed distance; unreachable pairs stay at infinity; and the 1-vertex case gives A^(0) = [[0]]
// and A^(1) = [[infinity]].
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

internal static class MinPlus
{
    internal const long Inf = long.MaxValue / 4;

    internal static long[,] FromWeights(int n, int[][] triples)
    {
        var w = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) w[i, j] = i == j ? 0 : Inf;
        }
        foreach (int[] t in triples) w[t[0], t[1]] = Math.Min(w[t[0], t[1]], t[2]);
        return w;
    }

    /** Adjacency-style matrix: infinity on the diagonal (no zero-length edge). */
    internal static long[,] EdgesOnly(long[,] w)
    {
        int n = w.GetLength(0);
        var a = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) a[i, j] = i == j ? Inf : w[i, j];
        }
        return a;
    }

    internal static long[,] Identity(int n)
    {
        var m = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) m[i, j] = i == j ? 0 : Inf;
        }
        return m;
    }

    internal static long[,] Multiply(long[,] a, long[,] b)
    {
        int n = a.GetLength(0);
        var c = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                long best = Inf;
                for (int t = 0; t < n; t++)
                {
                    if (a[i, t] >= Inf || b[t, j] >= Inf) continue;
                    long candidate = a[i, t] + b[t, j];
                    if (candidate < best) best = candidate;
                }
                c[i, j] = best;
            }
        }
        return c;
    }

    internal static long[,] Min(long[,] a, long[,] b)
    {
        int n = a.GetLength(0);
        var c = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) c[i, j] = Math.Min(a[i, j], b[i, j]);
        }
        return c;
    }

    internal static long[,] Power(long[,] a, int exponent)
    {
        int n = a.GetLength(0);
        long[,] result = Identity(n);
        long[,] basis = (long[,])a.Clone();
        int remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0) result = Multiply(result, basis);
            remaining >>= 1;
            if (remaining > 0) basis = Multiply(basis, basis);
        }
        return result;
    }

    internal static long[,] WalkDp(long[,] a, int edges)
    {
        int n = a.GetLength(0);
        long[,] current = Identity(n);
        for (int step = 0; step < edges; step++) current = Multiply(current, a);
        return current;
    }

    internal static long[,] FloydWarshall(long[,] w)
    {
        int n = w.GetLength(0);
        var d = (long[,])w.Clone();
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                if (d[i, k] >= Inf) continue;
                for (int j = 0; j < n; j++)
                {
                    if (d[k, j] >= Inf) continue;
                    long candidate = d[i, k] + d[k, j];
                    if (candidate < d[i, j]) d[i, j] = candidate;
                }
            }
        }
        return d;
    }

    internal static long[] Dijkstra(long[,] w, int source, int[]? parent)
    {
        int n = w.GetLength(0);
        var dist = new long[n];
        var done = new bool[n];
        for (int i = 0; i < n; i++)
        {
            dist[i] = Inf;
            if (parent != null) parent[i] = -1;
        }
        dist[source] = 0;
        for (int step = 0; step < n; step++)
        {
            int best = -1;
            for (int v = 0; v < n; v++)
            {
                if (!done[v] && (best < 0 || dist[v] < dist[best])) best = v;
            }
            if (best < 0 || dist[best] >= Inf) break;
            done[best] = true;
            for (int v = 0; v < n; v++)
            {
                if (v == best || w[best, v] >= Inf) continue;
                long candidate = dist[best] + w[best, v];
                if (candidate < dist[v])
                {
                    dist[v] = candidate;
                    if (parent != null) parent[v] = best;
                }
            }
        }
        return dist;
    }

    /** Minimum-weight simple path from s to t, by exhaustive DFS (only for tiny graphs). */
    internal static long MinSimplePath(long[,] w, int s, int t)
    {
        int n = w.GetLength(0);
        var visited = new bool[n];
        visited[s] = true;
        return Search(w, s, t, 0L, visited);
    }

    private static long Search(long[,] w, int at, int target, long cost, bool[] visited)
    {
        if (at == target) return cost;
        long best = Inf;
        int n = w.GetLength(0);
        for (int v = 0; v < n; v++)
        {
            if (visited[v] || v == at || w[at, v] >= Inf) continue;
            visited[v] = true;
            long sub = Search(w, v, target, cost + w[at, v], visited);
            visited[v] = false;
            if (sub < best) best = sub;
        }
        return best;
    }

    internal static long BruteMinWalk(long[,] a, int s, int t, int edges)
    {
        if (edges == 0) return s == t ? 0 : Inf;
        long best = Inf;
        int n = a.GetLength(0);
        for (int v = 0; v < n; v++)
        {
            if (a[s, v] >= Inf) continue;
            long sub = BruteMinWalk(a, v, t, edges - 1);
            if (sub < Inf) best = Math.Min(best, a[s, v] + sub);
        }
        return best;
    }

    /** Truncated Kleene closure: I_0 (+) A (+) ... (+) A^(n-1). */
    internal static long[,] Closure(long[,] a)
    {
        int n = a.GetLength(0);
        long[,] total = Identity(n);
        long[,] power = Identity(n);
        for (int k = 0; k < n - 1; k++)
        {
            power = Multiply(power, a);
            total = Min(total, power);
        }
        return total;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 557001UL;
        int n = 6;
        var edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                if (SplitMix64.Below(ref rng, 100) < 45)
                {
                    edges.Add(new[] { u, v, 1 + SplitMix64.Below(ref rng, 9) });
                }
            }
        }
        long[,] weights = MinPlus.FromWeights(n, edges.ToArray());
        long[,] a = MinPlus.EdgesOnly(weights);
        Check(weights[0, 0] == 0 && a[0, 0] == MinPlus.Inf, "the diagonal differs between the two forms");

        // Exactly-k tropical powers: DP and brute force agree.
        int cells = 0;
        for (int k = 0; k <= 6; k++)
        {
            long[,] fast = MinPlus.Power(a, k);
            long[,] dp = MinPlus.WalkDp(a, k);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Check(fast[i, j] == dp[i, j], "power != DP at k=" + k + " cell " + i + "," + j);
                    long brute = MinPlus.BruteMinWalk(a, i, j, k);
                    Check(fast[i, j] == brute, "power != brute force at k=" + k + " cell " + i + ","
                            + j + ": " + fast[i, j] + " vs " + brute);
                    cells += 2;
                }
            }
        }
        Check(cells == 2 * 7 * 36, "504 cell comparisons, got " + cells);

        // The truncated closure equals Floyd-Warshall and Dijkstra.
        long[,] closure = MinPlus.Closure(a);
        long[,] fw = MinPlus.FloydWarshall(weights);
        long[,] dijkstra = new long[n, n];
        for (int s = 0; s < n; s++)
        {
            long[] dist = MinPlus.Dijkstra(weights, s, null);
            for (int j = 0; j < n; j++) dijkstra[s, j] = dist[j];
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Check(closure[i, j] == fw[i, j], "closure != Floyd-Warshall at " + i + "," + j + ": "
                        + closure[i, j] + " vs " + fw[i, j]);
                Check(fw[i, j] == dijkstra[i, j], "Floyd-Warshall != Dijkstra at " + i + "," + j);
            }
            Check(closure[i, i] == 0, "the closure keeps the zero diagonal (no negative cycle here)");
        }

        // The documented caveat: exactly-k powers are not shortest-path distances.
        long[,] single = MinPlus.FromWeights(3, new[] { new[] { 0, 1, 1 } });
        long[,] singleA = MinPlus.EdgesOnly(single);
        Check(MinPlus.Power(singleA, 2)[0, 1] == MinPlus.Inf, "exactly-2-edge walk from 0 to 1 is infinite");
        Check(MinPlus.Closure(singleA)[0, 1] == 1, "but the closure correctly reports distance 1");
        Check(MinPlus.Power(singleA, 10)[0, 1] == MinPlus.Inf, "and any even power stays infinite");
        Check(MinPlus.Power(singleA, 1)[0, 1] == 1 && MinPlus.Power(singleA, 3)[0, 1] == MinPlus.Inf,
                "only the odd powers up to the path length see the edge");

        // Semiring laws on matrices with infinities.
        ulong lawRng = 557100UL;
        var mats = new List<long[,]>();
        for (int m = 0; m < 3; m++)
        {
            var t = new List<int[]>();
            for (int u = 0; u < 4; u++)
            {
                for (int v = 0; v < 4; v++)
                {
                    if (u != v && SplitMix64.Below(ref lawRng, 100) < 55)
                    {
                        t.Add(new[] { u, v, 1 + SplitMix64.Below(ref lawRng, 6) });
                    }
                }
            }
            mats.Add(MinPlus.EdgesOnly(MinPlus.FromWeights(4, t.ToArray())));
        }
        long[,] ma = mats[0], mb = mats[1], mc = mats[2];
        long[,] leftAssoc = MinPlus.Multiply(MinPlus.Multiply(ma, mb), mc);
        long[,] rightAssoc = MinPlus.Multiply(ma, MinPlus.Multiply(mb, mc));
        long[,] leftDistrib = MinPlus.Multiply(MinPlus.Min(ma, mb), mc);
        long[,] rightDistrib = MinPlus.Min(MinPlus.Multiply(ma, mc), MinPlus.Multiply(mb, mc));
        long[,] identity = MinPlus.Identity(4);
        long[,] withIdentity = MinPlus.Multiply(ma, identity);
        long[,] lowerMatrix = MinPlus.Min(ma, mb);
        long[,] lowerProduct = MinPlus.Multiply(lowerMatrix, mc);
        long[,] upperProduct = MinPlus.Multiply(ma, mc);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Check(leftAssoc[i, j] == rightAssoc[i, j], "associativity at " + i + "," + j);
                Check(leftDistrib[i, j] == rightDistrib[i, j], "distributivity at " + i + "," + j);
                Check(withIdentity[i, j] == ma[i, j], "I_0 is the multiplicative identity at " + i + "," + j);
                Check(MinPlus.Min(ma, ma)[i, j] == ma[i, j], "min is idempotent at " + i + "," + j);
                Check(lowerProduct[i, j] <= upperProduct[i, j],
                        "A <= B implies A product C <= B product C at " + i + "," + j);
            }
        }

        // Weighted DAG with negative edges: closure equals the brute-force minimum simple path.
        var dagTriples = new List<int[]>
        {
            new[] { 0, 1, 4 }, new[] { 0, 2, 5 }, new[] { 1, 2, -3 }, new[] { 1, 3, 6 },
            new[] { 2, 3, 2 }, new[] { 3, 4, -4 }, new[] { 0, 4, 20 },
        };
        long[,] dagWeights = MinPlus.FromWeights(5, dagTriples.ToArray());
        long[,] dagA = MinPlus.EdgesOnly(dagWeights);
        long[,] dagClosure = MinPlus.Closure(dagA);
        long[,] dagFw = MinPlus.FloydWarshall(dagWeights);
        Check(dagClosure[0, 0] == 0 && dagClosure[0, 2] == 1 && dagClosure[0, 3] == 3
                && dagClosure[0, 4] == -1,
                "hand-computed DAG distances 0->2 = 1 (0-1-2), 0->3 = 3 (0-1-2-3), 0->4 = -1, got "
                + dagClosure[0, 2] + "," + dagClosure[0, 3] + "," + dagClosure[0, 4]);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                Check(dagFw[i, j] == dagClosure[i, j], "DAG closure != Floyd-Warshall at " + i + "," + j);
                Check(dagClosure[i, j] == MinPlus.MinSimplePath(dagWeights, i, j),
                        "DAG closure != brute-force simple path at " + i + "," + j);
            }
        }

        // A negative cycle drives the closure diagonal negative: the honest signal for -infinity.
        var negCycle = new List<int[]>
        {
            new[] { 0, 1, 1 }, new[] { 1, 2, -2 }, new[] { 2, 1, -2 }, new[] { 2, 3, 5 },
        };
        long[,] negWeights = MinPlus.FromWeights(4, negCycle.ToArray());
        long[,] negClosure = MinPlus.FloydWarshall(negWeights);
        Check(negClosure[1, 1] < 0 && negClosure[2, 2] < 0, "the negative cycle shows up on the diagonal");
        Check(negClosure[0, 1] < 1, "a negative cycle also improves distances into it");

        // Hand-computed 4-vertex example with path reconstruction.
        var example = new List<int[]>
        {
            new[] { 0, 1, 2 }, new[] { 0, 2, 8 }, new[] { 1, 2, 3 }, new[] { 1, 3, 9 },
            new[] { 2, 3, 4 },
        };
        long[,] exampleWeights = MinPlus.FromWeights(4, example.ToArray());
        var parents = new int[4];
        long[] distFromZero = MinPlus.Dijkstra(exampleWeights, 0, parents);
        Check(distFromZero[0] == 0 && distFromZero[1] == 2 && distFromZero[2] == 5 && distFromZero[3] == 9,
                "hand-computed distances 0,2,5,9, got " + distFromZero[0] + "," + distFromZero[1] + ","
                + distFromZero[2] + "," + distFromZero[3]);
        long[,] exampleFw = MinPlus.FloydWarshall(exampleWeights);
        long[,] exampleClosure = MinPlus.Closure(MinPlus.EdgesOnly(exampleWeights));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Check(exampleFw[i, j] == exampleClosure[i, j], "example closure != Floyd-Warshall");
            }
        }
        var walk = new List<int>();
        for (int at = 3; at != -1; at = parents[at]) walk.Add(at);
        walk.Reverse();
        Check(walk[0] == 0 && walk[walk.Count - 1] == 3 && walk.Count == 4, "reconstructed path 0,1,2,3");
        Check(walk[1] == 1 && walk[2] == 2, "the parent pointers give the tight path");
        long walked = 0;
        for (int i = 0; i + 1 < walk.Count; i++) walked += exampleWeights[walk[i], walk[i + 1]];
        Check(walked == distFromZero[3], "the reconstructed walk weighs exactly the reported distance");

        // Unreachable pairs and the 1-vertex boundary.
        long[,] split = MinPlus.EdgesOnly(MinPlus.FromWeights(4, new[] { new[] { 0, 1, 1 },
                new[] { 2, 3, 1 } }));
        long[,] splitClosure = MinPlus.Closure(split);
        Check(splitClosure[0, 1] == 1 && splitClosure[0, 2] == MinPlus.Inf
                && splitClosure[0, 3] == MinPlus.Inf && splitClosure[1, 0] == MinPlus.Inf,
                "unreachable pairs stay at infinity");
        long[,] one = MinPlus.EdgesOnly(MinPlus.FromWeights(1, Array.Empty<int[]>()));
        Check(MinPlus.Power(one, 0)[0, 0] == 0, "A^(0) of the 1-vertex graph is [[0]]");
        Check(MinPlus.Power(one, 1)[0, 0] == MinPlus.Inf, "A^(1) of the 1-vertex graph is [[infinity]]");
        Check(MinPlus.Closure(one)[0, 0] == 0, "its closure is [[0]]");
        Check(MinPlus.Inf + MinPlus.Inf > MinPlus.Inf, "the sentinel cannot overflow when doubled");

        // Determinism.
        ulong sameRng = 557001UL;
        var repeat = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                if (SplitMix64.Below(ref sameRng, 100) < 45)
                {
                    repeat.Add(new[] { u, v, 1 + SplitMix64.Below(ref sameRng, 9) });
                }
            }
        }
        long[,] repeatWeights = MinPlus.FromWeights(n, repeat.ToArray());
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Check(repeatWeights[i, j] == weights[i, j], "the same seed reproduces the graph at "
                        + i + "," + j);
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
