// Task 535 - Implement dynamic programming on DAG with topological order caching. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
//
// The edge set of a directed graph becomes a reusable Dag object that computes the topological
// order exactly once - Kahn's algorithm with sorted adjacency lists, ready nodes appended in index
// order, so the order is a pure function of the edge set - and caches it in `Order`. Every dynamic
// program then walks that cached array instead of re-deriving an order: the longest chain ending at
// each node (chain[v] = max over in-edges of chain[u] + 1) and the number of distinct non-empty
// paths ending at each node (paths[v] = sum over in-edges of paths[u] + 1), the latter both exactly
// and modulo 1e9+7. A CachingDp object re-derives the same tables through a memoized depth-first
// recursion and counts how many table entries it actually filled, which is what makes the caching
// claim measurable rather than asserted.
//
// Documented checks: the cached order is a permutation whose edges all point forward and it is
// reproduced identically by an independent second construction; on an LCG-random 24-node DAG
// (seed 20035, 30% of the i<j pairs) the memoized tables equal the cached-order tables
// element-wise, the memo tables are filled exactly once per node (24 computations each) and a
// second full pass performs zero further computations, so the cache really is reused; the longest
// chains also agree with an independent Bellman-Ford relaxation; on a 9-node DAG the exact path
// table equals exhaustive enumeration of every path (long arithmetic, overflow guarded); on the
// Fibonacci DAG (edges i->i+1 and i->i+2, 21 nodes) the number of 0->20 paths equals F(21) = 10946
// and the longest chain to node 20 has 20 edges; one extra back edge (20->0), a 2-cycle and a
// self loop each destroy the cached order (Acyclic = false, Order empty); a single-node and a
// five-isolated-node graph cover the degenerate boundary; and the hand-checked diamond
// 0->1,0->2,1->2,1->3,2->3 has paths-ending-at-3 = 6 and 10 non-empty paths overall.
using System;
using System.Collections.Generic;

internal sealed class Dag
{
    internal const long Mod = 1000000007L;

    internal readonly int N;
    internal readonly List<int>[] Out;
    internal readonly List<int>[] In;
    internal readonly int[] Order;
    internal readonly bool Acyclic;

    internal Dag(int n, IReadOnlyList<int[]> edges)
    {
        N = n;
        Out = new List<int>[n];
        In = new List<int>[n];
        for (int v = 0; v < n; v++)
        {
            Out[v] = new List<int>();
            In[v] = new List<int>();
        }
        for (int i = 0; i < edges.Count; i++)
        {
            Out[edges[i][0]].Add(edges[i][1]);
            In[edges[i][1]].Add(edges[i][0]);
        }
        for (int v = 0; v < n; v++)
        {
            Out[v].Sort();
            In[v].Sort();
        }

        // Kahn's algorithm: the ready list is appended in index order, which fixes the order
        // deterministically for a fixed edge multiset.
        var order = new int[n];
        var indegree = new int[n];
        for (int v = 0; v < n; v++) indegree[v] = In[v].Count;
        var ready = new List<int>();
        for (int v = 0; v < n; v++)
        {
            if (indegree[v] == 0) ready.Add(v);
        }
        int placed = 0;
        for (int head = 0; head < ready.Count; head++)
        {
            int v = ready[head];
            order[placed++] = v;
            foreach (int w in Out[v])
            {
                if (--indegree[w] == 0) ready.Add(w);
            }
        }
        Acyclic = placed == n;
        Order = Acyclic ? order : Array.Empty<int>();
    }

    internal int SourceCount()
    {
        int count = 0;
        for (int v = 0; v < N; v++)
        {
            if (In[v].Count == 0) count++;
        }
        return count;
    }

    /** Longest chain (in edges) ending at every node, one forward pass over the cached order. */
    internal int[] LongestEndingIterative()
    {
        var chain = new int[N];
        foreach (int v in Order)
        {
            int best = 0;
            foreach (int u in In[v])
            {
                if (chain[u] + 1 > best) best = chain[u] + 1;
            }
            chain[v] = best;
        }
        return chain;
    }

    /** Independent method: relax every edge until the longest distances stop changing. */
    internal int[] LongestEndingBellmanFord()
    {
        var dist = new int[N];
        for (int pass = 0; pass < N; pass++)
        {
            bool changed = false;
            for (int v = 0; v < N; v++)
            {
                foreach (int w in Out[v])
                {
                    if (dist[v] + 1 > dist[w])
                    {
                        dist[w] = dist[v] + 1;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
        }
        return dist;
    }

    /** Paths ending at every node: exact (small graphs only) or modulo `mod`. */
    internal long[] PathsEndingIterative(long mod, bool exact)
    {
        var paths = new long[N];
        foreach (int v in Order)
        {
            long sum = 0;
            foreach (int u in In[v])
            {
                sum += paths[u] + 1;
                if (exact)
                {
                    if (sum > (long.MaxValue >> 2)) throw new InvalidOperationException("exact count overflow");
                }
                else
                {
                    sum %= mod;
                }
            }
            paths[v] = exact ? sum : sum % mod;
        }
        return paths;
    }

    /** Total number of non-empty paths, by exhaustive memoized recursion on in-edges. */
    internal long PathTotalBrute()
    {
        var memo = new long[N];
        var done = new bool[N];
        long total = 0;
        for (int v = 0; v < N; v++) total += CountEnding(v, memo, done);
        return total;
    }

    /** Longest chain in the whole graph, by exhaustive memoized recursion. */
    internal int LongestBrute()
    {
        var memo = new int[N];
        var done = new bool[N];
        int best = 0;
        for (int v = 0; v < N; v++)
        {
            int value = LongestEnding(v, memo, done);
            if (value > best) best = value;
        }
        return best;
    }

    /** Number of paths from s to t (includes the empty path when s == t), exhaustive recursion. */
    internal long PathsFromTo(int s, int t)
    {
        var memo = new long[N];
        var done = new bool[N];
        return CountFromTo(s, t, memo, done);
    }

    private long CountEnding(int v, long[] memo, bool[] done)
    {
        if (done[v]) return memo[v];
        done[v] = true;
        long sum = 0;
        foreach (int u in In[v]) sum += CountEnding(u, memo, done) + 1;
        memo[v] = sum;
        return sum;
    }

    private int LongestEnding(int v, int[] memo, bool[] done)
    {
        if (done[v]) return memo[v];
        done[v] = true;
        int best = 0;
        foreach (int u in In[v])
        {
            int value = LongestEnding(u, memo, done) + 1;
            if (value > best) best = value;
        }
        memo[v] = best;
        return best;
    }

    private long CountFromTo(int v, int t, long[] memo, bool[] done)
    {
        if (done[v]) return memo[v];
        done[v] = true;
        long total = v == t ? 1 : 0;
        foreach (int w in Out[v]) total += CountFromTo(w, t, memo, done);
        memo[v] = total;
        return total;
    }
}

internal sealed class CachingDp
{
    private readonly Dag dag;
    private readonly long mod;
    private readonly int[] chain;
    private readonly long[] paths;
    private readonly bool[] chainDone;
    private readonly bool[] pathsDone;

    internal int ChainComputations;
    internal int PathComputations;

    internal CachingDp(Dag dag, long mod)
    {
        this.dag = dag;
        this.mod = mod;
        chain = new int[dag.N];
        paths = new long[dag.N];
        chainDone = new bool[dag.N];
        pathsDone = new bool[dag.N];
    }

    internal int LongestEnding(int v)
    {
        if (chainDone[v]) return chain[v];
        ChainComputations++;
        int best = 0;
        foreach (int u in dag.In[v])
        {
            int value = LongestEnding(u) + 1;
            if (value > best) best = value;
        }
        chain[v] = best;
        chainDone[v] = true;
        return best;
    }

    internal long PathsEnding(int v)
    {
        if (pathsDone[v]) return paths[v];
        PathComputations++;
        long sum = 0;
        foreach (int u in dag.In[v]) sum = (sum + PathsEnding(u) + 1) % mod;
        paths[v] = sum;
        pathsDone[v] = true;
        return sum;
    }

    internal int[] AllChains()
    {
        var result = new int[dag.N];
        for (int v = 0; v < dag.N; v++) result[v] = LongestEnding(v);
        return result;
    }

    internal long[] AllPaths()
    {
        var result = new long[dag.N];
        for (int v = 0; v < dag.N; v++) result[v] = PathsEnding(v);
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Edges (i, j) with i < j drawn from a fixed LCG, keeping `densityPercent` of the pairs. */
    private static List<int[]> LcgEdges(int n, int seed, int densityPercent)
    {
        var edges = new List<int[]>();
        int state = seed;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                state = (state * 75 + 74) % 65537;
                if (state % 100 < densityPercent) edges.Add(new[] { i, j });
            }
        }
        return edges;
    }

    private static int MaxOf(int[] values)
    {
        int best = 0;
        foreach (int v in values)
        {
            if (v > best) best = v;
        }
        return best;
    }

    private static long SumOf(long[] values)
    {
        long total = 0;
        foreach (long v in values) total += v;
        return total;
    }

    internal static void Main()
    {
        const int n = 24;
        var edges = LcgEdges(n, 20035, 30);
        var dag = new Dag(n, edges);
        Check(dag.Acyclic, "LCG graph is acyclic");
        Check(edges.Count >= 60, "LCG graph is non-trivial (" + edges.Count + " edges)");
        Check(dag.Order.Length == n, "cached order covers every node");

        var seen = new bool[n];
        for (int i = 0; i < dag.Order.Length; i++)
        {
            Check(!seen[dag.Order[i]], "cached order has no repeats");
            seen[dag.Order[i]] = true;
        }
        var position = new int[n];
        for (int i = 0; i < dag.Order.Length; i++) position[dag.Order[i]] = i;
        foreach (int[] e in edges) Check(position[e[0]] < position[e[1]], "edge points forward in the cached order");

        var rebuilt = new Dag(n, edges);
        Check(string.Join(",", dag.Order) == string.Join(",", rebuilt.Order), "cached order is reproducible");
        Check(dag.SourceCount() >= 1, "a DAG has at least one source");

        int[] iterative = dag.LongestEndingIterative();
        int[] relaxation = dag.LongestEndingBellmanFord();
        for (int v = 0; v < n; v++)
        {
            Check(iterative[v] == relaxation[v], "Bellman-Ford longest matches the cached-order DP at " + v);
        }
        int longest = MaxOf(iterative);
        Check(longest >= 6, "the LCG DAG has a chain of at least 6 edges (got " + longest + ")");

        var cache = new CachingDp(dag, Dag.Mod);
        int[] memoized = cache.AllChains();
        for (int v = 0; v < n; v++) Check(memoized[v] == iterative[v], "memoized longest matches at " + v);
        Check(cache.ChainComputations == n, "memo fills one chain entry per node (" + cache.ChainComputations + ")");

        long[] memoPaths = cache.AllPaths();
        Check(cache.PathComputations == n, "memo fills one path entry per node (" + cache.PathComputations + ")");
        long[] cachedPaths = dag.PathsEndingIterative(Dag.Mod, false);
        for (int v = 0; v < n; v++)
        {
            Check(memoPaths[v] == cachedPaths[v], "memoized path count matches the cached order at " + v);
        }

        // Second full pass: every value must come from the cache.
        cache.AllChains();
        cache.AllPaths();
        Check(cache.ChainComputations == n && cache.PathComputations == n, "second pass reuses the cached tables");
        Check(SumOf(cachedPaths) > 0, "the LCG DAG contains non-empty paths");

        // Exact enumeration on a graph small enough to walk every path.
        var small = new Dag(9, LcgEdges(9, 777, 45));
        Check(small.Acyclic, "9-node DAG is acyclic");
        long[] exact = small.PathsEndingIterative(Dag.Mod, true);
        Check(SumOf(exact) == small.PathTotalBrute(), "exact DP equals exhaustive path enumeration");
        int[] smallChains = small.LongestEndingIterative();
        Check(MaxOf(smallChains) == small.LongestBrute(), "longest DP equals exhaustive longest chain");
        long[] smallMod = small.PathsEndingIterative(Dag.Mod, false);
        for (int v = 0; v < 9; v++) Check(smallMod[v] == exact[v] % Dag.Mod, "modular table consistent at " + v);

        // Fibonacci DAG: paths 0 -> 20 equal F(21).
        var fibEdges = new List<int[]>();
        const int fibN = 21;
        for (int i = 0; i + 1 < fibN; i++) fibEdges.Add(new[] { i, i + 1 });
        for (int i = 0; i + 2 < fibN; i++) fibEdges.Add(new[] { i, i + 2 });
        var fib = new Dag(fibN, fibEdges);
        Check(fib.PathsFromTo(0, 20) == 10946, "Fibonacci DAG 0->20 paths equal F(21) = 10946");
        Check(fib.LongestEndingIterative()[20] == 20, "Fibonacci DAG longest chain to node 20 is 20 edges");
        long[] fibTable = fib.PathsEndingIterative(Dag.Mod, true);
        Check(SumOf(fibTable) == fib.PathTotalBrute(), "Fibonacci DAG tables agree with brute force");

        // Hand-checked diamond: 0->1, 0->2, 1->2, 1->3, 2->3.
        var diamond = new Dag(4, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 }, new[] { 1, 3 }, new[] { 2, 3 },
        });
        long[] diamondPaths = diamond.PathsEndingIterative(Dag.Mod, true);
        Check(diamondPaths[3] == 6, "paths ending at node 3 of the diamond equal 6");
        Check(SumOf(diamondPaths) == 10, "the diamond has 10 non-empty paths");

        // Cyclic inputs: the cached order must disappear.
        var cyclicEdges = new List<int[]>(fibEdges) { new[] { 20, 0 } };
        var cyclic = new Dag(fibN, cyclicEdges);
        Check(!cyclic.Acyclic && cyclic.Order.Length == 0, "back edge 20->0 removes the cached order");
        Check(!new Dag(2, new List<int[]> { new[] { 0, 1 }, new[] { 1, 0 } }).Acyclic, "2-cycle detected");
        Check(!new Dag(3, new List<int[]> { new[] { 1, 1 } }).Acyclic, "self loop detected");

        // Degenerate boundaries.
        var trivial = new Dag(1, new List<int[]>());
        Check(trivial.Acyclic && trivial.Order.Length == 1, "single-node graph is acyclic");
        Check(trivial.PathsEndingIterative(Dag.Mod, true)[0] == 0, "single node has no non-empty paths");
        var isolated = new Dag(5, new List<int[]>());
        Check(isolated.SourceCount() == 5, "every isolated node is a source");
        Check(isolated.PathTotalBrute() == 0 && isolated.LongestBrute() == 0, "no edges means no paths");

        Console.WriteLine("PASS");
    }
}
