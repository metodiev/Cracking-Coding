// Task 552 - Implement VF2 algorithm skeleton for subgraph iso. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// VF2 (Cordella, Foggia, Sansone, Vento) for subgraph isomorphism, implemented for real instead of
// left as a skeleton. The state is a partial injective mapping core_1/core_2; the unmapped pattern
// vertices split into T1 (adjacent to the mapped core) and the free set, and VF2 always expands a
// vertex of T1 when that set is non-empty, choosing the one whose candidate list is shortest - the
// look-ahead step the paper credits for the algorithm's low branching factor. A pair (u, v) is
// feasible when it passes the necessary-compatibility matrix (degree filter plus Ullmann refinement
// to a fixed point), when the adjacency of u to every mapped pattern vertex reproduces the adjacency
// of v to its image (equality for induced subgraph isomorphism, "pattern edge implies target edge"
// for the non-induced monomorphism variant), and when the sound look-ahead rules hold: every unmapped
// pattern neighbour of u must have an unused candidate neighbour of v, and - induced case only -
// every unmapped non-neighbour of u must have an unused candidate non-neighbour of v. Extra sound
// pruning for the induced variant keeps the target partner of a free pattern vertex in the target
// free set. Random targets come from splitmix64 (documented 64-bit generator) rather than the repo's
// 16-bit LCG, whose short cycles would make the instances degenerate.
//
// Documented checks: over 7 patterns x 10 targets x 2 variants (140 pair/variant combinations, using
// P3, P4, C3, C4, K4, the 4-leaf star and the edgeless 3-vertex graph against K2..K5, C4, C5, the
// 4-leaf star, P5 and splitmix64 random graphs) the VF2 count equals an independent brute-force
// enumeration of injective mappings; the known values are reproduced exactly (K3 into K5 = P(5,3) =
// 60 in both variants, C3 into C5 = 0, P3 into C5 = 10, P4 into C5 = 10, P3 into K1,4 = 12, P4 into
// P3 = 0); the edgeless 3-vertex pattern into K5 is 60 non-induced versus 0 induced, so the induced
// semantics is really enforced; every reported mapping passes an independent verifier, the verifier
// rejects a deliberately broken mapping, and a mapping is reported exactly when the count is positive;
// induced <= non-induced for all pairs; and on P4 into a random 9-vertex graph the search visits
// strictly fewer states than the P(9,4) = 3024 injective mappings a naive enumeration would consider
// (a measurement, not a claimed asymptotic bound), with repeated runs visiting the same number of
// states and still matching brute force.
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

internal sealed class IsoGraph
{
    internal readonly int N;
    internal readonly bool[,] Adj;

    internal IsoGraph(int n)
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

    internal static IsoGraph Edgeless(int n)
    {
        return new IsoGraph(n);
    }

    internal static IsoGraph Path(int k)
    {
        var g = new IsoGraph(k);
        for (int i = 0; i + 1 < k; i++) g.AddEdge(i, i + 1);
        return g;
    }

    internal static IsoGraph Cycle(int k)
    {
        var g = new IsoGraph(k);
        for (int i = 0; i < k; i++) g.AddEdge(i, (i + 1) % k);
        return g;
    }

    internal static IsoGraph Clique(int k)
    {
        var g = new IsoGraph(k);
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++) g.AddEdge(i, j);
        }
        return g;
    }

    internal static IsoGraph Star(int leaves)
    {
        var g = new IsoGraph(leaves + 1);
        for (int i = 1; i <= leaves; i++) g.AddEdge(0, i);
        return g;
    }

    /** Deterministic Erdos-Renyi graph on n vertices with edge probability numerator/denominator. */
    internal static IsoGraph RandomGraph(int n, int numerator, int denominator, ulong seed)
    {
        var g = new IsoGraph(n);
        ulong state = seed;
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (SplitMix64.Below(ref state, denominator) < numerator) g.AddEdge(u, v);
            }
        }
        return g;
    }
}

internal static class Vf2
{
    internal sealed class Result
    {
        internal long Count;
        internal long States;
        internal int[]? First;
    }

    internal static Result Enumerate(IsoGraph p, IsoGraph t, bool induced, long limit)
    {
        var result = new Result();
        var search = new Search(p, t, induced, limit, result);
        search.Run();
        return result;
    }

    internal static bool Verify(IsoGraph p, IsoGraph t, bool induced, int[] mapping)
    {
        if (mapping.Length != p.N) return false;
        var seen = new bool[t.N];
        for (int u = 0; u < p.N; u++)
        {
            int v = mapping[u];
            if (v < 0 || v >= t.N || seen[v]) return false;
            seen[v] = true;
        }
        for (int u = 0; u < p.N; u++)
        {
            for (int x = u + 1; x < p.N; x++)
            {
                bool pe = p.Adj[u, x];
                bool te = t.Adj[mapping[u], mapping[x]];
                if (induced ? pe != te : (pe && !te)) return false;
            }
        }
        return true;
    }

    internal static long BruteForce(IsoGraph p, IsoGraph t, bool induced)
    {
        if (p.N > t.N) return 0;
        if (induced && p.EdgeCount() > t.EdgeCount()) return 0;
        var map = new int[p.N];
        var used = new bool[t.N];
        for (int i = 0; i < p.N; i++) map[i] = -1;
        return Brute(p, t, induced, map, used, 0);
    }

    private static long Brute(IsoGraph p, IsoGraph t, bool induced, int[] map, bool[] used, int depth)
    {
        if (depth == p.N) return 1;
        long total = 0;
        for (int v = 0; v < t.N; v++)
        {
            if (used[v]) continue;
            if (!Consistent(p, t, induced, map, depth, v)) continue;
            map[depth] = v;
            used[v] = true;
            total += Brute(p, t, induced, map, used, depth + 1);
            map[depth] = -1;
            used[v] = false;
        }
        return total;
    }

    internal static bool Consistent(IsoGraph p, IsoGraph t, bool induced, int[] map, int u, int v)
    {
        for (int x = 0; x < p.N; x++)
        {
            int wx = map[x];
            if (wx < 0) continue;
            bool pe = p.Adj[u, x];
            bool te = t.Adj[v, wx];
            if (induced ? pe != te : (pe && !te)) return false;
        }
        return true;
    }

    private sealed class Search
    {
        private readonly IsoGraph p;
        private readonly IsoGraph t;
        private readonly bool induced;
        private readonly long limit;
        private readonly Result result;
        private readonly int[] core1;
        private readonly int[] core2;
        private readonly bool[][] cand;

        internal Search(IsoGraph p, IsoGraph t, bool induced, long limit, Result result)
        {
            this.p = p;
            this.t = t;
            this.induced = induced;
            this.limit = limit;
            this.result = result;
            core1 = new int[p.N];
            core2 = new int[t.N];
            for (int i = 0; i < p.N; i++) core1[i] = -1;
            for (int i = 0; i < t.N; i++) core2[i] = -1;
            cand = NecessaryCandidates(p, t);
        }

        internal void Run()
        {
            if (p.N > t.N) return;
            if (induced && p.EdgeCount() > t.EdgeCount()) return;
            Recurse(0);
        }

        private void Recurse(int mapped)
        {
            if (limit > 0 && result.Count >= limit) return;
            if (mapped == p.N)
            {
                result.Count++;
                if (result.First == null) result.First = (int[])core1.Clone();
                return;
            }
            result.States++;
            int u = ChoosePatternVertex(out List<int> candidates);
            if (u < 0) return;
            foreach (int v in candidates)
            {
                core1[u] = v;
                core2[v] = u;
                Recurse(mapped + 1);
                core1[u] = -1;
                core2[v] = -1;
                if (limit > 0 && result.Count >= limit) return;
            }
        }

        /** T1 rule: expand a neighbour of the core when one exists, else a free pattern vertex. */
        private int ChoosePatternVertex(out List<int> candidates)
        {
            bool t1NonEmpty = false;
            for (int u = 0; u < p.N; u++)
            {
                if (core1[u] < 0 && HasMappedNeighbour(p, core1, u)) { t1NonEmpty = true; break; }
            }
            int best = -1;
            List<int>? bestList = null;
            for (int u = 0; u < p.N; u++)
            {
                if (core1[u] >= 0) continue;
                if (HasMappedNeighbour(p, core1, u) != t1NonEmpty) continue;
                var list = new List<int>();
                for (int v = 0; v < t.N; v++)
                {
                    if (core2[v] < 0 && Feasible(u, v)) list.Add(v);
                }
                if (list.Count == 0) { candidates = list; return u; }
                if (bestList == null || list.Count < bestList.Count)
                {
                    best = u;
                    bestList = list;
                }
            }
            candidates = bestList ?? new List<int>();
            return best;
        }

        private bool Feasible(int u, int v)
        {
            if (!cand[u][v]) return false;
            for (int w = 0; w < p.N; w++)
            {
                int wv = core1[w];
                if (wv < 0) continue;
                bool pe = p.Adj[u, w];
                bool te = t.Adj[v, wv];
                if (induced ? pe != te : (pe && !te)) return false;
            }
            bool uInT1 = HasMappedNeighbour(p, core1, u);
            if (induced && !uInT1 && HasMappedNeighbour(t, core2, v)) return false;
            for (int x = 0; x < p.N; x++)
            {
                if (x == u || core1[x] >= 0) continue;
                if (p.Adj[u, x])
                {
                    bool ok = false;
                    for (int y = 0; y < t.N; y++)
                    {
                        if (y == v || core2[y] >= 0) continue;
                        if (t.Adj[v, y] && cand[x][y]) { ok = true; break; }
                    }
                    if (!ok) return false;
                }
                else if (induced)
                {
                    bool ok = false;
                    for (int y = 0; y < t.N; y++)
                    {
                        if (y == v || core2[y] >= 0) continue;
                        if (!t.Adj[v, y] && cand[x][y]) { ok = true; break; }
                    }
                    if (!ok) return false;
                }
            }
            return true;
        }
    }

    private static bool HasMappedNeighbour(IsoGraph g, int[] core, int u)
    {
        for (int x = 0; x < g.N; x++)
        {
            if (x != u && core[x] >= 0 && g.Adj[u, x]) return true;
        }
        return false;
    }

    /** Necessary compatibility: degree filter, then Ullmann refinement to a fixed point. */
    private static bool[][] NecessaryCandidates(IsoGraph p, IsoGraph t)
    {
        var m = new bool[p.N][];
        for (int u = 0; u < p.N; u++)
        {
            m[u] = new bool[t.N];
            for (int v = 0; v < t.N; v++) m[u][v] = p.Degree(u) <= t.Degree(v);
        }
        for (int round = 0; round < 64; round++)
        {
            var next = new bool[p.N][];
            bool changed = false;
            for (int u = 0; u < p.N; u++)
            {
                next[u] = new bool[t.N];
                for (int v = 0; v < t.N; v++)
                {
                    bool keep = m[u][v];
                    for (int x = 0; x < p.N && keep; x++)
                    {
                        if (!p.Adj[u, x]) continue;
                        bool found = false;
                        for (int y = 0; y < t.N; y++)
                        {
                            if (t.Adj[v, y] && m[x][y]) { found = true; break; }
                        }
                        if (!found) keep = false;
                    }
                    next[u][v] = keep;
                    if (keep != m[u][v]) changed = true;
                }
            }
            m = next;
            if (!changed) break;
        }
        return m;
    }
}

public static class Solution
{
    public static void Main()
    {
        // Known values first: the VF2 count must reproduce them in both variants.
        var triangle = IsoGraph.Clique(3);
        var k5 = IsoGraph.Clique(5);
        var c5 = IsoGraph.Cycle(5);
        var star4 = IsoGraph.Star(4);
        Vf2.Result r = Vf2.Enumerate(triangle, k5, true, 0);
        Check(r.Count == 60, "VF2 induced K3 into K5 is P(5,3) = 60, got " + r.Count);
        Check(r.First != null && Vf2.Verify(triangle, k5, true, r.First!),
                "the first VF2 mapping passes the independent verifier");
        Check(Vf2.Enumerate(triangle, c5, true, 0).Count == 0, "C5 has no induced triangle");
        Check(Vf2.Enumerate(triangle, c5, false, 0).Count == 0, "C5 has no non-induced triangle");
        Check(Vf2.Enumerate(IsoGraph.Path(3), c5, true, 0).Count == 10, "P3 into C5 is 10");
        Check(Vf2.Enumerate(IsoGraph.Path(3), c5, false, 0).Count == 10, "P3 into C5 non-induced is 10");
        Check(Vf2.Enumerate(IsoGraph.Path(4), c5, true, 0).Count == 10, "P4 into C5 is 10");
        Check(Vf2.Enumerate(IsoGraph.Path(4), c5, false, 0).Count == 10, "P4 into C5 non-induced is 10");
        Check(Vf2.Enumerate(IsoGraph.Path(3), star4, true, 0).Count == 12, "P3 into K1,4 is 12");
        Check(Vf2.Enumerate(IsoGraph.Path(4), IsoGraph.Path(3), true, 0).Count == 0, "P4 into P3 is 0");

        // The induced and non-induced semantics must differ on the edgeless pattern.
        var edgeless3 = IsoGraph.Edgeless(3);
        Check(Vf2.Enumerate(edgeless3, k5, false, 0).Count == 60, "edgeless 3 into K5 non-induced is 60");
        Check(Vf2.Enumerate(edgeless3, k5, true, 0).Count == 0, "edgeless 3 into K5 induced is 0");

        // A broken mapping must be rejected by the verifier.
        var broken = new[] { 0, 1, 2 };
        Check(!Vf2.Verify(triangle, c5, true, broken), "verifier rejects 0,1,2 as a triangle in C5");
        Check(!Vf2.Verify(triangle, c5, false, broken), "verifier rejects it non-induced too");

        var patterns = new Dictionary<string, IsoGraph>
        {
            { "P3", IsoGraph.Path(3) },
            { "P4", IsoGraph.Path(4) },
            { "C3", IsoGraph.Clique(3) },
            { "C4", IsoGraph.Cycle(4) },
            { "K4", IsoGraph.Clique(4) },
            { "star4", star4 },
            { "edgeless3", edgeless3 },
        };
        var targets = new Dictionary<string, IsoGraph>
        {
            { "K2", IsoGraph.Clique(2) },
            { "K3", IsoGraph.Clique(3) },
            { "K4", IsoGraph.Clique(4) },
            { "K5", k5 },
            { "C4", IsoGraph.Cycle(4) },
            { "C5", c5 },
            { "star4", star4 },
            { "P5", IsoGraph.Path(5) },
            { "rand6a", IsoGraph.RandomGraph(6, 1, 2, 552001UL) },
            { "rand6b", IsoGraph.RandomGraph(6, 3, 5, 552002UL) },
        };

        int comparisons = 0;
        foreach (KeyValuePair<string, IsoGraph> pv in patterns)
        {
            foreach (KeyValuePair<string, IsoGraph> tv in targets)
            {
                foreach (bool induced in new[] { false, true })
                {
                    long expected = Vf2.BruteForce(pv.Value, tv.Value, induced);
                    Vf2.Result got = Vf2.Enumerate(pv.Value, tv.Value, induced, 0);
                    Check(got.Count == expected, "VF2 != brute force for " + pv.Key + " into " + tv.Key
                            + (induced ? " (induced)" : " (non-induced)") + ": " + got.Count
                            + " vs " + expected);
                    Check((got.First != null) == (got.Count > 0),
                            "a mapping is reported exactly when the count is positive for " + pv.Key
                            + " into " + tv.Key);
                    if (got.First != null)
                    {
                        Check(Vf2.Verify(pv.Value, tv.Value, induced, got.First),
                                "reported mapping is valid for " + pv.Key + " into " + tv.Key);
                    }
                    comparisons++;
                }
            }
        }
        Check(comparisons == 140, "the family comparison covers 7 patterns x 10 targets x 2 variants, "
                + "got " + comparisons);

        foreach (KeyValuePair<string, IsoGraph> pv in patterns)
        {
            foreach (KeyValuePair<string, IsoGraph> tv in targets)
            {
                long mono = Vf2.Enumerate(pv.Value, tv.Value, false, 0).Count;
                long ind = Vf2.Enumerate(pv.Value, tv.Value, true, 0).Count;
                Check(ind <= mono, "induced <= non-induced for " + pv.Key + " into " + tv.Key);
            }
        }

        // Pruning: fewer search states than the P(9,4) = 3024 injective maps a naive scan would try.
        IsoGraph big = IsoGraph.RandomGraph(9, 1, 2, 552100UL);
        Vf2.Result pruned = Vf2.Enumerate(IsoGraph.Path(4), big, true, 0);
        Check(pruned.States >= pruned.Count, "at least one state per found mapping");
        Check(pruned.States < 3024, "states visited (" + pruned.States + ") below P(9,4) = 3024");
        Vf2.Result prunedAgain = Vf2.Enumerate(IsoGraph.Path(4), big, true, 0);
        Check(prunedAgain.States == pruned.States && prunedAgain.Count == pruned.Count,
                "repeated runs are identical");
        Check(pruned.Count == Vf2.BruteForce(IsoGraph.Path(4), big, true),
                "the pruned search still agrees with brute force on 9 vertices");

        // The limit is honoured.
        Check(Vf2.Enumerate(triangle, k5, true, 10).Count == 10, "a limit of 10 yields 10");
        Check(Vf2.Enumerate(IsoGraph.Path(3), c5, false, 7).Count == 7, "a limit of 7 yields 7");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
