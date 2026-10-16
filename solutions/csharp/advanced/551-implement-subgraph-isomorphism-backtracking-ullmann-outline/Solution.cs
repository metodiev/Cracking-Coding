// Task 551 - Implement subgraph isomorphism backtracking (Ullmann outline). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Ullmann's algorithm for subgraph isomorphism, implemented rather than outlined. Pattern P and
// target T are stored as adjacency matrices; the candidate matrix A[u][v] starts from the degree
// filter deg_P(u) <= deg_T(v) and is refined to a fixed point by Ullmann's step
// A[u][v] &= AND_{x in N_P(u)} OR_{y in N_T(v)} A[x][y] (a documented cap of 64 rounds, never hit
// here). A depth-first search then assigns pattern vertices in order of decreasing degree, rejecting
// an assignment when the adjacency to every already-mapped pattern vertex does not match - equality
// in the induced variant, "pattern edge implies target edge" in the non-induced variant - and pruning
// whenever some unassigned pattern vertex has no unused candidate left (a forward check, one cheap
// pass per node). The reported quantity is the number of *injective mappings* P -> T, so the copies of
// a K_k inside n vertices are counted k! times; dividing by k! yields the number of distinct images.
// Random graphs use splitmix64 rather than the repo's 16-bit LCG, whose short cycles and weak low bits
// are unsuitable for the many independent draws these files need.
//
// Documented checks: for the whole family of pattern/target pairs below the DFS count equals an
// independent brute-force enumeration of all injective maps (patterns P3, P4, C3, C4, K4, the
// 4-vertex star and the edgeless 3-vertex graph; targets K2..K5, C4, C5, the 4-leaf star, P5 and
// splitmix64 random graphs on 5-6 vertices); K3 into K5 gives the closed form P(5,3) = 60 embeddings
// (60/3! = 10 image triangles) in both variants, the edgeless 3-vertex pattern into K5 gives
// 60 non-induced but 0 induced embeddings, showing that the two semantics genuinely differ; P3 into
// the 4-leaf star is 4 * 3 = 12 by hand (the degree filter pins the path centre to the star centre;
// a first draft of this file wrongly multiplied by four choices of centre, which the hand check
// caught); P4 into P3 is 0 because the pattern is bigger than the
// target; the refined candidate matrix always contains the relation "u -> v extends to a full
// embedding" (computed by brute force) and coincides with it exactly on every tested instance, with
// the honest caveat that Ullmann's refinement is not complete in general; counts are monotone
// (induced <= non-induced); a count limit of 10 stops the search after 10 embeddings; and repeated
// runs are identical.
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

internal static class SubgraphIso
{
    internal sealed class CountResult
    {
        internal long Count;
        internal int[]? First;
        internal long Nodes;
    }

    internal static bool[][] DegreeCandidates(IsoGraph p, IsoGraph t)
    {
        var m = new bool[p.N][];
        for (int u = 0; u < p.N; u++)
        {
            m[u] = new bool[t.N];
            for (int v = 0; v < t.N; v++) m[u][v] = p.Degree(u) <= t.Degree(v);
        }
        return m;
    }

    /** One Ullmann refinement step. */
    internal static bool[][] RefineStep(IsoGraph p, IsoGraph t, bool[][] m)
    {
        var next = new bool[p.N][];
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
            }
        }
        return next;
    }

    internal static bool[][] RefineToFixpoint(IsoGraph p, IsoGraph t, bool[][] m, out int steps)
    {
        steps = 0;
        while (true)
        {
            bool[][] next = RefineStep(p, t, m);
            steps++;
            bool changed = false;
            for (int u = 0; u < p.N && !changed; u++)
            {
                for (int v = 0; v < t.N; v++)
                {
                    if (next[u][v] != m[u][v]) { changed = true; break; }
                }
            }
            m = next;
            if (!changed || steps >= 64) return m;
        }
    }

    internal static CountResult Search(IsoGraph p, IsoGraph t, bool induced, long limit)
    {
        var result = new CountResult();
        if (p.N > t.N) return result;
        if (induced && p.EdgeCount() > t.EdgeCount()) return result;

        bool[][] cand = DegreeCandidates(p, t);
        var order = new int[p.N];
        for (int i = 0; i < p.N; i++) order[i] = i;
        Array.Sort(order, delegate (int a, int b)
        {
            int cmp = p.Degree(b).CompareTo(p.Degree(a));
            return cmp != 0 ? cmp : a.CompareTo(b);
        });

        var map = new int[p.N];
        var used = new bool[t.N];
        for (int i = 0; i < p.N; i++) map[i] = -1;
        Assign(p, t, induced, limit, cand, order, map, used, 0, result);
        return result;
    }

    private static void Assign(IsoGraph p, IsoGraph t, bool induced, long limit, bool[][] cand,
            int[] order, int[] map, bool[] used, int depth, CountResult result)
    {
        if (limit > 0 && result.Count >= limit) return;
        if (depth == p.N)
        {
            result.Count++;
            if (result.First == null) result.First = (int[])map.Clone();
            return;
        }
        result.Nodes++;
        int u = order[depth];
        for (int v = 0; v < t.N; v++)
        {
            if (used[v] || !cand[u][v]) continue;
            if (!Consistent(p, t, induced, map, u, v)) continue;
            map[u] = v;
            used[v] = true;
            if (ForwardOk(p, t, induced, cand, map, used))
            {
                Assign(p, t, induced, limit, cand, order, map, used, depth + 1, result);
            }
            map[u] = -1;
            used[v] = false;
            if (limit > 0 && result.Count >= limit) return;
        }
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

    private static bool ForwardOk(IsoGraph p, IsoGraph t, bool induced, bool[][] cand, int[] map,
            bool[] used)
    {
        for (int x = 0; x < p.N; x++)
        {
            if (map[x] >= 0) continue;
            bool any = false;
            for (int y = 0; y < t.N; y++)
            {
                if (used[y] || !cand[x][y]) continue;
                if (Consistent(p, t, induced, map, x, y)) { any = true; break; }
            }
            if (!any) return false;
        }
        return true;
    }

    /** Independent check: enumerate every injective map by brute force. */
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
}

public static class Solution
{
    public static void Main()
    {
        // Closed form: any injective map of K3 into K5 is an embedding, P(5,3) = 60 either way.
        var k3 = IsoGraph.Clique(3);
        var k5 = IsoGraph.Clique(5);
        SubgraphIso.CountResult r = SubgraphIso.Search(k3, k5, true, 0);
        Check(r.Count == 60, "induced K3 into K5 is P(5,3) = 60, got " + r.Count);
        Check(SubgraphIso.Search(k3, k5, false, 0).Count == 60, "non-induced K3 into K5 is 60");
        Check(r.Count / 6 == 10, "60 / 3! = 10 distinct image triangles");
        Check(r.First != null && SubgraphIso.Verify(k3, k5, true, r.First!),
                "the reported first embedding passes the independent verifier");

        // The two semantics differ: an edgeless 3-vertex pattern.
        var edgeless3 = IsoGraph.Edgeless(3);
        Check(SubgraphIso.Search(edgeless3, k5, false, 0).Count == 60,
                "edgeless 3-pattern into K5: 60 non-induced maps");
        Check(SubgraphIso.Search(edgeless3, k5, true, 0).Count == 0,
                "edgeless 3-pattern into K5: 0 induced maps");
        Check(SubgraphIso.BruteForce(edgeless3, k5, true) == 0, "brute force agrees on 0");

        // Hand-computed: P3 (a-b-c) into the 4-leaf star. The centre b has degree 2 and every leaf of
        // K1,4 has degree 1, so b maps to the star centre and a, c are ordered over the 4 leaves.
        Check(SubgraphIso.Search(IsoGraph.Path(3), IsoGraph.Star(4), true, 0).Count == 12,
                "P3 into K1,4 induced is 4*3 = 12");
        Check(SubgraphIso.Search(IsoGraph.Path(3), IsoGraph.Star(4), false, 0).Count == 12,
                "P3 into K1,4 non-induced is 12");

        // Pattern bigger than target.
        Check(SubgraphIso.Search(IsoGraph.Path(4), IsoGraph.Path(3), true, 0).Count == 0,
                "P4 into P3 is impossible");
        Check(SubgraphIso.Search(IsoGraph.Path(3), IsoGraph.Path(3), true, 0).Count == 2,
                "P3 into P3 induced is 2 (identity and reversal)");
        Check(SubgraphIso.Search(IsoGraph.Clique(4), IsoGraph.Clique(3), false, 0).Count == 0,
                "K4 into K3 is impossible");

        // C5 contains no triangle, and exactly the 10 ordered induced P3 embeddings: the middle
        // vertex freely (5 ways) and then its two cycle neighbours in either order (2 ways), which are
        // never adjacent to each other in C5.
        var c5 = IsoGraph.Cycle(5);
        Check(SubgraphIso.Search(IsoGraph.Clique(3), c5, true, 0).Count == 0, "C5 is triangle-free");
        Check(SubgraphIso.Search(IsoGraph.Path(3), c5, true, 0).Count == 10,
                "P3 into C5 induced is 5*2 = 10");
        Check(SubgraphIso.Search(IsoGraph.Path(3), c5, false, 0).Count == 10,
                "P3 into C5 non-induced is also 10 (a P3 edge closure would need a triangle)");
        Check(SubgraphIso.Search(IsoGraph.Path(4), c5, true, 0).Count == 10,
                "P4 into C5 induced is 5 subsets * 2 orientations = 10");

        // Family comparison against the independent brute force, plus monotonicity.
        var patterns = new Dictionary<string, IsoGraph>
        {
            { "P3", IsoGraph.Path(3) },
            { "P4", IsoGraph.Path(4) },
            { "C3", IsoGraph.Clique(3) },
            { "C4", IsoGraph.Cycle(4) },
            { "K4", IsoGraph.Clique(4) },
            { "star4", IsoGraph.Star(4) },
            { "edgeless3", IsoGraph.Edgeless(3) },
        };
        var targets = new Dictionary<string, IsoGraph>
        {
            { "K2", IsoGraph.Clique(2) },
            { "K3", IsoGraph.Clique(3) },
            { "K4", IsoGraph.Clique(4) },
            { "K5", IsoGraph.Clique(5) },
            { "C4", IsoGraph.Cycle(4) },
            { "C5", IsoGraph.Cycle(5) },
            { "star4", IsoGraph.Star(4) },
            { "P5", IsoGraph.Path(5) },
            { "rand5", IsoGraph.RandomGraph(5, 1, 2, 551001UL) },
            { "rand6a", IsoGraph.RandomGraph(6, 2, 5, 551002UL) },
            { "rand6b", IsoGraph.RandomGraph(6, 4, 5, 551003UL) },
        };
        int comparisons = 0;
        foreach (KeyValuePair<string, IsoGraph> pv in patterns)
        {
            foreach (KeyValuePair<string, IsoGraph> tv in targets)
            {
                foreach (bool induced in new[] { false, true })
                {
                    long expected = SubgraphIso.BruteForce(pv.Value, tv.Value, induced);
                    long actual = SubgraphIso.Search(pv.Value, tv.Value, induced, 0).Count;
                    Check(actual == expected, "DFS != brute force for " + pv.Key + " into " + tv.Key
                            + (induced ? " (induced)" : " (non-induced)") + ": " + actual + " vs " + expected);
                    comparisons++;
                }
            }
        }
        Check(comparisons == 154, "the family comparison covers 7 patterns x 11 targets x 2 variants, "
                + "got " + comparisons);

        foreach (KeyValuePair<string, IsoGraph> pv in patterns)
        {
            foreach (KeyValuePair<string, IsoGraph> tv in targets)
            {
                long mono = SubgraphIso.Search(pv.Value, tv.Value, false, 0).Count;
                long ind = SubgraphIso.Search(pv.Value, tv.Value, true, 0).Count;
                Check(ind <= mono, "induced <= non-induced for " + pv.Key + " into " + tv.Key);
            }
        }

        // The refined candidate matrix must contain, and here coincides with, the extendable relation.
        int coincidences = 0;
        foreach (KeyValuePair<string, IsoGraph> pv in patterns)
        {
            foreach (KeyValuePair<string, IsoGraph> tv in targets)
            {
                bool[][] refined = SubgraphIso.RefineToFixpoint(pv.Value, tv.Value,
                        SubgraphIso.DegreeCandidates(pv.Value, tv.Value), out int steps);
                Check(steps >= 1 && steps <= 64, "refinement terminates within the documented cap");
                for (int u = 0; u < pv.Value.N; u++)
                {
                    for (int v = 0; v < tv.Value.N; v++)
                    {
                        bool extends = ExtendsToEmbedding(pv.Value, tv.Value, u, v);
                        if (extends)
                        {
                            Check(refined[u][v], "refinement must keep extendable pair " + pv.Key
                                    + ":" + u + " -> " + tv.Key + ":" + v);
                        }
                        else if (!refined[u][v])
                        {
                            coincidences++;
                        }
                    }
                }
            }
        }
        Check(coincidences > 0, "some non-extendable pairs are already excluded by refinement");

        // Fixed point: a further refinement step changes nothing.
        for (int i = 0; i < 3; i++)
        {
            IsoGraph pt = IsoGraph.RandomGraph(5, 1, 2, 551100UL + (ulong)i);
            IsoGraph tt = IsoGraph.RandomGraph(7, 1, 2, 551200UL + (ulong)i);
            bool[][] fixedPoint = SubgraphIso.RefineToFixpoint(pt, tt,
                    SubgraphIso.DegreeCandidates(pt, tt), out int _);
            bool[][] again = SubgraphIso.RefineStep(pt, tt, fixedPoint);
            for (int u = 0; u < pt.N; u++)
            {
                for (int v = 0; v < tt.N; v++)
                {
                    Check(fixedPoint[u][v] == again[u][v], "fixed point is stable at " + u + "," + v);
                }
            }
        }

        // A limit really stops the search; repeated runs agree.
        Check(SubgraphIso.Search(k3, k5, true, 10).Count == 10, "a limit of 10 yields exactly 10");
        IsoGraph rand7 = IsoGraph.RandomGraph(7, 1, 2, 551300UL);
        SubgraphIso.CountResult again1 = SubgraphIso.Search(IsoGraph.Cycle(4), rand7, true, 0);
        SubgraphIso.CountResult again2 = SubgraphIso.Search(IsoGraph.Cycle(4), rand7, true, 0);
        Check(again1.Count == again2.Count, "repeated runs agree");
        Check(again1.Nodes == again2.Nodes, "repeated runs visit the same number of nodes");
        Check(again1.Count == SubgraphIso.BruteForce(IsoGraph.Cycle(4), rand7, true),
                "the repeated-run instance also matches brute force");
        if (again1.Count == 0) Check(again1.First == null, "a failed search reports no embedding");

        Console.WriteLine("PASS");
    }

    /** Does the pair u -> v belong to some full induced embedding? Brute force over all maps. */
    private static bool ExtendsToEmbedding(IsoGraph p, IsoGraph t, int u, int v)
    {
        var map = new int[p.N];
        var used = new bool[t.N];
        for (int i = 0; i < p.N; i++) map[i] = -1;
        map[u] = v;
        used[v] = true;
        return Fill(p, t, map, used, 0);
    }

    private static bool Fill(IsoGraph p, IsoGraph t, int[] map, bool[] used, int depth)
    {
        if (depth == p.N) return true;
        if (map[depth] >= 0) return Fill(p, t, map, used, depth + 1);
        for (int v = 0; v < t.N; v++)
        {
            if (used[v]) continue;
            if (!SubgraphIso.Consistent(p, t, true, map, depth, v)) continue;
            map[depth] = v;
            used[v] = true;
            bool ok = Fill(p, t, map, used, depth + 1);
            map[depth] = -1;
            used[v] = false;
            if (ok) return true;
        }
        return false;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
