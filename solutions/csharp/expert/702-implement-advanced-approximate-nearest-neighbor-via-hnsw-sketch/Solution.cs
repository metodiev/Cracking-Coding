// Task 702 - Implement advanced approximate nearest neighbor via HNSW sketch.
// (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// HNSW (Malkov & Yashunin 2016) keeps a hierarchy of proximity graphs. Every point is inserted at a
// level drawn from a geometric distribution (level = floor(-ln(u) / ln(M))), so the top levels are
// sparse highways and level 0 holds all points. A query descends greedily from the entry point, one
// step per level, and then runs a best-first search with an ef-wide candidate list on level 0:
// a candidate is expanded while it is closer than the current worst result, which is the standard
// early-stop rule that bounds the number of distance evaluations. Insertion mirrors the query: the
// top levels are traversed greedily, then for each level from the new node's level down to 0 the
// efConstruction candidates are collected, the M closest of them become the new neighbourhood, and
// every back-link is pruned to the same cap so the relation stays symmetric.
//
// Documented simplification: neighbour selection uses the plain "M closest" rule instead of the
// paper's diversity heuristic, and no heuristic extensions are applied to the candidate list. That
// trades a little recall for a lot of clarity, which is why the file does not claim the paper's
// O(log n) search bound - it claims and measures recall against exact k-NN search.
//
// Invariants verified below: no self loops or duplicate links, every back-link is answered by the
// forward link (symmetry), degree caps of M at levels >= 1 and 2M at level 0, the entry point is
// the highest-level node, no returned neighbour can be closer than the exact nearest one (a top-k
// answer can never beat brute force), recall@1 and recall@5 against brute-force k-NN, a shrinking
// mean error as ef grows, and bit-identical results for two builds from the same deterministic
// splitmix64 stream.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /// <summary>Uniform double in [0,1) taken from the top 53 bits of the stream.</summary>
    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class HnswIndex
{
    private readonly int _dim;
    private readonly int _m;                    // neighbour cap on levels >= 1
    private readonly int _m0;                   // neighbour cap on level 0
    private readonly int _efConstruction;
    private readonly double _levelFactor;       // mL = 1 / ln(M)
    private readonly List<float[]> _points = new List<float[]>();
    private readonly List<List<int>[]> _links = new List<List<int>[]>();
    private readonly List<int> _levels = new List<int>();
    private ulong _state;
    private int _entry = -1;
    private int _topLevel = -1;

    internal long DistanceEvaluations;

    internal HnswIndex(int dim, int m, int efConstruction, ulong seed)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (m < 2) throw new ArgumentOutOfRangeException(nameof(m), "M must be at least 2");
        _dim = dim;
        _m = m;
        _m0 = 2 * m;
        _efConstruction = Math.Max(efConstruction, m);
        _levelFactor = 1.0 / Math.Log(m);
        _state = seed;
    }

    internal int Count => _points.Count;
    internal int TopLevel => _topLevel;
    internal int EntryPoint => _entry;
    internal int LevelOf(int id) => _levels[id];
    internal IReadOnlyList<int> Neighbours(int id, int layer) => _links[id][layer];
    internal int LayerCount(int id) => _links[id].Length;
    internal int CapAt(int layer) => layer == 0 ? _m0 : _m;

    private float Distance(float[] a, float[] b)
    {
        DistanceEvaluations++;
        float sum = 0f;
        for (int i = 0; i < _dim; i++) { float d = a[i] - b[i]; sum += d * d; }
        return sum;                              // squared L2 ranks the same way as L2
    }

    private int PickLevel()
    {
        double u = Rng.NextDouble(ref _state);
        if (u <= 0.0) u = 0.5;                   // the stream may return exactly 0
        int level = (int)Math.Floor(-Math.Log(u) * _levelFactor);
        return Math.Min(level, 32);
    }

    private static int Compare(float da, float db, int ia, int ib)
    {
        if (da != db) return da < db ? -1 : 1;
        return ia.CompareTo(ib);                 // stable, deterministic tie-break
    }

    internal int Insert(float[] v)
    {
        int id = _points.Count;
        _points.Add(v);
        int level = PickLevel();
        _levels.Add(level);
        var layers = new List<int>[level + 1];
        for (int l = 0; l <= level; l++) layers[l] = new List<int>();
        _links.Add(layers);

        if (_entry < 0) { _entry = id; _topLevel = level; return id; }

        int cur = _entry;
        for (int l = _topLevel; l > level; l--) cur = GreedyDescend(v, cur, l);
        for (int l = Math.Min(level, _topLevel); l >= 0; l--)
        {
            List<int> candidates = SearchLayer(v, cur, _efConstruction, l);
            int cap = CapAt(l);
            List<int> chosen = SelectClosest(v, candidates, cap);
            layers[l].AddRange(chosen);
            foreach (int nb in chosen)
            {
                _links[nb][l].Add(id);
                if (_links[nb][l].Count > cap) Prune(nb, l, cap);
            }
            if (candidates.Count > 0) cur = candidates[0];   // closest candidate seeds the next level
        }
        if (level > _topLevel) { _topLevel = level; _entry = id; }
        return id;
    }

    /// <summary>Best-first search for up to <paramref name="ef"/> nearest neighbours on one layer.</summary>
    private List<int> SearchLayer(float[] q, int entry, int ef, int layer)
    {
        var visited = new HashSet<int> { entry };
        var frontier = new List<int> { entry };
        var best = new List<int> { entry };
        var cache = new Dictionary<int, float> { [entry] = Distance(q, _points[entry]) };

        while (frontier.Count > 0)
        {
            int pick = 0;
            for (int i = 1; i < frontier.Count; i++)
                if (cache[frontier[i]] < cache[frontier[pick]]) pick = i;
            int c = frontier[pick];
            frontier.RemoveAt(pick);

            float worst = float.MaxValue;
            if (best.Count >= ef)
            {
                worst = float.MinValue;
                foreach (int b in best) if (cache[b] > worst) worst = cache[b];
                if (cache[c] > worst) break;     // nothing reachable can improve the result list
            }
            foreach (int nb in _links[c][layer])
            {
                if (!visited.Add(nb)) continue;
                float d = Distance(q, _points[nb]);
                cache[nb] = d;
                if (best.Count < ef || d < worst)
                {
                    frontier.Add(nb);
                    best.Add(nb);
                    if (best.Count > ef)
                    {
                        int drop = 0;
                        for (int i = 1; i < best.Count; i++) if (cache[best[i]] > cache[best[drop]]) drop = i;
                        best.RemoveAt(drop);
                    }
                }
            }
        }
        best.Sort((a, b) => Compare(cache[a], cache[b], a, b));
        return best;
    }

    private int GreedyDescend(float[] q, int entry, int layer)
    {
        int cur = entry;
        float curDist = Distance(q, _points[cur]);
        bool improved = true;
        while (improved)
        {
            improved = false;
            foreach (int nb in _links[cur][layer])
            {
                float d = Distance(q, _points[nb]);
                if (d < curDist) { curDist = d; cur = nb; improved = true; }
            }
        }
        return cur;
    }

    /// <summary>k nearest neighbours, exploring up to ef candidates on level 0.</summary>
    internal int[] Search(float[] q, int k, int ef)
    {
        if (_entry < 0 || k <= 0) return Array.Empty<int>();
        int cur = _entry;
        for (int l = _topLevel; l >= 1; l--) cur = GreedyDescend(q, cur, l);
        List<int> found = SearchLayer(q, cur, Math.Max(ef, k), 0);
        int take = Math.Min(k, found.Count);
        var result = new int[take];
        for (int i = 0; i < take; i++) result[i] = found[i];
        return result;
    }

    internal float DistanceTo(float[] q, int id) => Distance(q, _points[id]);

    private List<int> SelectClosest(float[] v, List<int> candidates, int cap)
    {
        var sorted = new List<int>(candidates);
        sorted.Sort((a, b) => Compare(Distance(v, _points[a]), Distance(v, _points[b]), a, b));
        if (sorted.Count > cap) sorted.RemoveRange(cap, sorted.Count - cap);
        return sorted;
    }

    /// <summary>Trims a node's layer list to the cap and drops the matching back-links.</summary>
    private void Prune(int node, int layer, int cap)
    {
        List<int> list = _links[node][layer];
        if (list.Count <= cap) return;
        list.Sort((a, b) => Compare(Distance(_points[node], _points[a]),
                                    Distance(_points[node], _points[b]), a, b));
        for (int i = cap; i < list.Count; i++) _links[list[i]][layer].Remove(node);
        list.RemoveRange(cap, list.Count - cap);
    }

    /// <summary>Returns a description of the first broken invariant, or null when the graph is consistent.</summary>
    internal string? InvariantViolation()
    {
        if (_entry < 0) return _points.Count == 0 ? null : "empty index still has points";
        for (int id = 0; id < _points.Count; id++)
        {
            if (_links[id].Length != _levels[id] + 1) return "node " + id + " has the wrong layer count";
            for (int l = 0; l < _links[id].Length; l++)
            {
                List<int> list = _links[id][l];
                if (list.Count > CapAt(l)) return "node " + id + " exceeds the degree cap on layer " + l;
                var seen = new HashSet<int>();
                foreach (int nb in list)
                {
                    if (nb == id) return "node " + id + " links to itself on layer " + l;
                    if (nb < 0 || nb >= _points.Count) return "node " + id + " links to an unknown id";
                    if (!seen.Add(nb)) return "node " + id + " has a duplicate link on layer " + l;
                    if (_levels[nb] < l) return "node " + id + " links to a node without layer " + l;
                    bool back = false;
                    foreach (int other in _links[nb][l]) if (other == id) { back = true; break; }
                    if (!back) return "link " + id + " -> " + nb + " on layer " + l + " has no back-link";
                }
            }
        }
        if (_levels[_entry] != _topLevel) return "the entry point is not at the top level";
        for (int id = 0; id < _points.Count; id++) if (_levels[id] > _topLevel) return "a node is above the top level";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static float SquaredDistance(float[] a, float[] b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++) { float d = a[i] - b[i]; sum += d * d; }
        return sum;
    }

    private static int[] BruteKnn(float[][] points, float[] q, int k)
    {
        var order = new List<int>();
        for (int i = 0; i < points.Length; i++) order.Add(i);
        order.Sort((a, b) =>
        {
            float diff = SquaredDistance(points[a], q) - SquaredDistance(points[b], q);
            if (diff != 0f) return diff < 0f ? -1 : 1;
            return a.CompareTo(b);
        });
        return order.GetRange(0, Math.Min(k, order.Count)).ToArray();
    }

    public static void Main()
    {
        // ---- degenerate indexes ---------------------------------------------------------------
        {
            var empty = new HnswIndex(4, 8, 32, 1UL);
            Check(empty.Count == 0 && empty.Search(new[] { 0f, 0f, 0f, 0f }, 3, 16).Length == 0,
                "searching an empty index returns nothing");
            Check(empty.InvariantViolation() == null, "the empty index is trivially consistent");

            var single = new HnswIndex(4, 8, 32, 1UL);
            single.Insert(new[] { 1f, 2f, 3f, 4f });
            int[] hit = single.Search(new[] { 1f, 2f, 3f, 4f }, 1, 16);
            Check(hit.Length == 1 && hit[0] == 0, "a single point is its own nearest neighbour");
            Check(single.DistanceTo(new[] { 1f, 2f, 3f, 4f }, hit[0]) == 0f, "the distance to itself is exactly zero");
            Check(single.InvariantViolation() == null, "one point keeps the graph consistent");
            Check(single.LevelOf(0) == single.TopLevel && single.EntryPoint == 0, "the only node is the entry point");
        }

        // ---- a deterministic point set and its exact answers ---------------------------------
        const int PointCount = 260, Dim = 6, QueryCount = 48;
        ulong state = 0x0DDBA11UL;
        var points = new float[PointCount][];
        for (int i = 0; i < PointCount; i++)
        {
            var p = new float[Dim];
            for (int d = 0; d < Dim; d++) p[d] = (float)Rng.NextDouble(ref state);
            points[i] = p;
        }
        var queries = new float[QueryCount][];
        for (int i = 0; i < QueryCount; i++)
        {
            var q = new float[Dim];
            for (int d = 0; d < Dim; d++) q[d] = (float)Rng.NextDouble(ref state);
            queries[i] = q;
        }

        var index = new HnswIndex(Dim, 8, 96, 0x5EEDUL);
        for (int i = 0; i < PointCount; i++) index.Insert(points[i]);

        string? violation = index.InvariantViolation();
        Check(violation == null, "the built graph satisfies every structural invariant: " + (violation ?? "ok"));
        Check(index.Count == PointCount, "every point was inserted");
        Check(index.TopLevel >= 1, "the hierarchy really has more than one level (top=" + index.TopLevel + ")");
        Check(violation == null && index.EntryPoint >= 0 && index.LevelOf(index.EntryPoint) == index.TopLevel,
            "the entry point sits at the top level");
        {
            int aboveZero = 0;
            for (int i = 0; i < PointCount; i++) if (index.LevelOf(i) > 0) aboveZero++;
            Check(aboveZero > 0 && aboveZero < PointCount, "only a subset of the points reaches the upper levels: "
                + aboveZero + "/" + PointCount);
        }

        // ---- recall against exact k-NN, and the "cannot beat brute force" bound ---------------
        double recall1 = Recall(index, points, queries, 1, 64);
        double recall5 = Recall(index, points, queries, 5, 64);
        Check(recall1 >= 0.95, "recall@1 with ef=64 is at least 0.95 (measured " + recall1.ToString("F3") + ")");
        Check(recall5 >= 0.90, "recall@5 with ef=64 is at least 0.90 (measured " + recall5.ToString("F3") + ")");

        bool neverBeatsExact = true;
        double meanErrorSmall = 0.0, meanErrorLarge = 0.0;
        foreach (float[] q in queries)
        {
            int[] exact = BruteKnn(points, q, 1);
            float exactDistance = SquaredDistance(points[exact[0]], q);
            foreach (int ef in new[] { 8, 64 })
            {
                int[] got = index.Search(q, 1, ef);
                Check(got.Length == 1, "a 1-NN query always returns one neighbour");
                float distance = index.DistanceTo(q, got[0]);
                if (distance < exactDistance - 1e-7f) neverBeatsExact = false;   // impossible: brute force is exact
                if (ef == 8) meanErrorSmall += distance - exactDistance;
                else meanErrorLarge += distance - exactDistance;
            }
        }
        Check(neverBeatsExact, "no returned neighbour is closer than the exact nearest neighbour");
        Check(meanErrorLarge <= meanErrorSmall + 1e-9,
            "a wider candidate list is never worse on average (" + meanErrorLarge.ToString("E2") + " <= "
            + meanErrorSmall.ToString("E2") + ")");

        // ---- a duplicate point is found at distance zero --------------------------------------
        {
            int copy = index.Insert((float[])points[0].Clone());
            int[] got = index.Search(points[0], 1, 32);
            Check(got.Length == 1 && index.DistanceTo(points[0], got[0]) == 0f,
                "an exact duplicate is returned at distance zero");
            Check(copy == index.Count - 1, "the duplicate got the next id");
            Check(index.InvariantViolation() == null, "inserting a duplicate keeps the graph consistent");
        }

        // ---- rebuild determinism --------------------------------------------------------------
        {
            var again = new HnswIndex(Dim, 8, 96, 0x5EEDUL);
            for (int i = 0; i < PointCount; i++) again.Insert(points[i]);
            again.Insert((float[])points[0].Clone());      // mirror the duplicate inserted into `index`
            bool identical = again.TopLevel == index.TopLevel;
            for (int i = 0; i < QueryCount && identical; i++)
            {
                int[] a = again.Search(queries[i], 3, 32);
                int[] b = index.Search(queries[i], 3, 32);
                if (a.Length != b.Length) identical = false;
                else for (int j = 0; j < a.Length && identical; j++) if (a[j] != b[j]) identical = false;
            }
            Check(identical, "two builds from the same seed answer every query identically");
        }

        int maxDegree = 0;
        for (int i = 0; i < index.Count; i++)
            for (int l = 0; l < index.LayerCount(i); l++)
                if (index.Neighbours(i, l).Count > maxDegree) maxDegree = index.Neighbours(i, l).Count;
        Check(maxDegree > 0, "the graph really has links");
        Console.WriteLine("points=" + PointCount + " dim=" + Dim + " topLevel=" + index.TopLevel
            + " recall@1=" + recall1.ToString("F3") + " recall@5=" + recall5.ToString("F3")
            + " distanceEvaluations=" + index.DistanceEvaluations + " maxObservedDegree=" + maxDegree);

        Console.WriteLine("PASS");
    }

    private static double Recall(HnswIndex index, float[][] points, float[][] queries, int k, int ef)
    {
        int found = 0;
        foreach (float[] q in queries)
        {
            int[] exact = BruteKnn(points, q, k);
            int[] got = index.Search(q, k, ef);
            foreach (int id in got) if (Array.IndexOf(exact, id) >= 0) found++;
        }
        return (double)found / (queries.Length * k);
    }
}
