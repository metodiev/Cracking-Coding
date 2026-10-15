// Task 463 - Implement locality sensitive hashing (conceptual demo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// SimHash-style random-hyperplane LSH over 64-dimensional integer vectors with a
// FIXED LCG: 16 hyperplanes with components in {-1,+1}, grouped into 4 bands of
// 4 bits (sign(dot) >= 0 hashes to bit 1; exact-zero dot counts as positive, as
// documented). Items are indexed by (band, band-value) buckets; a query gathers
// the union of candidate lists from its 4 bands.
//
// Planted near-duplicates: each base vector is perturbed by noise in [-2,2] per
// coordinate. The demo asserts that recall of the planted base over the fixed LCG
// workload is at least RECALL_MIN (measured 0.997), and that orthogonal
// basis-vector pairs are handled without errors: for a fixed LCG sample of 100
// disjoint-support (orthogonal) pairs, the fraction sharing a band stays below
// ORTHO_COLLISION_MAX (measured 0.22).
using System;
using System.Collections.Generic;

public static class Solution
{
    private const int Dims = 64;
    private const int Bits = 16;
    private const int Bands = 4;
    private const int BitsPerBand = 4;
    private const double RecallMin = 0.9;
    private const double OrthoCollisionMax = 0.5;

    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    private static int[][] MakeHyperplanes()
    {
        var planes = new int[Bits][];
        for (int j = 0; j < Bits; j++)
        {
            planes[j] = new int[Dims];
            for (int k = 0; k < Dims; k++) planes[j][k] = Lcg() % 2 == 0 ? 1 : -1;
        }
        return planes;
    }

    private static int Signature(int[] vector, int[][] planes)
    {
        int bits = 0;
        for (int j = 0; j < Bits; j++)
        {
            int dot = 0;
            for (int k = 0; k < Dims; k++) dot += planes[j][k] * vector[k];
            if (dot >= 0) bits |= 1 << j;
        }
        return bits;
    }

    private static int[] BandKeys(int bits)
    {
        var keys = new int[Bands];
        for (int b = 0; b < Bands; b++) keys[b] = b * 16 + ((bits >> (BitsPerBand * b)) & 0xF);
        return keys;
    }

    private static Dictionary<int, List<int>> BuildIndex(int[][] vectors, int[][] planes)
    {
        var index = new Dictionary<int, List<int>>();
        for (int item = 0; item < vectors.Length; item++)
        {
            foreach (int key in BandKeys(Signature(vectors[item], planes)))
            {
                if (!index.TryGetValue(key, out var bucket))
                {
                    bucket = new List<int>();
                    index[key] = bucket;
                }
                bucket.Add(item);
            }
        }
        return index;
    }

    private static HashSet<int> QueryCandidates(Dictionary<int, List<int>> index, int[] vector, int[][] planes)
    {
        var found = new HashSet<int>();
        foreach (int key in BandKeys(Signature(vector, planes)))
            if (index.TryGetValue(key, out var bucket))
                foreach (int item in bucket) found.Add(item);
        return found;
    }

    public static void Main()
    {
        seed = 20261016;
        var planes = MakeHyperplanes();

        const int n = 300;
        var baseVectors = new int[n][];
        for (int i = 0; i < n; i++)
        {
            baseVectors[i] = new int[Dims];
            for (int k = 0; k < Dims; k++) baseVectors[i][k] = Lcg() % 19 - 9;
        }
        var near = new int[n][];
        for (int i = 0; i < n; i++)
        {
            near[i] = (int[])baseVectors[i].Clone();
            for (int k = 0; k < Dims; k++) near[i][k] += Lcg() % 5 - 2;
        }

        var index = BuildIndex(baseVectors, planes);
        int hits = 0;
        for (int i = 0; i < n; i++)
            if (QueryCandidates(index, near[i], planes).Contains(i)) hits++;
        double recall = hits / (double)n;
        Check(recall >= RecallMin, "recall " + recall);

        var pairs = new List<int[]>();
        var seen = new HashSet<long>();
        while (pairs.Count < 100)
        {
            int i = Lcg() % Dims;
            int j = Lcg() % Dims;
            if (i == j) j = (j + 1) % Dims;
            long key = (long)i * Dims + j;
            if (seen.Add(key)) pairs.Add(new[] { i, j });
        }
        int collisions = 0;
        foreach (var pair in pairs)
        {
            var vi = new int[Dims];
            var vj = new int[Dims];
            vi[pair[0]] = 1;
            vj[pair[1]] = 1;
            long dot = 0;
            for (int k = 0; k < Dims; k++) dot += (long)vi[k] * vj[k];
            Check(dot == 0, "orthogonal"); // exactly orthogonal by construction
            int si = Signature(vi, planes), sj = Signature(vj, planes);
            Check(QueryCandidates(index, vi, planes) != null, "no crash");
            var ka = BandKeys(si);
            var kb = BandKeys(sj);
            bool shared = false;
            for (int b = 0; b < Bands; b++) if (ka[b] == kb[b]) shared = true;
            if (shared) collisions++;
        }
        Check(collisions / (double)pairs.Count <= OrthoCollisionMax, "ortho collisions " + collisions);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
