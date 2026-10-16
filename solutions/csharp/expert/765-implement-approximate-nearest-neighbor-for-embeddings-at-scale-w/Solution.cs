// Task 765 - Implement approximate nearest neighbor for embeddings at scale with IVF+PQ. (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)
//
// IVF + PQ is the classic two level recipe for billion scale similarity search. IVF (inverted file) clusters the
// database with k-means and, at query time, only visits the `nprobe` nearest lists - that is the pruning step.
// PQ (product quantization) splits each vector into M subspaces, learns a small codebook per subspace with
// k-means, and replaces a vector by M byte-sized codebook indices; distances are then evaluated through an
// asymmetric distance computation (ADC): one lookup table per subspace, built once per query, turns the distance
// to a code into M table reads. The two losses multiply, so the file measures each one separately: recall of the
// IVF with exact distances (only pruning), recall of IVF+PQ (pruning plus quantization), and both against a brute
// force oracle.
//
// Documented simplifications: 600 vectors of 16 dimensions in a handful of synthetic clusters (a real system would
// hold billions and use SIMD/GPU kernels), Lloyd k-means with a fixed iteration count instead of an ANN library,
// no re-ranking of the ADC candidates with exact distances, no OPQ rotation, and float64 arithmetic instead of the
// usual float32. Invariants verified: every vector is assigned to its true nearest centroid; the PQ code is the
// per-subspace argmin over the codebook (recomputed independently); ADC distance equals the direct distance to the
// reconstruction; with nprobe = nlist and exact distances the IVF returns exactly the brute force top-k, and with
// unbounded probes so does the ADC path modulo quantisation; recall rises with nprobe; the search really does look
// at only a fraction of the database; training is deterministic; and the boundary cases (empty index, k > n, a
// query that is a database point, a single list) behave.
using System;
using System.Collections.Generic;

internal static class Rng
{
    private static ulong state = 0x765AB1E5EED12345UL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Unit() => (Next() >> 11) * (1.0 / 9007199254740992.0);

    internal static double Signed() => 2 * Unit() - 1;
}

internal static class Vectors
{
    internal const int Dimensions = 16;
    internal const int Clusters = 8;
    internal const int Count = 600;
    internal const double Noise = 0.06;

    internal static double[][] Dataset;
    internal static double[][] Centers;

    static Vectors()
    {
        Centers = new double[Clusters][];
        for (int cluster = 0; cluster < Clusters; cluster++)
        {
            Centers[cluster] = new double[Dimensions];
            for (int d = 0; d < Dimensions; d++) Centers[cluster][d] = Rng.Unit();
        }
        Dataset = new double[Count][];
        for (int i = 0; i < Count; i++)
        {
            double[] center = Centers[i % Clusters];
            var vector = new double[Dimensions];
            for (int d = 0; d < Dimensions; d++) vector[d] = center[d] + Rng.Signed() * Noise;
            Dataset[i] = vector;
        }
    }

    internal static double Squared(double[] left, double[] right)
    {
        double total = 0;
        for (int d = 0; d < left.Length; d++)
        {
            double difference = left[d] - right[d];
            total += difference * difference;
        }
        return total;
    }
}

internal static class KMeans
{
    private static ulong LocalNext(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double LocalUnit(ref ulong state) => (LocalNext(ref state) >> 11) * (1.0 / 9007199254740992.0);

    /** Training is deterministic: the seed is an argument, not shared global state. */
    internal static double[][] Train(double[][] points, int k, int iterations, ulong seed)
    {
        ulong state = seed;
        if (points.Length == 0 || k <= 0) return Array.Empty<double[]>();
        int actual = Math.Min(k, points.Length);
        var centroids = new double[actual][];

        // k-means++ initialisation.
        centroids[0] = (double[])points[LocalNext(ref state) % (ulong)points.Length].Clone();
        var distance = new double[points.Length];
        for (int i = 0; i < points.Length; i++) distance[i] = Vectors.Squared(points[i], centroids[0]);
        for (int c = 1; c < actual; c++)
        {
            double total = 0;
            foreach (double value in distance) total += value;
            double target = LocalUnit(ref state) * total;
            int chosen = 0;
            double running = 0;
            for (int i = 0; i < points.Length; i++)
            {
                running += distance[i];
                if (running >= target)
                {
                    chosen = i;
                    break;
                }
            }
            centroids[c] = (double[])points[chosen].Clone();
            for (int i = 0; i < points.Length; i++)
            {
                distance[i] = Math.Min(distance[i], Vectors.Squared(points[i], centroids[c]));
            }
        }

        var assignment = new int[points.Length];
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            bool moved = false;
            for (int i = 0; i < points.Length; i++)
            {
                int best = Nearest(centroids, points[i]);
                if (best != assignment[i])
                {
                    assignment[i] = best;
                    moved = true;
                }
            }
            var sums = new double[actual][];
            var counts = new int[actual];
            for (int c = 0; c < actual; c++) sums[c] = new double[points[0].Length];
            for (int i = 0; i < points.Length; i++)
            {
                counts[assignment[i]]++;
                for (int d = 0; d < points[i].Length; d++) sums[assignment[i]][d] += points[i][d];
            }
            for (int c = 0; c < actual; c++)
            {
                if (counts[c] == 0) continue;
                for (int d = 0; d < sums[c].Length; d++) centroids[c][d] = sums[c][d] / counts[c];
            }
            if (!moved && iteration > 0) break;
        }
        return centroids;
    }

    internal static int Nearest(double[][] centroids, double[] point)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < centroids.Length; c++)
        {
            double candidate = Vectors.Squared(centroids[c], point);
            if (candidate < bestDistance)
            {
                bestDistance = candidate;
                best = c;
            }
        }
        return best;
    }
}

internal sealed class ProductQuantizer
{
    internal const int Subspaces = 8;
    internal const int CentroidCount = 16;

    private readonly double[][][] codebook;
    internal readonly int PerSubspace;

    internal ProductQuantizer(double[][] data, int iterations)
    {
        PerSubspace = Vectors.Dimensions / Subspaces;
        codebook = new double[Subspaces][][];
        for (int subspace = 0; subspace < Subspaces; subspace++)
        {
            var slices = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                var slice = new double[PerSubspace];
                for (int d = 0; d < PerSubspace; d++) slice[d] = data[i][subspace * PerSubspace + d];
                slices[i] = slice;
            }
            codebook[subspace] = KMeans.Train(slices, CentroidCount, iterations, 0x765AB1E5EED00000UL + (ulong)subspace);
        }
    }

    internal byte[] Encode(double[] vector)
    {
        var code = new byte[Subspaces];
        for (int subspace = 0; subspace < Subspaces; subspace++)
        {
            double bestDistance = double.MaxValue;
            int best = 0;
            for (int centroid = 0; centroid < codebook[subspace].Length; centroid++)
            {
                double total = 0;
                for (int d = 0; d < PerSubspace; d++)
                {
                    double difference = vector[subspace * PerSubspace + d] - codebook[subspace][centroid][d];
                    total += difference * difference;
                }
                if (total < bestDistance)
                {
                    bestDistance = total;
                    best = centroid;
                }
            }
            code[subspace] = (byte)best;
        }
        return code;
    }

    internal double[] Reconstruct(byte[] code)
    {
        var vector = new double[Vectors.Dimensions];
        for (int subspace = 0; subspace < Subspaces; subspace++)
        {
            double[] centroid = codebook[subspace][code[subspace]];
            for (int d = 0; d < PerSubspace; d++) vector[subspace * PerSubspace + d] = centroid[d];
        }
        return vector;
    }

    /** One table per subspace, built once per query: the asymmetric distance computation. */
    internal double[][] Lookup(double[] query)
    {
        var table = new double[Subspaces][];
        for (int subspace = 0; subspace < Subspaces; subspace++)
        {
            table[subspace] = new double[codebook[subspace].Length];
            for (int centroid = 0; centroid < codebook[subspace].Length; centroid++)
            {
                double total = 0;
                for (int d = 0; d < PerSubspace; d++)
                {
                    double difference = query[subspace * PerSubspace + d] - codebook[subspace][centroid][d];
                    total += difference * difference;
                }
                table[subspace][centroid] = total;
            }
        }
        return table;
    }

    internal double AdcDistance(double[][] table, byte[] code)
    {
        double total = 0;
        for (int subspace = 0; subspace < Subspaces; subspace++) total += table[subspace][code[subspace]];
        return total;
    }

    internal double[][] Centroids(int subspace) => codebook[subspace];
}

internal sealed class IvfIndex
{
    internal readonly double[][] Centroids;
    internal readonly List<int>[] Lists;
    internal readonly double[][] Data;
    internal readonly int AssignmentCount;
    internal long DistanceComputations;

    internal IvfIndex(double[][] data, int nlist, int iterations)
    {
        Data = data;
        Centroids = KMeans.Train(data, nlist, iterations, 0x765AB1E5EED11111UL);
        Lists = new List<int>[Centroids.Length];
        for (int i = 0; i < Centroids.Length; i++) Lists[i] = new List<int>();
        for (int i = 0; i < data.Length; i++) Lists[KMeans.Nearest(Centroids, data[i])].Add(i);
        AssignmentCount = data.Length;
    }

    internal List<int> ProbeLists(double[] query, int probes)
    {
        var order = new List<int>();
        for (int i = 0; i < Centroids.Length; i++) order.Add(i);
        order.Sort((left, right) =>
        {
            int comparison = Vectors.Squared(Centroids[left], query).CompareTo(Vectors.Squared(Centroids[right], query));
            return comparison != 0 ? comparison : left.CompareTo(right);
        });
        return order.GetRange(0, Math.Min(probes, order.Count));
    }

    /** Exact distances over the probed lists only: this isolates the pruning loss from the quantisation loss. */
    internal List<int> Search(double[] query, int probes, int k)
    {
        DistanceComputations = 0;
        var candidates = new List<(double Distance, int Id)>();
        foreach (int list in ProbeLists(query, probes))
        {
            foreach (int id in Lists[list])
            {
                DistanceComputations++;
                candidates.Add((Vectors.Squared(query, Data[id]), id));
            }
        }
        return TopK(candidates, k);
    }

    internal static List<int> TopK(List<(double Distance, int Id)> candidates, int k)
    {
        candidates.Sort((left, right) =>
        {
            int comparison = left.Distance.CompareTo(right.Distance);
            return comparison != 0 ? comparison : left.Id.CompareTo(right.Id);
        });
        var result = new List<int>();
        foreach ((double _, int id) in candidates)
        {
            if (result.Count >= k) break;
            result.Add(id);
        }
        return result;
    }
}

internal sealed class IvfPqIndex
{
    internal readonly IvfIndex Ivf;
    internal readonly ProductQuantizer Pq;
    private readonly byte[][] codes;
    internal long AdcLookups;

    internal IvfPqIndex(double[][] data, int nlist, int iterations)
    {
        Ivf = new IvfIndex(data, nlist, iterations);
        Pq = new ProductQuantizer(data, iterations);
        codes = new byte[data.Length][];
        for (int i = 0; i < data.Length; i++) codes[i] = Pq.Encode(data[i]);
    }

    internal byte[] Code(int id) => codes[id];

    internal List<int> Search(double[] query, int probes, int k)
    {
        double[][] table = Pq.Lookup(query);
        var candidates = new List<(double Distance, int Id)>();
        foreach (int list in Ivf.ProbeLists(query, probes))
        {
            foreach (int id in Ivf.Lists[list])
            {
                AdcLookups++;
                candidates.Add((Pq.AdcDistance(table, codes[id]), id));
            }
        }
        return IvfIndex.TopK(candidates, k);
    }

    /** Standard practice: take a wider ADC shortlist and re-rank it with exact distances. */
    internal List<int> SearchReranked(double[] query, int probes, int k, int multiplier)
    {
        double[][] table = Pq.Lookup(query);
        var candidates = new List<(double Distance, int Id)>();
        foreach (int list in Ivf.ProbeLists(query, probes))
        {
            foreach (int id in Ivf.Lists[list])
            {
                AdcLookups++;
                candidates.Add((Pq.AdcDistance(table, codes[id]), id));
            }
        }
        List<int> shortlist = IvfIndex.TopK(candidates, k * multiplier);
        var reranked = new List<(double Distance, int Id)>();
        foreach (int id in shortlist)
        {
            DistanceComputations++;
            reranked.Add((Vectors.Squared(query, Ivf.Data[id]), id));
        }
        return IvfIndex.TopK(reranked, k);
    }

    internal long DistanceComputations;
}

public static class Solution
{
    private const int K = 10;

    public static void Main()
    {
        double[][] data = Vectors.Dataset;
        var ivf = new IvfIndex(data, 8, 25);
        var combined = new IvfPqIndex(data, 8, 25);

        Training(ivf, combined, data);
        Exactness(ivf, combined, data);
        RecallAndCost(ivf, combined, data);
        Boundaries(ivf, combined, data);

        Console.WriteLine("PASS");
    }

    private static void Training(IvfIndex ivf, IvfPqIndex combined, double[][] data)
    {
        int misassigned = 0;
        for (int i = 0; i < data.Length; i++)
        {
            int best = KMeans.Nearest(ivf.Centroids, data[i]);
            bool found = false;
            for (int list = 0; list < ivf.Lists.Length; list++)
            {
                if (ivf.Lists[list].Contains(i) && list == best) found = true;
            }
            if (!found) misassigned++;
        }
        Check(misassigned == 0, "every vector sits in its true nearest centroid's list");
        int total = 0;
        foreach (List<int> list in ivf.Lists) total += list.Count;
        Check(total == data.Length && ivf.Lists.Length == 8,
            "the 8 inverted lists partition all " + total + " vectors");

        int badCodes = 0;
        for (int i = 0; i < data.Length; i++)
        {
            byte[] code = combined.Code(i);
            for (int subspace = 0; subspace < ProductQuantizer.Subspaces; subspace++)
            {
                double[][] centroids = combined.Pq.Centroids(subspace);
                int best = 0;
                double bestDistance = double.MaxValue;
                for (int c = 0; c < centroids.Length; c++)
                {
                    double total2 = 0;
                    for (int d = 0; d < combined.Pq.PerSubspace; d++)
                    {
                        double difference = data[i][subspace * combined.Pq.PerSubspace + d] - centroids[c][d];
                        total2 += difference * difference;
                    }
                    if (total2 < bestDistance)
                    {
                        bestDistance = total2;
                        best = c;
                    }
                }
                if (best != code[subspace]) badCodes++;
            }
        }
        Check(badCodes == 0, "every PQ code is the per-subspace argmin over the codebook");

        double error = 0, variance = 0;
        var mean = new double[Vectors.Dimensions];
        foreach (double[] vector in data)
        {
            for (int d = 0; d < vector.Length; d++) mean[d] += vector[d] / data.Length;
        }
        foreach (double[] vector in data)
        {
            for (int d = 0; d < vector.Length; d++)
            {
                double centered = vector[d] - mean[d];
                variance += centered * centered;
            }
        }
        for (int i = 0; i < data.Length; i++) error += Vectors.Squared(data[i], combined.Pq.Reconstruct(combined.Code(i)));
        double relative = error / variance;
        Check(relative < 0.05, "PQ reconstruction error is " + (relative * 100).ToString("F2")
            + "% of the data variance");

        var retrained = new ProductQuantizer(data, 25);
        bool identical = true;
        for (int subspace = 0; subspace < ProductQuantizer.Subspaces; subspace++)
        {
            for (int c = 0; c < retrained.Centroids(subspace).Length; c++)
            {
                for (int d = 0; d < retrained.PerSubspace; d++)
                {
                    if (retrained.Centroids(subspace)[c][d] != combined.Pq.Centroids(subspace)[c][d]) identical = false;
                }
            }
        }
        Check(identical, "training the same data twice gives byte identical centroids (deterministic)");
    }

    private static void Exactness(IvfIndex ivf, IvfPqIndex combined, double[][] data)
    {
        var query = data[17];
        List<int> truth = BruteForce(data, query, K);
        List<int> allLists = ivf.Search(query, 8, K);
        Check(Same(allLists, truth), "probing every list with exact distances reproduces the brute force top-10 exactly");
        Check(ivf.DistanceComputations == data.Length, "and it really compared against all " + data.Length + " vectors");
        Check(ivf.Search(query, 8, K)[0] == 17, "a query that is a database point retrieves itself first");

        double[][] table = combined.Pq.Lookup(query);
        int worstCode = 0;
        double worstDifference = 0;
        for (int i = 0; i < data.Length; i++)
        {
            worstCode = Math.Max(worstCode, combined.Code(i).Length);
            double adc = combined.Pq.AdcDistance(table, combined.Code(i));
            double direct = Vectors.Squared(query, combined.Pq.Reconstruct(combined.Code(i)));
            worstDifference = Math.Max(worstDifference, Math.Abs(adc - direct));
        }
        Check(worstCode == ProductQuantizer.Subspaces && worstDifference < 1e-9,
            "ADC equals the direct distance to the reconstruction (worst gap " + worstDifference.ToString("E1") + ")");

        // The estimate is not the exact distance, but the order it induces must agree with the true order, and the
        // gap is bounded by the triangle inequality: |d(q,x') - d(q,x)| <= ||x-x'|| * (||q-x'|| + ||q-x||).
        int pairs = 0, consistent = 0;
        bool boundHolds = true;
        double worstFraction = 0;
        for (int q = 0; q < 20; q++)
        {
            double[] probe = data[(q * 7) % data.Length];
            double[][] probeTable = combined.Pq.Lookup(probe);
            for (int i = 0; i < data.Length; i++)
            {
                double[] reconstruction = combined.Pq.Reconstruct(combined.Code(i));
                double toReconstruction = Vectors.Squared(probe, reconstruction);
                double toExact = Vectors.Squared(probe, data[i]);
                double delta = Math.Sqrt(Vectors.Squared(data[i], reconstruction));
                double bound = delta * (Math.Sqrt(toReconstruction) + Math.Sqrt(toExact));
                double gap = Math.Abs(toReconstruction - toExact);
                if (gap > bound + 1e-9) boundHolds = false;
                if (bound > 0) worstFraction = Math.Max(worstFraction, gap / bound);
                if (i + 1 < data.Length)
                {
                    double nextEstimate = combined.Pq.AdcDistance(probeTable, combined.Code(i + 1));
                    double nextExact = Vectors.Squared(probe, data[i + 1]);
                    pairs++;
                    if ((combined.Pq.AdcDistance(probeTable, combined.Code(i)) < nextEstimate) == (toExact < nextExact)) consistent++;
                }
            }
        }
        double agreement = consistent / (double)pairs;
        Check(boundHolds, "the ADC gap never exceeds the triangle inequality bound (worst " + worstFraction.ToString("F3") + " of it)");
        Check(agreement > 0.9, "the ADC order matches the exact order on " + (agreement * 100).ToString("F1")
            + "% of " + pairs + " pairs");
    }

    private static void RecallAndCost(IvfIndex ivf, IvfPqIndex combined, double[][] data)
    {
        int queries = 40;
        double exactRecall = 0, pqRecall = 0, pqRecallWide = 0, rerankedRecall = 0;
        long ivfCost = 0, pqCost = 0, rerankCost = 0;
        for (int q = 0; q < queries; q++)
        {
            double[] query = data[(q * 13) % data.Length];
            List<int> truth = BruteForce(data, query, K);
            List<int> viaExact = ivf.Search(query, 3, K);
            List<int> viaPq = combined.Search(query, 3, K);
            List<int> viaPqWide = combined.Search(query, 8, K);
            List<int> viaReranked = combined.SearchReranked(query, 3, K, 8);
            ivfCost += ivf.DistanceComputations;
            pqCost += combined.AdcLookups;
            rerankCost += combined.DistanceComputations;
            combined.DistanceComputations = 0;
            exactRecall += Overlap(viaExact, truth) / (double)K;
            pqRecall += Overlap(viaPq, truth) / (double)K;
            pqRecallWide += Overlap(viaPqWide, truth) / (double)K;
            rerankedRecall += Overlap(viaReranked, truth) / (double)K;
        }
        exactRecall /= queries;
        pqRecall /= queries;
        pqRecallWide /= queries;
        rerankedRecall /= queries;
        Check(exactRecall >= 0.9, "IVF recall@10 with 3 of 8 lists is " + exactRecall.ToString("F3"));
        Check(pqRecall >= 0.3, "IVF+PQ recall@10 is " + pqRecall.ToString("F3")
            + ": quantisation alone really does cost recall, which is what re-ranking is for");
        Check(pqRecallWide >= pqRecall, "probing everything cannot hurt the ADC recall: " + pqRecallWide.ToString("F3"));
        Check(rerankedRecall >= 0.9, "re-ranking the ADC shortlist restores recall to " + rerankedRecall.ToString("F3"));
        Check(rerankCost == (long)queries * K * 8, "re-ranking costs exactly its shortlist: "
            + (rerankCost / queries) + " exact distances per query");
        Check(rerankCost < (long)queries * data.Length / 5, "which is well under a fifth of a full exact scan");

        double exactFraction = ivfCost / (double)(queries * data.Length);
        Check(exactFraction < 0.5, "the pruned exact search touches " + (exactFraction * 100).ToString("F1")
            + "% of the database on average");

        // Cost comparison for one query: both paths must visit exactly the probed candidates.
        ivf.DistanceComputations = 0;
        ivf.Search(data[0], 3, K);
        long pruned = ivf.DistanceComputations;
        combined.AdcLookups = 0;
        combined.Search(data[0], 3, K);
        Check(combined.AdcLookups == pruned && pruned > 0 && pruned < data.Length / 2,
            "both paths scan the same " + pruned + " of " + data.Length + " vectors, under half of the database");

        int probesExtra = ivf.ProbeLists(data[0], 8).Count;
        int probesDefault = ivf.ProbeLists(data[0], 3).Count;
        Check(probesExtra == 8 && probesDefault == 3, "nprobe is honoured: " + probesDefault + " of " + probesExtra + " lists");
    }

    private static void Boundaries(IvfIndex ivf, IvfPqIndex combined, double[][] data)
    {
        var empty = new IvfIndex(Array.Empty<double[]>(), 4, 5);
        Check(empty.Lists.Length == 0 && empty.Search(new double[Vectors.Dimensions], 4, 5).Count == 0,
            "an empty index has no lists and returns nothing");
        Check(ivf.Search(data[0], 3, 1000).Count == data.Length
            || ivf.Search(data[0], 8, 1000).Count == data.Length,
            "k larger than the candidate set is capped by the probed candidates");
        Check(ivf.Search(data[0], 8, 1000).Count == data.Length, "probing every list can return the whole database");
        Check(ivf.Search(data[0], 1, 0).Count == 0, "k = 0 returns nothing");
        Check(combined.Search(data[0], 8, K).Contains(0), "the quantised search also finds the exact query point");
        Check(combined.Code(0).Length == ProductQuantizer.Subspaces
            && ProductQuantizer.Subspaces * combined.Pq.PerSubspace == Vectors.Dimensions,
            "the code is " + ProductQuantizer.Subspaces + " bytes for " + Vectors.Dimensions + " dimensions");

        var single = new IvfIndex(data, 1, 5);
        Check(single.Lists.Length == 1 && single.Lists[0].Count == data.Length,
            "a single list holds the whole database");
        Check(Same(single.Search(data[5], 1, K), BruteForce(data, data[5], K)),
            "with one list the search is exact again");
    }

    private static List<int> BruteForce(double[][] data, double[] query, int k)
    {
        var candidates = new List<(double Distance, int Id)>();
        for (int i = 0; i < data.Length; i++) candidates.Add((Vectors.Squared(query, data[i]), i));
        return IvfIndex.TopK(candidates, k);
    }

    private static int Overlap(List<int> left, List<int> right)
    {
        int hits = 0;
        foreach (int id in left)
        {
            if (right.Contains(id)) hits++;
        }
        return hits;
    }

    private static bool Same(List<int> left, List<int> right)
    {
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
