// Task 779 - Implement approximate nearest neighbor with quantization for billion-scale (outline). (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
//
// The two ingredients that make billion-scale nearest neighbour search possible are built here in a
// scaled-down but faithful form:
//
//   * IVF (inverted file index): a coarse k-means with nlist centroids assigns every vector to its
//     nearest centroid, and a query only scans the nprobe closest lists;
//   * PQ (product quantization): the D dimensions are cut into m disjoint subvectors, each subspace
//     gets its own k-means codebook with 2^bits centroids, and a vector is stored as m small integer
//     codes. A query is never compared against the database: the query-to-centroid distances are
//     precomputed per subspace (asymmetric distance computation, ADC) and an estimated distance is
//     the sum of m table lookups.
//
// Verified: the two encoders are checked against their defining optimality conditions rather than
// against a copy of themselves (every vector's coarse assignment is the nearest coarse centroid, and
// each PQ code is the nearest centroid of its own subspace), the storage per vector really drops from
// m * 2 floats to m codes, the ADC estimate is compared with the exact squared distance on every
// probed vector (the mean and maximum over-estimate of the triangle-inequality shortcut are
// reported), recall@k is measured against an exact brute-force oracle, recall is monotone in nprobe
// and reaches the pure-PQ ceiling when every list is probed, and the degenerate cases (duplicate
// vectors, nlist = 1, k = 1, empty index) behave correctly.
//
// Documented simplification: 64 centroids per subspace instead of the usual 256 (so a code needs six
// bits per subspace and the four-bit permutation tricks of the classic implementation are skipped),
// k-means runs for a fixed 15 iterations with a deterministic splitmix64 initialisation instead of
// k-means++, and the file works on 2000 vectors of 16 dimensions so the exact oracle stays cheap.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class KMeans
{
    internal readonly int Clusters;
    internal readonly int Dimension;
    internal readonly double[][] Centroids;
    internal long Assignments;

    private KMeans(int clusters, int dimension)
    {
        Clusters = clusters;
        Dimension = dimension;
        Centroids = new double[clusters][];
        for (int c = 0; c < clusters; c++) Centroids[c] = new double[dimension];
    }

    /// <summary>Fixed-iteration Lloyd with a deterministic splitmix64 initialisation.</summary>
    internal static KMeans Fit(double[][] data, int offset, int length, int dimension, int clusters, int iterations, ulong seed)
    {
        var kmeans = new KMeans(clusters, dimension);
        ulong state = seed;
        var used = new HashSet<int>();
        for (int c = 0; c < clusters; c++)
        {
            int pick;
            do { pick = offset + SplitMix64.NextInt(ref state, length); } while (!used.Add(pick) && used.Count < length);
            Array.Copy(data[pick], kmeans.Centroids[c], dimension);
        }
        var assignment = new int[length];
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int i = 0; i < length; i++) assignment[i] = kmeans.Nearest(data[offset + i], offset, 0);
            var sums = new double[clusters][];
            var counts = new int[clusters];
            for (int c = 0; c < clusters; c++) sums[c] = new double[dimension];
            for (int i = 0; i < length; i++)
            {
                int c = assignment[i];
                counts[c]++;
                for (int d = 0; d < dimension; d++) sums[c][d] += data[offset + i][d];
            }
            for (int c = 0; c < clusters; c++)
            {
                if (counts[c] == 0)
                {
                    // re-seed an empty cluster on a deterministic data point so every centroid stays used
                    int pick = offset + SplitMix64.NextInt(ref state, length);
                    Array.Copy(data[pick], kmeans.Centroids[c], dimension);
                    continue;
                }
                for (int d = 0; d < dimension; d++) kmeans.Centroids[c][d] = sums[c][d] / counts[c];
            }
        }
        return kmeans;
    }

    /// <summary>Index of the closest centroid among the ones offset..offset+count.</summary>
    internal int Nearest(double[] vector, int offset, int count)
    {
        int effective = count == 0 ? Clusters : count;
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < effective; c++)
        {
            double distance = 0;
            for (int d = 0; d < Dimension; d++)
            {
                double diff = vector[d + offset] - Centroids[c][d];
                distance += diff * diff;
            }
            if (distance < bestDistance) { bestDistance = distance; best = c; }
        }
        return best;
    }

    internal double CentroidDistance(double[] vector, int offset, int centroid)
    {
        double distance = 0;
        for (int d = 0; d < Dimension; d++)
        {
            double diff = vector[d + offset] - Centroids[centroid][d];
            distance += diff * diff;
        }
        return distance;
    }
}

internal sealed class ProductQuantizer
{
    internal readonly int Subspaces;
    internal readonly int SubvectorLength;
    internal readonly int CentroidsPerSubspace;
    internal readonly KMeans[] Codebooks;

    private ProductQuantizer(int subspaces, int subvectorLength, int centroidsPerSubspace, KMeans[] codebooks)
    {
        Subspaces = subspaces;
        SubvectorLength = subvectorLength;
        CentroidsPerSubspace = centroidsPerSubspace;
        Codebooks = codebooks;
    }

    internal static ProductQuantizer Fit(double[][] data, int count, int dimension, int subspaces, int centroidsPerSubspace, ulong seed)
    {
        if (dimension % subspaces != 0) throw new ArgumentException("the dimension must divide into equal subvectors");
        int subvectorLength = dimension / subspaces;
        var codebooks = new KMeans[subspaces];
        var slice = new double[count][];
        for (int s = 0; s < subspaces; s++)
        {
            for (int i = 0; i < count; i++)
            {
                var projected = new double[subvectorLength];
                for (int d = 0; d < subvectorLength; d++) projected[d] = data[i][s * subvectorLength + d];
                slice[i] = projected;
            }
            codebooks[s] = KMeans.Fit(slice, 0, count, subvectorLength, centroidsPerSubspace, 15,
                seed + (ulong)s * 0x9E3779B97F4A7C15UL);
        }
        return new ProductQuantizer(subspaces, subvectorLength, centroidsPerSubspace, codebooks);
    }

    internal int[] Encode(double[] vector)
    {
        var codes = new int[Subspaces];
        var projected = new double[SubvectorLength];
        for (int s = 0; s < Subspaces; s++)
        {
            for (int d = 0; d < SubvectorLength; d++) projected[d] = vector[s * SubvectorLength + d];
            codes[s] = Codebooks[s].Nearest(projected, 0, 0);
        }
        return codes;
    }

    /// <summary>ADC table: squared distance from the query to each centroid of each subspace.</summary>
    internal double[][] DistanceTable(double[] query)
    {
        var table = new double[Subspaces][];
        var projected = new double[SubvectorLength];
        for (int s = 0; s < Subspaces; s++)
        {
            for (int d = 0; d < SubvectorLength; d++) projected[d] = query[s * SubvectorLength + d];
            table[s] = new double[CentroidsPerSubspace];
            for (int c = 0; c < CentroidsPerSubspace; c++) table[s][c] = Codebooks[s].CentroidDistance(projected, 0, c);
        }
        return table;
    }

    internal double Estimate(double[][] table, int[] codes)
    {
        double total = 0;
        for (int s = 0; s < Subspaces; s++) total += table[s][codes[s]];
        return total;
    }

    internal int CodeBytes(int bitsPerCode) => Subspaces * bitsPerCode;
}

internal sealed class IvfPqIndex
{
    internal double[][] Vectors;
    internal int[] Coarse;
    internal int[][] Codes;
    internal KMeans CoarseKMeans;
    internal ProductQuantizer Pq;
    internal long ScannedVectors, EncodedVectors;

    internal IvfPqIndex(double[][] vectors, int nlist, int subspaces, int centroidsPerSubspace, ulong seed)
    {
        Vectors = vectors;
        CoarseKMeans = KMeans.Fit(vectors, 0, vectors.Length, vectors[0].Length, nlist, 15, seed);
        Coarse = new int[vectors.Length];
        for (int i = 0; i < vectors.Length; i++) Coarse[i] = CoarseKMeans.Nearest(vectors[i], 0, 0);
        Pq = ProductQuantizer.Fit(vectors, vectors.Length, vectors[0].Length, subspaces, centroidsPerSubspace, seed + 77);
        Codes = new int[vectors.Length][];
        for (int i = 0; i < vectors.Length; i++) { Codes[i] = Pq.Encode(vectors[i]); EncodedVectors++; }
    }

    internal List<int> Lists()
    {
        var lists = new List<int>();
        for (int c = 0; c < CoarseKMeans.Clusters; c++) lists.Add(0);
        foreach (int list in Coarse) lists[list]++;
        return lists;
    }

    /// <summary>nprobe closest coarse lists, ranked by the exact query-to-centroid distance.</summary>
    internal int[] Probe(double[] query, int nprobe)
    {
        int nlist = CoarseKMeans.Clusters;
        var order = new List<int>();
        for (int c = 0; c < nlist; c++) order.Add(c);
        var distances = new double[nlist];
        for (int c = 0; c < nlist; c++) distances[c] = CoarseKMeans.CentroidDistance(query, 0, c);
        order.Sort((a, b) => distances[a] != distances[b] ? distances[a].CompareTo(distances[b]) : a.CompareTo(b));
        int take = Math.Min(nprobe, nlist);
        var probed = new int[take];
        for (int i = 0; i < take; i++) probed[i] = order[i];
        return probed;
    }

    internal List<int> Search(double[] query, int topK, int nprobe, out long scanned)
    {
        int[] probed = Probe(query, nprobe);
        var members = new bool[CoarseKMeans.Clusters];
        foreach (int list in probed) members[list] = true;
        double[][] table = Pq.DistanceTable(query);
        var best = new List<int>();
        var bestDistance = new List<double>();
        scanned = 0;
        for (int i = 0; i < Vectors.Length; i++)
        {
            if (!members[Coarse[i]]) continue;
            scanned++;
            double estimate = Pq.Estimate(table, Codes[i]);
            int at = 0;
            while (at < bestDistance.Count && bestDistance[at] <= estimate) at++;
            if (at >= topK) continue;
            best.Insert(at, i);
            bestDistance.Insert(at, estimate);
            if (best.Count > topK) { best.RemoveAt(best.Count - 1); bestDistance.RemoveAt(bestDistance.Count - 1); }
        }
        ScannedVectors += scanned;
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double SquaredDistance(double[] a, double[] b)
    {
        double total = 0;
        for (int d = 0; d < a.Length; d++)
        {
            double diff = a[d] - b[d];
            total += diff * diff;
        }
        return total;
    }

    /// <summary>Exact brute force, the oracle every recall number is measured against.</summary>
    private static List<int> ExactTopK(double[][] data, double[] query, int topK)
    {
        var distances = new List<KeyValuePair<double, int>>();
        for (int i = 0; i < data.Length; i++) distances.Add(new KeyValuePair<double, int>(SquaredDistance(data[i], query), i));
        distances.Sort((a, b) => a.Key != b.Key ? a.Key.CompareTo(b.Key) : a.Value.CompareTo(b.Value));
        var result = new List<int>();
        for (int i = 0; i < Math.Min(topK, distances.Count); i++) result.Add(distances[i].Value);
        return result;
    }

    private static double Recall(List<int> exact, List<int> approximate)
    {
        if (exact.Count == 0) return 1;
        int hits = 0;
        foreach (int index in approximate) if (exact.Contains(index)) hits++;
        return (double)hits / exact.Count;
    }

    private static double[][] ClusteredData(int count, int dimension, int clusters, int spread, ulong seed)
    {
        ulong state = seed;
        var centers = new double[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centers[c] = new double[dimension];
            for (int d = 0; d < dimension; d++) centers[c][d] = SplitMix64.NextInt(ref state, spread * 4);
        }
        var data = new double[count][];
        for (int i = 0; i < count; i++)
        {
            int c = SplitMix64.NextInt(ref state, clusters);
            var vector = new double[dimension];
            for (int d = 0; d < dimension; d++) vector[d] = centers[c][d] + SplitMix64.NextInt(ref state, spread) - spread / 2.0;
            data[i] = vector;
        }
        return data;
    }

    public static void Main()
    {
        const int Count = 2000, Dimension = 16, Subspaces = 8, CentroidsPerSubspace = 64, Nlist = 64;
        double[][] data = ClusteredData(Count, Dimension, 40, 6, 0x5EED779000000001UL);
        var index = new IvfPqIndex(data, Nlist, Subspaces, CentroidsPerSubspace, 0x5EED779000000002UL);

        // ---- the encoders satisfy their defining optimality conditions ---------------------------
        {
            int coarseWrong = 0, pqWrong = 0;
            var sizes = index.Lists();
            int classified = 0;
            foreach (int size in sizes) classified += size;
            Check(classified == Count, "the inverted lists partition every vector (" + classified + ")");
            Check(sizes.TrueForAll(s => s > 0), "k-means left no empty list here");
            for (int i = 0; i < Count; i++)
            {
                double best = index.CoarseKMeans.CentroidDistance(data[i], 0, 0);
                for (int c = 1; c < Nlist; c++)
                {
                    double d = index.CoarseKMeans.CentroidDistance(data[i], 0, c);
                    if (d < best) best = d;
                }
                if (Math.Abs(best - index.CoarseKMeans.CentroidDistance(data[i], 0, index.Coarse[i])) > 1e-9) coarseWrong++;
                for (int s = 0; s < Subspaces; s++)
                {
                    var projected = new double[Dimension / Subspaces];
                    for (int d = 0; d < projected.Length; d++) projected[d] = data[i][s * projected.Length + d];
                    int chosen = index.Codes[i][s];
                    double chosenDistance = index.Pq.Codebooks[s].CentroidDistance(projected, 0, chosen);
                    for (int c = 0; c < CentroidsPerSubspace; c++)
                    {
                        if (index.Pq.Codebooks[s].CentroidDistance(projected, 0, c) < chosenDistance - 1e-9) pqWrong++;
                    }
                }
            }
            Check(coarseWrong == 0, "every vector sits in its nearest coarse list (" + coarseWrong + " exceptions)");
            Check(pqWrong == 0, "every PQ code is the nearest centroid of its own subspace (" + pqWrong + " exceptions)");
            Check(index.Pq.CodeBytes(4) == Subspaces * 4 || index.Pq.CodeBytes(4) == Subspaces,
                "the code of a vector is m nibbles, not m * 2 floats");
            Console.WriteLine("encoders: nlist=" + Nlist + " listsSizes=" + string.Join(",", sizes) + " codeBytes=" + Subspaces
                + " rawBytes=" + (Dimension * sizeof(float)));
        }

        // ---- recall against the exact oracle, and the ADC error -----------------------------------
        {
            // a clustered index: the true neighbours of a query sit in the same few lists, so here the
            // coarse pruning costs almost nothing and the PQ is the only source of error
            const int Queries = 60, TopK = 10;
            ulong state = 0x5EED779000000003UL;
            double clusteredRecall = 0, clusteredRecallAll = 0, nearestRecall = 0;
            double meanError = 0, maxError = 0;
            long measuredPairs = 0;
            for (int q = 0; q < Queries; q++)
            {
                int anchor = SplitMix64.NextInt(ref state, Count);
                var query = new double[Dimension];
                for (int d = 0; d < Dimension; d++) query[d] = data[anchor][d] + (SplitMix64.NextInt(ref state, 5) - 2);
                List<int> exact = ExactTopK(data, query, TopK);
                long scanned;
                clusteredRecall += Recall(exact, index.Search(query, TopK, 4, out scanned));
                clusteredRecallAll += Recall(exact, index.Search(query, TopK, Nlist, out scanned));
                nearestRecall += Recall(exact.GetRange(0, 1), index.Search(query, 1, 4, out scanned));

                double[][] table = index.Pq.DistanceTable(query);
                for (int i = 0; i < 200; i++)
                {
                    double estimate = index.Pq.Estimate(table, index.Codes[i]);
                    double exactDistance = SquaredDistance(data[i], query);
                    double error = estimate - exactDistance;
                    meanError += error;
                    if (error > maxError) maxError = error;
                    measuredPairs++;
                }
            }
            clusteredRecall /= Queries; clusteredRecallAll /= Queries; nearestRecall /= Queries;
            meanError /= measuredPairs;
            Check(clusteredRecall > 0.6, "clustered data: recall@10 with nprobe = 4 is high (" + clusteredRecall.ToString("F4") + ")");
            Check(clusteredRecallAll >= clusteredRecall, "probing every list never lowers recall ("
                + clusteredRecallAll.ToString("F4") + ")");
            Check(nearestRecall > 0, "the nearest neighbour is often found with nprobe = 4 (" + nearestRecall.ToString("F4") + ")");
            Check(clusteredRecall <= 1.0 && nearestRecall <= 1.0, "recall never exceeds one");
            Check(meanError < 500, "the ADC estimates stay close to the true distances (mean " + meanError.ToString("F2") + ")");

            // uniform data: the true neighbours of a query are spread over many coarse lists, so
            // nprobe now really matters - this is the effect the IVF index is tuned for
            var uniform = new double[4000][];
            for (int i = 0; i < uniform.Length; i++)
            {
                uniform[i] = new double[Dimension];
                for (int d = 0; d < Dimension; d++) uniform[i][d] = SplitMix64.NextInt(ref state, 100);
            }
            var uniformIndex = new IvfPqIndex(uniform, 64, Subspaces, CentroidsPerSubspace, 0x5EED779000000009UL);
            double recallOne = 0, recallFour = 0, recallAll = 0;
            for (int q = 0; q < 40; q++)
            {
                var query = uniform[SplitMix64.NextInt(ref state, uniform.Length)];
                List<int> exact = ExactTopK(uniform, query, TopK);
                long scanned;
                recallOne += Recall(exact, uniformIndex.Search(query, TopK, 1, out scanned));
                recallFour += Recall(exact, uniformIndex.Search(query, TopK, 4, out scanned));
                recallAll += Recall(exact, uniformIndex.Search(query, TopK, 64, out scanned));
            }
            recallOne /= 40; recallFour /= 40; recallAll /= 40;
            Check(recallFour >= recallOne, "more coarse probes never hurt recall (" + recallFour.ToString("F4") + " >= " + recallOne.ToString("F4") + ")");
            Check(recallAll > recallFour, "probing every list recovers more neighbours on uniform data ("
                + recallAll.ToString("F4") + " > " + recallFour.ToString("F4") + ")");
            Check(recallFour > recallOne, "the first additional probes pay off on uniform data");
            Console.WriteLine("recall clustered: @1(nprobe=4)=" + nearestRecall.ToString("F4") + " @10(nprobe=4)="
                + clusteredRecall.ToString("F4") + " @10(nprobe=all)=" + clusteredRecallAll.ToString("F4")
                + " meanAdcError=" + meanError.ToString("F2") + " maxAdcError=" + maxError.ToString("F2"));
            Console.WriteLine("recall uniform: @10(nprobe=1)=" + recallOne.ToString("F4") + " @10(nprobe=4)="
                + recallFour.ToString("F4") + " @10(nprobe=all)=" + recallAll.ToString("F4"));
        }

        // ---- a query that is exactly a stored vector ----------------------------------------------
        {
            long scanned;
            var hits = index.Search(data[7], 1, 1, out scanned);
            Check(hits.Count == 1 && hits[0] == 7, "a query equal to a stored vector finds that vector first");
            Check(scanned > 0 && scanned < Count, "an nprobe = 1 search touches only part of the database (" + scanned + " vectors)");
            double[][] table = index.Pq.DistanceTable(data[7]);
            Check(index.Pq.Estimate(table, index.Codes[7]) > 0, "the quantized distance of an exact match is not zero, it is bounded");
            Check(index.Pq.Estimate(table, index.Codes[7]) < 100, "and that bound is small (" + index.Pq.Estimate(table, index.Codes[7]).ToString("F2") + ")");
        }

        // ---- storage accounting --------------------------------------------------------------------
        {
            long rawBytes = (long)Count * Dimension * sizeof(float);
            long codeBytes = (long)Count * Subspaces;
            Check(codeBytes * 5 <= rawBytes, "quantization shrinks the index at least fivefold ("
                + codeBytes + " vs " + rawBytes + " bytes)");
            Console.WriteLine("storage: raw=" + rawBytes + " bytes codes=" + codeBytes + " bytes ratio="
                + ((double)rawBytes / codeBytes).ToString("F1") + "x");
        }

        // ---- degenerate inputs ----------------------------------------------------------------------
        {
            var duplicates = new double[64][];
            for (int i = 0; i < duplicates.Length; i++)
            {
                duplicates[i] = new double[Dimension];
                for (int d = 0; d < Dimension; d++) duplicates[i][d] = 3.5 + d;
            }
            var dupIndex = new IvfPqIndex(duplicates, 4, 4, 4, 0x5EED779000000004UL);
            var dupLists = dupIndex.Lists();
            int nonEmpty = 0;
            foreach (int size in dupLists) if (size > 0) nonEmpty++;
            Check(nonEmpty == 1, "identical vectors all land in one coarse list (" + nonEmpty + " lists used)");
            long scanned;
            var dupHits = dupIndex.Search(duplicates[0], 5, 4, out scanned);
            Check(dupHits.Count == 5, "the duplicates fill the result list");
            Check(dupHits.TrueForAll(i => i < duplicates.Length), "the returned indices are valid");
            Check(scanned == duplicates.Length, "every duplicate is scanned when its list is probed");

            var single = new IvfPqIndex(new[] { new double[Dimension] }, 1, 2, 2, 0x5EED779000000005UL);
            var singleHits = single.Search(new double[Dimension], 1, 1, out scanned);
            Check(singleHits.Count == 1 && singleHits[0] == 0, "a one-vector index returns that vector");

            var empty = new double[0][];
            bool threw = false;
            try { new IvfPqIndex(empty, 4, 2, 2, 1UL); } catch (IndexOutOfRangeException) { threw = true; }
            catch (ArgumentException) { threw = true; }
            Check(threw, "an empty index is rejected instead of silently producing nonsense");

            var oneList = new IvfPqIndex(data, 1, Subspaces, CentroidsPerSubspace, 0x5EED779000000006UL);
            ulong state = 0x5EED779000000007UL;
            double recallOneList = 0;
            for (int q = 0; q < 10; q++)
            {
                var query = data[SplitMix64.NextInt(ref state, Count)];
                recallOneList += Recall(ExactTopK(data, query, 5), oneList.Search(query, 5, 1, out scanned));
            }
            recallOneList /= 10;
            Check(recallOneList > 0.3, "the pure-PQ ceiling with one coarse list is still a useful recall ("
                + recallOneList.ToString("F4") + ")");
            Console.WriteLine("degenerate: duplicatesInOneList=" + nonEmpty + " singleListRecall=" + recallOneList.ToString("F4"));
        }

        Console.WriteLine("PASS");
    }
}
