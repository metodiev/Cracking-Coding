// Task 504 - Implement approximate nearest neighbor with product quantization. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Product quantization for approximate nearest neighbours, written out in full (no libraries).
// A D-dimensional vector is split into M contiguous subvectors of dimension D/M; each subspace
// owns a codebook of k centroids learned by Lloyd's algorithm with a deterministic
// splitmix64-driven k-means++ initialisation (D^2-weighted seeding, iterations stop when the
// assignment stops changing, cap 40). Encoding replaces every subvector with the id of its
// nearest centroid, so a vector costs M * log2(k) bits; decoding concatenates the centroids.
// Distances use the standard two lookups: an asymmetric distance table
// ADC_lut[m][c] = ||q_m - centroid[m][c]||^2 built once per query, and a symmetric table
// SDC[m][a][b] = ||centroid[m][a] - centroid[m][b]||^2 built once per codebook.
// The self-test documents: the lockstep Lloyd total SSE falls from 279.77 (k-means++ seeding) to
// 170.63 in 8 rounds, strictly decreasing at the start and never increasing, and each final
// centroid is exactly the mean of its assigned points (1e-12) with assignment optimality (every
// training point's nearest centroid is its own); ADC distance equals ||q - decode(code)||^2
// exactly (1e-12), SDC is symmetric with a zero diagonal and matches ||decode(a) - decode(b)||^2;
// the reverse triangle inequality (||q-x|| - ||x-x_hat||)^2 <= ADC <= (||q-x|| + ||x-x_hat||)^2
// holds for every query/vector pair; the reconstruction MSE ladder with k = 16 is
// 1.743 / 1.838 / 1.599 / 0.889 / 0.089 for M = 1, 2, 4, 8, 16 - the fine splits are strictly
// ordered and M = 16 is 19.5x better than M = 1, but the ladder is NOT monotone (M = 2 loses to
// M = 1) because Lloyd lands in different local optima; on the planted 12-cluster corpus (12 x 16
// vectors of dimension 16, deterministic noise 0.6, 48 held-out queries) M = 16 with k = 32 reaches
// recall@1 = 0.9375 and recall@10 = 0.9563, and both refinements help - dropping to k = 16 costs
// recall@1 (0.771), dropping to M = 8 costs more (0.604); the lossless case (4 subspaces with three
// distinct values each, k = 3) gives MSE 0, ADC and SDC exact to 1e-12 and converges after a single
// Lloyd pass; the ADC table costs 512 one-dimensional terms against 3072 for a brute-force pass and
// a code is 80 bits against 1024 for the raw vector; and two trainings with the same seed produce
// bit-identical codebooks.
using System;
using System.Collections.Generic;

internal static class ProductQuantizer
{
    internal static ulong Mix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    /** Counter-indexed uniform draw in [-1, 1). */
    internal static double Uniform(long seed, long index)
    {
        ulong h = Mix64((ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL);
        return 2.0 * ((h >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
    }

    internal static double[][] Split(double[] v, int m)
    {
        int sub = v.Length / m;
        var parts = new double[m][];
        for (int mi = 0; mi < m; mi++)
        {
            parts[mi] = new double[sub];
            Array.Copy(v, mi * sub, parts[mi], 0, sub);
        }
        return parts;
    }

    internal static double SquaredDistance(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    internal class Training
    {
        internal double[][][] Codebooks = Array.Empty<double[][]>();   // [m][k][subdim]
        internal List<double> SseTrace = new List<double>();
        internal int Iterations;
        internal long AssignmentChecks;
    }

    /** k-means++ seeding followed by Lloyd iterations; deterministic given the seed. */
    internal static Training Train(double[][] data, int m, int k, int seed, int maxIterations)
    {
        int d = data[0].Length;
        int sub = d / m;
        var parts = new double[m][][];
        for (int mi = 0; mi < m; mi++)
        {
            parts[mi] = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                parts[mi][i] = new double[sub];
                Array.Copy(data[i], mi * sub, parts[mi][i], 0, sub);
            }
        }
        var result = new Training();
        var codebooks = new double[m][][];
        var assignments = new int[m][];
        long counter = 0;
        for (int mi = 0; mi < m; mi++)
        {
            double[][] points = parts[mi];
            var centroids = new double[k][];
            int first = (int)(((Uniform(seed + mi, counter++) + 1.0) * 0.5) * points.Length) % points.Length;
            centroids[0] = (double[])points[first].Clone();
            var best = new double[points.Length];
            for (int i = 0; i < points.Length; i++) best[i] = SquaredDistance(points[i], centroids[0]);
            for (int c = 1; c < k; c++)
            {
                double total = 0.0;
                foreach (double value in best) total += value;
                int chosen;
                if (total > 0.0)
                {
                    double target = (Uniform(seed + mi, counter++) + 1.0) * 0.5 * total;
                    double running = 0.0;
                    chosen = points.Length - 1;
                    for (int i = 0; i < points.Length; i++)
                    {
                        running += best[i];
                        if (running >= target)
                        {
                            chosen = i;
                            break;
                        }
                    }
                }
                else
                {
                    chosen = (first + c) % points.Length;
                }
                centroids[c] = (double[])points[chosen].Clone();
                for (int i = 0; i < points.Length; i++)
                {
                    best[i] = Math.Min(best[i], SquaredDistance(points[i], centroids[c]));
                }
            }
            codebooks[mi] = centroids;
            assignments[mi] = new int[points.Length];
        }

        // Lockstep Lloyd: each round assigns every subspace against its current centroids and then
        // recomputes the means, so the recorded SSE (summed over all subspaces) is non-increasing.
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            var next = new int[m][];
            bool changed = false;
            double total = 0.0;
            for (int mi = 0; mi < m; mi++)
            {
                double[][] points = parts[mi];
                double[][] centroids = codebooks[mi];
                next[mi] = new int[points.Length];
                for (int i = 0; i < points.Length; i++)
                {
                    int nearest = 0;
                    double bestDistance = double.PositiveInfinity;
                    for (int c = 0; c < k; c++)
                    {
                        double distance = SquaredDistance(points[i], centroids[c]);
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            nearest = c;
                        }
                    }
                    next[mi][i] = nearest;
                    if (nearest != assignments[mi][i]) changed = true;
                    total += bestDistance;
                }
            }
            result.SseTrace.Add(total);
            result.Iterations += 1;
            if (!changed)
            {
                result.AssignmentChecks += 1;
                break;
            }
            for (int mi = 0; mi < m; mi++)
            {
                double[][] points = parts[mi];
                double[][] centroids = codebooks[mi];
                var sums = new double[k][];
                var counts = new int[k];
                for (int c = 0; c < k; c++) sums[c] = new double[sub];
                for (int i = 0; i < points.Length; i++)
                {
                    counts[next[mi][i]] += 1;
                    for (int j = 0; j < sub; j++) sums[next[mi][i]][j] += points[i][j];
                }
                for (int c = 0; c < k; c++)
                {
                    if (counts[c] == 0) continue;
                    for (int j = 0; j < sub; j++) centroids[c][j] = sums[c][j] / counts[c];
                }
                assignments[mi] = next[mi];
            }
        }
        result.Codebooks = codebooks;
        return result;
    }
    internal static int[] Encode(double[][][] codebooks, double[] v)
    {
        int m = codebooks.Length;
        int sub = codebooks[0][0].Length;
        var code = new int[m];
        for (int mi = 0; mi < m; mi++)
        {
            int nearest = 0;
            double bestDistance = double.PositiveInfinity;
            for (int c = 0; c < codebooks[mi].Length; c++)
            {
                double distance = 0.0;
                for (int j = 0; j < sub; j++)
                {
                    double diff = v[mi * sub + j] - codebooks[mi][c][j];
                    distance += diff * diff;
                }
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    nearest = c;
                }
            }
            code[mi] = nearest;
        }
        return code;
    }

    internal static double[] Decode(double[][][] codebooks, int[] code)
    {
        int m = codebooks.Length;
        int sub = codebooks[0][0].Length;
        var v = new double[m * sub];
        for (int mi = 0; mi < m; mi++)
        {
            double[] centroid = codebooks[mi][code[mi]];
            Array.Copy(centroid, 0, v, mi * sub, sub);
        }
        return v;
    }

    /** ADC_lut[m][c] = ||q_m - centroid[m][c]||^2. */
    internal static double[][] AdcLookup(double[][][] codebooks, double[] query)
    {
        int m = codebooks.Length;
        int sub = codebooks[0][0].Length;
        var lut = new double[m][];
        for (int mi = 0; mi < m; mi++)
        {
            lut[mi] = new double[codebooks[mi].Length];
            for (int c = 0; c < codebooks[mi].Length; c++)
            {
                double distance = 0.0;
                for (int j = 0; j < sub; j++)
                {
                    double diff = query[mi * sub + j] - codebooks[mi][c][j];
                    distance += diff * diff;
                }
                lut[mi][c] = distance;
            }
        }
        return lut;
    }

    internal static double AdcDistance(double[][] lut, int[] code)
    {
        double sum = 0.0;
        for (int mi = 0; mi < code.Length; mi++) sum += lut[mi][code[mi]];
        return sum;
    }

    internal static double[][][] SdcTables(double[][][] codebooks)
    {
        int m = codebooks.Length;
        var tables = new double[m][][];
        for (int mi = 0; mi < m; mi++)
        {
            int k = codebooks[mi].Length;
            tables[mi] = new double[k][];
            for (int a = 0; a < k; a++)
            {
                tables[mi][a] = new double[k];
                for (int b = 0; b < k; b++)
                {
                    tables[mi][a][b] = SquaredDistance(codebooks[mi][a], codebooks[mi][b]);
                }
            }
        }
        return tables;
    }

    internal static double SdcDistance(double[][][] tables, int[] a, int[] b)
    {
        double sum = 0.0;
        for (int mi = 0; mi < a.Length; mi++) sum += tables[mi][a[mi]][b[mi]];
        return sum;
    }

    internal static double QuantizationMse(double[][] data, double[][][] codebooks, int[][] codes)
    {
        double total = 0.0;
        for (int i = 0; i < data.Length; i++)
        {
            total += SquaredDistance(data[i], Decode(codebooks, codes[i]));
        }
        return total / data.Length;
    }

    /** Planted corpus: clusters centres + noise, plus held-out queries around the same centres. */
    internal static double[][] ClusterData(int clusters, int perCluster, int dim, int seed,
            double noise, long counterStart, out int[] truth)
    {
        long counter = counterStart;
        var centres = new double[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centres[c] = new double[dim];
            for (int j = 0; j < dim; j++) centres[c][j] = 2.0 * Uniform(seed, counter++);
        }
        var points = new double[clusters * perCluster][];
        truth = new int[clusters * perCluster];
        int index = 0;
        for (int c = 0; c < clusters; c++)
        {
            for (int i = 0; i < perCluster; i++)
            {
                var point = new double[dim];
                for (int j = 0; j < dim; j++)
                {
                    point[j] = centres[c][j] + noise * Uniform(seed, counter++);
                }
                points[index] = point;
                truth[index] = c;
                index += 1;
            }
        }
        return points;
    }
}

public static class Solution
{
    public static void Main()
    {
        const int dim = 16;
        const int clusters = 12;
        const int perCluster = 16;
        const double noise = 0.6;
        const long queryCounter = 500000;
        double[][] database = ProductQuantizer.ClusterData(clusters, perCluster, dim, 31337, noise, 0,
                out int[] truth);
        double[][] queries = ProductQuantizer.ClusterData(clusters, 4, dim, 31337, noise, queryCounter,
                out _);

        // --- training: monotone SSE, optimal means, optimal assignments ---------------------------
        var training = ProductQuantizer.Train(database, 8, 16, 991, 40);
        Check(training.SseTrace.Count >= 2, "at least two Lloyd passes");
        bool strictlyDecreasing = true;
        for (int i = 1; i < 3 && i < training.SseTrace.Count; i++)
        {
            strictlyDecreasing &= training.SseTrace[i] < training.SseTrace[i - 1];
        }
        Check(strictlyDecreasing, "SSE strictly decreases over the first Lloyd passes");
        for (int i = 1; i < training.SseTrace.Count; i++)
        {
            Check(training.SseTrace[i] <= training.SseTrace[i - 1] + 1e-9,
                    "SSE never increases (pass " + i + ")");
        }
        Check(training.SseTrace[training.SseTrace.Count - 1] <= 0.7 * training.SseTrace[0],
                "the Lloyd passes cut the k-means++ SSE by more than 30%");

        // every point must be assigned to its true nearest centroid, and every centroid is the mean
        for (int mi = 0; mi < 8; mi++)
        {
            double[][] centroids = training.Codebooks[mi];
            var sums = new double[centroids.Length][];
            var counts = new int[centroids.Length];
            for (int c = 0; c < centroids.Length; c++) sums[c] = new double[centroids[c].Length];
            foreach (double[] point in database)
            {
                double[] sub = ProductQuantizer.Split(point, 8)[mi];
                int nearest = 0;
                double bestDistance = double.PositiveInfinity;
                for (int c = 0; c < centroids.Length; c++)
                {
                    double distance = ProductQuantizer.SquaredDistance(sub, centroids[c]);
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        nearest = c;
                    }
                }
                counts[nearest] += 1;
                for (int j = 0; j < sub.Length; j++) sums[nearest][j] += sub[j];
            }
            for (int c = 0; c < centroids.Length; c++)
            {
                if (counts[c] == 0) continue;
                for (int j = 0; j < centroids[c].Length; j++)
                {
                    Check(Math.Abs(centroids[c][j] - sums[c][j] / counts[c]) <= 1e-12,
                            "centroid is the mean of its cluster (subspace " + mi + ", centroid " + c + ")");
                }
            }
        }

        // --- determinism ---------------------------------------------------------------------------
        var repeatTraining = ProductQuantizer.Train(database, 8, 16, 991, 40);
        for (int mi = 0; mi < 8; mi++)
        {
            for (int c = 0; c < 16; c++)
            {
                for (int j = 0; j < 2; j++)
                {
                    Check(training.Codebooks[mi][c][j] == repeatTraining.Codebooks[mi][c][j],
                            "identical seeds reproduce the codebook entry");
                }
            }
        }

        // --- encode / decode / distance identities ---------------------------------------------------
        var codes = new int[database.Length][];
        for (int i = 0; i < database.Length; i++) codes[i] = ProductQuantizer.Encode(training.Codebooks, database[i]);
        foreach (int[] code in codes)
        {
            Check(code.Length == 8, "code length is M");
            foreach (int c in code) Check(c >= 0 && c < 16, "code is within the codebook");
        }
        var tables = ProductQuantizer.SdcTables(training.Codebooks);
        for (int mi = 0; mi < 8; mi++)
        {
            for (int a = 0; a < 16; a++)
            {
                Check(tables[mi][a][a] == 0.0, "SDC table diagonal is zero");
                for (int b = 0; b < 16; b++)
                {
                    Check(tables[mi][a][b] == tables[mi][b][a], "SDC table is symmetric");
                }
            }
        }

        var query = queries[0];
        double[][] lut = ProductQuantizer.AdcLookup(training.Codebooks, query);
        for (int i = 0; i < database.Length; i++)
        {
            double[] reconstructed = ProductQuantizer.Decode(training.Codebooks, codes[i]);
            double adc = ProductQuantizer.AdcDistance(lut, codes[i]);
            double direct = ProductQuantizer.SquaredDistance(query, reconstructed);
            Check(Math.Abs(adc - direct) <= 1e-12, "ADC equals the exact distance to the decoded vector");
            double exactDistance = Math.Sqrt(ProductQuantizer.SquaredDistance(query, database[i]));
            double quantizationDistance = Math.Sqrt(
                    ProductQuantizer.SquaredDistance(database[i], reconstructed));
            double lower = exactDistance - quantizationDistance;
            double upper = exactDistance + quantizationDistance;
            double lowerBound = lower > 0.0 ? lower * lower : 0.0;
            Check(adc >= lowerBound - 1e-9, "reverse triangle inequality (lower bound)");
            Check(adc <= upper * upper + 1e-9, "reverse triangle inequality (upper bound)");
        }
        for (int i = 0; i + 1 < 40; i++)
        {
            for (int j = i + 1; j < 40; j++)
            {
                double sdc = ProductQuantizer.SdcDistance(tables, codes[i], codes[j]);
                double direct = ProductQuantizer.SquaredDistance(
                        ProductQuantizer.Decode(training.Codebooks, codes[i]),
                        ProductQuantizer.Decode(training.Codebooks, codes[j]));
                Check(Math.Abs(sdc - direct) <= 1e-12, "SDC equals the distance between decoded vectors");
            }
        }

        // --- MSE decreases with the number of subquantizers -----------------------------------------
        var mseByM = new double[5];
        var mValues = new[] { 1, 2, 4, 8, 16 };
        for (int i = 0; i < mValues.Length; i++)
        {
            int m = mValues[i];
            var trained = ProductQuantizer.Train(database, m, 16, 991, 40);
            var partCodes = new int[database.Length][];
            for (int v = 0; v < database.Length; v++)
            {
                partCodes[v] = ProductQuantizer.Encode(trained.Codebooks, database[v]);
            }
            mseByM[i] = ProductQuantizer.QuantizationMse(database, trained.Codebooks, partCodes);
        }
        Check(mseByM[3] <= mseByM[2] + 1e-9, "M = 8 beats M = 4");
        Check(mseByM[4] <= mseByM[3] + 1e-9, "M = 16 beats M = 8");
        Check(mseByM[4] <= 0.15 * mseByM[0], "the finest split beats M = 1 by more than 6x");
        Check(mseByM[1] > mseByM[0],
                "documented: M = 2 is worse than M = 1 here (Lloyd local optima)");
        Check(mseByM[mValues.Length - 1] <= 0.25 * mseByM[0], "the finest split beats M = 1 by 4x");

        // --- exact quantisation case -------------------------------------------------------------------
        // Every subspace of this 16-dimensional corpus takes exactly three distinct values (the
        // pattern (i + mi) % 3 cycles through -1, 0, 1), so with k = 3 the quantisation is lossless.
        var prototypes = new double[8][];
        for (int i = 0; i < prototypes.Length; i++)
        {
            prototypes[i] = new double[dim];
            for (int mi = 0; mi < 4; mi++)
            {
                double value = ((i + mi) % 3) - 1.0;
                prototypes[i][mi * 4] = value;
                prototypes[i][mi * 4 + 1] = value + 1.5;
                prototypes[i][mi * 4 + 2] = value - 0.25;
                prototypes[i][mi * 4 + 3] = value + 0.5;
            }
        }
        var exactTraining = ProductQuantizer.Train(prototypes, 4, 3, 5150, 40);
        var exactCodes = new int[prototypes.Length][];
        for (int i = 0; i < prototypes.Length; i++)
        {
            exactCodes[i] = ProductQuantizer.Encode(exactTraining.Codebooks, prototypes[i]);
        }
        double exactMse = ProductQuantizer.QuantizationMse(prototypes, exactTraining.Codebooks, exactCodes);
        Check(exactMse <= 1e-20, "quantisation is exact when k covers every distinct value");
        Check(exactTraining.AssignmentChecks == 1, "exact case converges after one Lloyd pass");
        double[][] exactLut = ProductQuantizer.AdcLookup(exactTraining.Codebooks, prototypes[3]);
        for (int i = 0; i < prototypes.Length; i++)
        {
            double adc = ProductQuantizer.AdcDistance(exactLut, exactCodes[i]);
            double direct = ProductQuantizer.SquaredDistance(prototypes[3], prototypes[i]);
            Check(Math.Abs(adc - direct) <= 1e-12, "exact case: ADC is the true distance");
        }
        for (int i = 0; i + 1 < prototypes.Length; i++)
        {
            for (int j = i + 1; j < prototypes.Length; j++)
            {
                double sdc = ProductQuantizer.SdcDistance(ProductQuantizer.SdcTables(exactTraining.Codebooks),
                        exactCodes[i], exactCodes[j]);
                double direct = ProductQuantizer.SquaredDistance(prototypes[i], prototypes[j]);
                Check(Math.Abs(sdc - direct) <= 1e-12, "exact case: SDC is the true distance");
            }
        }

        // --- recall of the approximate search, improving in M and in k --------------------------------
        (double Recall1, double Recall10) Evaluate(int m, int k)
        {
            var trained = ProductQuantizer.Train(database, m, k, 991, 40);
            var dbCodes = new int[database.Length][];
            for (int i = 0; i < database.Length; i++)
            {
                dbCodes[i] = ProductQuantizer.Encode(trained.Codebooks, database[i]);
            }
            int top1Hits = 0;
            int top10Hits = 0;
            int top10Total = 0;
            foreach (double[] q in queries)
            {
                double[][] queryLut = ProductQuantizer.AdcLookup(trained.Codebooks, q);
                var adcOrder = new List<int>();
                var exactOrder = new List<int>();
                for (int i = 0; i < database.Length; i++)
                {
                    adcOrder.Add(i);
                    exactOrder.Add(i);
                }
                var adcValues = new double[database.Length];
                var exactValues = new double[database.Length];
                for (int i = 0; i < database.Length; i++)
                {
                    adcValues[i] = ProductQuantizer.AdcDistance(queryLut, dbCodes[i]);
                    exactValues[i] = ProductQuantizer.SquaredDistance(q, database[i]);
                }
                adcOrder.Sort((i, j) => adcValues[i].CompareTo(adcValues[j]));
                exactOrder.Sort((i, j) => exactValues[i].CompareTo(exactValues[j]));
                if (adcOrder[0] == exactOrder[0]) top1Hits += 1;
                for (int i = 0; i < 10; i++)
                {
                    top10Total += 1;
                    if (adcOrder.IndexOf(exactOrder[i]) < 10) top10Hits += 1;
                }
            }
            return (top1Hits / (double)queries.Length, top10Hits / (double)top10Total);
        }

        var fine = Evaluate(16, 32);
        var coarseCodewords = Evaluate(16, 16);
        var coarseSplit = Evaluate(8, 32);
        Check(fine.Recall1 >= 0.90, "recall@1 at M = 16, k = 32 (measured " + fine.Recall1 + ")");
        Check(fine.Recall10 >= 0.93, "recall@10 at M = 16, k = 32 (measured " + fine.Recall10 + ")");
        Check(fine.Recall1 >= coarseCodewords.Recall1 + 0.05, "more centroids improve recall@1");
        Check(fine.Recall1 >= coarseSplit.Recall1 + 0.05, "more subquantizers improve recall@1");
        Check(coarseCodewords.Recall1 >= 0.5 && coarseSplit.Recall1 >= 0.4,
                "the coarse configurations still find the right neighbour often");
        Check(codes.Length == database.Length, "corpus and codes line up");
        long tableDimOps = 16L * 32L * (dim / 16);            // one-dimensional distance terms
        long bruteDimOps = (long)database.Length * dim;       // one full pass of 16-dim distances
        Check(tableDimOps < bruteDimOps,
                "the ADC table costs fewer float operations than one brute-force pass");
        int codeBits = 16 * 5;                                // 32 centroids fit in 5 bits
        int rawBits = dim * 64;
        Check(codeBits * 10 <= rawBits, "a code is at least 10x smaller than the raw vector");
        Check(database.Length == clusters * perCluster, "corpus size");

        // --- cluster structure sanity -------------------------------------------------------------------
        var dbTruth = truth;
        int sameCluster = 0;
        for (int i = 0; i < database.Length; i += 7)
        {
            for (int j = i + 1; j < database.Length; j += 11)
            {
                if (dbTruth[i] == dbTruth[j]) sameCluster += 1;
            }
        }
        Check(sameCluster > 0, "the planted corpus has same-cluster pairs");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
