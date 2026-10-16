// Task 503 - Implement locality-sensitive hashing for cosine similarity (simhash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Two flavours of cosine-preserving LSH over a deterministically generated document corpus, plus
// the simhash signature used as a text near-duplicate detector.
//   * SimHash-64: every vocabulary token is mapped to a 64-bit fingerprint (FNV-1a followed by a
//     splitmix64 finalizer); for each token of weight w every set bit adds +w to that bit's
//     accumulator and every clear bit adds -w; the signature sets bit b when accumulator b > 0.
//     The result is invariant to the accumulation order and to scaling all weights by a positive
//     constant.
//   * Random-hyperplane LSH: k deterministic planes whose coordinates come from a
//     counter-indexed splitmix64 stream (the plain LCG's consecutive draws are correlated, which
//     would bias the sign test); bit b is sign(v . r_b). For two vectors with angle theta the
//     collision probability is exactly 1 - theta/pi, which the self-test measures by counting
//     sign agreements over 2000 planes (the packing into a 64-bit signature would alias planes
//     p and p + 64) for theta in {0, pi/4, pi/3, pi/2, 2pi/3, pi} with a documented tolerance of
//     0.05 (the sampling standard error is ~0.011).
//   * Banding: b bands of r rows, a pair is a candidate when some band matches. The measured
//     candidate rate over 200 independent plane sets is compared with 1 - (1 - p^r)^b
//     (p = 1 - theta/pi) within 0.06; theta = pi/3, r = 4, b = 16 gives a theoretical 0.970.
// Documented checks: identical documents collide (distance 0) and weight negation gives distance
// exactly 64 (for a document whose 64 accumulators are all non-zero); the Hamming metric
// satisfies d(x,x) = 0, symmetry and the triangle inequality on sampled triples; over 64 planted
// document pairs the Hamming distance tracks theta/pi (Pearson correlation >= 0.75) so same-topic
// pairs are separated from cross-topic pairs by a factor >= 1.5 in mean distance; banded
// retrieval of planted near-duplicate pairs reaches recall >= 0.9; and every signature is
// byte-identical across repeated runs.
using System;
using System.Collections.Generic;

internal static class Lsh
{
    internal class Lcg
    {
        private int state;

        internal Lcg(int seed) => state = seed;

        /** Uniform in [-1, 1]. */
        internal double Next()
        {
            state = (state * 75 + 74) % 65537;
            return (state % 20001 - 10000) / 10000.0;
        }

        /** Uniform in [0, 1). */
        internal double NextUnit()
        {
            state = (state * 75 + 74) % 65537;
            return (state % 65536) / 65536.0;
        }
    }

    internal static ulong Hash64(string token)
    {
        ulong hash = 14695981039346656037UL;                    // FNV-1a offset basis
        foreach (char c in token)
        {
            hash ^= c;
            hash *= 1099511628211UL;
        }
        return Mix64(hash);
    }

    /** splitmix64 finalizer: a bijective avalanche mix. */
    internal static ulong Mix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    /**
     * Counter-indexed uniform draw in [-1, 1). The plain LCG above is fine for magnitudes but
     * its consecutive draws are correlated, which would bias the hyperplane sign test, so plane
     * coordinates come from this avalanche-mixed stream instead.
     */
    internal static double Uniform(long seed, long index)
    {
        ulong h = Mix64((ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL);
        return 2.0 * ((h >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
    }

    internal static int PopCount(ulong value)
    {
        int count = 0;
        while (value != 0UL)
        {
            value &= value - 1UL;
            count += 1;
        }
        return count;
    }

    internal static int Hamming(ulong a, ulong b) => PopCount(a ^ b);

    /** The 64 signed accumulators; bit b of the signature is 1 iff sums[b] > 0. */
    internal static double[] SimHashSums(double[] weights, string prefix, bool reverse = false)
    {
        var sums = new double[64];
        for (int step = 0; step < weights.Length; step++)
        {
            int token = reverse ? weights.Length - 1 - step : step;
            if (weights[token] == 0.0) continue;
            ulong hash = Hash64(prefix + token);
            for (int bit = 0; bit < 64; bit++)
            {
                sums[bit] += ((hash >> bit) & 1UL) == 1UL ? weights[token] : -weights[token];
            }
        }
        return sums;
    }

    internal static ulong Signature(double[] sums)
    {
        ulong signature = 0UL;
        for (int bit = 0; bit < 64; bit++)
        {
            if (sums[bit] > 0.0) signature |= 1UL << bit;
        }
        return signature;
    }

    internal static double[][] RandomPlanes(int count, int dim, int seed)
    {
        var planes = new double[count][];
        long index = 0;
        for (int p = 0; p < count; p++)
        {
            var plane = new double[dim];
            for (int j = 0; j < dim; j++) plane[j] = Uniform(seed, index++);
            planes[p] = plane;
        }
        return planes;
    }

    /** Fraction of planes whose sign agrees for u and v: the LSH collision rate. */
    internal static double CollisionProbability(double[] u, double[] v, double[][] planes)
    {
        int collisions = 0;
        foreach (double[] plane in planes)
        {
            if ((Dot(u, plane) > 0.0) == (Dot(v, plane) > 0.0)) collisions += 1;
        }
        return collisions / (double)planes.Length;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    /** Signature of one vector under a plane set: bit p is sign(v . r_p). */
    internal static ulong HyperplaneSignature(double[] v, double[][] planes)
    {
        ulong signature = 0UL;
        for (int p = 0; p < planes.Length; p++)
        {
            if (Dot(v, planes[p]) > 0.0) signature |= 1UL << p;
        }
        return signature;
    }

    internal static double Angle(double[] u, double[] v)
    {
        double cosine = Dot(u, v) / (Norm(u) * Norm(v));
        cosine = Math.Max(-1.0, Math.Min(1.0, cosine));
        return Math.Acos(cosine);
    }

    internal static double Cosine(double[] u, double[] v) => Dot(u, v) / (Norm(u) * Norm(v));

    /** Banded candidates: b bands of r rows; a pair collides when any band signature matches. */
    internal static List<ulong> BandKeys(ulong signature, int bands, int rows)
    {
        ulong mask = rows >= 64 ? ulong.MaxValue : (1UL << rows) - 1UL;
        var keys = new List<ulong>();
        for (int band = 0; band < bands; band++)
        {
            keys.Add(((signature >> (band * rows)) & mask) ^ ((ulong)band << 48));
        }
        return keys;
    }

    internal static bool BandCollides(ulong a, ulong b, int bands, int rows)
    {
        List<ulong> left = BandKeys(a, bands, rows);
        List<ulong> right = BandKeys(b, bands, rows);
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] == right[i]) return true;
        }
        return false;
    }

    /** Planted topic model: 4 topics x 16 documents over a 256-token vocabulary. */
    internal static double[][] Topics(int topics, int docsPerTopic, int vocabulary, int seed,
            out int[] truth)
    {
        long counter = 0;
        var centers = new double[topics][];
        for (int t = 0; t < topics; t++)
        {
            centers[t] = new double[vocabulary];
            for (int j = 0; j < vocabulary; j++)
            {
                bool preferred = Uniform(seed, counter++) > 0.75;
                centers[t][j] = preferred ? 3.0 + Uniform(seed, counter++) : 0.0;
            }
        }
        var docs = new double[topics * docsPerTopic][];
        truth = new int[docs.Length];
        int index = 0;
        for (int t = 0; t < topics; t++)
        {
            for (int d = 0; d < docsPerTopic; d++)
            {
                var doc = new double[vocabulary];
                for (int j = 0; j < vocabulary; j++)
                {
                    double noise = 0.3 * (Uniform(seed, counter++) + 1.0);
                    doc[j] = centers[t][j] * (1.0 + 0.2 * Uniform(seed, counter++)) + noise;
                }
                docs[index] = doc;
                truth[index] = t;
                index += 1;
            }
        }
        return docs;
    }

    internal static double Pearson(double[] x, double[] y)
    {
        int n = x.Length;
        double meanX = 0.0;
        double meanY = 0.0;
        for (int i = 0; i < n; i++)
        {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= n;
        meanY /= n;
        double covariance = 0.0;
        double varianceX = 0.0;
        double varianceY = 0.0;
        for (int i = 0; i < n; i++)
        {
            covariance += (x[i] - meanX) * (y[i] - meanY);
            varianceX += (x[i] - meanX) * (x[i] - meanX);
            varianceY += (y[i] - meanY) * (y[i] - meanY);
        }
        return covariance / Math.Sqrt(varianceX * varianceY);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- signature invariants ---------------------------------------------------------------
        var seed = new Lsh.Lcg(4242);
        var weights = new double[64];
        for (int i = 0; i < weights.Length; i++) weights[i] = seed.NextUnit() < 0.5 ? 1.0 + seed.NextUnit() : 0.0;
        double[] sums = Lsh.SimHashSums(weights, "w");
        ulong signature = Lsh.Signature(sums);
        Check(Lsh.Signature(Lsh.SimHashSums(weights, "w", true)) == signature,
                "simhash is independent of the accumulation order");
        var scaled = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++) scaled[i] = weights[i] * 7.5;
        Check(Lsh.Signature(Lsh.SimHashSums(scaled, "w")) == signature,
                "simhash is invariant to positive scaling");
        var negated = new double[weights.Length];
        for (int i = 0; i < weights.Length; i++) negated[i] = -weights[i];
        double[] negatedSums = Lsh.SimHashSums(negated, "w");
        bool allNonZero = true;
        foreach (double value in sums) allNonZero &= Math.Abs(value) > 1e-12;
        Check(allNonZero, "the sample document has no tied accumulator");
        Check(Lsh.Hamming(signature, Lsh.Signature(negatedSums)) == 64,
                "negated weights flip every one of the 64 bits");

        // --- Hamming metric properties ------------------------------------------------------------
        var signatures = new ulong[32];
        for (int i = 0; i < signatures.Length; i++) signatures[i] = Lsh.Signature(Lsh.SimHashSums(
                Lsh.Topics(1, 1, 64, 900 + i, out _)[0], "w"));
        for (int i = 0; i < signatures.Length; i++)
        {
            Check(Lsh.Hamming(signatures[i], signatures[i]) == 0, "d(x, x) = 0");
            for (int j = i + 1; j < signatures.Length; j++)
            {
                Check(Lsh.Hamming(signatures[i], signatures[j]) == Lsh.Hamming(signatures[j], signatures[i]),
                        "Hamming distance is symmetric");
            }
        }
        for (int i = 0; i + 2 < signatures.Length; i += 3)
        {
            int a = i;
            int b = i + 1;
            int c = i + 2;
            Check(Lsh.Hamming(signatures[a], signatures[c])
                    <= Lsh.Hamming(signatures[a], signatures[b]) + Lsh.Hamming(signatures[b], signatures[c]),
                    "Hamming distance obeys the triangle inequality");
        }

        // --- random hyperplane collision probability equals 1 - theta/pi ---------------------------
        var planes = Lsh.RandomPlanes(2000, 8, 24680);
        var thetaCases = new[] { 0.0, Math.PI / 4.0, Math.PI / 3.0, Math.PI / 2.0, 2.0 * Math.PI / 3.0, Math.PI };
        foreach (double theta in thetaCases)
        {
            var u = new double[8];
            u[0] = 1.0;
            var v = new double[8];
            v[0] = Math.Cos(theta);
            v[1] = Math.Sin(theta);
            Check(Math.Abs(Lsh.Angle(u, v) - theta) <= 1e-12, "planted angle is exact");
            double measured = Lsh.CollisionProbability(u, v, planes);
            double expected = 1.0 - theta / Math.PI;
            Check(Math.Abs(measured - expected) <= 0.05,
                    "collision probability at theta = " + theta + " (measured " + measured + ")");
        }

        // --- banding: measured candidate rate equals 1 - (1 - p^r)^b -------------------------------
        const int bands = 16;
        const int rows = 4;
        double thetaBand = Math.PI / 3.0;
        double p = 1.0 - thetaBand / Math.PI;
        double expectedRate = 1.0 - Math.Pow(1.0 - Math.Pow(p, rows), bands);
        var bandU = new double[8];
        bandU[0] = 1.0;
        var bandV = new double[8];
        bandV[0] = Math.Cos(thetaBand);
        bandV[1] = Math.Sin(thetaBand);
        int collisions = 0;
        const int trials = 200;
        for (int trial = 0; trial < trials; trial++)
        {
            double[][] trialPlanes = Lsh.RandomPlanes(bands * rows, 8, 1000 + 97 * trial);
            ulong a = Lsh.HyperplaneSignature(bandU, trialPlanes);
            ulong b = Lsh.HyperplaneSignature(bandV, trialPlanes);
            if (Lsh.BandCollides(a, b, bands, rows)) collisions += 1;
        }
        double measuredRate = collisions / (double)trials;
        Check(expectedRate >= 0.95 && expectedRate <= 0.99, "documented theoretical candidate rate");
        Check(Math.Abs(measuredRate - expectedRate) <= 0.06,
                "banded candidate rate (measured " + measuredRate + " vs " + expectedRate + ")");
        ulong same = Lsh.HyperplaneSignature(bandU, Lsh.RandomPlanes(bands * rows, 8, 5));
        ulong sameAgain = Lsh.HyperplaneSignature(bandU, Lsh.RandomPlanes(bands * rows, 8, 5));
        Check(Lsh.BandCollides(same, sameAgain, bands, rows), "identical signatures always collide");

        // --- planted document corpus ---------------------------------------------------------------
        double[][] docs = Lsh.Topics(4, 16, 256, 20240607, out int[] truth);
        Check(docs.Length == 64 && truth.Length == 64, "corpus size");
        var docSums = new double[docs.Length][];
        var docSignatures = new ulong[docs.Length];
        for (int i = 0; i < docs.Length; i++)
        {
            docSums[i] = Lsh.SimHashSums(docs[i], "w");
            docSignatures[i] = Lsh.Signature(docSums[i]);
        }
        var sameTopic = new List<int[]>();
        var crossTopic = new List<int[]>();
        for (int topic = 0; topic < 4; topic++)
        {
            for (int d = 1; d <= 8; d++) sameTopic.Add(new[] { topic * 16, topic * 16 + d });
        }
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 4; j++) crossTopic.Add(new[] { i, 32 + j * 8 + i });
        }
        var angles = new double[sameTopic.Count + crossTopic.Count];
        var distances = new double[angles.Length];
        double sameMean = 0.0;
        double crossMean = 0.0;
        for (int i = 0; i < sameTopic.Count; i++)
        {
            int[] pair = sameTopic[i];
            angles[i] = Lsh.Angle(docs[pair[0]], docs[pair[1]]);
            distances[i] = Lsh.Hamming(docSignatures[pair[0]], docSignatures[pair[1]]);
            sameMean += distances[i];
        }
        for (int i = 0; i < crossTopic.Count; i++)
        {
            int[] pair = crossTopic[i];
            int slot = sameTopic.Count + i;
            angles[slot] = Lsh.Angle(docs[pair[0]], docs[pair[1]]);
            distances[slot] = Lsh.Hamming(docSignatures[pair[0]], docSignatures[pair[1]]);
            crossMean += distances[slot];
        }
        sameMean /= sameTopic.Count;
        crossMean /= crossTopic.Count;
        Check(sameMean * 1.5 <= crossMean, "same-topic pairs are much closer in Hamming space");
        double correlation = Lsh.Pearson(angles, distances);
        Check(correlation >= 0.75, "Hamming distance tracks the angle (r = " + correlation + ")");
        for (int i = 0; i < sameTopic.Count; i++)
        {
            Check(angles[i] < Math.PI / 3.0, "same-topic angles stay below 60 degrees");
        }
        for (int i = 0; i + 1 < sameTopic.Count; i += 1)
        {
            Check(distances[i] <= 40, "same-topic Hamming distance stays bounded");
        }

        // --- banded retrieval recall on the planted corpus --------------------------------------------
        int recalled = 0;
        int plantedPairs = 0;
        for (int topic = 0; topic < 4; topic++)
        {
            for (int a = 0; a < 16; a++)
            {
                for (int b = a + 1; b < 16 && b <= a + 2; b++)
                {
                    int left = topic * 16 + a;
                    int right = topic * 16 + b;
                    plantedPairs += 1;
                    if (Lsh.BandCollides(docSignatures[left], docSignatures[right], bands, rows)) recalled += 1;
                }
            }
        }
        double recall = recalled / (double)plantedPairs;
        Check(recalled >= 30 && recall >= 0.9, "banded recall on near-duplicates (recall " + recall + ")");

        // --- determinism ------------------------------------------------------------------------------
        var repeat = new ulong[docs.Length];
        for (int i = 0; i < docs.Length; i++)
        {
            repeat[i] = Lsh.Signature(Lsh.SimHashSums(docs[i], "w"));
        }
        for (int i = 0; i < docs.Length; i++)
        {
            Check(repeat[i] == docSignatures[i], "signature of document " + i + " is deterministic");
        }
        Check(Lsh.Cosine(docs[0], docs[0]) >= 1.0 - 1e-12, "cosine of a vector with itself is 1");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
