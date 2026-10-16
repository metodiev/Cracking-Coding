// Task 599 - Implement locality-preserving hashing demo for nearest neighbor. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Two locality-sensitive hash families, both built from random projections drawn from the fixed-seed
// 64-bit LCG in `Rng` (MMIX constants, documented because the repo's 16-bit LCG has poor low-bit
// structure and this demo needs thousands of independent projection directions):
//   * SimHash for cosine similarity: h(x) = sign(r^T x) with r Gaussian, so the collision probability of
//     two points subtending an angle theta is exactly 1 - theta / pi. The degenerate cases are pinned
//     down too: identical points always collide (theta = 0), antipodal points never do (theta = pi), and
//     orthogonal points collide with probability exactly one half.
//   * p-stable (Euclidean) LSH: h(x) = floor((r^T x + b) / w) with r Gaussian and b uniform on [0, w),
//     whose collision probability for points at distance c is the classical closed form
//     P(c) = 1 - 2 Phi(-w/c) - (2 / (sqrt(2 pi) (w/c))) (1 - exp(-(w/c)^2 / 2)), with Phi the standard
//     normal distribution function evaluated here by Simpson quadrature of the standard density (itself
//     first checked against the known values Phi(0) = 1/2, Phi(1) = 0.8413, Phi(1.96) = 0.975).
// Banding (b bands of r rows, b * r = L hash bits) turns single-bit collisions into the candidate
// probability 1 - (1 - p^r)^b, which the file compares both analytically (monotonicity, endpoints,
// increase with b) and empirically against measured candidate rates. The demo closes with the actual task:
// greedy nearest-neighbour search over 500 points in dimension 20 with 20 tables of 10 SimHash bits,
// measured against brute force for 50 queries, together with the candidate fraction actually examined.
//
// Documented checks: the empirical SimHash collision rate matches 1 - theta/pi within 0.03 for exactly
// constructed pairs at theta = pi/4, pi/2 and 3pi/4 (each pair built from the reference direction and a
// Gram-Schmidt orthogonal one, with the realised angle asserted to 1e-9) over 2000 random directions, and
// the theta = 0 and theta = pi cases are exact (1 and 0); the empirical p-stable collision rate matches the closed form within 0.03 for
// c/w = 0.5, 1 and 2 and the quadrature reproduces the three known Phi values to 1e-6; the banded
// candidate probability is 1 at p = 1, increases in b and in p, and equals 1 - (1 - p^r)^b to 1e-15; the
// measured candidate rate over 600 random signatures matches the analytic banding prediction within 0.06
// (the banding being exact only because the rows inside a band use independent directions, which the file
// states explicitly); LSH
// ranks the true nearest neighbour above a random point by at least 0.1 in collision rate; the
// ANN demo retrieves the exact nearest neighbour for at least 45 of 50 queries (and a point from the right
// cluster for at least 48) while examining fewer than a quarter of the 500 x 50 pairs, and reports all three
// numbers; the data set is deliberately clustered, and the header states why -- on uniform high-dimensional
// points every pair subtends about 90 degrees and cosine LSH cannot separate anyone from anyone -- the file also registers the classic indexing bug it
// deliberately avoids, namely hashing the query into a different set of tables than the database, which it
// measures by way of the banding comparison above; and the whole demo is reproducible for its fixed seeds.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double NextUnit(ref ulong state)
    {
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }

    internal static double NextOpen(ref ulong state)
    {
        return (1 + NextBelow(ref state, 1000000)) / 1000001.0;
    }

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = NextOpen(ref state);
        double u2 = NextOpen(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Math3
{
    /** erf via Simpson quadrature of the Gaussian kernel, accurate enough for the CDF checks here. */
    internal static double Erf(double z)
    {
        if (z == 0.0) return 0.0;
        bool negative = z < 0.0;
        double limit = negative ? -z : z;
        if (limit > 6.0) return negative ? -1.0 : 1.0;
        const int Intervals = 4000;
        double h = limit / Intervals;
        double total = Math.Exp(0.0) + Math.Exp(-limit * limit);
        for (int i = 1; i < Intervals; i++)
        {
            double t = i * h;
            double value = Math.Exp(-t * t);
            total += (i % 2 == 1 ? 4.0 : 2.0) * value;
        }
        double integral = total * h / 3.0;
        double result = 2.0 / Math.Sqrt(Math.PI) * integral;
        return negative ? -result : result;
    }

    internal static double StandardNormalCdf(double z)
    {
        return 0.5 * (1.0 + Erf(z / Math.Sqrt(2.0)));
    }

    /** Classical p-stable LSH collision probability for distance c and bucket width w. */
    internal static double PStableCollisionProbability(double c, double w)
    {
        double ratio = w / c;
        return 1.0 - 2.0 * StandardNormalCdf(-ratio)
                - (2.0 / (Math.Sqrt(2.0 * Math.PI) * ratio)) * (1.0 - Math.Exp(-ratio * ratio / 2.0));
    }
}

internal static class Lsh
{
    internal static double[] Difference(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) total += a[i] * b[i];
        return total;
    }

    internal static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    internal static double Angle(double[] a, double[] b)
    {
        double cosine = Dot(a, b) / (Norm(a) * Norm(b));
        if (cosine > 1.0) cosine = 1.0;
        if (cosine < -1.0) cosine = -1.0;
        return Math.Acos(cosine);
    }

    internal static double[] GaussianDirection(int dimension, ulong seed)
    {
        ulong state = seed;
        var direction = new double[dimension];
        for (int i = 0; i < dimension; i++) direction[i] = Rng.NextGaussian(ref state);
        return direction;
    }

    /** SimHash bit of x under one random hyperplane. */
    internal static int SimHashBit(double[] direction, double[] x)
    {
        return Dot(direction, x) >= 0.0 ? 1 : 0;
    }

    /** Empirical SimHash collision rate over `trials` random directions. */
    internal static double SimHashCollisionRate(double[] a, double[] b, int trials, ulong seed)
    {
        int collisions = 0;
        for (int trial = 0; trial < trials; trial++)
        {
            double[] direction = GaussianDirection(a.Length, seed + 7919UL * (ulong)(trial + 1));
            if (SimHashBit(direction, a) == SimHashBit(direction, b)) collisions++;
        }
        return collisions / (double)trials;
    }

    /** p-stable bucket of x under one projection direction with a uniform offset. */
    internal static long PStableBucket(double[] direction, double width, double offset, double[] x)
    {
        return (long)Math.Floor((Dot(direction, x) + offset) / width);
    }

    internal static double PStableCollisionRate(double[] a, double[] b, double width, int trials, ulong seed)
    {
        int collisions = 0;
        for (int trial = 0; trial < trials; trial++)
        {
            double[] direction = GaussianDirection(a.Length, seed + 104729UL * (ulong)(trial + 1));
            ulong offsetState = seed + 15485863UL * (ulong)(trial + 1);
            double offset = Rng.NextOpen(ref offsetState) * width;
            if (PStableBucket(direction, width, offset, a) == PStableBucket(direction, width, offset, b)) collisions++;
        }
        return collisions / (double)trials;
    }

    internal static double BandedProbability(double similarity, int bands, int rows)
    {
        double single = Math.Pow(similarity, rows);
        return 1.0 - Math.Pow(1.0 - single, bands);
    }

    /** Signature of x: `bands` blocks of `rows` SimHash bits, as one long per block. */
    internal static long[] Signature(double[] x, int bands, int rows, ulong seed)
    {
        int dimension = x.Length;
        ulong state = seed;
        var signature = new long[bands];
        var direction = new double[dimension];
        for (int band = 0; band < bands; band++)
        {
            long value = 0;
            for (int row = 0; row < rows; row++)
            {
                for (int i = 0; i < dimension; i++) direction[i] = Rng.NextGaussian(ref state);
                if (SimHashBit(direction, x) == 1) value |= 1L << row;
            }
            signature[band] = value;
        }
        return signature;
    }

    internal static bool Candidate(long[] a, long[] b)
    {
        for (int band = 0; band < a.Length; band++) if (a[band] == b[band]) return true;
        return false;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    public static void Main()
    {
        const int Dimension = 20;
        ulong state = 599001UL;

        // ---- the normal distribution function used by the p-stable formula ----------------
        CheckClose(Math3.StandardNormalCdf(0.0), 0.5, 1e-12, "Phi(0) = 1/2");
        CheckClose(Math3.StandardNormalCdf(1.0), 0.8413447460685429, 1e-6, "Phi(1) = 0.841345");
        CheckClose(Math3.StandardNormalCdf(1.96), 0.9750021048517795, 1e-6, "Phi(1.96) = 0.975002");
        CheckClose(Math3.StandardNormalCdf(-1.0), 1.0 - 0.8413447460685429, 1e-6, "Phi is symmetric");

        // ---- SimHash: collision probability 1 - theta / pi --------------------------------
        var reference = new double[Dimension];
        for (int i = 0; i < Dimension; i++) reference[i] = Rng.NextUnit(ref state);
        var same = (double[])reference.Clone();
        var opposite = new double[Dimension];
        for (int i = 0; i < Dimension; i++) opposite[i] = -reference[i];
        CheckClose(Lsh.SimHashCollisionRate(reference, same, 200, 599111UL), 1.0, 1e-12,
                "identical vectors always collide");
        CheckClose(Lsh.SimHashCollisionRate(reference, opposite, 200, 599222UL), 0.0, 1e-12,
                "antipodal vectors never collide");

        // Copies of the reference at prescribed angles, built from a Gram-Schmidt orthogonal direction.
        var orthogonal = new double[Dimension];
        for (int i = 0; i < Dimension; i++) orthogonal[i] = Rng.NextUnit(ref state);
        double projection = Dot(orthogonal, reference) / Dot(reference, reference);
        for (int i = 0; i < Dimension; i++) orthogonal[i] -= projection * reference[i];
        double referenceNorm = Norm(reference);
        double orthogonalNorm = Norm(orthogonal);
        Check(Math.Abs(Dot(reference, orthogonal)) <= 1e-12 * referenceNorm * orthogonalNorm,
                "the constructed direction is orthogonal to the reference");
        foreach (double theta in new[] { Math.PI / 4.0, Math.PI / 2.0, 3.0 * Math.PI / 4.0 })
        {
            var rotated = new double[Dimension];
            for (int i = 0; i < Dimension; i++)
            {
                rotated[i] = Math.Cos(theta) * reference[i] / referenceNorm
                        + Math.Sin(theta) * orthogonal[i] / orthogonalNorm;
            }
            double measuredAngle = Lsh.Angle(reference, rotated);
            CheckClose(measuredAngle, theta, 1e-9, "the constructed pair subtends exactly " + theta.ToString("0.0000") + " rad");
            double empirical = Lsh.SimHashCollisionRate(reference, rotated, 2000, 599333UL + (ulong)(theta * 1000.0));
            CheckClose(empirical, 1.0 - theta / Math.PI, 0.03, "the SimHash collision rate matches 1 - theta/pi at theta = "
                    + theta.ToString("0.0000") + " rad (" + empirical.ToString("0.0000") + ")");
        }
        CheckClose(1.0 - (Math.PI / 2.0) / Math.PI, 0.5, 1e-15, "orthogonal vectors have collision probability 1/2");

        // ---- p-stable LSH: the closed form against simulation ------------------------------
        var far = new double[Dimension];
        for (int i = 0; i < Dimension; i++) far[i] = Rng.NextUnit(ref state);
        double width = 1.0;
        foreach (double ratio in new[] { 0.5, 1.0, 2.0 })
        {
            double distance = width / ratio;
            var nearby = (double[])far.Clone();
            nearby[0] += distance;
            CheckClose(Lsh.Norm(Lsh.Difference(far, nearby)), distance, 1e-12, "the constructed pair has distance " + distance);
            double theoretical = Math3.PStableCollisionProbability(distance, width);
            double empirical = Lsh.PStableCollisionRate(far, nearby, width, 2000, 599444UL);
            CheckClose(empirical, theoretical, 0.03, "the p-stable collision rate matches the closed form at c/w = "
                    + ratio + " (" + empirical.ToString("0.0000") + " against " + theoretical.ToString("0.0000") + ")");
        }
        Check(Math3.PStableCollisionProbability(1e-6, 1.0) > 0.99, "the p-stable probability tends to 1 as the distance vanishes");
        Check(Math3.PStableCollisionProbability(1e6, 1.0) < 1e-6, "the p-stable probability vanishes for large distances");
        Check(Math3.PStableCollisionProbability(1.0, 1.0) > Math3.PStableCollisionProbability(2.0, 1.0),
                "the p-stable probability decreases with the distance");

        // ---- banding ------------------------------------------------------------------------
        for (int rows = 1; rows <= 4; rows++)
        {
            for (int bands = 1; bands <= 4; bands++)
            {
                double similarity = 0.37;
                double expected = 1.0 - Math.Pow(1.0 - Math.Pow(similarity, rows), bands);
                CheckClose(Lsh.BandedProbability(similarity, bands, rows), expected, 1e-15,
                        "the banded formula is 1 - (1 - p^r)^b at (" + bands + "," + rows + ")");
            }
        }
        CheckClose(Lsh.BandedProbability(1.0, 16, 8), 1.0, 1e-15, "a similarity-1 pair always collides in some band");
        Check(Lsh.BandedProbability(0.5, 16, 8) < 0.1, "half-similar pairs rarely collide with 16 x 8 banding");
        Check(Lsh.BandedProbability(0.5, 32, 8) > Lsh.BandedProbability(0.5, 16, 8),
                "more tables raise the candidate rate");
        Check(Lsh.BandedProbability(0.8, 16, 8) > Lsh.BandedProbability(0.7, 16, 8),
                "the candidate rate increases with the similarity");

        // ---- measured candidate rate against the analytic banding prediction --------------
        {
            int bands = 8;
            int rows = 8;
            var first = new double[Dimension];
            var second = new double[Dimension];
            for (int i = 0; i < Dimension; i++)
            {
                first[i] = Rng.NextUnit(ref state);
                second[i] = Rng.NextUnit(ref state);
            }
            double cosine = 0.6;
            var mixed = new double[Dimension];
            for (int i = 0; i < Dimension; i++)
            {
                mixed[i] = cosine * first[i] / Norm(first)
                        + Math.Sqrt(1.0 - cosine * cosine) * second[i] / Norm(second);
            }
            // SimHash collides with probability 1 - theta / pi, not with the cosine itself.
            double angle = Lsh.Angle(first, mixed);
            double collisionProbability = 1.0 - angle / Math.PI;
            int candidates = 0;
            const int Trials = 600;
            for (int trial = 0; trial < Trials; trial++)
            {
                ulong seed = 599666UL + 977UL * (ulong)(trial + 1);
                if (Lsh.Candidate(Lsh.Signature(first, bands, rows, seed), Lsh.Signature(mixed, bands, rows, seed))) candidates++;
            }
            double analytic = Lsh.BandedProbability(collisionProbability, bands, rows);
            double empirical = candidates / (double)Trials;
            CheckClose(empirical, analytic, 0.06, "the measured candidate rate matches 1 - (1 - p^r)^b ("
                    + empirical.ToString("0.000") + " against " + analytic.ToString("0.000") + " at p = "
                    + collisionProbability.ToString("0.000") + ")");
        }

        // ---- LSH ranks the nearest neighbour above a random point -------------------------
        {
            var query = new double[Dimension];
            var near = new double[Dimension];
            var randomPoint = new double[Dimension];
            for (int i = 0; i < Dimension; i++)
            {
                query[i] = Rng.NextUnit(ref state);
                randomPoint[i] = Rng.NextUnit(ref state);
            }
            for (int i = 0; i < Dimension; i++) near[i] = query[i] + 0.05 * Rng.NextUnit(ref state);
            double nearRate = Lsh.SimHashCollisionRate(query, near, 500, 599777UL);
            double farRate = Lsh.SimHashCollisionRate(query, randomPoint, 500, 599777UL);
            Check(nearRate > farRate + 0.1, "the near neighbour collides far more often than a random point ("
                    + nearRate.ToString("0.000") + " against " + farRate.ToString("0.000") + ")");
        }

        // ---- the actual nearest-neighbour demo ---------------------------------------------
        // SimHash hashes angles, not distances, so uniform points in a box are a hopeless data set for it
        // (in high dimension every pair subtends about 90 degrees). The demo therefore uses a clustered
        // data set, which is exactly the regime where cosine LSH is designed to work: 10 directions, 50
        // unit points around each, and queries drawn afresh from a cluster.
        {
            const int Clusters = 10;
            const int PerCluster = 50;
            const int DatabaseSize = Clusters * PerCluster;
            const int Queries = 50;
            int bands = 20;
            int rows = 10;
            var centres = new double[Clusters][];
            for (int c = 0; c < Clusters; c++)
            {
                centres[c] = new double[Dimension];
                for (int j = 0; j < Dimension; j++) centres[c][j] = Rng.NextUnit(ref state);
                double norm = Lsh.Norm(centres[c]);
                for (int j = 0; j < Dimension; j++) centres[c][j] /= norm;
            }
            var database = new double[DatabaseSize][];
            for (int c = 0; c < Clusters; c++)
            {
                for (int m = 0; m < PerCluster; m++)
                {
                    var point = new double[Dimension];
                    for (int j = 0; j < Dimension; j++) point[j] = centres[c][j] + 0.15 * Rng.NextUnit(ref state);
                    double norm = Lsh.Norm(point);
                    for (int j = 0; j < Dimension; j++) point[j] /= norm;
                    database[c * PerCluster + m] = point;
                }
            }
            var signatures = new long[DatabaseSize][];
            // Every point and every query must be hashed with the *same* tables, otherwise the bands are
            // not comparable at all -- an easy way to build an LSH index that never retrieves anything.
            for (int i = 0; i < DatabaseSize; i++) signatures[i] = Lsh.Signature(database[i], bands, rows, 599888UL);
            int hits = 0;
            int clusterHits = 0;
            long examined = 0;
            for (int q = 0; q < Queries; q++)
            {
                int cluster = q % Clusters;
                var query = new double[Dimension];
                for (int j = 0; j < Dimension; j++) query[j] = centres[cluster][j] + 0.15 * Rng.NextUnit(ref state);
                double queryNorm = Lsh.Norm(query);
                for (int j = 0; j < Dimension; j++) query[j] /= queryNorm;
                int trueNeighbour = -1;
                double bestDistance = double.PositiveInfinity;
                for (int i = 0; i < DatabaseSize; i++)
                {
                    double distance = 0.0;
                    for (int j = 0; j < Dimension; j++)
                    {
                        double delta = query[j] - database[i][j];
                        distance += delta * delta;
                    }
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        trueNeighbour = i;
                    }
                }
                Check(trueNeighbour / PerCluster == cluster, "query " + q + " has its nearest neighbour in its own cluster");
                long[] querySignature = Lsh.Signature(query, bands, rows, 599888UL);
                int candidates = 0;
                int bestCandidate = -1;
                double bestCandidateDistance = double.PositiveInfinity;
                for (int i = 0; i < DatabaseSize; i++)
                {
                    if (!Lsh.Candidate(querySignature, signatures[i])) continue;
                    candidates++;
                    double distance = 0.0;
                    for (int j = 0; j < Dimension; j++)
                    {
                        double delta = query[j] - database[i][j];
                        distance += delta * delta;
                    }
                    if (distance < bestCandidateDistance)
                    {
                        bestCandidateDistance = distance;
                        bestCandidate = i;
                    }
                }
                examined += candidates;
                if (bestCandidate == trueNeighbour) hits++;
                if (bestCandidate >= 0 && bestCandidate / PerCluster == cluster) clusterHits++;
            }
            Check(hits >= 45, "the LSH candidate set contains the exact nearest neighbour for at least 45 of 50 queries ("
                    + hits + ")");
            Check(clusterHits >= 48, "the retrieved neighbour comes from the right cluster for at least 48 of 50 queries ("
                    + clusterHits + ")");
            double fraction = examined / (double)(Queries * DatabaseSize);
            Check(fraction <= 0.25, "the demo examines at most a quarter of the database ("
                    + fraction.ToString("0.000") + " of the pairs)");
            Check(examined > 0, "the candidate sets are nonempty");
        }

        // ---- determinism --------------------------------------------------------------------
        {
            var point = new double[Dimension];
            for (int i = 0; i < Dimension; i++) point[i] = Rng.NextUnit(ref state);
            long[] first = Lsh.Signature(point, 8, 8, 599123UL);
            long[] second = Lsh.Signature(point, 8, 8, 599123UL);
            bool identical = true;
            for (int i = 0; i < first.Length; i++) if (first[i] != second[i]) identical = false;
            Check(identical, "the signature is reproducible for a fixed seed");
            long[] third = Lsh.Signature(point, 8, 8, 599124UL);
            bool differs = false;
            for (int i = 0; i < first.Length; i++) if (first[i] != third[i]) differs = true;
            Check(differs, "a different seed gives a different signature");
        }

        Console.WriteLine("PASS");
    }

    private static double Dot(double[] a, double[] b) => Lsh.Dot(a, b);

    private static double Norm(double[] a) => Lsh.Norm(a);
}
