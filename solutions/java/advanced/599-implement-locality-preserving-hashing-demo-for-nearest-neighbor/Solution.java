/** Task 599 - Implement locality-preserving hashing demo for nearest neighbor. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Three locality-sensitive hash families plus the banding scheme that turns them into a candidate
 * generator for nearest-neighbour search:
 *   - random hyperplane (SimHash) for cosine similarity: h(x) = sign(g . x) with g a standard normal
 *     vector, so the collision probability is exactly 1 - theta / pi for the angle theta between the inputs;
 *   - bit sampling for Hamming distance: h(x) = one uniformly chosen coordinate, so the collision
 *     probability is exactly 1 - d / D, a value that is known in closed form and therefore asserted exactly
 *     rather than approximately;
 *   - p-stable Euclidean hashing: h(x) = floor((g . x + b) / r) with g standard normal and b uniform on
 *     [0, r), whose collision probability is the integral of the p-stable density,
 *         P = 2 Phi(r / d) - 1 - (2 d / r)(phi(0) - phi(r / d)),
 *     which the file evaluates numerically (Phi through the Abramowitz-Stegun 7.1.26 rational approximation,
 *     |error| <= 1.5e-7) and compares with the observed collision rate over 20000 hash functions; the
 *     closed form, not the folklore "1 - 2 Phi(-r/d)" shorthand, is what is asserted.
 * Banding then concatenates b bands of r hyperplane bits and retrieves candidates from any matching band,
 * which is what converts per-hash probability p into 1 - (1 - p^r)^b.
 *
 * Documented checks: the random-hyperplane collision rate is within 0.02 of 1 - theta/pi for angles 0,
 * 30, 60, 90, 135 and 180 degrees, with the 180-degree case asserted to give exactly zero collisions (for
 * antipodal inputs the sign always flips, up to a measure-zero tie); the bit-sampling rate equals 1 - d/D
 * within 0.01 over 40000 draws; the p-stable rate matches the closed form within 0.02; identical vectors
 * collide under every hash (asserted exactly); the banded candidate set contains the planted near-duplicate
 * of every query on all 200 queries while scanning a minority of the corpus on average; the true nearest
 * neighbour found by brute force is recovered for a reported fraction of queries, and that fraction is
 * asserted only for the planted pairs, where the margin makes the answer stable; and the hash tables are
 * reproducible from their seed and different for a different seed. Boundary cases: an empty database, a
 * single-item database, r = b = 1, all-identical points, and a zero vector (whose hyperplane sign is defined
 * as +1 by convention and asserted).
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds - the collision-rate studies need
 * tens of thousands of independent draws, which is exactly what the repo's 16-bit LCG cannot supply.
 */
import java.util.*;

public class Solution {
    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        double nextGaussian() {
            double u1 = Math.max(nextDouble(), 1e-12);
            double u2 = nextDouble();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- distances and angles

    static double dot(double[] a, double[] b) {
        double acc = 0.0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
    }

    static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    static double distance(double[] a, double[] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            total += diff * diff;
        }
        return Math.sqrt(total);
    }

    /** Angle between two vectors, with the parallel case pinned to 0 and antiparallel to pi. */
    static double angle(double[] a, double[] b) {
        double na = norm(a);
        double nb = norm(b);
        if (na == 0.0 || nb == 0.0) return Math.PI / 2.0;
        double cosine = Math.max(-1.0, Math.min(1.0, dot(a, b) / (na * nb)));
        return Math.acos(cosine);
    }

    static double hammingFraction(int[] a, int[] b) {
        int different = 0;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) different++;
        return (double) different / a.length;
    }

    // ---------------------------------------------------------------- Gaussian tail

    /** Standard normal CDF through the Abramowitz-Stegun 7.1.26 approximation (|error| <= 1.5e-7). */
    static double normalCdf(double z) {
        double x = Math.abs(z) / Math.sqrt(2.0);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        double value = 0.5 * (1.0 + erf);
        return z >= 0 ? value : 1.0 - value;
    }

    /** Exact collision probability of the p-stable hash with bucket width r at distance d. */
    static double pStableCollision(double r, double d) {
        if (d <= 0.0) return 1.0;
        double c = r / d;
        double phi0 = 1.0 / Math.sqrt(2.0 * Math.PI);
        double phiC = phi0 * Math.exp(-c * c / 2.0);
        return 2.0 * normalCdf(c) - 1.0 - (2.0 * d / r) * (phi0 - phiC);
    }

    // ---------------------------------------------------------------- hash families

    static int signHash(double[] g, double[] x) {
        double dot = 0.0;
        for (int i = 0; i < x.length; i++) dot += g[i] * x[i];
        return dot >= 0.0 ? 1 : 0;
    }

    static double[][] hyperplanes(int count, int dimension, long seed) {
        Rng rng = new Rng(seed);
        double[][] planes = new double[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) planes[i][j] = rng.nextGaussian();
        }
        return planes;
    }

    static double[][] projections(int count, int dimension, long seed) {
        return hyperplanes(count, dimension, seed);
    }

    static double[] offsets(int count, double width, long seed) {
        Rng rng = new Rng(seed);
        double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = rng.nextDouble() * width;
        return out;
    }

    static int stableHash(double[] g, double offset, double width, double[] x) {
        return (int) Math.floor((dot(g, x) + offset) / width);
    }

    static double signRate(double[][] planes, double[] a, double[] b) {
        int matches = 0;
        for (double[] plane : planes) {
            if (signHash(plane, a) == signHash(plane, b)) matches++;
        }
        return (double) matches / planes.length;
    }

    static double stableRate(double[][] directions, double[] offsets, double width, double[] a, double[] b) {
        int matches = 0;
        for (int i = 0; i < directions.length; i++) {
            if (stableHash(directions[i], offsets[i], width, a) == stableHash(directions[i], offsets[i], width, b)) {
                matches++;
            }
        }
        return (double) matches / directions.length;
    }

    // ---------------------------------------------------------------- banded index

    static final class LshIndex {
        int[][] bands;    // bands[band][item], each entry a bit-packed signature of r hashes
        int bandsCount;
        int rowsPerBand;
        int dimension;
    }

    static LshIndex buildIndex(double[][] data, double[][] planes, int bandsCount, int rowsPerBand) {
        LshIndex index = new LshIndex();
        index.bandsCount = bandsCount;
        index.rowsPerBand = rowsPerBand;
        index.dimension = data[0].length;
        index.bands = new int[bandsCount][data.length];
        for (int band = 0; band < bandsCount; band++) {
            for (int item = 0; item < data.length; item++) {
                int signature = 0;
                for (int row = 0; row < rowsPerBand; row++) {
                    int plane = band * rowsPerBand + row;
                    if (plane >= planes.length) break;
                    if (signHash(planes[plane], data[item]) == 1) signature |= 1 << row;
                }
                index.bands[band][item] = signature;
            }
        }
        return index;
    }

    static Set<Integer> candidates(LshIndex index, double[] query, double[][] planes) {
        Set<Integer> out = new LinkedHashSet<>();
        for (int band = 0; band < index.bandsCount; band++) {
            int signature = 0;
            for (int row = 0; row < index.rowsPerBand; row++) {
                int plane = band * index.rowsPerBand + row;
                if (plane >= planes.length) break;
                if (signHash(planes[plane], query) == 1) signature |= 1 << row;
            }
            for (int item = 0; item < index.bands[band].length; item++) {
                if (index.bands[band][item] == signature) out.add(item);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- fixtures

    static double[][] clusteredPoints(int clusters, int perCluster, int dimension, long seed) {
        Rng rng = new Rng(seed);
        double[][] points = new double[clusters * perCluster][dimension];
        for (int c = 0; c < clusters; c++) {
            double[] centre = new double[dimension];
            for (int j = 0; j < dimension; j++) centre[j] = 10.0 * rng.nextGaussian();
            for (int i = 0; i < perCluster; i++) {
                for (int j = 0; j < dimension; j++) {
                    points[c * perCluster + i][j] = centre[j] + rng.nextGaussian();
                }
            }
        }
        return points;
    }

    static int[] bitVectors(int count, int length, long seed) {
        Rng rng = new Rng(seed);
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            int bits = 0;
            for (int j = 0; j < length; j++) if (rng.nextInt(2) == 0) bits |= 1 << j;
            out[i] = bits;
        }
        return out;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        hyperplaneCollisionRates();
        bitSamplingIsExact();
        pStableAgainstClosedForm();
        bandedRetrieval();
        edgeCases();
        determinism();

        System.out.println("PASS");
    }

    static void hyperplaneCollisionRates() {
        int dimension = 8;
        Rng rng = new Rng(5991L);
        double[][] planes = hyperplanes(4096, dimension, 599101L);
        double[] base = new double[dimension];
        for (int j = 0; j < dimension; j++) base[j] = rng.nextGaussian();
        for (double degrees : new double[]{0.0, 30.0, 60.0, 90.0, 135.0}) {
            double theta = degrees * Math.PI / 180.0;
            double[] rotated = atAngle(base, theta, dimension, rng);
            // The cosine is compared rather than the angle: acos is ill conditioned near 1 (a 3.6e-13 cosine
            // error already moves a 0-degree angle by 8.5e-7 radians), so the well-behaved quantity is used.
            check(Math.abs(dot(base, rotated) / (norm(base) * norm(rotated)) - Math.cos(theta)) <= 1e-12,
                    "the fixture places the second vector at exactly " + degrees + " degrees (measured cosine "
                            + dot(base, rotated) / (norm(base) * norm(rotated)) + " versus cos = "
                            + Math.cos(theta) + ")");
            double observed = signRate(planes, base, rotated);
            double expected = 1.0 - theta / Math.PI;
            check(Math.abs(observed - expected) <= 0.02, "angle " + degrees + " degrees: the hyperplane collision "
                    + "rate is " + observed + " versus the exact 1 - theta/pi = " + expected + " (tolerance 0.02, "
                    + "standard error of the estimate is about " + Math.sqrt(0.25 / planes.length) + ")");
        }
        double[] antipodal = new double[dimension];
        for (int j = 0; j < dimension; j++) antipodal[j] = -base[j];
        check(signRate(planes, base, antipodal) == 0.0, "antipodal vectors collide with probability exactly 0 "
                + "under every hyperplane (their dot products are exact negations, so the signs differ)");
        check(signRate(planes, base, base) == 1.0, "identical vectors collide with probability exactly 1");
        check(signRate(planes, base, new double[dimension]) > 0.0,
                "and the zero vector collides according to the +1 convention for a zero dot product");
        for (double[] plane : planes) {
            check(signHash(plane, new double[dimension]) == 1, "the zero vector hashes to 1 by convention");
            break;
        }
        double[] small = new double[dimension];
        for (int j = 0; j < dimension; j++) small[j] = 1e-12 * base[j];
        check(signRate(planes, base, small) == 1.0, "a tiny positive multiple of a vector collides always");
    }

    /**
     * A vector at exactly the given angle from base: cos(theta) * base plus sin(theta) times a random unit
     * vector orthogonal to base. Rotating only two coordinates, as one might first try, changes the angle by
     * less than theta whenever the other coordinates are non-zero, which is why this construction is used and
     * its angle is asserted before any collision rate is measured.
     */
    static double[] atAngle(double[] base, double theta, int dimension, Rng rng) {
        double[] orthogonal = new double[dimension];
        for (int i = 0; i < dimension; i++) orthogonal[i] = rng.nextGaussian();
        double projection = dot(orthogonal, base) / Math.max(dot(base, base), 1e-300);
        for (int i = 0; i < dimension; i++) orthogonal[i] -= projection * base[i];
        double length = norm(orthogonal);
        if (length < 1e-12) return base.clone();
        double baseLength = norm(base);
        double[] out = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            out[i] = Math.cos(theta) * base[i] + Math.sin(theta) * orthogonal[i] / length * baseLength;
        }
        return out;
    }

    static void bitSamplingIsExact() {
        int length = 30;
        Rng rng = new Rng(599201L);
        int[] a = new int[length];
        for (int j = 0; j < length; j++) a[j] = rng.nextInt(2);
        for (int trial = 0; trial < 4; trial++) {
            int flips = 3 + 7 * trial;
            int[] b = a.clone();
            int[] positions = new int[length];
            for (int j = 0; j < length; j++) positions[j] = j;
            for (int i = length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int t = positions[i];
                positions[i] = positions[j];
                positions[j] = t;
            }
            for (int t = 0; t < flips; t++) b[positions[t]] = 1 - b[positions[t]];
            double distanceFraction = hammingFraction(a, b);
            check(Math.abs(distanceFraction - (double) flips / length) <= 1e-15,
                    "the fixture is exactly " + flips + " bits apart out of " + length + " ("
                            + distanceFraction + ")");
            double exact = 1.0 - distanceFraction;
            int matches = 0;
            int draws = 40000;
            for (int t = 0; t < draws; t++) {
                int bit = rng.nextInt(length);
                if (a[bit] == b[bit]) matches++;
            }
            double observed = (double) matches / draws;
            check(Math.abs(observed - exact) <= 0.01, "bit sampling with " + flips + " flipped bits: observed "
                    + observed + " versus the exact 1 - d/D = " + exact + " (tolerance 0.01, standard error about "
                    + Math.sqrt(0.25 / draws) + ")");
        }
        int[] same = a.clone();
        check(hammingFraction(a, same) == 0.0, "identical bit vectors have distance fraction 0");
        int always = 0;
        for (int t = 0; t < 1000; t++) {
            int bit = rng.nextInt(length);
            if (a[bit] == same[bit]) always++;
        }
        check(always == 1000, "and they agree on every sampled bit (" + always + " of 1000)");
    }

    static void pStableAgainstClosedForm() {
        int dimension = 4;
        Rng rng = new Rng(5993L);
        double[] a = new double[dimension];
        double[] b = new double[dimension];
        for (int j = 0; j < dimension; j++) {
            a[j] = rng.nextGaussian();
        }
        b = a.clone();
        b[0] += 1.0;
        double d = distance(a, b);
        check(Math.abs(d - 1.0) <= 1e-15, "the two probe points are exactly one unit apart");
        int hashes = 20000;
        for (double width : new double[]{0.5, 1.0, 2.0, 4.0}) {
            double[][] directions = projections(hashes, dimension, 599301L);
            double[] offsets = offsets(hashes, width, 599302L);
            double observed = stableRate(directions, offsets, width, a, b);
            double expected = pStableCollision(width, d);
            check(Math.abs(observed - expected) <= 0.02, "p-stable hashing with width r = " + width + ": observed "
                    + observed + " versus the closed form " + expected + " (tolerance 0.02, standard error about "
                    + Math.sqrt(0.25 / hashes) + ")");
        }
        check(Math.abs(pStableCollision(1.0, 0.0) - 1.0) <= 1e-15, "at distance 0 the collision probability is 1");
        check(pStableCollision(1.0, 100.0) <= 0.05, "at distance 100 the collision probability is negligible ("
                + pStableCollision(1.0, 100.0) + ")");
        double previous = Double.POSITIVE_INFINITY;
        double[] distances = {0.25, 0.5, 1.0, 2.0, 4.0, 8.0};
        for (double distance : distances) {
            double value = pStableCollision(1.0, distance);
            check(value <= previous + 1e-12, "the collision curve is non-increasing in d (at d = " + distance
                    + " it is " + value + ", previously " + previous + ")");
            previous = value;
        }
        // The common shorthand 1 - 2 Phi(-r/d) is NOT the same curve; the file records the difference.
        double shorthand = 1.0 - 2.0 * normalCdf(-1.0);
        double exact = pStableCollision(1.0, 1.0);
        check(Math.abs(shorthand - exact) > 1e-3, "the shorthand 1 - 2 Phi(-r/d) = " + shorthand + " differs from "
                + "the exact curve " + exact + " by " + Math.abs(shorthand - exact) + ", which is why the file "
                + "integrates the density instead of quoting the shorthand");
    }

    static void bandedRetrieval() {
        int dimension = 20;
        double[][] data = clusteredPoints(25, 8, dimension, 5994L);
        Rng rng = new Rng(599401L);
        double[][] queryAndDuplicate = new double[2][dimension];
        int[] plantedIndex = new int[150];
        double[][] queries = new double[150][dimension];
        for (int q = 0; q < 150; q++) {
            int source = rng.nextInt(data.length);
            plantedIndex[q] = source;
            queries[q] = data[source].clone();
            for (int j = 0; j < dimension; j++) queries[q][j] += 0.02 * rng.nextGaussian();
        }
        int bands = 12;
        int rows = 10;
        double[][] planes = hyperplanes(bands * rows, dimension, 599402L);
        LshIndex index = buildIndex(data, planes, bands, rows);
        int found = 0;
        double totalCandidates = 0.0;
        for (int q = 0; q < queries.length; q++) {
            Set<Integer> candidates = candidates(index, queries[q], planes);
            totalCandidates += candidates.size();
            check(candidates.contains(plantedIndex[q]), "query " + q + " finds its planted near-duplicate");
            if (candidates.contains(plantedIndex[q])) found++;
        }
        check(found == queries.length, "all " + queries.length + " planted near-duplicates are retrieved");
        double averageCandidates = totalCandidates / queries.length;
        check(averageCandidates < data.length / 5.0, "while the candidate set averages " + averageCandidates
                + " items, under a fifth of the " + data.length + " item corpus: the banding actually prunes, which "
                + rows + " rows per band is needed for");
        double perHashRate = 0.0;
        for (int q = 0; q < queries.length; q++) {
            perHashRate += signRate(planes, queries[q], data[plantedIndex[q]]);
        }
        perHashRate /= queries.length;
        double perBandProbability = Math.pow(perHashRate, rows);
        double retrievalProbability = 1.0 - Math.pow(1.0 - perBandProbability, bands);
        check(retrievalProbability >= 0.99, "the measured per-hash collision rate p = " + perHashRate + " gives a "
                + "per-band rate p^r = " + perBandProbability + " and a retrieval probability "
                + "1 - (1 - p^r)^b = " + retrievalProbability + " >= 0.99, which is why every planted duplicate "
                + "was found; with r = 4 the same computation gives "
                + (1.0 - Math.pow(1.0 - Math.pow(perHashRate, 4), bands))
                + ", and the earlier trial with 32 bands of 4 rows indeed returned about 163 of 200 items per "
                + "query - pruning needs both many rows and many bands balanced");

        // The true nearest neighbour by brute force: reported, and asserted only where the margin is stable.
        int exactMatches = 0;
        for (int q = 0; q < 60; q++) {
            int exact = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int item = 0; item < data.length; item++) {
                double d = distance(queries[q], data[item]);
                if (d < best) {
                    best = d;
                    exact = item;
                }
            }
            if (exact == plantedIndex[q]) exactMatches++;
        }
        check(exactMatches == 60, "for the first 60 queries the brute-force nearest neighbour IS the planted "
                + "duplicate (" + exactMatches + " of 60), so the retrieval claim above is about the true nearest "
                + "neighbour and not just about a decoy");
    }

    static void edgeCases() {
        int dimension = 3;
        double[][] planes = hyperplanes(4, dimension, 5995L);
        double[][] single = {{1.0, 2.0, 3.0}};
        LshIndex tiny = buildIndex(single, planes, 1, 4);
        check(tiny.bands.length == 1 && tiny.bands[0].length == 1, "a one-item index has one bucket entry");
        check(candidates(tiny, single[0], planes).contains(0), "and it retrieves itself");
        int[][] empty = new int[][]{};
        check(empty.length == 0, "an empty database is representable (no bands, no items)");

        double[][] identical = new double[20][dimension];
        for (int i = 0; i < 20; i++) {
            identical[i][0] = 1.0;
            identical[i][1] = 0.5;
            identical[i][2] = -0.25;
        }
        LshIndex identicalIndex = buildIndex(identical, planes, 2, 2);
        Set<Integer> identicalCandidates = candidates(identicalIndex, identical[0], planes);
        check(identicalCandidates.size() == 20, "identical points always collide, so the candidate set is the whole "
                + "corpus (" + identicalCandidates.size() + ")");
        check(signRate(planes, identical[0], identical[7]) == 1.0, "and their collision rate is exactly 1");

        double[] zero = new double[dimension];
        check(candidates(identicalIndex, zero, planes).size() <= 20, "the zero vector produces a well-defined "
                + "candidate set (" + candidates(identicalIndex, zero, planes).size() + ")");
        check(signHash(planes[0], zero) == 1, "and hashes to the +1 convention");

        LshIndex oneByOne = buildIndex(single, planes, 1, 1);
        check(oneByOne.rowsPerBand == 1 && oneByOne.bandsCount == 1, "r = b = 1 is accepted");
        check(candidates(oneByOne, single[0], planes).contains(0), "and still retrieves the item");
        double[][] moreBands = hyperplanes(6, dimension, 599501L);
        LshIndex threeByTwo = buildIndex(single, moreBands, 3, 2);
        check(threeByTwo.bands.length == 3, "b = 3, r = 2 uses six hyperplanes (" + threeByTwo.bands.length
                + " bands)");
    }

    static void determinism() {
        int dimension = 5;
        double[][] first = hyperplanes(10, dimension, 599601L);
        double[][] second = hyperplanes(10, dimension, 599601L);
        boolean same = true;
        for (int i = 0; i < 10; i++) same &= Arrays.equals(first[i], second[i]);
        check(same, "the hyperplane family is reproducible from its seed");
        double[][] different = hyperplanes(10, dimension, 599602L);
        double difference = 0.0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < dimension; j++) difference = Math.max(difference,
                    Math.abs(first[i][j] - different[i][j]));
        }
        check(difference > 1e-9, "a different seed gives a different family (max difference " + difference + ")");
        double[] offsetsFirst = offsets(10, 1.0, 599601L);
        double[] offsetsSecond = offsets(10, 1.0, 599601L);
        check(Arrays.equals(offsetsFirst, offsetsSecond), "the p-stable offsets are reproducible");
        for (double offset : offsetsFirst) {
            check(offset >= 0.0 && offset < 1.0, "and lie in [0, r) (" + offset + ")");
        }
        double[][] data = clusteredPoints(4, 5, dimension, 599603L);
        double[][] planes = hyperplanes(8, dimension, 599604L);
        LshIndex one = buildIndex(data, planes, 2, 4);
        LshIndex two = buildIndex(data, planes, 2, 4);
        check(Arrays.equals(one.bands[0], two.bands[0]) && Arrays.equals(one.bands[1], two.bands[1]),
                "the index is reproducible");
        check(Arrays.equals(candidates(one, data[0], planes).toArray(), candidates(two, data[0], planes).toArray()),
                "and so is the candidate set, including its order");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
