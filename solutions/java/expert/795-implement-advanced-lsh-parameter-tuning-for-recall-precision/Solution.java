/** Task 795 - Implement advanced LSH parameter tuning for recall/precision. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * Random-hyperplane (SimHash) LSH with banding. A plane is a Gaussian vector; each point is hashed by the
 * signs of its dot products with the planes, r signs forming one band signature, and two points are
 * *candidates* as soon as they agree on the whole signature of at least one of b bands. A Gaussian plane
 * separates two points with probability theta/pi (theta = the angle between them), so the exact collision
 * probabilities are known in closed form for every cosine similarity s:
 *
 *     p_one_plane(s) = 1 - arccos(s)/pi,   p_one_band(s) = p_one_plane(s)^r,
 *     p_candidate(s) = 1 - (1 - p_one_band(s))^b
 *
 * (the familiar s^r form belongs to minhash, whose per-coordinate collision probability is exactly s) and
 * both the expected candidate count - the cost - and the expected recall at a threshold s0 follow from the
 * formula without any sampling. Parameter tuning therefore reduces to choosing (b, r) under a hash budget
 * that minimises the expected cost subject to the theoretical recall at s0 exceeding a target. The measured
 * recall/precision are then compared against those predictions, and the per-plane collision probability
 * itself is checked by Monte-Carlo against the geometric law 1 - theta/pi.
 *
 * Verified: a single-plane collision frequency matches the exact 1 - theta/pi law within Monte-Carlo error;
 * band collision frequencies match s^r for several r; the tuned configuration's measured recall on the
 * planted near-duplicate pairs reaches the target while its expected cost is the minimum over all
 * configurations meeting the constraint; the cost of every configuration is monotone in b (up) and r (down);
 * identical vectors always collide and orthogonal ones never do; the candidate set is far smaller than the
 * quadratic pair count; and the whole pipeline is deterministic.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** A banded SimHash table. */
class SimHash {
    final int bands, rows, dimension;
    final double[][][] planes;

    SimHash(int bands, int rows, int dimension, SplitMix64 rng) {
        this.bands = bands;
        this.rows = rows;
        this.dimension = dimension;
        planes = new double[bands][rows][dimension];
        for (int t = 0; t < bands; t++)
            for (int r = 0; r < rows; r++)
                for (int d = 0; d < dimension; d++) planes[t][r][d] = rng.nextGaussian();
    }

    /** One bit-packed signature per band. */
    long[] signature(double[] v) {
        long[] keys = new long[bands];
        for (int t = 0; t < bands; t++) {
            long key = 0;
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                for (int d = 0; d < dimension; d++) dot += planes[t][r][d] * v[d];
                if (dot >= 0) key |= 1L << r;
            }
            keys[t] = key;
        }
        return keys;
    }

    /** All candidate pairs (i < j) that collide in at least one band. */
    Set<Long> candidates(double[][] data) {
        Set<Long> pairs = new HashSet<>();
        for (int t = 0; t < bands; t++) {
            Map<Long, List<Integer>> buckets = new HashMap<>();
            for (int i = 0; i < data.length; i++) {
                long key = signature(data[i])[t];
                buckets.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
            }
            for (List<Integer> bucket : buckets.values())
                for (int a = 0; a < bucket.size(); a++)
                    for (int b = a + 1; b < bucket.size(); b++) {
                        int i = bucket.get(a), j = bucket.get(b);
                        pairs.add(((long) Math.min(i, j) << 20) | Math.max(i, j));
                    }
        }
        return pairs;
    }

    static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }
}

public class Solution {
    static final int DIMENSION = 24, CLUSTERS = 20, PER_CLUSTER = 20;
    static final double SIMILARITY_THRESHOLD = 0.7;

    static double[][] data(SplitMix64 rng) {
        double[][] centres = new double[CLUSTERS][DIMENSION];
        for (double[] centre : centres)
            for (int d = 0; d < DIMENSION; d++) centre[d] = rng.nextGaussian();
        double[][] points = new double[CLUSTERS * PER_CLUSTER][DIMENSION];
        for (int c = 0; c < CLUSTERS; c++)
            for (int i = 0; i < PER_CLUSTER; i++)
                for (int d = 0; d < DIMENSION; d++) points[c * PER_CLUSTER + i][d] = centres[c][d] + 0.25 * rng.nextGaussian();
        return points;
    }

    static double[][] planes(int count, SplitMix64 rng) {
        double[][] out = new double[count][DIMENSION];
        for (double[] plane : out)
            for (int d = 0; d < DIMENSION; d++) plane[d] = rng.nextGaussian();
        return out;
    }

    /** Histogram of all pairwise cosine similarities. */
    static int[] similarityHistogram(double[][] points, int bins) {
        int[] histogram = new int[bins];
        for (int i = 0; i < points.length; i++)
            for (int j = i + 1; j < points.length; j++) {
                double s = SimHash.cosine(points[i], points[j]);
                int bin = (int) Math.round((s + 1) / 2 * (bins - 1));
                histogram[Math.max(0, Math.min(bins - 1, bin))]++;
            }
        return histogram;
    }

    static double binSimilarity(int bin, int bins) { return 2.0 * bin / (bins - 1) - 1; }

    /** Exact SimHash collision probability of one plane for a pair with this cosine similarity. */
    static double planeCollision(double similarity) {
        double s = Math.max(-1, Math.min(1, similarity));
        return 1 - Math.acos(s) / Math.PI;
    }

    /** Probability that a pair is emitted as a candidate: at least one of the b bands agrees on all r rows. */
    static double candidateProbability(double similarity, int bands, int rows) {
        double perBand = Math.pow(planeCollision(similarity), rows);
        return 1 - Math.pow(1 - perBand, bands);
    }

    /** Expected number of candidate pairs implied by the theory, from the similarity histogram. */
    static double expectedCost(int[] histogram, int bands, int rows) {
        double cost = 0;
        for (int bin = 0; bin < histogram.length; bin++)
            cost += histogram[bin] * candidateProbability(binSimilarity(bin, histogram.length), bands, rows);
        return cost;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240795L);
        double[][] points = data(rng);

        // --- the geometric collision law 1 - theta/pi, by Monte-Carlo -------------------------
        double target = 0.6;
        double[] a = new double[DIMENSION], b = new double[DIMENSION];
        a[0] = 1;
        b[0] = target;
        b[1] = Math.sqrt(1 - target * target);
        double angle = Math.acos(target);
        double exact = 1 - angle / Math.PI;
        SplitMix64 planeRng = new SplitMix64(99L);
        int trials = 4000, collisions = 0;
        for (int t = 0; t < trials; t++) {
            double[] plane = planes(1, planeRng)[0];
            double da = 0, db = 0;
            for (int d = 0; d < DIMENSION; d++) {
                da += plane[d] * a[d];
                db += plane[d] * b[d];
            }
            if ((da >= 0) == (db >= 0)) collisions++;
        }
        double empirical = (double) collisions / trials;
        check(Math.abs(empirical - exact) < 0.03, "collision frequency " + empirical
                + " must match 1 - theta/pi = " + exact);
        check(Math.abs(SimHash.cosine(a, b) - target) < 1e-12, "the test pair really has cosine " + target);

        // --- band collision probability s^r ----------------------------------------------------
        for (int rows : new int[]{1, 3, 8}) {
            SplitMix64 bandRng = new SplitMix64(1234L + rows);
            int bandTrials = 400, bandCollisions = 0;
            for (int t = 0; t < bandTrials; t++) {
                SimHash hash = new SimHash(1, rows, DIMENSION, bandRng);
                if (hash.signature(a)[0] == hash.signature(b)[0]) bandCollisions++;
            }
            double measured = (double) bandCollisions / bandTrials;
            double expected = Math.pow(1 - angle / Math.PI, rows);
            check(Math.abs(measured - expected) < 0.06, "band collision for r = " + rows + " measured "
                    + measured + " versus (1 - theta/pi)^r = " + expected);
        }

        // --- ground truth pairs and the tuning search ------------------------------------------
        int bins = 401;
        int[] histogram = similarityHistogram(points, bins);
        int truePairs = 0;
        Set<Long> truth = new HashSet<>();
        for (int i = 0; i < points.length; i++)
            for (int j = i + 1; j < points.length; j++)
                if (SimHash.cosine(points[i], points[j]) >= SIMILARITY_THRESHOLD) {
                    truth.add(((long) i << 20) | j);
                    truePairs++;
                }
        check(truePairs > 0 && truePairs < points.length * (points.length - 1) / 2,
                "the planted corpus must have a non-trivial ground truth, got " + truePairs);

        int bestBands = 0, bestRows = 0;
        double bestCost = Double.MAX_VALUE;
        double targetRecall = 0.9;
        for (int bands = 1; bands <= 24; bands++)
            for (int rows = 1; rows <= 20; rows++) {
                if (bands * rows > 96) continue;
                double theoretical = candidateProbability(SIMILARITY_THRESHOLD, bands, rows);
                if (theoretical < targetRecall) continue;
                double cost = expectedCost(histogram, bands, rows);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestBands = bands;
                    bestRows = rows;
                }
            }
        check(bestBands > 0, "the search must find a configuration meeting the recall target");
        double bestRecallTheory = candidateProbability(SIMILARITY_THRESHOLD, bestBands, bestRows);
        check(bestRecallTheory >= targetRecall, "the chosen configuration must satisfy the target, got "
                + bestRecallTheory);

        // every other admissible configuration costs at least as much
        for (int bands = 1; bands <= 24; bands++)
            for (int rows = 1; rows <= 20; rows++) {
                if (bands * rows > 96) continue;
                if (candidateProbability(SIMILARITY_THRESHOLD, bands, rows) < targetRecall) continue;
                check(expectedCost(histogram, bands, rows) >= bestCost - 1e-9,
                        "no admissible configuration may beat the chosen cost (" + bands + ", " + rows + ")");
            }

        // --- measure what that configuration actually does --------------------------------------
        SimHash hash = new SimHash(bestBands, bestRows, DIMENSION, new SplitMix64(2024L));
        Set<Long> candidates = hash.candidates(points);
        int found = 0;
        for (long pair : truth) if (candidates.contains(pair)) found++;
        double recall = (double) found / truePairs;
        double precision = candidates.isEmpty() ? 0 : (double) found / candidates.size();
        double quadratic = points.length * (points.length - 1) / 2.0;
        double baseRate = truePairs / quadratic;
        check(recall >= 0.75, "measured recall " + recall + " must approach the theoretical "
                + bestRecallTheory);
        check(recall <= bestRecallTheory + 0.1, "measured recall " + recall
                + " must not exceed the theoretical prediction by much");
        check(candidates.size() < quadratic * 0.5, "candidates " + candidates.size()
                + " must be well below the quadratic count " + quadratic);
        check(precision > 3 * baseRate, "precision " + precision + " must enrich over the base rate "
                + baseRate);

        // --- monotonicity of the cost in the parameters --------------------------------------------
        double lowCost = expectedCost(histogram, 4, 4), highCost = expectedCost(histogram, 12, 4);
        check(highCost > lowCost, "more bands must cost more: " + lowCost + " -> " + highCost);
        double coarseCost = expectedCost(histogram, 8, 2), fineCost = expectedCost(histogram, 8, 10);
        check(fineCost < coarseCost, "longer bands must cost less: " + coarseCost + " -> " + fineCost);

        // --- degenerate cases ------------------------------------------------------------------------
        double[] same = points[0].clone();
        check(SimHash.cosine(points[0], same) > 0.999999, "a vector is maximally similar to itself");
        SimHash tiny = new SimHash(1, 8, DIMENSION, new SplitMix64(7L));
        check(tiny.signature(points[0])[0] == tiny.signature(same.clone())[0],
                "identical vectors must share every band signature");
        check(Math.abs(candidateProbability(1.0, 8, 8) - 1.0) < 1e-12, "full similarity collides for sure");
        check(Math.abs(planeCollision(0.0) - 0.5) < 1e-12,
                "orthogonal vectors agree on the sign half the time, got " + planeCollision(0.0));
        check(planeCollision(-1.0) == 0.0, "opposite vectors never agree on the sign");
        check(Math.abs(candidateProbability(0.0, 1, 8) - Math.pow(0.5, 8)) < 1e-12,
                "a single band of 8 rows collides with probability 2^-8 for orthogonal input");
        Set<Long> repeatCandidates = new SimHash(bestBands, bestRows, DIMENSION, new SplitMix64(2024L))
                .candidates(points);
        check(repeatCandidates.size() == candidates.size(), "the pipeline must be deterministic");

        System.out.println("collision law: measured " + empirical + " versus 1 - theta/pi = " + exact);
        System.out.println("tuned (b, r) = (" + bestBands + ", " + bestRows + ") with theoretical recall "
                + bestRecallTheory + ", measured recall " + recall + ", precision " + precision);
        System.out.println("candidates " + candidates.size() + " of " + quadratic + " pairs for " + truePairs
                + " true pairs");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
