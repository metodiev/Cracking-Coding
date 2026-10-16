/** Task 912 - Implement end-to-end pipeline for privacy-preserving analytics with DP budgeting.
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * The pipeline answers count, sum, mean and histogram queries over a small dataset while spending a global
 * differential-privacy budget. Two ingredients make the answers certifiable rather than merely noisy:
 *
 *   Laplace mechanism  a query with sensitivity Delta is answered with noise drawn from Lap(Delta/epsilon),
 *                      whose density is calibrated so that neighbouring datasets produce indistinguishable
 *                      answers; the noise must therefore have mean absolute deviation Delta/epsilon and
 *                      variance 2(Delta/epsilon)^2 - both are measured below, not assumed;
 *   accountant         sequential composition charges the sum of the epsilons, parallel composition over
 *                      disjoint partitions charges only the maximum, and a query whose cost would exceed the
 *                      remaining budget is refused *before* it runs, so it consumes nothing.
 *
 * The checks exercise the statistics (calibration of the noise, its CDF, unbiasedness, the 1/eps^2 variance
 * law), the composition rules, the exact-vs-noisy oracle, and the boundaries: eps <= 0, zero sensitivity,
 * an empty dataset, and a budget that is exhausted exactly.
 *
 * Simplification: deterministic splitmix64 sampling instead of a hardware RNG, 40 records, and a handful of
 * query types. Invariants verified: the accountant never overspends, refused queries never consume budget,
 * noise calibration matches the analytic Laplace moments, and every released answer is reproducible.
 */
import java.util.*;

/** splitmix64 with a uniform double, used to invert the Laplace CDF. */
final class Rng912 {
    private long state;

    Rng912(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** Inverse-CDF Laplace sample with scale b: u in (-1/2, 1/2) maps monotonically onto the line. */
    double nextLaplace(double scale) {
        double u = nextDouble() - 0.5;
        double magnitude = -scale * Math.log(1 - 2 * Math.abs(u));
        return u < 0 ? -magnitude : magnitude;
    }
}

/** One privacy request: a query, its sensitivity and the epsilon it wants to spend. */
final class PrivacyRequest {
    final String name;
    final double sensitivity;
    final double epsilon;
    final boolean parallelGroup;

    PrivacyRequest(String name, double sensitivity, double epsilon, boolean parallelGroup) {
        this.name = name;
        this.sensitivity = sensitivity;
        this.epsilon = epsilon;
        this.parallelGroup = parallelGroup;
    }
}

/** Sequential-plus-parallel composition accountant. */
final class Accountant {
    final double budget;
    final Map<String, Double> spentPerGroup = new TreeMap<>();
    double spent;
    int refused;

    Accountant(double budget) { this.budget = budget; }

    double remaining() { return budget - spent; }

    /** True when the request fits; a refused request must leave the accountant untouched. */
    boolean canAfford(PrivacyRequest request) {
        if (request.epsilon <= 0) return false;
        if (request.parallelGroup) {
            double already = spentPerGroup.getOrDefault(request.name, 0.0);
            return Math.max(already, request.epsilon) <= budget - spent + already;
        }
        return spent + request.epsilon <= budget + 1e-12;
    }

    boolean charge(PrivacyRequest request) {
        if (!canAfford(request)) {
            refused++;
            return false;
        }
        if (request.parallelGroup) {
            double already = spentPerGroup.getOrDefault(request.name, 0.0);
            double increase = Math.max(0, request.epsilon - already);
            spentPerGroup.put(request.name, Math.max(already, request.epsilon));
            spent += increase;
        } else {
            spent += request.epsilon;
        }
        return true;
    }
}

/** The dataset and the exact answers used as the oracle. */
final class Dataset {
    final double[] values;
    final int[] buckets;

    Dataset(double[] values, int buckets) {
        this.values = values;
        this.buckets = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            this.buckets[i] = (int) Math.min(buckets - 1, Math.floor(values[i] * buckets));
        }
    }

    double count() { return values.length; }

    double sum() {
        double total = 0;
        for (double value : values) total += value;
        return total;
    }

    double mean() { return values.length == 0 ? 0 : sum() / values.length; }

    double[] histogram(int size) {
        double[] counts = new double[size];
        for (int bucket : buckets) counts[bucket]++;
        return counts;
    }
}

public class Solution {
    static final double BUDGET = 4.0;
    static final int RECORDS = 40;
    static final int NOISE_SAMPLES = 20000;

    public static void main(String[] args) {
        Dataset data = makeData(RECORDS, 0x912ABCDEL);

        // ---- 1. the Laplace mechanism is calibrated, measured over many draws --------------------------
        double scale = 2.0;
        Rng912 noiseRng = new Rng912(0x9120001L);
        double absoluteDeviation = 0;
        double variance = 0;
        double mean = 0;
        double[] samples = new double[NOISE_SAMPLES];
        for (int i = 0; i < NOISE_SAMPLES; i++) {
            double sample = noiseRng.nextLaplace(scale);
            samples[i] = sample;
            absoluteDeviation += Math.abs(sample);
            mean += sample;
        }
        absoluteDeviation /= NOISE_SAMPLES;
        mean /= NOISE_SAMPLES;
        for (double sample : samples) variance += (sample - mean) * (sample - mean);
        variance /= NOISE_SAMPLES;
        check(Math.abs(absoluteDeviation - scale) < 0.03 * scale,
                "the mean absolute deviation must equal the scale: " + absoluteDeviation + " vs " + scale);
        check(Math.abs(variance - 2 * scale * scale) < 0.1 * 2 * scale * scale,
                "the variance must equal 2*scale^2: " + variance + " vs " + (2 * scale * scale));
        check(Math.abs(mean) < 4 * Math.sqrt(variance / NOISE_SAMPLES) + 0.01,
                "Laplace noise must be symmetric around zero, sample mean " + mean);

        // ---- 2. the empirical CDF must match the analytic Laplace CDF ---------------------------------
        double[] points = {-2 * scale, -scale, 0, scale, 2 * scale};
        for (double point : points) {
            int below = 0;
            for (double sample : samples) if (sample <= point) below++;
            double empirical = below / (double) NOISE_SAMPLES;
            double analytic = point < 0 ? 0.5 * Math.exp(point / scale) : 1 - 0.5 * Math.exp(-point / scale);
            check(Math.abs(empirical - analytic) < 0.02,
                    "empirical CDF at " + point + " is " + empirical + ", analytic " + analytic);
        }

        // ---- 3. variance scales as 1/epsilon^2 ---------------------------------------------------------
        double baseVariance = empiricalVariance(0x9120002L, 1.0, 2000);
        double halfEpsilonVariance = empiricalVariance(0x9120002L, 0.5, 2000);
        double quarterEpsilonVariance = empiricalVariance(0x9120002L, 0.25, 2000);
        check(Math.abs(halfEpsilonVariance / baseVariance - 4) < 1.2,
                "halving epsilon should quadruple the variance: " + halfEpsilonVariance / baseVariance);
        check(Math.abs(quarterEpsilonVariance / baseVariance - 16) < 8,
                "quartering epsilon should give 16x the variance: " + quarterEpsilonVariance / baseVariance);

        // ---- 4. the released answers are unbiased and the oracle is exact ------------------------------
        double trueCount = data.count();
        double trueSum = data.sum();
        double trueMean = data.mean();
        Rng912 releaseRng = new Rng912(0x9120003L);
        double countTotal = 0;
        double sumTotal = 0;
        for (int repetition = 0; repetition < 2000; repetition++) {
            countTotal += trueCount + releaseRng.nextLaplace(1.0 / 0.5);
            sumTotal += trueSum + releaseRng.nextLaplace(1.0 / 0.5);
        }
        double countMean = countTotal / 2000;
        double sumMean = sumTotal / 2000;
        double countNoiseSd = Math.sqrt(2 * 4.0 / 2000);
        check(Math.abs(countMean - trueCount) < 4 * countNoiseSd + 0.05,
                "the noisy count must be unbiased: " + countMean + " vs " + trueCount);
        check(Math.abs(sumMean - trueSum) < 4 * Math.sqrt(2 * 4.0 / 2000) + 0.5,
                "the noisy sum must be unbiased: " + sumMean + " vs " + trueSum);
        check(Math.abs(data.mean() - trueSum / RECORDS) < 1e-12, "the exact mean must match the exact sum");
        double[] histogram = data.histogram(5);
        double histogramTotal = 0;
        for (double bucket : histogram) histogramTotal += bucket;
        check(Math.abs(histogramTotal - trueCount) < 1e-12,
                "the exact histogram must sum to the record count");
        check(trueMean > 0 && trueMean < 1, "the fixture values must lie in [0,1], mean " + trueMean);

        // ---- 5. the accountant: sequential composition and refusals -----------------------------------
        Accountant accountant = new Accountant(BUDGET);
        PrivacyRequest count = new PrivacyRequest("count", 1.0, 0.5, false);
        for (int i = 0; i < 4; i++) {
            check(accountant.charge(count), "the first four half-epsilon queries must fit");
        }
        check(Math.abs(accountant.spent - 2.0) < 1e-12, "four half-epsilon queries spend 2.0, got "
                + accountant.spent);
        PrivacyRequest huge = new PrivacyRequest("mean", 0.025, 3.0, false);
        check(!accountant.charge(huge), "a query above the remaining budget must be refused");
        check(Math.abs(accountant.spent - 2.0) < 1e-12, "a refused query must not consume budget: "
                + accountant.spent);
        check(accountant.refused == 1, "the refusal must be counted, got " + accountant.refused);
        check(accountant.remaining() > 0, "the budget must not be exhausted yet");

        // ---- 6. parallel composition over disjoint partitions -----------------------------------------
        Accountant parallel = new Accountant(BUDGET);
        for (int group = 0; group < 5; group++) {
            check(parallel.charge(new PrivacyRequest("partition-" + group, 1.0, 0.2, false)),
                    "each partition group must be affordable");
        }
        check(Math.abs(parallel.spent - 1.0) < 1e-12, "five sequential groups spend 1.0, got " + parallel.spent);
        Accountant disjoint = new Accountant(BUDGET);
        for (int group = 0; group < 5; group++) {
            // The same group id means the same histogram over five disjoint partitions: charge the maximum.
            check(disjoint.charge(new PrivacyRequest("histogram", 1.0, 0.2, true)),
                    "each disjoint partition query must be granted");
        }
        check(Math.abs(disjoint.spent - 0.2) < 1e-12,
                "disjoint partitions charge the maximum, not the sum: " + disjoint.spent);
        check(disjoint.spent < parallel.spent, "parallel composition must beat sequential composition");

        // ---- 7. boundaries ------------------------------------------------------------------------------
        Accountant exhausted = new Accountant(1.0);
        check(exhausted.charge(new PrivacyRequest("a", 1.0, 1.0, false)), "an exact-fit query must be granted");
        check(Math.abs(exhausted.remaining()) < 1e-12, "the budget must be exactly exhausted");
        check(!exhausted.charge(new PrivacyRequest("b", 1.0, 1e-9, false)),
                "nothing fits once the budget is exhausted");
        check(!new Accountant(BUDGET).charge(new PrivacyRequest("bad", 1.0, 0.0, false)),
                "epsilon = 0 is not a private query and must be refused");
        check(!new Accountant(BUDGET).charge(new PrivacyRequest("bad", 1.0, -1.0, false)),
                "negative epsilon must be refused");
        Dataset empty = makeData(0, 0x9120004L);
        check(empty.count() == 0 && empty.sum() == 0, "an empty dataset must have zero count and sum");
        double[] emptyHistogram = empty.histogram(5);
        double emptyTotal = 0;
        for (double bucket : emptyHistogram) emptyTotal += bucket;
        check(emptyTotal == 0, "an empty dataset has an all-zero histogram");
        check(!answerable(empty), "a mean query over an empty dataset must be refused, not answered with 0");
        check(answerable(data), "a populated dataset must be answerable");

        // ---- 8. determinism ------------------------------------------------------------------------------
        check(empiricalVariance(0x9120002L, 1.0, 2000) == baseVariance, "noise draws must be reproducible");
        Accountant replay = new Accountant(BUDGET);
        for (int i = 0; i < 4; i++) replay.charge(count);
        check(Math.abs(replay.spent - accountant.spent) < 1e-12, "accounting must be reproducible");
        check(replay.refused == 0, "a replay of the affordable queries must refuse nothing");

        System.out.println("records " + RECORDS + ", true count " + trueCount + " sum " + round(trueSum)
                + " mean " + round(trueMean));
        System.out.println("Laplace(2.0): mean |x| " + round(absoluteDeviation) + " (theory 2.0), variance "
                + round(variance) + " (theory 8.0)");
        System.out.println("variance law: " + round(baseVariance) + " -> " + round(halfEpsilonVariance)
                + " -> " + round(quarterEpsilonVariance) + " for eps 1 / 0.5 / 0.25; budget spent "
                + accountant.spent + " of " + BUDGET + ", parallel groups " + disjoint.spent);
        System.out.println("PASS");
    }

    static Dataset makeData(int count, long seed) {
        Rng912 rng = new Rng912(seed);
        double[] values = new double[count];
        for (int i = 0; i < count; i++) values[i] = rng.nextDouble();
        return new Dataset(values, 5);
    }

    /** Variance of the released count for a given epsilon, measured over many repetitions. */
    static double empiricalVariance(long seed, double epsilon, int repetitions) {
        Rng912 rng = new Rng912(seed);
        double scale = 1.0 / epsilon;
        double total = 0;
        double totalSquared = 0;
        for (int i = 0; i < repetitions; i++) {
            double value = rng.nextLaplace(scale);
            total += value;
            totalSquared += value * value;
        }
        double mean = total / repetitions;
        return totalSquared / repetitions - mean * mean;
    }

    static boolean answerable(Dataset data) { return data.count() > 0; }

    static double round(double value) { return Math.round(value * 10000) / 10000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
