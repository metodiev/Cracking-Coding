/** Task 710 - Implement private-count sketches with Laplace mechanism simulation.
 * (Tier: Expert, CTCI Beyond -- Security & Privacy)
 *
 * A Count-Min sketch answers "how often did key x appear?" in O(width * depth) counters: every key is
 * hashed to one counter per row, the row counters are increased by the item's weight, and the query
 * returns the *minimum* over the rows. Because each row only ever accumulates the weights of keys that
 * collide there, the estimate is never below the true count (the classic one-sided error), and with
 * epsilon_cm = e / width the overestimate is at most epsilon_cm * N with high probability.
 *
 * Privacy is added on release: the sketch itself is exact, and the answer handed to the analyst is
 * estimate(x) + Laplace(1/epsilon). The estimate has sensitivity 1 - adding or removing a single
 * record increases one counter per row by 1, so the minimum over rows moves by at most 1 - which makes
 * the released answer epsilon-differentially private by the Laplace mechanism.
 *
 * Simplified: noise is injected once per query (query-level DP) rather than into every counter (the
 * counter-level construction needs a separate analysis of the sensitivity of the minimum over noisy
 * counters, cf. the private Count-Min literature), and only the Laplace mechanism is used.
 *
 * Verified: the one-sided error of the exact sketch (estimate >= truth for every key of a skewed
 * stream), the epsilon_cm * N bound on the overestimate, the degenerate width = 1 case where every key
 * reports the total, top-k recovery of heavy hitters, the exact epsilon-DP probability ratio of the
 * released answer computed from the Laplace CDF (plus the negative control that under-noising breaks
 * it), the measured utility 2/epsilon^2, the monotone utility/space tradeoff in the sketch width, and
 * rejection of malformed parameters.
 */
import java.util.*;

class CountMinSketch {
    private final int width;
    private final int depth;
    private final long[] counters;
    private final long seed;
    private long total;

    CountMinSketch(int width, int depth, long seed) {
        if (width < 1) throw new IllegalArgumentException("width must be positive: " + width);
        if (depth < 1) throw new IllegalArgumentException("depth must be positive: " + depth);
        this.width = width;
        this.depth = depth;
        this.seed = seed;
        this.counters = new long[width * depth];
    }

    int width() {
        return width;
    }

    int depth() {
        return depth;
    }

    long counterCount() {
        return (long) width * depth;
    }

    long total() {
        return total;
    }

    void add(long key, long weight) {
        if (weight < 0) throw new IllegalArgumentException("counts must be non-negative");
        for (int row = 0; row < depth; row++) counters[row * width + index(key, row)] += weight;
        total += weight;
    }

    long estimate(long key) {
        long best = Long.MAX_VALUE;
        for (int row = 0; row < depth; row++) best = Math.min(best, counters[row * width + index(key, row)]);
        return best;
    }

    /** splitmix64 over (seed, key, row): deterministic, no java.util.Random. */
    private int index(long key, int row) {
        long z = seed + 0x9E3779B97F4A7C15L * (key * 0x100000001B3L + row + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, width);
    }
}

class LaplaceNoise {
    private final double scale;
    private long state;

    LaplaceNoise(double scale, long seed) {
        if (!(scale > 0)) throw new IllegalArgumentException("scale must be positive");
        this.scale = scale;
        this.state = seed;
    }

    double sample() {
        double uniform = nextUniform() - 0.5;
        double magnitude = -scale * Math.log(1.0 - 2.0 * Math.abs(uniform));
        return uniform < 0 ? -magnitude : magnitude;
    }

    private double nextUniform() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        double uniform = (z >>> 11) * 0x1.0p-53;
        return uniform == 0.0 ? Double.MIN_VALUE : uniform;
    }

    static double cdf(double t, double scale) {
        return t <= 0 ? 0.5 * Math.exp(t / scale) : 1.0 - 0.5 * Math.exp(-t / scale);
    }

    /** Exact worst-case ratio between answers differing by one record (sensitivity 1). */
    static double worstRatio(double scale, int bins, double radius) {
        double worst = 0;
        double binWidth = 2 * radius / bins;
        for (int bin = 0; bin < bins; bin++) {
            double low = -radius + bin * binWidth;
            double high = low + binWidth;
            double numerator = cdf(high, scale) - cdf(low, scale);
            double denominator = cdf(high - 1, scale) - cdf(low - 1, scale);
            if (denominator <= 0) continue;
            worst = Math.max(worst, numerator / denominator);
        }
        return worst;
    }
}

public class Solution {
    static final int KEYS = 200;
    static final long[] TRUE_COUNTS = new long[KEYS + 1];

    /** Skewed stream: key k appears max(1, 4000 / k) times, so a few heavy hitters dominate. */
    static long[] buildStream() {
        List<Long> stream = new ArrayList<>();
        Arrays.fill(TRUE_COUNTS, 0);
        for (int key = 1; key <= KEYS; key++) {
            long repeat = Math.max(1, 4000 / key);
            TRUE_COUNTS[key] = repeat;
            for (long i = 0; i < repeat; i++) stream.add((long) key);
        }
        long[] array = new long[stream.size()];
        for (int i = 0; i < array.length; i++) array[i] = stream.get(i);
        return array;
    }

    public static void main(String[] args) {
        long[] stream = buildStream();
        long streamTotal = 0;
        for (long key : stream) streamTotal++;

        // --- the exact sketch: one-sided error and the CM bound -----------------
        CountMinSketch sketch = new CountMinSketch(256, 4, 0xC0FFEEL);
        for (long key : stream) sketch.add(key, 1);
        check(sketch.total() == streamTotal, "the sketch counts every element");
        check(sketch.counterCount() == 256L * 4, "the sketch uses width * depth counters");
        double collisionFactor = Math.E / sketch.width();
        long worstOverestimate = 0;
        for (int key = 1; key <= KEYS; key++) {
            long estimate = sketch.estimate(key);
            check(estimate >= TRUE_COUNTS[key],
                    "key " + key + ": estimate " + estimate + " below truth " + TRUE_COUNTS[key]);
            worstOverestimate = Math.max(worstOverestimate, estimate - TRUE_COUNTS[key]);
        }
        check(worstOverestimate <= collisionFactor * streamTotal,
                "overestimate " + worstOverestimate + " above the CM bound "
                        + (collisionFactor * streamTotal));
        check(sketch.estimate(KEYS + 1) >= 0, "an unseen key cannot report a negative count");

        // --- degenerate width: everything collides ------------------------------
        CountMinSketch single = new CountMinSketch(1, 1, 1L);
        for (long key : stream) single.add(key, 1);
        check(single.estimate(1) == streamTotal && single.estimate(KEYS) == streamTotal,
                "width one reports the whole stream for every key");
        CountMinSketch empty = new CountMinSketch(32, 3, 7L);
        check(empty.estimate(5) == 0 && empty.total() == 0, "an empty sketch reports zero");
        for (int key = 1; key <= KEYS; key++) {
            check(empty.estimate(key) == 0, "every key of an empty sketch is zero");
        }

        // --- heavy hitters ------------------------------------------------------
        List<Integer> heavyHitters = new ArrayList<>();
        for (int key = 1; key <= KEYS; key++) heavyHitters.add(key);
        heavyHitters.sort((left, right) -> Long.compare(sketch.estimate(right), sketch.estimate(left)));
        List<Integer> truth = new ArrayList<>();
        for (int key = 1; key <= KEYS; key++) truth.add(key);
        truth.sort((left, right) -> Long.compare(TRUE_COUNTS[right], TRUE_COUNTS[left]));
        for (int rank = 0; rank < 10; rank++) {
            check(heavyHitters.get(rank).equals(truth.get(rank)),
                    "rank " + rank + ": sketch says " + heavyHitters.get(rank) + ", truth "
                            + truth.get(rank));
        }

        // --- the utility/space tradeoff ----------------------------------------
        long previousError = Long.MAX_VALUE;
        long narrowestError = -1;
        long widestError = -1;
        for (int width : new int[]{16, 64, 256, 1024}) {
            CountMinSketch narrowed = new CountMinSketch(width, 4, 0xC0FFEEL);
            for (long key : stream) narrowed.add(key, 1);
            long totalOverestimate = 0;
            for (int key = 1; key <= KEYS; key++) {
                totalOverestimate += narrowed.estimate(key) - TRUE_COUNTS[key];
            }
            check(totalOverestimate <= previousError,
                    "a wider sketch must not be worse: " + width + " -> " + totalOverestimate);
            previousError = totalOverestimate;
            if (narrowestError < 0) narrowestError = totalOverestimate;
            widestError = totalOverestimate;
        }
        check(narrowestError > widestError, "space buys accuracy: " + narrowestError + " vs " + widestError);
        check(widestError < streamTotal / 4,
                "a 1024-wide sketch keeps the total overestimate small: " + widestError);

        // --- epsilon-differential privacy of the released answer ----------------
        int bins = 4000;
        double radius = 60;
        for (double epsilon : new double[]{0.1, 0.5, 1.0, 2.0}) {
            double scale = 1.0 / epsilon;                      // sensitivity of the estimate is 1
            double ratio = LaplaceNoise.worstRatio(scale, bins, radius);
            check(ratio <= Math.exp(epsilon) * (1 + 1e-9),
                    "epsilon = " + epsilon + ": ratio " + ratio + " above e^epsilon");
            check(ratio >= Math.exp(epsilon) * 0.98,
                    "the bound is tight for epsilon = " + epsilon + ": " + ratio);
        }
        check(LaplaceNoise.worstRatio(1.0 / 1.0, bins, radius) > Math.exp(0.5),
                "noise tuned for epsilon = 1 must not pass the epsilon = 0.5 test");

        // --- measured utility of the released counts ----------------------------
        double epsilon = 1.0;
        LaplaceNoise noise = new LaplaceNoise(1.0 / epsilon, 0x5EEDL);
        double errorSum = 0;
        double biasSum = 0;
        int released = 0;
        int insideBound = 0;
        for (int round = 0; round < 400; round++) {
            for (int key = 1; key <= KEYS; key++) {
                double exact = sketch.estimate(key);
                double releasedValue = exact + noise.sample();
                errorSum += (releasedValue - exact) * (releasedValue - exact);
                biasSum += releasedValue - exact;
                released++;
                if (Math.abs(releasedValue - exact) <= 3.0 / epsilon) insideBound++;
            }
        }
        double mse = errorSum / released;
        double bias = biasSum / released;
        check(Math.abs(mse - 2.0 / (epsilon * epsilon)) < 0.05 * 2.0 / (epsilon * epsilon),
                "released mse " + mse + " should be 2/epsilon^2");
        check(Math.abs(bias) < 0.05, "the noise is unbiased: " + bias);
        check(insideBound > 0.9 * released,
                "most released values stay within 3/epsilon of the exact estimate: " + insideBound
                        + "/" + released);

        // --- more budget, less error -------------------------------------------
        double previousMse = Double.MAX_VALUE;
        for (double budget : new double[]{0.25, 0.5, 1.0, 2.0, 4.0}) {
            LaplaceNoise generator = new LaplaceNoise(1.0 / budget, 0xBEEFL);
            double error = 0;
            int count = 20000;
            for (int i = 0; i < count; i++) {
                double value = generator.sample();
                error += value * value;
            }
            double measured = error / count;
            check(measured < previousMse, "epsilon = " + budget + " must beat the previous budget");
            check(Math.abs(measured - 2.0 / (budget * budget)) < 0.06 * 2.0 / (budget * budget),
                    "epsilon = " + budget + ": measured " + measured + " vs theory "
                            + (2.0 / (budget * budget)));
            previousMse = measured;
        }

        // --- rejected parameters and determinism --------------------------------
        check(rejects(() -> new CountMinSketch(0, 4, 1L)), "zero width rejected");
        check(rejects(() -> new CountMinSketch(16, 0, 1L)), "zero depth rejected");
        check(rejects(() -> new CountMinSketch(16, 2, 1L).add(1L, -5L)), "negative weight rejected");
        check(rejects(() -> new LaplaceNoise(0.0, 1L)), "zero noise scale rejected");
        CountMinSketch rebuilt = new CountMinSketch(256, 4, 0xC0FFEEL);
        for (long key : stream) rebuilt.add(key, 1);
        for (int key = 1; key <= KEYS; key++) {
            check(rebuilt.estimate(key) == sketch.estimate(key), "the sketch must be deterministic");
        }

        System.out.println("worst overestimate = " + worstOverestimate + ", released mse at epsilon = "
                + epsilon + " is " + String.format(Locale.ROOT, "%.3f", mse)
                + " against a theory of 2.0");
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
