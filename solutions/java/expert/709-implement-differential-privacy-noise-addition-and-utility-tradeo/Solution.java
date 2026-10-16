/** Task 709 - Implement differential privacy noise addition and utility tradeoff demo.
 * (Tier: Expert, CTCI Beyond -- Security & Privacy)
 *
 * The Laplace mechanism answers a numeric query f with f(D) + Lap(b), where b = sensitivity / epsilon.
 * Two things are demonstrated and verified here without any cryptography: the *privacy* guarantee, i.e.
 * that for two neighbouring data sets whose answers differ by at most the sensitivity the probability
 * of every output set changes by at most a factor e^epsilon, and the *utility* tradeoff, i.e. that the
 * mean squared error of the released answer is 2 b^2 = 2 sensitivity^2 / epsilon^2.
 *
 * Noise comes from the inverse-CDF sampler of the Laplace distribution, u ~ U(-1/2, 1/2) mapped by
 * x = -b sign(u) ln(1 - 2|u|), driven by a fixed-seed splitmix64 stream so every run is identical.
 * The privacy side is verified *exactly*, not by sampling: for a grid of output intervals the
 * probabilities under the two neighbouring answers are computed from the closed-form CDF, and the
 * largest ratio over the grid is compared against e^epsilon - including the negative control that a
 * smaller noise scale (a larger epsilon than advertised) really does break the bound. Composition is
 * checked the same way on the convolved two-query distribution.
 *
 * Simplified: a single numeric query with a fixed sensitivity (a count, sensitivity 1) and two
 * neighbouring data sets, rather than a general DP framework with arbitrary mechanisms; the Gaussian
 * mechanism, privacy amplification and advanced composition are out of scope.
 */
import java.util.*;

class LaplaceMechanism {
    private final double scale;
    private long state;

    LaplaceMechanism(double scale, long seed) {
        if (!(scale > 0)) throw new IllegalArgumentException("scale must be positive: " + scale);
        this.scale = scale;
        this.state = seed;
    }

    double scale() {
        return scale;
    }

    /** Inverse-CDF sampler; the stream is a deterministic splitmix64. */
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

    static double intervalProbability(double low, double high, double scale) {
        return cdf(high, scale) - cdf(low, scale);
    }

    /** Exact worst-case probability ratio between two answers shifted by `shift`. */
    static double worstRatio(double shift, double scale, int bins, double radius) {
        double worst = 0;
        double width = 2 * radius / bins;
        for (int bin = 0; bin < bins; bin++) {
            double low = -radius + bin * width;
            double high = low + width;
            double numerator = intervalProbability(low, high, scale);
            double denominator = intervalProbability(low - shift, high - shift, scale);
            if (denominator <= 0) continue;
            worst = Math.max(worst, numerator / denominator);
        }
        return worst;
    }
}

public class Solution {
    public static void main(String[] args) {
        // --- the closed form tail ----------------------------------------------
        check(LaplaceMechanism.cdf(0.0, 2.0) == 0.5, "the median is zero");
        check(LaplaceMechanism.cdf(-1e9, 2.0) < 1e-9, "the left tail vanishes");
        check(LaplaceMechanism.cdf(1e9, 2.0) > 1 - 1e-9, "the right tail saturates");
        check(Math.abs(LaplaceMechanism.cdf(-1.0, 2.0) - 0.5 * Math.exp(-0.5)) < 1e-12,
                "cdf below zero");
        check(Math.abs(LaplaceMechanism.cdf(1.0, 2.0) - (1 - 0.5 * Math.exp(-0.5))) < 1e-12,
                "cdf above zero");
        check(LaplaceMechanism.intervalProbability(-1e9, 1e9, 2.0) > 1 - 1e-9, "the law is normalised");

        // --- sampling matches the law ------------------------------------------
        int samples = 200000;
        double scale = 1.5;
        LaplaceMechanism mechanism = new LaplaceMechanism(scale, 0x5EEDL);
        double sum = 0;
        double squareSum = 0;
        int negative = 0;
        int[] histogram = new int[21];
        for (int i = 0; i < samples; i++) {
            double value = mechanism.sample();
            sum += value;
            squareSum += value * value;
            if (value < 0) negative++;
            int bucket = (int) Math.floor((value + 10) / 1.0);
            if (bucket < 0) bucket = 0;
            if (bucket > 20) bucket = 20;
            histogram[bucket]++;
        }
        double mean = sum / samples;
        double variance = squareSum / samples - mean * mean;
        check(Math.abs(mean) < 0.05, "empirical mean is centred: " + mean);
        check(Math.abs(variance - 2 * scale * scale) < 0.05 * 2 * scale * scale,
                "variance " + variance + " should be close to 2 b^2 = " + (2 * scale * scale));
        check(Math.abs(negative / (double) samples - 0.5) < 0.01, "the law is symmetric");
        for (int bucket = 0; bucket <= 20; bucket++) {
            // the outer buckets absorb the clamped tails, so their expectation is unbounded
            double low = bucket == 0 ? -1e9 : -10 + bucket;
            double high = bucket == 20 ? 1e9 : -9 + bucket;
            double expected = LaplaceMechanism.intervalProbability(low, high, scale) * samples;
            check(Math.abs(histogram[bucket] - expected) < 4 * Math.sqrt(Math.max(expected, 1)) + 30,
                    "histogram bucket " + bucket + ": " + histogram[bucket] + " vs " + expected);
        }
        LaplaceMechanism replay = new LaplaceMechanism(scale, 0x5EEDL);
        check(Math.abs(replay.sample() - new LaplaceMechanism(scale, 0x5EEDL).sample()) == 0.0,
                "the generator is deterministic");
        boolean rejected = false;
        try {
            new LaplaceMechanism(0.0, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a non-positive scale must be rejected");

        // --- epsilon-differential privacy, verified exactly --------------------
        int bins = 4000;
        double radius = 60;
        double[] budgets = {0.1, 0.5, 1.0, 2.0, 4.0};
        for (double epsilon : budgets) {
            double noiseScale = 1.0 / epsilon;                 // sensitivity 1, a count query
            double ratio = LaplaceMechanism.worstRatio(1.0, noiseScale, bins, radius);
            check(ratio <= Math.exp(epsilon) * (1 + 1e-9),
                    "epsilon = " + epsilon + " ratio " + ratio + " above e^epsilon");
            check(ratio >= Math.exp(epsilon) * 0.98,
                    "the bound is tight for epsilon = " + epsilon + ": " + ratio);
        }

        // --- negative control: under-noising breaks the promised bound ---------
        double claimed = 0.5;
        double actual = 1.0;
        double noisyScale = 1.0 / actual;                      // too little noise for the claim
        double cheatingRatio = LaplaceMechanism.worstRatio(1.0, noisyScale, bins, radius);
        check(cheatingRatio > Math.exp(claimed),
                "noise tuned for epsilon = " + actual + " must not satisfy epsilon = " + claimed
                        + ": " + cheatingRatio);
        check(cheatingRatio <= Math.exp(actual) * (1 + 1e-9),
                "it does satisfy the epsilon it was actually tuned for");

        // --- utility tradeoff --------------------------------------------------
        double previousError = Double.MAX_VALUE;
        for (double epsilon : budgets) {
            double noiseScale = 1.0 / epsilon;
            LaplaceMechanism generator = new LaplaceMechanism(noiseScale, 0xABCDEFL);
            double error = 0;
            for (int i = 0; i < 100000; i++) {
                double noise = generator.sample();
                error += noise * noise;
            }
            double mse = error / 100000;
            double predicted = 2.0 / (epsilon * epsilon);
            check(Math.abs(mse - predicted) < 0.05 * predicted,
                    "mse " + mse + " should be 2/epsilon^2 = " + predicted);
            check(mse < previousError, "more privacy budget must mean less error");
            previousError = mse;
        }
        check(Math.abs(previousError - 2.0 / 16.0) < 0.05 * 2.0 / 16.0,
                "the cheapest budget still follows 2/epsilon^2");

        // --- composition: the budget adds up and utility pays for it ------------
        double share = 0.5;                                    // per-query budget, two queries
        double perQueryScale = 1.0 / share;
        int gridBins = 60;
        double gridRadius = 40;
        double width = 2 * gridRadius / gridBins;
        double composedMax = 0;
        for (int i = 0; i < gridBins; i++) {
            double lowI = -gridRadius + i * width;
            double numeratorI = LaplaceMechanism.intervalProbability(lowI, lowI + width, perQueryScale);
            double denominatorI =
                    LaplaceMechanism.intervalProbability(lowI - 1, lowI + width - 1, perQueryScale);
            if (denominatorI <= 0) continue;
            for (int j = 0; j < gridBins; j++) {
                double lowJ = -gridRadius + j * width;
                double numeratorJ = LaplaceMechanism.intervalProbability(lowJ, lowJ + width, perQueryScale);
                double denominatorJ =
                        LaplaceMechanism.intervalProbability(lowJ - 1, lowJ + width - 1, perQueryScale);
                if (denominatorJ <= 0) continue;
                composedMax = Math.max(composedMax,
                        (numeratorI * numeratorJ) / (denominatorI * denominatorJ));
            }
        }
        double summedBudget = 2 * share;
        check(composedMax <= Math.exp(summedBudget) * (1 + 1e-9),
                "two queries of epsilon/2 together respect epsilon = " + summedBudget + ": " + composedMax);
        check(composedMax > Math.exp(share),
                "a single epsilon/2 budget would not cover the composition: " + composedMax);
        check(composedMax >= Math.exp(summedBudget) * 0.9,
                "the composed bound is tight: " + composedMax);

        double composedVariance = 2 * perQueryScale * perQueryScale * 2;
        LaplaceMechanism composed = new LaplaceMechanism(perQueryScale, 0x1234L);
        int runs = 200000;
        double composedSum = 0;
        double composedSquare = 0;
        for (int run = 0; run < runs; run++) {
            double value = composed.sample() + composed.sample();
            composedSum += value;
            composedSquare += value * value;
        }
        double composedMean = composedSum / runs;
        double composedMeasured = composedSquare / runs - composedMean * composedMean;
        check(Math.abs(composedMeasured - composedVariance) < 0.05 * composedVariance,
                "composed variance " + composedMeasured + " vs " + composedVariance);
        double singleVariance = 2.0;          // one query with the whole budget epsilon = 1
        check(composedMeasured > singleVariance,
                "composition costs utility: " + composedMeasured + " > " + singleVariance);
        check(Math.abs(composedMean) < 0.05, "the composed noise is still centred: " + composedMean);

        System.out.println("epsilon budget spent = " + summedBudget + ", composed variance = "
                + String.format(Locale.ROOT, "%.3f", composedMeasured)
                + ", single query variance = " + singleVariance);

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
