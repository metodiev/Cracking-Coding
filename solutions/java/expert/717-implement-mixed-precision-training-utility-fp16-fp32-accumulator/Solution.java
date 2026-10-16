/** Task 717 - Implement mixed-precision training utility (FP16/FP32 accumulator).
 * (Tier: Expert, CTCI Beyond -- Machine Learning & AI)
 *
 * binary16 has 1 sign bit, 5 exponent bits and 10 mantissa bits, so it can hold values up to 65504 and
 * subnormals down to 2^-24 but represents them with about three decimal digits of precision. Two
 * consequences dominate mixed-precision training, and both are implemented and measured here:
 *
 *   - accumulation: summing many FP16 values *in FP16* loses almost everything, because once the
 *     running total grows the step size of the format exceeds each addend and the additions round away.
 *     Keeping the accumulator in FP32 (only the storage in FP16) fixes this, which is the whole reason
 *     hardware accumulates in higher precision.
 *   - gradient underflow: gradients often sit below 2^-24 and would flush to zero in FP16. Loss scaling
 *     multiplies the loss (and hence the gradients) by a large power of two, converts those scaled
 *     gradients to FP16, and divides the scale out again in FP32 before the optimiser step, so no
 *     information is lost. The dynamic variant watches for overflow (infinity after scaling), skips
 *     that step entirely and backs the scale off, growing it again after a run of clean steps.
 *
 * The conversion is exact and round-to-nearest-even, implemented from the raw IEEE bits with
 * Math.rint for the tie case, which the checks verify against the published encodings (1.0 = 0x3C00,
 * 2^-24 = 0x0001, 65504 = 0x7BFF, 65520 = 0x7C00 = infinity) and on the tie values 2049 and 2051.
 *
 * Verified: the encoding table and tie handling, the relative error bound, overflow/underflow/NaN
 * behaviour, the accumulator comparison (FP16 accumulation versus FP32 accumulation of FP16-rounded
 * data), loss scaling recovering a gradient that would otherwise flush to zero, the dynamic scaler
 * schedule and its skipped steps, determinism, and a training loop showing that FP32 master weights
 * with loss-scaled FP16 gradients actually learn while pure FP16 does not.
 */
import java.util.*;

class Half {
    static final short POSITIVE_INFINITY = (short) 0x7C00;
    static final short NEGATIVE_INFINITY = (short) 0xFC00;
    static final short CANONICAL_NAN = (short) 0x7E00;
    static final double MIN_SUBNORMAL = Math.scalb(1.0, -24);
    static final double MIN_NORMAL = Math.scalb(1.0, -14);
    static final double MAX_VALUE = 65504.0;

    /** Round a finite double (or a special) to the nearest binary16 bits, ties to even. */
    static short toHalfBits(double value) {
        if (Double.isNaN(value)) return CANONICAL_NAN;
        long bits = Double.doubleToRawLongBits(value);
        int sign = ((bits >>> 63) & 1L) != 0 ? 0x8000 : 0;
        double magnitude = Math.abs(value);
        if (Double.isInfinite(magnitude) || magnitude >= 65520.0) return (short) (sign | 0x7C00);
        if (magnitude == 0.0) return (short) sign;
        if (magnitude < MIN_SUBNORMAL / 2) return (short) sign;               // rounds to signed zero
        if (magnitude < MIN_NORMAL) {
            // subnormal half: the value is an integer multiple of 2^-24
            long steps = (long) Math.rint(magnitude / MIN_SUBNORMAL);
            if (steps == 0) return (short) sign;
            if (steps >= 1024) return (short) (sign | (1 << 10));             // the smallest normal
            return (short) (sign | (int) steps);
        }
        int exponent = (int) ((bits >>> 52) & 0x7FF) - 1023;
        long significand = (long) Math.rint(magnitude / Math.scalb(1.0, exponent - 10));
        int biasedExponent = exponent + 15;
        if (significand == 2048) {                                            // rounded up a power of two
            significand = 1024;
            biasedExponent++;
        }
        if (biasedExponent >= 31) return (short) (sign | 0x7C00);
        return (short) (sign | (biasedExponent << 10) | (int) (significand - 1024));
    }

    static double fromHalfBits(short bits) {
        int raw = bits & 0xFFFF;
        int sign = (raw >>> 15) & 1;
        int exponent = (raw >>> 10) & 0x1F;
        int mantissa = raw & 0x3FF;
        double magnitude;
        if (exponent == 0x1F) {
            magnitude = mantissa != 0 ? Double.NaN : Double.POSITIVE_INFINITY;
        } else if (exponent == 0) {
            magnitude = mantissa * MIN_SUBNORMAL;
        } else {
            magnitude = (1024 + mantissa) * Math.scalb(1.0, exponent - 25);
        }
        return sign == 1 ? -magnitude : magnitude;
    }

    static double roundToHalf(double value) {
        return fromHalfBits(toHalfBits(value));
    }

    /** Text bits, handy for the encoding checks. */
    static String describe(double value) {
        return String.format("0x%04X", toHalfBits(value) & 0xFFFF);
    }
}

class DynamicLossScaler {
    private double scale;
    private int healthySteps;
    private final int growthInterval;
    private int skipped;
    private int grown;

    DynamicLossScaler(double initialScale, int growthInterval) {
        if (!(initialScale > 0)) throw new IllegalArgumentException("scale must be positive");
        if (growthInterval < 1) throw new IllegalArgumentException("growth interval must be positive");
        this.scale = initialScale;
        this.growthInterval = growthInterval;
    }

    double scale() {
        return scale;
    }

    int skippedSteps() {
        return skipped;
    }

    int growths() {
        return grown;
    }

    /** Returns true when the step may be applied; false means it was skipped and the scale halved. */
    boolean step(boolean overflowed) {
        if (overflowed) {
            scale = Math.max(scale / 2, 1.0);
            healthySteps = 0;
            skipped++;
            return false;
        }
        healthySteps++;
        if (healthySteps >= growthInterval) {
            scale = Math.min(scale * 2, Math.scalb(1.0, 24));
            healthySteps = 0;
            grown++;
        }
        return true;
    }
}

/** Linear regression trained with loss-scaled FP16 gradients and FP32 master weights. */
class MixedPrecisionTrainer {
    private final double[][] features;
    private final double[] targets;
    private double[] masterWeights;
    private final double learningRate;
    private final DynamicLossScaler scaler;
    private final boolean dynamicScaling;
    private final double fixedScale;

    MixedPrecisionTrainer(double[][] features, double[] targets, double learningRate,
                          double initialScale, boolean dynamicScaling) {
        this.features = features;
        this.targets = targets;
        this.learningRate = learningRate;
        this.masterWeights = new double[features[0].length];
        this.dynamicScaling = dynamicScaling;
        this.fixedScale = initialScale;
        this.scaler = new DynamicLossScaler(initialScale, 5);
    }

    double[] weights() {
        return masterWeights.clone();
    }

    DynamicLossScaler scaler() {
        return scaler;
    }

    double loss() {
        double total = 0;
        for (int sample = 0; sample < features.length; sample++) {
            double prediction = 0;
            for (int k = 0; k < masterWeights.length; k++) prediction += masterWeights[k] * features[sample][k];
            double error = prediction - targets[sample];
            total += error * error;
        }
        return total / features.length;
    }

    /** A step whose gradients are gradients * scale, rounded to FP16 before use. */
    private double[] scaledGradient(double scale) {
        double[] gradient = new double[masterWeights.length];
        for (int sample = 0; sample < features.length; sample++) {
            double prediction = 0;
            for (int k = 0; k < masterWeights.length; k++) prediction += masterWeights[k] * features[sample][k];
            double error = 2 * (prediction - targets[sample]) / features.length;
            for (int k = 0; k < masterWeights.length; k++) gradient[k] += error * features[sample][k];
        }
        for (int k = 0; k < gradient.length; k++) {
            gradient[k] = Half.roundToHalf(gradient[k] * scale);
        }
        return gradient;
    }

    /** @param simulateOverflow forces the overflow path so the skip behaviour can be checked. */
    boolean step(boolean simulateOverflow) {
        double scale = dynamicScaling ? scaler.scale() : fixedScale;
        double[] scaled = scaledGradient(scale);
        boolean overflowed = simulateOverflow;
        for (double value : scaled) {
            if (Double.isNaN(value) || Double.isInfinite(value)) overflowed = true;
        }
        if (dynamicScaling) {
            if (!scaler.step(overflowed)) return false;
        } else if (overflowed) {
            return false;                       // even a static scale skips an overflowing step
        }
        for (int k = 0; k < masterWeights.length; k++) {
            masterWeights[k] -= learningRate * (scaled[k] / scale);
        }
        return true;
    }

    /** Magnitude of the unscaled gradient at the current master weights. */
    double rawGradientMagnitude() {
        double[] gradient = scaledGradient(1.0);
        double total = 0;
        for (double value : gradient) total += Math.abs(value);
        return total;
    }

    /** The FP16 snapshot of the master weights, as a Cast operator would produce. */
    double[] halfWeights() {
        double[] half = new double[masterWeights.length];
        for (int k = 0; k < masterWeights.length; k++) half[k] = Half.roundToHalf(masterWeights[k]);
        return half;
    }
}

public class Solution {
    private static long state = 0x7171707L;

    static double nextDouble() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (z >>> 11) * 0x1.0p-53;
    }

    public static void main(String[] args) {
        // --- the published encodings ---------------------------------------------
        check(Half.toHalfBits(0.0) == 0x0000, "zero is 0x0000");
        check(Short.toUnsignedInt(Half.toHalfBits(-0.0)) == 0x8000, "negative zero keeps its sign");
        check(Half.toHalfBits(1.0) == 0x3C00, "one is 0x3C00, got " + Half.describe(1.0));
        check(Half.toHalfBits(2.0) == 0x4000, "two is 0x4000");
        check(Half.toHalfBits(0.5) == 0x3800, "a half is 0x3800");
        check(Short.toUnsignedInt(Half.toHalfBits(-1.0)) == 0xBC00, "minus one is 0xBC00");
        check(Half.toHalfBits(Half.MIN_NORMAL) == 0x0400, "2^-14 is the smallest normal");
        check(Half.toHalfBits(Half.MIN_SUBNORMAL) == 0x0001, "2^-24 is the smallest subnormal");
        check(Half.toHalfBits(Half.MAX_VALUE) == 0x7BFF, "65504 is the largest finite value");
        check(Half.toHalfBits(65520.0) == Half.POSITIVE_INFINITY, "65520 rounds up to infinity");
        check(Half.toHalfBits(-1e300) == Half.NEGATIVE_INFINITY, "a huge negative overflows");
        check(Half.toHalfBits(1e-20) == 0x0000, "a tiny positive flushes to zero");
        check(Short.toUnsignedInt(Half.toHalfBits(-1e-20)) == 0x8000, "a tiny negative flushes to signed zero");
        check(Short.toUnsignedInt(Half.toHalfBits(Double.NaN)) == 0x7E00, "NaN has a canonical encoding");

        // --- round trips and ties --------------------------------------------------
        check(Half.fromHalfBits((short) 0x3C00) == 1.0, "decoding one");
        check(Half.fromHalfBits(Half.POSITIVE_INFINITY) == Double.POSITIVE_INFINITY, "decoding infinity");
        check(Double.isNaN(Half.fromHalfBits(Half.CANONICAL_NAN)), "decoding NaN");
        check(Half.fromHalfBits((short) 0x0001) == Half.MIN_SUBNORMAL, "decoding the smallest subnormal");
        check(Half.roundToHalf(2049.0) == 2048.0, "2049 is a tie that rounds to even");
        check(Half.roundToHalf(2051.0) == 2052.0, "2051 is a tie that rounds to even");
        check(Half.roundToHalf(1.0 + Math.scalb(1.0, -11)) == 1.0, "the tie above one rounds down");
        check(Half.roundToHalf(1.0 + Math.scalb(1.0, -11) + Math.scalb(1.0, -40)) == 1.0
                        + Math.scalb(1.0, -10),
                "just above the tie rounds up");
        for (double value : new double[]{0.1, 1.0 / 3, 1234.5678, -7.25, 0.0001, 54321.0}) {
            double rounded = Half.roundToHalf(value);
            check(Math.abs(rounded - value) <= Math.abs(value) * Math.scalb(1.0, -11) + Half.MIN_SUBNORMAL,
                    "relative error bound for " + value + ": " + rounded);
            check(Half.roundToHalf(rounded) == rounded, "rounding must be idempotent for " + value);
        }

        // --- accumulation: FP16 versus an FP32 accumulator -------------------------
        int terms = 10000;
        double exact = terms / 3.0;
        double naive = 0;
        double stored = 0;
        double accumulator = 0;
        for (int i = 0; i < terms; i++) {
            double addend = Half.roundToHalf(1.0 / 3);
            naive = Half.roundToHalf(naive + addend);
            stored += addend;
            accumulator += 1.0 / 3;
        }
        double naiveError = Math.abs(naive - exact) / exact;
        double storedError = Math.abs(stored - exact) / exact;
        double accumulatorError = Math.abs(accumulator - exact) / exact;
        check(Math.abs(stored - 3332.5195) < 0.5, "storage rounding shifts the sum slightly: " + stored);
        check(storedError < 1e-3, "FP16 storage with an exact accumulator stays accurate: " + storedError);
        check(accumulatorError < 1e-10, "a double accumulator stays accurate to a part in a billion: "
                + accumulatorError);
        check(naive < exact, "FP16 accumulation loses value once the running total grows: " + naive);
        check(naiveError > 0.05, "the naive FP16 sum is badly wrong: " + naiveError);
        check(naiveError > 100 * storedError, "accumulator precision is the difference: " + naiveError
                + " vs " + storedError);

        // --- loss scaling recovers a gradient that would flush to zero -----------
        double tinyGradient = 1e-9;
        double flushed = Half.roundToHalf(tinyGradient);
        check(flushed == 0.0, "a gradient of 1e-9 flushes to zero in FP16");
        double lossScale = Math.scalb(1.0, 16);
        double scaled = Half.roundToHalf(tinyGradient * lossScale);
        check(scaled > 0, "the scaled gradient survives FP16");
        double recovered = scaled / lossScale;
        double recoveredError = Math.abs(recovered - tinyGradient) / tinyGradient;
        check(recoveredError < 1e-3, "unscaled gradient error must be small: " + recoveredError);
        check(scaled / lossScale > 0.5 * tinyGradient, "the recovered gradient keeps its magnitude");
        check(Half.roundToHalf(Math.scalb(1.0, 20) * 1e-9) > 0, "a larger scale helps even more");

        // --- the dynamic loss scaler ----------------------------------------------
        DynamicLossScaler scaler = new DynamicLossScaler(1024, 3);
        check(scaler.scale() == 1024, "the initial scale is honoured");
        check(scaler.step(false) && scaler.scale() == 1024, "one healthy step keeps the scale");
        check(scaler.step(false) && scaler.scale() == 1024, "two healthy steps keep the scale");
        check(scaler.step(false) && scaler.scale() == 2048, "the third healthy step doubles the scale");
        check(!scaler.step(true) && scaler.scale() == 1024, "an overflow halves the scale");
        check(scaler.skippedSteps() == 1, "the overflow was counted");
        check(scaler.step(false) && scaler.step(false) && !scaler.step(true) && scaler.scale() == 512,
                "the healthy-step counter resets after an overflow");
        check(rejects(() -> new DynamicLossScaler(0, 4)), "zero initial scale rejected");
        check(rejects(() -> new DynamicLossScaler(8, 0)), "zero growth interval rejected");

        // --- a training loop: FP16 alone stalls, mixed precision learns -----------
        // Features are tiny (1e-4) so the gradients land around 1e-9, which is below the smallest
        // binary16 subnormal: they flush to zero unless the loss is scaled first.
        int samples = 24;
        int dimensions = 3;
        double[] truth = {1.0, -0.5, 2.0};
        double[][] features = new double[samples][dimensions];
        double[] targets = new double[samples];
        for (int sample = 0; sample < samples; sample++) {
            double target = 0;
            for (int k = 0; k < dimensions; k++) {
                features[sample][k] = nextDouble() * 1e-4;
                target += truth[k] * features[sample][k];
            }
            targets[sample] = target;
        }
        double learningRate = 2e7;
        MixedPrecisionTrainer mixed = new MixedPrecisionTrainer(features, targets, learningRate, 1024, true);
        MixedPrecisionTrainer plainHalf = new MixedPrecisionTrainer(features, targets, learningRate, 1.0, false);
        double initialLoss = mixed.loss();
        check(Math.abs(plainHalf.loss() - initialLoss) < 1e-30, "both trainers start from zero weights");
        check(plainHalf.rawGradientMagnitude() < Half.MIN_SUBNORMAL / 2,
                "the raw gradient sits below the smallest binary16 subnormal: "
                        + plainHalf.rawGradientMagnitude());
        for (int step = 0; step < 200; step++) {
            mixed.step(false);
            plainHalf.step(false);
        }
        double mixedLoss = mixed.loss();
        double halfLoss = plainHalf.loss();
        check(mixedLoss < 0.5 * initialLoss,
                "loss scaling must make progress: " + mixedLoss + " vs " + initialLoss);
        check(Arrays.equals(plainHalf.weights(), new double[dimensions]),
                "pure FP16 gradients flush to zero, so the weights never move: "
                        + Arrays.toString(plainHalf.weights()) + " raw gradient "
                        + plainHalf.rawGradientMagnitude());
        check(halfLoss == initialLoss, "the unscaled trainer makes no progress at all: " + halfLoss);
        check(mixed.scaler().scale() > 1024, "the scaler grew during healthy training");

        // --- an overflow skips the step and halves the scale ----------------------
        MixedPrecisionTrainer skipped = new MixedPrecisionTrainer(features, targets, learningRate, 1024, true);
        skipped.step(false);
        double[] before = skipped.weights();
        double scaleBefore = skipped.scaler().scale();
        check(!skipped.step(true), "an overflowing step reports that it was skipped");
        check(Arrays.equals(before, skipped.weights()), "a skipped step must not touch the weights");
        check(skipped.scaler().scale() == scaleBefore / 2, "a skipped step halves the scale");
        check(skipped.scaler().skippedSteps() == 1, "the skip is counted");

        // --- master weights versus FP16 weights -----------------------------------
        double[] master = mixed.weights();
        double[] halved = mixed.halfWeights();
        for (int k = 0; k < master.length; k++) {
            check(Math.abs(halved[k] - master[k])
                            <= Math.abs(master[k]) * Math.scalb(1.0, -11) + Half.MIN_SUBNORMAL,
                    "the FP16 snapshot of the master weights stays within a relative 2^-11");
        }
        check(Math.abs(master[0] - truth[0]) < 0.05,
                "the mixed precision trainer found the first weight: " + master[0]);
        check(Math.abs(master[2] - truth[2]) < 0.05,
                "the mixed precision trainer found the third weight: " + master[2]);

        // --- rejection and determinism --------------------------------------------
        check(rejects(() -> new MixedPrecisionTrainer(features, targets, 0.5, 0.0, true)),
                "zero loss scale rejected");
        MixedPrecisionTrainer first = new MixedPrecisionTrainer(features, targets, learningRate, 1024, true);
        MixedPrecisionTrainer second = new MixedPrecisionTrainer(features, targets, learningRate, 1024, true);
        for (int step = 0; step < 50; step++) {
            first.step(false);
            second.step(false);
        }
        check(Arrays.equals(first.weights(), second.weights()), "training must be deterministic");

        System.out.println("naive FP16 sum " + String.format(Locale.ROOT, "%.3f", naive) + " vs exact "
                + String.format(Locale.ROOT, "%.3f", exact) + " vs stored-FP16/FP32-accumulator "
                + String.format(Locale.ROOT, "%.3f", stored));
        System.out.println("loss " + String.format(Locale.ROOT, "%.3e", initialLoss) + " -> mixed "
                + String.format(Locale.ROOT, "%.3e", mixedLoss) + ", pure FP16 "
                + String.format(Locale.ROOT, "%.3e", halfLoss));
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
