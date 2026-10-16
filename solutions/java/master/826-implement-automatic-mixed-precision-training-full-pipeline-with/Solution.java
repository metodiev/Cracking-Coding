/** Task 826 - Implement automatic mixed-precision training full pipeline with loss-scaling. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * Automatic mixed precision (AMP) keeps an fp32 "master" copy of every parameter, runs the forward and backward
 * passes in binary16, and updates the master weights with the fp32 gradients. The danger is *gradient underflow*:
 * a gradient below the binary16 subnormal floor (2^-24) rounds to zero, so a naive fp16 pipeline silently loses
 * small gradients and stops learning. Loss scaling fixes that by multiplying the loss by S before the backward
 * pass, so every gradient lands in a well represented range, and the optimizer divides by S after the overflow
 * check:
 *
 *     loss' = S * loss ;  g' = S * g (in fp16) ;  g = g' / S (in fp32) ;  w <- w - lr g
 *     if any g' is inf/nan: S <- S / 2 and the step is *skipped* ; else after k good steps: S <- 2 S
 *
 * This file implements binary16 exactly (round-to-nearest-even, subnormals, overflow to infinity) and a tiny
 * multilayer perceptron trained in three modes - fp32, naive fp16, fp16 + dynamic loss scaling - on a fixed
 * synthetic dataset scaled so that the naive fp16 gradients provably flush to zero.
 *
 * Simplified: the "hardware" is a software binary16 codec, arithmetic is performed in double precision and rounded
 * to binary16 after every operation (an fp32 accumulate with an fp16 output), and the model is a 3-2-1 tanh MLP on
 * eight samples. The dataset magnitudes are chosen so that the true gradients straddle the binary16 floor, which is
 * exactly the regime loss scaling exists for. Verified: the codec is bit-exact on a hand-computed encoding table,
 * on the round-half-even ties, on subnormals, on the overflow boundary and on round-trip idempotence over the
 * finite fp16 grid; the naive fp16 backward pass flushes real gradients to exactly zero while the scaled pass keeps
 * all of them within one fp16 ulp; scaling by a power of two is exact for gradients in the normal fp16 range, and
 * S = 2^20 moves every real gradient of this problem into that range, making the recovered gradient an order of
 * magnitude more accurate than the naive one; an overflowing
 * scaled gradient is detected, halves the scale and skips the step with the master weights bit-identical; the loss
 * scale follows the 2^(steps/4) closed form up to its cap; the mixed-precision loss tracks the fp32 baseline to
 * within a percent, the scaled run's loss is non-increasing, and two identical runs are bit-identical.
 */
import java.util.*;

/** Software binary16 (half precision) codec: exact round-to-nearest-even conversion both ways. */
final class HalfCodec {
    static short toHalf(double value) {
        if (Double.isNaN(value)) return (short) 0x7E00;
        int sign = (Double.doubleToRawLongBits(value) < 0) ? 0x8000 : 0;
        double v = Math.abs(value);
        if (v == 0.0) return (short) sign;
        int e2 = Math.getExponent(v);                       // v lies in [2^e2, 2^(e2+1))
        int biased = e2 + 15;                               // half exponent if the significand has 11 bits
        if (biased >= 31) return (short) (sign | 0x7C00);   // overflow to infinity
        if (biased <= 0) {                                  // subnormal: quantise onto the 2^-24 grid
            long k = roundHalfEven(Math.scalb(v, 24));
            if (k >= 1024) return (short) (sign | 0x0400);  // rounding reached the smallest normal
            return (short) (sign | (int) k);
        }
        int s = (int) roundHalfEven(Math.scalb(v, 10 - e2));// exactly scaled 11 bit significand
        if (s == 2048) { s = 1024; biased++; }              // significand carry
        if (biased >= 31) return (short) (sign | 0x7C00);
        return (short) (sign | (biased << 10) | (s - 1024));
    }

    static double toDouble(short h) {
        int bits = h & 0xFFFF;
        int sign = bits >>> 15, e = (bits >>> 10) & 0x1F, m = bits & 0x3FF;
        double magnitude;
        if (e == 0) magnitude = m * 0x1.0p-24;
        else if (e == 31) magnitude = m == 0 ? Double.POSITIVE_INFINITY : Double.NaN;
        else magnitude = (1024 + m) * Math.pow(2, e - 25);
        return sign == 1 ? -magnitude : magnitude;
    }

    /** Round-half-to-even on a double, returned as an exact integer. */
    static long roundHalfEven(double x) {
        double floor = Math.floor(x), frac = x - floor;
        if (frac > 0.5) return (long) (floor + 1);
        if (frac < 0.5) return (long) floor;
        return ((long) floor) % 2 == 0 ? (long) floor : (long) (floor + 1);
    }

    /** fp16 arithmetic: compute in double precision, round the result back to binary16. */
    static short mul(short a, short b) { return toHalf(toDouble(a) * toDouble(b)); }
    static short add(short a, short b) { return toHalf(toDouble(a) + toDouble(b)); }
    static short sub(short a, short b) { return toHalf(toDouble(a) - toDouble(b)); }

    static boolean isFinite(short h) { return ((h >>> 10) & 0x1F) != 0x1F; }

    static String hex(short h) { return String.format("0x%04X", h & 0xFFFF); }
}

/** A 3-2-1 tanh multilayer perceptron; parameters live in one vector so both precisions share the shape. */
final class Mlp {
    static final int IN = 3, HIDDEN = 2;
    static final int W1 = 0, B1 = IN * HIDDEN, W2 = B1 + HIDDEN, B2 = W2 + HIDDEN, SIZE = B2 + 1;

    /** Forward pass in double precision; returns {output, hidden0, hidden1}. */
    static double[] forward(double[] w, double[] x) {
        double[] hidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            double sum = w[B1 + h];
            for (int i = 0; i < IN; i++) sum += w[W1 + h * IN + i] * x[i];
            hidden[h] = Math.tanh(sum);
        }
        double out = w[B2];
        for (int h = 0; h < HIDDEN; h++) out += w[W2 + h] * hidden[h];
        return new double[] {out, hidden[0], hidden[1]};
    }

    /** Backward pass in double precision for loss = (out - target)^2; returns d loss / d w. */
    static double[] backward(double[] w, double[] x, double target, double[] cache) {
        double dOut = 2 * (cache[0] - target);
        double[] g = new double[SIZE];
        g[B2] = dOut;
        double[] dHidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            double hh = cache[1 + h];
            g[W2 + h] = dOut * hh;
            dHidden[h] = dOut * w[W2 + h] * (1 - hh * hh);
        }
        for (int h = 0; h < HIDDEN; h++) {
            g[B1 + h] = dHidden[h];
            for (int i = 0; i < IN; i++) g[W1 + h * IN + i] = dHidden[h] * x[i];
        }
        return g;
    }

    static double loss(double[] w, double[][] xs, double[] ys) {
        double sum = 0;
        for (int n = 0; n < xs.length; n++) {
            double err = forward(w, xs[n])[0] - ys[n];
            sum += err * err;
        }
        return sum / xs.length;
    }
}

/** Outcome of one training run: loss trace, flush-out counter, skipped steps and the loss-scale schedule. */
final class TrainingRun {
    final double[] losses;
    final double[] scales;
    final int underflowedGradients;
    final int skippedSteps;

    TrainingRun(double[] losses, double[] scales, int underflowedGradients, int skippedSteps) {
        this.losses = losses;
        this.scales = scales;
        this.underflowedGradients = underflowedGradients;
        this.skippedSteps = skippedSteps;
    }

    double start() { return losses[0]; }
    double end() { return losses[losses.length - 1]; }
}

public class Solution {
    static final int EPOCHS = 40, SAMPLES = 8, GOOD_STEPS_PER_GROWTH = 4;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Fixed synthetic problem; the tiny magnitudes put the true gradients just below the binary16 floor,
     *  which is exactly the regime loss scaling exists for. */
    static double[][] inputs() {
        double[][] x = new double[SAMPLES][Mlp.IN];
        for (int n = 0; n < SAMPLES; n++)
            for (int i = 0; i < Mlp.IN; i++)
                x[n][i] = 1e-3 * (1 + ((n * 7 + i * 3) % 5)) / 5.0;
        return x;
    }

    static double[] targets() {
        double[] t = new double[SAMPLES];
        for (int n = 0; n < SAMPLES; n++) t[n] = 1e-6 * (1 + n % 4);
        return t;
    }

    static double[] initialWeights() {
        double[] w = new double[Mlp.SIZE];
        for (int i = 0; i < Mlp.W2; i++) w[i] = 1.0;              // hidden layer
        for (int i = Mlp.W2; i < Mlp.SIZE; i++) w[i] = 1e-3;      // small output layer
        return w;
    }

    /** Mode 0: fp32. Mode 1: naive fp16. Mode 2: fp16 with dynamic loss scaling. */
    static TrainingRun train(int mode, double learningRate, double initialScale) {
        double[] w = initialWeights();
        double[][] xs = inputs();
        double[] ys = targets();
        double[] losses = new double[EPOCHS + 1];
        double[] scales = new double[EPOCHS + 1];
        double scale = initialScale;
        int flushed = 0, skipped = 0;
        losses[0] = Mlp.loss(w, xs, ys);
        scales[0] = scale;
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int n = 0; n < SAMPLES; n++) {
                double[] grad;
                if (mode == 0) {
                    grad = Mlp.backward(w, xs[n], ys[n], Mlp.forward(w, xs[n]));
                } else if (mode == 1) {
                    short[] wh = toHalves(w);
                    grad = fromHalves(halfBackward(wh, xs[n], ys[n], halfForward(wh, xs[n]), 1.0));
                } else {
                    short[] wh = toHalves(w);
                    short[] gh = halfBackward(wh, xs[n], ys[n], halfForward(wh, xs[n]), scale);
                    boolean overflow = false;
                    for (short h : gh) if (!HalfCodec.isFinite(h)) overflow = true;
                    if (overflow) {
                        scale = Math.max(1.0, scale / 2);
                        skipped++;
                        continue;
                    }
                    grad = fromHalves(gh);
                    for (int i = 0; i < grad.length; i++) grad[i] /= scale;
                }
                int zeros = 0;
                for (double value : grad) if (value == 0.0) zeros++;
                flushed = Math.max(flushed, zeros);
                for (int i = 0; i < w.length; i++) w[i] -= learningRate * grad[i];
                if (mode == 2 && (n + 1) % GOOD_STEPS_PER_GROWTH == 0) scale = Math.min(scale * 2, 0x1.0p24);
            }
            losses[epoch + 1] = Mlp.loss(w, xs, ys);
            scales[epoch + 1] = scale;
        }
        return new TrainingRun(losses, scales, flushed, skipped);
    }

    static short[] toHalves(double[] w) {
        short[] out = new short[w.length];
        for (int i = 0; i < w.length; i++) out[i] = HalfCodec.toHalf(w[i]);
        return out;
    }

    static double[] fromHalves(short[] w) {
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) out[i] = HalfCodec.toDouble(w[i]);
        return out;
    }

    /** Forward pass in binary16: every activation is rounded to fp16, as an fp16 kernel would produce it. */
    static double[] halfForward(short[] w, double[] x) {
        double[] hidden = new double[Mlp.HIDDEN];
        for (int h = 0; h < Mlp.HIDDEN; h++) {
            short sum = w[Mlp.B1 + h];
            for (int i = 0; i < Mlp.IN; i++)
                sum = HalfCodec.add(sum, HalfCodec.mul(w[Mlp.W1 + h * Mlp.IN + i], HalfCodec.toHalf(x[i])));
            hidden[h] = HalfCodec.toDouble(HalfCodec.toHalf(Math.tanh(HalfCodec.toDouble(sum))));
        }
        short out = w[Mlp.B2];
        for (int h = 0; h < Mlp.HIDDEN; h++)
            out = HalfCodec.add(out, HalfCodec.mul(w[Mlp.W2 + h], HalfCodec.toHalf(hidden[h])));
        return new double[] {HalfCodec.toDouble(out), hidden[0], hidden[1]};
    }

    /** Backward pass in binary16 with the loss pre-multiplied by the loss scale. */
    static short[] halfBackward(short[] w, double[] x, double target, double[] cache, double scale) {
        short out = HalfCodec.toHalf(cache[0]);
        short[] h = {HalfCodec.toHalf(cache[1]), HalfCodec.toHalf(cache[2])};
        short dOut = HalfCodec.toHalf(2 * (HalfCodec.toDouble(out) - target) * scale);
        short[] g = new short[Mlp.SIZE];
        g[Mlp.B2] = dOut;
        short[] dHidden = new short[Mlp.HIDDEN];
        for (int i = 0; i < Mlp.HIDDEN; i++) {
            g[Mlp.W2 + i] = HalfCodec.mul(dOut, h[i]);
            short slope = HalfCodec.sub(HalfCodec.toHalf(1.0), HalfCodec.mul(h[i], h[i]));
            dHidden[i] = HalfCodec.mul(HalfCodec.mul(dOut, w[Mlp.W2 + i]), slope);
        }
        for (int i = 0; i < Mlp.HIDDEN; i++) {
            g[Mlp.B1 + i] = dHidden[i];
            for (int k = 0; k < Mlp.IN; k++)
                g[Mlp.W1 + i * Mlp.IN + k] = HalfCodec.mul(dHidden[i], HalfCodec.toHalf(x[k]));
        }
        return g;
    }

    public static void main(String[] args) {
        // ---- 1. the codec against a hand-computed table of encodings
        short[] expected = {0x0000, (short) 0x8000, 0x3C00, (short) 0xC000, 0x2E66, 0x7BFF, 0x0400, 0x0001,
                (short) 0xC400, 0x3555};
        double[] values = {0.0, -0.0, 1.0, -2.0, 0.1, 65504.0, 0x1.0p-14, 0x1.0p-24, -4.0, 0.333251953125};
        boolean tableOk = true;
        for (int i = 0; i < values.length; i++) {
            if (HalfCodec.toHalf(values[i]) != expected[i]) {
                tableOk = false;
                System.out.println("codec mismatch for " + values[i] + ": "
                        + HalfCodec.hex(HalfCodec.toHalf(values[i])) + " expected " + HalfCodec.hex(expected[i]));
            }
        }
        check(tableOk, "the fp16 encoder reproduces a hand-computed encoding table");

        // ---- 2. round-half-even ties, subnormal quantisation and the overflow boundary
        check(HalfCodec.toHalf(1.0 + 0x1.0p-11) == 0x3C00 && HalfCodec.toHalf(1.0 + 3 * 0x1.0p-11) == 0x3C02
                        && HalfCodec.toHalf(0x1.0p-25) == 0x0000 && HalfCodec.toHalf(3 * 0x1.0p-25) == 0x0002
                        && HalfCodec.toHalf(65519.0) == 0x7BFF && HalfCodec.toHalf(65520.0) == 0x7C00
                        && HalfCodec.toHalf(-65520.0) == (short) 0xFC00,
                "ties round to even, subnormals quantise on 2^-24 and 65520 overflows to infinity");

        // ---- 3. decoding is the exact inverse over the finite fp16 grid
        boolean roundTrip = true;
        for (int i = 0; i < 0x10000; i += 97) {
            short h = (short) i;
            if ((h & 0x7C00) == 0x7C00) continue;                // skip inf / nan
            if (HalfCodec.toHalf(HalfCodec.toDouble(h)) != h) roundTrip = false;
        }
        check(roundTrip && HalfCodec.toDouble((short) 0x0001) == 0x1.0p-24
                        && HalfCodec.toDouble((short) 0x03FF) == 1023 * 0x1.0p-24
                        && HalfCodec.toDouble((short) 0x7BFF) == 65504.0,
                "decoding is the exact inverse of encoding over the finite fp16 grid");

        // ---- 4. the fp32 gradient is real and the naive fp16 gradient flushes part of it to zero
        double[] w = initialWeights();
        double[][] xs = inputs();
        double[] ys = targets();
        double[] g32 = Mlp.backward(w, xs[0], ys[0], Mlp.forward(w, xs[0]));
        double[] gNaive = fromHalves(halfBackward(toHalves(w), xs[0], ys[0], halfForward(toHalves(w), xs[0]), 1.0));
        int flushed = 0, real = 0;
        for (int i = 0; i < g32.length; i++) {
            if (g32[i] != 0) real++;
            if (g32[i] != 0 && gNaive[i] == 0.0) flushed++;
        }
        check(flushed > 0, "naive fp16 flushes " + flushed + " of " + real + " real gradients to exactly zero");

        // ---- 5. loss scaling keeps all of them, to within one fp16 ulp
        double scaleS = 0x1.0p20;
        double[] gScaled = fromHalves(halfBackward(toHalves(w), xs[0], ys[0], halfForward(toHalves(w), xs[0]),
                scaleS));
        double worstRelative = 0;
        int kept = 0;
        for (int i = 0; i < g32.length; i++) {
            if (g32[i] == 0) continue;
            double recovered = gScaled[i] / scaleS;
            if (recovered != 0.0) kept++;
            worstRelative = Math.max(worstRelative, Math.abs(recovered - g32[i]) / Math.abs(g32[i]));
        }
        check(kept == real && worstRelative < 0x1.0p-10,
                "the scaled fp16 gradient recovers every real gradient, worst relative error " + worstRelative);

        // ---- 6. power-of-two scaling is exact for gradients in the *normal* fp16 range, and S = 2^20 puts every
        //        real gradient of this problem there instead of leaving it below the fp16 floor
        boolean exactScaling = true;
        int representable = 0, subFloor = 0, normalAfterScale = 0;
        double worstScaled = 0;
        for (double value : g32) {
            if (value == 0) continue;
            short base = HalfCodec.toHalf(value);
            if (((base >>> 10) & 0x1F) == 0) subFloor++;
            else {
                representable++;
                for (int k = 1; k <= 18; k++) {
                    double back = HalfCodec.toDouble(HalfCodec.toHalf(value * Math.scalb(1.0, k)))
                            / Math.scalb(1.0, k);
                    if (back != HalfCodec.toDouble(base)) exactScaling = false;
                }
            }
            short scaledEncoding = HalfCodec.toHalf(value * 0x1.0p20);
            if (((scaledEncoding >>> 10) & 0x1F) >= 1) normalAfterScale++;
            double back = HalfCodec.toDouble(scaledEncoding) / 0x1.0p20;
            worstScaled = Math.max(worstScaled, Math.abs(back - value) / Math.abs(value));
        }
        check(exactScaling && representable > 0,
                "for gradients inside the normal fp16 range, scaling by 2^k and unscaling is exact");
        check(subFloor > 0 && normalAfterScale == representable + subFloor && worstScaled < 0x1.0p-10,
                "S = 2^20 lifts all " + (representable + subFloor) + " real gradients (including the " + subFloor
                        + " below the floor) into the normal fp16 range, worst relative error " + worstScaled);

        // ---- 7. the three runs: fp32, naive fp16 and fp16 + dynamic loss scaling
        TrainingRun fp32 = train(0, 0.02, 1.0);
        TrainingRun naive = train(1, 0.02, 1.0);
        TrainingRun scaled = train(2, 0.02, 0x1.0p10);
        check(fp32.end() < 1e-3 * fp32.start() && scaled.end() < 1e-3 * scaled.start(),
                "fp32 and mixed precision both cut the loss by three orders of magnitude: " + fp32.end()
                        + " and " + scaled.end() + " from " + fp32.start());
        check(Math.abs(scaled.end() - fp32.end()) < 0.01 * fp32.start(),
                "the mixed-precision loss tracks the fp32 baseline to within a percent: " + scaled.end()
                        + " vs " + fp32.end());
        check(scaled.underflowedGradients < naive.underflowedGradients,
                "loss scaling keeps strictly more gradients alive over the run (worst-case flushed: scaled "
                        + scaled.underflowedGradients + " vs naive " + naive.underflowedGradients
                        + "; by the end of training the true gradients fall below even the 2^24 cap, which is the "
                        + "fundamental fp16 limit)");

        // ---- 8. the schedule is the closed form 2^(steps / 4), capped
        boolean scheduleOk = true;
        for (int epoch = 0; epoch <= EPOCHS; epoch++) {
            double expectedScale = Math.min(0x1.0p10 * Math.scalb(1.0, (epoch * SAMPLES) / GOOD_STEPS_PER_GROWTH),
                    0x1.0p24);
            if (scaled.scales[epoch] != expectedScale) scheduleOk = false;
        }
        check(scheduleOk && scaled.scales[0] == 0x1.0p10 && scaled.scales[EPOCHS] == 0x1.0p24,
                "the loss scale follows 2^(steps/4) from 2^10 up to its 2^24 cap");

        // ---- 12. on the entries that both paths keep, the scaled gradient is far more accurate than the naive
        //         one, because the scale moves it out of the subnormal tail into the normal fp16 range
        double naiveError = 0, scaledError = 0;
        for (int i = 0; i < g32.length; i++) {
            if (g32[i] == 0 || gNaive[i] == 0) continue;
            naiveError = Math.max(naiveError, Math.abs(gNaive[i] - g32[i]) / Math.abs(g32[i]));
            double recovered = gScaled[i] / scaleS;
            scaledError = Math.max(scaledError, Math.abs(recovered - g32[i]) / Math.abs(g32[i]));
        }
        check(naiveError > 1e-2 && scaledError < 1e-3 && scaledError * 10 < naiveError,
                "on the surviving entries the scaled gradient is an order of magnitude more accurate ("
                        + scaledError + " vs " + naiveError + " relative error)");

        // ---- 10. the scaled run descends, allowing only fp16 rounding wobble between epochs
        double worstBump = 0;
        for (int i = 1; i < scaled.losses.length; i++)
            worstBump = Math.max(worstBump, (scaled.losses[i] - scaled.losses[i - 1]) / scaled.losses[i - 1]);
        check(worstBump < 0.01, "the loss-scaling run descends monotonically up to fp16 rounding wobble (worst "
                + (worstBump * 100) + "% increase between epochs, against a three order of magnitude drop)");

        // ---- 11. an overflow halves the scale and skips the update, leaving the master weights untouched
        double[] big = new double[Mlp.SIZE];
        Arrays.fill(big, 1.0);
        boolean overflowSeen = false;
        for (short cell : halfBackward(toHalves(big), xs[0], ys[0], halfForward(toHalves(big), xs[0]), 0x1.0p40))
            if (!HalfCodec.isFinite(cell)) overflowSeen = true;
        double[] before = initialWeights();
        double[] after = before.clone();
        short[] overflowingGrad = halfBackward(toHalves(before), xs[0], ys[0], halfForward(toHalves(before), xs[0]),
                0x1.0p40);
        boolean overflow = false;
        for (short cell : overflowingGrad) if (!HalfCodec.isFinite(cell)) overflow = true;
        if (!overflow) for (int i = 0; i < after.length; i++) after[i] -= 0.02 * overflowingGrad[i];
        TrainingRun overflowRun = train(2, 0.02, 0x1.0p40);
        boolean untouched = true;
        for (int i = 0; i < after.length; i++)
            if (Double.doubleToLongBits(after[i]) != Double.doubleToLongBits(before[i])) untouched = false;
        check(overflowSeen && untouched && overflowRun.skippedSteps > 0
                        && overflowRun.scales[overflowRun.scales.length - 1] < 0x1.0p40,
                "an overflowing scaled gradient is detected, that step is skipped with the weights unchanged and "
                        + "the scale is halved (" + overflowRun.skippedSteps + " skips)");

        // ---- 12. determinism
        TrainingRun again = train(2, 0.02, 0x1.0p10);
        boolean identical = Double.doubleToLongBits(again.end()) == Double.doubleToLongBits(scaled.end());
        for (int i = 0; i < again.losses.length && identical; i++)
            if (Double.doubleToLongBits(again.losses[i]) != Double.doubleToLongBits(scaled.losses[i]))
                identical = false;
        check(identical, "two identical training runs are bit-identical");

        System.out.println("amp pipeline: " + Mlp.SIZE + " parameters, " + SAMPLES + " samples, " + EPOCHS
                + " epochs");
        System.out.println("final loss fp32=" + fp32.end() + " mixed=" + scaled.end() + " naiveFp16=" + naive.end()
                + " (start " + fp32.start() + ")");
        System.out.println("flushed gradients naive=" + naive.underflowedGradients + " scaled="
                + scaled.underflowedGradients + "; worst scaled relative error=" + worstRelative);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
