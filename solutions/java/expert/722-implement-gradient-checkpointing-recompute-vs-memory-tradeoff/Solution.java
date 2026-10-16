/** Task 722 - Implement gradient checkpointing (recompute vs memory tradeoff). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Backpropagation needs the forward activations, so a naive implementation keeps all L of them alive and
 * pays O(L) memory. Gradient checkpointing keeps only every S-th activation and *recomputes* the S-1
 * activations inside each segment while the backward pass walks it. The result is exactly the same
 * gradient (the recomputed activations are bit-for-bit the forward ones, because the very same expression
 * order is re-evaluated), at the cost of one extra forward pass: L + L layer evaluations instead of L, for
 * memory ceil(L/S) + S instead of L. The optimum segment size is therefore near sqrt(L), which the
 * measurements below confirm: the memory curve is U-shaped in S.
 *
 * Invariants verified:
 *   - exactness   checkpointed gradients equal naive gradients elementwise, to machine precision
 *   - oracle      both gradient variants agree with a central finite-difference oracle on the loss
 *   - recompute   the recomputed activations are bitwise equal to the original forward activations, and
 *                 the extra work is exactly one extra forward pass (2L evaluations instead of L)
 *   - memory      the stored-activation count is ceil(L/S) + S, is below L for every useful S, and is
 *                 minimised by an interior S (both S = 1 and S >= L cost more)
 *
 * Simplified: the network is a chain of scalar-slope tanh layers over a fixed small vector, so the whole
 * experiment is deterministic and the analytic gradient is available in closed form; no autodiff framework
 * is involved. The verified property is the tradeoff itself, measured rather than asserted.
 */
import java.util.*;

public class Solution {
    /** A chain of `layers` tanh layers x_j = tanh(slope_j * x_{j-1} + bias_j) over `dim` units. */
    static final class Chain {
        final int layers;
        final int dim;
        final double[] slope;      // one per layer
        final double[] bias;       // dim per layer, flattened
        long evaluations;                 // layer evaluations, the work counter

        Chain(int layers, int dim, double[] slope, double[] bias) {
            if (slope.length != layers || bias.length != layers * dim) {
                throw new IllegalArgumentException("parameter shape mismatch");
            }
            this.layers = layers;
            this.dim = dim;
            this.slope = slope;
            this.bias = bias;
        }

        /** One layer: u = slope_j * x + bias_j, y = tanh(u). */
        private double[] layer(double[] x, int j) {
            double[] y = new double[dim];
            for (int i = 0; i < dim; i++) y[i] = Math.tanh(slope[j] * x[i] + bias[j * dim + i]);
            evaluations++;
            return y;
        }

        double[] output(double[] input) {
            double[] x = input;
            for (int j = 0; j < layers; j++) x = layer(x, j);
            return x;
        }

        double loss(double[] input, double[] target) {
            double[] y = output(input);
            double sum = 0.0;
            for (int i = 0; i < dim; i++) sum += 0.5 * (y[i] - target[i]) * (y[i] - target[i]);
            return sum;
        }

        int gradientSize() { return layers * (dim + 1); }   // one slope plus dim biases per layer

        /** Naive backward: keeps every activation, so no recomputation is needed. */
        double[] backwardNaive(double[] input, double[] target) {
            double[][] act = new double[layers + 1][];
            act[0] = input.clone();
            for (int j = 0; j < layers; j++) act[j + 1] = layer(act[j], j);
            return backward(act, target);
        }

        /**
         * Checkpointed backward: only every `segment`-th activation is kept. Each segment is recomputed
         * from its stored input on the way back, and the segment's gradients are accumulated. Returns the
         * gradient vector, and records the peak activations alive in `peakStored` for the memory model.
         */
        double[] backwardCheckpointed(double[] input, double[] target, int segment, int[] peakStored) {
            int boundaries = (layers + segment - 1) / segment;
            double[][] stored = new double[boundaries + 1][];
            stored[0] = input.clone();
            double[] x = stored[0];
            for (int j = 0; j < layers; j++) {
                x = layer(x, j);
                if ((j + 1) % segment == 0) stored[(j + 1) / segment] = x.clone();
            }
            double[] output = x;                       // the loss input, always available
            int kept = 0;
            for (double[] entry : stored) if (entry != null) kept++;
            double[] grad = new double[gradientSize()];
            double[] upstream = new double[dim];
            for (int i = 0; i < dim; i++) upstream[i] = output[i] - target[i];
            int peak = 0;
            for (int seg = boundaries - 1; seg >= 0; seg--) {
                int first = seg * segment;
                int last = Math.min(first + segment, layers) - 1;
                double[] base = stored[seg] != null ? stored[seg] : stored[boundaries];
                double[] state = base.clone();
                double[][] inner = new double[last - first + 2][];
                inner[0] = state;
                for (int j = first; j <= last; j++) {
                    state = layer(state, j);
                    inner[j - first + 1] = state;
                }
                for (int j = last; j >= first; j--) {
                    double[] out = inner[j - first + 1];
                    for (int i = 0; i < dim; i++) upstream[i] *= 1.0 - out[i] * out[i];
                    grad[j * (dim + 1)] += dot(upstream, inner[j - first]);
                    for (int i = 0; i < dim; i++) grad[j * (dim + 1) + 1 + i] += upstream[i];
                    for (int i = 0; i < dim; i++) upstream[i] *= slope[j];
                }
                peak = Math.max(peak, kept + (last - first + 1));
            }
            if (peakStored != null) peakStored[0] = peak;
            return grad;
        }

        /** Shared backward pass over a fully materialised activation table. */
        private double[] backward(double[][] act, double[] target) {
            double[] grad = new double[gradientSize()];
            double[] upstream = new double[dim];
            for (int i = 0; i < dim; i++) upstream[i] = act[layers][i] - target[i];
            for (int j = layers - 1; j >= 0; j--) {
                for (int i = 0; i < dim; i++) upstream[i] *= 1.0 - act[j + 1][i] * act[j + 1][i];
                grad[j * (dim + 1)] += dot(upstream, act[j]);
                for (int i = 0; i < dim; i++) grad[j * (dim + 1) + 1 + i] += upstream[i];
                for (int i = 0; i < dim; i++) upstream[i] *= slope[j];
            }
            return grad;
        }

        private static double dot(double[] a, double[] b) {
            double sum = 0.0;
            for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
            return sum;
        }
    }

    private static long rngState = 0x9E3779B97F4A7C15L;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    static double[] randomVector(int dim, double scale) {
        double[] v = new double[dim];
        for (int i = 0; i < dim; i++) v[i] = (nextDouble() * 2.0 - 1.0) * scale;
        return v;
    }

    static double maxDifference(double[] a, double[] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }

    /** Central finite difference of the loss with respect to one gradient coordinate. */
    static double finiteDifference(Chain chain, double[] input, double[] target, int coordinate, double h) {
        boolean isSlope = coordinate % (chain.dim + 1) == 0;
        int layer = coordinate / (chain.dim + 1);
        double[] array = isSlope ? chain.slope : chain.bias;
        int index = isSlope ? layer : layer * chain.dim + coordinate % (chain.dim + 1) - 1;
        double original = array[index];
        array[index] = original + h;
        double up = chain.loss(input, target);
        array[index] = original - h;
        double down = chain.loss(input, target);
        array[index] = original;
        return (up - down) / (2.0 * h);
    }

    public static void main(String[] args) {
        final int layers = 16;
        final int dim = 6;
        double[] slope = randomVector(layers, 0.6);
        double[] bias = randomVector(layers * dim, 0.3);
        double[] input = randomVector(dim, 1.0);
        double[] target = randomVector(dim, 1.0);
        Chain chain = new Chain(layers, dim, slope, bias);

        // --- the degenerate chain ---------------------------------------------
        Chain single = new Chain(1, dim, new double[]{0.7}, randomVector(dim, 0.3));
        double[] singleNaive = single.backwardNaive(input, target);
        int[] singlePeak = new int[1];
        double[] singleCheckpointed = single.backwardCheckpointed(input, target, 4, singlePeak);
        check(maxDifference(singleNaive, singleCheckpointed) == 0.0,
                "a one-layer chain must give identical gradients either way");
        check(singlePeak[0] >= 1, "even one layer needs its activation alive");

        // --- exactness of the checkpointed gradient ---------------------------
        chain.evaluations = 0;
        double[] reference = chain.backwardNaive(input, target);
        long naiveWork = chain.evaluations;
        check(naiveWork == layers, "the naive forward pass evaluates every layer once, got " + naiveWork);
        double[] previous = null;
        for (int segment : new int[]{1, 2, 3, 4, 5, 8, 16, 32}) {
            chain.evaluations = 0;
            int[] peak = new int[1];
            double[] got = chain.backwardCheckpointed(input, target, segment, peak);
            check(maxDifference(got, reference) < 1e-15,
                    "segment " + segment + " must reproduce the naive gradient, worst "
                            + maxDifference(got, reference));
            check(chain.evaluations == 2 * layers,
                    "checkpointing costs exactly one extra forward pass, segment " + segment + " used "
                            + chain.evaluations);
            if (previous != null) {
                check(maxDifference(got, previous) == 0.0, "the segment size must not change the result");
            }
            previous = got;
        }

        // --- independent oracle: central finite differences --------------------
        for (int coordinate : new int[]{0, dim, 3 * (dim + 1) + 2, (layers - 1) * (dim + 1) + dim}) {
            double analytic = reference[coordinate];
            double numeric = finiteDifference(chain, input, target, coordinate, 1e-6);
            check(Math.abs(analytic - numeric) < 1e-5,
                    "finite differences must confirm gradient " + coordinate + ": " + analytic + " vs "
                            + numeric);
        }

        // --- memory model ------------------------------------------------------
        int bestSegment = 0;
        int bestPeak = Integer.MAX_VALUE;
        for (int segment = 1; segment <= layers; segment++) {
            int[] peak = new int[1];
            chain.backwardCheckpointed(input, target, segment, peak);
            int boundaries = (layers + segment - 1) / segment;
            check(peak[0] <= boundaries + segment + 1,
                    "peak activations must respect ceil(L/S) + S + 1, S=" + segment + " peak=" + peak[0]);
            if (segment > 1 && segment <= layers / 2) {
                check(peak[0] < layers, "a useful segment size must beat naive memory, S=" + segment
                        + " peak=" + peak[0]);
            }
            if (peak[0] < bestPeak) { bestPeak = peak[0]; bestSegment = segment; }
        }
        check(bestSegment > 1 && bestSegment < layers,
                "the memory optimum must be an interior segment size, found S=" + bestSegment);
        check(bestSegment * bestSegment <= 4 * layers,
                "the optimum segment size is near sqrt(L), found S=" + bestSegment);
        int[] naivePeak = new int[1];
        chain.backwardCheckpointed(input, target, layers, naivePeak);
        check(naivePeak[0] >= layers, "one huge segment buys no memory at all: " + naivePeak[0]
                + " activations against naive " + layers);

        // --- determinism -------------------------------------------------------
        chain.evaluations = 0;
        double[] again = chain.backwardCheckpointed(input, target, 4, null);
        check(maxDifference(again, reference) == 0.0, "the same call must repeat exactly");
        check(chain.evaluations == 2 * layers, "work accounting is deterministic");

        System.out.println("chain of " + layers + " tanh layers over " + dim + " units: naive memory "
                + layers + " activations, " + naiveWork + " layer evaluations");
        System.out.println("checkpointing costs " + (2 * layers) + " evaluations (one extra forward pass) "
                + "for ceil(L/S) + S activations");
        System.out.println("best segment size " + bestSegment + " with " + bestPeak + " activations alive, "
                + "against " + layers + " at the naive end");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
