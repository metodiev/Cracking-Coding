/** Task 716 - Implement quantized neural network inference simulation (8-bit).
 * (Tier: Expert, CTCI Beyond -- Machine Learning & AI)
 *
 * 8-bit inference replaces every float with an affine map q = round(x / scale) + zero_point, so that a
 * tensor is stored as bytes plus one scale (and a zero point for asymmetric ranges). The payoff is that
 * a fully connected layer becomes pure integer arithmetic: multiply the int8 activations by the int8
 * weights into an int32 accumulator, add the bias - which is pre-scaled by the *product* of the input
 * and weight scales so it can join the accumulator directly - and then requantise the accumulator by
 * scale_in * scale_weight / scale_out, rounding and saturating back to int8. ReLU on a symmetric tensor
 * is simply max(0, q).
 *
 * The implementation provides both flavours: symmetric quantisation (scale = max|x| / 127, zero point
 * 0, the usual choice for weights) and asymmetric quantisation (scale = (max - min) / 255 with the zero
 * point chosen so that min maps to qmin, which keeps ranges that do not straddle zero precise). The
 * reference float network is built from exactly the same weights, so every reported error is a genuine
 * quantisation error and not a difference in the model.
 *
 * Verified: per-element round-trip error bounded by half a quantisation step, the extremes of a range
 * mapping to -128/127 (and saturation of out-of-range inputs), the all-zero tensor case, the zero
 * point of an asymmetric tensor mapping to integer zero, the int32 accumulator bound (127 * 127 * terms
 * against 2^31 - 1), the layer output against the float reference, the two-layer classification
 * agreement between the quantised and float networks, and a deterministic run.
 */
import java.util.*;

class Tensor8 {
    final byte[] data;
    final double scale;
    final int zeroPoint;
    final boolean symmetric;

    private Tensor8(byte[] data, double scale, int zeroPoint, boolean symmetric) {
        this.data = data;
        this.scale = scale;
        this.zeroPoint = zeroPoint;
        this.symmetric = symmetric;
    }

    static Tensor8 symmetric(double[] values) {
        double largest = 0;
        for (double value : values) {
            requireFinite(value);
            largest = Math.max(largest, Math.abs(value));
        }
        double scale = largest == 0 ? 1.0 : largest / 127.0;
        byte[] data = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = (byte) clamp(Math.round(values[i] / scale), -127, 127);
        }
        return new Tensor8(data, scale, 0, true);
    }

    static Tensor8 asymmetric(double[] values) {
        double smallest = Double.POSITIVE_INFINITY;
        double largest = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            requireFinite(value);
            smallest = Math.min(smallest, value);
            largest = Math.max(largest, value);
        }
        double range = largest - smallest;
        double scale = range == 0 ? 1.0 : range / 255.0;
        // the offset places the minimum at -128 and the maximum at 127; for a range that does not
        // contain zero the offset itself leaves the int8 interval, which is exactly why asymmetric
        // quantisation is normally stored unsigned - here it is kept exact and stored as an int
        int offset = -128 - (int) Math.round(smallest / scale);
        byte[] data = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            data[i] = (byte) clamp(Math.round(values[i] / scale) + offset, -128, 127);
        }
        return new Tensor8(data, scale, offset, false);
    }

    private static void requireFinite(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException("tensor entries must be finite");
        }
    }

    static int clamp(long value, int low, int high) {
        if (value < low) return low;
        if (value > high) return high;
        return (int) value;
    }

    double[] dequantize() {
        double[] values = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            values[i] = (data[i] - zeroPoint) * scale;
        }
        return values;
    }

    int at(int index) {
        return data[index];
    }

    int size() {
        return data.length;
    }
}

class QuantizedLayer {
    private final Tensor8 weights;
    private final int[] biasInAccumulatorUnits;
    private final int rows;
    private final int columns;

    QuantizedLayer(double[][] denseWeights, double[] bias, double inputScale, double outputScale) {
        rows = denseWeights.length;
        columns = denseWeights[0].length;
        if (bias.length != rows) throw new IllegalArgumentException("bias length must match the rows");
        if (!(inputScale > 0) || !(outputScale > 0)) {
            throw new IllegalArgumentException("scales must be positive");
        }
        double[] flat = new double[rows * columns];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(denseWeights[row], 0, flat, row * columns, columns);
        }
        weights = Tensor8.symmetric(flat);
        biasInAccumulatorUnits = new int[rows];
        for (int row = 0; row < rows; row++) {
            biasInAccumulatorUnits[row] = (int) Math.round(bias[row] / (inputScale * weights.scale));
        }
    }

    double weightScale() {
        return weights.scale;
    }

    /** Integer matmul with an int32 accumulator plus requantisation, optionally followed by ReLU. */
    byte[] forward(int[] activations, double inputScale, double outputScale, boolean relu) {
        if (activations.length != columns) throw new IllegalArgumentException("activation size mismatch");
        byte[] output = new byte[rows];
        for (int row = 0; row < rows; row++) {
            int accumulator = biasInAccumulatorUnits[row];
            for (int column = 0; column < columns; column++) {
                accumulator += weights.at(row * columns + column) * activations[column];
            }
            double scaleFactor = inputScale * weights.scale / outputScale;
            output[row] = (byte) Tensor8.clamp(Math.round(accumulator * scaleFactor), -128, 127);
            if (relu && output[row] < 0) output[row] = 0;
        }
        return output;
    }

    static int[] toIntegers(Tensor8 tensor) {
        int[] integers = new int[tensor.size()];
        for (int i = 0; i < integers.length; i++) integers[i] = tensor.at(i);
        return integers;
    }
}

public class Solution {
    private static long state = 0x7160F00DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double[] vector(int size) {
        double[] values = new double[size];
        for (int i = 0; i < size; i++) values[i] = ((state >>> 11) * 0x1.0p-53) * 2 - 1;
        state = nextLong();
        return values;
    }

    static double[] applyFloat(double[][] weights, double[] bias, double[] input, boolean relu) {
        double[] output = new double[weights.length];
        for (int row = 0; row < weights.length; row++) {
            double sum = bias[row];
            for (int column = 0; column < input.length; column++) sum += weights[row][column] * input[column];
            output[row] = relu ? Math.max(0, sum) : sum;
        }
        return output;
    }

    public static void main(String[] args) {
        // --- symmetric quantisation of a known range ------------------------------
        double[] values = {-1.0, -0.5, 0.0, 0.5, 1.0};
        Tensor8 symmetric = Tensor8.symmetric(values);
        check(symmetric.at(4) == 127, "the largest entry maps to 127");
        check(symmetric.at(0) == -127, "the most negative entry maps to -127");
        check(symmetric.at(2) == 0, "zero maps to zero in a symmetric tensor");
        check(Math.abs(symmetric.scale - 1.0 / 127.0) < 1e-15, "the symmetric scale is max / 127");
        double[] restored = symmetric.dequantize();
        for (int i = 0; i < values.length; i++) {
            check(Math.abs(restored[i] - values[i]) <= symmetric.scale / 2 + 1e-12,
                    "round trip error must be within half a step at " + i);
        }

        // --- asymmetric quantisation and its zero point ---------------------------
        double[] shifted = {2.0, 3.0, 4.0, 5.0, 6.0};
        Tensor8 asymmetric = Tensor8.asymmetric(shifted);
        check(asymmetric.at(0) == -128, "the minimum maps to -128");
        check(asymmetric.at(4) == 127, "the maximum maps to 127");
        check(Math.abs(asymmetric.scale - 4.0 / 255.0) < 1e-15, "the asymmetric scale is range / 255");
        double[] asymRestored = asymmetric.dequantize();
        for (int i = 0; i < shifted.length; i++) {
            check(Math.abs(asymRestored[i] - shifted[i]) <= asymmetric.scale / 2 + 1e-12,
                    "asymmetric round trip at " + i);
        }
        double[] straddlingZero = {-0.25, 0.0, 0.75};
        Tensor8 straddle = Tensor8.asymmetric(straddlingZero);
        check(Math.abs(straddle.dequantize()[1]) <= straddle.scale / 2 + 1e-12,
                "the zero point keeps zero close to zero");

        // --- degenerate tensors ---------------------------------------------------
        Tensor8 zeros = Tensor8.symmetric(new double[]{0, 0, 0});
        check(zeros.scale == 1.0, "an all-zero tensor gets a unit scale instead of dividing by zero");
        for (double value : zeros.dequantize()) {
            check(value == 0.0, "an all-zero tensor dequantises to zeros");
        }
        Tensor8 constant = Tensor8.symmetric(new double[]{7, 7, 7});
        check(constant.at(0) == 127 && constant.at(2) == 127, "a constant tensor saturates at the top");
        check(Math.abs(constant.dequantize()[0] - 7.0) <= constant.scale / 2 + 1e-12,
                "a constant tensor round trips");
        Tensor8 single = Tensor8.asymmetric(new double[]{-3.5});
        check(single.size() == 1 && Math.abs(single.dequantize()[0] + 3.5) <= single.scale / 2 + 1e-12,
                "a single element tensor round trips");

        // --- saturation -----------------------------------------------------------
        Tensor8 saturated = Tensor8.symmetric(new double[]{1.0, 1.5, -2.0});
        check(saturated.at(0) > 0 && saturated.at(1) > saturated.at(0),
                "the scale follows the largest magnitude");
        check(saturated.at(1) <= 127 && saturated.at(2) == -127, "all entries stay inside int8");
        check(Tensor8.clamp(1000, -128, 127) == 127 && Tensor8.clamp(-1000, -128, 127) == -128,
                "clamping saturates both ends");

        // --- the integer accumulator cannot overflow ------------------------------
        int terms = 512;
        long worstAccumulator = 127L * 127L * terms;
        check(worstAccumulator < Integer.MAX_VALUE,
                "the int32 accumulator bound for " + terms + " terms: " + worstAccumulator);
        check(127L * 127L * 200000L > Integer.MAX_VALUE,
                "the bound is real: enough terms would overflow");

        // --- one quantised layer against the float reference ----------------------
        int inputs = 8;
        int outputs = 6;
        double[][] weights = new double[outputs][inputs];
        double[] bias = new double[outputs];
        for (int row = 0; row < outputs; row++) {
            for (int column = 0; column < inputs; column++) weights[row][column] = vector(1)[0] * 1.5;
            bias[row] = vector(1)[0] * 0.5;
        }
        double[] input = vector(inputs);
        Tensor8 quantizedInput = Tensor8.symmetric(input);
        final double inputScale = quantizedInput.scale;
        double[] floatOutput = applyFloat(weights, bias, input, true);
        double rawLayerScale = 0;
        for (double value : floatOutput) rawLayerScale = Math.max(rawLayerScale, Math.abs(value));
        final double layerOutputScale = rawLayerScale == 0 ? 1 : rawLayerScale / 127.0;
        QuantizedLayer layer = new QuantizedLayer(weights, bias, inputScale, layerOutputScale);
        byte[] quantizedOutput = layer.forward(QuantizedLayer.toIntegers(quantizedInput),
                inputScale, layerOutputScale, true);
        double worstLayerError = 0;
        for (int row = 0; row < outputs; row++) {
            double dequantized = quantizedOutput[row] * layerOutputScale;
            worstLayerError = Math.max(worstLayerError,
                    Math.abs(dequantized - floatOutput[row]) / Math.max(1e-6, Math.abs(floatOutput[row])));
        }
        check(worstLayerError < 0.05,
                "quantised layer must stay within a few percent of the float reference: " + worstLayerError);
        for (byte value : quantizedOutput) {
            check(value >= 0, "ReLU on a symmetric tensor keeps everything non-negative");
        }

        // --- two layers: classification agreement ---------------------------------
        int hidden = 16;
        int classes = 3;
        double[][] firstWeights = new double[hidden][inputs];
        double[] firstBias = new double[hidden];
        for (int row = 0; row < hidden; row++) {
            for (int column = 0; column < inputs; column++) firstWeights[row][column] = vector(1)[0];
            firstBias[row] = vector(1)[0] * 0.2;
        }
        double[][] secondWeights = new double[classes][hidden];
        double[] secondBias = new double[classes];
        for (int row = 0; row < classes; row++) {
            for (int column = 0; column < hidden; column++) secondWeights[row][column] = vector(1)[0];
            secondBias[row] = vector(1)[0] * 0.2;
        }
        int agreed = 0;
        int trials = 200;
        double worstAbsoluteError = 0;
        double largestLogit = 0;
        for (int trial = 0; trial < trials; trial++) {
            double[] sample = vector(inputs);
            double sampleScale = 0;
            for (double value : sample) sampleScale = Math.max(sampleScale, Math.abs(value));
            sampleScale = sampleScale / 127.0;
            Tensor8 packed = Tensor8.symmetric(sample);
            double[] hiddenFloat = applyFloat(firstWeights, firstBias, sample, true);
            double hiddenScale = 0;
            for (double value : hiddenFloat) hiddenScale = Math.max(hiddenScale, value);
            hiddenScale = hiddenScale == 0 ? 1 : hiddenScale / 127.0;
            QuantizedLayer firstLayer = new QuantizedLayer(firstWeights, firstBias, sampleScale, hiddenScale);
            byte[] hiddenQuantized = firstLayer.forward(QuantizedLayer.toIntegers(packed), sampleScale,
                    hiddenScale, true);
            double[] outputFloat = applyFloat(secondWeights, secondBias, hiddenFloat, false);
            double outputScale = 0;
            for (double value : outputFloat) outputScale = Math.max(outputScale, Math.abs(value));
            outputScale = outputScale == 0 ? 1 : outputScale / 127.0;
            QuantizedLayer secondLayer = new QuantizedLayer(secondWeights, secondBias, hiddenScale,
                    outputScale);
            int[] hiddenActivations = new int[hiddenQuantized.length];
            for (int index = 0; index < hiddenActivations.length; index++) {
                hiddenActivations[index] = hiddenQuantized[index];
            }
            byte[] outputQuantized = secondLayer.forward(hiddenActivations, hiddenScale, outputScale,
                    false);
            for (int index = 0; index < classes; index++) {
                worstAbsoluteError = Math.max(worstAbsoluteError,
                        Math.abs(outputQuantized[index] * outputScale - outputFloat[index]));
                largestLogit = Math.max(largestLogit, Math.abs(outputFloat[index]));
            }
            int floatBest = 0;
            int quantizedBest = 0;
            for (int index = 1; index < classes; index++) {
                if (outputFloat[index] > outputFloat[floatBest]) floatBest = index;
                if (outputQuantized[index] > outputQuantized[quantizedBest]) quantizedBest = index;
            }
            if (floatBest == quantizedBest) agreed++;
        }
        check(agreed >= 0.9 * trials,
                "the quantised network must classify like the float one: " + agreed + "/" + trials);
        check(worstAbsoluteError <= 0.05 * largestLogit,
                "two layers of quantisation must stay within five percent of the logit scale: "
                        + worstAbsoluteError + " against logits of size " + largestLogit);

        // --- rejection and determinism --------------------------------------------
        check(rejects(() -> Tensor8.symmetric(new double[]{Double.NaN})), "NaN rejected");
        check(rejects(() -> Tensor8.symmetric(new double[]{Double.POSITIVE_INFINITY})), "infinity rejected");
        check(rejects(() -> new QuantizedLayer(new double[][]{{1, 1}}, new double[]{0, 0}, 1, 1)),
                "bias length mismatch rejected");
        check(rejects(() -> layer.forward(new int[inputs + 1], inputScale, layerOutputScale, true)),
                "activation size mismatch rejected");
        Tensor8 again = Tensor8.symmetric(values);
        check(Arrays.equals(again.data, symmetric.data), "quantisation must be deterministic");

        System.out.println("layer error " + String.format(Locale.ROOT, "%.5f", worstLayerError)
                + ", agreement " + agreed + "/" + trials);
        System.out.println("PASS");
    }

    /** Wraps raw int8 activations so the caller does not need a Tensor8 from a float vector. */
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
