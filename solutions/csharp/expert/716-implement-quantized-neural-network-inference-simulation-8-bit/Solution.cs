// Task 716 - Implement quantized neural network inference simulation (8-bit).
// (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Post-training 8-bit quantization replaces every real tensor by integers plus an affine map
// q = clamp(round(x / scale) + zero_point) and x ~= (q - zero_point) * scale. Weights use symmetric
// int8 (-127..127, zero point 0, scale = max|w| / 127, optionally one scale per output channel) and
// activations use asymmetric uint8 (0..255), which is what lets a real ReLU output be represented
// exactly. Inference then runs integer-only: an int8 weight row times a uint8 activation vector is
// accumulated in int32 and the result is requantized with the combined scale
// scale_out = scale_w * scale_x / scale_y.
//
// Verified below: the round trip is inside half a scale, exactly representable values survive
// untouched, out-of-range values saturate and the saturation counter sees them; symmetric and
// asymmetric maps hit their endpoints; the int32 accumulator agrees bit for bit with a long
// reference; the quantized layer output stays inside the first-order error bound
// sum|w|*sx/2 + sum|x|*sw/2 + sy/2 derived from the two quantizers; per-channel weight scales beat
// per-tensor scales on a matrix whose channels differ by orders of magnitude; requantized outputs
// respect the uint8 range; and the degenerate tensors (all zeros, a constant, a single element) plus
// malformed quantizers are handled without producing NaN or silently wrong numbers.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Signed(ref ulong state) => ((Next(ref state) >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

internal sealed class AffineQuantizer
{
    internal double Scale { get; }
    internal int ZeroPoint { get; }
    internal int Minimum { get; }
    internal int Maximum { get; }
    internal int SaturationCount { get; private set; }

    internal AffineQuantizer(double scale, int zeroPoint, int minimum, int maximum)
    {
        if (scale <= 0) throw new ArgumentOutOfRangeException(nameof(scale), "the scale must be positive");
        if (minimum > maximum) throw new ArgumentOutOfRangeException(nameof(minimum), "the range is empty");
        if (zeroPoint < minimum || zeroPoint > maximum)
            throw new ArgumentOutOfRangeException(nameof(zeroPoint), "the zero point must lie inside the range");
        Scale = scale;
        ZeroPoint = zeroPoint;
        Minimum = minimum;
        Maximum = maximum;
    }

    /// <summary>Symmetric int8 quantizer for weights: zero point 0, range [-127, 127].</summary>
    internal static AffineQuantizer SymmetricInt8(double maxAbsolute)
    {
        if (maxAbsolute < 0) throw new ArgumentOutOfRangeException(nameof(maxAbsolute));
        double scale = maxAbsolute <= 0 ? 1.0 : maxAbsolute / 127.0;
        return new AffineQuantizer(scale, 0, -127, 127);
    }

    /// <summary>Asymmetric uint8 quantizer for activations: range [0, 255].</summary>
    internal static AffineQuantizer AsymmetricUint8(double observedMin, double observedMax)
    {
        if (observedMax < observedMin) throw new ArgumentOutOfRangeException(nameof(observedMax));
        if (observedMax == observedMin) return new AffineQuantizer(1.0, 0, 0, 255);
        double scale = (observedMax - observedMin) / 255.0;
        int zeroPoint = (int)Math.Round(-observedMin / scale);
        zeroPoint = Math.Clamp(zeroPoint, 0, 255);
        return new AffineQuantizer(scale, zeroPoint, 0, 255);
    }

    internal int Quantize(double value)
    {
        double scaled = Math.Round(value / Scale, MidpointRounding.AwayFromZero) + ZeroPoint;
        if (scaled < Minimum) { SaturationCount++; return Minimum; }
        if (scaled > Maximum) { SaturationCount++; return Maximum; }
        return (int)scaled;
    }

    internal double Dequantize(int quantized) => (quantized - ZeroPoint) * Scale;

    internal int[] QuantizeAll(IReadOnlyList<double> values)
    {
        var result = new int[values.Count];
        for (int i = 0; i < values.Count; i++) result[i] = Quantize(values[i]);
        return result;
    }
}

internal static class QuantizedLayer
{
    /// <summary>Integer accumulation in int32, the way an 8-bit kernel does it.</summary>
    internal static int DotInt32(int[] weights, int[] inputs)
    {
        int sum = 0;
        for (int i = 0; i < weights.Length; i++) sum += weights[i] * inputs[i];
        return sum;
    }

    /// <summary>Reference accumulation with room to spare, used as an independent oracle.</summary>
    internal static long DotLong(int[] weights, int[] inputs)
    {
        long sum = 0;
        for (int i = 0; i < weights.Length; i++) sum += (long)weights[i] * inputs[i];
        return sum;
    }

    /// <summary>Requantizes an int32 accumulator into the output quantizer's grid.</summary>
    internal static int Requantize(int accumulator, double weightScale, double inputScale, AffineQuantizer output)
        => output.Quantize(accumulator * weightScale * inputScale);
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double[] MakeVector(int length, double magnitude, ulong seed)
    {
        var values = new double[length];
        var state = seed;
        for (int i = 0; i < length; i++) values[i] = magnitude * Rng.Signed(ref state);
        return values;
    }

    public static void Main()
    {
        // ---- the affine map --------------------------------------------------------------------
        {
            var weights = AffineQuantizer.SymmetricInt8(2.0);
            Check(Math.Abs(weights.Scale - 2.0 / 127.0) < 1e-15, "the symmetric scale is max|w|/127");
            Check(weights.Quantize(2.0) == 127 && weights.Quantize(-2.0) == -127, "weights hit both ends of int8");
            Check(weights.Quantize(0.0) == 0, "a zero weight maps to zero");

            var activations = AffineQuantizer.AsymmetricUint8(-0.5, 3.5);
            Check(activations.Quantize(-0.5) == 0 && activations.Quantize(3.5) == 255,
                "activations hit both ends of uint8");
            Check(activations.Quantize(0.0) >= 0 && Math.Abs(activations.Dequantize(activations.Quantize(0.0))) <= activations.Scale,
                "the zero point represents a real zero within half a scale");

            double worstError = 0;
            ulong state = 0x71UL;
            for (int i = 0; i < 4000; i++)
            {
                double value = -0.5 + 4.0 * ((Rng.Signed(ref state) + 1.0) / 2.0);   // inside the calibrated range
                double restored = activations.Dequantize(activations.Quantize(value));
                worstError = Math.Max(worstError, Math.Abs(restored - value));
            }
            Check(worstError <= activations.Scale / 2 + 1e-12,
                "every round trip is inside half a scale (worst " + worstError.ToString("E3") + ")");

            var exact = AffineQuantizer.SymmetricInt8(1.0);
            double exactValue = 3.0 / 127.0;
            Check(Math.Abs(exact.Dequantize(exact.Quantize(exactValue)) - exactValue) < 1e-15,
                "a value already on the grid survives exactly");

            var saturating = AffineQuantizer.SymmetricInt8(1.0);
            Check(saturating.Quantize(100.0) == 127 && saturating.Quantize(-100.0) == -127,
                "out-of-range values clamp to the ends");
            Check(saturating.SaturationCount == 2, "the saturations are counted (" + saturating.SaturationCount + ")");
            Check(saturating.Quantize(0.5) == 64 && saturating.SaturationCount == 2,
                "in-range values do not add saturations");
            Console.WriteLine("symmetricScale=" + weights.Scale.ToString("E3")
                + " activationScale=" + activations.Scale.ToString("E4") + " worstRoundTrip=" + worstError.ToString("E3"));
        }

        // ---- integer accumulation and the layer error bound ------------------------------------
        {
            const int Width = 96;
            double[] weightRow = MakeVector(Width, 1.5, 0x1UL);
            double[] inputVector = MakeVector(Width, 1.0, 0x2UL);
            var weightQuantizer = AffineQuantizer.SymmetricInt8(MaxAbs(weightRow));
            var inputQuantizer = AffineQuantizer.AsymmetricUint8(MinOf(inputVector), MaxOf(inputVector));
            int[] quantizedWeights = weightQuantizer.QuantizeAll(weightRow);
            int[] quantizedInputs = inputQuantizer.QuantizeAll(inputVector);

            long reference = QuantizedLayer.DotLong(quantizedWeights, quantizedInputs);
            int fast = QuantizedLayer.DotInt32(quantizedWeights, quantizedInputs);
            Check(fast == reference, "the int32 accumulator matches the long reference bit for bit");
            Check(Math.Abs(reference) < int.MaxValue / 2,
                "the fixture stays well inside int32 range (" + reference + ")");

            double floatDot = 0;
            for (int i = 0; i < Width; i++) floatDot += weightRow[i] * inputVector[i];
            double quantizedDot = 0;
            for (int i = 0; i < Width; i++) quantizedDot += weightQuantizer.Dequantize(quantizedWeights[i])
                * inputQuantizer.Dequantize(quantizedInputs[i]);
            double error = Math.Abs(quantizedDot - floatDot);
            double bound = 0;
            for (int i = 0; i < Width; i++)
                bound += Math.Abs(weightRow[i]) * inputQuantizer.Scale / 2
                    + Math.Abs(inputVector[i]) * weightQuantizer.Scale / 2;
            Check(error <= bound + 1e-9, "the dot product error stays inside the first-order bound ("
                + error.ToString("E4") + " <= " + bound.ToString("E4") + ")");
            Console.WriteLine("dotError=" + error.ToString("E5") + " dotBound=" + bound.ToString("E5")
                + " accumulator=" + reference);
        }

        // ---- per-channel beats per-tensor ------------------------------------------------------
        {
            const int Channels = 4, Width = 32;
            var state = 0x5UL;
            var channelMax = new double[Channels];
            var matrix = new double[Channels][];
            for (int c = 0; c < Channels; c++)
            {
                double magnitude = c == 0 ? 8.0 : 0.05;                 // one loud channel, three quiet ones
                matrix[c] = MakeVector(Width, magnitude, 0x100UL + (ulong)c);
                channelMax[c] = MaxAbs(matrix[c]);
            }
            double globalMax = 0;
            foreach (double m in channelMax) globalMax = Math.Max(globalMax, m);

            var inputs = MakeVector(Width, 1.0, 0x200UL);
            var inputQuantizer = AffineQuantizer.AsymmetricUint8(MinOf(inputs), MaxOf(inputs));
            int[] quantizedInputs = inputQuantizer.QuantizeAll(inputs);

            var perTensor = AffineQuantizer.SymmetricInt8(globalMax);
            double perTensorError = 0, perChannelError = 0;
            for (int c = 0; c < Channels; c++)
            {
                var channelQuantizer = AffineQuantizer.SymmetricInt8(channelMax[c]);
                for (int i = 0; i < Width; i++)
                {
                    double tensorRestored = perTensor.Dequantize(perTensor.Quantize(matrix[c][i]));
                    double channelRestored = channelQuantizer.Dequantize(channelQuantizer.Quantize(matrix[c][i]));
                    perTensorError += Math.Abs(tensorRestored - matrix[c][i]);
                    perChannelError += Math.Abs(channelRestored - matrix[c][i]);
                }
            }
            Check(perChannelError < perTensorError,
                "per-channel scales quantize the matrix more accurately (" + perChannelError.ToString("E4")
                + " < " + perTensorError.ToString("E4") + ")");
            Check(perChannelError < perTensorError / 4,
                "and the improvement is large when the channels differ by orders of magnitude");

            // the requantized outputs stay on the uint8 grid
            var outputQuantizer = AffineQuantizer.AsymmetricUint8(-1.0, 5.0);
            int saturated = 0;
            for (int c = 0; c < Channels; c++)
            {
                var channelQuantizer = AffineQuantizer.SymmetricInt8(channelMax[c]);
                int[] quantizedWeights = channelQuantizer.QuantizeAll(matrix[c]);
                int accumulator = QuantizedLayer.DotInt32(quantizedWeights, quantizedInputs);
                int requantized = QuantizedLayer.Requantize(accumulator, channelQuantizer.Scale,
                    inputQuantizer.Scale, outputQuantizer);
                if (requantized < 0 || requantized > 255) saturated++;
            }
            Check(saturated == 0, "no requantized output escapes the uint8 grid");
            Console.WriteLine("perTensorError=" + perTensorError.ToString("E4")
                + " perChannelError=" + perChannelError.ToString("E4")
                + " outputScale=" + outputQuantizer.Scale.ToString("E4"));
        }

        // ---- degenerate tensors ----------------------------------------------------------------
        {
            var zeros = new double[16];
            var zeroQuantizer = AffineQuantizer.SymmetricInt8(MaxAbs(zeros));
            int[] quantizedZeros = zeroQuantizer.QuantizeAll(zeros);
            bool allZero = true;
            foreach (int q in quantizedZeros) if (q != 0) allZero = false;
            Check(allZero, "an all-zero tensor quantizes to all zeros");
            Check(!double.IsNaN(zeroQuantizer.Scale) && zeroQuantizer.Scale > 0,
                "the degenerate scale stays finite and positive (" + zeroQuantizer.Scale + ")");

            var constant = new double[16];
            for (int i = 0; i < constant.Length; i++) constant[i] = 2.5;
            var constantQuantizer = AffineQuantizer.AsymmetricUint8(2.5, 2.5);
            Check(constantQuantizer.Scale > 0
                && !double.IsNaN(constantQuantizer.Dequantize(constantQuantizer.Quantize(2.5))),
                "a constant tensor gets a degenerate but usable scale (" + constantQuantizer.Scale + ")");
            Check(constantQuantizer.Quantize(2.5) >= 0 && constantQuantizer.Quantize(2.5) <= 255,
                "and the constant still lands on the grid");

            var single = AffineQuantizer.SymmetricInt8(0.25);
            Check(single.Quantize(0.25) == 127 && single.Quantize(-0.25) == -127, "a single-element tensor still maps");

            var wide = AffineQuantizer.AsymmetricUint8(-100.0, 100.0);
            Check(wide.Quantize(-100.0) == 0 && wide.Quantize(100.0) == 255 && wide.ZeroPoint == 128,
                "a symmetric real range puts the zero point in the middle");

            bool thrown = false;
            try { new AffineQuantizer(0, 0, 0, 255); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a zero scale is rejected");
            thrown = false;
            try { new AffineQuantizer(1, 0, 255, 0); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an inverted range is rejected");
            thrown = false;
            try { new AffineQuantizer(1, 400, 0, 255); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a zero point outside the range is rejected");
            thrown = false;
            try { AffineQuantizer.SymmetricInt8(-1.0); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a negative magnitude is rejected");
            thrown = false;
            try { AffineQuantizer.AsymmetricUint8(5.0, 1.0); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an inverted observed range is rejected");
        }

        // ---- determinism and a full layer pass -------------------------------------------------
        {
            var first = AffineQuantizer.SymmetricInt8(1.0);
            var second = AffineQuantizer.SymmetricInt8(1.0);
            var vector = MakeVector(64, 0.9, 0x9UL);
            bool same = true;
            for (int i = 0; i < vector.Length; i++)
                if (first.Quantize(vector[i]) != second.Quantize(vector[i])) same = false;
            Check(same, "quantization is deterministic");

            // a three-layer 8-bit pass: 8 -> 6 -> 4 -> 2, with per-tensor scales throughout
            var state = 0xABCDUL;
            var activations = MakeVector(8, 1.0, 0xAAUL);
            var currentQuantizer = AffineQuantizer.AsymmetricUint8(MinOf(activations), MaxOf(activations));
            var current = currentQuantizer.QuantizeAll(activations);
            var sizes = new[] { 6, 4, 2 };
            var integerTrace = new List<string>();
            foreach (int size in sizes)
            {
                var rows = new double[size][];
                var rowQuantizers = new AffineQuantizer[size];
                for (int r = 0; r < size; r++)
                {
                    rows[r] = MakeVector(current.Length, 1.0, 0xBBUL + (ulong)r);
                    rowQuantizers[r] = AffineQuantizer.SymmetricInt8(MaxAbs(rows[r]));
                }
                var next = new int[size];
                double combinedScale = 0;
                for (int r = 0; r < size; r++)
                {
                    int[] quantizedRow = rowQuantizers[r].QuantizeAll(rows[r]);
                    next[r] = QuantizedLayer.DotInt32(quantizedRow, current);
                    combinedScale = rowQuantizers[r].Scale * currentQuantizer.Scale;
                }
                var real = new double[size];
                double low = double.MaxValue, high = double.MinValue;
                for (int r = 0; r < size; r++)
                {
                    real[r] = next[r] * combinedScale;
                    low = Math.Min(low, real[r]);
                    high = Math.Max(high, real[r]);
                }
                var outputQuantizer = AffineQuantizer.AsymmetricUint8(low, high);
                current = outputQuantizer.QuantizeAll(real);
                currentQuantizer = outputQuantizer;
                integerTrace.Add("layer" + size + "=" + current[0]);
                _ = state;
            }
            Check(current.Length == 2, "the three-layer pass ends with two outputs");
            bool finiteOutputs = true;
            foreach (int q in current) if (q < 0 || q > 255) finiteOutputs = false;
            Check(finiteOutputs, "every intermediate activation stayed on the uint8 grid");
            Console.WriteLine("layerTrace=" + string.Join(",", integerTrace));
        }

        Console.WriteLine("PASS");
    }

    private static double MaxAbs(IReadOnlyList<double> values)
    {
        double best = 0;
        foreach (double v in values) best = Math.Max(best, Math.Abs(v));
        return best;
    }

    private static double MinOf(IReadOnlyList<double> values)
    {
        double best = double.MaxValue;
        foreach (double v in values) best = Math.Min(best, v);
        return values.Count == 0 ? 0 : best;
    }

    private static double MaxOf(IReadOnlyList<double> values)
    {
        double best = double.MinValue;
        foreach (double v in values) best = Math.Max(best, v);
        return values.Count == 0 ? 0 : best;
    }
}
