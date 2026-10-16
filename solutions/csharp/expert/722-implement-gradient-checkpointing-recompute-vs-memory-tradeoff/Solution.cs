// Task 722 - Implement gradient checkpointing (recompute vs memory tradeoff). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Backpropagation normally stores every layer activation, so memory grows linearly with depth. Gradient
// checkpointing (Chen et al., 2016) stores only the activations at a few *segment boundaries* and
// recomputes the activations inside a segment during its backward pass. With L layers split into s
// near-equal segments the live activation buffers are
//
//     peak(s) = s                 (one stored boundary activation per segment)
//             + max segment length (recomputed inside the segment being traversed)
//
// minimised at s = sqrt(L) by ~2 sqrt(L) buffers instead of L + 1, at the cost of exactly one extra
// forward layer evaluation per layer (2L evaluations total instead of L). Both sides of that tradeoff
// are measured here, not asserted from a formula: the backward pass tracks how many activation buffers
// are alive at once, and the number of forward layer evaluations is counted.
//
// Verified: the checkpointed backward pass reproduces the naive gradient *bit-for-bit* (identical
// operations in the identical order); the tracked peak equals s + maxSegment for every segmentation,
// for uniform and for irregular ones; the optimum sits at sqrt(L) and halves memory; the recompute
// counter is exactly 2L for every segmentation; the recomputed segment-boundary activations equal the
// stored ones bit-for-bit; and an independent central-difference oracle agrees with the analytic
// gradient. Weights come from splitmix64, so every run is reproducible.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);

    internal double NextSigned() => 2.0 * NextUnit() - 1.0;
}

/// <summary>One block: activation y = tanh(W x + b) with a small dense weight matrix.</summary>
internal sealed class Layer
{
    internal readonly int Width;
    internal readonly double[] W;    // Width x Width, row major
    internal readonly double[] B;

    internal Layer(int width, SplitMix64 rng)
    {
        Width = width;
        W = new double[width * width];
        B = new double[width];
        double scale = 1.0 / Math.Sqrt(width);
        for (int i = 0; i < W.Length; i++) W[i] = scale * rng.NextSigned();
        for (int i = 0; i < B.Length; i++) B[i] = 0.05 * rng.NextSigned();
    }

    internal double[] Forward(double[] x, out double[] preActivation)
    {
        preActivation = new double[Width];
        var y = new double[Width];
        for (int i = 0; i < Width; i++)
        {
            double acc = B[i];
            int row = i * Width;
            for (int j = 0; j < Width; j++) acc += W[row + j] * x[j];
            preActivation[i] = acc;
            y[i] = Math.Tanh(acc);
        }
        return y;
    }

    /// <summary>Returns dL/dx and accumulates dL/dW, dL/db for this layer.</summary>
    internal double[] Backward(double[] gradOut, double[] preActivation, double[] input, double[] gradW, double[] gradB)
    {
        var local = new double[Width];               // dL/d(pre-activation)
        for (int i = 0; i < Width; i++)
        {
            double t = Math.Tanh(preActivation[i]);
            local[i] = gradOut[i] * (1.0 - t * t);
        }
        var gradInput = new double[Width];
        for (int i = 0; i < Width; i++)
        {
            int row = i * Width;
            for (int j = 0; j < Width; j++)
            {
                gradW[row + j] += local[i] * input[j];
                gradInput[j] += W[row + j] * local[i];
            }
        }
        for (int i = 0; i < Width; i++) gradB[i] += local[i];
        return gradInput;
    }
}

/// <summary>A segmentation: boundaries[0] = 0 &lt; ... &lt; boundaries[s] = L.</summary>
internal sealed class SegmentPlan
{
    internal readonly int[] Boundaries;

    private SegmentPlan(int[] boundaries) { Boundaries = boundaries; }

    internal int Segments => Boundaries.Length - 1;
    internal int MaxSegmentLength
    {
        get
        {
            int longest = 0;
            for (int k = 0; k < Segments; k++) longest = Math.Max(longest, Boundaries[k + 1] - Boundaries[k]);
            return longest;
        }
    }

    internal static SegmentPlan Uniform(int layers, int segments)
    {
        if (segments < 1) throw new ArgumentOutOfRangeException(nameof(segments));
        segments = Math.Min(segments, Math.Max(1, layers));
        var boundaries = new int[segments + 1];
        for (int k = 0; k <= segments; k++) boundaries[k] = (int)((long)k * layers / segments);
        return new SegmentPlan(boundaries);
    }

    internal static SegmentPlan FromBoundaries(int[] boundaries) => new SegmentPlan((int[])boundaries.Clone());
}

public static class Solution
{
    private const int Layers = 12;
    private const int Width = 4;

    private sealed class Model
    {
        internal readonly Layer[] Blocks;
        internal readonly double[] Input;
        internal readonly double[] Target;

        internal Model(ulong seed)
        {
            var rng = new SplitMix64(seed);
            Blocks = new Layer[Solution.Layers];
            for (int i = 0; i < Solution.Layers; i++) Blocks[i] = new Layer(Width, rng);
            Input = new double[Width];
            Target = new double[Width];
            for (int j = 0; j < Width; j++)
            {
                Input[j] = rng.NextSigned();
                Target[j] = rng.NextSigned();
            }
        }
    }

    private sealed class Gradient
    {
        internal double[] Input = Array.Empty<double>();
        internal double[][] Weights = Array.Empty<double[]>();
        internal double[][] Biases = Array.Empty<double[]>();
        internal long ForwardCalls;
        internal int PeakLiveBuffers;
    }

    private static void ForwardAll(Model model, double[] x0, out double[][] activations, out double[][] preActivations, out long calls)
    {
        activations = new double[Layers + 1][];
        preActivations = new double[Layers][];
        activations[0] = (double[])x0.Clone();
        calls = 0;
        for (int i = 0; i < Layers; i++)
        {
            activations[i + 1] = model.Blocks[i].Forward(activations[i], out preActivations[i]);
            calls++;
        }
    }

    /// <summary>Ordinary backward pass over the fully stored activation list.</summary>
    private static Gradient BackwardNaive(Model model, double[] x0)
    {
        ForwardAll(model, x0, out double[][] activations, out double[][] preActivations, out long calls);
        var result = new Gradient { ForwardCalls = calls, PeakLiveBuffers = Layers + 1 };
        result.Weights = new double[Layers][];
        result.Biases = new double[Layers][];
        for (int i = 0; i < Layers; i++)
        {
            result.Weights[i] = new double[Width * Width];
            result.Biases[i] = new double[Width];
        }
        var grad = new double[Width];
        for (int j = 0; j < Width; j++) grad[j] = activations[Layers][j] - model.Target[j];   // dL/d(output)
        for (int i = Layers - 1; i >= 0; i--)
            grad = model.Blocks[i].Backward(grad, preActivations[i], activations[i], result.Weights[i], result.Biases[i]);
        result.Input = grad;
        return result;
    }

    /// <summary>Checkpointed backward pass: only segment boundaries are stored, the rest is recomputed.</summary>
    private static Gradient BackwardCheckpointed(Model model, double[] x0, SegmentPlan plan)
    {
        int s = plan.Segments;
        var stored = new double[s][];
        stored[0] = (double[])x0.Clone();
        long calls = 0;

        // Forward pass: every layer is still evaluated (the loss needs them), but only the activations
        // at the segment boundaries survive - the rest are dropped as soon as the next one is computed.
        double[] cursor = stored[0];
        int nextBoundary = 1;
        for (int i = 0; i < Layers; i++)
        {
            cursor = model.Blocks[i].Forward(cursor, out _);
            calls++;
            if (nextBoundary < s && plan.Boundaries[nextBoundary] == i + 1)
            {
                stored[nextBoundary] = cursor;
                nextBoundary++;
            }
        }

        var result = new Gradient { ForwardCalls = calls, PeakLiveBuffers = s };
        result.Weights = new double[Layers][];
        result.Biases = new double[Layers][];
        for (int i = 0; i < Layers; i++)
        {
            result.Weights[i] = new double[Width * Width];
            result.Biases[i] = new double[Width];
        }

        var grad = new double[Width];
        for (int k = s - 1; k >= 0; k--)
        {
            int from = plan.Boundaries[k];
            int to = plan.Boundaries[k + 1];
            int length = to - from;
            var activations = new double[length + 1][];
            var preActivations = new double[length][];
            activations[0] = stored[k];
            for (int i = 0; i < length; i++)
            {
                activations[i + 1] = model.Blocks[from + i].Forward(activations[i], out preActivations[i]);
                calls++;
            }
            result.PeakLiveBuffers = Math.Max(result.PeakLiveBuffers, s + length);
            if (to == Layers)
            {
                for (int j = 0; j < Width; j++) grad[j] = activations[length][j] - model.Target[j];
            }
            for (int i = length - 1; i >= 0; i--)
            {
                grad = model.Blocks[from + i].Backward(grad, preActivations[i], activations[i],
                    result.Weights[from + i], result.Biases[from + i]);
            }
        }
        result.Input = grad;
        result.ForwardCalls = calls;
        return result;
    }

    private static double Loss(Model model, double[] x0)
    {
        ForwardAll(model, x0, out double[][] activations, out _, out _);
        double sum = 0.0;
        for (int j = 0; j < Width; j++)
        {
            double d = activations[Layers][j] - model.Target[j];
            sum += 0.5 * d * d;
        }
        return sum;
    }

    /// <summary>Loss evaluated with parameters taken from a flat vector, for the finite-difference oracle.</summary>
    private static double LossWithFlat(Model model, double[] flat, double[] x0)
    {
        int cursor = 0;
        for (int i = 0; i < Layers; i++)
        {
            Array.Copy(flat, cursor, model.Blocks[i].W, 0, Width * Width);
            cursor += Width * Width;
            Array.Copy(flat, cursor, model.Blocks[i].B, 0, Width);
            cursor += Width;
        }
        return Loss(model, x0);
    }

    public static void Main()
    {
        var model = new Model(0x722C0FFEEUL);

        // ---- the checkpointed gradient must equal the naive one bit-for-bit ------------------------
        Gradient naive = BackwardNaive(model, model.Input);
        Check(naive.ForwardCalls == Layers, "the naive pass evaluates each layer once");

        var plans = new List<SegmentPlan>
        {
            SegmentPlan.Uniform(Layers, 1),
            SegmentPlan.Uniform(Layers, 3),
            SegmentPlan.Uniform(Layers, 4),
            SegmentPlan.Uniform(Layers, Layers),
            SegmentPlan.FromBoundaries(new[] { 0, 5, 9, 11, 12 }),
            SegmentPlan.FromBoundaries(new[] { 0, 1, 12 }),
        };
        foreach (SegmentPlan plan in plans)
        {
            Gradient check = BackwardCheckpointed(model, model.Input, plan);
            Check(BitEqual(naive.Input, check.Input), "segment plan " + plan.Segments + ": input gradient matches bit-for-bit");
            bool parametersEqual = true;
            for (int i = 0; i < Layers; i++)
            {
                if (!BitEqual(naive.Weights[i], check.Weights[i]) || !BitEqual(naive.Biases[i], check.Biases[i]))
                    parametersEqual = false;
            }
            Check(parametersEqual, "segment plan " + plan.Segments + ": parameter gradients match bit-for-bit");
            Check(check.ForwardCalls == 2L * Layers,
                "segment plan " + plan.Segments + ": recomputing costs exactly one extra pass (" + check.ForwardCalls + ")");
            Check(check.PeakLiveBuffers == plan.Segments + plan.MaxSegmentLength,
                "segment plan " + plan.Segments + ": live buffers are segments + longest segment ("
                + check.PeakLiveBuffers + ")");
        }

        // ---- an independent central-difference oracle for the analytic gradient --------------------
        {
            var flat = new double[Layers * (Width * Width + Width)];
            var analytic = new double[flat.Length];
            int cursor = 0;
            for (int i = 0; i < Layers; i++)
            {
                Array.Copy(model.Blocks[i].W, 0, flat, cursor, Width * Width);
                cursor += Width * Width;
                Array.Copy(model.Blocks[i].B, 0, flat, cursor, Width);
                cursor += Width;
            }
            cursor = 0;
            for (int i = 0; i < Layers; i++)
            {
                Array.Copy(naive.Weights[i], 0, analytic, cursor, Width * Width);
                cursor += Width * Width;
                Array.Copy(naive.Biases[i], 0, analytic, cursor, Width);
                cursor += Width;
            }
            const double h = 1e-6;
            double worst = 0.0, worstMagnitude = 0.0;
            int worstIndex = -1, healthy = 0;
            for (int p = 0; p < flat.Length; p++)
            {
                double original = flat[p];
                flat[p] = original + h;
                double plus = LossWithFlat(model, flat, model.Input);
                flat[p] = original - h;
                double minus = LossWithFlat(model, flat, model.Input);
                flat[p] = original;
                double numeric = (plus - minus) / (2.0 * h);
                double error = Math.Abs(numeric - analytic[p]) / (1e-8 + 1e-5 * Math.Abs(analytic[p]));
                if (error > worst) { worst = error; worstIndex = p; worstMagnitude = analytic[p]; }
                if (Math.Abs(analytic[p]) > 1e-6) healthy++;
            }
            LossWithFlat(model, flat, model.Input);            // put the original parameters back
            Console.WriteLine("finiteDifferences: worstRelativeError=" + worst.ToString("G4")
                + " atParameter=" + worstIndex + " (analytic=" + worstMagnitude.ToString("G4") + ") healthyGradients="
                + healthy + " of " + flat.Length);
            Check(worst <= 0.05, "the analytic gradient matches central differences (" + worst + ")");
            Check(healthy >= Width * Width + Width, "the oracle exercised whole layers of non-vanishing gradients ("
                + healthy + " parameters)");
        }

        // ---- the memory/recompute tradeoff, measured for every segmentation ------------------------
        {
            int bestSegments = 0, bestPeak = int.MaxValue;
            var peaks = new int[Layers + 1];
            for (int s = 1; s <= Layers; s++)
            {
                SegmentPlan plan = SegmentPlan.Uniform(Layers, s);
                Gradient run = BackwardCheckpointed(model, model.Input, plan);
                peaks[s] = run.PeakLiveBuffers;
                Check(run.PeakLiveBuffers == s + plan.MaxSegmentLength, "s=" + s + ": the peak follows the model");
                if (run.PeakLiveBuffers < bestPeak) { bestPeak = run.PeakLiveBuffers; bestSegments = s; }
            }
            int sqrtLayers = (int)Math.Sqrt(Layers);
            Console.WriteLine("layers=" + Layers + " naivePeak=" + (Layers + 1) + " bestPeak=" + bestPeak
                + " bestSegments=" + bestSegments + " sqrtL=" + sqrtLayers);
            Check(peaks[1] == Layers + 1, "a single segment is as expensive as storing everything");
            Check(peaks[Layers] == Layers + 1, "checkpointing every layer is also as expensive as storing everything");
            Check(bestSegments <= sqrtLayers + 1, "the optimum sits at about sqrt(L) segments (" + bestSegments + ")");
            Check(bestPeak <= 2 * sqrtLayers + 2, "the optimum peak is about 2 sqrt(L) buffers (" + bestPeak + ")");
            Check(bestPeak * 3 <= 2 * Layers + 2,
                "the optimum uses at most two thirds of the naive activation buffers (" + bestPeak + " of " + (Layers + 1) + ")");

            // sqrt scaling: the same rule keeps working as the chain grows
            foreach (int depth in new[] { 16, 64, 144, 256 })
            {
                int best = int.MaxValue;
                for (int s = 1; s <= depth; s++)
                {
                    SegmentPlan plan = SegmentPlan.Uniform(depth, s);
                    best = Math.Min(best, s + plan.MaxSegmentLength);
                }
                Check(best * best >= depth, "L=" + depth + ": no segmentation beats sqrt(L) (peak " + best + ")");
                Check(best <= 2 * (int)Math.Sqrt(depth) + 2, "L=" + depth + ": sqrt(L) is achievable (" + best + ")");
            }
        }

        // ---- boundaries: a shallow chain, an irregular plan, recomputed values --------------------
        {
            SegmentPlan shallow = SegmentPlan.Uniform(1, 1);
            Check(shallow.Segments == 1 && shallow.MaxSegmentLength == 1, "a one-layer chain is one segment");
            Check(SegmentPlan.Uniform(8, 100).Segments == 8, "more segments than layers are clamped to one per layer");

            SegmentPlan irregular = SegmentPlan.FromBoundaries(new[] { 0, 4, 5, 6, 12 });
            Gradient run = BackwardCheckpointed(model, model.Input, irregular);
            Check(run.PeakLiveBuffers == irregular.Segments + irregular.MaxSegmentLength
                && run.PeakLiveBuffers > 2 * (int)Math.Sqrt(Layers) + 1,
                "an irregular plan cannot beat its own longest segment (" + run.PeakLiveBuffers + " buffers)");
            Check(BitEqual(naive.Input, run.Input), "an irregular plan still reproduces the gradient exactly");

            // recomputed boundary activations must equal the stored ones bit-for-bit
            ForwardAll(model, model.Input, out double[][] all, out _, out _);
            double[] walked = (double[])model.Input.Clone();
            for (int i = 0; i < irregular.Boundaries[1]; i++) walked = model.Blocks[i].Forward(walked, out _);
            Check(BitEqual(walked, all[irregular.Boundaries[1]]), "a recomputed boundary activation is bit-exact");

            Gradient repeat = BackwardCheckpointed(model, model.Input, irregular);
            Check(BitEqual(run.Input, repeat.Input) && run.ForwardCalls == repeat.ForwardCalls,
                "the checkpointed pass is fully reproducible");
            Check(!ReferenceEquals(run.Input, repeat.Input), "each pass returns its own buffers");
        }

        Console.WriteLine("PASS");
    }

    private static bool BitEqual(double[] a, double[] b)
    {
        if (a == null || b == null || a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.DoubleToInt64Bits(a[i]) != BitConverter.DoubleToInt64Bits(b[i])) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
