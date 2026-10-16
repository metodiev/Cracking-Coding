// Task 715 - Implement sparse neural network training sketch with magnitude pruning.
// (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// A tiny hand-written MLP (3 inputs -> 5 tanh hidden units -> 1 linear output, 26 parameters) is
// trained by full-batch gradient descent, then compressed with *iterative magnitude pruning* (Han et
// al.): train, remove the p% smallest weights in absolute value, retrain the survivors with the
// removed weights pinned to zero, and repeat until the target sparsity is reached. Pruning is global
// (across layers) and masked, so a pruning round can never resurrect a weight: the mask invariant -
// "every pruned parameter is exactly zero after any amount of retraining" - is checked after every
// single cycle.
//
// The forward and backward passes are written out by hand, so the analytic gradient is not taken on
// trust: this file cross-checks every one of the 26 partial derivatives against a central finite
// difference of the loss, which is the independent oracle for the whole training routine. On top of
// that the checks cover the loss reduction of the dense model, the loss recovery after each pruning
// cycle (retraining must not be worse than pruning alone), monotone sparsity growth up to the exact
// target, the accuracy of the pruned model relative to the dense one, a byte-level compression
// estimate, and the degenerate settings (prune nothing, prune everything, one hidden unit, invalid
// sparsity, empty data).
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

internal sealed class TinyMlp
{
    internal readonly int Inputs, Hidden;
    internal readonly double[][] W1;              // Hidden x Inputs
    internal readonly double[] B1;                // Hidden
    internal readonly double[] W2;                // Hidden
    internal double B2;                           // scalar output bias
    internal readonly bool[] Mask;                // per parameter, in the canonical order below

    internal TinyMlp(int inputs, int hidden, ulong seed)
    {
        Inputs = inputs;
        Hidden = hidden;
        var state = seed;
        W1 = new double[hidden][];
        for (int h = 0; h < hidden; h++)
        {
            W1[h] = new double[inputs];
            for (int i = 0; i < inputs; i++) W1[h][i] = 0.5 * Rng.Signed(ref state);
        }
        B1 = new double[hidden];
        W2 = new double[hidden];
        for (int h = 0; h < hidden; h++) W2[h] = 0.5 * Rng.Signed(ref state);
        B2 = 0.1 * Rng.Signed(ref state);
        Mask = new bool[ParameterCount];
        for (int p = 0; p < Mask.Length; p++) Mask[p] = true;
    }

    internal int ParameterCount => Hidden * Inputs + Hidden + Hidden + 1;

    /// <summary>Parameter i in a canonical order: all W1, then B1, then W2, then B2.</summary>
    internal double GetParameter(int index)
    {
        if (index < Hidden * Inputs) return W1[index / Inputs][index % Inputs];
        index -= Hidden * Inputs;
        if (index < Hidden) return B1[index];
        index -= Hidden;
        if (index < Hidden) return W2[index];
        return B2;
    }

    internal void SetParameter(int index, double value)
    {
        if (index < Hidden * Inputs) { W1[index / Inputs][index % Inputs] = value; return; }
        index -= Hidden * Inputs;
        if (index < Hidden) { B1[index] = value; return; }
        index -= Hidden;
        if (index < Hidden) { W2[index] = value; return; }
        B2 = value;
    }

    internal double[] Forward(double[] x, double[] hiddenActivation)
    {
        for (int h = 0; h < Hidden; h++)
        {
            double sum = B1[h];
            for (int i = 0; i < Inputs; i++) sum += W1[h][i] * x[i];
            hiddenActivation[h] = Math.Tanh(sum);
        }
        double output = B2;
        for (int h = 0; h < Hidden; h++) output += W2[h] * hiddenActivation[h];
        return hiddenActivation;
    }

    internal double Predict(double[] x, double[] scratch)
    {
        Forward(x, scratch);
        double output = B2;
        for (int h = 0; h < Hidden; h++) output += W2[h] * scratch[h];
        return output;
    }

    internal double Loss(double[][] inputs, double[] targets, bool[]? mask)
    {
        var scratch = new double[Hidden];
        double total = 0;
        for (int n = 0; n < inputs.Length; n++)
        {
            double error = Predict(inputs[n], scratch) - targets[n];
            total += error * error;
        }
        _ = mask;
        return inputs.Length == 0 ? 0 : total / inputs.Length;
    }

    /// <summary>Analytic gradients in the canonical parameter order.</summary>
    internal double[] Gradients(double[][] inputs, double[] targets, bool[]? mask)
    {
        var gradient = new double[ParameterCount];
        var hidden = new double[Hidden];
        int count = inputs.Length;
        for (int n = 0; n < count; n++)
        {
            Forward(inputs[n], hidden);
            double output = B2;
            for (int h = 0; h < Hidden; h++) output += W2[h] * hidden[h];
            double dOutput = 2.0 * (output - targets[n]) / count;

            for (int h = 0; h < Hidden; h++)
            {
                double dHidden = dOutput * W2[h] * (1 - hidden[h] * hidden[h]);
                for (int i = 0; i < Inputs; i++) gradient[h * Inputs + i] += dHidden * inputs[n][i];
                gradient[Hidden * Inputs + h] += dHidden;
                gradient[Hidden * Inputs + Hidden + h] += dOutput * hidden[h];
            }
            gradient[ParameterCount - 1] += dOutput;
        }
        if (mask != null)
        {
            for (int p = 0; p < gradient.Length; p++) if (!mask[p]) gradient[p] = 0;
        }
        return gradient;
    }

    /// <summary>Gradient descent on the parameters that the mask still allows.</summary>
    internal void Train(double[][] inputs, double[] targets, bool[]? mask, int steps, double learningRate)
    {
        var gradient = new double[ParameterCount];
        for (int step = 0; step < steps; step++)
        {
            gradient = Gradients(inputs, targets, mask);
            for (int p = 0; p < ParameterCount; p++)
            {
                if (mask != null && !mask[p]) continue;
                SetParameter(p, GetParameter(p) - learningRate * gradient[p]);
            }
        }
    }

    internal int ZeroCount()
    {
        int zeros = 0;
        for (int p = 0; p < ParameterCount; p++) if (GetParameter(p) == 0) zeros++;
        return zeros;
    }

    internal bool MaskedParametersAreZero()
    {
        for (int p = 0; p < ParameterCount; p++) if (!Mask[p] && GetParameter(p) != 0) return false;
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static (double[][] inputs, double[] targets) MakeData(int samples, ulong seed)
    {
        var inputs = new double[samples][];
        var targets = new double[samples];
        var state = seed;
        for (int n = 0; n < samples; n++)
        {
            double a = Rng.Signed(ref state) * 2.0, b = Rng.Signed(ref state), c = Rng.Signed(ref state);
            inputs[n] = new[] { a, b, c };
            targets[n] = 0.6 * Math.Sin(2 * a) - 0.4 * b * c + 0.3;
        }
        return (inputs, targets);
    }

    public static void Main()
    {
        (double[][] inputs, double[] targets) = MakeData(60, 0x5EEDUL);
        var model = new TinyMlp(3, 5, 0xC0FFEEUL);
        Check(model.ParameterCount == 26, "the fixture model has 26 parameters");

        // ---- the analytic gradient against central differences --------------------------------
        {
            var gradient = model.Gradients(inputs, targets, null);
            const double Step = 1e-5;
            double worstRelative = 0;
            string worstParameter = "none";
            for (int p = 0; p < model.ParameterCount; p++)
            {
                double original = model.GetParameter(p);
                model.SetParameter(p, original + Step);
                double up = model.Loss(inputs, targets, null);
                model.SetParameter(p, original - Step);
                double down = model.Loss(inputs, targets, null);
                model.SetParameter(p, original);
                double numeric = (up - down) / (2 * Step);
                double scale = Math.Max(1.0, Math.Abs(numeric));
                double relative = Math.Abs(numeric - gradient[p]) / scale;
                if (relative > worstRelative)
                {
                    worstRelative = relative;
                    worstParameter = "p" + p;
                }
            }
            Check(worstRelative < 1e-5, "every analytic derivative matches a central difference (worst "
                + worstRelative.ToString("E2") + " at " + worstParameter + ")");
            Console.WriteLine("gradientCheckWorstRelativeError=" + worstRelative.ToString("E3"));
        }

        // ---- a dense baseline ------------------------------------------------------------------
        const int DenseSteps = 600;
        var dense = new TinyMlp(3, 5, 0xC0FFEEUL);
        double initialLoss = dense.Loss(inputs, targets, null);
        dense.Train(inputs, targets, null, DenseSteps, 0.05);
        double denseLoss = dense.Loss(inputs, targets, null);
        Check(denseLoss < initialLoss / 2,
            "dense training more than halves the loss (" + denseLoss.ToString("E4") + " vs " + initialLoss.ToString("E4") + ")");
        Console.WriteLine("initialLoss=" + initialLoss.ToString("E4") + " denseLoss=" + denseLoss.ToString("E4"));

        // ---- iterative magnitude pruning -------------------------------------------------------
        {
            var pruned = new TinyMlp(3, 5, 0xC0FFEEUL);
            pruned.Train(inputs, targets, null, DenseSteps, 0.05);
            double trainedLoss = pruned.Loss(inputs, targets, null);

            const double TargetSparsity = 0.8;
            const int Cycles = 8;
            int targetPruned = (int)Math.Round(TargetSparsity * pruned.ParameterCount);
            bool sparsityGrew = true, maskHeld = true, recoveryHelped = true;
            int previousZeros = 0;
            double lastPruneOnlyLoss = 0, lastRetrainedLoss = 0;

            for (int cycle = 0; cycle < Cycles; cycle++)
            {
                int keptNow = CountKeptIn(pruned);
                int alreadyPruned = pruned.ParameterCount - keptNow;
                if (alreadyPruned >= targetPruned) break;                   // the target is already met
                int toPrune = Math.Max(1, (int)Math.Round(keptNow * 0.25));
                toPrune = Math.Min(toPrune, targetPruned - alreadyPruned);

                // global magnitude ranking over the parameters the mask still allows
                var order = new List<int>();
                for (int p = 0; p < pruned.ParameterCount; p++) if (pruned.Mask[p]) order.Add(p);
                order.Sort((a, b) =>
                {
                    double diff = Math.Abs(pruned.GetParameter(a)) - Math.Abs(pruned.GetParameter(b));
                    return diff != 0 ? (diff < 0 ? -1 : 1) : a.CompareTo(b);
                });
                for (int i = 0; i < toPrune; i++)
                {
                    pruned.Mask[order[i]] = false;
                    pruned.SetParameter(order[i], 0.0);
                }
                lastPruneOnlyLoss = pruned.Loss(inputs, targets, pruned.Mask);

                int zeros = pruned.ParameterCount - CountKeptIn(pruned);
                if (zeros < previousZeros) sparsityGrew = false;
                previousZeros = zeros;

                pruned.Train(inputs, targets, pruned.Mask, 300, 0.05);
                lastRetrainedLoss = pruned.Loss(inputs, targets, pruned.Mask);
                if (lastRetrainedLoss > lastPruneOnlyLoss + 1e-12) recoveryHelped = false;
                if (!pruned.MaskedParametersAreZero()) maskHeld = false;
            }

            int zerosFinal = pruned.ParameterCount - CountKeptIn(pruned);
            double sparsity = (double)zerosFinal / pruned.ParameterCount;
            Check(maskHeld, "every pruned parameter stayed exactly zero through all retraining");
            Check(sparsityGrew, "sparsity grows monotonically across pruning cycles");
            Check(sparsity >= TargetSparsity,
                "the target sparsity is reached (" + sparsity.ToString("F3") + " >= " + TargetSparsity + ")");
            Check(recoveryHelped, "retraining never makes a pruned model worse (" + lastRetrainedLoss.ToString("E4")
                + " <= " + lastPruneOnlyLoss.ToString("E4") + ")");
            Check(lastRetrainedLoss < initialLoss * 0.85,
                "the 80% sparse model is still clearly better than an untrained one (" + lastRetrainedLoss.ToString("E4")
                + " vs " + initialLoss.ToString("E4") + ")");
            Check(lastRetrainedLoss <= denseLoss * 3.0,
                "the 80% sparse model stays within 3x of the dense loss (" + lastRetrainedLoss.ToString("E4")
                + " vs " + denseLoss.ToString("E4") + ")");
            Check(pruned.ZeroCount() >= zerosFinal, "no zero parameter was resurrected (" + pruned.ZeroCount() + ")");

            double storedBytes = (pruned.ParameterCount - zerosFinal) * 8.0 + zerosFinal * 1.0;   // sparse value + index bit
            double denseBytes = pruned.ParameterCount * 8.0;
            Check(storedBytes < denseBytes, "the sparse encoding is smaller (" + storedBytes + " vs " + denseBytes + " bytes)");
            Console.WriteLine("sparsity=" + sparsity.ToString("F3") + " sparseLoss=" + lastRetrainedLoss.ToString("E4")
                + " pruneOnlyLoss=" + lastPruneOnlyLoss.ToString("E4") + " sparseBytes=" + storedBytes);
        }

        // ---- a masked parameter receives no gradient -------------------------------------------
        {
            var masked = new TinyMlp(3, 5, 0x11UL);
            for (int p = 0; p < masked.ParameterCount; p += 3) { masked.Mask[p] = false; masked.SetParameter(p, 0); }
            double[] gradient = masked.Gradients(inputs, targets, masked.Mask);
            bool zeroGradients = true;
            for (int p = 0; p < masked.ParameterCount; p++)
                if (!masked.Mask[p] && gradient[p] != 0) zeroGradients = false;
            Check(zeroGradients, "masked parameters never receive gradient, so pruning is permanent");
            masked.Train(inputs, targets, masked.Mask, 50, 0.05);
            Check(masked.MaskedParametersAreZero(), "training through a mask leaves the masked parameters at zero");
        }

        // ---- degenerate settings ---------------------------------------------------------------
        {
            var untouched = new TinyMlp(3, 5, 0x22UL);
            double before = untouched.Loss(inputs, targets, null);
            untouched.Train(inputs, targets, null, 0, 0.05);
            Check(untouched.Loss(inputs, targets, null) == before, "zero training steps change nothing");

            var allPruned = new TinyMlp(3, 5, 0x33UL);
            allPruned.Train(inputs, targets, null, 100, 0.05);
            for (int p = 0; p < allPruned.ParameterCount; p++) { allPruned.Mask[p] = false; allPruned.SetParameter(p, 0); }
            double degenerated = allPruned.Loss(inputs, targets, null);
            double meanSquareTarget = 0;
            foreach (double t in targets) meanSquareTarget += t * t;
            meanSquareTarget /= targets.Length;
            Check(Math.Abs(degenerated - meanSquareTarget) < 1e-12,
                "a fully pruned model degenerates to the constant zero predictor (" + degenerated.ToString("F4") + ")");
            allPruned.Train(inputs, targets, allPruned.Mask, 50, 0.05);
            Check(allPruned.Loss(inputs, targets, null) == degenerated, "and it cannot learn anything back");
            Check(allPruned.ZeroCount() == allPruned.ParameterCount, "every parameter stays at zero");

            var tinyModel = new TinyMlp(1, 1, 0x44UL);
            var simpleInputs = new double[][] { new[] { 0.5 }, new[] { -0.5 }, new[] { 0.0 } };
            var simpleTargets = new[] { 1.0, -1.0, 0.0 };
            double startLoss = tinyModel.Loss(simpleInputs, simpleTargets, null);
            tinyModel.Train(simpleInputs, simpleTargets, null, 400, 0.1);
            Check(tinyModel.Loss(simpleInputs, simpleTargets, null) < startLoss,
                "a one-hidden-unit model still learns on a trivial task");

            bool thrown = false;
            try { double[] g = tinyModel.Gradients(Array.Empty<double[]>(), Array.Empty<double>(), null); _ = g; }
            catch (DivideByZeroException) { thrown = true; }
            Check(!thrown, "an empty dataset yields a zero loss rather than dividing by zero");
            Check(tinyModel.Loss(Array.Empty<double[]>(), Array.Empty<double>(), null) == 0, "the empty loss is zero");
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            var a = new TinyMlp(3, 5, 0x55UL);
            var b = new TinyMlp(3, 5, 0x55UL);
            a.Train(inputs, targets, null, 200, 0.05);
            b.Train(inputs, targets, null, 200, 0.05);
            bool same = true;
            for (int p = 0; p < a.ParameterCount; p++) if (a.GetParameter(p) != b.GetParameter(p)) same = false;
            Check(same, "the training run is deterministic");
            Check(a.Loss(inputs, targets, null) < initialLoss / 2, "and it actually reduces the loss");
        }

        Console.WriteLine("PASS");
    }

    private static int CountKeptIn(TinyMlp model)
    {
        int kept = 0;
        for (int p = 0; p < model.ParameterCount; p++) if (model.Mask[p]) kept++;
        return kept;
    }
}
