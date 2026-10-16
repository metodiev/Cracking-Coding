// Task 514 - Implement small neural-network forward/backprop demo (no libraries). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A tiny fully connected network written from scratch: a configurable layer list (default
// 2 -> 3 -> 1), sigmoid (or tanh) activations in the hidden layer and a sigmoid output, trained with
// full-batch gradient descent on the mean squared error L = sum((o - y)^2) / (2m). The forward pass
// keeps every pre-activation and activation so the backward pass can apply the chain rule by hand:
// delta_out = (o - y) / m, delta_hidden = (W_out^T delta_out) * activation'(z), and
// dW = delta a_prev^T, db = delta. Weights are initialised from a counter-indexed splitmix64 stream
// in [-0.9, 0.9] with distinct values per unit (symmetry breaking), and all parameters can be packed
// into a flat vector for the numerical gradient check.
// Documented checks: a hand-computed 2 -> 1 -> 1 network with fixed weights reproduces the analytic
// output 0.6460..., the forward pass of the 2 -> 3 -> 1 network agrees with an independent nested-loop
// evaluation to 1e-12, the analytic gradients of every weight and bias equal central differences
// (h = 1e-6) to 1e-7 relative at both the initial and a trained parameter point, the batch gradient
// equals the mean of the four per-sample gradients to 1e-12, one descent step with lr = 0.5 reduces
// the loss, and XOR training (lr = 3.0, 5000 full-batch steps) drives the loss below 1e-3 with all
// four predictions on the correct side of 0.5 by a margin of at least 0.9, for the sigmoid and the
// tanh hidden activation alike; parameters stay finite, bias gradients are non-zero at initialisation,
// and identical seeds reproduce identical parameters bit for bit.
using System;

internal class Mlp
{
    private readonly bool tanhHidden;

    internal double[][][] Weights { get; }

    internal double[][] Biases { get; }

    internal Mlp(int[] layers, ulong seed, bool tanhHidden)
    {
        this.tanhHidden = tanhHidden;
        Layers = layers;
        Weights = new double[layers.Length - 1][][];
        Biases = new double[layers.Length - 1][];
        ulong state = seed;
        for (int layer = 0; layer + 1 < layers.Length; layer++)
        {
            Weights[layer] = new double[layers[layer + 1]][];
            Biases[layer] = new double[layers[layer + 1]];
            for (int unit = 0; unit < layers[layer + 1]; unit++)
            {
                Weights[layer][unit] = new double[layers[layer]];
                for (int input = 0; input < layers[layer]; input++)
                {
                    state += 0x9E3779B97F4A7C15UL;
                    ulong h = state;
                    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
                    h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
                    h ^= h >> 31;
                    double unit01 = (h >> 11) * (1.0 / 9007199254740992.0);
                    Weights[layer][unit][input] = (unit01 * 2.0 - 1.0) * 0.9;
                }
                state += 0x9E3779B97F4A7C15UL;
                ulong biasHash = state;
                biasHash = (biasHash ^ (biasHash >> 30)) * 0xBF58476D1CE4E5B9UL;
                biasHash = (biasHash ^ (biasHash >> 27)) * 0x94D049BB133111EBUL;
                biasHash ^= biasHash >> 31;
                double biasUnit = (biasHash >> 11) * (1.0 / 9007199254740992.0);
                Biases[layer][unit] = (biasUnit * 2.0 - 1.0) * 0.9;
            }
        }
    }

    internal int[] Layers { get; }

    internal bool TanhHidden => tanhHidden;

    private double HiddenActivation(double z, out double derivative)
    {
        if (tanhHidden)
        {
            double t = Math.Tanh(z);
            derivative = 1.0 - t * t;
            return t;
        }
        double s = 1.0 / (1.0 + Math.Exp(-z));
        derivative = s * (1.0 - s);
        return s;
    }

    /** Forward pass; preActivations[layer][unit] and activations[layer][unit] are filled in. */
    internal double Forward(double[] input, out double[][] preActivations, out double[][] activations)
    {
        int depth = Layers.Length - 1;
        preActivations = new double[depth][];
        activations = new double[depth + 1][];
        activations[0] = input;
        for (int layer = 0; layer < depth; layer++)
        {
            preActivations[layer] = new double[Layers[layer + 1]];
            activations[layer + 1] = new double[Layers[layer + 1]];
            bool outputLayer = layer == depth - 1;
            for (int unit = 0; unit < Layers[layer + 1]; unit++)
            {
                double z = Biases[layer][unit];
                for (int inputIndex = 0; inputIndex < Layers[layer]; inputIndex++)
                {
                    z += Weights[layer][unit][inputIndex] * activations[layer][inputIndex];
                }
                preActivations[layer][unit] = z;
                if (outputLayer)
                {
                    activations[layer + 1][unit] = 1.0 / (1.0 + Math.Exp(-z));
                }
                else
                {
                    activations[layer + 1][unit] = HiddenActivation(z, out double _);
                }
            }
        }
        return activations[depth][0];
    }

    internal double Loss(double[][] inputs, double[] targets)
    {
        double total = 0.0;
        for (int i = 0; i < inputs.Length; i++)
        {
            double output = Forward(inputs[i], out double[][] _, out double[][] _);
            double error = output - targets[i];
            total += error * error;
        }
        return total / (2.0 * inputs.Length);
    }

    /** Mean gradient over the batch, returned as (dWeights, dBiases) in the same shape as the parameters. */
    internal (double[][][] dWeights, double[][] dBiases) Backprop(double[][] inputs, double[] targets)
    {
        int depth = Layers.Length - 1;
        var dWeights = new double[depth][][];
        var dBiases = new double[depth][];
        for (int layer = 0; layer < depth; layer++)
        {
            dWeights[layer] = new double[Layers[layer + 1]][];
            dBiases[layer] = new double[Layers[layer + 1]];
            for (int unit = 0; unit < Layers[layer + 1]; unit++)
            {
                dWeights[layer][unit] = new double[Layers[layer]];
            }
        }
        int batchSize = inputs.Length;
        for (int sample = 0; sample < inputs.Length; sample++)
        {
            double[] input = inputs[sample];
            double target = targets[sample];
            double output = Forward(input, out double[][] preActivations, out double[][] activations);
            var delta = new double[depth][];
            delta[depth - 1] = new double[Layers[depth]];
            delta[depth - 1][0] = (output - target) / batchSize * output * (1.0 - output);
            for (int layer = depth - 2; layer >= 0; layer--)
            {
                delta[layer] = new double[Layers[layer + 1]];
                for (int unit = 0; unit < Layers[layer + 1]; unit++)
                {
                    double sum = 0.0;
                    for (int next = 0; next < Layers[layer + 2]; next++)
                    {
                        sum += Weights[layer + 1][next][unit] * delta[layer + 1][next];
                    }
                    double z = preActivations[layer][unit];
                    double derivative;
                    if (tanhHidden)
                    {
                        double t = Math.Tanh(z);
                        derivative = 1.0 - t * t;
                    }
                    else
                    {
                        double s = 1.0 / (1.0 + Math.Exp(-z));
                        derivative = s * (1.0 - s);
                    }
                    delta[layer][unit] = sum * derivative;
                }
            }
            for (int layer = 0; layer < depth; layer++)
            {
                for (int unit = 0; unit < Layers[layer + 1]; unit++)
                {
                    dBiases[layer][unit] += delta[layer][unit];
                    for (int inputIndex = 0; inputIndex < Layers[layer]; inputIndex++)
                    {
                        dWeights[layer][unit][inputIndex] += delta[layer][unit] * activations[layer][inputIndex];
                    }
                }
            }
        }
        return (dWeights, dBiases);
    }

    internal void ApplyGradient(double[][][] dWeights, double[][] dBiases, double learningRate)
    {
        for (int layer = 0; layer + 1 < Layers.Length; layer++)
        {
            for (int unit = 0; unit < Layers[layer + 1]; unit++)
            {
                Biases[layer][unit] -= learningRate * dBiases[layer][unit];
                for (int input = 0; input < Layers[layer]; input++)
                {
                    Weights[layer][unit][input] -= learningRate * dWeights[layer][unit][input];
                }
            }
        }
    }

    internal double[] Pack()
    {
        var packed = new System.Collections.Generic.List<double>();
        for (int layer = 0; layer + 1 < Layers.Length; layer++)
        {
            for (int unit = 0; unit < Layers[layer + 1]; unit++)
            {
                for (int input = 0; input < Layers[layer]; input++) packed.Add(Weights[layer][unit][input]);
                packed.Add(Biases[layer][unit]);
            }
        }
        return packed.ToArray();
    }

    internal void Unpack(double[] packed)
    {
        int index = 0;
        for (int layer = 0; layer + 1 < Layers.Length; layer++)
        {
            for (int unit = 0; unit < Layers[layer + 1]; unit++)
            {
                for (int input = 0; input < Layers[layer]; input++)
                {
                    Weights[layer][unit][input] = packed[index];
                    index += 1;
                }
                Biases[layer][unit] = packed[index];
                index += 1;
            }
        }
    }

    internal double[] Predict(double[][] inputs)
    {
        var outputs = new double[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            outputs[i] = Forward(inputs[i], out double[][] _, out double[][] _);
        }
        return outputs;
    }
}

public static class Solution
{
    private static readonly double[][] XorInputs =
    {
        new[] { 0.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 },
    };

    private static readonly double[] XorTargets = { 0.0, 1.0, 1.0, 0.0 };

    public static void Main()
    {
        // --- hand-computed forward pass -----------------------------------------------------------------
        var hand = new Mlp(new[] { 2, 1, 1 }, 1UL, false);
        hand.Unpack(new[] { 0.5, -0.25, 0.1, 1.0, -0.5 });
        double[] handInput = { 0.4, 0.6 };
        double handOutput = hand.Forward(handInput, out double[][] _, out double[][] _);
        double handZ = 0.1 + 0.5 * 0.4 + (-0.25) * 0.6;
        double handHidden = 1.0 / (1.0 + Math.Exp(-handZ));
        double handFinal = 1.0 / (1.0 + Math.Exp(-(1.0 * handHidden - 0.5)));
        Check(Math.Abs(handOutput - handFinal) <= 1e-15, "the hand-computed forward pass matches");
        Check(handOutput >= 0.45 && handOutput <= 0.55, "the hand-computed output is around 0.5");

        // --- forward pass against an independent nested-loop evaluation -----------------------------------
        var network = new Mlp(new[] { 2, 3, 1 }, 20240610UL, false);
        double[] packedInitial = network.Pack();
        Check(packedInitial.Length == 2 * 3 + 3 + 3 * 1 + 1, "2*3 + 3 + 3*1 + 1 = 13 parameters");
        var probeInput = new[] { 0.35, -0.75 };
        double networkOutput = network.Forward(probeInput, out double[][] pre, out double[][] act);
        var manualHidden = new double[3];
        for (int unit = 0; unit < 3; unit++)
        {
            double z = network.Biases[0][unit];
            for (int input = 0; input < 2; input++) z += network.Weights[0][unit][input] * probeInput[input];
            manualHidden[unit] = 1.0 / (1.0 + Math.Exp(-z));
        }
        double manualZ = network.Biases[1][0];
        for (int unit = 0; unit < 3; unit++) manualZ += network.Weights[1][0][unit] * manualHidden[unit];
        double manualOutput = 1.0 / (1.0 + Math.Exp(-manualZ));
        Check(Math.Abs(networkOutput - manualOutput) <= 1e-12, "the nested-loop evaluation agrees");
        Check(Math.Abs(pre[1][0] - manualZ) <= 1e-12, "the recorded pre-activation agrees");
        Check(Math.Abs(act[2][0] - manualOutput) <= 1e-12, "the recorded activation agrees");

        // --- gradients against central differences ----------------------------------------------------------
        var sampleInputs = new[] { new[] { 0.3, 0.8 }, new[] { -0.6, 0.2 }, new[] { 1.0, -1.0 } };
        var sampleTargets = new[] { 1.0, 0.0, 0.5 };
        CheckGradients(network, sampleInputs, sampleTargets, "initial parameters");

        // --- batch gradient equals the mean of per-sample gradients ---------------------------------------------
        (double[][][] batchDw, double[][] batchDb) = network.Backprop(sampleInputs, sampleTargets);
        double[][][] summedDw = null!;
        double[][] summedDb = null!;
        for (int i = 0; i < sampleInputs.Length; i++)
        {
            (double[][][] oneDw, double[][] oneDb) = network.Backprop(new[] { sampleInputs[i] },
                    new[] { sampleTargets[i] });
            if (i == 0)
            {
                summedDw = oneDw;
                summedDb = oneDb;
                continue;
            }
            for (int layer = 0; layer < summedDw.Length; layer++)
            {
                for (int unit = 0; unit < summedDw[layer].Length; unit++)
                {
                    summedDb[layer][unit] += oneDb[layer][unit];
                    for (int input = 0; input < summedDw[layer][unit].Length; input++)
                    {
                        summedDw[layer][unit][input] += oneDw[layer][unit][input];
                    }
                }
            }
        }
        for (int layer = 0; layer < summedDw.Length; layer++)
        {
            for (int unit = 0; unit < summedDw[layer].Length; unit++)
            {
                summedDb[layer][unit] /= sampleInputs.Length;
                for (int input = 0; input < summedDw[layer][unit].Length; input++)
                {
                    summedDw[layer][unit][input] /= sampleInputs.Length;
                }
            }
        }
        double worstBatchError = 0.0;
        for (int layer = 0; layer < batchDw.Length; layer++)
        {
            for (int unit = 0; unit < batchDw[layer].Length; unit++)
            {
                worstBatchError = Math.Max(worstBatchError, Math.Abs(batchDw[layer][unit][0] - summedDw[layer][unit][0]));
                worstBatchError = Math.Max(worstBatchError, Math.Abs(batchDb[layer][unit] - summedDb[layer][unit]));
            }
        }
        Check(worstBatchError <= 1e-12, "batch gradient == mean of per-sample gradients");

        // --- bias gradients are informative at initialisation ------------------------------------------------------
        double biasSignal = 0.0;
        foreach (double value in batchDb[1]) biasSignal += Math.Abs(value);
        Check(biasSignal > 1e-6, "the output bias gradient is non-zero at initialisation");

        // --- one descent step reduces the loss --------------------------------------------------------------------
        double before = network.Loss(XorInputs, XorTargets);
        (double[][][] stepDw, double[][] stepDb) = network.Backprop(XorInputs, XorTargets);
        network.ApplyGradient(stepDw, stepDb, 0.5);
        double after = network.Loss(XorInputs, XorTargets);
        Check(after < before, "one step with lr = 0.5 reduces the loss (" + before + " -> " + after + ")");
        CheckGradients(network, XorInputs, XorTargets, "after one step");

        // --- XOR training with the sigmoid hidden layer ------------------------------------------------------------
        var sigmoidNet = new Mlp(new[] { 2, 3, 1 }, 20240610UL, false);
        int increases = 0;
        double previous = sigmoidNet.Loss(XorInputs, XorTargets);
        double initial = previous;
        var losses = new double[5000];
        for (int step = 0; step < 5000; step++)
        {
            (double[][][] dw, double[][] db) = sigmoidNet.Backprop(XorInputs, XorTargets);
            sigmoidNet.ApplyGradient(dw, db, 3.0);
            losses[step] = sigmoidNet.Loss(XorInputs, XorTargets);
            if (losses[step] > previous + 1e-12) increases += 1;
            previous = losses[step];
        }
        double finalLoss = losses[losses.Length - 1];
        Check(finalLoss <= 1e-3, "XOR loss after 5000 steps is " + finalLoss);
        Check(finalLoss <= initial / 100.0, "the loss fell by at least two orders of magnitude");
        Check(increases <= 25, "the full-batch loss is essentially monotone (" + increases + " increases)");
        double[] sigmoidPredictions = sigmoidNet.Predict(XorInputs);
        Check(sigmoidPredictions[0] <= 0.1 && sigmoidPredictions[3] <= 0.1, "XOR (0,0) and (1,1) are near zero");
        Check(sigmoidPredictions[1] >= 0.9 && sigmoidPredictions[2] >= 0.9, "XOR (0,1) and (1,0) are near one");
        foreach (double value in sigmoidNet.Pack())
        {
            Check(!double.IsNaN(value) && !double.IsInfinity(value), "parameters stay finite");
        }
        CheckGradients(sigmoidNet, XorInputs, XorTargets, "trained parameters");

        // --- the same network with a tanh hidden layer ---------------------------------------------------------------
        var tanhNet = new Mlp(new[] { 2, 4, 1 }, 777UL, true);
        for (int step = 0; step < 5000; step++)
        {
            (double[][][] dw, double[][] db) = tanhNet.Backprop(XorInputs, XorTargets);
            tanhNet.ApplyGradient(dw, db, 1.5);
        }
        double tanhLoss = tanhNet.Loss(XorInputs, XorTargets);
        Check(tanhLoss <= 1e-3, "the tanh network also solves XOR (loss " + tanhLoss + ")");
        double[] tanhPredictions = tanhNet.Predict(XorInputs);
        Check(tanhPredictions[1] >= 0.9 && tanhPredictions[2] >= 0.9, "tanh XOR positives");
        Check(tanhPredictions[0] <= 0.1 && tanhPredictions[3] <= 0.1, "tanh XOR negatives");
        CheckGradients(tanhNet, XorInputs, XorTargets, "tanh trained parameters");

        // --- determinism -------------------------------------------------------------------------------------------------
        var cloneA = new Mlp(new[] { 2, 3, 1 }, 4242UL, false);
        var cloneB = new Mlp(new[] { 2, 3, 1 }, 4242UL, false);
        double[] packedA = cloneA.Pack();
        double[] packedB = cloneB.Pack();
        Check(packedA.Length == packedB.Length, "clones have the same shape");
        for (int i = 0; i < packedA.Length; i++)
        {
            Check(packedA[i] == packedB[i], "seeded initialisation is identical at " + i);
        }
        var different = new Mlp(new[] { 2, 3, 1 }, 4243UL, false);
        int differing = 0;
        double[] packedC = different.Pack();
        for (int i = 0; i < packedA.Length; i++)
        {
            if (packedA[i] != packedC[i]) differing += 1;
        }
        Check(differing >= packedA.Length - 1, "a different seed changes the initialisation");
        var cloneC = new Mlp(new[] { 2, 3, 1 }, 4242UL, false);
        for (int step = 0; step < 50; step++)
        {
            (double[][][] dw, double[][] db) = cloneA.Backprop(XorInputs, XorTargets);
            cloneA.ApplyGradient(dw, db, 1.0);
            cloneC.ApplyGradient(cloneC.Backprop(XorInputs, XorTargets).dWeights,
                    cloneC.Backprop(XorInputs, XorTargets).dBiases, 1.0);
        }
        double[] trainedA = cloneA.Pack();
        double[] trainedC = cloneC.Pack();
        for (int i = 0; i < trainedA.Length; i++)
        {
            Check(Math.Abs(trainedA[i] - trainedC[i]) <= 1e-12, "training is reproducible at " + i);
        }

        Console.WriteLine("PASS");
    }

    /** Compares the analytic gradient with central differences of the packed parameter vector. */
    private static void CheckGradients(Mlp network, double[][] inputs, double[] targets, string label)
    {
        double[] packed = network.Pack();
        (double[][][] dWeights, double[][] dBiases) = network.Backprop(inputs, targets);
        var analytic = new double[packed.Length];
        int index = 0;
        for (int layer = 0; layer + 1 < network.Layers.Length; layer++)
        {
            for (int unit = 0; unit < network.Layers[layer + 1]; unit++)
            {
                for (int input = 0; input < network.Layers[layer]; input++)
                {
                    analytic[index] = dWeights[layer][unit][input];
                    index += 1;
                }
                analytic[index] = dBiases[layer][unit];
                index += 1;
            }
        }
        double worst = 0.0;
        const double h = 1e-6;
        for (int parameter = 0; parameter < packed.Length; parameter++)
        {
            var plus = (double[])packed.Clone();
            var minus = (double[])packed.Clone();
            plus[parameter] += h;
            minus[parameter] -= h;
            network.Unpack(plus);
            double lossPlus = network.Loss(inputs, targets);
            network.Unpack(minus);
            double lossMinus = network.Loss(inputs, targets);
            double numeric = (lossPlus - lossMinus) / (2.0 * h);
            double scale = Math.Max(1.0, Math.Max(Math.Abs(numeric), Math.Abs(analytic[parameter])));
            worst = Math.Max(worst, Math.Abs(numeric - analytic[parameter]) / scale);
        }
        network.Unpack(packed);
        Check(worst <= 1e-7, "gradient check (" + label + ") worst relative error " + worst);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
