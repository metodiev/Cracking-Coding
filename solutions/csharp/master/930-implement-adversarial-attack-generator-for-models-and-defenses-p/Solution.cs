// Task 930 - Implement adversarial attack generator for models and defenses (PGD). (Tier: Master, Beyond — Machine Learning & AI)
//
// A small 4-8-3 ReLU network is trained by hand-written full-batch gradient descent on a three-cluster Gaussian
// dataset (forward pass, softmax cross-entropy, analytic backpropagation), and then attacked. FGSM takes one step of
// size eps in the sign of the input gradient; PGD takes k steps of size alpha from a deterministic random start
// inside the eps-ball and projects back onto the ball after every step, which is the standard l-infinity attack. The
// defence is adversarial training: the network keeps training on a 50/50 mix of clean and PGD-adversarial examples,
// and the checks measure what that costs (clean accuracy) against what it buys (robust accuracy). Every random
// choice comes from a hand-written splitmix64, so the whole experiment - initialisation, data, random starts - is
// reproducible.
//
// Simplifications: tiny network, full-batch (not minibatch) gradients, no momentum/weight decay, l-infinity
// threat model only, fixed step schedule, integer-free toy data, no GPU. Verified: the perturbation never leaves the
// eps-ball (exact boundary at eps = 0), PGD beats FGSM which beats the clean baseline, the mean loss grows under
// attack, a larger budget is never easier to defend, adversarial training improves robust accuracy while keeping the
// clean accuracy reasonable, and two runs with the same seed agree bit for bit.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    internal static double Gaussian(ref ulong s)
    {
        double u = ((Next(ref s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double v = (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}

internal sealed class Net
{
    internal const int In = 4, Hidden = 8, Out = 3;
    internal readonly double[,] W1 = new double[Hidden, In];
    internal readonly double[] B1 = new double[Hidden];
    internal readonly double[,] W2 = new double[Out, Hidden];
    internal readonly double[] B2 = new double[Out];
    internal Net(ulong seed)
    {
        ulong state = seed;
        for (int h = 0; h < Hidden; h++)
        {
            for (int i = 0; i < In; i++) W1[h, i] = 0.5 * Rng.Signed(ref state);
            B1[h] = 0.1 * Rng.Signed(ref state);
            for (int o = 0; o < Out; o++) W2[o, h] = 0.5 * Rng.Signed(ref state);
        }
        for (int o = 0; o < Out; o++) B2[o] = 0.1 * Rng.Signed(ref state);
    }
    internal double[] Logits(double[] x)
    {
        var hidden = new double[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            double sum = B1[h];
            for (int i = 0; i < In; i++) sum += W1[h, i] * x[i];
            hidden[h] = Math.Max(0.0, sum);
        }
        var logits = new double[Out];
        for (int o = 0; o < Out; o++)
        {
            double sum = B2[o];
            for (int h = 0; h < Hidden; h++) sum += W2[o, h] * hidden[h];
            logits[o] = sum;
        }
        return logits;
    }
    internal int Predict(double[] x) { double[] l = Logits(x); return l[0] >= l[1] && l[0] >= l[2] ? 0 : (l[1] >= l[2] ? 1 : 2); }
    internal static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        var e = logits.Select(v => Math.Exp(v - max)).ToArray();
        double sum = e.Sum();
        return e.Select(v => v / sum).ToArray();
    }
    internal static double Loss(double[] logits, int label) => -Math.Log(Math.Max(1e-12, Softmax(logits)[label]));
    /** Finite-difference input gradient, used only by the independent gradient check. */
    internal double[] NumericGradient(double[] x, int label, double h)
    {
        var g = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double original = x[i];
            x[i] = original + h; double up = Loss(Logits(x), label);
            x[i] = original - h; double down = Loss(Logits(x), label);
            x[i] = original;
            g[i] = (up - down) / (2.0 * h);
        }
        return g;
    }
    /** Analytic input gradient of the cross-entropy through ReLU, written by hand. */
    internal double[] InputGradient(double[] x, int label)
    {
        var pre = new double[Hidden];
        var hidden = new double[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            double sum = B1[h];
            for (int i = 0; i < In; i++) sum += W1[h, i] * x[i];
            pre[h] = sum; hidden[h] = Math.Max(0.0, sum);
        }
        double[] logits = Logits(x);
        double[] probability = Softmax(logits);
        var dLogit = new double[Out];
        for (int o = 0; o < Out; o++) dLogit[o] = probability[o] - (o == label ? 1.0 : 0.0);
        var g = new double[In];
        for (int h = 0; h < Hidden; h++)
        {
            if (pre[h] <= 0.0) continue;
            double dHidden = 0;
            for (int o = 0; o < Out; o++) dHidden += dLogit[o] * W2[o, h];
            for (int i = 0; i < In; i++) g[i] += dHidden * W1[h, i];
        }
        return g;
    }
    /** One full-batch gradient-descent epoch on the cross-entropy loss. */
    internal void Train(List<double[]> xs, List<int> labels, int epochs, double rate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var gW1 = new double[Hidden, In]; var gB1 = new double[Hidden];
            var gW2 = new double[Out, Hidden]; var gB2 = new double[Out];
            for (int n = 0; n < xs.Count; n++)
            {
                double[] x = xs[n];
                var pre = new double[Hidden]; var hidden = new double[Hidden];
                for (int h = 0; h < Hidden; h++)
                {
                    double sum = B1[h];
                    for (int i = 0; i < In; i++) sum += W1[h, i] * x[i];
                    pre[h] = sum; hidden[h] = Math.Max(0.0, sum);
                }
                double[] logits = Logits(x);
                double[] p = Softmax(logits);
                var dLogit = new double[Out];
                for (int o = 0; o < Out; o++) dLogit[o] = (p[o] - (o == labels[n] ? 1.0 : 0.0)) / xs.Count;
                for (int h = 0; h < Hidden; h++)
                {
                    for (int o = 0; o < Out; o++) gW2[o, h] += dLogit[o] * hidden[h];
                    if (pre[h] <= 0.0) continue;
                    double dHidden = 0;
                    for (int o = 0; o < Out; o++) dHidden += dLogit[o] * W2[o, h];
                    for (int i = 0; i < In; i++) gW1[h, i] += dHidden * x[i];
                    gB1[h] += dHidden;
                }
                for (int o = 0; o < Out; o++) gB2[o] += dLogit[o];
            }
            for (int h = 0; h < Hidden; h++)
            {
                for (int i = 0; i < In; i++) W1[h, i] -= rate * gW1[h, i];
                B1[h] -= rate * gB1[h];
                for (int o = 0; o < Out; o++) W2[o, h] -= rate * gW2[o, h];
            }
            for (int o = 0; o < Out; o++) B2[o] -= rate * gB2[o];
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    /** FGSM: one step of size eps along sign(gradient), projected onto the eps-ball. */
    private static double[] Fgsm(Net net, double[] x, int label, double eps)
    {
        double[] g = net.InputGradient(x, label);
        var adversarial = new double[x.Length];
        for (int i = 0; i < x.Length; i++) adversarial[i] = x[i] + eps * Math.Sign(g[i]);
        return adversarial;
    }
    /** PGD: k projected gradient steps of size alpha from a random start inside the ball. */
    private static double[] Pgd(Net net, double[] x, int label, double eps, double alpha, int steps, ref ulong state)
    {
        var adversarial = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            adversarial[i] = x[i] + eps * Rng.Signed(ref state);
            if (adversarial[i] > x[i] + eps) adversarial[i] = x[i] + eps;
            if (adversarial[i] < x[i] - eps) adversarial[i] = x[i] - eps;
        }
        for (int step = 0; step < steps; step++)
        {
            double[] g = net.InputGradient(adversarial, label);
            for (int i = 0; i < x.Length; i++)
            {
                adversarial[i] += alpha * Math.Sign(g[i]);
                if (adversarial[i] > x[i] + eps) adversarial[i] = x[i] + eps;
                if (adversarial[i] < x[i] - eps) adversarial[i] = x[i] - eps;
            }
        }
        return adversarial;
    }
    private static double MaxAbs(double[] a, double[] b) => a.Select((v, i) => Math.Abs(v - b[i])).Max();
    private static double Accuracy(Net net, List<double[]> xs, List<int> labels, Func<double[], int, double[]>? attack)
    {
        int correct = 0;
        for (int n = 0; n < xs.Count; n++)
        {
            double[] input = attack == null ? xs[n] : attack(xs[n], labels[n]);
            if (net.Predict(input) == labels[n]) correct++;
        }
        return correct / (double)xs.Count;
    }
    public static void Main()
    {
        ulong state = 930UL;
        var centers = new[] { new[] { 1.2, 0.0, 0.0, 0.0 }, new[] { -0.6, 1.0, 0.1, 0.0 }, new[] { -0.6, -1.0, -0.1, 0.0 } };
        var xs = new List<double[]>(); var labels = new List<int>();
        for (int c = 0; c < 3; c++)
            for (int n = 0; n < 20; n++)
            {
                var x = new double[Net.In];
                for (int i = 0; i < Net.In; i++) x[i] = centers[c][i] + 0.45 * Rng.Gaussian(ref state);
                xs.Add(x); labels.Add(c);
            }
        var net = new Net(930UL);
        net.Train(xs, labels, 300, 0.4);
        Check(Accuracy(net, xs, labels, null) > 0.85, "the clean accuracy is " + Accuracy(net, xs, labels, null).ToString("F3"));

        // 1. the analytic input gradient agrees with finite differences
        double[] gradient = net.InputGradient(xs[0], labels[0]);
        double[] numeric = net.NumericGradient((double[])xs[0].Clone(), labels[0], 1e-5);
        Check(gradient.Select((g, i) => Math.Abs(g - numeric[i])).Max() < 1e-5, "the hand-written input gradient matches central differences (max gap " + gradient.Select((g, i) => Math.Abs(g - numeric[i])).Max().ToString("E1") + ")");

        // 2. FGSM respects the budget and raises the loss in the ascent direction
        const double eps = 0.5;
        double worstFgsm = 0, lossBefore = 0, lossAfter = 0;
        foreach (int n in Enumerable.Range(0, xs.Count))
        {
            double[] adversarial = Fgsm(net, xs[n], labels[n], eps);
            worstFgsm = Math.Max(worstFgsm, MaxAbs(adversarial, xs[n]));
            lossBefore += Net.Loss(net.Logits(xs[n]), labels[n]);
            lossAfter += Net.Loss(net.Logits(adversarial), labels[n]);
        }
        Check(worstFgsm <= eps + 1e-12, "FGSM stays inside the eps-ball (worst |delta| = " + worstFgsm.ToString("F6") + ")");
        Check(lossAfter > lossBefore, "FGSM increases the mean loss (" + (lossBefore / 60).ToString("F4") + " -> " + (lossAfter / 60).ToString("F4") + ")");

        // 3. PGD is stronger than FGSM, which is stronger than the clean baseline
        ulong attackState = 7UL;
        double fgsmAccuracy = Accuracy(net, xs, labels, (x, l) => Fgsm(net, x, l, eps));
        double pgdAccuracy = Accuracy(net, xs, labels, (x, l) => Pgd(net, x, l, eps, eps / 5.0, 20, ref attackState));
        double clean = Accuracy(net, xs, labels, null);
        Check(pgdAccuracy <= fgsmAccuracy && pgdAccuracy < clean, "accuracy drops clean " + clean.ToString("F3") + " -> FGSM " + fgsmAccuracy.ToString("F3") + " -> PGD " + pgdAccuracy.ToString("F3"));
        double worstPgd = 0;
        attackState = 7UL;
        foreach (int n in Enumerable.Range(0, xs.Count)) worstPgd = Math.Max(worstPgd, MaxAbs(Pgd(net, xs[n], labels[n], eps, eps / 5.0, 20, ref attackState), xs[n]));
        Check(worstPgd <= eps + 1e-12, "PGD stays inside the eps-ball (worst |delta| = " + worstPgd.ToString("F6") + ")");

        // 4. the budget is monotone: zero budget is the identity, a larger budget is never easier
        double[] identity = Fgsm(net, xs[3], labels[3], 0.0);
        Check(MaxAbs(identity, xs[3]) == 0.0 && net.Predict(identity) == net.Predict(xs[3]), "eps = 0 leaves the input untouched and the prediction unchanged (boundary)");
        ulong smallState = 3UL, largeState = 3UL;
        double small = Accuracy(net, xs, labels, (x, l) => Pgd(net, x, l, 0.10, 0.02, 20, ref smallState));
        double large = Accuracy(net, xs, labels, (x, l) => Pgd(net, x, l, 0.90, 0.18, 20, ref largeState));
        Check(large <= small + 1e-12, "a larger budget is never easier (0.10 -> " + small.ToString("F3") + ", 0.90 -> " + large.ToString("F3") + ")");

        // 5. adversarial training buys robust accuracy at its own training budget
        const double epsTrain = 0.25;
        ulong baselineState = 7UL;
        double baselineAtBudget = Accuracy(net, xs, labels, (x, l) => Pgd(net, x, l, epsTrain, epsTrain / 5.0, 20, ref baselineState));
        var hardened = new Net(930UL);
        hardened.Train(xs, labels, 300, 0.4);
        ulong mix = 99UL;
        for (int round = 0; round < 6; round++)
        {
            var mixed = new List<double[]>(); var mixedLabels = new List<int>();
            for (int n = 0; n < xs.Count; n++)
            {
                mixed.Add(xs[n]); mixedLabels.Add(labels[n]);
                mixed.Add(Pgd(hardened, xs[n], labels[n], epsTrain, epsTrain / 5.0, 20, ref mix)); mixedLabels.Add(labels[n]);
            }
            hardened.Train(mixed, mixedLabels, 40, 0.1);               // regenerate the attack set every round
        }
        ulong hardenedState = 7UL;
        double hardenedPgd = Accuracy(hardened, xs, labels, (x, l) => Pgd(hardened, x, l, epsTrain, epsTrain / 5.0, 20, ref hardenedState));
        Check(hardenedPgd > baselineAtBudget, "adversarial training lifts robust accuracy at eps " + epsTrain + ": " + baselineAtBudget.ToString("F3") + " -> " + hardenedPgd.ToString("F3"));
        Check(Accuracy(hardened, xs, labels, null) > 0.7, "the hardened model keeps its clean accuracy (" + Accuracy(hardened, xs, labels, null).ToString("F3") + ")");

        // 6. reproducibility
        var replay = new Net(930UL);
        replay.Train(xs, labels, 300, 0.4);
        ulong replayState = 7UL;
        attackState = 7UL;
        bool identical = Enumerable.Range(0, xs.Count).All(n =>
            Pgd(replay, xs[n], labels[n], eps, eps / 5.0, 20, ref replayState).SequenceEqual( Pgd(net, xs[n], labels[n], eps, eps / 5.0, 20, ref attackState)));
        Check(identical, "the same seed reproduces the same network and the same adversarial examples");

        Console.WriteLine("attack: eps " + eps + ", FGSM worst |d| " + worstFgsm.ToString("F6") + ", PGD worst |d| " + worstPgd.ToString("F6") + ", mean loss " + (lossBefore / 60).ToString("F4") + " -> " + (lossAfter / 60).ToString("F4"));
        Console.WriteLine("accuracy: clean " + clean.ToString("F3") + ", FGSM " + fgsmAccuracy.ToString("F3") + ", PGD " + pgdAccuracy.ToString("F3") + ", adversarially trained PGD " + hardenedPgd.ToString("F3") + " (clean " + Accuracy(hardened, xs, labels, null).ToString("F3") + ")");
        Console.WriteLine("PASS");
    }
}


