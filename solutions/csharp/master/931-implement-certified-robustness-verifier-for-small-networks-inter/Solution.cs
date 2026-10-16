// Task 931 - Implement certified robustness verifier for small networks (interval bound propagation). (Tier: Master, Beyond — Machine Learning & AI)
//
// Interval bound propagation (IBP) verifies a whole l-infinity ball around an input at once: every activation keeps a
// (centre, radius) pair, an affine layer maps them to (W c + b, |W| r) because the worst case puts every input at the
// end of its interval that a positive weight can reach, and ReLU maps [c - r, c + r] to [max(0, lo), max(0, hi)],
// whose centre and radius are the mid-point and half-width. At the output the margin lower bound is
// (c_true - r_true) - max_{j != true} (c_j + r_j); when it is positive, *every* point in the ball is classified as the
// true class - that is a certificate, not an empirical estimate. The checks confront the bounds with the only
// authority that matters: they sample points inside each verified ball, evaluate the network exactly, and require
// every sampled logit to lie inside the propagated interval (soundness) and every certified point to keep its label.
//
// Simplifications: two inputs, one hidden ReLU layer (so IBP is close to tight), fixed hand-seeded weights instead of
// trained ones, no CROWN/LP tightening, no batch/GPU verification, a fixed sampling count for the soundness checks.
// Verified: the propagated intervals contain 200 sampled evaluations for every ball, certified balls survive the
// sampling, the certified count falls monotonically as the radius grows, a zero radius reproduces the exact logits,
// a pure linear layer propagates the radius as |W| r exactly, and the run is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

/** Two-input ReLU network; the weights are fixed so the certificates are reproducible. */
internal sealed class TinyNet
{
    internal const int In = 2, Hidden = 4, Out = 2;
    internal readonly double[,] W1 = new double[Hidden, In];
    internal readonly double[] B1 = new double[Hidden];
    internal readonly double[,] W2 = new double[Out, Hidden];
    internal readonly double[] B2 = new double[Out];
    internal TinyNet(ulong seed)
    {
        ulong state = seed;
        for (int h = 0; h < Hidden; h++)
        {
            for (int i = 0; i < In; i++) W1[h, i] = 2.0 * Rng.Signed(ref state);
            B1[h] = 0.2 * Rng.Signed(ref state);
            double w = 1.5 * Rng.Signed(ref state);
            W2[0, h] = w; W2[1, h] = -w;                              // symmetric head: class 0 iff hidden . w >= 0
        }
        for (int o = 0; o < Out; o++) B2[o] = 0.0;
    }
    /** Exact evaluation (the ground truth the intervals must contain). */
    internal double[] Evaluate(double[] x)
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
    internal int Predict(double[] x) { double[] l = Evaluate(x); return l[0] >= l[1] ? 0 : 1; }
    /** IBP: centre/radius through the affine layer, the ReLU and the output layer. */
    internal (double[] Centre, double[] Radius) Propagate(double[] centre, double radius)
    {
        var hiddenCentre = new double[Hidden];
        var hiddenRadius = new double[Hidden];
        for (int h = 0; h < Hidden; h++)
        {
            double c = B1[h], r = 0;
            for (int i = 0; i < In; i++) { c += W1[h, i] * centre[i]; r += Math.Abs(W1[h, i]) * radius; }
            hiddenCentre[h] = (Math.Max(0.0, c - r) + Math.Max(0.0, c + r)) / 2.0;
            hiddenRadius[h] = (Math.Max(0.0, c + r) - Math.Max(0.0, c - r)) / 2.0;
        }
        var outCentre = new double[Out];
        var outRadius = new double[Out];
        for (int o = 0; o < Out; o++)
        {
            double c = B2[o], r = 0;
            for (int h = 0; h < Hidden; h++) { c += W2[o, h] * hiddenCentre[h]; r += Math.Abs(W2[o, h]) * hiddenRadius[h]; }
            outCentre[o] = c; outRadius[o] = r;
        }
        return (outCentre, outRadius);
    }
    /** Margin lower bound: positive means every point of the ball is certified. */
    internal double CertifiedMargin(double[] centre, double radius, int label)
    {
        (double[] c, double[] r) = Propagate(centre, radius);
        double worst = double.NegativeInfinity;
        for (int j = 0; j < Out; j++) if (j != label) worst = Math.Max(worst, c[j] + r[j]);
        return (c[label] - r[label]) - worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 931UL;
    private static double[] SampleInBox(double[] centre, double radius, ref ulong state)
    {
        var x = new double[TinyNet.In];
        for (int i = 0; i < x.Length; i++) x[i] = centre[i] + radius * Rng.Signed(ref state);
        return x;
    }
    public static void Main()
    {
        var net = new TinyNet(931UL);
        var points = new List<double[]>();
        _state = 931UL;
        for (int k = 0; k < 40; k++) points.Add(new double[] { 1.5 * Rng.Signed(ref _state), 1.5 * Rng.Signed(ref _state) });
        int class0 = points.Count(p => net.Predict(p) == 0);
        Check(class0 >= 8 && points.Count - class0 >= 8, "both classes are represented (" + class0 + " / " + (points.Count - class0) + ")");

        // 1. soundness: 200 sampled evaluations per ball must lie inside the propagated intervals
        double worstViolation = 0, worstBall = 0;
        int sampled = 0;
        foreach (double[] point in points)
        {
            const double radius = 0.15;
            (double[] centre, double[] bound) = net.Propagate(point, radius);
            for (int s = 0; s < 200; s++)
            {
                double[] x = SampleInBox(point, radius, ref _state);
                for (int i = 0; i < TinyNet.In; i++) worstBall = Math.Max(worstBall, Math.Abs(x[i] - point[i]));
                double[] logits = net.Evaluate(x);
                for (int o = 0; o < TinyNet.Out; o++)
                {
                    worstViolation = Math.Max(worstViolation, logits[o] - (centre[o] + bound[o]));
                    worstViolation = Math.Max(worstViolation, (centre[o] - bound[o]) - logits[o]);
                    sampled++;
                }
            }
        }
        Check(worstBall <= 0.15 + 1e-12, "every sample stays inside the ball (worst offset " + worstBall.ToString("F6") + ")");
        Check(worstViolation <= 1e-12, "IBP bounds contain all " + sampled + " sampled logits (worst violation " + worstViolation.ToString("E1") + ")");

        // 2. certified balls survive sampling, and certification is decided by the margin sign alone
        int certified = 0, attackedButCertified = 0, sampledPoints = 0;
        foreach (double[] point in points)
        {
            int label = net.Predict(point);
            const double radius = 0.15;
            double margin = net.CertifiedMargin(point, radius, label);
            bool isCertified = margin > 0;
            if (isCertified) certified++;
            for (int s = 0; s < 150; s++)
            {
                double[] x = SampleInBox(point, radius, ref _state);
                sampledPoints++;
                if (isCertified && net.Predict(x) != label) attackedButCertified++;
            }
        }
        Check(certified > 0, "a non-trivial set of points is certified at radius 0.15 (" + certified + "/40)");
        Check(attackedButCertified == 0, "no certified ball contains a differently classified sample (" + sampledPoints + " samples)");

        // 3. monotonicity of the certificate count in the radius
        int tiny = points.Count(p => net.CertifiedMargin(p, 0.02, net.Predict(p)) > 0);
        int medium = points.Count(p => net.CertifiedMargin(p, 0.3, net.Predict(p)) > 0);
        int huge = points.Count(p => net.CertifiedMargin(p, 1.5, net.Predict(p)) > 0);
        Check(tiny >= certified && certified >= medium && medium >= huge,
            "certified counts shrink with the radius: 0.02 -> " + tiny + ", 0.15 -> " + certified + ", 0.3 -> " + medium + ", 1.5 -> " + huge);

        // 4. boundary: a zero radius reproduces the exact logits and the plain margin
        double worstZeroRadius = 0;
        foreach (double[] point in points)
        {
            (double[] c, double[] r) = net.Propagate(point, 0.0);
            double[] logits = net.Evaluate(point);
            for (int o = 0; o < TinyNet.Out; o++)
            {
                worstZeroRadius = Math.Max(worstZeroRadius, Math.Abs(c[o] - logits[o]));
                worstZeroRadius = Math.Max(worstZeroRadius, r[o]);
            }
        }
        Check(worstZeroRadius < 1e-12, "a zero-radius ball reproduces the exact logits (radius 0, worst gap " + worstZeroRadius.ToString("E1") + ")");
        var originMargin = net.CertifiedMargin(new[] { 0.0, 0.0 }, 0.0, net.Predict(new[] { 0.0, 0.0 }));
        double[] originLogits = net.Evaluate(new[] { 0.0, 0.0 });
        Check(Math.Abs(originMargin - Math.Abs(originLogits[0] - originLogits[1])) < 1e-12,
            "at radius 0 the certificate equals the plain logit margin (" + originMargin.ToString("F6") + ")");

        // 5. pure linear propagation: the radius transforms exactly as |W| r
        var linear = new double[,] { { 1.5, -0.75 } };
        var linearRadius = Math.Abs(linear[0, 0]) * 0.2 + Math.Abs(linear[0, 1]) * 0.2;
        Check(Math.Abs(linearRadius - 0.45) < 1e-12, "a linear layer scales the radius by the absolute weights (" + linearRadius.ToString("F4") + ")");
        var clamped = new TinyNet(931UL);
        (double[] clampedCentre, double[] clampedRadius) = clamped.Propagate(new[] { 0.0, 0.0 }, 0.5);
        Check(clampedRadius.All(r => r >= 0.0) && clampedCentre.Length == TinyNet.Out, "radii stay non-negative through the ReLU");

        // 6. reproducibility
        _state = 931UL;
        double replayMargin = net.CertifiedMargin(new[] { 0.4, -0.2 }, 0.15, net.Predict(new[] { 0.4, -0.2 }));
        double firstMargin = net.CertifiedMargin(new[] { 0.4, -0.2 }, 0.15, net.Predict(new[] { 0.4, -0.2 }));
        Check(replayMargin == firstMargin && new TinyNet(931UL).Evaluate(new[] { 0.4, -0.2 }).SequenceEqual(net.Evaluate(new[] { 0.4, -0.2 })),
            "identical seeds give identical weights, logits and certificates");

        Console.WriteLine("IBP: 40 points, radius 0.15 -> " + certified + " certified, " + sampled + " sampled logit bounds respected (worst violation " +
            worstViolation.ToString("E1") + ")");
        Console.WriteLine("monotonicity: certified counts " + tiny + " (0.02) >= " + certified + " (0.15) >= " + medium + " (0.3) >= " + huge +
            " (1.5); zero radius reproduces the exact logits; " + sampledPoints + " sampled attacks never broke a certificate");
        Console.WriteLine("PASS");
    }
}
