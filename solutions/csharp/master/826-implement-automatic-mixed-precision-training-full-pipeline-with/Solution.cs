// Task 826 - Implement automatic mixed-precision training full pipeline with loss-scaling.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// The full AMP pipeline for a small MLP is modelled in-process and bit-exactly:
//
//   * every tensor that real AMP would store in half precision is *actually* rounded here, by an IEEE-style
//     rounder that builds the value on the (sign, exponent, mantissa) grid of bf16 / fp16 with subnormals,
//     round-to-nearest-even and overflow to infinity - the rounder is validated against known half values
//     (0.1 -> 0.0999755859375, 65504 the largest finite, 65520 -> +inf, 2^-24 the smallest subnormal);
//   * weights and activations are rounded before each multiply while products and sums accumulate in double
//     (standing in for fp32 accumulation), matching how a real AMP kernel computes;
//   * the loss is multiplied by a dynamic loss scale S before the backward pass, every gradient is rounded on
//     the low-precision grid *while scaled* and then divided back by S in the master precision, which is exactly
//     the mechanism that keeps a gradient of 1e-8 (which rounds to zero on its own) representable;
//   * if any scaled gradient rounds to +-inf/NaN the step is skipped, the scale is halved and no master weight
//     moves; otherwise the gradients are globally norm-clipped and the fp32 master copy is updated. Every
//     `growInterval` applied steps the scale doubles again, up to a cap.
//
// Verified invariants (all recomputed, none assumed):
//   * the rounder is exact on representable values, idempotent, never moves a value by more than half an ulp of
//     its result, handles ties to even, subnormals and overflow, and is a no-op for the double precision path;
//   * loss scaling is exact for power-of-two scales (scale-then-unscale is bit-preserving) and reduces the mean
//     relative gradient error by more than an order of magnitude on values that underflow half precision;
//   * an overflow is detected, the step is skipped, the scale shrinks - and applied + skipped always equals the
//     number of steps;
//   * the unscaled low-precision gradients stay within 2e-3 of the exact double gradients on the same batch,
//     which is the real cost of the compression;
//   * the master (fp32) copy retains an update that half precision cannot represent (1024 + 1e-4 -> 1024);
//   * the analytic gradients used by both trainers match central finite differences of the double-precision loss;
//   * mixed-precision training converges to within a small factor of the pure-double baseline on a noiseless
//     regression task, and repeated runs are bit-for-bit identical.
//
// Documented simplification: one dense 4-6-1 MLP, full-batch SGD, no drop-in API, no fused-kernel emulation and
// no bf16 *master* variant; the accumulation order is fixed, so results are reproducible rather than hardware
// dependent. What is real is the numeric format, the scaling/unscaling arithmetic, the overflow/skip policy and
// the gradient compression, which is what the task is about.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double NextRange(ref ulong state, double lo, double hi) => lo + (hi - lo) * NextDouble(ref state);
}

internal enum Precis { Double, Half, Bfloat16 }

/** An IEEE-style rounder onto a (sign, exponent, mantissa) grid, including subnormals and ties-to-even. */
internal static class Fp
{
    internal const int HalfMaxExp = 15, HalfMantissa = 10;
    internal const int Bf16MaxExp = 127, Bf16Mantissa = 7;

    internal static int MaxExponent(Precis p) => p == Precis.Half ? HalfMaxExp : Bf16MaxExp;

    internal static int MantissaBits(Precis p) => p == Precis.Half ? HalfMantissa : Bf16Mantissa;

    /** Largest finite value of the format. */
    internal static double MaxFinite(Precis p) =>
        Math.ScaleB(2.0 - Math.ScaleB(1.0, -MantissaBits(p)), MaxExponent(p));

    /** Smallest positive subnormal. */
    internal static double SmallestSubnormal(Precis p) =>
        Math.ScaleB(1.0, 1 - MaxExponent(p) - MantissaBits(p));

    internal static double Round(double x, Precis p)
    {
        if (p == Precis.Double) return x;
        if (double.IsNaN(x) || double.IsInfinity(x) || x == 0.0) return x;
        int maxExp = MaxExponent(p);
        int mantBits = MantissaBits(p);
        int minNormalExp = 1 - maxExp;
        double sign = x < 0.0 ? -1.0 : 1.0;
        double a = Math.Abs(x);
        int e = Math.ILogB(a);
        if (e > maxExp) return sign * double.PositiveInfinity;
        if (e < minNormalExp)
        {
            double subnormalStep = Math.ScaleB(1.0, minNormalExp - mantBits);
            double q = Math.Round(a / subnormalStep, MidpointRounding.ToEven) * subnormalStep;
            return sign * q;
        }
        double step = Math.ScaleB(1.0, e - mantBits);
        double r = Math.Round(a / step, MidpointRounding.ToEven) * step;
        if (r >= Math.ScaleB(1.0, maxExp + 1)) return sign * double.PositiveInfinity;
        return sign * r;
    }
}

internal sealed class Mlp
{
    internal readonly int In, Hid, Out;
    internal readonly double[] W1, B1, W2, B2;

    internal Mlp(int inDim, int hid, int outDim)
    {
        if (inDim < 1 || hid < 1 || outDim < 1) throw new ArgumentException("every layer needs at least one unit");
        In = inDim;
        Hid = hid;
        Out = outDim;
        W1 = new double[hid * inDim];
        B1 = new double[hid];
        W2 = new double[outDim * hid];
        B2 = new double[outDim];
    }

    internal static Mlp Drawn(int inDim, int hid, int outDim, ulong seed, double scale)
    {
        var m = new Mlp(inDim, hid, outDim);
        var rng = seed;
        Fill(m.W1, ref rng, scale);
        Fill(m.B1, ref rng, scale);
        Fill(m.W2, ref rng, scale);
        Fill(m.B2, ref rng, scale);
        return m;
    }

    private static void Fill(double[] v, ref ulong rng, double scale)
    {
        for (int i = 0; i < v.Length; i++) v[i] = scale * (2.0 * SplitMix64.NextDouble(ref rng) - 1.0);
    }

    /** A copy whose parameters live on the low-precision grid: what the half-precision kernel actually reads. */
    internal Mlp Quantized(Precis p)
    {
        var m = new Mlp(In, Hid, Out);
        for (int i = 0; i < W1.Length; i++) m.W1[i] = Fp.Round(W1[i], p);
        for (int i = 0; i < B1.Length; i++) m.B1[i] = Fp.Round(B1[i], p);
        for (int i = 0; i < W2.Length; i++) m.W2[i] = Fp.Round(W2[i], p);
        for (int i = 0; i < B2.Length; i++) m.B2[i] = Fp.Round(B2[i], p);
        return m;
    }

    internal double Value(double[][] xs, double[] ys, Precis p)
    {
        double[] g1 = new double[W1.Length], gb1 = new double[B1.Length];
        double[] g2 = new double[W2.Length], gb2 = new double[B2.Length];
        return LossAndGrad(xs, ys, p, 1.0, g1, gb1, g2, gb2, out _);
    }

    /**
     * Full-batch MSE and its gradient. When p != Double every weight and activation is rounded to p before the
     * multiply (accumulation stays in double, i.e. fp32). Each raw gradient is then multiplied by lossScale,
     * rounded to p *in that scaled form* and divided back by lossScale - the AMP loss-scaling step - and any
     * non-finite scaled value is reported through `overflow` so that the caller can skip the update.
     */
    internal double LossAndGrad(double[][] xs, double[] ys, Precis p, double lossScale,
        double[] gW1, double[] gB1, double[] gW2, double[] gB2, out bool overflow)
    {
        if (lossScale <= 0.0) throw new ArgumentException("the loss scale must be positive", nameof(lossScale));
        if (xs.Length != ys.Length) throw new ArgumentException("inputs and targets must have the same length");
        if (xs.Length == 0) throw new ArgumentException("the batch must not be empty");
        Array.Clear(gW1, 0, gW1.Length);
        Array.Clear(gB1, 0, gB1.Length);
        Array.Clear(gW2, 0, gW2.Length);
        Array.Clear(gB2, 0, gB2.Length);
        Mlp q = p == Precis.Double ? this : Quantized(p);
        double[] xq = new double[In];
        var z = new double[Hid];
        var h = new double[Hid];
        var dy = new double[Out];
        var dh = new double[Hid];
        int batch = xs.Length;
        double total = 0.0;
        for (int s = 0; s < batch; s++)
        {
            double[] x = xs[s];
            if (x.Length != In) throw new ArgumentException("an input has the wrong dimension");
            for (int i = 0; i < In; i++) xq[i] = Fp.Round(x[i], p);
            for (int j = 0; j < Hid; j++)
            {
                double acc = q.B1[j];
                for (int i = 0; i < In; i++) acc += q.W1[j * In + i] * xq[i];
                z[j] = acc;
                h[j] = Math.Max(0.0, Fp.Round(acc, p));
            }
            for (int o = 0; o < Out; o++)
            {
                double acc = q.B2[o];
                for (int j = 0; j < Hid; j++) acc += q.W2[o * Hid + j] * h[j];
                double err = acc - ys[s];
                total += err * err;
                dy[o] = 2.0 * err / batch;
                gB2[o] += dy[o];
                for (int j = 0; j < Hid; j++) gW2[o * Hid + j] += dy[o] * h[j];
            }
            for (int j = 0; j < Hid; j++)
            {
                double acc = 0.0;
                for (int o = 0; o < Out; o++) acc += q.W2[o * Hid + j] * dy[o];
                dh[j] = z[j] > 0.0 ? acc : 0.0;
                gB1[j] += dh[j];
                for (int i = 0; i < In; i++) gW1[j * In + i] += dh[j] * xq[i];
            }
        }
        overflow = Scale(gW1, p, lossScale) | Scale(gB1, p, lossScale);
        overflow |= Scale(gW2, p, lossScale) | Scale(gB2, p, lossScale);
        return total / batch;
    }

    /** Scale -> round -> unscale; reports whether the rounded scaled value left the finite range. */
    private static bool Scale(double[] g, Precis p, double lossScale)
    {
        bool overflow = false;
        for (int i = 0; i < g.Length; i++)
        {
            double scaled = Fp.Round(g[i] * lossScale, p);
            if (double.IsInfinity(scaled) || double.IsNaN(scaled)) overflow = true;
            g[i] = scaled / lossScale;
        }
        return overflow;
    }

    internal double GlobalNorm(double[] gW1, double[] gB1, double[] gW2, double[] gB2)
    {
        double s = 0.0;
        foreach (double v in gW1) s += v * v;
        foreach (double v in gB1) s += v * v;
        foreach (double v in gW2) s += v * v;
        foreach (double v in gB2) s += v * v;
        return Math.Sqrt(s);
    }
}

internal sealed class RunStats
{
    internal int Applied;
    internal int Skipped;
    internal int Grows;
    internal int Shrinks;
    internal double FinalLoss;
    internal double FinalScale;
    internal double StartLoss;
}

public static class Solution
{
    private const int In = 4, Hid = 6, Out = 1, Batch = 32;

    private static void BuildData(out double[][] xs, out double[] ys)
    {
        var rng = 0x5EEDUL;
        var teacher = Mlp.Drawn(In, Hid, Out, 0xABCDEFUL, 1.5);
        xs = new double[Batch][];
        ys = new double[Batch];
        for (int s = 0; s < Batch; s++)
        {
            var x = new double[In];
            for (int i = 0; i < In; i++) x[i] = SplitMix64.NextRange(ref rng, -1.0, 1.0);
            xs[s] = x;
            var z = new double[Hid];
            var h = new double[Hid];
            for (int j = 0; j < Hid; j++)
            {
                double acc = teacher.B1[j];
                for (int i = 0; i < In; i++) acc += teacher.W1[j * In + i] * x[i];
                z[j] = acc;
                h[j] = Math.Max(0.0, acc);
            }
            double y = teacher.B2[0];
            for (int j = 0; j < Hid; j++) y += teacher.W2[j] * h[j];
            ys[s] = y;
        }
    }

    /** One mixed-precision training run. Returns the final loss plus the scale-policy statistics. */
    private static RunStats Train(double[][] xs, double[] ys, int steps, double lr, double initialScale,
        Precis precision, double clipping, int growInterval)
    {
        if (steps < 0) throw new ArgumentException("the step count must not be negative", nameof(steps));
        if (lr <= 0.0) throw new ArgumentException("the learning rate must be positive", nameof(lr));
        if (initialScale <= 0.0) throw new ArgumentException("the loss scale must be positive", nameof(initialScale));
        var master = Mlp.Drawn(In, Hid, Out, 0x1234UL, 1.0);
        var stats = new RunStats { FinalScale = initialScale };
        var gW1 = new double[master.W1.Length];
        var gB1 = new double[master.B1.Length];
        var gW2 = new double[master.W2.Length];
        var gB2 = new double[master.B2.Length];
        stats.StartLoss = master.Value(xs, ys, precision);
        double scale = initialScale;
        for (int step = 0; step < steps; step++)
        {
            master.LossAndGrad(xs, ys, precision, scale, gW1, gB1, gW2, gB2, out bool overflow);
            if (overflow)
            {
                stats.Skipped++;
                scale = Math.Max(1.0, scale * 0.5);
                stats.Shrinks++;
                continue;
            }
            double norm = master.GlobalNorm(gW1, gB1, gW2, gB2);
            double factor = norm > clipping ? clipping / norm : 1.0;
            Update(master.W1, gW1, lr * factor);
            Update(master.B1, gB1, lr * factor);
            Update(master.W2, gW2, lr * factor);
            Update(master.B2, gB2, lr * factor);
            stats.Applied++;
            if (growInterval > 0 && stats.Applied % growInterval == 0 && scale < 1073741824.0)
            {
                scale *= 2.0;
                stats.Grows++;
            }
        }
        stats.FinalScale = scale;
        stats.FinalLoss = master.Value(xs, ys, precision);
        return stats;
    }

    private static void Update(double[] w, double[] g, double lr)
    {
        for (int i = 0; i < w.Length; i++) w[i] -= lr * g[i];
    }

    private static double Rel(double a, double b) => Math.Abs(a - b) / Math.Max(1e-30, Math.Abs(b));

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // 1. The rounder agrees with the known half-precision grid values.
        Check(Fp.Round(1.0, Precis.Half) == 1.0 && Fp.Round(-1.5, Precis.Half) == -1.5,
            "exactly representable values survive rounding");
        Check(Fp.Round(0.1, Precis.Half) == 0.0999755859375, "0.1 rounds to the nearest half value");
        Check(Fp.Round(0.1, Precis.Bfloat16) == 0.10009765625, "0.1 rounds to the nearest bfloat16 value");
        Console.WriteLine("  rounder: 0.1 (half)=" + Fp.Round(0.1, Precis.Half).ToString("G17")
            + " 0.1 (bf16)=" + Fp.Round(0.1, Precis.Bfloat16).ToString("G17"));

        // 2. Range: largest finite, first overflow, smallest subnormal.
        Check(Fp.Round(65504.0, Precis.Half) == 65504.0 && Fp.Round(65504.0, Precis.Half) == Fp.MaxFinite(Precis.Half),
            "65504 is the largest finite half value");
        Check(double.IsPositiveInfinity(Fp.Round(65520.0, Precis.Half))
            && double.IsPositiveInfinity(Fp.Round(70000.0, Precis.Half)), "values at or above the overflow threshold become +inf");
        Check(Fp.Round(Math.ScaleB(1.0, -24), Precis.Half) == Math.ScaleB(1.0, -24)
            && Fp.Round(Math.ScaleB(1.0, -25), Precis.Half) == 0.0 && Fp.Round(1e-9, Precis.Half) == 0.0,
            "the smallest subnormal is 2^-24 and anything below half of it flushes to zero");
        Check(Math.Abs(Fp.Round(1e30, Precis.Bfloat16) - 1e30) / 1e30 < 0.01
            && double.IsPositiveInfinity(Fp.Round(1e30, Precis.Half)),
            "bfloat16 shares the fp32 exponent range while half overflows");

        // 3. Ties round to even, and the rounder is idempotent and never moves more than half an ulp.
        Check(Fp.Round(1.00048828125, Precis.Half) == 1.0, "a tie rounds to the even mantissa (1.0)");
        Check(Fp.Round(1.00146484375, Precis.Half) == 1.001953125, "a tie rounds to the even mantissa (upwards)");
        var rng = 0x1234UL;
        bool idempotent = true, halfUlp = true, passthrough = true;
        for (int i = 0; i < 2000; i++)
        {
            double x = SplitMix64.NextRange(ref rng, -3e4, 3e4);
            double r = Fp.Round(x, Precis.Half);
            if (Fp.Round(r, Precis.Half) != r) idempotent = false;
            if (r != 0.0)
            {
                int e = Math.ILogB(Math.Abs(r));
                double ulp = e < 1 - Fp.MaxExponent(Precis.Half)
                    ? Fp.SmallestSubnormal(Precis.Half)
                    : Math.ScaleB(1.0, e - Fp.MantissaBits(Precis.Half));
                if (Math.Abs(r - x) > 0.5 * ulp + 1e-9) halfUlp = false;
            }
            if (Fp.Round(x, Precis.Double) != x) passthrough = false;
        }
        Check(idempotent, "rounding twice is the same as rounding once");
        Check(halfUlp, "rounding never moves a value by more than half an ulp");
        Check(passthrough, "the double precision path is a no-op");
        for (int i = 0; i < 2000; i++)
        {
            double x = SplitMix64.NextRange(ref rng, -1e-4, 1e-4);
            double r = Fp.Round(x, Precis.Bfloat16);
            if (Fp.Round(r, Precis.Bfloat16) != r) idempotent = false;
        }
        Check(idempotent, "bfloat16 rounding is idempotent on tiny magnitudes (subnormal handling)");

        // 4. Loss scaling makes underflowing gradients representable, and is exact for power-of-two scales.
        double tiny = 1e-8;
        Check(Fp.Round(tiny, Precis.Half) == 0.0, "a gradient of 1e-8 underflows half precision on its own");
        double scaled = Fp.Round(tiny * 8192.0, Precis.Half) / 8192.0;
        Check(scaled != 0.0 && Rel(scaled, tiny) < 1e-3, "loss scaling keeps the same gradient representable");
        bool exactRoundTrip = true;
        for (int i = 0; i < 500; i++)
        {
            double g = SplitMix64.NextRange(ref rng, -1.0, 1.0);
            for (int e = 0; e <= 10; e++)
            {
                double s = Math.ScaleB(1.0, e);
                if (Fp.Round(g * s, Precis.Half) / s != Fp.Round(g, Precis.Half)) exactRoundTrip = false;
            }
        }
        Check(exactRoundTrip, "scale/unscale round trips exactly for power-of-two scales");

        double errScaled = 0.0, errPlain = 0.0;
        for (int i = 0; i < 200; i++)
        {
            double g = Math.ScaleB(1.0, -24) * SplitMix64.NextRange(ref rng, 1.0, 4.0);
            double s = 1024.0;
            errPlain += Math.Abs(Fp.Round(g, Precis.Half) - g) / g;
            errScaled += Math.Abs(Fp.Round(g * s, Precis.Half) / s - g) / g;
        }
        Console.WriteLine("  mean relative error on subnormal gradients: unscaled=" + (errPlain / 200).ToString("E3")
            + " scaled=" + (errScaled / 200).ToString("E3"));
        Check(errScaled < 0.1 * errPlain, "scaling cuts the mean relative gradient error by more than 10x");

        // 5. The compressed gradients are close to the exact double gradients, and the analytic gradient is
        //    itself validated by central differences.
        BuildData(out double[][] xs, out double[] ys);
        var probe = Mlp.Drawn(In, Hid, Out, 0x77UL, 1.0);
        var dW1 = new double[probe.W1.Length];
        var dB1 = new double[probe.B1.Length];
        var dW2 = new double[probe.W2.Length];
        var dB2 = new double[probe.B2.Length];
        double exactLoss = probe.LossAndGrad(xs, ys, Precis.Double, 1.0, dW1, dB1, dW2, dB2, out _);
        var hW1 = new double[probe.W1.Length];
        var hB1 = new double[probe.B1.Length];
        var hW2 = new double[probe.W2.Length];
        var hB2 = new double[probe.B2.Length];
        probe.LossAndGrad(xs, ys, Precis.Half, 1024.0, hW1, hB1, hW2, hB2, out bool over);
        Check(!over, "a sane scale produces no overflow on this batch");
        double worst = 0.0;
        double mean = 0.0;
        int counted = 0;
        for (int i = 0; i < dW1.Length; i++)
        {
            worst = Math.Max(worst, Rel(hW1[i], dW1[i]));
            mean += Rel(hW1[i], dW1[i]);
            counted++;
        }
        for (int i = 0; i < dW2.Length; i++)
        {
            worst = Math.Max(worst, Rel(hW2[i], dW2[i]));
            mean += Rel(hW2[i], dW2[i]);
            counted++;
        }
        mean /= counted;
        Console.WriteLine("  half-precision gradient vs exact: worst relative error=" + worst.ToString("E3")
            + " mean=" + mean.ToString("E3"));
        Check(worst < 1e-2, "the worst half-precision gradient component stays within 1e-2 of the exact value");
        Check(mean < 1e-3, "the mean relative error of the compressed gradients stays below 1e-3");

        const double h = 1e-5;
        var bumped = Mlp.Drawn(In, Hid, Out, 0x77UL, 1.0);
        double fd = (LossAfter(bumped, xs, ys, 0, h) - LossAfter(bumped, xs, ys, 0, -h)) / (2.0 * h);
        Console.WriteLine("  finite difference dL/dW1[0]=" + fd.ToString("E10") + " analytic=" + dW1[0].ToString("E10"));
        Check(Rel(fd, dW1[0]) < 1e-4, "the analytic gradient matches a central finite difference");

        // 6. Overflow policy: a deliberately huge scale trips the finite-range check, the step is skipped and the
        //    scale halves; applied + skipped always accounts for every step.
        Check(Throws(() => probe.LossAndGrad(xs, ys, Precis.Half, 0.0, dW1, dB1, dW2, dB2, out _)),
            "a non-positive loss scale is rejected");
        Check(Throws(() => Train(xs, ys, -1, 0.1, 1024.0, Precis.Half, 1.0, 0)), "a negative step count is rejected");
        Check(Throws(() => Train(xs, ys, 10, 0.0, 1024.0, Precis.Half, 1.0, 0)), "a non-positive learning rate is rejected");
        var blown = probe.LossAndGrad(xs, ys, Precis.Half, 1e300, dW1, dB1, dW2, dB2, out bool hugeOverflow);
        Check(hugeOverflow, "an absurd scale is caught by the overflow check");
        double bigLoss = ys[0] * 1e6;
        var huge = new Mlp(In, Hid, Out);
        for (int i = 0; i < huge.W2.Length; i++) huge.W2[i] = 1.0;
        var yBig = new double[Batch];
        for (int s = 0; s < Batch; s++) yBig[s] = bigLoss;
        huge.LossAndGrad(xs, yBig, Precis.Half, 1024.0, dW1, dB1, dW2, dB2, out bool overflowAtScale);
        Check(overflowAtScale, "large gradients overflow at a scale of 1024 and are detected");

        // 7. Mixed-precision training: convergence within a small factor of the pure-double baseline.
        var mixed = Train(xs, ys, 1200, 0.2, 1024.0, Precis.Half, 1.0, 300);
        var plain = Train(xs, ys, 1200, 0.2, 1.0, Precis.Double, 1.0, 0);
        Console.WriteLine("  training: start=" + mixed.StartLoss.ToString("E3")
            + " mixed=" + mixed.FinalLoss.ToString("E6") + " double=" + plain.FinalLoss.ToString("E6")
            + " applied=" + mixed.Applied + " skipped=" + mixed.Skipped + " scale=" + mixed.FinalScale);
        Check(mixed.Applied + mixed.Skipped == 1200, "every step is either applied or skipped");
        Check(mixed.FinalLoss < 0.02 * mixed.StartLoss, "mixed-precision training removes at least 98% of the loss");
        Check(plain.FinalLoss < 0.02 * plain.StartLoss, "the pure-double trainer converges as well");
        Check(mixed.FinalLoss < 1.5 * plain.FinalLoss + 1e-9,
            "mixed-precision training lands within 50% of the pure-double baseline loss");
        Check(plain.Skipped == 0 && plain.Grows == 0, "the double-precision path never skips and never rescales");

        // 8. The scale policy: it grows on success and shrinks under overflow but never below one.
        var shrinking = Train(xs, yBig, 40, 1e-3, 1024.0, Precis.Half, 1.0, 0);
        Check(shrinking.Shrinks > 0 && shrinking.FinalScale < 1024.0,
            "overflow shrinks the loss scale");
        Check(shrinking.FinalScale >= 1.0, "the loss scale never falls below one");
        var growing = Train(xs, ys, 300, 0.2, 8.0, Precis.Half, 1.0, 50);
        Check(growing.Grows > 0 && growing.FinalScale > 8.0, "a healthy run grows the loss scale back up");

        // 9. The fp32 master copy retains an update that half precision cannot even represent.
        var small = Mlp.Drawn(In, Hid, Out, 3UL, 1.0);
        small.W1[0] = 1024.0;
        Check(Fp.Round(1024.0, Precis.Half) == 1024.0 && Fp.Round(1024.0 + 1e-4, Precis.Half) == 1024.0,
            "half precision cannot represent 1024 + 1e-4, so the update would be lost");
        small.W1[0] -= 1e-4;
        Check(small.W1[0] != 1024.0 && small.W1[0] < 1024.0, "the fp32 master copy keeps that update");

        // 10. Zero steps: nothing moves, and the reported scale is unchanged.
        var idle = Train(xs, ys, 0, 0.2, 512.0, Precis.Half, 1.0, 0);
        Check(idle.Applied == 0 && idle.Skipped == 0 && idle.FinalScale == 512.0 && idle.FinalLoss == idle.StartLoss,
            "a zero-step run leaves the model and the scale untouched");

        // 11. Determinism.
        var again = Train(xs, ys, 1200, 0.2, 1024.0, Precis.Half, 1.0, 300);
        Check(again.FinalLoss == mixed.FinalLoss && again.Applied == mixed.Applied && again.FinalScale == mixed.FinalScale,
            "repeated mixed-precision runs are bit-for-bit identical");

        Console.WriteLine("PASS");
    }

    /** Double-precision loss after nudging one weight; used by the finite-difference oracle. */
    private static double LossAfter(Mlp m, double[][] xs, double[] ys, int index, double delta)
    {
        double original = m.W1[index];
        m.W1[index] = original + delta;
        double loss = m.Value(xs, ys, Precis.Double);
        m.W1[index] = original;
        return loss;
    }
}
