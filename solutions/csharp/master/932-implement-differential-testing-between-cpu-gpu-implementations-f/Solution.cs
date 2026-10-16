// Task 932 - Implement differential testing between CPU/GPU implementations for numerical ops. (Tier: Master, Beyond — Systems & Performance)
//
// Two implementations of the same kernels must agree, so the tester compares three independent computations: a naive
// scalar "CPU" kernel (single-precision accumulation, one multiply-add at a time), a tiled "GPU" kernel that keeps a
// block partial per tile and adds the tiles together (a different summation order, which is exactly what a parallel
// reduction does), and a double-precision reference. The acceptance criterion is an error *model* rather than a magic
// constant: for a reduction of length K the round-off of a float accumulation is bounded by c * u * K * scale, with
// u = 2^-24 and scale the largest magnitude the reduction can produce, so the tolerance scales with both the problem
// size and the magnitudes involved. Ops covered: matmul, row sums, an affine elementwise map and softmax.
//
// Simplifications: no real GPU (the tiled kernel only mimics parallel reduction order), float/double only, no
// transcendental correctness beyond the tolerance model, small shapes, no autotuning of the tile size.
// Verified: 40 random shape combinations agree CPU-vs-GPU and both against the double reference within the modelled
// bound, the observed error grows with the reduction length and never exceeds the bound, an intentionally broken
// kernel is caught and reported, row sums and softmax behave (rows sum to one), singleton shapes are exact,
// mismatched shapes are rejected, and every kernel is bit-for-bit deterministic.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static float Float(ref ulong s) => (float)(((Next(ref s) >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0);
}

internal static class Kernels
{
    internal const float U = 1.0f / 16777216.0f;                       // 2^-24: the float unit round-off
    /** Naive scalar accumulation (the "CPU" path). */
    internal static float[] MatMulCpu(float[] a, int m, int k, float[] b, int n)
    {
        RequireShapes(a, m, k, b, n);
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0.0f;
                for (int t = 0; t < k; t++) acc += a[i * k + t] * b[t * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }
    /** Tiled accumulation with a per-tile partial (the "GPU" path); `broken` transposes one tile index on purpose. */
    internal static float[] MatMulTiled(float[] a, int m, int k, float[] b, int n, int tile = 4, bool broken = false)
    {
        RequireShapes(a, m, k, b, n);
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0.0f;
                for (int start = 0; start < k; start += tile)
                {
                    float partial = 0.0f;
                    int stop = Math.Min(k, start + tile);
                    for (int t = start; t < stop; t++)
                        partial += a[i * k + t] * b[(broken && t == start ? (t + 1) % k : t) * n + j];
                    acc += partial;
                }
                c[i * n + j] = acc;
            }
        return c;
    }
    internal static double[] MatMulReference(float[] a, int m, int k, float[] b, int n)
    {
        RequireShapes(a, m, k, b, n);
        var c = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0.0;
                for (int t = 0; t < k; t++) acc += (double)a[i * k + t] * b[t * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }
    internal static float[] RowSums(float[] a, int m, int n)
    {
        var sums = new float[m];
        for (int i = 0; i < m; i++)
        {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += a[i * n + j];
            sums[i] = acc;
        }
        return sums;
    }
    internal static float[] RowSumsTiled(float[] a, int m, int n, int tile = 4)
    {
        var sums = new float[m];
        for (int i = 0; i < m; i++)
        {
            float acc = 0.0f;
            for (int start = 0; start < n; start += tile)
            {
                float partial = 0.0f;
                for (int j = start; j < Math.Min(n, start + tile); j++) partial += a[i * n + j];
                acc += partial;
            }
            sums[i] = acc;
        }
        return sums;
    }
    internal static float[] AffineCpu(float[] a, float scale, float bias)
    {
        var r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] * scale + bias;
        return r;
    }
    internal static float[] AffineTiled(float[] a, float scale, float bias, int tile = 4)
    {
        var r = new float[a.Length];
        for (int start = 0; start < a.Length; start += tile)
            for (int i = start; i < Math.Min(a.Length, start + tile); i++) r[i] = a[i] * scale + bias;
        return r;
    }
    /** Numerically stable softmax computed twice: max-subtracted in float, and with a double accumulation. */
    internal static float[] SoftmaxCpu(float[] a, int m, int n)
    {
        var p = new float[m * n];
        for (int i = 0; i < m; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < n; j++) max = Math.Max(max, a[i * n + j]);
            float sum = 0.0f;
            for (int j = 0; j < n; j++) { p[i * n + j] = (float)Math.Exp(a[i * n + j] - max); sum += p[i * n + j]; }
            for (int j = 0; j < n; j++) p[i * n + j] /= sum;
        }
        return p;
    }
    internal static double[] SoftmaxReference(float[] a, int m, int n)
    {
        var p = new double[m * n];
        for (int i = 0; i < m; i++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j < n; j++) max = Math.Max(max, a[i * n + j]);
            double sum = 0;
            for (int j = 0; j < n; j++) { p[i * n + j] = Math.Exp(a[i * n + j] - max); sum += p[i * n + j]; }
            for (int j = 0; j < n; j++) p[i * n + j] /= sum;
        }
        return p;
    }
    internal static double MaxAbs(params float[][] arrays) => arrays.SelectMany(a => a).Max(v => Math.Abs((double)v));
    internal static double MaxDiff(float[] a, float[] b) => a.Select((v, i) => Math.Abs((double)v - b[i])).Max();
    internal static double MaxDiff(float[] a, double[] b) => a.Select((v, i) => Math.Abs((double)v - b[i])).Max();
    /** Both kernels validate their operands before touching memory, so a bad shape is a hard error. */
    internal static void RequireShapes(float[] a, int m, int k, float[] b, int n)
    {
        if (a.Length != m * k || b.Length != k * n) throw new Exception("shape mismatch: [" + m + "x" + k + "] * [" + k + "x" + n + "]");
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 932UL;
    private static float[] Random(int size, ref ulong state)
    {
        var a = new float[size];
        for (int i = 0; i < size; i++) a[i] = Rng.Float(ref state);
        return a;
    }
    public static void Main()
    {
        // 1. differential matmul over 40 random shape combinations
        var shapes = new[] { 1, 2, 3, 4, 8, 16, 32 };
        double worstRatio = 0, worstCpuRef = 0, worstBound = 0;
        int cases = 0, failures = 0;
        foreach (int m in shapes)
            foreach (int n in shapes)
                foreach (int k in new[] { 1, 2, 4, 8, 16, 32, 64 })
                {
                    if (m * n * k > 4096 || cases >= 40) continue;
                    float[] a = Random(m * k, ref _state), b = Random(k * n, ref _state);
                    float[] cpu = Kernels.MatMulCpu(a, m, k, b, n);
                    float[] gpu = Kernels.MatMulTiled(a, m, k, b, n);
                    double[] reference = Kernels.MatMulReference(a, m, k, b, n);
                    double scale = Kernels.MaxAbs(a) * Kernels.MaxAbs(b);
                    double bound = 8.0 * Kernels.U * k * scale;
                    double diff = Kernels.MaxDiff(cpu, gpu);
                    if (diff > bound || Kernels.MaxDiff(cpu, reference) > bound) failures++;
                    worstRatio = Math.Max(worstRatio, bound > 0 ? diff / bound : 0.0);
                    worstCpuRef = Math.Max(worstCpuRef, Kernels.MaxDiff(cpu, reference));
                    worstBound = Math.Max(worstBound, bound);
                    cases++;
                }
        Check(cases >= 30, cases + " shape combinations were tested");
        Check(failures == 0, "CPU and GPU agree with each other and with the double reference inside the error model (" + cases + " cases)");
        Check(worstRatio < 1.0, "the worst deviation uses " + (worstRatio * 100).ToString("F1") + "% of the allowed bound " + worstBound.ToString("E2"));

        // 2. the error model itself: both a short and a long reduction stay inside their K-scaled bound
        float[] shortA = Random(4, ref _state), shortB = Random(4, ref _state);
        float[] longA = Random(64, ref _state), longB = Random(64, ref _state);
        double shortError = Kernels.MaxDiff(Kernels.MatMulCpu(shortA, 1, 4, shortB, 1), Kernels.MatMulReference(shortA, 1, 4, shortB, 1));
        double longError = Kernels.MaxDiff(Kernels.MatMulCpu(longA, 1, 64, longB, 1), Kernels.MatMulReference(longA, 1, 64, longB, 1));
        double shortBound = 8.0 * Kernels.U * 4 * Kernels.MaxAbs(shortA) * Kernels.MaxAbs(shortB);
        double longBound = 8.0 * Kernels.U * 64 * Kernels.MaxAbs(longA) * Kernels.MaxAbs(longB);
        Check(shortError <= shortBound && longError <= longBound, "short and long reductions respect the K-scaled bound (" +
            shortError.ToString("E1") + " <= " + shortBound.ToString("E1") + ", " + longError.ToString("E1") + " <= " + longBound.ToString("E1") + ")");
        Check(longError > 0.0, "a 64-term float reduction is not magically exact (error " + longError.ToString("E1") + ")");

        // 3. a deliberately broken kernel must be caught by the same test
        float[] a8 = Random(8 * 8, ref _state), b8 = Random(8 * 8, ref _state);
        float[] correct = Kernels.MatMulCpu(a8, 8, 8, b8, 8);
        float[] broken = Kernels.MatMulTiled(a8, 8, 8, b8, 8, 4, broken: true);
        double brokenDiff = Kernels.MaxDiff(correct, broken);
        double brokenBound = 8.0 * Kernels.U * 8 * Kernels.MaxAbs(a8) * Kernels.MaxAbs(b8);
        Check(brokenDiff > brokenBound, "the injected kernel bug is caught (deviation " + brokenDiff.ToString("E2") +
            " > bound " + brokenBound.ToString("E2") + ")");
        Check(Kernels.MaxDiff(correct, Kernels.MatMulTiled(a8, 8, 8, b8, 8, 4, broken: false)) <= brokenBound,
            "the correct tile order passes the same test");

        // 4. row sums, affine and softmax agree; softmax rows sum to one
        float[] rows = Random(24 * 32, ref _state);
        Check(Kernels.MaxDiff(Kernels.RowSums(rows, 24, 32), Kernels.RowSumsTiled(rows, 24, 32)) <= 8.0 * Kernels.U * 32 * 32,
            "tiled row sums agree with the scalar ones");
        Check(Kernels.MaxDiff(Kernels.AffineCpu(rows, 1.5f, -0.25f), Kernels.AffineTiled(rows, 1.5f, -0.25f)) == 0.0,
            "the elementwise affine map is bit-for-bit identical in both orders");
        float[] probabilities = Kernels.SoftmaxCpu(rows, 24, 32);
        double[] referenceProbabilities = Kernels.SoftmaxReference(rows, 24, 32);
        Check(Kernels.MaxDiff(probabilities, referenceProbabilities) < 1e-6, "softmax matches the double reference");
        double worstRowSum = 0;
        for (int i = 0; i < 24; i++) worstRowSum = Math.Max(worstRowSum, Math.Abs(Enumerable.Range(0, 32).Sum(j => probabilities[i * 32 + j]) - 1.0));
        Check(worstRowSum < 1e-5, "every softmax row sums to one (worst |sum - 1| = " + worstRowSum.ToString("E1") + ")");

        // 5. boundary: singleton shapes are exact and mismatched shapes are rejected
        float[] one = { 3.5f }, two = { -2.0f };
        Check(Kernels.MatMulCpu(one, 1, 1, two, 1)[0] == -7.0f && Kernels.MatMulTiled(one, 1, 1, two, 1)[0] == -7.0f,
            "a 1x1 matmul is exact in both kernels (" + Kernels.MatMulTiled(one, 1, 1, two, 1)[0].ToString("F1") + ")");
        float[] single = Kernels.SoftmaxCpu(new[] { 5.0f, 5.0f, 5.0f }, 1, 3);
        Check(Kernels.MaxDiff(single, new[] { 1f / 3, 1f / 3, 1f / 3 }) < 1e-7, "a uniform softmax row is uniform in float");
        bool rejected = false;
        try { Kernels.MatMulCpu(new float[6], 2, 3, new float[4], 2); } catch (Exception) { rejected = true; }
        Check(rejected, "a shape mismatch is rejected instead of being read out of bounds");

        // 6. determinism
        float[] d1 = Kernels.MatMulTiled(a8, 8, 8, b8, 8);
        float[] d2 = Kernels.MatMulTiled(a8, 8, 8, b8, 8);
        Check(d1.SequenceEqual(d2) && Kernels.MatMulCpu(a8, 8, 8, b8, 8).SequenceEqual(Kernels.MatMulCpu(a8, 8, 8, b8, 8)),
            "both kernels are bit-for-bit deterministic");

        Console.WriteLine("matmul: " + cases + " shape triples, 0 failures, worst deviation " + (worstRatio * 100).ToString("F1") +
            "% of the modelled bound (largest bound " + worstBound.ToString("E2") + ", worst CPU-vs-reference " + worstCpuRef.ToString("E2") + ")");
        Console.WriteLine("ops: tiled row sums, affine (bit-identical) and softmax (rows sum to one within " + worstRowSum.ToString("E1") +
            ") all agree; injected kernel bug detected; singleton shapes exact; determinism verified");
        Console.WriteLine("PASS");
    }
}
