// Task 539 - Implement bitset convolution using FWHT (XOR convolution) small N. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// Walsh-Hadamard transform and the XOR convolution it diagonalises. The transform is the in-place
// butterfly `for len = 1,2,4,...: (u,v) -> (u+v, u-v)` in natural (Hadamard) order, so a needs no
// bit-reversal; the XOR convolution is c = F^-1(F(a) . F(b)) with F^-1 = F / N, and it reproduces
// the textbook c[k] = sum over i of a[i] * b[i ^ k] exactly in integers (all arithmetic is long,
// values stay far below overflow for the sizes used). The "bitset" reading of the same routine is
// also checked: for 0/1 indicators of two subsets of {0,1}^m, c[k] counts the pairs whose XOR is k,
// and for all-ones vectors it counts the N representations of every k.
//
// Documented checks: FWHT(c) = c * FWHT for scaled inputs (linearity) and F(F(a)) = N * a
// (involution); the delta vector is the convolution identity; the all-ones pair gives c[k] = N for
// every k = 2^m; for N in {1,2,4,8,16,32,64} with LCG integer vectors the transform result equals
// both the O(N^2) naive convolution and the independent single-loop form c[k] = sum_i a[i] b[i^k];
// convolution commutes and distributes over addition; a 0/1 bitset example reproduces small
// by-hand counts (in particular XOR of two disjoint singleton sets); and re-running the whole file
// of checks is deterministic because every vector comes from a fixed LCG.
using System;
using System.Collections.Generic;

internal static class Fwht
{
    /** In-place unnormalised Walsh-Hadamard transform (natural order, no bit reversal). */
    internal static void Transform(long[] a)
    {
        int n = a.Length;
        for (int len = 1; len < n; len <<= 1)
        {
            for (int start = 0; start < n; start += len << 1)
            {
                for (int j = 0; j < len; j++)
                {
                    long u = a[start + j];
                    long v = a[start + j + len];
                    a[start + j] = u + v;
                    a[start + j + len] = u - v;
                }
            }
        }
    }

    /** XOR convolution: c[k] = sum over i of a[i] * b[i ^ k]. */
    internal static long[] XorConvolution(long[] a, long[] b)
    {
        int n = a.Length;
        var fa = (long[])a.Clone();
        var fb = (long[])b.Clone();
        Transform(fa);
        Transform(fb);
        for (int i = 0; i < n; i++) fa[i] *= fb[i];
        Transform(fa);
        for (int i = 0; i < n; i++)
        {
            if (fa[i] % n != 0) throw new InvalidOperationException("inverse cost is not divisible at " + i);
            fa[i] /= n;
        }
        return fa;
    }

    internal static long[] NaivePairwise(long[] a, long[] b)
    {
        int n = a.Length;
        var c = new long[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) c[i ^ j] += a[i] * b[j];
        }
        return c;
    }

    /** Independent single-loop formulation: c[k] = sum_i a[i] * b[i ^ k]. */
    internal static long[] NaiveByTarget(long[] a, long[] b)
    {
        int n = a.Length;
        var c = new long[n];
        for (int k = 0; k < n; k++)
        {
            long sum = 0;
            for (int i = 0; i < n; i++) sum += a[i] * b[i ^ k];
            c[k] = sum;
        }
        return c;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int state = 539001;

    private static void LcgVector(long[] target, int limit)
    {
        for (int i = 0; i < target.Length; i++)
        {
            state = (state * 75 + 74) % 65537;
            target[i] = state % limit;
        }
    }

    private static string Render(long[] values)
    {
        return string.Join(",", values);
    }

    internal static void Main()
    {
        foreach (int m in new[] { 0, 1, 2, 3, 4, 5, 6 })
        {
            int n = 1 << m;
            var a = new long[n];
            var b = new long[n];
            LcgVector(a, 10);
            LcgVector(b, 10);
            long[] fast = Fwht.XorConvolution(a, b);
            long[] pairwise = Fwht.NaivePairwise(a, b);
            long[] byTarget = Fwht.NaiveByTarget(a, b);
            Check(Render(fast) == Render(pairwise), "FWHT matches the O(N^2) convolution for N=" + n);
            Check(Render(fast) == Render(byTarget), "FWHT matches the per-target formulation for N=" + n);
            Check(Render(Fwht.XorConvolution(b, a)) == Render(fast), "convolution commutes for N=" + n);
        }

        var a16 = new long[16];
        var b16 = new long[16];
        var c16 = new long[16];
        LcgVector(a16, 7);
        LcgVector(b16, 7);
        LcgVector(c16, 7);
        var sum = new long[16];
        for (int i = 0; i < 16; i++) sum[i] = b16[i] + c16[i];
        var left = Fwht.XorConvolution(a16, sum);
        var right = Fwht.XorConvolution(a16, b16);
        var right2 = Fwht.XorConvolution(a16, c16);
        for (int i = 0; i < 16; i++) right[i] += right2[i];
        Check(Render(left) == Render(right), "convolution distributes over addition");

        var scaleSource = new long[8];
        LcgVector(scaleSource, 5);
        var scaled = new long[8];
        for (int i = 0; i < 8; i++) scaled[i] = 3 * scaleSource[i];
        var tf = (long[])scaleSource.Clone();
        var tfScaled = (long[])scaled.Clone();
        Fwht.Transform(tf);
        Fwht.Transform(tfScaled);
        for (int i = 0; i < 8; i++) Check(tfScaled[i] == 3 * tf[i], "FWHT is linear at index " + i);

        var involution = new long[16];
        LcgVector(involution, 9);
        var original = (long[])involution.Clone();
        Fwht.Transform(involution);
        Fwht.Transform(involution);
        for (int i = 0; i < 16; i++) Check(involution[i] == 16 * original[i], "F(F(a)) = N * a at index " + i);

        var delta = new long[8];
        delta[0] = 1;
        var identitySource = new long[8];
        LcgVector(identitySource, 11);
        Check(Render(Fwht.XorConvolution(identitySource, delta)) == Render(identitySource),
                "the delta vector is the convolution identity");
        Check(Render(Fwht.XorConvolution(delta, delta)) == Render(delta), "delta squared is delta");

        foreach (int m in new[] { 1, 2, 3, 4, 5 })
        {
            int n = 1 << m;
            var ones = new long[n];
            for (int i = 0; i < n; i++) ones[i] = 1;
            long[] all = Fwht.XorConvolution(ones, ones);
            for (int k = 0; k < n; k++)
            {
                Check(all[k] == n, "all-ones convolution gives N at index " + k + " (N=" + n + ")");
            }

            // Bitset reading: 0/1 indicators count XOR pairs.
            var p = new long[n];
            var q = new long[n];
            LcgVector(p, 2);
            LcgVector(q, 2);
            long[] counts = Fwht.XorConvolution(p, q);
            long[] expected = Fwht.NaivePairwise(p, q);
            Check(Render(counts) == Render(expected), "bitset pair counts match for N=" + n);
            long total = 0;
            foreach (long value in counts) total += value;
            long supportP = 0;
            long supportQ = 0;
            foreach (long value in p) supportP += value;
            foreach (long value in q) supportQ += value;
            Check(total == supportP * supportQ, "the pair counts sum to |P| * |Q|");
        }

        // By-hand bitset example: {1} xor {2} = {3}, {1} xor {1} = {0}.
        var single = new long[4];
        var single2 = new long[4];
        single[1] = 1;
        single2[2] = 1;
        long[] hand = Fwht.XorConvolution(single, single2);
        Check(hand[3] == 1 && hand[0] == 0 && hand[1] == 0 && hand[2] == 0, "{1} xor {2} = {3}");
        Check(Fwht.XorConvolution(single, single)[0] == 1, "{1} xor {1} = {0}");

        Console.WriteLine("PASS");
    }
}
