// Task 302 - Convolution via number-theoretic transform (outline). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public const long Mod = 998244353L;
    public const long PrimitiveRoot = 3L;

    public static long Power(long baseValue, long exponent, long modulus)
    {
        long result = 1;
        baseValue %= modulus;
        while (exponent > 0)
        {
            if ((exponent & 1) == 1) result = result * baseValue % modulus;
            baseValue = baseValue * baseValue % modulus;
            exponent >>= 1;
        }
        return result;
    }

    /// <summary>In-place iterative NTT; `values` holds coefficients modulo Mod.</summary>
    public static void Ntt(long[] values, bool invert)
    {
        int n = values.Length;
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j |= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }

        for (int length = 2; length <= n; length <<= 1)
        {
            long root = Power(PrimitiveRoot, (Mod - 1) / length, Mod);
            if (invert) root = Power(root, Mod - 2, Mod);
            int half = length >> 1;
            for (int start = 0; start < n; start += length)
            {
                long w = 1;
                for (int k = 0; k < half; k++)
                {
                    long u = values[start + k];
                    long v = values[start + k + half] * w % Mod;
                    values[start + k] = (u + v) % Mod;
                    values[start + k + half] = ((u - v) % Mod + Mod) % Mod;
                    w = w * root % Mod;
                }
            }
        }

        if (invert)
        {
            long inverseN = Power(n, Mod - 2, Mod);
            for (int i = 0; i < n; i++) values[i] = values[i] * inverseN % Mod;
        }
    }

    public static long[] ConvolutionNtt(long[] a, long[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<long>();
        int size = 1;
        while (size < a.Length + b.Length - 1) size <<= 1;
        var left = new long[size];
        var right = new long[size];
        for (int i = 0; i < a.Length; i++) left[i] = a[i] % Mod;
        for (int i = 0; i < b.Length; i++) right[i] = b[i] % Mod;
        Ntt(left, false);
        Ntt(right, false);
        for (int i = 0; i < size; i++) left[i] = left[i] * right[i] % Mod;
        Ntt(left, true);
        return left.Take(a.Length + b.Length - 1).ToArray();
    }

    public static long[] ConvolutionNaiveMod(long[] a, long[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<long>();
        var outList = new long[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                outList[i + j] = (outList[i + j] + a[i] * b[j]) % Mod;
        return outList;
    }

    private static int seed = 7788;

    private static long[] LcgVector(int size)
    {
        var vector = new long[size];
        for (int i = 0; i < size; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            vector[i] = seed % 1000;
        }
        return vector;
    }

    private static string Join(long[] values) => string.Join(",", values);

    public static void Main()
    {
        Check(Power(PrimitiveRoot, Mod - 1, Mod) == 1, "Fermat root of unity");
        Check((Mod - 1) % (1 << 23) == 0 && (Mod - 1) / (1 << 23) == 119, "modulus shape");

        Check(Join(ConvolutionNtt(new long[] { 1, 2, 3 }, new long[] { 0, 1, 1 })) == "0,1,3,5,3", "classic");
        Check(Join(ConvolutionNtt(new long[] { 1, 1 }, new long[] { 1, 1 })) == "1,2,1", "square");
        Check(Join(ConvolutionNtt(new long[] { 7 }, new long[] { 9 })) == "63", "scalars");
        Check(ConvolutionNtt(Array.Empty<long>(), new long[] { 1 }).Length == 0, "empty operand");
        Check(Join(ConvolutionNtt(new long[] { Mod - 1 }, new long[] { 2 })) == (Mod - 2).ToString(), "negation mod p");

        var vector = new long[] { 3, 1, 4, 1, 5, 9, 2, 6 };
        var copy = (long[])vector.Clone();
        Ntt(copy, false);
        Check(Join(copy) != Join(vector), "forward transform changes the vector");
        Ntt(copy, true);
        Check(Join(copy) == Join(vector), "inverse restores the vector");

        for (int trial = 0; trial < 40; trial += 2)
        {
            var a = LcgVector(1 + (trial + seed) % 7);
            var b = LcgVector(1 + seed % 7);
            Check(Join(ConvolutionNtt(a, b)) == Join(ConvolutionNaiveMod(a, b)), "random forward " + trial);
            Check(Join(ConvolutionNtt(b, a)) == Join(ConvolutionNaiveMod(a, b)), "random swapped " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
