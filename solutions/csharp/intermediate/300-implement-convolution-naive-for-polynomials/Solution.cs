// Task 300 - Implement convolution (naive) for polynomials. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Naive O(n*m) convolution: result[k] = sum of a[i] * b[k - i].</summary>
    public static long[] Convolve(long[] a, long[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<long>();
        var result = new long[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                result[i + j] += a[i] * b[j];
            }
        }
        return result;
    }

    /// <summary>Horner evaluation used as an independent oracle for the LCG cross-checks.</summary>
    public static long Evaluate(long[] polynomial, long x)
    {
        long value = 0;
        for (int i = polynomial.Length - 1; i >= 0; i--) value = value * x + polynomial[i];
        return value;
    }

    private static int seed = 70707;

    private static long[] LcgVector(int size)
    {
        var values = new long[size];
        for (int i = 0; i < size; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 10;
        }
        return values;
    }

    public static void Main()
    {
        Check(Same(Convolve(new long[] { 1, 2, 3 }, new long[] { 0, 1, 1 }), new long[] { 0, 1, 3, 5, 3 }), "sample");
        Check(Same(Convolve(new long[] { 5 }, new long[] { 7 }), new long[] { 35 }), "single terms");
        Check(Same(Convolve(new long[] { 5 }, new long[] { 0 }), new long[] { 0 }), "zero factor");
        Check(Convolve(Array.Empty<long>(), new long[] { 1, 2 }).Length == 0, "empty left");
        Check(Convolve(new long[] { 1, 2 }, Array.Empty<long>()).Length == 0, "empty right");

        var a = new long[] { 4, 1, 2, 3 };
        var b = new long[] { 2, 0, 5 };
        var c = new long[] { 1, 1, 0, 2 };
        Check(Same(Convolve(a, b), Convolve(b, a)), "commutativity");
        Check(Same(Convolve(Convolve(a, b), c), Convolve(a, Convolve(b, c))), "associativity");

        Check(Same(Convolve(a, new long[] { 1 }), a), "identity right");
        Check(Same(Convolve(new long[] { 1 }, a), a), "identity left");
        Check(Same(Convolve(new long[] { 0, 0, 1 }, new long[] { 1, 2, 3 }), new long[] { 0, 0, 1, 2, 3 }), "delta shift");
        Check(Same(Convolve(new long[] { 1, 0, 0 }, new long[] { 3, 4 }), new long[] { 3, 4, 0, 0 }), "trailing zeros");

        for (int trial = 0; trial < 120; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 6;
            seed = (seed * 75 + 74) % 65537;
            int m = 1 + seed % 6;
            var left = LcgVector(n);
            var right = LcgVector(m);
            var product = Convolve(left, right);
            Check(product.Length == n + m - 1, "length " + trial);
            Check(Same(product, Convolve(right, left)), "commutativity " + trial);
            foreach (long x in new long[] { 2, 3, 5 })
            {
                Check(Evaluate(product, x) == Evaluate(left, x) * Evaluate(right, x), "evaluation " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static bool Same(long[] a, long[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
