// Task 585 - Implement modular FFT for polynomial multiplication (NTT) small prime. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// Number-theoretic transform: the Cooley-Tukey FFT run in F_p instead of C, so every butterfly is
// exact integer arithmetic (no rounding, no epsilon) and the round trip is an identity up to the
// modular inverse of n. The file works modulo the NTT-friendly prime p = 119 * 2^23 + 1 =
// 998244353 with the primitive root 3, using the standard radix-2 iterative in-place butterfly with
// bit-reversal permutation: at stage `len` the twiddle is w = 3^((p-1)/len), replaced by its inverse
// for the inverse transform, which is finished by multiplying by n^-1 mod p. Polynomial
// multiplication is cyclic convolution of length n (`c = INTT(NTT(a) . NTT(b))` computes
// c[k] = sum_i a[i] b[(k-i) mod n]) and linear convolution is obtained by zero-padding to the next
// power of two above deg(a) + deg(b); the file also shows the documented wrap identity
// cyclic[k] = linear[k] + linear[k + n]. The "small prime" caveat is handled explicitly: one prime
// only recovers coefficients below p, so a second prime q = 5 * 2^25 + 1 = 167772161 (also with
// primitive root 3) is combined with the first by CRT to recover integer convolution coefficients
// below p * q ~ 1.67e17 exactly. Everything is deterministic: inputs come from the documented 64-bit
// LCG in `Lcg64` (MMIX constants), chosen over the repo's 16-bit LCG because these tests need many
// independent draws.
//
// Documented checks: for every power of two n in {2,...,1024} the chosen stage root satisfies
// w^n = 1 and w^(n/2) = p-1 (i.e. w really is a primitive n-th root of unity), and 3^((p-1)/2) =
// p-1 confirms 3 is a non-residue; INTT(NTT(a)) = a holds exactly (term-by-term integer equality) on
// LCG vectors of every size, and NTT(INTT(a)) = a likewise; NTT is additive; the transform of the
// delta is the all-ones vector and the inverse transform of the all-ones vector is the delta, so the
// delta is the convolution identity; the length-4 transform of [1,1,0,0] gives the hand-checkable
// values X = [2, 1+i, 0, 1-i] with i the modular square root of -1 = 3^((p-1)/4); NTT-based linear
// convolution equals the naive O(nm) mod-p convolution for every degree pair in 1..8 (zero padded, with
// the leading coefficient of each product asserted nonzero, so the high coefficients are genuinely
// checked); convolution commutes and is associative up
// to floating-free modular arithmetic on a fixed instance; the cyclic result equals the sum of the
// wrapped linear coefficients; multiplying anything by the delta is a no-op; the same construction
// modulo the second prime also satisfies the root and primitivity checks, and its convolution residues feed
// the CRT recombination; and the CRT recombination
// of the two NTT-derived residues reproduces the exact integer convolution (checked against
// `long`-arithmetic naive convolution) for coefficients large enough to exceed either prime alone
// and for a hand instance whose coefficient 1225 = 35^2 is verified literally.
using System;
using System.Collections.Generic;

internal static class Ntt
{
    internal const long P1 = 998244353L;   // 119 * 2^23 + 1
    internal const long P2 = 167772161L;   // 5 * 2^25 + 1
    internal const long Root = 3L;

    internal static long PowMod(long b, long e, long mod)
    {
        b %= mod;
        if (b < 0) b += mod;
        long result = 1L;
        while (e > 0)
        {
            if ((e & 1L) != 0L) result = result * b % mod;
            b = b * b % mod;
            e >>= 1;
        }
        return result;
    }

    /** Radix-2 iterative NTT of a power-of-two length in F_mod (bit reversal, in place). */
    internal static void Transform(long[] a, bool invert, long mod, long root)
    {
        int n = a.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0)
            {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j)
            {
                long tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            long w = PowMod(root, (mod - 1) / len, mod);
            if (invert) w = PowMod(w, mod - 2, mod);
            for (int start = 0; start < n; start += len)
            {
                long wn = 1L;
                int half = len >> 1;
                for (int j = 0; j < half; j++)
                {
                    long u = a[start + j];
                    long v = a[start + j + half] * wn % mod;
                    a[start + j] = (u + v) % mod;
                    a[start + j + half] = (u - v + mod) % mod;
                    wn = wn * w % mod;
                }
            }
        }
        if (invert)
        {
            long inv = PowMod(n % mod, mod - 2, mod);
            for (int i = 0; i < n; i++) a[i] = a[i] * inv % mod;
        }
    }

    internal static int NextPowerOfTwo(int value)
    {
        int size = 1;
        while (size < value) size <<= 1;
        return size;
    }

    /** Linear convolution mod p via zero padding, any lengths. */
    internal static long[] MultiplyLinear(long[] a, long[] b, long mod, long root)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<long>();
        int size = NextPowerOfTwo(a.Length + b.Length - 1);
        var fa = new long[size];
        var fb = new long[size];
        Array.Copy(a, fa, a.Length);
        Array.Copy(b, fb, b.Length);
        Transform(fa, false, mod, root);
        Transform(fb, false, mod, root);
        for (int i = 0; i < size; i++) fa[i] = fa[i] * fb[i] % mod;
        Transform(fa, true, mod, root);
        var result = new long[a.Length + b.Length - 1];
        Array.Copy(fa, result, result.Length);
        return result;
    }

    /** Cyclic convolution of length n, requiring both inputs to have exactly n entries. */
    internal static long[] MultiplyCyclic(long[] a, long[] b, long mod, long root)
    {
        int n = a.Length;
        var fa = (long[])a.Clone();
        var fb = (long[])b.Clone();
        Transform(fa, false, mod, root);
        Transform(fb, false, mod, root);
        for (int i = 0; i < n; i++) fa[i] = fa[i] * fb[i] % mod;
        Transform(fa, true, mod, root);
        return fa;
    }

    internal static long[] MultiplyNaive(long[] a, long[] b, long mod)
    {
        var result = new long[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                result[i + j] = (result[i + j] + a[i] * b[j]) % mod;
            }
        }
        return result;
    }

    /** Exact integer convolution in long arithmetic, used to validate the CRT recombination. */
    internal static long[] MultiplyNaiveInteger(int[] a, int[] b)
    {
        var result = new long[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++) result[i + j] += (long)a[i] * b[j];
        }
        return result;
    }

    /** Chinese remainder recombination of residues mod P1 and P2 into [0, P1 * P2). */
    internal static long CombineCrt(long r1, long r2)
    {
        long inverse = PowMod(P1 % P2, P2 - 2, P2);
        long delta = ((r2 - r1) % P2 + P2) % P2;
        long t = delta * inverse % P2;
        return r1 + P1 * t;
    }
}

internal static class Lcg64
{
    /** 64-bit LCG with the MMIX constants, used instead of the repo's 16-bit LCG. */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double a, double b, double tolerance, string label)
    {
        if (Math.Abs(a - b) > tolerance) throw new Exception("FAILED: " + label + " (" + a + " vs " + b + ")");
    }

    private static void CheckVectors(long[] a, long[] b, string label)
    {
        Check(a.Length == b.Length, label + ": length " + a.Length + " vs " + b.Length);
        for (int i = 0; i < a.Length; i++) Check(a[i] == b[i], label + ": coefficient " + i + " (" + a[i] + " vs " + b[i] + ")");
    }

    private static long[] RandomPoly(int degree, ulong seed, int bound)
    {
        var poly = new long[degree + 1];
        ulong state = seed;
        for (int i = 0; i <= degree; i++) poly[i] = Lcg64.NextBelow(ref state, bound);
        if (poly[degree] == 0) poly[degree] = 1;
        return poly;
    }

    public static void Main()
    {
        // ---- roots of unity --------------------------------------------------------------
        Check((119L * (1L << 23) + 1L) == Ntt.P1, "P1 factors as 119 * 2^23 + 1");
        Check((5L * (1L << 25) + 1L) == Ntt.P2, "P2 factors as 5 * 2^25 + 1");
        Check(Ntt.PowMod(Ntt.Root, (Ntt.P1 - 1) / 2, Ntt.P1) == Ntt.P1 - 1,
                "3 is a non-residue mod P1, so it is a primitive root of P1");
        Check(Ntt.PowMod(Ntt.Root, (Ntt.P2 - 1) / 2, Ntt.P2) == Ntt.P2 - 1,
                "3 is a non-residue mod P2 as well");
        for (int exponent = 1; exponent <= 10; exponent++)
        {
            int n = 1 << exponent;
            long w = Ntt.PowMod(Ntt.Root, (Ntt.P1 - 1) / n, Ntt.P1);
            Check(Ntt.PowMod(w, n, Ntt.P1) == 1L, "the stage root for n = " + n + " has order dividing n");
            Check(Ntt.PowMod(w, n >> 1, Ntt.P1) == Ntt.P1 - 1,
                    "the stage root for n = " + n + " is primitive (w^(n/2) = -1)");
        }
        long imaginary = Ntt.PowMod(Ntt.Root, (Ntt.P1 - 1) / 4, Ntt.P1);
        Check(imaginary * imaginary % Ntt.P1 == Ntt.P1 - 1, "3^((p-1)/4) squares to -1, the modular i");

        // ---- round trips -----------------------------------------------------------------
        for (int exponent = 1; exponent <= 10; exponent++)
        {
            int n = 1 << exponent;
            ulong state = 585000UL + (ulong)n;
            var values = new long[n];
            for (int i = 0; i < n; i++) values[i] = Lcg64.NextBelow(ref state, 1000000);
            var transformed = (long[])values.Clone();
            Ntt.Transform(transformed, false, Ntt.P1, Ntt.Root);
            var restored = (long[])transformed.Clone();
            Ntt.Transform(restored, true, Ntt.P1, Ntt.Root);
            CheckVectors(restored, values, "INTT(NTT(a)) = a at n = " + n);
            var forwardAgain = (long[])restored.Clone();
            Ntt.Transform(forwardAgain, false, Ntt.P1, Ntt.Root);
            CheckVectors(forwardAgain, transformed, "NTT(INTT(a)) = a at n = " + n);
        }

        // ---- delta behaviour and a hand-checkable transform ------------------------------
        var delta = new long[8];
        delta[0] = 1L;
        var deltaTransform = (long[])delta.Clone();
        Ntt.Transform(deltaTransform, false, Ntt.P1, Ntt.Root);
        for (int i = 0; i < 8; i++) Check(deltaTransform[i] == 1L, "the transform of the delta is all ones at " + i);
        var ones = new long[8];
        for (int i = 0; i < 8; i++) ones[i] = 1L;
        Ntt.Transform(ones, true, Ntt.P1, Ntt.Root);
        Check(ones[0] == 1L, "the inverse transform of ones is the delta");
        for (int i = 1; i < 8; i++) Check(ones[i] == 0L, "the inverse transform of ones vanishes at " + i);

        var two = new long[] { 1L, 1L, 0L, 0L };
        Ntt.Transform(two, false, Ntt.P1, Ntt.Root);
        Check(two[0] == 2L, "X[0] = 2 for [1,1,0,0]");
        Check(two[2] == 0L, "X[2] = 0 for [1,1,0,0]");
        Check((two[1] + two[3]) % Ntt.P1 == 2L, "X[1] + X[3] = 2 (the conjugate pair sums to 2)");
        Check(two[1] * two[3] % Ntt.P1 == 2L, "X[1] * X[3] = 2 (their product is 1 - i^2 = 2)");
        Check(two[1] % Ntt.P1 == (1L + imaginary) % Ntt.P1, "X[1] = 1 + i");
        Check(two[3] % Ntt.P1 == ((1L - imaginary) % Ntt.P1 + Ntt.P1) % Ntt.P1, "X[3] = 1 - i");

        // ---- linearity -------------------------------------------------------------------
        ulong linSeed = 585777UL;
        var linA = new long[16];
        var linB = new long[16];
        ulong state2 = linSeed;
        for (int i = 0; i < 16; i++)
        {
            linA[i] = Lcg64.NextBelow(ref state2, 1000);
            linB[i] = Lcg64.NextBelow(ref state2, 1000);
        }
        var sum = new long[16];
        for (int i = 0; i < 16; i++) sum[i] = (linA[i] + linB[i]) % Ntt.P1;
        var ta = (long[])linA.Clone();
        var tb = (long[])linB.Clone();
        var tsum = (long[])sum.Clone();
        Ntt.Transform(ta, false, Ntt.P1, Ntt.Root);
        Ntt.Transform(tb, false, Ntt.P1, Ntt.Root);
        Ntt.Transform(tsum, false, Ntt.P1, Ntt.Root);
        for (int i = 0; i < 16; i++) Check(tsum[i] == (ta[i] + tb[i]) % Ntt.P1, "NTT is additive at " + i);

        // ---- polynomial multiplication ---------------------------------------------------
        for (int degreeA = 1; degreeA <= 8; degreeA++)
        {
            for (int degreeB = 1; degreeB <= 8; degreeB++)
            {
                var a = RandomPoly(degreeA, 585100UL + (ulong)(degreeA * 31 + degreeB), 50);
                var b = RandomPoly(degreeB, 585200UL + (ulong)(degreeA * 31 + degreeB), 50);
                long[] fast = Ntt.MultiplyLinear(a, b, Ntt.P1, Ntt.Root);
                long[] naive = Ntt.MultiplyNaive(a, b, Ntt.P1);
                CheckVectors(fast, naive, "linear convolution deg " + degreeA + " x deg " + degreeB);
                Check(fast.Length == degreeA + degreeB + 1, "product degree is the sum for "
                        + degreeA + " x " + degreeB);
                Check(fast[degreeA + degreeB] != 0L, "the leading coefficient is nonzero for "
                        + degreeA + " x " + degreeB);
                long[] reversed = Ntt.MultiplyLinear(b, a, Ntt.P1, Ntt.Root);
                CheckVectors(fast, reversed, "multiplication commutes for " + degreeA + " x " + degreeB);
            }
        }

        var unit = new long[] { 1L };
        ulong edgeSeed = 585900UL;
        var edgePoly = new long[7];
        for (int i = 0; i < 7; i++) edgePoly[i] = Lcg64.NextBelow(ref edgeSeed, 100);
        CheckVectors(Ntt.MultiplyLinear(edgePoly, unit, Ntt.P1, Ntt.Root), edgePoly, "multiplying by 1 is a no-op");
        var empty = new long[0];
        Check(Ntt.MultiplyLinear(empty, edgePoly, Ntt.P1, Ntt.Root).Length == 0, "an empty polynomial gives an empty product");
        CheckNttAssociative();

        // ---- cyclic convolution and the wrap identity ------------------------------------
        var cycA = new long[] { 3L, 1L, 4L, 1L, 5L, 9L, 2L, 6L };
        var cycB = new long[] { 2L, 7L, 1L, 8L, 2L, 8L, 1L, 8L };
        long[] cyclic = Ntt.MultiplyCyclic(cycA, cycB, Ntt.P1, Ntt.Root);
        long[] linear = Ntt.MultiplyLinear(cycA, cycB, Ntt.P1, Ntt.Root);
        for (int k = 0; k < 8; k++)
        {
            long wrapped = linear[k];
            if (k + 8 < linear.Length) wrapped += linear[k + 8];
            Check(cyclic[k] == wrapped % Ntt.P1, "cyclic[k] = linear[k] + linear[k + 8] at " + k);
        }
        var directCyclic = new long[8];
        for (int k = 0; k < 8; k++)
        {
            for (int i = 0; i < 8; i++) directCyclic[k] = (directCyclic[k] + cycA[i] * cycB[(k - i + 8) % 8]) % Ntt.P1;
        }
        CheckVectors(cyclic, directCyclic, "the NTT cyclic convolution matches the double sum");

        // ---- second prime and CRT recombination ------------------------------------------
        for (int exponent = 1; exponent <= 8; exponent++)
        {
            int n = 1 << exponent;
            long w = Ntt.PowMod(Ntt.Root, (Ntt.P2 - 1) / n, Ntt.P2);
            Check(Ntt.PowMod(w, n, Ntt.P2) == 1L, "the second-prime stage root has order dividing n = " + n);
            Check(Ntt.PowMod(w, n >> 1, Ntt.P2) == Ntt.P2 - 1, "the second-prime root is primitive at n = " + n);
        }
        ulong crtSeed = 585333UL;
        var crtA = new int[6];
        var crtB = new int[6];
        for (int i = 0; i < 6; i++)
        {
            crtA[i] = 50000 + Lcg64.NextBelow(ref crtSeed, 40001);
            crtB[i] = 50000 + Lcg64.NextBelow(ref crtSeed, 40001);
        }
        var crtALong = new long[crtA.Length];
        var crtBLong = new long[crtB.Length];
        for (int i = 0; i < crtA.Length; i++)
        {
            crtALong[i] = crtA[i];
            crtBLong[i] = crtB[i];
        }
        long[] residue1 = Ntt.MultiplyLinear(crtALong, crtBLong, Ntt.P1, Ntt.Root);
        long[] residue2 = Ntt.MultiplyLinear(crtALong, crtBLong, Ntt.P2, Ntt.Root);
        long[] exact = Ntt.MultiplyNaiveInteger(crtA, crtB);
        Check(exact.Length == residue1.Length && exact.Length == residue2.Length, "all three products have the same degree");
        for (int i = 0; i < exact.Length; i++)
        {
            long combined = Ntt.CombineCrt(residue1[i], residue2[i]);
            Check(combined == exact[i], "CRT recovers coefficient " + i + " (" + combined + " vs " + exact[i] + ")");
            Check(exact[i] > Ntt.P1, "coefficient " + i + " exceeds P1, so two primes are needed");
        }
        var bigA = new long[] { 35L, 35L };
        var bigB = new long[] { 35L, 35L };
        long[] big1 = Ntt.MultiplyLinear(bigA, bigB, Ntt.P1, Ntt.Root);
        long[] big2 = Ntt.MultiplyLinear(bigA, bigB, Ntt.P2, Ntt.Root);
        Check(Ntt.CombineCrt(big1[1], big2[1]) == 2450L, "the middle coefficient of (35 + 35x)^2 is 2450");
        Check(Ntt.CombineCrt(big1[2], big2[2]) == 1225L, "the leading coefficient is 35^2 = 1225");
        Check(Ntt.PowMod(7, 0, Ntt.P1) == 1L, "the exponent zero case of PowMod is 1");
        Check(Ntt.PowMod(2, 10, Ntt.P1) == 1024L, "PowMod agrees with repeated squaring on 2^10");

        Console.WriteLine("PASS");
    }

    private static void CheckNttAssociative()
    {
        var p = new long[] { 2L, 3L, 5L, 7L, 11L };
        var q = new long[] { 1L, 4L, 9L, 2L };
        var r = new long[] { 6L, 5L, 3L };
        long[] left = Ntt.MultiplyLinear(Ntt.MultiplyLinear(p, q, Ntt.P1, Ntt.Root), r, Ntt.P1, Ntt.Root);
        long[] right = Ntt.MultiplyLinear(p, Ntt.MultiplyLinear(q, r, Ntt.P1, Ntt.Root), Ntt.P1, Ntt.Root);
        CheckVectors(left, right, "polynomial multiplication is associative");
    }
}
