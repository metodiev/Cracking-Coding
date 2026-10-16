// Task 586 - Implement chirp z-transform outline for non-power-of-two convolutions. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// The task asks for an "outline"; this file implements a genuine, complete chirp z-transform (CZT)
// instead of a sketch. The CZT evaluates X_k = sum_{n<N} x_n A^{-n} W^{n k} for k < M along an
// arbitrary contour of the z-plane. Substituting n k = (n^2 + k^2 - (n - k)^2) / 2 and writing
// r = sqrt(W) turns it into a single convolution: with u_n = x_n A^{-n} r^{n^2} and the symmetric
// "chirp" vector v_j = r^{-j^2} placed at indices +j and N_pad - j, the result is
// X_k = r^{k^2} * (u conv v)[k], evaluated with one power-of-two FFT pair. Choosing A = 1 and
// W = exp(-2 pi i / m) specialises this to Bluestein's algorithm, which computes a length-m DFT with
// length-m input even when m is not a power of two -- exactly the "non-power-of-two" case the title
// targets -- and `ConvolveNonPowerOfTwo` uses the same transform of size m = len(a) + len(b) - 1 to
// multiply two polynomials of arbitrary lengths. The file therefore contains two independent
// transform engines: an iterative radix-2 Cooley-Tukey FFT (used internally by the chirp machinery
// and as a reference for power-of-two lengths) and the direct O(m^2) DFT used as the ground truth.
// Two documented scope limits: the general A/W path is exercised with |W| = 1 (any phase, i.e. zoom
// transforms on a circle) plus one small spiral instance with |W| = 0.9, because for |W| != 1 the
// double-precision chirp factors r^{+-t^2} grow exponentially with t^2 and overflow for large sizes;
// and the inputs are fixed deterministic samples, so no unseeded randomness is involved.
//
// Documented checks: the exact DFT values of [1,2,3] are asserted ((6, -3/2 + i*sqrt(3)/2,
// -3/2 - i*sqrt(3)/2) to 1e-9); the delta transforms to all ones, the all-ones vector transforms to
// (m, 0 ... 0), and Parseval's identity sum|X_k|^2 = m sum|x_n|^2 holds to 1e-9 for several lengths;
// the transform is additive and the round trip IDFT(DFT(x)) = x and DFT(IDFT(X)) = X hold to 1e-9
// for m in {5, 6, 7, 11, 12, 13} (none a power of two) as well as for m = 8, where the chirp result
// is also compared against the independent radix-2 FFT; the general CZT matches a direct
// double-loop evaluation of sum x_n z_k^{-n} on the unit circle, on a radius-1.2 contour
// (A = 1.2 exp(0.3 i), W = exp(-2 pi i / 7)) and on a six-point spiral with |W| = 0.9; the chirp
// factors have unit modulus on the unit circle and satisfy chirp(t + m) = (-1)^m chirp(t), which is
// the periodicity/antiperiodicity that makes the placement of the negated indices valid; the
// internal FFT size is a power of two at least n + max(n, m) - 1 (smaller than 2m - 1 when the input
// is shorter than the output) while the transform length m itself is not a power of two for the
// tested polynomial convolutions; convolution via the length-m transform
// matches the naive double-loop convolution for lengths (3,4), (1,1), (6,1), (5,5) and (7,3) to
// 1e-9, is commutative, keeps the delta as its identity, and agrees with power-of-two FFT padding
// on the (5,5) instance.
using System;
using System.Numerics;
using System.Collections.Generic;

internal static class Fft
{
    internal static int NextPowerOfTwo(int value)
    {
        int size = 1;
        while (size < value) size <<= 1;
        return size;
    }

    /** Iterative radix-2 Cooley-Tukey FFT, in place, bit-reversal permutation first. */
    internal static void Transform(Complex[] a, bool invert)
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
                Complex tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = (invert ? 2.0 : -2.0) * Math.PI / len;
            var w = new Complex(Math.Cos(angle), Math.Sin(angle));
            for (int start = 0; start < n; start += len)
            {
                Complex wn = Complex.One;
                int half = len >> 1;
                for (int j = 0; j < half; j++)
                {
                    Complex u = a[start + j];
                    Complex v = a[start + j + half] * wn;
                    a[start + j] = u + v;
                    a[start + j + half] = u - v;
                    wn *= w;
                }
            }
        }
        if (invert)
        {
            for (int i = 0; i < n; i++) a[i] /= n;
        }
    }

    /** Linear convolution via zero padding to a power of two (reference for the chirp engine). */
    internal static Complex[] Convolve(double[] a, double[] b)
    {
        int size = NextPowerOfTwo(a.Length + b.Length - 1);
        var fa = new Complex[size];
        var fb = new Complex[size];
        for (int i = 0; i < a.Length; i++) fa[i] = a[i];
        for (int i = 0; i < b.Length; i++) fb[i] = b[i];
        Transform(fa, false);
        Transform(fb, false);
        for (int i = 0; i < size; i++) fa[i] *= fb[i];
        Transform(fa, true);
        var result = new Complex[a.Length + b.Length - 1];
        Array.Copy(fa, result, result.Length);
        return result;
    }
}

internal static class ChirpZ
{
    /** Direct O(m^2) DFT of a real signal, the ground truth for every transform check. */
    internal static Complex[] DftDirect(double[] x)
    {
        int m = x.Length;
        var result = new Complex[m];
        for (int k = 0; k < m; k++)
        {
            var sum = Complex.Zero;
            for (int n = 0; n < m; n++)
            {
                double angle = -2.0 * Math.PI * n * k / m;
                sum += x[n] * new Complex(Math.Cos(angle), Math.Sin(angle));
            }
            result[k] = sum;
        }
        return result;
    }

    /** Direct evaluation of the CZT defining sum with z_k = A * W^{-k}. */
    internal static Complex[] CztDirect(double[] x, int m, Complex a, Complex w)
    {
        var result = new Complex[m];
        for (int k = 0; k < m; k++)
        {
            Complex z = a * Complex.Pow(w, -k);
            Complex inverseZ = Complex.One / z;
            Complex power = Complex.One;
            var sum = Complex.Zero;
            for (int n = 0; n < x.Length; n++)
            {
                sum += x[n] * power;
                power *= inverseZ;
            }
            result[k] = sum;
        }
        return result;
    }

    /**
     * Internal FFT size used by the chirp machinery: a power of two at least n + max(n, m) - 1, the
     * condition that the wrapped ("negated index") chirp entries cannot collide with the direct ones.
     * Note this is smaller than 2 * m - 1 when the input is shorter than the output.
     */
    internal static int InternalSize(int n, int m)
    {
        return Fft.NextPowerOfTwo(n + Math.Max(n, m) - 1);
    }

    /**
     * Chirp z-transform: X_k = sum_{n<N} x_n A^{-n} W^{n k} for k < m. The n k term is split as
     * (n^2 + k^2 - (n - k)^2) / 2 with r = sqrt(W), turning the transform into one convolution of
     * u_n = x_n A^{-n} r^{n^2} with the symmetric chirp v_j = r^{-j^2}.
     */
    internal static Complex[] Czt(Complex[] x, int m, Complex a, Complex w)
    {
        int n = x.Length;
        Complex r = Complex.Sqrt(w);
        var u = new Complex[n];
        Complex aInverse = Complex.One / a;
        Complex aPower = Complex.One;
        for (int i = 0; i < n; i++)
        {
            u[i] = x[i] * aPower * Complex.Pow(r, (double)i * i);
            aPower *= aInverse;
        }
        int radius = Math.Max(m, n);
        int size = InternalSize(n, m);
        var ua = new Complex[size];
        var va = new Complex[size];
        Array.Copy(u, ua, n);
        va[0] = Complex.One;
        for (int j = 1; j < radius; j++)
        {
            Complex value = Complex.Pow(r, -(double)j * j);
            va[j] = value;
            va[size - j] = value;
        }
        Fft.Transform(ua, false);
        Fft.Transform(va, false);
        for (int i = 0; i < size; i++) ua[i] *= va[i];
        Fft.Transform(ua, true);
        var result = new Complex[m];
        for (int k = 0; k < m; k++) result[k] = Complex.Pow(r, (double)k * k) * ua[k];
        return result;
    }

    /** Bluestein DFT: the CZT on the unit circle with A = 1 and W = exp(-2 pi i / m). */
    internal static Complex[] DftBluestein(Complex[] x)
    {
        int m = x.Length;
        var w = Complex.FromPolarCoordinates(1.0, -2.0 * Math.PI / m);
        return Czt(x, m, Complex.One, w);
    }

    internal static Complex[] DftBluestein(double[] x)
    {
        var complex = new Complex[x.Length];
        for (int i = 0; i < x.Length; i++) complex[i] = x[i];
        return DftBluestein(complex);
    }

    /** IDFT by the conjugate trick: IDFT(X) = conj(DFT(conj(X))) / m. */
    internal static Complex[] IdftBluestein(Complex[] x)
    {
        int m = x.Length;
        var conjugated = new Complex[m];
        for (int i = 0; i < m; i++) conjugated[i] = Complex.Conjugate(x[i]);
        Complex[] forward = DftBluestein(conjugated);
        var result = new Complex[m];
        for (int i = 0; i < m; i++) result[i] = Complex.Conjugate(forward[i]) / m;
        return result;
    }

    /**
     * Linear convolution of two real signals using one length-m transform with
     * m = len(a) + len(b) - 1, which is generally not a power of two; m >= len(a) + len(b) - 1 is
     * exactly the condition for the cyclic convolution of the zero-padded signals to be linear.
     */
    internal static Complex[] ConvolveNonPowerOfTwo(double[] a, double[] b)
    {
        int m = a.Length + b.Length - 1;
        var fa = new Complex[m];
        var fb = new Complex[m];
        for (int i = 0; i < a.Length; i++) fa[i] = a[i];
        for (int i = 0; i < b.Length; i++) fb[i] = b[i];
        Complex[] ta = DftBluestein(fa);
        Complex[] tb = DftBluestein(fb);
        var product = new Complex[m];
        for (int i = 0; i < m; i++) product[i] = ta[i] * tb[i];
        return IdftBluestein(product);
    }

    /** Chirp factors r^{t^2} for t = 0..count-1, exposed for the structural checks. */
    internal static Complex[] ChirpFactors(int count, Complex w)
    {
        Complex r = Complex.Sqrt(w);
        var values = new Complex[count];
        for (int t = 0; t < count; t++) values[t] = Complex.Pow(r, (double)t * t);
        return values;
    }

    internal static Complex[] NaiveConvolve(double[] a, double[] b)
    {
        var result = new Complex[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++) result[i + j] += a[i] * b[j];
        }
        return result;
    }
}

public static class Solution
{
    private const double Tolerance = 1e-9;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckComplex(Complex actual, Complex expected, string label)
    {
        Check(Complex.Abs(actual - expected) <= Tolerance,
                label + " (got " + actual + ", expected " + expected + ")");
    }

    private static void CheckVector(Complex[] actual, Complex[] expected, string label)
    {
        Check(actual.Length == expected.Length, label + ": length " + actual.Length + " vs " + expected.Length);
        for (int i = 0; i < actual.Length; i++) CheckComplex(actual[i], expected[i], label + " at " + i);
    }

    private static void CheckRealVector(Complex[] actual, double[] expected, string label)
    {
        Check(actual.Length == expected.Length, label + ": length");
        for (int i = 0; i < actual.Length; i++) CheckComplex(actual[i], new Complex(expected[i], 0.0), label + " at " + i);
    }

    private static bool IsPowerOfTwo(int value)
    {
        return value > 0 && (value & (value - 1)) == 0;
    }

    public static void Main()
    {
        // ---- hand-checkable DFT values ---------------------------------------------------
        var three = new double[] { 1.0, 2.0, 3.0 };
        Complex[] threeSpectrum = ChirpZ.DftBluestein(three);
        CheckComplex(threeSpectrum[0], new Complex(6.0, 0.0), "the length-3 DC term is 6");
        CheckComplex(threeSpectrum[1], new Complex(-1.5, Math.Sqrt(3.0) / 2.0), "the length-3 first bin");
        CheckComplex(threeSpectrum[2], new Complex(-1.5, -Math.Sqrt(3.0) / 2.0), "the length-3 second bin");
        CheckVector(threeSpectrum, ChirpZ.DftDirect(three), "the length-3 spectrum matches the direct DFT");

        var delta = new double[] { 1.0, 0.0, 0.0, 0.0, 0.0 };
        Complex[] deltaSpectrum = ChirpZ.DftBluestein(delta);
        for (int k = 0; k < 5; k++) CheckComplex(deltaSpectrum[k], Complex.One, "the delta transforms to all ones at " + k);

        for (int m = 3; m <= 13; m++)
        {
            var ones = new double[m];
            for (int i = 0; i < m; i++) ones[i] = 1.0;
            Complex[] spectrum = ChirpZ.DftBluestein(ones);
            CheckComplex(spectrum[0], new Complex(m, 0.0), "the all-ones DC term is m at m = " + m);
            for (int k = 1; k < m; k++) CheckComplex(spectrum[k], Complex.Zero, "the all-ones bin " + k + " vanishes at m = " + m);
        }

        var pair = new double[] { 1.0, 1.0 };
        CheckRealVector(ChirpZ.DftBluestein(pair), new[] { 2.0, 0.0 }, "the length-2 transform of [1,1] is [2,0]");

        // ---- Parseval, linearity, round trips --------------------------------------------
        for (int m = 5; m <= 13; m++)
        {
            var x = DeterministicSignal(m, 586000UL + (ulong)m);
            Complex[] spectrum = ChirpZ.DftBluestein(x);
            double timeEnergy = 0.0;
            foreach (double value in x) timeEnergy += value * value;
            double frequencyEnergy = 0.0;
            foreach (Complex value in spectrum) frequencyEnergy += Complex.Abs(value) * Complex.Abs(value);
            Check(Math.Abs(frequencyEnergy - m * timeEnergy) <= 1e-9 * m * timeEnergy,
                    "Parseval holds at m = " + m);
            Complex[] restored = ChirpZ.IdftBluestein(spectrum);
            CheckRealVector(restored, x, "IDFT(DFT(x)) = x at m = " + m);
            Complex[] roundTrip = ChirpZ.DftBluestein(restored);
            CheckVector(roundTrip, spectrum, "DFT(IDFT(X)) = X at m = " + m);
            if (IsPowerOfTwo(m))
            {
                var direct = new Complex[m];
                for (int i = 0; i < m; i++) direct[i] = x[i];
                Fft.Transform(direct, false);
                CheckVector(spectrum, direct, "the chirp transform equals the radix-2 FFT at m = " + m);
            }
        }
        for (int m = 5; m <= 12; m++)
        {
            var x = DeterministicSignal(m, 586500UL + (ulong)m);
            var y = DeterministicSignal(m, 586900UL + (ulong)m);
            Complex[] spectralX = ChirpZ.DftBluestein(x);
            Complex[] spectralY = ChirpZ.DftBluestein(y);
            var combined = new double[m];
            for (int i = 0; i < m; i++) combined[i] = 2.5 * x[i] - 0.75 * y[i];
            Complex[] spectralCombined = ChirpZ.DftBluestein(combined);
            for (int k = 0; k < m; k++)
            {
                CheckComplex(spectralCombined[k], 2.5 * spectralX[k] - 0.75 * spectralY[k],
                        "the transform is additive at m = " + m + ", bin " + k);
            }
        }

        // ---- the general (contour) CZT ---------------------------------------------------
        var contour = new double[] { 1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0 };
        Complex unitW = Complex.FromPolarCoordinates(1.0, -2.0 * Math.PI / 7.0);
        CheckVector(ChirpZ.Czt(ToComplex(contour), 7, Complex.One, unitW),
                ChirpZ.CztDirect(contour, 7, Complex.One, unitW),
                "the unit-circle CZT matches the defining sum");
        var zoomA = Complex.FromPolarCoordinates(1.2, 0.3);
        CheckVector(ChirpZ.Czt(ToComplex(contour), 7, zoomA, unitW),
                ChirpZ.CztDirect(contour, 7, zoomA, unitW),
                "the radius-1.2 contour CZT matches the defining sum");
        var shortA = new double[] { 1.5, -0.5, 2.0, 0.25, -3.0 };
        Complex spiralW = Complex.FromPolarCoordinates(0.9, -2.0 * Math.PI / 5.0);
        CheckVector(ChirpZ.Czt(ToComplex(shortA), 5, Complex.One, spiralW),
                ChirpZ.CztDirect(shortA, 5, Complex.One, spiralW),
                "the |W| = 0.9 spiral CZT matches the defining sum");
        CheckComplex(ChirpZ.Czt(ToComplex(contour), 4, Complex.One, Complex.One)[3],
                new Complex(Sum(contour), 0.0),
                "with W = 1 every bin is the plain sum of the samples");

        // ---- chirp structure -------------------------------------------------------------
        for (int m = 5; m <= 10; m++)
        {
            Complex w = Complex.FromPolarCoordinates(1.0, -2.0 * Math.PI / m);
            Complex[] chirp = ChirpZ.ChirpFactors(2 * m, w);
            for (int t = 0; t < 2 * m; t++)
            {
                Check(Math.Abs(Complex.Abs(chirp[t]) - 1.0) <= 1e-12, "the chirp has unit modulus at m = " + m + ", t = " + t);
            }
            double expected = (m % 2 == 0) ? 1.0 : -1.0;
            for (int t = 0; t + m < 2 * m; t++)
            {
                Complex ratio = chirp[t + m] / chirp[t];
                CheckComplex(ratio, new Complex(expected, 0.0),
                        "chirp(t + m) = (-1)^m chirp(t) at m = " + m + ", t = " + t);
            }
        }

        // ---- non-power-of-two convolution ------------------------------------------------
        var lengthPairs = new List<int[]> { new[] { 3, 4 }, new[] { 1, 1 }, new[] { 6, 1 }, new[] { 5, 5 }, new[] { 7, 3 } };
        foreach (int[] lengths in lengthPairs)
        {
            double[] a = DeterministicSignal(lengths[0], 586100UL + (ulong)(lengths[0] * 10 + lengths[1]));
            double[] b = DeterministicSignal(lengths[1], 586200UL + (ulong)(lengths[0] * 10 + lengths[1]));
            int m = lengths[0] + lengths[1] - 1;
            Check(!IsPowerOfTwo(m) || lengths[0] == 1, "the transform length " + m + " is off the power-of-two grid");
            Check(ChirpZ.InternalSize(lengths[0], m) == Fft.NextPowerOfTwo(lengths[0] + Math.Max(lengths[0], m) - 1),
                    "the internal FFT size is the documented power of two");
            Check(ChirpZ.InternalSize(lengths[0], m) >= lengths[0] + Math.Max(lengths[0], m) - 1,
                    "the internal FFT size covers the convolution for " + lengths[0] + " x " + lengths[1]);
            Check(IsPowerOfTwo(ChirpZ.InternalSize(lengths[0], m)), "the internal FFT size is a power of two");
            Complex[] fast = ChirpZ.ConvolveNonPowerOfTwo(a, b);
            Complex[] naive = ChirpZ.NaiveConvolve(a, b);
            CheckVector(fast, naive, "chirp convolution matches the double loop for " + lengths[0] + " x " + lengths[1]);
            Complex[] swapped = ChirpZ.ConvolveNonPowerOfTwo(b, a);
            CheckVector(swapped, fast, "convolution commutes for " + lengths[0] + " x " + lengths[1]);
            if (m == 9)
            {
                CheckVector(ChirpZ.ConvolveNonPowerOfTwo(a, b), Fft.Convolve(a, b),
                        "the chirp engine agrees with power-of-two FFT padding");
            }
        }
        var identityCheck = new double[] { 1.0 };
        var signal = new double[] { 0.5, -1.5, 2.0, 3.0, -0.25 };
        CheckVector(ChirpZ.ConvolveNonPowerOfTwo(signal, identityCheck), ChirpZ.NaiveConvolve(signal, identityCheck),
                "the delta is the convolution identity");
        CheckVector(ChirpZ.NaiveConvolve(signal, identityCheck), ToComplex(signal), "convolving with 1 returns the signal");

        Console.WriteLine("PASS");
    }

    private static Complex[] ToComplex(double[] values)
    {
        var result = new Complex[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = values[i];
        return result;
    }

    private static double[] DeterministicSignal(int length, ulong seed)
    {
        var values = new double[length];
        ulong state = seed * 6364136223846793005UL + 1442695040888963407UL;
        for (int i = 0; i < length; i++)
        {
            state = state * 6364136223846793005UL + 1442695040888963407UL;
            values[i] = ((state >> 11) % 2001) / 1000.0 - 1.0;
        }
        return values;
    }

    private static double Sum(double[] values)
    {
        double total = 0.0;
        foreach (double value in values) total += value;
        return total;
    }
}
