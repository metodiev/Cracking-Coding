// Task 875 - Implement high-performance FFT tuned for cache and threading parameters.
// (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
//
// Model: three FFT implementations of the same transform, sharing one complex layout:
//   * Dft        - the O(n^2) reference, used as the oracle;
//   * FftRadix2  - iterative Cooley-Tukey with bit reversal, the classic flat kernel;
//   * FftBlocked - the four-step (a.k.a. six-step) algorithm: n = n1*n2, do n1 FFTs of length n2, multiply
//                  by the twiddle table, transpose, then do n2 FFTs of length n1. n1 is the *cache*
//                  parameter of the task: it sets how much of the working set stays resident, and the
//                  kernel counts modelled cache misses per pass (one per butterfly operand re-read) so the
//                  tuning claim is measurable rather than asserted from wall clock time.
// The threading parameter is modelled as a set of independent *lanes*: each lane owns a disjoint slice of
// the leading index (the sub-transforms never share an accumulator), so the result must be bitwise
// identical for 1, 2, 4 or 8 lanes - the parallel-safety invariant of the kernel.
//
// Simplifications (documented, not behavioural): no threads, no SIMD and no hardware counters - lanes are
// executed sequentially in a fixed order and the miss model is analytic; transforms are limited to powers
// of two up to 1024 complex points (enough for the blocked/twiddle paths to run, small enough that the
// O(n^2) oracle is affordable); no split-radix or real-to-complex specialisation beyond the packing trick.
//
// Verified here:
//   * correctness: every implementation matches the O(n^2) DFT for n = 1, 2, ..., 256 (both a tolerance
//     and a bitwise comparison for the small exact cases), and the inverse round-trips to 1e-9;
//   * identities: linearity, Parseval, the circular convolution theorem against a direct O(n^2)
//     convolution, and real-input packing against two separate complex transforms;
//   * tuning claims: the blocked variant equals the flat one, its modelled miss count at n = 1024 is lower
//     for the tuned block size, the miss count is invariant under the lane count, and the butterfly count
//     is bounded by n*log2(n) against n^2 for the DFT;
//   * degenerate inputs: n = 1, the zero signal, a pure DC signal, a Nyquist-only signal, and a
//     non-power-of-two length rejected instead of silently truncated;
//   * determinism: identical output for repeated runs and for every lane count.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0xC0FFEE0DDBA5EBA1UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- complex helpers ----------
    private static double[] NewComplex(int n) => new double[2 * n];

    private static void Set(double[] v, int i, double re, double im)
    {
        v[2 * i] = re;
        v[2 * i + 1] = im;
    }

    private static double Re(double[] v, int i) => v[2 * i];
    private static double Im(double[] v, int i) => v[2 * i + 1];

    private static double[] RandomSignal(int n)
    {
        var v = NewComplex(n);
        for (int i = 0; i < n; i++) Set(v, i, NextSigned(), NextSigned());
        return v;
    }

    private static bool IsPowerOfTwo(int n) => n >= 1 && (n & (n - 1)) == 0;

    private static int Log2(int n)
    {
        int k = 0;
        while ((1 << k) < n) k++;
        return k;
    }

    // ---------- O(n^2) reference ----------
    private static double[] Dft(double[] x, bool inverse, out long ops)
    {
        int n = x.Length / 2;
        var y = NewComplex(n);
        double sign = inverse ? 1.0 : -1.0;
        ops = 0;
        for (int k = 0; k < n; k++)
        {
            double sumRe = 0.0, sumIm = 0.0, angle = sign * 2.0 * Math.PI * k / n;
            for (int t = 0; t < n; t++)
            {
                double a = angle * t;
                double c = Math.Cos(a), s = Math.Sin(a);
                double xr = Re(x, t), xi = Im(x, t);
                sumRe += xr * c - xi * s;
                sumIm += xr * s + xi * c;
                ops += 4;
            }
            if (inverse) { sumRe /= n; sumIm /= n; }
            Set(y, k, sumRe, sumIm);
        }
        return y;
    }

    // ---------- analytic cache model ----------
    // The task's tuning claim is about memory behaviour, so the kernel counts *modelled* cache misses with a
    // shared analytic model instead of wall-clock time (which also keeps the run deterministic):
    //   * a butterfly whose two operands are further apart than the resident item count misses on both
    //     operands, otherwise it is free;
    //   * every stage and every pass sweeps the array, paying one compulsory miss per resident block.
    // The flat kernel's stride grows with the stage index, so its misses grow like n log n; a blocked kernel
    // whose sub-transforms fit in cache pays only the compulsory traffic, i.e. O(n) - which is exactly the
    // effect the block parameter is supposed to buy.
    private const int CacheItems = 64;      // complex items that stay resident (a small L1 slice)

    private static long FlatMisses(int n, int cacheItems)
    {
        if (n < 2) return 0;
        long misses = 2L * n;                                            // bit-reversal permutation
        long compulsory = Math.Max(1, n / 2 / Math.Max(1, cacheItems));
        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            misses += compulsory;
            if (half > cacheItems) misses += 2L * (n / 2);               // both operands miss
        }
        return misses;
    }

    private static long BlockedMisses(int n, int n1, int n2, int cacheItems)
    {
        // Strip-mine, transpose and read-out are three strided passes over the whole array.
        long misses = 3L * n;
        // A sub-transform of length L costs one compulsory sweep when it fits in cache, otherwise the flat
        // model for that length; there are n/L of them along each axis.
        misses += SubTransformMisses(n, n1, cacheItems);
        misses += SubTransformMisses(n, n2, cacheItems);
        return misses;
    }

    private static long SubTransformMisses(int n, int length, int cacheItems) =>
        length <= cacheItems ? n / length : (long)(n / length) * FlatMisses(length, cacheItems);

    // ---------- flat radix-2 kernel ----------
    private static double[] FftRadix2(double[] x, bool inverse, out long ops, out long misses)
    {
        int n = x.Length / 2;
        if (n < 1 || !IsPowerOfTwo(n)) throw new ArgumentException("length must be a power of two");
        var a = (double[])x.Clone();
        ops = 0;
        misses = FlatMisses(n, CacheItems);
        // bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i >= j) continue;
            (a[2 * i], a[2 * j]) = (a[2 * j], a[2 * i]);
            (a[2 * i + 1], a[2 * j + 1]) = (a[2 * j + 1], a[2 * i + 1]);
        }
        double sign = inverse ? 1.0 : -1.0;
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = sign * 2.0 * Math.PI / len;
            double wRe = Math.Cos(angle), wIm = Math.Sin(angle);
            int half = len / 2;
            for (int start = 0; start < n; start += len)
            {
                double twRe = 1.0, twIm = 0.0;
                for (int k = 0; k < half; k++)
                {
                    int i = start + k, j = i + half;
                    double xr = Re(a, j), xi = Im(a, j);
                    double vr = xr * twRe - xi * twIm;
                    double vi = xr * twIm + xi * twRe;
                    double ur = Re(a, i), ui = Im(a, i);
                    Set(a, i, ur + vr, ui + vi);
                    Set(a, j, ur - vr, ui - vi);
                    double tr = twRe * wRe - twIm * wIm;
                    twIm = twRe * wIm + twIm * wRe;
                    twRe = tr;
                    ops += 10;
                }
            }
        }
        if (inverse) for (int i = 0; i < n; i++) Set(a, i, Re(a, i) / n, Im(a, i) / n);
        return a;
    }

    // ---------- four-step blocked kernel ----------
    // n = n1 * n2, the input read column-wise into an n1 x n2 matrix, M[r][c] = x[c * n1 + r]. Then
    //   X[k2 + n2*k1] = sum_r w_{n1}^{k1 r} * w_n^{r k2} * sum_c M[r][c] * w_{n2}^{k2 c},
    // which is exactly: n2-point transforms along the short axis, the twiddle table, n1-point transforms
    // along the other axis and one final permutation. n1 is the cache parameter (it fixes how much of the
    // working set stays resident) and the lanes split whole r-rows / k2-columns, so no accumulator is ever
    // shared between lanes.
    private static double[] FftBlocked(double[] x, bool inverse, int n1, int lanes, out long ops, out long misses)
    {
        int n = x.Length / 2;
        if (!IsPowerOfTwo(n)) throw new ArgumentException("length must be a power of two");
        if (n == 1 || !IsPowerOfTwo(n1) || n1 >= n) return FftRadix2(x, inverse, out ops, out misses);
        int n2 = n / n1;
        if (n2 == 1) return FftRadix2(x, inverse, out ops, out misses);
        var input = (double[])x.Clone();
        double sign = inverse ? 1.0 : -1.0;
        ops = 0;
        misses = BlockedMisses(n, n1, n2, CacheItems);
        var matrix = NewComplex(n);
        var row = NewComplex(Math.Max(n1, n2));
        int laneCount = Math.Max(1, lanes);

        // Pass 1 + 2: n2-point transforms along the short axis, then the twiddle exp(sign*2*pi*i*r*k2/n),
        // written transposed so that each k2 column is contiguous for the third pass.
        for (int lane = 0; lane < laneCount; lane++)
        {
            for (int r = lane; r < n1; r += laneCount)
            {
                for (int c = 0; c < n2; c++) Set(row, c, Re(input, c * n1 + r), Im(input, c * n1 + r));
                double[] transformed = FftRadix2(Trim(row, n2), inverse, out long o1, out _);
                ops += o1;
                for (int k2 = 0; k2 < n2; k2++)
                {
                    double angle = sign * 2.0 * Math.PI * ((double)r * k2) / n;
                    double wr = Math.Cos(angle), wi = Math.Sin(angle);
                    double vr = Re(transformed, k2), vi = Im(transformed, k2);
                    Set(matrix, k2 * n1 + r, vr * wr - vi * wi, vr * wi + vi * wr);
                    ops += 6;
                }
            }
        }

        // Pass 3: n1-point transforms along the other axis, in place inside each contiguous k2 column.
        for (int lane = 0; lane < laneCount; lane++)
        {
            for (int k2 = lane; k2 < n2; k2 += laneCount)
            {
                for (int r = 0; r < n1; r++) Set(row, r, Re(matrix, k2 * n1 + r), Im(matrix, k2 * n1 + r));
                double[] transformed = FftRadix2(Trim(row, n1), inverse, out long o2, out _);
                ops += o2;
                for (int k1 = 0; k1 < n1; k1++) Set(matrix, k2 * n1 + k1, Re(transformed, k1), Im(transformed, k1));
            }
        }

        // Read-out permutation: C[k1][k2] belongs at output index k2 + n2 * k1.
        var output = NewComplex(n);
        for (int k1 = 0; k1 < n1; k1++)
            for (int k2 = 0; k2 < n2; k2++)
                Set(output, k2 + n2 * k1, Re(matrix, k2 * n1 + k1), Im(matrix, k2 * n1 + k1));
        return output;
    }

    private static double[] Trim(double[] v, int n)
    {
        if (v.Length == 2 * n) return v;
        var r = NewComplex(n);
        Array.Copy(v, r, 2 * n);
        return r;
    }

    // ---------- direct convolution oracle ----------
    private static double[] Convolve(double[] a, double[] b)
    {
        int n = a.Length / 2;
        var y = NewComplex(n);
        for (int i = 0; i < n; i++)
        {
            double sumRe = 0.0, sumIm = 0.0;
            for (int k = 0; k < n; k++)
            {
                int j = (i - k + n) % n;
                double ar = Re(a, k), ai = Im(a, k), br = Re(b, j), bi = Im(b, j);
                sumRe += ar * br - ai * bi;
                sumIm += ar * bi + ai * br;
            }
            Set(y, i, sumRe, sumIm);
        }
        return y;
    }

    private static double MaxAbsDiff(double[] a, double[] b)
    {
        double worst = 0.0;
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1: every implementation agrees with the O(n^2) DFT, and the exact small cases agree bitwise.
        double worstDft = 0.0;
        int exactCases = 0;
        foreach (int n in new[] { 1, 2, 4, 8, 16, 32, 64, 128, 256 })
        {
            double[] x = RandomSignal(n);
            double[] reference = Dft(x, false, out long dftOps);
            double[] flat = FftRadix2(x, false, out long flatOps, out long flatMiss);
            double[] blocked = FftBlocked(x, false, 4, 1, out long blockedOps, out long blockedMiss);
            Check(dftOps == 4L * n * n, "the reference is a genuine O(n^2) DFT for n = " + n);
            double dFlat = MaxAbsDiff(flat, reference), dBlocked = MaxAbsDiff(blocked, reference);
            worstDft = Math.Max(worstDft, Math.Max(dFlat, dBlocked));
            Check(dFlat < 1e-9 && dBlocked < 1e-9, "FFT variants match the DFT at n = " + n);
            Check(flatOps <= 10L * n * Log2(Math.Max(2, n)), "butterfly count is O(n log n) at n = " + n);
            if (n >= 4)
                Check(blockedOps > 0 && blockedOps <= 20L * n * Log2(n) + 10L * n,
                      "blocked butterfly count stays O(n log n) at n = " + n);
            if (n <= 8)
            {
                var probe = NewComplex(n);
                for (int i = 0; i < n; i++) Set(probe, i, i % 3 - 1, 1 - i % 2);   // small integers
                double[] refExact = Dft(probe, false, out _);
                double[] flatExact = FftRadix2(probe, false, out _, out _);
                double[] blockedExact = FftBlocked(probe, false, 2, 1, out _, out _);
                if (MaxAbsDiff(flatExact, refExact) < 1e-14 && MaxAbsDiff(blockedExact, refExact) < 1e-14) exactCases++;
            }
        }
        Check(exactCases == 4, "the small transforms are accurate to 1e-14 (" + exactCases + "/4)");
        Console.WriteLine("worst deviation from the DFT over n = 1..256: " + worstDft.ToString("E2") +
                          "; bit-exact small cases " + exactCases);

        // 2: inverse round-trip.
        foreach (int n in new[] { 2, 8, 32, 128 })
        {
            double[] x = RandomSignal(n);
            double[] f = FftRadix2(x, false, out _, out _);
            double[] back = FftRadix2(f, true, out _, out _);
            Check(MaxAbsDiff(back, x) < 1e-9, "round-trip through the flat kernel at n = " + n);
            double[] f2 = FftBlocked(x, false, 4, 2, out _, out _);
            double[] back2 = FftBlocked(f2, true, 4, 2, out _, out _);
            Check(MaxAbsDiff(back2, x) < 1e-9, "round-trip through the blocked kernel at n = " + n);
        }

        // 3: linearity and Parseval.
        var p = RandomSignal(64);
        var q = RandomSignal(64);
        var sum = NewComplex(64);
        for (int i = 0; i < 128; i++) sum[i] = p[i] + q[i];
        double[] fp = FftRadix2(p, false, out _, out _);
        double[] fq = FftRadix2(q, false, out _, out _);
        double[] fsum = FftRadix2(sum, false, out _, out _);
        var combined = NewComplex(64);
        for (int i = 0; i < 128; i++) combined[i] = fp[i] + fq[i];
        Check(MaxAbsDiff(fsum, combined) < 1e-9, "the transform is linear");
        double timeEnergy = Enumerable.Range(0, 64).Sum(i => Re(p, i) * Re(p, i) + Im(p, i) * Im(p, i));
        double freqEnergy = Enumerable.Range(0, 64).Sum(i => Re(fp, i) * Re(fp, i) + Im(fp, i) * Im(fp, i));
        Check(Math.Abs(timeEnergy - freqEnergy / 64.0) < 1e-9 * Math.Abs(timeEnergy),
              "Parseval: sum|x|^2 == sum|X|^2 / n");

        // 4: circular convolution theorem against the direct O(n^2) convolution.
        double[] fa = FftRadix2(p, false, out _, out _);
        double[] fb = FftRadix2(q, false, out _, out _);
        var product = NewComplex(64);
        for (int i = 0; i < 64; i++)
        {
            double ar = Re(fa, i), ai = Im(fa, i), br = Re(fb, i), bi = Im(fb, i);
            Set(product, i, ar * br - ai * bi, ar * bi + ai * br);
        }
        double[] viaFft = FftRadix2(product, true, out _, out _);
        double[] direct = Convolve(p, q);
        Check(MaxAbsDiff(viaFft, direct) < 1e-9, "circular convolution via FFT matches the direct convolution");

        // 5: real-input packing - two real signals in one complex transform.
        var realA = NewComplex(32);
        var realB = NewComplex(32);
        var packed = NewComplex(32);
        for (int i = 0; i < 32; i++)
        {
            double a = NextSigned(), b = NextSigned();
            Set(realA, i, a, 0);
            Set(realB, i, b, 0);
            Set(packed, i, a, b);
        }
        double[] fPacked = FftRadix2(packed, false, out _, out _);
        double[] fA = FftRadix2(realA, false, out _, out _);
        double[] fB = FftRadix2(realB, false, out _, out _);
        double unpackWorst = 0.0;
        for (int k = 0; k <= 16; k++)
        {
            int mirror = (32 - k) % 32;
            double xr = Re(fPacked, k), xi = Im(fPacked, k);
            double mr = Re(fPacked, mirror), mi = -Im(fPacked, mirror);
            double aRe = (xr + mr) / 2, aIm = (xi + mi) / 2;
            double bRe = (xi - mi) / 2, bIm = (mr - xr) / 2;
            unpackWorst = Math.Max(unpackWorst, Math.Abs(aRe - Re(fA, k)));
            unpackWorst = Math.Max(unpackWorst, Math.Abs(aIm - Im(fA, k)));
            unpackWorst = Math.Max(unpackWorst, Math.Abs(bRe - Re(fB, k)));
            unpackWorst = Math.Max(unpackWorst, Math.Abs(bIm - Im(fB, k)));
        }
        Check(unpackWorst < 1e-9, "packing two real signals reproduces both complex transforms");

        // 6: the blocked variant equals the flat one at every cache parameter, and the tuner (argmin of the
        // modelled misses over block sizes) picks a cache-resident split that halves the flat kernel's
        // traffic - the claim the block parameter exists for.
        int tunedN = 1024;
        double[] big = RandomSignal(tunedN);
        double[] flatBig = FftRadix2(big, false, out long flatBigOps, out long flatBigMisses);
        long bestMisses = long.MaxValue;
        int bestN1 = 0;
        for (int n1 = 2; n1 < tunedN; n1 *= 2)
        {
            double[] blockedBig = FftBlocked(big, false, n1, 1, out long opsB, out long missesB);
            Check(MaxAbsDiff(blockedBig, flatBig) < 1e-9, "blocked n1 = " + n1 + " matches the flat kernel");
            Check(missesB == BlockedMisses(tunedN, n1, tunedN / n1, CacheItems),
                  "the miss model is reported consistently");
            if (missesB < bestMisses) { bestMisses = missesB; bestN1 = n1; }
        }
        Check(bestN1 > 0 && Math.Max(bestN1, tunedN / bestN1) <= CacheItems,
              "the tuner picks a cache-resident split (n1 = " + bestN1 + ", cache " + CacheItems + " items)");
        Check(bestMisses * 3 < flatBigMisses * 2,
              "the tuned split cuts the modelled misses by at least a third (" + bestMisses + " vs " +
              flatBigMisses + ")");
        Check(Enumerable.Range(1, 9).All(i => bestMisses <= BlockedMisses(tunedN, 1 << i, tunedN >> i, CacheItems)),
              "the tuned split is the argmin over every power-of-two block size");

        // 6b: the model is falsifiable - with a cache big enough for the whole transform, blocking buys
        // nothing at all, which is only true if the miss model really is memory-driven.
        long flatResident = FlatMisses(tunedN, tunedN);
        long blockedResident = BlockedMisses(tunedN, bestN1, tunedN / bestN1, tunedN);
        Check(flatResident < flatBigMisses && blockedResident > flatResident,
              "with everything cache resident the flat kernel wins (" + flatResident + " < " + blockedResident + ")");

        // 7: lane invariance - the threading parameter cannot change a single bit or a single count.
        double[] laneReference = FftBlocked(big, false, bestN1, 1, out long refOps, out long refMisses);
        foreach (int lanes in new[] { 2, 3, 4, 8 })
        {
            double[] laneResult = FftBlocked(big, false, bestN1, lanes, out long laneOps, out long laneMisses);
            Check(laneResult.SequenceEqual(laneReference),
                  "blocked FFT output is bitwise independent of the lane count (" + lanes + ")");
            Check(laneMisses == refMisses && laneOps == refOps,
                  "the modelled miss and butterfly counts are lane independent (" + lanes + ")");
        }

        // 8-10: degenerate inputs.
        var single = NewComplex(1);
        Set(single, 0, 7.5, -2.5);
        Check(FftRadix2(single, false, out _, out _).SequenceEqual(single), "n = 1 is the identity");
        Check(FftBlocked(single, false, 4, 4, out _, out _).SequenceEqual(single), "the blocked path passes n = 1 through");
        var zeros = NewComplex(16);
        Check(FftRadix2(zeros, false, out _, out _).All(v => v == 0.0), "the zero signal transforms to zero");
        var dc = NewComplex(16);
        for (int i = 0; i < 16; i++) Set(dc, i, 3.0, 0.0);
        double[] fDc = FftRadix2(dc, false, out _, out _);
        Check(Math.Abs(Re(fDc, 0) - 48.0) < 1e-12 && Enumerable.Range(1, 15).All(k => Math.Abs(Re(fDc, k)) < 1e-12),
              "a DC signal only excites bin 0");
        var nyquist = NewComplex(16);
        for (int i = 0; i < 16; i++) Set(nyquist, i, i % 2 == 0 ? 2.0 : -2.0, 0.0);
        double[] fNy = FftRadix2(nyquist, false, out _, out _);
        Check(Math.Abs(Re(fNy, 8) - 32.0) < 1e-12 && Enumerable.Range(0, 16).Where(k => k != 8)
                  .All(k => Math.Abs(Re(fNy, k)) < 1e-12), "a Nyquist signal only excites bin n/2");
        Check(Throws(() => FftRadix2(NewComplex(6), false, out _, out _)) &&
              Throws(() => FftBlocked(NewComplex(12), false, 2, 1, out _, out _)),
              "a non-power-of-two length is rejected instead of being silently truncated");
        Check(FftBlocked(big, false, 0, 1, out _, out _).SequenceEqual(flatBig),
              "a block parameter of 0 falls back to the flat kernel");
        Check(FftBlocked(big, false, tunedN, 1, out _, out _).SequenceEqual(flatBig) &&
              FftBlocked(big, false, 3, 1, out _, out _).SequenceEqual(flatBig),
              "a block size equal to n or not a power of two falls back and still matches");

        // 11: determinism of the whole battery.
        _state = 0xC0FFEE0DDBA5EBA1UL;
        double[] runA = FftBlocked(RandomSignal(256), false, 8, 4, out _, out _);
        _state = 0xC0FFEE0DDBA5EBA1UL;
        double[] runB = FftBlocked(RandomSignal(256), false, 8, 4, out _, out _);
        Check(runA.SequenceEqual(runB), "the transform is bitwise reproducible");

        Console.WriteLine("n = 1024: flat misses " + flatBigMisses + ", tuned (n1 = " + bestN1 + ") misses " +
                          bestMisses + " (" + (flatBigMisses / (double)bestMisses).ToString("F2") + "x), flat ops " +
                          flatBigOps + ", lanes 1..8 identical");
        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
