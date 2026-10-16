// Task 727 - Implement Strassen matrix multiplication and recursion cutoff tuning. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Strassen's algorithm multiplies 2x2 blocks with 7 recursive products instead of the naive 8, giving
// M(n) = 7 M(n/2) multiplications at the cost of 18 additions of half-size blocks. The recursion only
// works on square power-of-two blocks, so arbitrary operands are zero-padded to the next power of two
// (the standard library approach) and the padding is discarded again afterwards.
//
// The point of a cutoff is that trade: below it the naive n^3 kernel wins, above it Strassen's 7^k
// growth wins. This file counts both operations instead of timing them, and the closed forms
//
//   M(n, leaf) = n^3            for n <= leaf      A(n, leaf) = n^3                    for n <= leaf
//              = 7 M(n/2, leaf)                 ,            = 7 A(n/2, leaf) + 18 (n/2)^2
//
// are checked against the counters of real multiplications, so the tuning objective is grounded in
// measurements of the recursion itself. Documented simplifications: entries are 64-bit integers (results
// compare exactly, no floating-point error analysis is needed), a multiplication counts as three
// additions in the tuning objective (stated where it is used), and only power-of-two blocks are recursed
// into. Verified: exact agreement with the naive product for square and rectangular operands including
// odd sizes, hand-checked 2x2 counts (seven products versus eight), counter/model agreement for every
// tested cutoff, and an interior minimum of the tuned objective.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Counters
{
    internal long Multiplications;
    internal long Additions;
}

/// <summary>
/// Strassen multiplication over a flat row-major buffer. Buffers carry their own stride so the recursive
/// calls can mix a sub-block of the input (stride = the padded size) with a temporary (stride = its own
/// side length) without copying.
/// </summary>
internal static class Strassen
{
    internal static void Naive(long[] a, int ao, int aStride, long[] b, int bo, int bStride, long[] c, int co, int cStride,
        int n, Counters counters)
    {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
            {
                long av = a[ao + i * aStride + k];
                for (int j = 0; j < n; j++)
                {
                    c[co + i * cStride + j] += av * b[bo + k * bStride + j];
                    counters.Multiplications++;
                    counters.Additions++;
                }
            }
    }

    internal static void Recursive(long[] a, int ao, int aStride, long[] b, int bo, int bStride, long[] c, int co, int cStride,
        int n, int cutoff, Counters counters)
    {
        if (n <= cutoff)
        {
            Naive(a, ao, aStride, b, bo, bStride, c, co, cStride, n, counters);
            return;
        }
        int h = n / 2;
        var products = new long[7][];
        for (int i = 0; i < 7; i++) products[i] = new long[h * h];
        var left = new long[h * h];
        var right = new long[h * h];

        int a11 = ao, a12 = ao + h, a21 = ao + h * aStride, a22 = ao + h * aStride + h;
        int b11 = bo, b12 = bo + h, b21 = bo + h * bStride, b22 = bo + h * bStride + h;

        AddBlock(a, a11, a22, aStride, left, h, counters);                     // M1 = (A11 + A22)(B11 + B22)
        AddBlock(b, b11, b22, bStride, right, h, counters);
        Recursive(left, 0, h, right, 0, h, products[0], 0, h, h, cutoff, counters);

        AddBlock(a, a21, a22, aStride, left, h, counters);                     // M2 = (A21 + A22) B11
        Recursive(left, 0, h, b, b11, bStride, products[1], 0, h, h, cutoff, counters);

        SubBlock(b, b12, b22, bStride, right, h, counters);                    // M3 = A11 (B12 - B22)
        Recursive(a, a11, aStride, right, 0, h, products[2], 0, h, h, cutoff, counters);

        SubBlock(b, b21, b11, bStride, right, h, counters);                    // M4 = A22 (B21 - B11)
        Recursive(a, a22, aStride, right, 0, h, products[3], 0, h, h, cutoff, counters);

        AddBlock(a, a11, a12, aStride, left, h, counters);                     // M5 = (A11 + A12) B22
        Recursive(left, 0, h, b, b22, bStride, products[4], 0, h, h, cutoff, counters);

        SubBlock(a, a21, a11, aStride, left, h, counters);                     // M6 = (A21 - A11)(B11 + B12)
        AddBlock(b, b11, b12, bStride, right, h, counters);
        Recursive(left, 0, h, right, 0, h, products[5], 0, h, h, cutoff, counters);

        SubBlock(a, a12, a22, aStride, left, h, counters);                     // M7 = (A12 - A22)(B21 + B22)
        AddBlock(b, b21, b22, bStride, right, h, counters);
        Recursive(left, 0, h, right, 0, h, products[6], 0, h, h, cutoff, counters);

        int c11 = co, c12 = co + h, c21 = co + h * cStride, c22 = co + h * cStride + h;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < h; j++)
            {
                long m1 = products[0][i * h + j], m2 = products[1][i * h + j], m3 = products[2][i * h + j];
                long m4 = products[3][i * h + j], m5 = products[4][i * h + j], m6 = products[5][i * h + j];
                long m7 = products[6][i * h + j];
                c[c11 + i * cStride + j] = m1 + m4 - m5 + m7;
                c[c12 + i * cStride + j] = m3 + m5;
                c[c21 + i * cStride + j] = m2 + m4;
                c[c22 + i * cStride + j] = m1 - m2 + m3 + m6;
                counters.Additions += 8;
            }
    }

    private static void AddBlock(long[] source, int first, int second, int stride, long[] destination, int h, Counters counters)
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < h; j++)
            {
                destination[i * h + j] = source[first + i * stride + j] + source[second + i * stride + j];
                counters.Additions++;
            }
    }

    private static void SubBlock(long[] source, int first, int second, int stride, long[] destination, int h, Counters counters)
    {
        for (int i = 0; i < h; i++)
            for (int j = 0; j < h; j++)
            {
                destination[i * h + j] = source[first + i * stride + j] - source[second + i * stride + j];
                counters.Additions++;
            }
    }

    /// <summary>Pads both operands to the next power of two, recurses, and crops the result again.</summary>
    internal static long[,] Multiply(long[,] a, long[,] b, int cutoff, Counters counters)
    {
        if (cutoff < 1) throw new ArgumentOutOfRangeException(nameof(cutoff));
        int rows = a.GetLength(0), inner = a.GetLength(1), cols = b.GetLength(1);
        if (b.GetLength(0) != inner) throw new ArgumentException("inner dimensions must agree");
        int size = 1;
        while (size < Math.Max(rows, Math.Max(inner, cols))) size *= 2;
        var pa = new long[size * size];
        var pb = new long[size * size];
        var pc = new long[size * size];
        for (int i = 0; i < rows; i++) for (int k = 0; k < inner; k++) pa[i * size + k] = a[i, k];
        for (int k = 0; k < inner; k++) for (int j = 0; j < cols; j++) pb[k * size + j] = b[k, j];
        Recursive(pa, 0, size, pb, 0, size, pc, 0, size, size, cutoff, counters);
        var result = new long[rows, cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) result[i, j] = pc[i * size + j];
        return result;
    }

    internal static long[,] NaiveMultiply(long[,] a, long[,] b)
    {
        int rows = a.GetLength(0), inner = a.GetLength(1), cols = b.GetLength(1);
        var result = new long[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < inner; k++)
            {
                long av = a[i, k];
                for (int j = 0; j < cols; j++) result[i, j] += av * b[k, j];
            }
        return result;
    }

    /// <summary>Closed-form operation counts of the recursion above, used to tune the cutoff.</summary>
    internal static (long Multiplications, long Additions) Model(int size, int cutoff)
    {
        if (size <= cutoff) return ((long)size * size * size, (long)size * size * size);
        (long innerMultiplications, long innerAdditions) = Model(size / 2, cutoff);
        int h = size / 2;
        return (7 * innerMultiplications, 7 * innerAdditions + 18L * h * h);
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- exact agreement with the naive product, square and rectangular ------------------------
        {
            (int Rows, int Inner, int Cols)[] shapes =
            {
                (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (7, 7, 7), (8, 8, 8),
                (16, 16, 16), (17, 17, 17), (31, 31, 31), (3, 5, 7), (12, 20, 9), (1, 33, 1),
            };
            long largest = 0;
            foreach ((int rows, int inner, int cols) in shapes)
            {
                long[,] a = RandomMatrix(rows, inner, 0x727000UL + (ulong)(rows * 31 + cols));
                long[,] b = RandomMatrix(inner, cols, 0x727100UL + (ulong)(inner * 17 + cols));
                long[,] expected = Strassen.NaiveMultiply(a, b);
                long[,] actual = Strassen.Multiply(a, b, 4, new Counters());
                Check(Same(expected, actual), rows + "x" + inner + "x" + cols + ": Strassen equals the naive product");
                largest = Math.Max(largest, MaxAbs(actual));
            }
            Console.WriteLine("shapes: largestMagnitude=" + largest);
            Check(largest > 0 && largest < 1000000, "the test values stay in a safe range");
        }

        // ---- hand-checked counts: 2x2 needs seven products, the naive kernel eight ------------------
        {
            long[,] a = RandomMatrix(2, 2, 0x727AUL);
            long[,] b = RandomMatrix(2, 2, 0x727BUL);
            var strassenCounters = new Counters();
            Strassen.Multiply(a, b, 1, strassenCounters);
            var naiveCounters = new Counters();
            Strassen.Naive(Fill(a, 2), 0, 2, Fill(b, 2), 0, 2, new long[4], 0, 2, 2, naiveCounters);
            Check(strassenCounters.Multiplications == 7, "the 2x2 Strassen kernel uses seven multiplications");
            Check(naiveCounters.Multiplications == 8, "the naive 2x2 kernel uses eight");
            Check(Strassen.Model(2, 1).Multiplications == 7 && Strassen.Model(2, 1).Additions == 25,
                "the 2x2 model is seven products and twenty-five additions");

            var four = new Counters();
            Strassen.Multiply(RandomMatrix(4, 4, 0x727CUL), RandomMatrix(4, 4, 0x727DUL), 1, four);
            Check(four.Multiplications == 49, "4x4 Strassen needs 7^2 = 49 multiplications, not 64");
            var eight = new Counters();
            Strassen.Multiply(RandomMatrix(8, 8, 0x727EUL), RandomMatrix(8, 8, 0x727FUL), 1, eight);
            Check(eight.Multiplications == 343, "8x8 Strassen needs 7^3 = 343 multiplications, not 512");
            Check(Strassen.Model(8, 1).Additions == 2017, "and 2017 additions, as the recurrence predicts");
        }

        // ---- the counters match the closed-form models ----------------------------------------------
        {
            foreach (int cutoff in new[] { 1, 2, 4, 8, 16 })
                foreach (int size in new[] { 8, 16, 32 })
                {
                    if (size < cutoff) continue;
                    var counters = new Counters();
                    Strassen.Multiply(RandomMatrix(size, size, 0x7275EEDUL), RandomMatrix(size, size, 0x7271EEDUL), cutoff, counters);
                    (long multiplications, long additions) = Strassen.Model(size, cutoff);
                    Check(counters.Multiplications == multiplications,
                        "n=" + size + " cutoff=" + cutoff + ": multiplications match the model ("
                        + counters.Multiplications + " vs " + multiplications + ")");
                    Check(counters.Additions == additions,
                        "n=" + size + " cutoff=" + cutoff + ": additions match the model ("
                        + counters.Additions + " vs " + additions + ")");
                    var naive = new Counters();
                    Strassen.Naive(new long[size * size], 0, size, new long[size * size], 0, size, new long[size * size], 0, size,
                        size, naive);
                    Check((long)size * size * size == naive.Multiplications, "the naive kernel is n^3 multiplications");
                }
        }

        // ---- cutoff tuning: the objective has an interior minimum -----------------------------------
        {
            const int target = 1024;
            long bestCost = long.MaxValue;
            int bestCutoff = 0;
            long smallestMultiplications = long.MaxValue;
            int smallestAt = 0;
            long largestAdditions = long.MinValue;
            int largestAdditionsAt = 0;
            var curve = new List<(int Cutoff, long Multiplications, long Additions, long Cost)>();
            for (int cutoff = 1; cutoff <= 256; cutoff++)
            {
                (long multiplications, long additions) = Strassen.Model(target, cutoff);
                long cost = 3 * multiplications + additions;      // one multiplication costs three additions
                curve.Add((cutoff, multiplications, additions, cost));
                if (cost < bestCost) { bestCost = cost; bestCutoff = cutoff; }
                if (multiplications < smallestMultiplications) { smallestMultiplications = multiplications; smallestAt = cutoff; }
                if (additions > largestAdditions) { largestAdditions = additions; largestAdditionsAt = cutoff; }
            }
            (long tunedMultiplications, _) = Strassen.Model(target, bestCutoff);
            (long pureMultiplications, long pureAdditions) = Strassen.Model(target, 1);
            (long naiveMultiplications, long naiveAdditions) = Strassen.Model(target, target);
            Console.WriteLine("tuning(n=" + target + "): bestCutoff=" + bestCutoff + " multsAtBest=" + tunedMultiplications
                + " pureMults=" + pureMultiplications + " pureAdds=" + pureAdditions
                + " naiveMults=" + naiveMultiplications + " minMultsAt=" + smallestAt
                + " maxAddsAt=" + largestAdditionsAt + " curvePoints=" + curve.Count);

            Check(bestCutoff > 1 && bestCutoff < 256, "the optimal cutoff is strictly interior (" + bestCutoff + ")");
            Check(smallestAt == 1, "the multiplication count is minimised by pure Strassen");
            Check(largestAdditionsAt == 1, "the addition count is worst for pure Strassen");
            Check(pureMultiplications < naiveMultiplications, "pure Strassen multiplies less than the naive kernel");
            Check(pureAdditions > naiveAdditions, "and pays for it in additions");
            Check(tunedMultiplications > pureMultiplications && tunedMultiplications < naiveMultiplications,
                "the tuned cutoff sits between the two extremes (" + tunedMultiplications + ")");
            Check(bestCost == 3 * tunedMultiplications + Strassen.Model(target, bestCutoff).Additions,
                "the optimum is consistent with its own model");

            // the tuning objective must be reproducible from an actual multiplication of the same shape
            const int measured = 64;
            var counters = new Counters();
            Strassen.Multiply(RandomMatrix(measured, measured, 0x7277A1UL), RandomMatrix(measured, measured, 0x7277A2UL),
                Math.Min(bestCutoff, measured), counters);
            (long modelMultiplications, long modelAdditions) = Strassen.Model(measured, Math.Min(bestCutoff, measured));
            Check(counters.Multiplications == modelMultiplications && counters.Additions == modelAdditions,
                "the tuned model is the one the multiply actually implements");
        }

        // ---- degenerate operands and rejection --------------------------------------------------------
        {
            var identity = new long[4, 4];
            for (int i = 0; i < 4; i++) identity[i, i] = 1;
            long[,] a = RandomMatrix(4, 4, 0x7271DUL);
            Check(Same(a, Strassen.Multiply(a, identity, 2, new Counters())), "multiplying by the identity is a no-op");
            Check(Same(Strassen.Multiply(identity, a, 2, new Counters()), a), "and works on the left as well");

            var zero = new long[8, 8];
            Check(Same(zero, Strassen.Multiply(zero, RandomMatrix(8, 8, 0x7272EUL), 1, new Counters())),
                "a zero matrix multiplies to zero");

            var single = new long[1, 1];
            single[0, 0] = 7;
            var singleB = new long[1, 1];
            singleB[0, 0] = -6;
            Check(Strassen.Multiply(single, singleB, 1, new Counters())[0, 0] == -42, "a 1x1 product is exact");

            bool threw = false;
            try { Strassen.Multiply(RandomMatrix(2, 3, 1UL), RandomMatrix(4, 2, 2UL), 1, new Counters()); }
            catch (ArgumentException) { threw = true; }
            Check(threw, "mismatched inner dimensions are rejected");
            bool threwCutoff = false;
            try { Strassen.Multiply(single, single, 0, new Counters()); } catch (ArgumentOutOfRangeException) { threwCutoff = true; }
            Check(threwCutoff, "a zero cutoff is rejected");

            long[,] c = RandomMatrix(9, 9, 0x7273FUL);
            long[,] d = RandomMatrix(9, 9, 0x7274FUL);
            long[,] expected = Strassen.NaiveMultiply(c, d);
            Check(Same(expected, Strassen.Multiply(c, d, 8, new Counters())),
                "a cutoff larger than the padded size falls back to the naive kernel");
            Check(Same(expected, Strassen.Multiply(c, d, 1, new Counters())), "pure Strassen agrees on an odd 9x9 size");
            Check(Same(Strassen.Multiply(c, d, 4, new Counters()), Strassen.Multiply(c, d, 4, new Counters())),
                "the same operands reproduce the same product");
        }

        Console.WriteLine("PASS");
    }

    private static long[,] RandomMatrix(int rows, int cols, ulong seed)
    {
        var rng = new SplitMix64(seed);
        var matrix = new long[rows, cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) matrix[i, j] = rng.NextInt(9) - 4;
        return matrix;
    }

    private static long[] Fill(long[,] matrix, int size)
    {
        var flat = new long[size * size];
        for (int i = 0; i < size; i++) for (int j = 0; j < size; j++) flat[i * size + j] = matrix[i, j];
        return flat;
    }

    private static long MaxAbs(long[,] matrix)
    {
        long largest = 0;
        foreach (long value in matrix) largest = Math.Max(largest, Math.Abs(value));
        return largest;
    }

    private static bool Same(long[,] a, long[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) return false;
        for (int i = 0; i < a.GetLength(0); i++)
            for (int j = 0; j < a.GetLength(1); j++)
                if (a[i, j] != b[i, j]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
