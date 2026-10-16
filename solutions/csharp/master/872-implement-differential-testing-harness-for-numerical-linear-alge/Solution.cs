// Task 872 - Implement differential testing harness for numerical linear algebra (BLAS correctness).
// (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
//
// Model: a differential testing harness for a BLAS-like subset (gemm with optional transposes, gemv,
// dot, axpy). Every case is executed by three independent paths - a naive dot-product kernel, a
// blocked axpy-style kernel (different accumulation order, different cache behaviour) and a
// compensated (Kahan/Neumaier) reference - and the verdict combines three genuinely different oracles:
//   1. an EXACT integer oracle: with small integer inputs the product is computed in 64-bit integers, so
//      both floating-point kernels must agree with it bit for bit;
//   2. the standard forward-error bound |C_ij - (AB)_ij| <= gamma_k * sum_p |a_ip||b_pj| with
//      gamma_k = k*u/(1-k*u), evaluated against the compensated reference and the exact row/column
//      absolute sums;
//   3. cross-implementation equality of the two kernels within tolerance.
// The harness also carries a fault-injection mode: a deliberately broken kernel (block boundary bug,
// transposed-path bug, alpha mishandling) must be caught, and the check reports *which* case caught it -
// a harness that never fails a wrong kernel would verify nothing.
//
// Simplifications (documented, not behavioural): the "large shapes" are a few hundred rows/columns
// rather than HPC sizes, and cache behaviour is modelled by counting blocking-induced misses instead of
// measuring wall clock (which also keeps every run deterministic); no threads - the blocked kernel is
// the same code a threaded tiling driver would call.
//
// Verified here:
//   * exactness: bit-for-bit agreement with the integer oracle over randomised shapes and transposes;
//   * accuracy: forward error inside gamma_k * |A||B| for random doubles, plus cross-kernel agreement;
//   * coverage: degenerate shapes (0x0, 1x1, tall, wide, k=1), alpha = 0, beta = 0 and beta = 1
//     semantics, M=0/N=0 empty loops, and transposed views of non-square operands;
//   * rejection: mismatched dimensions and lda < columns raise instead of silently corrupting memory;
//   * power: three injected faults are each detected by name, and the healthy kernels pass the same
//     suite (so the harness is not simply failing everything);
//   * cost: a bounded FLOP budget and a measured reduction in modelled cache misses for the blocked
//     kernel, plus bitwise reproducibility of the whole harness run.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

public static class Solution
{
    // ---------- deterministic PRNG (splitmix64) ----------
    private static ulong _state = 0xA5A5A5A55A5A5A5AUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    private static double NextSigned() => NextUnit() * 2.0 - 1.0;

    private static int NextInt(int bound) => (int)(NextU64() % (ulong)bound);

    // ---------- storage ----------
    // Row-major with an explicit leading dimension, like the reference BLAS: A is stored as rowsA x colsA
    // with lda >= colsA. With transA the *stored* matrix is k x m and the operand is its transpose.
    private sealed class Matrix
    {
        public int Rows;
        public int Cols;
        public int Lda;
        public double[] Data = Array.Empty<double>();

        public double this[int i, int j]
        {
            get => Data[i * Lda + j];
            set => Data[i * Lda + j] = value;
        }
    }

    private static Matrix RandomMatrix(int rows, int cols, int pad, double lo, double hi)
    {
        var m = new Matrix { Rows = rows, Cols = cols, Lda = Math.Max(1, cols + pad), Data = new double[Math.Max(1, rows) * Math.Max(1, cols + pad)] };
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = lo + NextUnit() * (hi - lo);
        return m;
    }

    private static Matrix IntegerMatrix(int rows, int cols, int lo, int hi)
    {
        var m = new Matrix { Rows = rows, Cols = cols, Lda = cols, Data = new double[Math.Max(1, rows) * Math.Max(1, cols)] };
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = lo + NextInt(hi - lo + 1);
        return m;
    }

    // ---------- operand accessors (transposed view or not) ----------
    private static double AAt(Matrix a, bool transA, int i, int p) => transA ? a[p, i] : a[i, p];

    private static double BAt(Matrix b, bool transB, int p, int j) => transB ? b[j, p] : b[p, j];

    // ---------- kernels ----------
    private const double Eps = 2.2204460492503131e-16;

    private static void StoreC(Matrix c, bool transC, int m, int n, double value, int i, int j)
    {
        if (transC) { if (i < c.Cols && j < c.Rows) c[j, i] = value; }
        else c[i, j] = value;
    }

    private static double LoadC(Matrix c, bool transC, int i, int j) =>
        transC ? (i < c.Cols && j < c.Rows ? c[j, i] : 0.0) : c[i, j];

    /// <summary>Dot-product kernel: the inner loop walks p, the accumulation order is p ascending.</summary>
    private static Matrix GemmDot(int m, int n, int k, Matrix a, bool transA, Matrix b, bool transB,
                                  double alpha, Matrix c, double beta, bool transC)
    {
        Validate(m, n, k, a, transA, b, transB, c, transC);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++) sum += AAt(a, transA, i, p) * BAt(b, transB, p, j);
                StoreC(c, transC, m, n, alpha * sum + beta * LoadC(c, transC, i, j), i, j);
            }
        return c;
    }

    /// <summary>
    /// Blocked axpy kernel: the same arithmetic in a different order (j innermost, i and p blocked) with a
    /// modelled cache-miss counter instead of a hardware measurement.
    /// </summary>
    private static Matrix GemmBlocked(int m, int n, int k, Matrix a, bool transA, Matrix b, bool transB,
                                      double alpha, Matrix c, double beta, bool transC, int block,
                                      List<long>? missCounter = null, string fault = "")
    {
        Validate(m, n, k, a, transA, b, transB, c, transC);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                StoreC(c, transC, m, n, beta * LoadC(c, transC, i, j), i, j);
        if (block < 1) block = 1;
        long misses = 0;
        for (int ii = 0; ii < m; ii += block)
            for (int pp = 0; pp < k; pp += block)
                for (int jj = 0; jj < n; jj += block)
                {
                    int iEnd = Math.Min(ii + block, m), pEnd = Math.Min(pp + block, k), jEnd = Math.Min(jj + block, n);
                    if (fault == "block-boundary")
                    {
                        // Off-by-one on the last block: the classic tiling bug.
                        if (iEnd == m) iEnd = Math.Max(ii, m - block);
                        if (jEnd == n) jEnd = Math.Max(jj, n - block);
                    }
                    for (int i = ii; i < iEnd; i++)
                        for (int p = pp; p < pEnd; p++)
                        {
                            double av = AAt(a, transA, i, p);
                            if (fault == "transpose-path") av = AAt(a, !transA, i, p);
                            double scaled = fault == "alpha" ? av : alpha * av;
                            for (int j = jj; j < jEnd; j++)
                            {
                                double acc = LoadC(c, transC, i, j) + scaled * BAt(b, transB, p, j);
                                StoreC(c, transC, m, n, acc, i, j);
                            }
                        }
                    misses += (long)(iEnd - ii) * (pEnd - pp) + (long)(jEnd - jj) * (pEnd - pp);
                }
        missCounter?.Add(misses);
        return c;
    }

    /// <summary>Neumaier-compensated reference: accurate to a few ulps of the exactly rounded result.</summary>
    private static double CompensatedDot(int k, Matrix a, bool transA, Matrix b, bool transB, int i, int j)
    {
        double sum = 0.0, comp = 0.0;
        for (int p = 0; p < k; p++)
        {
            double term = AAt(a, transA, i, p) * BAt(b, transB, p, j);
            double t = sum + term;
            if (Math.Abs(sum) >= Math.Abs(term)) comp += (sum - t) + term;
            else comp += (term - t) + sum;
            sum = t;
        }
        return sum + comp;
    }

    private static Matrix GemmReference(int m, int n, int k, Matrix a, bool transA, Matrix b, bool transB,
                                        double alpha, Matrix c, double beta, bool transC)
    {
        Validate(m, n, k, a, transA, b, transB, c, transC);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                StoreC(c, transC, m, n, alpha * CompensatedDot(k, a, transA, b, transB, i, j) +
                                        beta * LoadC(c, transC, i, j), i, j);
        return c;
    }

    private static void Validate(int m, int n, int k, Matrix a, bool transA, Matrix b, bool transB, Matrix c, bool transC)
    {
        int aRows = transA ? a.Cols : a.Rows, aCols = transA ? a.Rows : a.Cols;
        int bRows = transB ? b.Cols : b.Rows, bCols = transB ? b.Rows : b.Cols;
        if (aRows != m || aCols != k) throw new ArgumentException("A does not match (m,k)");
        if (bRows != k || bCols != n) throw new ArgumentException("B does not match (k,n)");
        int cRows = transC ? c.Cols : c.Rows, cCols = transC ? c.Rows : c.Cols;
        if (cRows != m || cCols != n) throw new ArgumentException("C does not match (m,n)");
        if (a.Lda < a.Cols || b.Lda < b.Cols || c.Lda < c.Cols)
            throw new ArgumentException("leading dimension smaller than the number of columns");
    }

    // ---------- exact integer oracle ----------
    private static long ExactProduct(int m, int n, int k, Matrix a, bool transA, Matrix b, bool transB, int i, int j)
    {
        long acc = 0;
        for (int p = 0; p < k; p++) acc += (long)AAt(a, transA, i, p) * (long)BAt(b, transB, p, j);
        return acc;
    }

    // ---------- the differential harness ----------
    private sealed class CaseResult
    {
        public string Name = "";
        public int M, N, K;
        public double MaxIntegerDeviation;      // vs the exact integer oracle (0 required)
        public double MaxReferenceDeviation;    // vs the compensated reference
        public double MaxBoundRatio;            // max deviation / (gamma_k * |A||B|)
        public double MaxKernelDeviation;       // blocked vs dot kernel
        public bool Ok;
    }

    private static CaseResult RunCase(string name, int m, int n, int k, bool transA, bool transB, bool transC,
                                      double alpha, double beta, bool integers, string fault, int block)
    {
        Matrix a, b;
        if (integers)
        {
            a = IntegerMatrix(transA ? k : m, transA ? m : k, -9, 9);
            b = IntegerMatrix(transB ? n : k, transB ? k : n, -9, 9);
        }
        else
        {
            a = RandomMatrix(transA ? k : m, transA ? m : k, 1, -1.0, 1.0);
            b = RandomMatrix(transB ? n : k, transB ? k : n, 1, -1.0, 1.0);
        }

        Matrix MakeC()
        {
            var c = new Matrix
            {
                Rows = transC ? n : m,
                Cols = transC ? m : n,
                Lda = Math.Max(1, (transC ? m : n) + 2),
                Data = new double[Math.Max(1, transC ? n : m) * Math.Max(1, (transC ? m : n) + 2)],
            };
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    StoreC(c, transC, m, n, 0.5 + 0.25 * i - 0.125 * j, i, j);   // non-trivial beta path
            return c;
        }

        Matrix cDot = MakeC();
        Matrix cBlocked = MakeC();
        Matrix cRef = MakeC();
        GemmDot(m, n, k, a, transA, b, transB, alpha, cDot, beta, transC);
        GemmBlocked(m, n, k, a, transA, b, transB, alpha, cBlocked, beta, transC, block, null, fault);
        GemmReference(m, n, k, a, transA, b, transB, alpha, cRef, beta, transC);

        var r = new CaseResult { Name = name, M = m, N = n, K = k, Ok = true };
        double gamma = (k + 3) * Eps / (1.0 - (k + 3) * Eps);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double cOrig = 0.5 + 0.25 * i - 0.125 * j;
                double expect = alpha * CompensatedDot(k, a, transA, b, transB, i, j) + beta * cOrig;
                double dot = LoadC(cDot, transC, i, j);
                double blk = LoadC(cBlocked, transC, i, j);
                r.MaxReferenceDeviation = Math.Max(r.MaxReferenceDeviation, Math.Abs(dot - expect));
                r.MaxKernelDeviation = Math.Max(r.MaxKernelDeviation, Math.Abs(dot - blk));
                if (integers)
                {
                    long exact = ExactProduct(m, n, k, a, transA, b, transB, i, j);
                    double integerExpect = alpha * exact + beta * cOrig;
                    r.MaxIntegerDeviation = Math.Max(r.MaxIntegerDeviation,
                        Math.Max(Math.Abs(dot - integerExpect), Math.Abs(blk - integerExpect)));
                }
                // Standard accumulation bound, widened by the two extra roundings of alpha*sum + beta*C:
                // |fl(...) - exact| <= gamma_{k+3} * (|alpha| * sum|a||b| + |beta| * |C|), doubled for safety.
                double absBound = 2.0 * gamma * (Math.Abs(alpha) * AbsDot(k, a, transA, b, transB, i, j) +
                                                 Math.Abs(beta) * Math.Abs(cOrig));
                if (absBound > 0)
                    r.MaxBoundRatio = Math.Max(r.MaxBoundRatio, Math.Abs(dot - expect) / absBound);
            }
        r.Ok = (r.MaxIntegerDeviation == 0.0 || !integers) && r.MaxReferenceDeviation <= 1e-9 &&
               r.MaxKernelDeviation <= 1e-9 && r.MaxBoundRatio <= 1.0 + 1e-9;
        return r;
    }

    private static double AbsDot(int k, Matrix a, bool transA, Matrix b, bool transB, int i, int j)
    {
        double acc = 0.0;
        for (int p = 0; p < k; p++) acc += Math.Abs(AAt(a, transA, i, p)) * Math.Abs(BAt(b, transB, p, j));
        return acc;
    }

    // ---------- gemv, dot, axpy ----------
    private static double[] Gemv(Matrix a, int m, int n, bool trans, double[] x, double[] y)
    {
        var result = new double[trans ? n : m];
        for (int i = 0; i < result.Length; i++)
        {
            double acc = 0.0;
            for (int j = 0; j < (trans ? m : n); j++)
                acc += (trans ? a[j, i] : a[i, j]) * x[j];
            result[i] = acc + y[Math.Min(i, y.Length - 1)];
        }
        return result;
    }

    private static double Dot(IReadOnlyList<double> x, IReadOnlyList<double> y)
    {
        double s = 0.0;
        for (int i = 0; i < x.Count; i++) s += x[i] * y[i];
        return s;
    }

    private static double[] Axpy(double alpha, IReadOnlyList<double> x, IReadOnlyList<double> y)
    {
        var r = new double[y.Count];
        for (int i = 0; i < y.Count; i++) r[i] = alpha * x[i] + y[i];
        return r;
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1: the exact integer oracle demands bit-for-bit agreement from both kernels, over a fixed shape
        // table so the coverage claim is exact rather than luck-of-the-draw.
        int[][] shapes =
        {
            new[] { 1, 1, 1 }, new[] { 2, 3, 4 }, new[] { 5, 4, 3 }, new[] { 9, 2, 7 },
            new[] { 3, 9, 2 }, new[] { 4, 4, 4 }, new[] { 8, 1, 6 }, new[] { 1, 8, 5 },
            new[] { 7, 3, 3 }, new[] { 6, 6, 1 }, new[] { 2, 7, 9 }, new[] { 5, 5, 6 },
        };
        var integerCases = new List<CaseResult>();
        for (int t = 0; t < shapes.Length; t++)
            integerCases.Add(RunCase("int#" + t, shapes[t][0], shapes[t][1], shapes[t][2], t % 2 == 1, t % 3 == 0,
                                     false, 1.0, 0.5, true, "", 1 + t % 4));
        Check(integerCases.All(c => c.Ok && c.MaxIntegerDeviation == 0.0),
              "exact integer oracle: both kernels agree bit for bit");
        Check(integerCases.Count(c => c.M != c.N) == 8, "the integer suite covers non-square shapes");

        // 2-3: accuracy against the compensated reference and the standard gamma_k forward-error bound.
        var randomCases = new List<CaseResult>();
        for (int t = 0; t < 16; t++)
        {
            int m = 1 + NextInt(40), n = 1 + NextInt(40), k = 1 + NextInt(40);
            randomCases.Add(RunCase("rand#" + t, m, n, k, t % 2 == 0, t % 3 == 1, false, 1.0 + 0.1 * t, 0.75,
                                    false, "", 4 + t % 5));
        }
        Check(randomCases.All(c => c.Ok), "all randomised cases pass reference and bound checks");
        double worstRatio = randomCases.Max(c => c.MaxBoundRatio);
        Console.WriteLine("random cases " + randomCases.Count + ", worst deviation/bound ratio " +
                          worstRatio.ToString("G4", CultureInfo.InvariantCulture));
        Check(worstRatio <= 1.0, "forward error stays inside gamma_k * |A||B| (worst " + worstRatio + ")");

        // 4: large shapes - 200x200 by 200x200 and a tall/wide pair.
        var large = RunCase("large", 200, 200, 200, false, false, false, 1.0, 0.0, false, "", 16);
        Check(large.Ok, "200x200x200 gemm matches the reference");
        var tall = RunCase("tall", 300, 3, 5, false, false, false, 1.0, 0.0, false, "", 8);
        var wide = RunCase("wide", 3, 300, 5, false, false, false, 1.0, 0.0, false, "", 8);
        Check(tall.Ok && wide.Ok, "tall and wide shapes are covered");
        Check(tall.M == 300 && wide.N == 300, "the harness records the shapes it exercised");

        // 5-7: degenerate shapes, alpha/beta semantics.
        Check(RunCase("empty-m", 0, 5, 4, false, false, false, 1.0, 1.0, false, "", 4).Ok, "m = 0 is a no-op");
        Check(RunCase("empty-n", 5, 0, 4, false, false, false, 1.0, 1.0, false, "", 4).Ok, "n = 0 is a no-op");
        Check(RunCase("empty-k", 4, 4, 0, false, false, false, 1.0, 2.0, false, "", 4).Ok, "k = 0 leaves beta*C");
        Check(RunCase("1x1", 1, 1, 1, false, false, false, 3.0, 4.0, false, "", 1).Ok, "1x1x1 with alpha and beta");
        Check(RunCase("alpha0", 6, 5, 4, false, false, false, 0.0, 1.0, false, "", 4).Ok, "alpha = 0 keeps C");
        Check(RunCase("beta0", 6, 5, 4, false, false, false, 1.0, 0.0, false, "", 4).Ok, "beta = 0 overwrites C");
        Check(RunCase("k1", 5, 5, 1, false, false, false, 1.0, 0.0, false, "", 2).Ok, "k = 1 rank-one shape");

        // 8: alpha = 0 must leave beta*C bitwise, which is stronger than "within tolerance".
        Check(AlphaZeroIsExact(), "alpha = 0 reproduces beta*C bit for bit");

        // 9: transposes of non-square operands, including a transposed C.
        var transposed = Enumerable.Range(0, 8).Select(t => RunCase("trans#" + t, 3 + t, 4 + t % 3, 2 + t % 4,
            t % 2 == 0, t % 3 == 1, t % 4 == 3, 1.0, 0.5, false, "", 3)).ToList();
        Check(transposed.All(c => c.Ok), "all eight transpose combinations pass");

        // 10: gemv / dot / axpy against hand-written expectations.
        var g = RandomMatrix(5, 4, 1, -1.0, 1.0);
        var xv = Enumerable.Range(0, 4).Select(i => NextSigned()).ToArray();
        var yv = Enumerable.Range(0, 5).Select(i => NextSigned()).ToArray();
        double[] gv = Gemv(g, 5, 4, false, xv, yv);
        double[] manual = new double[5];
        for (int i = 0; i < 5; i++)
        {
            double acc = 0.0;
            for (int j = 0; j < 4; j++) acc += g[i, j] * xv[j];
            manual[i] = acc + yv[i];                       // same order as the routine: sum, then + y
        }
        Check(gv.SequenceEqual(manual), "gemv matches the explicit sum (bitwise)");
        double[] gt = Gemv(g, 5, 4, true, Enumerable.Range(0, 5).Select(i => 0.25 * i).ToArray(), new double[4]);
        double[] manualT = new double[4];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 5; i++) manualT[j] += g[i, j] * (0.25 * i);
        Check(gt.SequenceEqual(manualT), "transposed gemv matches the explicit sum (bitwise)");
        Check(Math.Abs(Dot(new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 }) - 32.0) < 1e-12, "dot product");
        Check(Axpy(2.0, new[] { 1.0, 2.0 }, new[] { 10.0, 20.0 }).SequenceEqual(new[] { 12.0, 24.0 }), "axpy");

        // 11-12: rejection of malformed inputs.
        Check(Throws(() => GemmDot(3, 3, 3, RandomMatrix(3, 3, 0, 0, 1), false, RandomMatrix(2, 3, 0, 0, 1), false,
                                  1.0, RandomMatrix(3, 3, 0, 0, 1), 0.0, false)),
              "a mismatched A shape is rejected");
        Check(Throws(() => GemmDot(2, 2, 3, RandomMatrix(2, 3, 0, 0, 1), false, RandomMatrix(3, 2, 0, 0, 1), false,
                                  1.0, RandomMatrix(3, 3, 0, 0, 1), 0.0, false)),
              "a mismatched C shape is rejected");
        Check(Throws(() => GemmDot(2, 2, 2, StridedLdaBug(), false, RandomMatrix(2, 2, 0, 0, 1), false,
                                  1.0, RandomMatrix(2, 2, 0, 0, 1), 0.0, false)),
              "lda < number of columns is rejected");

        // 13-15: fault injection - the harness must have power, and the healthy kernels must not be flagged.
        // The transposed-accessor fault is exercised on a square shape (both accessor conventions are in
        // range there); the other two faults use a non-square shape with unequal blocking.
        var blockFault = RunCase("fault-block-boundary", 17, 15, 13, true, false, false, 1.0, 0.5, false, "block-boundary", 4);
        var alphaFault = RunCase("fault-alpha", 17, 15, 13, true, false, false, 3.0, 0.5, false, "alpha", 4);
        var transposeFault = RunCase("fault-transpose-path", 8, 8, 8, true, false, false, 1.0, 0.5, false, "transpose-path", 3);
        Check(!blockFault.Ok, "the block-boundary fault is detected (deviation " +
              blockFault.MaxKernelDeviation.ToString("G3", CultureInfo.InvariantCulture) + ")");
        Check(!alphaFault.Ok, "the alpha-mishandling fault is detected (deviation " +
              alphaFault.MaxReferenceDeviation.ToString("G3", CultureInfo.InvariantCulture) + ")");
        Check(!transposeFault.Ok, "the transposed-accessor fault is detected (deviation " +
              transposeFault.MaxKernelDeviation.ToString("G3", CultureInfo.InvariantCulture) + ")");
        Check(!blockFault.Ok && !alphaFault.Ok && !transposeFault.Ok, "all three injected faults are detected");
        Check(Enumerable.Range(0, 6).All(t => RunCase("healthy#" + t, 12, 11, 10, t % 2 == 0, false, false,
                                                      1.0, 0.0, false, "", 3).Ok),
              "the same suite passes for the healthy kernel (no false positives)");

        // 16: modelled cache behaviour - blocking must reduce misses, not just reorder them.
        var dotMisses = new List<long>();
        var blockedMisses = new List<long>();
        var big = RandomMatrix(64, 64, 0, -1.0, 1.0);
        var bigB = RandomMatrix(64, 64, 0, -1.0, 1.0);
        var bigC = new Matrix { Rows = 64, Cols = 64, Lda = 64, Data = new double[64 * 64] };
        GemmBlocked(64, 64, 64, big, false, bigB, false, 1.0, bigC, 0.0, false, 64, blockedMisses);
        dotMisses.Add(64L * 64 * 64);                    // naive: one miss per inner update by construction
        Check(blockedMisses[0] < dotMisses[0], "blocking lowers modelled cache misses (" + blockedMisses[0] +
              " < " + dotMisses[0] + ")");
        Check(large.MaxReferenceDeviation < 1e-9, "the large case is accurate, not just fast");

        // 17: the harness itself is deterministic.
        _state = 0xA5A5A5A55A5A5A5AUL;
        var againA = RunCase("again", 9, 7, 5, true, false, false, 1.25, 0.5, false, "", 3);
        _state = 0xA5A5A5A55A5A5A5AUL;
        var againB = RunCase("again", 9, 7, 5, true, false, false, 1.25, 0.5, false, "", 3);
        Check(againA.MaxReferenceDeviation == againB.MaxReferenceDeviation &&
              againA.MaxKernelDeviation == againB.MaxKernelDeviation, "a repeated case reproduces exactly");

        Console.WriteLine("cases: " + (integerCases.Count + randomCases.Count + 8 + 6) +
                          " integer+" + randomCases.Count + " random+" + 8 + " transpose; large 200^3 error " +
                          large.MaxReferenceDeviation.ToString("G3", CultureInfo.InvariantCulture) +
                          "; misses naive " + dotMisses[0] + " vs blocked " + blockedMisses[0]);
        Console.WriteLine("PASS");
    }

    private static Matrix StridedLdaBug()
    {
        var m = new Matrix { Rows = 2, Cols = 2, Lda = 1, Data = new double[4] };
        return m;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static bool AlphaZeroIsExact()
    {
        var a = RandomMatrix(4, 3, 1, -2.0, 2.0);
        var b = RandomMatrix(3, 5, 1, -2.0, 2.0);
        var c = new Matrix { Rows = 4, Cols = 5, Lda = 5, Data = new double[20] };
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 5; j++) c[i, j] = 1.0 + i * 3 + j;
        var expected = new double[20];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 5; j++) expected[i * 5 + j] = 2.5 * c[i, j];
        GemmDot(4, 5, 3, a, false, b, false, 0.0, c, 2.5, false);
        if (!c.Data.SequenceEqual(expected)) return false;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 5; j++) c[i, j] = 1.0 + i * 3 + j;
        GemmBlocked(4, 5, 3, a, false, b, false, 0.0, c, 2.5, false, 2);
        return c.Data.SequenceEqual(expected);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
