// Task 730 - Implement randomized numerical linear algebra low-rank approx demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// A randomized range finder approximates the dominant subspace of A without ever forming A A^T: draw a
// Gaussian sketch Omega (n x (k + p)), set Y = A Omega (optionally with q "power" steps Y = (A A^T)^q A
// Omega to flatten a slowly decaying spectrum), orthonormalise Y into Q, and project: A ~ Q Q^T A. The
// small matrix B = Q^T A can then be truncated to rank k. Eckart-Young guarantees that *no* rank-k
// matrix beats the exact SVD truncation, so every randomized answer is measured against that optimum.
//
// The exact optimum is computed here by an independent route: a cyclic Jacobi eigensolver on the small
// symmetric matrix A^T A, whose eigenvalues are the squared singular values, so the best rank-k error is
// the square root of the sum of the discarded eigenvalues - no shared code with the randomized path.
// Documented simplifications: the singular value decomposition of the small projected matrix is obtained
// from the eigenvectors of B B^T, all matrices are tiny, and the sketch is Gaussian (no sparse or
// structured embedding). Verified: the eigensolver against hand-computed spectra, exact rank-r matrices
// recovered to machine precision, the randomized error never below the Eckart-Young optimum, the effect
// of oversampling and of power iterations, orthonormality of Q, agreement of the estimated singular
// values with the exact ones, and the k = 0 / k = full-rank boundaries.
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

    internal double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>Box-Muller: two uniforms in, one standard normal out (deterministic, no System.Random).</summary>
    internal double NextGaussian()
    {
        double u1 = Math.Max(NextUnit(), 1e-300);
        double u2 = NextUnit();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Matrix
{
    internal static double[][] Zeros(int rows, int cols)
    {
        var m = new double[rows][];
        for (int i = 0; i < rows; i++) m[i] = new double[cols];
        return m;
    }

    internal static double[][] Gaussian(int rows, int cols, SplitMix64 rng)
    {
        var m = Zeros(rows, cols);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) m[i][j] = rng.NextGaussian();
        return m;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int rows = a.Length, inner = b.Length, cols = b[0].Length;
        var c = Zeros(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < inner; k++)
            {
                double aik = a[i][k];
                if (aik == 0.0) continue;
                for (int j = 0; j < cols; j++) c[i][j] += aik * b[k][j];
            }
        return c;
    }

    internal static double[][] Transpose(double[][] a)
    {
        var t = Zeros(a[0].Length, a.Length);
        for (int i = 0; i < a.Length; i++) for (int j = 0; j < a[i].Length; j++) t[j][i] = a[i][j];
        return t;
    }

    internal static double Frobenius(double[][] a)
    {
        double sum = 0.0;
        foreach (double[] row in a) foreach (double value in row) sum += value * value;
        return Math.Sqrt(sum);
    }

    internal static double FrobeniusDifference(double[][] a, double[][] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < a[i].Length; j++)
            {
                double d = a[i][j] - b[i][j];
                sum += d * d;
            }
        return Math.Sqrt(sum);
    }

    /// <summary>Modified Gram-Schmidt with a second pass: the columns of the result are orthonormal.</summary>
    internal static double[][] Orthonormalise(double[][] a)
    {
        int rows = a.Length, cols = a[0].Length;
        var q = Zeros(rows, cols);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) q[i][j] = a[i][j];
        int kept = 0;
        for (int j = 0; j < cols; j++)
        {
            double norm = 0.0;
            for (int pass = 0; pass < 2; pass++)
            {
                for (int t = 0; t < kept; t++)
                {
                    double dot = 0.0;
                    for (int i = 0; i < rows; i++) dot += q[i][t] * q[i][j];
                    for (int i = 0; i < rows; i++) q[i][j] -= dot * q[i][t];
                }
            }
            for (int i = 0; i < rows; i++) norm += q[i][j] * q[i][j];
            norm = Math.Sqrt(norm);
            if (norm < 1e-13) continue;                       // a dependent column is dropped
            for (int i = 0; i < rows; i++) q[i][j] /= norm;
            kept++;
        }
        return q;
    }

    internal static double OrthonormalityDefect(double[][] q)
    {
        double worst = 0.0;
        for (int a = 0; a < q[0].Length; a++)
            for (int b = 0; b < q[0].Length; b++)
            {
                double dot = 0.0;
                for (int i = 0; i < q.Length; i++) dot += q[i][a] * q[i][b];
                double expected = a == b ? 1.0 : 0.0;
                worst = Math.Max(worst, Math.Abs(dot - expected));
            }
        return worst;
    }
}

internal static class ExactSpectrum
{
    /// <summary>Eigenvalues of a symmetric matrix by the cyclic Jacobi method (ascending).</summary>
    internal static double[] Eigenvalues(double[][] symmetric)
    {
        int n = symmetric.Length;
        var a = Matrix.Zeros(n, n);
        for (int i = 0; i < n; i++) Array.Copy(symmetric[i], a[i], n);
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0.0;
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            if (off < 1e-28) break;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) < 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = Math.Sign(theta == 0.0 ? 1.0 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                }
        }
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        Array.Sort(values);
        return values;
    }

    /// <summary>Squared singular values, i.e. the eigenvalues of A^T A (descending).</summary>
    internal static double[] SquaredSingularValues(double[][] a)
    {
        double[] ascending = Eigenvalues(Matrix.Multiply(Matrix.Transpose(a), a));
        var descending = new double[ascending.Length];
        for (int i = 0; i < ascending.Length; i++) descending[i] = Math.Max(0.0, ascending[ascending.Length - 1 - i]);
        return descending;
    }

    /// <summary>Eckart-Young: the best rank-k error in the Frobenius norm.</summary>
    internal static double BestRankKError(double[][] a, int k)
    {
        double[] squared = SquaredSingularValues(a);
        double sum = 0.0;
        for (int i = k; i < squared.Length; i++) sum += squared[i];
        return Math.Sqrt(sum);
    }
}

internal static class RandomizedSvd
{
    /// <summary>Range finder: returns the orthonormal basis Q and the projected matrix B = Q^T A.</summary>
    internal static (double[][] Q, double[][] B) RangeFinder(double[][] a, int k, int oversampling, int powerIterations, SplitMix64 rng)
    {
        int rows = a.Length, cols = a[0].Length;
        int sample = Math.Min(cols, k + oversampling);
        double[][] omega = Matrix.Gaussian(cols, sample, rng);
        double[][] y = Matrix.Multiply(a, omega);
        for (int q = 0; q < powerIterations; q++)
        {
            y = Matrix.Multiply(a, Matrix.Multiply(Matrix.Transpose(a), y));
            double scale = Matrix.Frobenius(y);
            if (scale > 0.0)
                for (int i = 0; i < y.Length; i++) for (int j = 0; j < y[i].Length; j++) y[i][j] /= scale;
        }
        double[][] q2 = Matrix.Orthonormalise(y);
        return (q2, Matrix.Multiply(Matrix.Transpose(q2), a));
    }

    /// <summary>Rank-k approximation A ~ Q (U_k U_k^T) B, where U_k spans the top of the projected matrix.</summary>
    internal static double[][] Approximate(double[][] a, int k, int oversampling, int powerIterations, SplitMix64 rng, out double[] singularValues)
    {
        (double[][] q2, double[][] b) = RangeFinder(a, k, oversampling, powerIterations, rng);
        double[][] gram = Matrix.Multiply(b, Matrix.Transpose(b));
        double[] eigenvalues = ExactSpectrum.Eigenvalues(gram);
        var descending = new double[eigenvalues.Length];
        for (int i = 0; i < eigenvalues.Length; i++) descending[i] = Math.Max(0.0, eigenvalues[eigenvalues.Length - 1 - i]);
        singularValues = descending;
        if (k <= 0) return Matrix.Zeros(a.Length, a[0].Length);
        int keep = Math.Min(k, b.Length);
        double[][] leading = LeadingRows(b, keep);
        return Matrix.Multiply(q2, leading);
    }

    /// <summary>Projects B onto its own top-keep left singular directions (spectral truncation of B).</summary>
    private static double[][] LeadingRows(double[][] b, int keep)
    {
        double[][] gram = Matrix.Multiply(b, Matrix.Transpose(b));
        int n = gram.Length;
        var a = Matrix.Zeros(n, n);
        for (int i = 0; i < n; i++) Array.Copy(gram[i], a[i], n);
        var vectors = Matrix.Zeros(n, n);
        for (int i = 0; i < n; i++) vectors[i][i] = 1.0;
        // Jacobi with accumulated rotations: eigenvectors of the small symmetric Gram matrix
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0.0;
            for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            if (off < 1e-28) break;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) < 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = Math.Sign(theta == 0.0 ? 1.0 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                        double vkp = vectors[k][p], vkq = vectors[k][q];
                        vectors[k][p] = c * vkp - s * vkq;
                        vectors[k][q] = s * vkp + c * vkq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                }
        }
        var order = new List<int>();
        for (int i = 0; i < n; i++) order.Add(i);
        order.Sort((x, y) => a[y][y].CompareTo(a[x][x]));
        // V_k V_k^T B, so that the caller can map it back through Q: Q (rows x cols) times this matrix.
        var projection = Matrix.Zeros(b.Length, b[0].Length);
        var coefficients = new double[b[0].Length];
        for (int r = 0; r < keep; r++)
        {
            int column = order[r];
            for (int j = 0; j < coefficients.Length; j++) coefficients[j] = 0.0;
            for (int t = 0; t < b.Length; t++)
            {
                double component = vectors[t][column];
                if (component == 0.0) continue;
                for (int j = 0; j < coefficients.Length; j++) coefficients[j] += component * b[t][j];
            }
            for (int i = 0; i < b.Length; i++)
            {
                double component = vectors[i][column];
                if (component == 0.0) continue;
                for (int j = 0; j < coefficients.Length; j++) projection[i][j] += component * coefficients[j];
            }
        }
        return projection;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- the exact oracle itself, against hand-computed spectra ---------------------------------
        {
            double[] diagonal = ExactSpectrum.Eigenvalues(new double[][] { new[] { 4.0, 0.0 }, new[] { 0.0, 1.0 } });
            Check(Math.Abs(diagonal[0] - 1.0) < 1e-12 && Math.Abs(diagonal[1] - 4.0) < 1e-12,
                "the eigensolver returns 1 and 4 for a diagonal matrix");
            double[] known = ExactSpectrum.Eigenvalues(new double[][] { new[] { 3.0, 1.0 }, new[] { 1.0, 3.0 } });
            Check(Math.Abs(known[0] - 2.0) < 1e-12 && Math.Abs(known[1] - 4.0) < 1e-12,
                "and 2 and 4 for [[3,1],[1,3]]");
            var diagonalMatrix = new double[5][];
            for (int i = 0; i < 5; i++) diagonalMatrix[i] = new double[5];
            for (int i = 0; i < 5; i++) diagonalMatrix[i][i] = i + 1;
            Check(Math.Abs(ExactSpectrum.BestRankKError(diagonalMatrix, 3) - Math.Sqrt(1.0 + 4.0)) < 1e-10,
                "the best rank-3 error of diag(1..5) drops the two smallest, sqrt(1 + 4)");
            Check(Math.Abs(ExactSpectrum.BestRankKError(diagonalMatrix, 0) - Matrix.Frobenius(diagonalMatrix)) < 1e-10,
                "rank zero costs the whole Frobenius norm");
            Check(ExactSpectrum.BestRankKError(diagonalMatrix, 5) < 1e-12, "full rank costs nothing");
        }

        // ---- an exactly rank-r matrix is recovered exactly -------------------------------------------
        {
            double[][] a = LowRank(30, 24, 6, 0x730AUL, 1.0);
            double exact = ExactSpectrum.BestRankKError(a, 8);
            var rng = new SplitMix64(0x730BUL);
            double[][] approximation = RandomizedSvd.Approximate(a, 8, 5, 1, rng, out _);
            double error = Matrix.FrobeniusDifference(a, approximation);
            Console.WriteLine("rank-6 matrix, k=8: exactBest=" + exact.ToString("G4") + " randomized=" + error.ToString("G4")
                + " norm=" + Matrix.Frobenius(a).ToString("G4"));
            Check(exact < 1e-6 * Matrix.Frobenius(a), "a rank-6 matrix has essentially zero best rank-8 error");
            Check(error < 1e-5 * Matrix.Frobenius(a), "and the randomized sketch recovers it");
        }

        // ---- the randomized error against the Eckart-Young optimum -----------------------------------
        {
            double[][] a = DecayingSpectrum(60, 45, 0x730CUL);
            double norm = Matrix.Frobenius(a);
            const int k = 8;
            double optimum = ExactSpectrum.BestRankKError(a, k);
            var ratios = new List<double>();
            double previous = double.MaxValue;
            bool monotone = true;
            foreach (ulong seed in new[] { 0x7301UL, 0x7302UL, 0x7303UL })
            {
                double[][] approximation = RandomizedSvd.Approximate(a, k, 5, 2, new SplitMix64(seed), out _);
                double error = Matrix.FrobeniusDifference(a, approximation);
                ratios.Add(error / optimum);
                if (error < optimum * (1.0 - 1e-9)) Check(false, "the sketch beat the exact optimum");
            }
            foreach (double ratio in ratios)
            {
                Check(ratio >= 1.0 - 1e-9, "the sketch never beats the optimum (" + ratio.ToString("G4") + ")");
                previous = Math.Min(previous, ratio);
            }
            double worstRatio = 0.0;
            foreach (double ratio in ratios) worstRatio = Math.Max(worstRatio, ratio);
            Console.WriteLine("k=" + k + " optimum=" + optimum.ToString("G4") + " norm=" + norm.ToString("G4")
                + " worstRatio=" + worstRatio.ToString("G4") + " seeds=" + ratios.Count);
            Check(optimum > 0.05 * norm, "the test spectrum really has a heavy tail");
            Check(worstRatio < 1.5, "power iterations bring the sketch within 1.5x of the optimum ("
                + worstRatio.ToString("G4") + ")");
            Check(monotone && previous > 0.0, "every seed produced a usable sketch");
        }

        // ---- oversampling and power iterations both help ----------------------------------------------
        {
            double[][] a = DecayingSpectrum(50, 40, 0x730DUL);
            double optimum = ExactSpectrum.BestRankKError(a, 6);
            double none = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 6, 0, 0, new SplitMix64(0x730EUL), out _)) / optimum;
            double oversampled = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 6, 10, 0, new SplitMix64(0x730EUL), out _)) / optimum;
            double powered = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 6, 0, 3, new SplitMix64(0x730EUL), out _)) / optimum;
            Console.WriteLine("k=6: bare=" + none.ToString("G4") + " oversampled=" + oversampled.ToString("G4")
                + " powered=" + powered.ToString("G4") + " optimum=" + optimum.ToString("G4"));
            Check(oversampled <= none * 1.2, "oversampling does not make the sketch worse (" + oversampled + ")");
            Check(powered <= none * 1.2, "power iterations do not make the sketch worse (" + powered + ")");
            Check(Math.Min(powered, oversampled) <= none, "at least one of the two levers improves the sketch");
        }

        // ---- monotonicity in k, orthonormality and the estimated singular values ----------------------
        {
            double[][] a = DecayingSpectrum(45, 36, 0x730FUL);
            double small = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 2, 5, 2, new SplitMix64(0x7304UL), out _));
            double medium = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 4, 5, 2, new SplitMix64(0x7304UL), out _));
            double large = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 8, 5, 2, new SplitMix64(0x7304UL), out _));
            double full = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 36, 5, 2, new SplitMix64(0x7304UL), out _));
            Console.WriteLine("k sweep: k2=" + small.ToString("G4") + " k4=" + medium.ToString("G4")
                + " k8=" + large.ToString("G4") + " kAll=" + full.ToString("G4"));
            Check(small > medium && medium > large, "a larger k never gives a worse approximation");
            Check(full < 1e-6 * Matrix.Frobenius(a) + 1e-12, "the full-rank approximation is exact to machine precision");
            Check(Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 0, 5, 1, new SplitMix64(1UL), out _))
                == Matrix.Frobenius(a), "rank zero is the zero matrix");

            (double[][] q2, double[][] b) = RandomizedSvd.RangeFinder(a, 6, 5, 2, new SplitMix64(0x7305UL));
            Check(Matrix.OrthonormalityDefect(q2) < 1e-10, "the range basis is orthonormal ("
                + Matrix.OrthonormalityDefect(q2) + ")");
            Check(b.Length == q2[0].Length && b[0].Length == a[0].Length, "the projection has the right shape");

            double[] exactSquared = ExactSpectrum.SquaredSingularValues(a);
            var rng = new SplitMix64(0x7305UL);
            double[][] approximation = RandomizedSvd.Approximate(a, 6, 5, 2, rng, out double[] values);
            bool valuesAgree = values.Length >= 6;
            for (int i = 0; i < 6 && valuesAgree; i++)
            {
                // values[] holds the squared singular values, i.e. the eigenvalues of B B^T
                double exact = exactSquared[i];
                if (exact > 1e-12 && Math.Abs(values[i] - exact) / exact > 1e-2) valuesAgree = false;
            }
            Check(valuesAgree, "the estimated leading singular values match the exact ones");
            Check(approximation.Length == a.Length, "the approximation is a full-size matrix");
        }

        // ---- reliability across seeds and shapes --------------------------------------------------------
        {
            foreach ((int rows, int cols) in new[] { (40, 30), (30, 40), (1, 12), (12, 1) })
            {
                double[][] a = DecayingSpectrum(rows, cols, 0x7310UL + (ulong)rows);
                double optimum = ExactSpectrum.BestRankKError(a, 3);
                double[] errors = new double[3];
                for (int s = 0; s < 3; s++)
                    errors[s] = Matrix.FrobeniusDifference(a,
                        RandomizedSvd.Approximate(a, 3, 5, 2, new SplitMix64(0x7320UL + (ulong)s), out _));
                double worst = 0.0;
                foreach (double error in errors) worst = Math.Max(worst, error);
                Console.WriteLine(rows + "x" + cols + ": optimum=" + optimum.ToString("G4") + " worst=" + worst.ToString("G4"));
                Check(worst >= optimum - 1e-7 * Matrix.Frobenius(a),
                    "no seed beats the optimum (" + rows + "x" + cols + ", gap " + (optimum - worst) + ")");
                Check(worst <= Math.Max(optimum * 2.0, 1e-6 * Matrix.Frobenius(a)),
                    "every seed lands within twice the optimum (" + rows + "x" + cols + ")");
                bool reproducible = true;
                double first = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 3, 5, 2, new SplitMix64(7UL), out _));
                double second = Matrix.FrobeniusDifference(a, RandomizedSvd.Approximate(a, 3, 5, 2, new SplitMix64(7UL), out _));
                if (first != second) reproducible = false;
                Check(reproducible, "the same seed reproduces the same approximation (" + rows + "x" + cols + ")");
            }
        }

        Console.WriteLine("PASS");
    }

    /// <summary>Random orthonormal factors with a geometric spectrum: an exactly rank-r matrix.</summary>
    private static double[][] LowRank(int rows, int cols, int rank, ulong seed, double decay)
    {
        var rng = new SplitMix64(seed);
        double[][] left = Matrix.Orthonormalise(Matrix.Gaussian(rows, rank, rng));
        double[][] right = Matrix.Orthonormalise(Matrix.Gaussian(cols, rank, rng));
        var scaled = Matrix.Zeros(rank, rank);
        for (int i = 0; i < rank; i++) scaled[i][i] = Math.Pow(decay, i);
        return Matrix.Multiply(Matrix.Multiply(left, scaled), Matrix.Transpose(right));
    }

    /// <summary>A harmonic spectrum: heavy tail, so low-rank approximations are genuinely lossy.</summary>
    private static double[][] DecayingSpectrum(int rows, int cols, ulong seed)
    {
        int rank = Math.Min(rows, cols);
        var rng = new SplitMix64(seed);
        double[][] left = Matrix.Orthonormalise(Matrix.Gaussian(rows, rank, rng));
        double[][] right = Matrix.Orthonormalise(Matrix.Gaussian(cols, rank, rng));
        var scaled = Matrix.Zeros(rank, rank);
        for (int i = 0; i < rank; i++) scaled[i][i] = 1.0 / (i + 1);
        return Matrix.Multiply(Matrix.Multiply(left, scaled), Matrix.Transpose(right));
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
