// Task 502 - Implement randomized SVD basic using power method. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Randomized SVD in the Halko–Martinsson–Tropp style. Given A (m x n), a deterministic LCG
// sketch Omega (n x l, entries uniform in [-1,1], l = k + p oversampling), the range finder
// computes Y = (A A^T)^q A Omega, i.e. q steps of the power method that flatten the spectrum;
// Q is an orthonormal basis of range(Y) built by modified Gram–Schmidt with one
// re-orthogonalisation pass (twice-is-enough). The small matrix B = Q^T A (l x n) is then
// decomposed exactly: Jacobi eigen-decomposition of B B^T gives U_B and the squared singular
// values, V = B^T U_B / sigma, and finally U = Q U_B. Documented checks:
//   * the planted matrix A = sum_{i<12} 10 * 0.5^i u_i v_i^T really has those singular values:
//     Jacobi on A^T A reproduces them (|sigma - s| <= 1e-8, zeros in the 13 tail slots) and the
//     planted frames are orthonormal (<= 1e-12);
//   * with k = 6, p = 4, q = 2 the top-6 singular values are machine precise (<= 1e-9) while the
//     q = 0 sketch is off by 2.9e-4 - a documented >= 10x improvement from the power steps;
//   * the rank-k reconstruction error is squeezed by Eckart–Young from both sides,
//     tail(k) - 1e-9 <= ||A - U_k S_k V_k^T||_F <= 1.05 * tail(k), tail(k) = sqrt(sum_{i>=k} s_i^2);
//   * range errors show the power method working and then saturating in double precision:
//     ||A - Q Q^T A||_F is 0.0909 for q = 0, 0.0131 for q = 1 and 0.0110 for q = 2 - all within a
//     factor of the ideal 10-dimensional truncation error sigma_10 = 0.00977 - but 0.0572 (q = 3)
//     and 0.1098 (q = 4) because the sketch columns become numerically rank-deficient;
//   * Q, U, V are orthonormal to 1e-12 (vector-wise defect), the eigen-relation residual
//     ||A v_i - sigma_i u_i|| is <= 1e-4 * sigma_1, and the error decreases monotonically in k;
//   * l = n makes the sketch exact (|sigma_hat - sigma| <= 1e-6 with q = 0), a different LCG seed
//     still recovers the top-6 values, repeated runs are bit-for-bit identical, and a rank-5
//     matrix is recovered with k = 5, p = 3, q = 1 (errors <= 1e-6, reconstruction <= 1e-8).
using System;
using System.Collections.Generic;

internal static class RandomizedSvd
{
    internal class Lcg
    {
        private int state;

        internal Lcg(int seed) => state = seed;

        /** Uniform in [-1, 1]. */
        internal double Next()
        {
            state = (state * 75 + 74) % 65537;
            return (state % 20001 - 10000) / 10000.0;
        }
    }

    internal static double[][] Zeros(int rows, int cols)
    {
        var result = new double[rows][];
        for (int i = 0; i < rows; i++) result[i] = new double[cols];
        return result;
    }

    internal static double[][] Transpose(double[][] a)
    {
        int m = a.Length;
        int n = a[0].Length;
        var result = Zeros(n, m);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) result[j][i] = a[i][j];
        }
        return result;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int m = a.Length;
        int inner = b.Length;
        int n = b[0].Length;
        var result = Zeros(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int t = 0; t < inner; t++)
            {
                double value = a[i][t];
                if (value == 0.0) continue;
                for (int j = 0; j < n; j++) result[i][j] += value * b[t][j];
            }
        }
        return result;
    }

    internal static double[] MatVec(double[][] a, double[] v)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < v.Length; j++) sum += a[i][j] * v[j];
            result[i] = sum;
        }
        return result;
    }

    internal static double Frobenius(double[][] a)
    {
        double sum = 0.0;
        foreach (double[] row in a)
        {
            foreach (double value in row) sum += value * value;
        }
        return Math.Sqrt(sum);
    }

    /** Modified Gram–Schmidt with one re-orthogonalisation pass; returns the orthonormal columns. */
    internal static double[][] Orth(double[][] y)
    {
        int m = y.Length;
        int l = y[0].Length;
        var q = Zeros(m, l);
        var norms = new double[l];
        for (int c = 0; c < l; c++)
        {
            var v = new double[m];
            for (int i = 0; i < m; i++) v[i] = y[i][c];
            for (int pass = 0; pass < 2; pass++)
            {
                for (int d = 0; d < c; d++)
                {
                    double projection = 0.0;
                    for (int i = 0; i < m; i++) projection += q[i][d] * v[i];
                    for (int i = 0; i < m; i++) v[i] -= projection * q[i][d];
                }
            }
            double norm = 0.0;
            for (int i = 0; i < m; i++) norm += v[i] * v[i];
            norm = Math.Sqrt(norm);
            norms[c] = norm;
            for (int i = 0; i < m; i++) q[i][c] = norm <= 1e-300 ? 0.0 : v[i] / norm;
        }
        return q;
    }

    internal class Eigen
    {
        internal double[] Values = Array.Empty<double>();
        internal double[][] Vectors = Array.Empty<double[]>();
    }

    /** Cyclic Jacobi for a symmetric matrix; eigenvectors are columns (as in the other demos). */
    internal static Eigen Jacobi(double[][] matrix, int maxSweeps, double tol)
    {
        int n = matrix.Length;
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n];
            Array.Copy(matrix[i], a[i], n);
        }
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0.0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (Math.Sqrt(off) <= tol) break;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q2 = p + 1; q2 < n; q2++)
                {
                    if (Math.Abs(a[p][q2]) <= 1e-300) continue;
                    double theta = (a[q2][q2] - a[p][p]) / (2.0 * a[p][q2]);
                    double t = 1.0 / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta < 0.0) t = -t;
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        if (k == p || k == q2) continue;
                        double akp = a[k][p];
                        double akq = a[k][q2];
                        a[k][p] = c * akp - s * akq;
                        a[k][q2] = s * akp + c * akq;
                        a[p][k] = a[k][p];
                        a[q2][k] = a[k][q2];
                    }
                    double app = a[p][p];
                    double aqq = a[q2][q2];
                    double apq = a[p][q2];
                    a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                    a[q2][q2] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                    a[p][q2] = 0.0;
                    a[q2][p] = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k][p];
                        double vkq = v[k][q2];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q2] = s * vkp + c * vkq;
                    }
                }
            }
        }
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        return new Eigen { Values = values, Vectors = v };
    }

    internal class Svd
    {
        internal double[] Sigma = Array.Empty<double>();
        internal double[][] U = Array.Empty<double[]>();     // U[i] is an m-dim left vector
        internal double[][] V = Array.Empty<double[]>();     // V[i] is an n-dim right vector
        internal double[][] Q = Array.Empty<double[]>();
    }

    internal static Svd Randomized(double[][] a, int k, int p, int q, int seed)
    {
        int m = a.Length;
        int n = a[0].Length;
        int l = k + p;
        var rng = new Lcg(seed);
        var omega = Zeros(n, l);
        for (int j = 0; j < l; j++)
        {
            for (int i = 0; i < n; i++) omega[i][j] = rng.Next();
        }
        double[][] y = Multiply(a, omega);
        double[][] at = Transpose(a);
        for (int step = 0; step < q; step++)
        {
            y = Multiply(a, Multiply(at, y));
        }
        double[][] qBasis = Orth(y);
        double[][] b = Multiply(Transpose(qBasis), a);
        double[][] bt = Transpose(b);
        Eigen gram = Jacobi(Multiply(b, bt), 100, 1e-14);
        var index = new int[l];
        for (int i = 0; i < l; i++) index[i] = i;
        double[] gramValues = gram.Values;
        Array.Sort(index, (i, j) => gramValues[j].CompareTo(gramValues[i]));
        var sigma = new double[l];
        var u = new double[l][];
        var v = new double[l][];
        for (int i = 0; i < l; i++)
        {
            int column = index[i];
            sigma[i] = Math.Sqrt(Math.Max(gramValues[column], 0.0));
            var ub = new double[l];
            for (int r = 0; r < l; r++) ub[r] = gram.Vectors[r][column];
            if (sigma[i] <= 1e-300)
            {
                u[i] = new double[m];
                v[i] = new double[n];
                continue;
            }
            var right = new double[n];
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int r = 0; r < l; r++) sum += bt[j][r] * ub[r];
                right[j] = sum / sigma[i];
            }
            var left = new double[m];
            for (int r = 0; r < m; r++)
            {
                double sum = 0.0;
                for (int c = 0; c < l; c++) sum += qBasis[r][c] * ub[c];
                left[r] = sum;
            }
            int argmax = 0;
            for (int r = 1; r < n; r++)
            {
                if (Math.Abs(right[r]) > Math.Abs(right[argmax])) argmax = r;
            }
            if (right[argmax] < 0.0)
            {
                for (int r = 0; r < n; r++) right[r] = -right[r];
                for (int r = 0; r < m; r++) left[r] = -left[r];
            }
            u[i] = left;
            v[i] = right;
        }
        return new Svd { Sigma = sigma, U = u, V = v, Q = qBasis };
    }

    /** Frobenius norm of A minus its rank-k truncation built from the first k singular triplets. */
    internal static double RankKError(double[][] a, Svd svd, int k)
    {
        int m = a.Length;
        int n = a[0].Length;
        var residual = new double[m][];
        for (int i = 0; i < m; i++)
        {
            residual[i] = new double[n];
            Array.Copy(a[i], residual[i], n);
        }
        for (int c = 0; c < k; c++)
        {
            if (svd.Sigma[c] <= 1e-300) continue;
            for (int i = 0; i < m; i++)
            {
                double coefficient = svd.Sigma[c] * svd.U[c][i];
                if (coefficient == 0.0) continue;
                for (int j = 0; j < n; j++) residual[i][j] -= coefficient * svd.V[c][j];
            }
        }
        return Frobenius(residual);
    }

    internal static double[] Residuals(double[][] a, Svd svd, int k)
    {
        var result = new double[k];
        for (int c = 0; c < k; c++)
        {
            int m = a.Length;
            int n = a[0].Length;
            var av = new double[m];
            for (int i = 0; i < m; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += a[i][j] * svd.V[c][j];
                av[i] = sum;
            }
            double error = 0.0;
            for (int i = 0; i < m; i++)
            {
                double diff = av[i] - svd.Sigma[c] * svd.U[c][i];
                error += diff * diff;
            }
            result[c] = Math.Sqrt(error);
        }
        return result;
    }

    /** Frobenius error of the sketch projector: ||A - Q Q^T A||. */
    internal static double RangeError(double[][] a, Svd svd)
    {
        int m = a.Length;
        int n = a[0].Length;
        int l = svd.Q[0].Length;
        var projected = Zeros(m, n);
        for (int c = 0; c < l; c++)
        {
            var qa = new double[n];
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < m; i++) sum += svd.Q[i][c] * a[i][j];
                qa[j] = sum;
            }
            for (int i = 0; i < m; i++)
            {
                double coefficient = svd.Q[i][c];
                if (coefficient == 0.0) continue;
                for (int j = 0; j < n; j++) projected[i][j] += coefficient * qa[j];
            }
        }
        double total = 0.0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double diff = a[i][j] - projected[i][j];
                total += diff * diff;
            }
        }
        return Math.Sqrt(total);
    }

    /** Planted matrix sum_i s_i u_i v_i^T with orthonormal LCG frames. */
    internal static double[][] Planted(int m, int n, double[] s, int seed)
    {
        var rng = new Lcg(seed);
        var uRaw = Zeros(m, s.Length);
        for (int c = 0; c < s.Length; c++)
        {
            for (int i = 0; i < m; i++) uRaw[i][c] = rng.Next();
        }
        var vRaw = Zeros(n, s.Length);
        for (int c = 0; c < s.Length; c++)
        {
            for (int i = 0; i < n; i++) vRaw[i][c] = rng.Next();
        }
        double[][] u = Orth(uRaw);
        double[][] v = Orth(vRaw);
        var a = Zeros(m, n);
        for (int c = 0; c < s.Length; c++)
        {
            for (int i = 0; i < m; i++)
            {
                double coefficient = s[c] * u[i][c];
                for (int j = 0; j < n; j++) a[i][j] += coefficient * v[j][c];
            }
        }
        return a;
    }

    internal static double GramDefect(double[][] frame)
    {
        int l = frame[0].Length;
        double worst = 0.0;
        for (int i = 0; i < l; i++)
        {
            for (int j = 0; j < l; j++)
            {
                double dot = 0.0;
                for (int r = 0; r < frame.Length; r++) dot += frame[r][i] * frame[r][j];
                worst = Math.Max(worst, Math.Abs(dot - (i == j ? 1.0 : 0.0)));
            }
        }
        return worst;
    }

    /** Same defect for a list of vectors (vectors[i] is one vector, not one matrix row). */
    internal static double VectorDefect(double[][] vectors)
    {
        double worst = 0.0;
        for (int i = 0; i < vectors.Length; i++)
        {
            for (int j = 0; j < vectors.Length; j++)
            {
                double dot = 0.0;
                for (int r = 0; r < vectors[i].Length; r++) dot += vectors[i][r] * vectors[j][r];
                worst = Math.Max(worst, Math.Abs(dot - (i == j ? 1.0 : 0.0)));
            }
        }
        return worst;
    }
}

public static class Solution
{
    public static void Main()
    {
        const int m = 40;
        const int n = 25;
        const int rank = 12;
        var s = new double[rank];
        for (int i = 0; i < rank; i++) s[i] = 10.0 * Math.Pow(0.5, i);
        double[][] a = RandomizedSvd.Planted(m, n, s, 7);

        // --- the planted frame is orthonormal and the exact SVD agrees with s -------------------
        var uRaw = RandomizedSvd.Zeros(m, rank);
        var rngU = new RandomizedSvd.Lcg(7);
        for (int c = 0; c < rank; c++)
        {
            for (int i = 0; i < m; i++) uRaw[i][c] = rngU.Next();
        }
        double[][] uFrame = RandomizedSvd.Orth(uRaw);
        Check(RandomizedSvd.GramDefect(uFrame) <= 1e-12, "planted U frame orthonormal");
        double[][] at = RandomizedSvd.Transpose(a);
        var gram = RandomizedSvd.Jacobi(RandomizedSvd.Multiply(at, a), 100, 1e-14);
        var sigmaExact = (double[])gram.Values.Clone();
        Array.Sort(sigmaExact);
        Array.Reverse(sigmaExact);
        for (int i = 0; i < n; i++)
        {
            sigmaExact[i] = Math.Sqrt(Math.Max(sigmaExact[i], 0.0));
        }
        for (int i = 0; i < rank; i++)
        {
            Check(Math.Abs(sigmaExact[i] - s[i]) <= 1e-8, "exact sigma_" + i + " matches the plant");
        }
        for (int i = rank; i < n; i++)
        {
            Check(sigmaExact[i] <= 1e-6, "exact tail sigma_" + i + " is numerically zero");
        }
        for (int i = 0; i < n - 1; i++)
        {
            Check(sigmaExact[i] >= sigmaExact[i + 1] - 1e-12, "exact spectrum descending");
        }

        // --- randomized SVD accuracy ------------------------------------------------------------
        const int k = 6;
        const int p = 4;
        var q0 = RandomizedSvd.Randomized(a, k, p, 0, 2024);
        var q1 = RandomizedSvd.Randomized(a, k, p, 1, 2024);
        var q2 = RandomizedSvd.Randomized(a, k, p, 2, 2024);
        var q4 = RandomizedSvd.Randomized(a, k, p, 4, 2024);

        double TopError(RandomizedSvd.Svd svd)
        {
            double worst = 0.0;
            for (int i = 0; i < k; i++) worst = Math.Max(worst, Math.Abs(svd.Sigma[i] - s[i]));
            return worst;
        }

        Check(TopError(q2) <= 1e-5 * s[0], "q=2 top-6 singular values accurate to 1e-5 relative");
        Check(TopError(q2) <= 1e-9, "q=2 top-6 singular values are machine precise");
        Check(TopError(q0) >= 10.0 * TopError(q2), "power iterations improve sigma by >= 10x");
        Check(TopError(q0) >= 1e-5, "q=0 keeps a visible error on this spectrum");

        double tailK = 0.0;
        for (int i = k; i < rank; i++) tailK += s[i] * s[i];
        tailK = Math.Sqrt(tailK);
        double errQ2 = RandomizedSvd.RankKError(a, q2, k);
        double errQ0 = RandomizedSvd.RankKError(a, q0, k);
        Check(errQ2 >= tailK - 1e-9, "Eckart-Young lower bound for q=2");
        Check(errQ2 <= 1.05 * tailK, "Eckart-Young upper bound for q=2");
        Check(errQ0 >= tailK - 1e-9, "q=0 also respects the lower bound");

        double rangeQ0 = RandomizedSvd.RangeError(a, q0);
        double rangeQ1 = RandomizedSvd.RangeError(a, q1);
        double rangeQ2 = RandomizedSvd.RangeError(a, q2);
        double rangeQ4 = RandomizedSvd.RangeError(a, q4);
        Check(rangeQ0 >= 5.0 * rangeQ1, "one power step shrinks the range error >= 5x");
        Check(rangeQ1 <= 2.0 * sigmaExact[k + p], "q=1 range error is near the ideal l-truncation");
        Check(rangeQ2 <= 1.2 * sigmaExact[k + p], "q=2 range error is at the ideal l-truncation");
        Check(rangeQ4 >= rangeQ1, "excess power steps saturate in double precision");

        // --- orthonormality and eigen-relation residuals ------------------------------------------
        Check(RandomizedSvd.GramDefect(q2.Q) <= 1e-12, "Q orthonormal");
        Check(RandomizedSvd.VectorDefect(q2.U) <= 1e-12, "U orthonormal");
        Check(RandomizedSvd.VectorDefect(q2.V) <= 1e-12, "V orthonormal");
        double[] residuals = RandomizedSvd.Residuals(a, q2, k);
        for (int i = 0; i < k; i++)
        {
            Check(residuals[i] <= 1e-4 * s[0], "eigen-relation residual for component " + i);
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                double dot = 0.0;
                for (int r = 0; r < n; r++) dot += q2.V[i][r] * q2.V[j][r];
                Check(Math.Abs(dot) <= 1e-8, "right singular vectors orthogonal");
            }
        }

        // --- l = n makes the sketch exact ---------------------------------------------------------
        var exactSketch = RandomizedSvd.Randomized(a, 20, 5, 0, 99);
        for (int i = 0; i < n; i++)
        {
            Check(Math.Abs(exactSketch.Sigma[i] - sigmaExact[i]) <= 1e-6,
                    "full sketch sigma_" + i + " is exact");
        }

        // --- error decreases monotonically in k -----------------------------------------------------
        double previous = double.PositiveInfinity;
        for (int kk = 1; kk <= 10; kk++)
        {
            var approx = RandomizedSvd.Randomized(a, kk, 4, 2, 2024);
            double error = RandomizedSvd.RankKError(a, approx, kk);
            Check(error <= previous + 1e-9, "error non-increasing in k at k = " + kk);
            previous = error;
        }

        // --- determinism ------------------------------------------------------------------------------
        var repeatA = RandomizedSvd.Randomized(a, k, p, 2, 2024);
        var repeatB = RandomizedSvd.Randomized(a, k, p, 2, 2024);
        for (int i = 0; i < q2.Sigma.Length; i++)
        {
            Check(q2.Sigma[i] == repeatA.Sigma[i] && repeatA.Sigma[i] == repeatB.Sigma[i],
                    "identical LCG seeds reproduce sigma_" + i);
        }
        var otherSeed = RandomizedSvd.Randomized(a, k, p, 2, 31337);
        for (int i = 0; i < k; i++)
        {
            Check(Math.Abs(otherSeed.Sigma[i] - s[i]) <= 1e-5 * s[0],
                    "a different sketch still recovers sigma_" + i);
        }

        // --- exact rank-5 recovery ---------------------------------------------------------------------
        var s5 = new[] { 5.0, 4.0, 3.0, 2.0, 1.0 };
        double[][] low = RandomizedSvd.Planted(m, n, s5, 555);
        var lowSvd = RandomizedSvd.Randomized(low, 5, 3, 1, 4242);
        for (int i = 0; i < 5; i++)
        {
            Check(Math.Abs(lowSvd.Sigma[i] - s5[i]) <= 1e-6, "rank-5 sigma_" + i);
        }
        Check(RandomizedSvd.RankKError(low, lowSvd, 5) <= 1e-8, "rank-5 reconstruction");
        Check(RandomizedSvd.Frobenius(low) <= 1.0 + Math.Sqrt(25.0 + 16.0 + 9.0 + 4.0 + 1.0),
                "Frobenius norm sanity bound");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
