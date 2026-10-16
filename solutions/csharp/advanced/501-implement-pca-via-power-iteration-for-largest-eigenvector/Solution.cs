// Task 501 - Implement PCA via power iteration for largest eigenvector. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// PCA built from first principles: centre the data, form the biased (1/n) covariance matrix
// C = X^T X / n, then extract eigenpairs. The leading eigenpair comes from power iteration
// (w = C v, normalise, repeat until ||v_{k+1} - v_k|| <= 1e-13 or 500 iterations; the
// largest-magnitude component is forced positive so the iterate cannot flip sign), reported
// with its Rayleigh quotient lambda = v^T C v / v^T v and the residual ||C v - lambda v||.
// The remaining eigenpairs come from a full cyclic Jacobi decomposition of the same symmetric
// matrix, which doubles as the independent oracle (Jacobi sweeps until the off-diagonal
// Frobenius norm <= 1e-14, cap 100 sweeps). Documented checks:
//   * closed-form covariance of (1,0,0), (-1,0,0), (0,0.5,0), (0,-0.5,0) is diag(0.5, 0.125, 0),
//     so the leading eigenpair is exactly (0.5, e1) and the Jacobi spectrum must agree;
//   * the classic C = [[2,1],[1,2]] has lambda1 = 3 with eigenvector (1,1)/sqrt(2);
//   * |Rayleigh - lambda| <= 1e-12, residual <= 1e-10, and the Rayleigh quotient of the power
//     iterates is non-decreasing (power iteration is exact gradient ascent on R(v));
//   * lambda1 is the maximum Rayleigh quotient over 4000 deterministic LCG directions;
//   * trace identity sum_i lambda_i = trace(C) and the rank-k reconstruction identity
//     mean ||x - P_k x||^2 = sum_{i>k} lambda_i for k = 0..3 (checked against a real
//     projection-and-reconstruct pass);
//   * shift invariance (add 1000 to every coordinate -> identical eigenpairs) and quadratic
//     scaling (3x data -> 9x eigenvalues, same directions);
//   * on the planted 3-D model (orthonormal axes with amplitudes 3.0 / 1.5 / 0.5 and LCG
//     coefficients in [-1,1]) the first component recovers the planted axis (|cos| >= 1-1e-2)
//     and explains the documented >= 75% of the total variance.
using System;
using System.Collections.Generic;

internal static class LinearAlgebra
{
    internal static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    internal static double[] MatVec(double[][] m, double[] v)
    {
        var result = new double[m.Length];
        for (int i = 0; i < m.Length; i++)
        {
            double[] row = m[i];
            double sum = 0.0;
            for (int j = 0; j < v.Length; j++) sum += row[j] * v[j];
            result[i] = sum;
        }
        return result;
    }

    internal static double[] Sub(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }

    internal static double[] Scale(double[] a, double factor)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] * factor;
        return result;
    }

    internal static double[] Unit(double[] a)
    {
        double norm = Norm(a);
        return Scale(a, 1.0 / norm);
    }

    internal static double Rayleigh(double[][] m, double[] v) => Dot(v, MatVec(m, v)) / Dot(v, v);

    internal class Eigen
    {
        internal double[] Values = Array.Empty<double>();
        internal double[][] Vectors = Array.Empty<double[]>();   // Vectors[c] is column c
        internal int Sweeps;
    }

    /** Cyclic Jacobi eigen-decomposition of a symmetric matrix (eigenvectors are columns). */
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
        int sweeps = 0;
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0.0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (Math.Sqrt(off) <= tol) break;
            sweeps += 1;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) <= 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = 1.0 / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta < 0.0) t = -t;
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        if (k == p || k == q) continue;
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                        a[p][k] = a[k][p];
                        a[q][k] = a[k][q];
                    }
                    double app = a[p][p];
                    double aqq = a[q][q];
                    double apq = a[p][q];
                    a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                    a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        return new Eigen { Values = values, Vectors = v, Sweeps = sweeps };
    }

    /** Eigenvalues descending; vectors are re-ordered and flipped so the largest entry is positive. */
    internal static void SortDescending(Eigen eigen)
    {
        int n = eigen.Values.Length;
        var index = new int[n];
        for (int i = 0; i < n; i++) index[i] = i;
        double[] values = eigen.Values;
        Array.Sort(index, (i, j) => values[j].CompareTo(values[i]));
        var sortedValues = new double[n];
        var sortedVectors = new double[n][];
        for (int i = 0; i < n; i++) sortedValues[i] = values[index[i]];
        for (int i = 0; i < n; i++)
        {
            var column = new double[n];
            for (int r = 0; r < n; r++) column[r] = eigen.Vectors[r][index[i]];
            int argmax = 0;
            for (int r = 1; r < n; r++)
            {
                if (Math.Abs(column[r]) > Math.Abs(column[argmax])) argmax = r;
            }
            if (column[argmax] < 0.0) column = Scale(column, -1.0);
            sortedVectors[i] = column;
        }
        eigen.Values = sortedValues;
        eigen.Vectors = sortedVectors;
    }
}

internal static class Pca
{
    internal static double[][] Center(double[][] data, out double[] mean)
    {
        int n = data.Length;
        int d = data[0].Length;
        mean = new double[d];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++) mean[j] += data[i][j];
        }
        for (int j = 0; j < d; j++) mean[j] /= n;
        var centered = new double[n][];
        for (int i = 0; i < n; i++)
        {
            centered[i] = new double[d];
            for (int j = 0; j < d; j++) centered[i][j] = data[i][j] - mean[j];
        }
        return centered;
    }

    /** Biased covariance (divide by n) of already-centred data. */
    internal static double[][] Covariance(double[][] centered)
    {
        int n = centered.Length;
        int d = centered[0].Length;
        var c = new double[d][];
        for (int i = 0; i < d; i++) c[i] = new double[d];
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < d; i++)
            {
                double xi = centered[k][i];
                for (int j = i; j < d; j++) c[i][j] += xi * centered[k][j];
            }
        }
        for (int i = 0; i < d; i++)
        {
            for (int j = i; j < d; j++)
            {
                double value = c[i][j] / n;
                c[i][j] = value;
                c[j][i] = value;
            }
        }
        return c;
    }

    internal class PowerResult
    {
        internal double[] Vector = Array.Empty<double>();
        internal double Eigenvalue;
        internal int Iterations;
        internal bool Converged;
        internal double Residual;
        internal List<double> RayleighTrace = new List<double>();
    }

    /** Power iteration for the dominant eigenpair of a symmetric matrix. */
    internal static PowerResult PowerIteration(double[][] matrix, double[] start, int maxIterations,
            double tolerance)
    {
        int d = matrix.Length;
        var v = LinearAlgebra.Unit((double[])start.Clone());
        var result = new PowerResult();
        result.RayleighTrace.Add(LinearAlgebra.Rayleigh(matrix, v));
        for (int iteration = 1; iteration <= maxIterations; iteration++)
        {
            double[] w = LinearAlgebra.MatVec(matrix, v);
            double norm = LinearAlgebra.Norm(w);
            if (norm <= 1e-300)
            {
                result.Converged = true;   // zero matrix: every direction is an eigenvector
                break;
            }
            w = LinearAlgebra.Scale(w, 1.0 / norm);
            int argmax = 0;
            for (int i = 1; i < d; i++)
            {
                if (Math.Abs(w[i]) > Math.Abs(w[argmax])) argmax = i;
            }
            if (w[argmax] < 0.0) w = LinearAlgebra.Scale(w, -1.0);
            double shift = LinearAlgebra.Norm(LinearAlgebra.Sub(w, v));
            v = w;
            result.Iterations = iteration;
            result.RayleighTrace.Add(LinearAlgebra.Rayleigh(matrix, v));
            if (shift <= tolerance)
            {
                result.Converged = true;
                break;
            }
        }
        result.Vector = v;
        result.Eigenvalue = LinearAlgebra.Rayleigh(matrix, v);
        result.Residual = LinearAlgebra.Norm(LinearAlgebra.Sub(LinearAlgebra.MatVec(matrix, v),
                LinearAlgebra.Scale(v, result.Eigenvalue)));
        return result;
    }

    /** Mean squared error of the rank-k PCA reconstruction of centred data. */
    internal static double ReconstructionMse(double[][] centered, double[][] components, int k)
    {
        double total = 0.0;
        foreach (double[] point in centered)
        {
            var reconstruction = new double[point.Length];
            for (int c = 0; c < k; c++)
            {
                double coefficient = LinearAlgebra.Dot(point, components[c]);
                for (int j = 0; j < point.Length; j++) reconstruction[j] += coefficient * components[c][j];
            }
            double[] error = LinearAlgebra.Sub(point, reconstruction);
            total += LinearAlgebra.Dot(error, error);
        }
        return total / centered.Length;
    }

    internal static double Trace(double[][] matrix)
    {
        double sum = 0.0;
        for (int i = 0; i < matrix.Length; i++) sum += matrix[i][i];
        return sum;
    }

    /** Deterministic LCG unit directions in d dimensions. */
    internal static double[][] LcgDirections(int count, int d, ref int seed)
    {
        var directions = new double[count][];
        for (int c = 0; c < count; c++)
        {
            var direction = new double[d];
            for (int j = 0; j < d; j++)
            {
                seed = (seed * 75 + 74) % 65537;
                direction[j] = (seed % 2001 - 1000) / 1000.0;
            }
            directions[c] = LinearAlgebra.Unit(direction);
        }
        return directions;
    }

    /** Planted 3-D model: x = a1 t1 v1 + a2 t2 v2 + a3 t3 v3 with LCG coefficients. */
    internal static double[][] PlantedData(int n, double[] amplitudes, double[][] axes, ref int seed)
    {
        var points = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var point = new double[axes.Length];
            for (int a = 0; a < axes.Length; a++)
            {
                seed = (seed * 75 + 74) % 65537;
                double coefficient = amplitudes[a] * (seed % 2001 - 1000) / 1000.0;
                for (int j = 0; j < axes[a].Length; j++) point[j] += coefficient * axes[a][j];
            }
            points[i] = point;
        }
        return points;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- closed-form diagonal covariance --------------------------------------------------
        double[][] axisAligned = {
            new[] { 1.0, 0.0, 0.0 }, new[] { -1.0, 0.0, 0.0 },
            new[] { 0.0, 0.5, 0.0 }, new[] { 0.0, -0.5, 0.0 },
        };
        double[][] centered = Pca.Center(axisAligned, out _);
        double[][] cov = Pca.Covariance(centered);
        Check(Math.Abs(cov[0][0] - 0.5) <= 1e-15 && Math.Abs(cov[1][1] - 0.125) <= 1e-15
                && Math.Abs(cov[2][2]) <= 1e-15, "closed-form covariance diag(0.5, 0.125, 0)");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (i != j) Check(Math.Abs(cov[i][j]) <= 1e-15, "closed-form covariance is diagonal");
            }
        }
        var diagonalPower = Pca.PowerIteration(cov, new[] { 1.0, 1.0, 1.0 }, 500, 1e-13);
        Check(diagonalPower.Converged, "power iteration converged on the diagonal covariance");
        Check(Math.Abs(diagonalPower.Eigenvalue - 0.5) <= 1e-12, "lambda1 = 0.5");
        Check(Math.Abs(Math.Abs(diagonalPower.Vector[0]) - 1.0) <= 1e-12
                && Math.Abs(diagonalPower.Vector[1]) <= 1e-12
                && Math.Abs(diagonalPower.Vector[2]) <= 1e-12, "leading vector is e1");
        Check(diagonalPower.Residual <= 1e-12, "diagonal covariance residual");
        var diagonalEigen = LinearAlgebra.Jacobi(cov, 100, 1e-14);
        LinearAlgebra.SortDescending(diagonalEigen);
        Check(Math.Abs(diagonalEigen.Values[0] - 0.5) <= 1e-12
                && Math.Abs(diagonalEigen.Values[1] - 0.125) <= 1e-12
                && Math.Abs(diagonalEigen.Values[2]) <= 1e-12, "Jacobi spectrum diag(0.5, 0.125, 0)");

        // --- classic 2x2 with lambda1 = 3 -----------------------------------------------------
        double[][] classic = { new[] { 2.0, 1.0 }, new[] { 1.0, 2.0 } };
        var classicPower = Pca.PowerIteration(classic, new[] { 1.0, 0.0 }, 500, 1e-13);
        Check(Math.Abs(classicPower.Eigenvalue - 3.0) <= 1e-12, "[[2,1],[1,2]]: lambda1 = 3");
        Check(Math.Abs(Math.Abs(classicPower.Vector[0]) - Math.Sqrt(0.5)) <= 1e-12
                && Math.Abs(Math.Abs(classicPower.Vector[1]) - Math.Sqrt(0.5)) <= 1e-12,
                "[[2,1],[1,2]]: leading vector (1,1)/sqrt(2)");
        Check(classicPower.Residual <= 1e-12, "classic residual");
        Check(Math.Abs(LinearAlgebra.Rayleigh(classic, classicPower.Vector) - classicPower.Eigenvalue)
                <= 1e-12, "Rayleigh quotient equals reported eigenvalue");

        // --- Rayleigh trace is non-decreasing -------------------------------------------------
        for (int i = 1; i < diagonalPower.RayleighTrace.Count; i++)
        {
            Check(diagonalPower.RayleighTrace[i] >= diagonalPower.RayleighTrace[i - 1] - 1e-15,
                    "Rayleigh quotient non-decreasing at step " + i);
        }

        // --- planted 3-D model -----------------------------------------------------------------
        var axes = new double[][] {
            LinearAlgebra.Unit(new[] { 1.0, 1.0, 1.0 }),
            LinearAlgebra.Unit(new[] { 1.0, -1.0, 0.0 }),
            LinearAlgebra.Unit(new[] { 1.0, 1.0, -2.0 }),
        };
        Check(Math.Abs(LinearAlgebra.Dot(axes[0], axes[1])) <= 1e-15
                && Math.Abs(LinearAlgebra.Dot(axes[0], axes[2])) <= 1e-15
                && Math.Abs(LinearAlgebra.Dot(axes[1], axes[2])) <= 1e-15, "planted axes orthonormal");
        int seed = 12345;
        double[][] data = Pca.PlantedData(200, new[] { 3.0, 1.5, 0.5 }, axes, ref seed);
        double[][] dataCentered = Pca.Center(data, out double[] mean);
        double[][] dataCov = Pca.Covariance(dataCentered);
        var power = Pca.PowerIteration(dataCov, new[] { 1.0, 0.0, 0.0 }, 500, 1e-13);
        var eigen = LinearAlgebra.Jacobi(dataCov, 100, 1e-14);
        LinearAlgebra.SortDescending(eigen);
        Check(power.Converged && power.Iterations <= 200, "planted data converged");
        Check(Math.Abs(power.Eigenvalue - eigen.Values[0]) <= 1e-10,
                "power eigenvalue matches Jacobi lambda1");
        Check(Math.Abs(LinearAlgebra.Dot(power.Vector, eigen.Vectors[0])) >= 1.0 - 1e-10,
                "power vector matches Jacobi vector");
        Check(Math.Abs(LinearAlgebra.Dot(power.Vector, axes[0])) >= 1.0 - 1e-2,
                "leading direction recovers the planted axis");
        Check(eigen.Values[0] > eigen.Values[1] && eigen.Values[1] > eigen.Values[2]
                && eigen.Values[2] > 0.0, "three distinct positive eigenvalues");
        double trace = Pca.Trace(dataCov);
        Check(Math.Abs(eigen.Values[0] + eigen.Values[1] + eigen.Values[2] - trace) <= 1e-10,
                "trace identity");
        double explained = eigen.Values[0] / trace;
        Check(explained >= 0.75 && explained <= 1.0, "PC1 explains at least 75% of the variance");
        for (int j = 0; j < 3; j++)
        {
            double spread = Math.Sqrt(dataCov[j][j]);
            Check(Math.Abs(mean[j]) <= 0.25 * spread, "planted data centred within sampling noise");
        }

        // --- lambda1 is the maximum Rayleigh quotient over many directions ----------------------
        int directionSeed = 777;
        double[][] directions = Pca.LcgDirections(4000, 3, ref directionSeed);
        double best = double.NegativeInfinity;
        foreach (double[] direction in directions)
        {
            best = Math.Max(best, LinearAlgebra.Rayleigh(dataCov, direction));
        }
        Check(best <= eigen.Values[0] + 1e-12, "lambda1 dominates every sampled Rayleigh quotient");
        Check(best >= 0.99 * eigen.Values[0], "sampled directions get close to lambda1");

        // --- rank-k reconstruction identity -----------------------------------------------------
        double tail = eigen.Values[0] + eigen.Values[1] + eigen.Values[2];
        for (int k = 0; k <= 3; k++)
        {
            double mse = Pca.ReconstructionMse(dataCentered, eigen.Vectors, k);
            Check(Math.Abs(mse - tail) <= 1e-10, "reconstruction identity at k = " + k);
            if (k < 3) tail -= eigen.Values[k];
        }
        Check(Pca.ReconstructionMse(dataCentered, eigen.Vectors, 3) <= 1e-20,
                "full-rank reconstruction is exact");
        Check(Pca.ReconstructionMse(dataCentered, eigen.Vectors, 1)
                < Pca.ReconstructionMse(dataCentered, eigen.Vectors, 0) - 1e-9,
                "one component already removes substantial variance");

        // --- shift invariance and quadratic scaling ---------------------------------------------
        var shifted = new double[data.Length][];
        for (int i = 0; i < data.Length; i++)
        {
            shifted[i] = new[] { data[i][0] + 1000.0, data[i][1] + 1000.0, data[i][2] + 1000.0 };
        }
        double[][] shiftedCov = Pca.Covariance(Pca.Center(shifted, out _));
        var shiftedPower = Pca.PowerIteration(shiftedCov, new[] { 1.0, 0.0, 0.0 }, 500, 1e-13);
        Check(Math.Abs(shiftedPower.Eigenvalue - power.Eigenvalue) <= 1e-9, "shift invariance: lambda1");
        Check(Math.Abs(LinearAlgebra.Dot(shiftedPower.Vector, power.Vector)) >= 1.0 - 1e-9,
                "shift invariance: eigenvector");

        var scaled = new double[data.Length][];
        for (int i = 0; i < data.Length; i++)
        {
            scaled[i] = LinearAlgebra.Scale(data[i], 3.0);
        }
        double[][] scaledCov = Pca.Covariance(Pca.Center(scaled, out _));
        var scaledPower = Pca.PowerIteration(scaledCov, new[] { 1.0, 0.0, 0.0 }, 500, 1e-13);
        Check(Math.Abs(scaledPower.Eigenvalue - 9.0 * power.Eigenvalue) <= 1e-8,
                "quadratic scaling: lambda1 scales by 9");
        Check(Math.Abs(LinearAlgebra.Dot(scaledPower.Vector, power.Vector)) >= 1.0 - 1e-8,
                "quadratic scaling: eigenvector unchanged");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
