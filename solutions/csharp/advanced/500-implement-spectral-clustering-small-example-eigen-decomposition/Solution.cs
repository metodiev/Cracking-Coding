// Task 500 - Implement spectral clustering small example (eigen decomposition). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Spectral 2-clustering of a handful of points: build the Gaussian similarity matrix
// W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), form the combinatorial Laplacian L = D - W (D the
// row sums), and take the Fiedler vector - the eigenvector of the second-smallest eigenvalue -
// via cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below 1e-12,
// documented cap 100 sweeps). The split threshold is the mean of the Fiedler vector, with the
// sign fixed by making point 0 lie at or above the mean. Documented checks: the zero eigenvalue
// and the residual ||L v - lambda2 v|| (<= 1e-9), lambda2 < lambda3 with the vector orthogonal
// to the constant, and the resulting cut equals the minimum over all balanced 6-6 cuts of the
// canonical two-blob example (the planted split). The canonical example is two six-point
// clusters around (0,0) and (10,10) with sigma = 3.0 (cross-cluster similarities ~1e-4, so
// lambda2 ~ 4.85e-4 - comfortably above the Jacobi tolerance); the tiny second example is two
// LCG-jittered 4-point blobs around (0,0) and (3,0) with sigma = 0.5, where the Fiedler sign
// pattern must again reproduce the planted partition.
using System;
using System.Collections.Generic;

internal static class SpectralDemo
{
    internal static readonly double[][] Canonical = {
        new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 },
        new[] { 0.5, 1.5 }, new[] { 1.5, 0.5 }, new[] { 10.0, 10.0 }, new[] { 11.0, 10.0 },
        new[] { 10.0, 11.0 }, new[] { 11.0, 11.0 }, new[] { 10.5, 10.5 }, new[] { 9.5, 10.5 },
    };
    internal static readonly int[] CanonicalTruth = { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };
    internal const double SigmaCanonical = 3.0;
    internal const double SigmaTiny = 0.5;

    internal static double Sqdist(double[] a, double[] b)
    {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    /** Two 4-point blobs around (0,0) and (3,0), jitter +/-0.4 in 0.01 steps via the LCG. */
    internal static double[][] TinyBlobs(int seed, out int[] truth)
    {
        var points = new double[8][];
        int index = 0;
        foreach (double cx in new[] { 0.0, 3.0 })
        {
            for (int i = 0; i < 4; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                double x = cx + (seed % 81 - 40) / 100.0;
                seed = (seed * 75 + 74) % 65537;
                double y = (seed % 81 - 40) / 100.0;
                points[index++] = new[] { x, y };
            }
        }
        truth = new[] { 0, 0, 0, 0, 1, 1, 1, 1 };
        return points;
    }

    internal static double[][] Similarity(double[][] points, double sigma)
    {
        int n = points.Length;
        var w = new double[n][];
        for (int i = 0; i < n; i++) w[i] = new double[n];
        double denom = 2.0 * sigma * sigma;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double value = Math.Exp(-Sqdist(points[i], points[j]) / denom);
                w[i][j] = value;
                w[j][i] = value;
            }
        }
        return w;
    }

    internal static double[][] Laplacian(double[][] w)
    {
        int n = w.Length;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            double degree = 0.0;
            for (int j = 0; j < n; j++) degree += w[i][j];
            result[i] = new double[n];
            for (int j = 0; j < n; j++) result[i][j] = (i == j ? degree : 0.0) - w[i][j];
        }
        return result;
    }

    internal class Eigen
    {
        internal double[] Values = Array.Empty<double>();
        internal double[][] Vectors = Array.Empty<double[]>();
        internal int Sweeps;
    }

    /** Cyclic Jacobi eigen-decomposition of a symmetric matrix; eigenvectors are columns. */
    internal static Eigen JacobiEigh(double[][] matrix, int maxSweeps, double tol)
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

    internal class Split
    {
        internal int[] Labels = Array.Empty<int>();
        internal double Lam2;
        internal double Residual;
        internal double Orthogonality;
        internal int Sweeps;
    }

    internal static Split SpectralSplit(double[][] points, double sigma)
    {
        int n = points.Length;
        double[][] w = Similarity(points, sigma);
        double[][] lmat = Laplacian(w);
        Eigen eigen = JacobiEigh(lmat, 100, 1e-12);
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (x, y) => eigen.Values[x].CompareTo(eigen.Values[y]));
        int fiedler = order[1];
        double lam2 = eigen.Values[fiedler];
        var vec = new double[n];
        double mean = 0.0;
        for (int i = 0; i < n; i++)
        {
            vec[i] = eigen.Vectors[i][fiedler];
            mean += vec[i];
        }
        mean /= n;
        if (vec[0] < mean)
        {
            for (int i = 0; i < n; i++) vec[i] = -vec[i];
        }
        var labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = vec[i] >= mean ? 0 : 1;
        double residual = 0.0;
        double orthogonality = 0.0;
        for (int i = 0; i < n; i++)
        {
            double acc = 0.0;
            for (int j = 0; j < n; j++) acc += lmat[i][j] * vec[j];
            residual += (acc - lam2 * vec[i]) * (acc - lam2 * vec[i]);
            orthogonality += vec[i];
        }
        return new Split
        {
            Labels = labels,
            Lam2 = lam2,
            Residual = Math.Sqrt(residual),
            Orthogonality = Math.Abs(orthogonality),
            Sweeps = eigen.Sweeps,
        };
    }

    internal static double CutWeight(double[][] w, int[] side)
    {
        int n = w.Length;
        double total = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (side[i] == 0 && side[j] == 1) total += w[i][j];
            }
        }
        return total;
    }

    /** Brute force over all balanced half-half splits (point 0 in side A, mirrors skipped). */
    internal static double MinBalancedCut(double[][] w, int half)
    {
        int n = w.Length;
        int choose = half - 1;
        var idx = new int[choose];
        for (int i = 0; i < choose; i++) idx[i] = i + 1;
        double best = double.PositiveInfinity;
        while (true)
        {
            var side = new int[n];
            for (int i = 0; i < n; i++) side[i] = 1;
            side[0] = 0;
            foreach (int i in idx) side[i] = 0;
            if (side[0] != side[n - 1]) best = Math.Min(best, CutWeight(w, side));
            int i2 = choose - 1;
            while (i2 >= 0 && idx[i2] == n - choose + i2) i2--;
            if (i2 < 0) break;
            idx[i2] += 1;
            for (int j = i2 + 1; j < choose; j++) idx[j] = idx[j - 1] + 1;
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        SpectralDemo.Eigen small = SpectralDemo.JacobiEigh(
                new[] { new[] { 2.0, 1.0 }, new[] { 1.0, 2.0 } }, 100, 1e-12);
        double[] smallSorted = (double[])small.Values.Clone();
        Array.Sort(smallSorted);
        Check(Math.Abs(smallSorted[0] - 1.0) <= 1e-12 && Math.Abs(smallSorted[1] - 3.0) <= 1e-12,
                "2x2 eigenvalues");

        SpectralDemo.Split canonical = SpectralDemo.SpectralSplit(
                SpectralDemo.Canonical, SpectralDemo.SigmaCanonical);
        Check(canonical.Residual <= 1e-9, "eigen-pair residual");
        Check(canonical.Orthogonality <= 1e-8, "Fiedler vector orthogonal to the constant");
        Check(canonical.Lam2 >= 1e-6 && canonical.Lam2 <= 1e-2,
                "two separated blobs: tiny positive Fiedler value");
        Check(canonical.Sweeps <= 100, "sweep cap");

        double[][] w = SpectralDemo.Similarity(SpectralDemo.Canonical,
                SpectralDemo.SigmaCanonical);
        SpectralDemo.Eigen spectrum = SpectralDemo.JacobiEigh(SpectralDemo.Laplacian(w), 100, 1e-12);
        double[] asc = (double[])spectrum.Values.Clone();
        Array.Sort(asc);
        Check(Math.Abs(asc[0]) <= 1e-9, "connected graph: one zero eigenvalue");
        Check(asc[1] < asc[2], "Fiedler eigenvalue is strictly second smallest");
        Check(asc[1] <= 1e-2 * asc[2], "documented: the Fiedler value separates the blobs");

        for (int i = 0; i < SpectralDemo.CanonicalTruth.Length; i++)
        {
            Check(canonical.Labels[i] == SpectralDemo.CanonicalTruth[i],
                    "canonical split must match the planted structure at " + i);
        }
        double spectralCut = SpectralDemo.CutWeight(w, canonical.Labels);
        double bestBalanced = SpectralDemo.MinBalancedCut(w, 6);
        Check(Math.Abs(spectralCut - bestBalanced) <= 1e-9,
                "spectral cut equals the min 6-6 cut");
        double totalMass = 0.0;
        foreach (double[] row in w)
        {
            foreach (double value in row) totalMass += value;
        }
        Check(spectralCut <= 1e-3 * totalMass, "cut fraction");

        double[][] tiny = SpectralDemo.TinyBlobs(2024, out int[] truth);
        SpectralDemo.Split tinySplit = SpectralDemo.SpectralSplit(tiny, SpectralDemo.SigmaTiny);
        for (int i = 0; i < truth.Length; i++)
        {
            Check(tinySplit.Labels[i] == truth[i], "tiny LCG blobs split correctly at " + i);
        }
        Check(tinySplit.Residual <= 1e-9 && tinySplit.Orthogonality <= 1e-8, "tiny residuals");
        Check(tinySplit.Lam2 >= 1e-8 && tinySplit.Lam2 <= 1e-2, "tiny positive Fiedler value");
        SpectralDemo.Split tinyRepeat = SpectralDemo.SpectralSplit(tiny, SpectralDemo.SigmaTiny);
        for (int i = 0; i < truth.Length; i++)
        {
            Check(tinyRepeat.Labels[i] == tinySplit.Labels[i], "deterministic re-run at " + i);
        }

        var spaced = new double[tiny.Length][];
        for (int i = 0; i < tiny.Length; i++)
        {
            spaced[i] = new[] { tiny[i][0] * 10.0, tiny[i][1] * 10.0 };
        }
        SpectralDemo.Split spacedSplit = SpectralDemo.SpectralSplit(
                spaced, SpectralDemo.SigmaTiny * 10.0);
        for (int i = 0; i < truth.Length; i++)
        {
            Check(spacedSplit.Labels[i] == truth[i], "scaled blobs split correctly at " + i);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
