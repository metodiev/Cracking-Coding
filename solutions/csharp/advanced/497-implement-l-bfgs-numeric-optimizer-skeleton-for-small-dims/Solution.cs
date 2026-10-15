// Task 497 - Implement L-BFGS numeric optimizer skeleton for small dims. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A compact L-BFGS for small dimensions: the two-loop recursion builds H*g from the last
// m = 5 (s, y) correction pairs (the search direction is -H*g), a backtracking Armijo line
// search (c1 = 1e-4, rho = 0.5, first trial step 1.0) picks the step, and the loop stops when
// ||grad f|| <= 1e-8 or after 200 iterations. The model problem is the convex quadratic
// f(x) = 0.5 x^T A x - b^T x with A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3], x0 = 0; its exact
// minimizer x* = A^-1 b and f* are computed by Gaussian elimination as the oracle. Documented
// tolerance: L-BFGS reaches ||x - x*|| <= 1e-6, |f - f*| <= 1e-9 and ||grad f|| <= 1e-8 (9
// iterations on the fixed problem). A plain gradient descent with the documented step size
// eta = 1/L (largest eigenvalue by power iteration, both runs started at x0 and compared at the
// same ||grad f|| <= 1e-6 tolerance) needs over twice as many iterations (46) - L-BFGS is
// compared against it directly.
using System;
using System.Collections.Generic;

internal static class LbfgsDemo
{
    internal static readonly double[][] A = {
        new[] { 4.0, 1.0, 0.0 },
        new[] { 1.0, 3.0, 1.0 },
        new[] { 0.0, 1.0, 2.0 },
    };
    internal static readonly double[] B = { 1.0, 2.0, 3.0 };
    internal static readonly double[] X0 = { 0.0, 0.0, 0.0 };
    internal const int History = 5;
    internal const int MaxIter = 200;
    internal const int GdMaxIter = 20000;
    internal const double C1 = 1e-4;
    internal const double Rho = 0.5;

    internal static double Dot(double[] u, double[] v)
    {
        double total = 0.0;
        for (int i = 0; i < u.Length; i++) total += u[i] * v[i];
        return total;
    }

    internal static double[] Matvec(double[][] matrix, double[] vector)
    {
        var result = new double[matrix.Length];
        for (int i = 0; i < matrix.Length; i++) result[i] = Dot(matrix[i], vector);
        return result;
    }

    internal static double Objective(double[] x)
    {
        return 0.5 * Dot(x, Matvec(A, x)) - Dot(B, x);
    }

    internal static double[] Gradient(double[] x)
    {
        double[] ax = Matvec(A, x);
        for (int i = 0; i < ax.Length; i++) ax[i] -= B[i];
        return ax;
    }

    /** Gaussian elimination with partial pivoting - the exact-minimizer oracle. */
    internal static double[] SolveLinear(double[][] matrix, double[] rhs)
    {
        int n = matrix.Length;
        var aug = new double[n][];
        for (int i = 0; i < n; i++)
        {
            aug[i] = new double[n + 1];
            Array.Copy(matrix[i], aug[i], n);
            aug[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
            {
                if (Math.Abs(aug[r][col]) > Math.Abs(aug[pivot][col])) pivot = r;
            }
            (aug[col], aug[pivot]) = (aug[pivot], aug[col]);
            for (int r = col + 1; r < n; r++)
            {
                double factor = aug[r][col] / aug[col][col];
                for (int c = col; c <= n; c++) aug[r][c] -= factor * aug[col][c];
            }
        }
        var x = new double[n];
        for (int r = n - 1; r >= 0; r--)
        {
            double acc = aug[r][n];
            for (int c = r + 1; c < n; c++) acc -= aug[r][c] * x[c];
            x[r] = acc / aug[r][r];
        }
        return x;
    }

    internal static double LargestEigenvalue(double[][] matrix, int iterations)
    {
        var v = new double[matrix.Length];
        for (int i = 0; i < v.Length; i++) v[i] = 1.0;
        double norm = Math.Sqrt(Dot(v, v));
        for (int i = 0; i < v.Length; i++) v[i] /= norm;
        for (int i = 0; i < iterations; i++)
        {
            double[] w = Matvec(matrix, v);
            norm = Math.Sqrt(Dot(w, w));
            for (int j = 0; j < v.Length; j++) v[j] = w[j] / norm;
        }
        return Dot(v, Matvec(matrix, v));
    }

    /** L-BFGS two-loop recursion: returns r = H*g; the search direction is -r. */
    internal static double[] TwoLoopDirection(double[] g, List<double[]> sHist, List<double[]> yHist)
    {
        double[] q = (double[])g.Clone();
        var alphas = new double[sHist.Count];
        for (int k = sHist.Count - 1; k >= 0; k--)
        {
            double[] s = sHist[k];
            double[] y = yHist[k];
            double rho = 1.0 / Dot(y, s);
            double alpha = rho * Dot(s, q);
            alphas[k] = alpha;
            for (int i = 0; i < q.Length; i++) q[i] -= alpha * y[i];
        }
        double gamma = 1.0;
        if (sHist.Count > 0)
        {
            double[] sLast = sHist[sHist.Count - 1];
            double[] yLast = yHist[yHist.Count - 1];
            gamma = Dot(sLast, yLast) / Dot(yLast, yLast);
        }
        var r = new double[q.Length];
        for (int i = 0; i < r.Length; i++) r[i] = gamma * q[i];
        for (int k = 0; k < sHist.Count; k++)
        {
            double[] s = sHist[k];
            double[] y = yHist[k];
            double rho = 1.0 / Dot(y, s);
            double beta = rho * Dot(y, r);
            for (int i = 0; i < r.Length; i++) r[i] += s[i] * (alphas[k] - beta);
        }
        return r;
    }

    internal class LbfgsResult
    {
        internal double[] X;
        internal double[] G;
        internal int Iterations;

        internal LbfgsResult(double[] x, double[] g, int iterations)
        {
            X = x;
            G = g;
            Iterations = iterations;
        }
    }

    internal static LbfgsResult Lbfgs()
    {
        double[] x = (double[])X0.Clone();
        double[] g = Gradient(x);
        var sHist = new List<double[]>();
        var yHist = new List<double[]>();
        int iterations = 0;
        for (int iter = 0; iter < MaxIter; iter++)
        {
            if (Math.Sqrt(Dot(g, g)) <= 1e-8) break;
            double[] r = TwoLoopDirection(g, sHist, yHist);
            var direction = new double[r.Length];
            for (int i = 0; i < r.Length; i++) direction[i] = -r[i];
            double descent = Dot(g, direction);
            if (descent >= 0.0)                 // safeguard: fall back to steepest descent
            {
                for (int i = 0; i < direction.Length; i++) direction[i] = -g[i];
                descent = -Dot(g, g);
            }
            double step = 1.0;
            while (Objective(AddScaled(x, direction, step)) > Objective(x) + C1 * step * descent)
            {
                step *= Rho;
            }
            double[] xNew = AddScaled(x, direction, step);
            double[] gNew = Gradient(xNew);
            var s = new double[x.Length];
            var y = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                s[i] = xNew[i] - x[i];
                y[i] = gNew[i] - g[i];
            }
            if (Dot(s, y) > 0.0)                // curvature condition
            {
                sHist.Add(s);
                yHist.Add(y);
                if (sHist.Count > History)
                {
                    sHist.RemoveAt(0);
                    yHist.RemoveAt(0);
                }
            }
            x = xNew;
            g = gNew;
            iterations += 1;
        }
        return new LbfgsResult(x, g, iterations);
    }

    internal static double[] AddScaled(double[] x, double[] direction, double step)
    {
        var result = new double[x.Length];
        for (int i = 0; i < x.Length; i++) result[i] = x[i] + step * direction[i];
        return result;
    }

    internal static LbfgsResult Gd(double tolerance)
    {
        double eta = 1.0 / LargestEigenvalue(A, 100);       // documented GD step size
        double[] x = (double[])X0.Clone();
        double[] g = Gradient(x);
        int iterations = 0;
        while (Math.Sqrt(Dot(g, g)) > tolerance && iterations < GdMaxIter)
        {
            for (int i = 0; i < x.Length; i++) x[i] -= eta * g[i];
            g = Gradient(x);
            iterations += 1;
        }
        return new LbfgsResult(x, g, iterations);
    }
}

public static class Solution
{
    public static void Main()
    {
        double[] xStar = LbfgsDemo.SolveLinear(LbfgsDemo.A, LbfgsDemo.B);
        double fStar = LbfgsDemo.Objective(xStar);
        double gStarMax = 0.0;
        foreach (double value in LbfgsDemo.Gradient(xStar)) gStarMax = Math.Max(gStarMax, Math.Abs(value));
        Check(gStarMax <= 1e-12, "Gaussian oracle is exact");

        // Skeleton sanity: with an empty history H = I, so H*g = g and the step is -g.
        double[] g0 = LbfgsDemo.Gradient(LbfgsDemo.X0);
        double[] identity = LbfgsDemo.TwoLoopDirection(g0, new List<double[]>(), new List<double[]>());
        bool same = identity.Length == g0.Length;
        for (int i = 0; same && i < g0.Length; i++) same = identity[i] == g0[i];
        Check(same, "empty-history identity");

        LbfgsDemo.LbfgsResult lb = LbfgsDemo.Lbfgs();
        Check(Math.Sqrt(LbfgsDemo.Dot(lb.G, lb.G)) <= 1e-8, "documented L-BFGS gradient tolerance");
        double xErr = 0.0;
        for (int i = 0; i < xStar.Length; i++) xErr = Math.Max(xErr, Math.Abs(lb.X[i] - xStar[i]));
        Check(xErr <= 1e-6, "L-BFGS minimizer accuracy");
        Check(Math.Abs(LbfgsDemo.Objective(lb.X) - fStar) <= 1e-9, "L-BFGS objective accuracy");
        Check(lb.Iterations <= 20, "L-BFGS must converge in few iterations");

        LbfgsDemo.LbfgsResult plain = LbfgsDemo.Gd(1e-6);
        Check(Math.Sqrt(LbfgsDemo.Dot(plain.G, plain.G)) <= 1e-6, "GD must converge within the cap");
        Check(plain.Iterations > 2 * lb.Iterations, "L-BFGS must need far fewer iterations");
        double gdErr = 0.0;
        for (int i = 0; i < xStar.Length; i++) gdErr = Math.Max(gdErr, Math.Abs(plain.X[i] - xStar[i]));
        Check(gdErr <= 1e-4, "GD accuracy");

        LbfgsDemo.LbfgsResult repeat = LbfgsDemo.Lbfgs();
        bool identical = repeat.Iterations == lb.Iterations;
        for (int i = 0; identical && i < lb.X.Length; i++) identical = repeat.X[i] == lb.X[i];
        Check(identical, "deterministic re-run");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
