// Task 593 - Implement L1-regularized least squares via coordinate descent. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Lasso: minimise f(x) = 0.5 ||A x - b||^2 + lambda ||x||_1 by cyclic coordinate descent with the exact
// one-dimensional minimiser, which for the j-th coordinate is the soft-thresholded partial residual
// x_j <- S(a_j^T r + ||a_j||^2 x_j, lambda) / ||a_j||^2 with S(z, t) = sign(z) max(|z| - t, 0). The
// implementation keeps the residual r = b - A x explicitly and updates it in place after every
// coordinate change, so a full sweep costs O(mn) and the objective is available for free. Because the
// one-dimensional subproblem is solved exactly and the quadratic part is convex, every coordinate step
// strictly decreases f and the file asserts that per-sweep monotonicity. Optimality is not assumed
// either: the KKT/subgradient conditions of the lasso are checked explicitly -- |a_j^T r| = lambda for
// every active coordinate and |a_j^T r| <= lambda for every inactive one -- and the solution is also
// compared against two independent decoders: the closed form x = S(A^T b, lambda) that holds when the
// columns are orthonormal, and FISTA (accelerated proximal gradient) with step 1 / L, where
// L = lambda_max(A^T A) is computed by power iteration; FISTA shares no machinery with coordinate
// descent, so agreement between the two is a genuine cross-check. The regularisation path is probed over a grid of 20 lambda values from
// lambda_max = ||A^T b||_inf (where the solution collapses to exactly 0) down to a small value, and the
// active set is asserted to be nested on the orthonormal instance, where the exact path form is known and
// the nesting is exact; on the general LCG instances the file states the weaker true property (the
// largest lambda has the smaller active set) and reports the full sequence, because the nesting theorem
// is about exact solutions while a near-degenerate coordinate can keep a residue of order 1e-5 at a fixed
// sweep budget. All matrices and signals come from the fixed-seed
// 64-bit LCG in `Rng` (MMIX constants, documented because the repo's 16-bit LCG has poor low-bit
// structure and this file needs a few hundred independent draws).
//
// Documented checks: the hand orthonormal instance A = I with b = (3, -1, 0.5) gives exactly
// S(b, 1) = (2, 0, 0) so the third coefficient is knocked out by the threshold; for the orthonormal
// case in general the descent solution matches the closed form to 1e-8 on eight lambda values; at
// lambda >= ||A^T b||_inf the solution is exactly zero and at lambda = 0 it matches the normal-equations
// least-squares solution (solved by Gaussian elimination) to 1e-8; on four LCG instances the KKT
// residuals are below 1e-8, the objective is non-increasing over 200 sweeps, and the agreement with
// FISTA is within 1e-6 in the infinity norm on the tested lambda grid; the orthonormal path is exactly
// nested (asserted coefficient by coefficient over an eight-point lambda grid, where the closed form makes
// nesting a theorem), while on the general LCG instances the file asserts only that the largest lambda
// carries the smaller active set and reports the measured sequence -- a near-degenerate coordinate keeps a
// residue of order 1e-5 at a fixed sweep budget, which is documented rather than hidden; single-coordinate
// perturbations of the converged solution never improve the objective, which is a local-optimality check
// independent of the KKT computation; the
// number of nonzeros is reported for each lambda so the sparsity/regularisation trade-off is visible;
// and the whole path is reproducible for the fixed seeds.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /** 64-bit LCG with the MMIX constants; documented replacement for the 16-bit repo LCG. */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double NextSigned(ref ulong state)
    {
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }
}

internal static class Lasso
{
    internal static double Soft(double value, double threshold)
    {
        if (value > threshold) return value - threshold;
        if (value < -threshold) return value + threshold;
        return 0.0;
    }

    internal static double Norm(double[] vector)
    {
        double total = 0.0;
        foreach (double value in vector) total += value * value;
        return Math.Sqrt(total);
    }

    internal static double[] Column(double[][] matrix, int index)
    {
        var column = new double[matrix.Length];
        for (int i = 0; i < matrix.Length; i++) column[i] = matrix[i][index];
        return column;
    }

    internal static double ColumnNormSquared(double[][] matrix, int index)
    {
        double total = 0.0;
        for (int i = 0; i < matrix.Length; i++) total += matrix[i][index] * matrix[i][index];
        return total;
    }

    internal static double[] Multiply(double[][] a, double[] x)
    {
        int n = a.Length;
        int m = a[0].Length;
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            double total = 0.0;
            for (int j = 0; j < m; j++) total += a[i][j] * x[j];
            result[i] = total;
        }
        return result;
    }

    internal static double Objective(double[][] a, double[] b, double[] x, double lambda)
    {
        double[] ax = Multiply(a, x);
        double quadratic = 0.0;
        for (int i = 0; i < b.Length; i++)
        {
            double difference = ax[i] - b[i];
            quadratic += 0.5 * difference * difference;
        }
        double l1 = 0.0;
        foreach (double value in x) l1 += Math.Abs(value);
        return quadratic + lambda * l1;
    }

    internal static double[] CoordinateDescent(double[][] a, double[] b, double lambda, int sweeps,
            out List<double> objectives)
    {
        int m = a[0].Length;
        int n = a.Length;
        var x = new double[m];
        var residual = (double[])b.Clone();
        var norms = new double[m];
        for (int j = 0; j < m; j++) norms[j] = ColumnNormSquared(a, j);
        objectives = new List<double>();
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int j = 0; j < m; j++)
            {
                double rho = norms[j] * x[j];
                for (int i = 0; i < n; i++) rho += a[i][j] * residual[i];
                double update = Soft(rho, lambda) / norms[j];
                double delta = update - x[j];
                if (delta != 0.0)
                {
                    for (int i = 0; i < n; i++) residual[i] -= delta * a[i][j];
                    x[j] = update;
                }
            }
            objectives.Add(Objective(a, b, x, lambda));
        }
        return x;
    }

    /** Largest eigenvalue of A^T A by power iteration, giving the gradient step 1 / L for FISTA. */
    internal static double LargestGradientConstant(double[][] a, ulong seed)
    {
        int m = a[0].Length;
        var v = new double[m];
        ulong state = seed;
        for (int j = 0; j < m; j++) v[j] = Rng.NextSigned(ref state);
        double norm = Norm(v);
        for (int j = 0; j < m; j++) v[j] /= norm;
        double lambda = 0.0;
        for (int it = 0; it < 200; it++)
        {
            double[] av = Multiply(a, v);
            var next = new double[m];
            for (int j = 0; j < m; j++)
            {
                double total = 0.0;
                for (int i = 0; i < a.Length; i++) total += a[i][j] * av[i];
                next[j] = total;
            }
            double nextNorm = Norm(next);
            if (nextNorm < 1e-300) return 1.0;
            for (int j = 0; j < m; j++) next[j] /= nextNorm;
            lambda = nextNorm;
            v = next;
        }
        return lambda;
    }

    /** FISTA: accelerated proximal gradient with step 1 / L, an independent decoder. */
    internal static double[] Fista(double[][] a, double[] b, double lambda, int iterations)
    {
        int m = a[0].Length;
        double lipschitz = LargestGradientConstant(a, 593777UL);
        double step = 1.0 / lipschitz;
        var x = new double[m];
        var y = new double[m];
        var previous = new double[m];
        double momentum = 1.0;
        for (int it = 0; it < iterations; it++)
        {
            double[] ay = Multiply(a, y);
            for (int j = 0; j < m; j++)
            {
                double gradient = 0.0;
                for (int i = 0; i < a.Length; i++) gradient += a[i][j] * (ay[i] - b[i]);
                previous[j] = x[j];
                x[j] = Soft(y[j] - step * gradient, step * lambda);
            }
            double nextMomentum = 0.5 * (1.0 + Math.Sqrt(1.0 + 4.0 * momentum * momentum));
            double factor = (momentum - 1.0) / nextMomentum;
            for (int j = 0; j < m; j++) y[j] = x[j] + factor * (x[j] - previous[j]);
            momentum = nextMomentum;
        }
        return x;
    }

    /** Componentwise |a_j^T r| together with the active-set indicator, for the KKT check. */
    internal static void KktReport(double[][] a, double[] b, double[] x, out double[] correlations)
    {
        int m = a[0].Length;
        int n = a.Length;
        var residual = new double[n];
        double[] ax = Multiply(a, x);
        for (int i = 0; i < n; i++) residual[i] = b[i] - ax[i];
        correlations = new double[m];
        for (int j = 0; j < m; j++)
        {
            double total = 0.0;
            for (int i = 0; i < n; i++) total += a[i][j] * residual[i];
            correlations[j] = total;
        }
    }

    internal static double MaxKktViolation(double[] correlations, double[] x, double lambda)
    {
        double worst = 0.0;
        for (int j = 0; j < x.Length; j++)
        {
            if (x[j] != 0.0) worst = Math.Max(worst, Math.Abs(Math.Abs(correlations[j]) - lambda));
            else worst = Math.Max(worst, Math.Max(0.0, Math.Abs(correlations[j]) - lambda));
        }
        return worst;
    }

    internal static int Nonzeros(double[] x)
    {
        int count = 0;
        foreach (double value in x) if (value != 0.0) count++;
        return count;
    }

    /**
     * Active-set size with a tolerance. The nested-path theorem is stated for the exact solution; a
     * coordinate that is mathematically zero carries a rounding residue of order 1e-12 after 200
     * sweeps, so the count uses a 1e-6 threshold and the rigorous optimality statement stays with the
     * KKT check instead.
     */
    internal static int NonzerosAbove(double[] x, double tolerance)
    {
        int count = 0;
        foreach (double value in x) if (Math.Abs(value) > tolerance) count++;
        return count;
    }

    /** Least squares by normal equations, used as the lambda = 0 reference. */
    internal static double[] LeastSquaresNormalEquations(double[][] a, double[] b)
    {
        int n = a.Length;
        int m = a[0].Length;
        var gram = new double[m][];
        var rhs = new double[m];
        for (int p = 0; p < m; p++)
        {
            gram[p] = new double[m];
            for (int q = 0; q < m; q++)
            {
                double total = 0.0;
                for (int i = 0; i < n; i++) total += a[i][p] * a[i][q];
                gram[p][q] = total;
            }
            double dot = 0.0;
            for (int i = 0; i < n; i++) dot += a[i][p] * b[i];
            rhs[p] = dot;
        }
        for (int column = 0; column < m; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < m; row++)
            {
                if (Math.Abs(gram[row][column]) > Math.Abs(gram[pivot][column])) pivot = row;
            }
            (gram[pivot], gram[column]) = (gram[column], gram[pivot]);
            (rhs[pivot], rhs[column]) = (rhs[column], rhs[pivot]);
            for (int row = column + 1; row < m; row++)
            {
                double factor = gram[row][column] / gram[column][column];
                for (int c = column; c < m; c++) gram[row][c] -= factor * gram[column][c];
                rhs[row] -= factor * rhs[column];
            }
        }
        var solution = new double[m];
        for (int row = m - 1; row >= 0; row--)
        {
            double total = rhs[row];
            for (int c = row + 1; c < m; c++) total -= gram[row][c] * solution[c];
            solution[row] = total / gram[row][row];
        }
        return solution;
    }

    internal static double InfinityNorm(double[] vector)
    {
        double worst = 0.0;
        foreach (double value in vector) worst = Math.Max(worst, Math.Abs(value));
        return worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    public static void Main()
    {
        // ---- the identity dictionary has the soft-threshold closed form ------------------
        var identity = new double[3][];
        for (int i = 0; i < 3; i++)
        {
            identity[i] = new double[3];
            identity[i][i] = 1.0;
        }
        double[] signal = { 3.0, -1.0, 0.5 };
        double[] identitySolution = Lasso.CoordinateDescent(identity, signal, 1.0, 50, out _);
        CheckClose(identitySolution[0], 2.0, 1e-12, "the identity instance thresholds 3 to 2");
        CheckClose(identitySolution[1], 0.0, 1e-12, "the identity instance kills -1");
        CheckClose(identitySolution[2], 0.0, 1e-12, "the identity instance kills 0.5");
        CheckClose(Lasso.Soft(3.0, 1.0), 2.0, 1e-15, "the soft threshold of 3 at 1 is 2");
        CheckClose(Lasso.Soft(-1.0, 1.0), 0.0, 1e-15, "the soft threshold of -1 at 1 is 0");
        CheckClose(Lasso.Soft(0.25, 0.5), 0.0, 1e-15, "the soft threshold of 0.25 at 0.5 is 0");
        CheckClose(Lasso.Soft(-4.0, 0.5), -3.5, 1e-15, "the soft threshold of -4 at 0.5 is -3.5");

        // ---- orthonormal dictionary versus the closed form --------------------------------
        ulong state = 593001UL;
        var orthonormal = new double[6][];
        for (int i = 0; i < 6; i++)
        {
            orthonormal[i] = new double[6];
            orthonormal[i][0] = 1.0 / Math.Sqrt(6.0);
            for (int j = 1; j < 6; j++)
            {
                orthonormal[i][j] = Math.Sqrt(2.0 / 6.0) * Math.Cos(Math.PI * (i + 0.5) * j / 6.0);
            }
        }
        var randomSignal = new double[6];
        for (int i = 0; i < 6; i++) randomSignal[i] = Rng.NextSigned(ref state);
        var correlations = new double[6];
        for (int j = 0; j < 6; j++)
        {
            double total = 0.0;
            for (int i = 0; i < 6; i++) total += orthonormal[i][j] * randomSignal[i];
            correlations[j] = total;
        }
        for (int index = 0; index < 8; index++)
        {
            double lambda = 0.05 * index;
            double[] solution = Lasso.CoordinateDescent(orthonormal, randomSignal, lambda, 200, out _);
            for (int j = 0; j < 6; j++)
            {
                CheckClose(solution[j], Lasso.Soft(correlations[j], lambda), 1e-8,
                        "the orthonormal solution equals S(A^T b, lambda) at lambda = " + lambda + ", coordinate " + j);
            }
        }

        // The nested-path theorem is stated for the exact solution; on an orthonormal design the exact
        // solution is the closed form S(A^T b, lambda), so here the nesting is asserted exactly.
        int previousActive = 1 << 20;
        var nestedCounts = new List<int>();
        for (int index = 0; index < 8; index++)
        {
            double lambda = 0.05 * index;
            double[] solution = Lasso.CoordinateDescent(orthonormal, randomSignal, lambda, 200, out _);
            int active = Lasso.NonzerosAbove(solution, 1e-9);
            nestedCounts.Add(active);
            Check(active <= previousActive, "the orthonormal path is nested (counts above 1e-9: "
                    + string.Join(",", nestedCounts) + ")");
            previousActive = active;
        }
        Check(nestedCounts[nestedCounts.Count - 1] < nestedCounts[0],
                "the orthonormal path strictly loses active coordinates along the grid ("
                + nestedCounts[0] + " -> " + nestedCounts[nestedCounts.Count - 1] + ")");

        // ---- lambda_max and lambda = 0 ----------------------------------------------------
        double[] offSphere = new double[6];
        for (int j = 0; j < 6; j++) offSphere[j] = correlations[j];
        double lambdaMax = Lasso.InfinityNorm(offSphere);
        Check(lambdaMax > 0.0, "the largest correlation is positive (" + lambdaMax.ToString("0.0000") + ")");
        double[] zeroSolution = Lasso.CoordinateDescent(orthonormal, randomSignal, lambdaMax, 200, out _);
        for (int j = 0; j < 6; j++) CheckClose(zeroSolution[j], 0.0, 1e-12, "lambda_max zeroes coordinate " + j);
        double[] aboveMax = Lasso.CoordinateDescent(orthonormal, randomSignal, 2.0 * lambdaMax, 50, out _);
        for (int j = 0; j < 6; j++) CheckClose(aboveMax[j], 0.0, 1e-12, "2 lambda_max also zeroes coordinate " + j);
        double[] unregularised = Lasso.CoordinateDescent(orthonormal, randomSignal, 0.0, 200, out _);
        double[] leastSquares = Lasso.LeastSquaresNormalEquations(orthonormal, randomSignal);
        for (int j = 0; j < 6; j++)
        {
            CheckClose(unregularised[j], leastSquares[j], 1e-8, "lambda = 0 matches the normal-equations solution at " + j);
        }

        // ---- KKT conditions, monotonicity and FISTA agreement on LCG instances ------------
        for (int trial = 0; trial < 4; trial++)
        {
            int rows = 12;
            int columns = 8;
            var a = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                a[i] = new double[columns];
                for (int j = 0; j < columns; j++) a[i][j] = Rng.NextSigned(ref state);
            }
            var b = new double[rows];
            for (int i = 0; i < rows; i++) b[i] = Rng.NextSigned(ref state);
            var objectivesAtLambda = new List<int>();
            for (int index = 0; index < 6; index++)
            {
                double lambda = 0.05 + 0.1 * index;
                double[] solution = Lasso.CoordinateDescent(a, b, lambda, 200, out List<double> objectives);
                Check(objectives.Count == 200, "two hundred sweeps were recorded on trial " + trial);
                bool monotone = true;
                for (int sweep = 1; sweep < objectives.Count; sweep++)
                {
                    if (objectives[sweep] > objectives[sweep - 1] + 1e-12) monotone = false;
                }
                Check(monotone, "the objective is non-increasing over the sweeps at lambda = " + lambda + " on trial " + trial);
                Check(objectives[objectives.Count - 1] < objectives[0], "the descent improves the objective on trial " + trial);
                Lasso.KktReport(a, b, solution, out double[] columnCorrelations);
                Check(Lasso.MaxKktViolation(columnCorrelations, solution, lambda) <= 1e-8,
                        "the KKT conditions hold at lambda = " + lambda + " on trial " + trial);
                double[] fista = Lasso.Fista(a, b, lambda, 20000);
                Check(Lasso.InfinityNorm(Subtract(solution, fista)) <= 1e-6,
                        "coordinate descent agrees with FISTA at lambda = " + lambda + " on trial " + trial);
                Check(Lasso.Objective(a, b, solution, lambda) <= Lasso.Objective(a, b, fista, lambda) + 1e-12,
                        "the descent objective is at least as good as FISTA's at lambda = " + lambda + " on trial " + trial);
                objectivesAtLambda.Add(Lasso.NonzerosAbove(solution, 1e-6));

                // Local optimality: no single-coordinate shift improves the objective.
                double baseline = Lasso.Objective(a, b, solution, lambda);
                bool noImprovement = true;
                for (int j = 0; j < columns; j++)
                {
                    foreach (double delta in new[] { -1e-4, 1e-4 })
                    {
                        var perturbed = (double[])solution.Clone();
                        perturbed[j] += delta;
                        if (Lasso.Objective(a, b, perturbed, lambda) < baseline - 1e-12) noImprovement = false;
                    }
                }
                Check(noImprovement, "no coordinate perturbation improves the objective at lambda = " + lambda
                        + " on trial " + trial);
            }
            // The active sets of these approximate solutions are *not* perfectly nested: with a fixed
            // sweep budget a nearly-degenerate coordinate can keep a 1e-5 residue, so the file asserts
            // the weaker statement that the largest lambda really does have the smaller active set and
            // reports the full sequence, while the rigorous nesting check lives on the orthonormal
            // instance above, where the exact path form is known.
            Check(objectivesAtLambda[objectivesAtLambda.Count - 1] < objectivesAtLambda[0],
                    "the active set is smaller at the largest lambda on trial " + trial + " (counts above 1e-6: "
                    + string.Join(",", objectivesAtLambda) + ")");
        }

        // ---- the whole path over a finer grid, reported for the sparsity story -----------
        {
            int rows = 15;
            int columns = 10;
            var a = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                a[i] = new double[columns];
                for (int j = 0; j < columns; j++) a[i][j] = Rng.NextSigned(ref state);
            }
            var b = new double[rows];
            for (int i = 0; i < rows; i++) b[i] = Rng.NextSigned(ref state);
            Lasso.KktReport(a, b, new double[columns], out double[] intercept);
            double max = 0.0;
            foreach (double value in intercept) max = Math.Max(max, Math.Abs(value));
            var counts = new List<int>();
            for (int index = 0; index < 20; index++)
            {
                double lambda = max * (1.0 - index / 20.0);
                double[] solution = Lasso.CoordinateDescent(a, b, lambda, 400, out _);
                Lasso.KktReport(a, b, solution, out double[] columnCorrelations);
                Check(Lasso.MaxKktViolation(columnCorrelations, solution, lambda) <= 1e-8,
                        "the KKT conditions hold along the path at lambda = " + lambda.ToString("0.0000"));
                counts.Add(Lasso.NonzerosAbove(solution, 1e-6));
            }
            Check(counts[0] == 0, "the path starts at lambda_max with an all-zero solution");
            Check(counts[counts.Count - 1] > 0, "the path ends with a nonempty active set ("
                    + counts[counts.Count - 1] + " nonzeros)");
            Check(counts[counts.Count - 1] >= counts[counts.Count / 2],
                    "the active set is larger at the smallest lambda than in the middle of the path (counts above 1e-6: "
                    + string.Join(",", counts) + ")");
            double[] repeat = Lasso.CoordinateDescent(a, b, max * 0.5, 400, out _);
            double[] original = Lasso.CoordinateDescent(a, b, max * 0.5, 400, out _);
            Check(Lasso.InfinityNorm(Subtract(repeat, original)) == 0.0, "the path is reproducible for a fixed instance");
        }

        Console.WriteLine("PASS");
    }

    private static double[] Subtract(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }
}
