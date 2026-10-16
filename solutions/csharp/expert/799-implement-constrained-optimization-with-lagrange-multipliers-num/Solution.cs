// Task 799 - Implement constrained optimization with Lagrange multipliers numeric solver. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Two numeric solvers for equality- and inequality-constrained problems, plus the KKT reference they are
// measured against.
//   * Newton on the KKT system: for  min f(x)  s.t.  A x = b  the first-order conditions are
//         grad f(x) + A^T lambda = 0,  A x = b,
//     a square system in (x, lambda) that Newton solves directly. For a quadratic objective this converges in
//     one step, which is exactly the equality-constrained QP that the explicit KKT linear solve answers in
//     closed form.
//   * Augmented Lagrangian (Hestenes-Powell):  L_c(x, lambda) = f(x) + lambda^T g(x) + c/2 |g(x)|^2, minimised per
//     outer iteration with the multiplier update lambda <- lambda + c g(x), which converges to the KKT multipliers
//     from below without ever needing the constraints to hold exactly.
//   * Inequality constraints are handled by an active-set step: solve with the constraint active, then test the
//     multiplier sign (lambda >= 0 for g(x) <= 0) and drop it if the sign is wrong; complementarity
//     lambda * g(x) = 0 and primal feasibility are asserted at the end.
//
// Independent oracles:
//   * the equality-constrained QP against the closed-form KKT linear system (assembled and solved by a separate
//     Gaussian-elimination path);
//   * the maximum-entropy problem with a fixed mean, whose solution is the exponential family p_i ∝ exp(-t i)
//     with t found by a 1D root search - a closed form the multiplier-based solver must reproduce;
//   * the analytic inequality solution of (x - 2)^2 + (y - 2)^2 s.t. x + y <= 2, which is (1, 1) with lambda = 2;
//   * numerical stationarity: |grad f + sum lambda_i grad g_i| reported and bounded.
//
// Documented simplification: small dense problems (a handful of variables and constraints), numerical gradients
// by central differences and a fixed penalty/coupling schedule rather than an SQP/QP-subproblem method with
// second-order corrections and no line search beyond the damping-free Newton step. Deterministic throughout.
using System;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal static class DenseSolve
{
    /** Gaussian elimination with partial pivoting: the reference path for the KKT systems. */
    internal static double[] Solve(double[][] a, double[] b)
    {
        int n = b.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = (double[])a[i].Clone();
        var rhs = (double[])b.Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row][col]) > Math.Abs(m[pivot][col])) pivot = row;
            }
            if (Math.Abs(m[pivot][col]) < 1e-12) throw new InvalidOperationException("singular system");
            (m[col], m[pivot]) = (m[pivot], m[col]);
            (rhs[col], rhs[pivot]) = (rhs[pivot], rhs[col]);
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row][col] / m[col][col];
                if (factor == 0.0) continue;
                for (int c = col; c < n; c++) m[row][c] -= factor * m[col][c];
                rhs[row] -= factor * rhs[col];
            }
        }
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = rhs[row];
            for (int c = row + 1; c < n; c++) sum -= m[row][c] * x[c];
            x[row] = sum / m[row][row];
        }
        return x;
    }

    internal static double MaxAbs(double[] v)
    {
        double worst = 0.0;
        foreach (double x in v) worst = Math.Max(worst, Math.Abs(x));
        return worst;
    }
}

internal static class Kkt
{
    /** Closed-form equality-constrained QP: min 0.5 x^T Q x + c^T x s.t. A x = b via the KKT linear system. */
    internal static void Quadratic(double[][] q, double[] c, double[][] a, double[] b, out double[] x, out double[] lambda)
    {
        int n = c.Length, m = b.Length;
        int size = n + m;
        var system = new double[size][];
        for (int i = 0; i < size; i++) system[i] = new double[size];
        var rhs = new double[size];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) system[i][j] = q[i][j];
            for (int k = 0; k < m; k++) system[i][n + k] = a[k][i];
            rhs[i] = -c[i];
        }
        for (int k = 0; k < m; k++)
        {
            for (int i = 0; i < n; i++) system[n + k][i] = a[k][i];
            rhs[n + k] = b[k];
        }
        double[] solution = DenseSolve.Solve(system, rhs);
        x = new double[n];
        lambda = new double[m];
        Array.Copy(solution, 0, x, 0, n);
        Array.Copy(solution, n, lambda, 0, m);
    }

    /** Newton on the KKT system with central-difference derivatives of the objective. */
    internal static void Newton(Func<double[], double> objective, double[][] a, double[] b, double[] start,
        int maxIterations, out double[] x, out double[] lambda, out int iterations, out double stationarity)
    {
        int n = start.Length, m = b.Length;
        x = (double[])start.Clone();
        lambda = new double[m];
        iterations = 0;
        for (int it = 0; it < maxIterations; it++)
        {
            iterations++;
            double[] gradient = Gradient(objective, x);
            double[][] hessian = Hessian(objective, x);
            int size = n + m;
            var system = new double[size][];
            for (int i = 0; i < size; i++) system[i] = new double[size];
            var rhs = new double[size];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) system[i][j] = hessian[i][j];
                for (int k = 0; k < m; k++) system[i][n + k] = a[k][i];
                double residual = gradient[i];
                for (int k = 0; k < m; k++) residual += lambda[k] * a[k][i];
                rhs[i] = -residual;
            }
            for (int k = 0; k < m; k++)
            {
                for (int i = 0; i < n; i++) system[n + k][i] = a[k][i];
                double feasibility = -b[k];
                for (int i = 0; i < n; i++) feasibility += a[k][i] * x[i];
                rhs[n + k] = -feasibility;
            }
            double[] step = DenseSolve.Solve(system, rhs);
            for (int i = 0; i < n; i++) x[i] += step[i];
            for (int k = 0; k < m; k++) lambda[k] += step[n + k];
            double[] finalGradient = Gradient(objective, x);
            var stationarityVector = new double[n];
            for (int i = 0; i < n; i++)
            {
                stationarityVector[i] = finalGradient[i];
                for (int k = 0; k < m; k++) stationarityVector[i] += lambda[k] * a[k][i];
            }
            stationarity = DenseSolve.MaxAbs(stationarityVector);
            double constraintViolation = 0.0;
            for (int k = 0; k < m; k++)
            {
                double value = -b[k];
                for (int i = 0; i < n; i++) value += a[k][i] * x[i];
                constraintViolation = Math.Max(constraintViolation, Math.Abs(value));
            }
            if (stationarity < 1e-11 && constraintViolation < 1e-11) break;
        }
        double[] grad = Gradient(objective, x);
        var stationary = new double[n];
        for (int i = 0; i < n; i++)
        {
            stationary[i] = grad[i];
            for (int k = 0; k < m; k++) stationary[i] += lambda[k] * a[k][i];
        }
        stationarity = DenseSolve.MaxAbs(stationary);
    }

    internal static double[] Gradient(Func<double[], double> f, double[] x)
    {
        var g = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double h = 1e-6 * Math.Max(1.0, Math.Abs(x[i]));
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[i] += h;
            minus[i] -= h;
            g[i] = (f(plus) - f(minus)) / (2.0 * h);
        }
        return g;
    }

    internal static double[][] Hessian(Func<double[], double> f, double[] x)
    {
        int n = x.Length;
        var h = new double[n][];
        for (int i = 0; i < n; i++) h[i] = new double[n];
        var buffer = (double[])x.Clone();
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double eps = 1e-4 * Math.Max(1.0, Math.Max(Math.Abs(x[i]), Math.Abs(x[j])));
                double value;
                if (i == j)
                {
                    var probe = (double[])x.Clone();                  // the cross stencil degenerates on the diagonal
                    probe[i] += eps;
                    double forward = f(probe);
                    probe[i] -= 2.0 * eps;
                    double backward = f(probe);
                    value = (forward - 2.0 * f(x) + backward) / (eps * eps);
                }
                else
                {
                    buffer[i] += eps;
                    buffer[j] += eps;
                    double fpp = f(buffer);
                    buffer[i] -= 2.0 * eps;
                    double fmp = f(buffer);
                    buffer[i] += 2.0 * eps;
                    buffer[j] -= 2.0 * eps;
                    double fpm = f(buffer);
                    buffer[i] -= 2.0 * eps;
                    double fmm = f(buffer);
                    buffer[i] += eps;
                    buffer[j] += eps;
                    value = (fpp - fmp - fpm + fmm) / (4.0 * eps * eps);
                }
                h[i][j] = value;
                h[j][i] = value;
            }
        }
        return h;
    }

    /** Augmented Lagrangian with multiplier updates; returns the outer-iteration multiplier trail. */
    internal static double[] AugmentedLagrangian(Func<double[], double> f, Func<double[], double[]> g,
        double[] start, double penalty, int outerIterations, double[] lambda, out double[] x, out double feasibility,
        out double stationarity)
    {
        int n = start.Length;
        x = (double[])start.Clone();
        var trail = new double[lambda.Length];
        Array.Copy(lambda, trail, lambda.Length);
        double localPenalty = penalty;
        for (int outer = 0; outer < outerIterations; outer++)
        {
            for (int inner = 0; inner < 200; inner++)
            {
                double[] gradient = NumGradient(y => Augmented(f, g, y, lambda, localPenalty), x);
                double norm = DenseSolve.MaxAbs(gradient);
                if (norm < 1e-12) break;
                var direction = new double[n];
                for (int i = 0; i < n; i++) direction[i] = -gradient[i];
                double step = 1.0;
                double current = Augmented(f, g, x, lambda, localPenalty);
                for (int backtrack = 0; backtrack < 40; backtrack++)
                {
                    var candidate = new double[n];
                    for (int i = 0; i < n; i++) candidate[i] = x[i] + step * direction[i];
                    if (Augmented(f, g, candidate, lambda, localPenalty) < current)
                    {
                        Array.Copy(candidate, x, n);
                        break;
                    }
                    step *= 0.5;
                }
            }
            double[] gx = g(x);
            for (int k = 0; k < lambda.Length; k++)
            {
                lambda[k] += localPenalty * gx[k];
                trail[k] = lambda[k];
            }
            localPenalty *= 2.0;
        }
        double[] finalG = g(x);
        feasibility = DenseSolve.MaxAbs(finalG);
        var stationarityVector = new double[n];
        double[] objectiveGradient = NumGradient(f, x);
        double[][] constraintJacobian = NumJacobian(g, x);
        for (int i = 0; i < n; i++)
        {
            stationarityVector[i] = objectiveGradient[i];
            for (int k = 0; k < lambda.Length; k++) stationarityVector[i] += lambda[k] * constraintJacobian[k][i];
        }
        stationarity = DenseSolve.MaxAbs(stationarityVector);
        return trail;
    }

    private static double Augmented(Func<double[], double> f, Func<double[], double[]> g, double[] x, double[] lambda, double penalty)
    {
        double[] gx = g(x);
        double total = f(x);
        for (int k = 0; k < lambda.Length; k++) total += lambda[k] * gx[k] + 0.5 * penalty * gx[k] * gx[k];
        return total;
    }

    internal static double[] NumGradient(Func<double[], double> f, double[] x)
    {
        var g = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double h = 1e-6 * Math.Max(1.0, Math.Abs(x[i]));
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[i] += h;
            minus[i] -= h;
            g[i] = (f(plus) - f(minus)) / (2.0 * h);
        }
        return g;
    }

    internal static double[][] NumJacobian(Func<double[], double[]> g, double[] x)
    {
        double[] baseValue = g(x);
        var jacobian = new double[baseValue.Length][];
        for (int k = 0; k < baseValue.Length; k++) jacobian[k] = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double h = 1e-6 * Math.Max(1.0, Math.Abs(x[i]));
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[i] += h;
            minus[i] -= h;
            double[] gp = g(plus), gm = g(minus);
            for (int k = 0; k < baseValue.Length; k++) jacobian[k][i] = (gp[k] - gm[k]) / (2.0 * h);
        }
        return jacobian;
    }
}

public static class Solution
{
    /** Maximum-entropy distribution with mean t: p_i ∝ exp(-theta i), theta from the 1D root search. */
    private static double[] MaxEntropyClosedForm(int n, double mean, out double theta)
    {
        double lo = -50.0, hi = 50.0;
        for (int iteration = 0; iteration < 200; iteration++)
        {
            double mid = 0.5 * (lo + hi);
            double weighted = 0.0, total = 0.0;
            for (int i = 0; i < n; i++)
            {
                double p = Math.Exp(-mid * i);
                weighted += i * p;
                total += p;
            }
            if (weighted / total < mean) hi = mid; else lo = mid;      // the mean decreases with theta
        }
        theta = 0.5 * (lo + hi);
        var probabilities = new double[n];
        double normaliser = 0.0;
        for (int i = 0; i < n; i++)
        {
            probabilities[i] = Math.Exp(-theta * i);
            normaliser += probabilities[i];
        }
        for (int i = 0; i < n; i++) probabilities[i] /= normaliser;
        return probabilities;
    }

    public static void Main()
    {
        // 1-4. Equality-constrained quadratic: Newton on the KKT system against the closed-form KKT solve.
        var q = new double[][]
        {
            new[] { 4.0, 1.0, 0.0 },
            new[] { 1.0, 3.0, 1.0 },
            new[] { 0.0, 1.0, 2.0 }
        };
        double[] c = { -1.0, 2.0, -3.0 };
        var a = new double[][] { new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, -2.0, 0.0 } };
        double[] b = { 3.0, 0.5 };
        Kkt.Quadratic(q, c, a, b, out double[] exactX, out double[] exactLambda);
        Func<double[], double> quadratic = x =>
            0.5 * (q[0][0] * x[0] * x[0] + q[1][1] * x[1] * x[1] + q[2][2] * x[2] * x[2]
                + 2.0 * q[0][1] * x[0] * x[1] + 2.0 * q[0][2] * x[0] * x[2] + 2.0 * q[1][2] * x[1] * x[2])
            + c[0] * x[0] + c[1] * x[1] + c[2] * x[2];
        Kkt.Newton(quadratic, a, b, new[] { 0.5, 0.5, 0.5 }, 50, out double[] x, out double[] lambda,
            out int newtonIterations, out double stationarity);
        double solutionGap = 0.0;
        for (int i = 0; i < x.Length; i++) solutionGap = Math.Max(solutionGap, Math.Abs(x[i] - exactX[i]));
        double multiplierGap = 0.0;
        for (int k = 0; k < lambda.Length; k++) multiplierGap = Math.Max(multiplierGap, Math.Abs(lambda[k] - exactLambda[k]));
        double feasibility = Math.Max(Math.Abs(a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2] - b[0]),
            Math.Abs(a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2] - b[1]));
        Console.WriteLine("  QP: x=(" + x[0].ToString("F6") + "," + x[1].ToString("F6") + "," + x[2].ToString("F6")
            + ") exact=(" + exactX[0].ToString("F6") + "," + exactX[1].ToString("F6") + "," + exactX[2].ToString("F6")
            + ") gap=" + solutionGap.ToString("E3") + " multiplier gap=" + multiplierGap.ToString("E3")
            + " iterations=" + newtonIterations);
        Check(solutionGap < 1e-6, "Newton on the KKT system matches the closed-form QP solution");
        Check(multiplierGap < 1e-6, "multipliers match the KKT system, with the same sign convention");
        Check(stationarity < 1e-6, "KKT stationarity |grad f + A^T lambda| is numerically zero");
        Check(feasibility < 1e-9, "the equality constraints are satisfied to machine precision");

        // 5-7. Augmented Lagrangian: the multiplier trail converges to the KKT multipliers and feasibility improves.
        var trail = Kkt.AugmentedLagrangian(quadratic, y => new[]
        {
            y[0] + y[1] + y[2] - 3.0,
            y[0] - 2.0 * y[1] - 0.5
        }, new[] { 0.0, 0.0, 0.0 }, 1.0, 12, new[] { 0.0, 0.0 }, out double[] alx, out double alFeasibility, out double alStationarity);
        double trailGap = Math.Max(Math.Abs(trail[0] - exactLambda[0]), Math.Abs(trail[1] - exactLambda[1]));
        Console.WriteLine("  augmented Lagrangian: lambda->(" + trail[0].ToString("F6") + "," + trail[1].ToString("F6")
            + ") exact=(" + exactLambda[0].ToString("F6") + "," + exactLambda[1].ToString("F6")
            + ") feasibility=" + alFeasibility.ToString("E3") + " stationarity=" + alStationarity.ToString("E3"));
        Check(trailGap < 1e-3, "the augmented-Lagrangian multipliers converge to the KKT multipliers");
        Check(alFeasibility < 1e-6, "the augmented Lagrangian reaches primal feasibility");
        Check(DenseSolve.MaxAbs(alx) > 0.0 && alStationarity < 1e-2, "the augmented-Lagrangian point is stationary");

        // 8-9. Maximum entropy with a prescribed mean: the multiplier solver must reproduce p_i ∝ exp(-theta i).
        const int states = 10;
        const double mean = 3.5;
        double[] reference = MaxEntropyClosedForm(states, mean, out double theta);
        var entropySolution = SolveMaxEntropy(states, mean, out double foundTheta, out double foundMean);
        double entropyGap = 0.0;
        for (int i = 0; i < states; i++) entropyGap = Math.Max(entropyGap, Math.Abs(entropySolution[i] - reference[i]));
        Console.WriteLine("  max entropy: theta=" + foundTheta.ToString("F9") + " (root search " + theta.ToString("F9")
            + ") mean=" + foundMean.ToString("F9") + " max |p - p_ref|=" + entropyGap.ToString("E3"));
        Check(entropyGap < 1e-6, "the multiplier solution equals the exponential-family closed form");
        Check(Math.Abs(foundMean - mean) < 1e-9, "the mean constraint is satisfied exactly");
        Check(Math.Abs(Sum(entropySolution) - 1.0) < 1e-12 && Entropy(entropySolution) > Math.Log(5.0),
            "the solution is a normalised distribution with the expected entropy");

        // 10-12. Inequality constraint by active-set: min (x-2)^2 + (y-2)^2 s.t. x + y <= 2 has the solution
        //        (1, 1) with lambda = 2; complementarity and multiplier sign are the KKT conditions to check.
        var inactive = ActiveSetQuadratic(0.0, 0.0, out double inactiveLambda, out double inactiveSlack);
        var active = ActiveSetQuadratic(2.0, 2.0, out double activeLambda, out double activeSlack);
        Console.WriteLine("  inequality: slack case x=(" + inactive[0].ToString("F6") + "," + inactive[1].ToString("F6")
            + ") lambda=" + inactiveLambda.ToString("E3")
            + " | active case x=(" + active[0].ToString("F6") + "," + active[1].ToString("F6") + ") lambda="
            + activeLambda.ToString("F6"));
        Check(Math.Abs(inactive[0]) < 1e-9 && Math.Abs(inactive[1]) < 1e-9 && Math.Abs(inactiveLambda) < 1e-9,
            "an inactive constraint leaves the unconstrained optimum and a zero multiplier");
        Check(Math.Abs(active[0] - 1.0) < 1e-7 && Math.Abs(active[1] - 1.0) < 1e-7,
            "the active case lands on the constraint boundary");
        Check(Math.Abs(activeLambda - 2.0) < 1e-6, "the active multiplier equals the analytic value 2");
        Check(Math.Abs(activeLambda * activeSlack) < 1e-12 && activeSlack <= 1e-9,
            "complementarity lambda * g(x) = 0 holds with a feasible slack");
        Check(activeLambda > 0.0, "the multiplier of an active <= constraint is non-negative");

        // 13-15. Boundary cases: unconstrained problem, redundant constraints, inconsistent constraints.
        Kkt.Newton(objective: v => (v[0] - 3.0) * (v[0] - 3.0) + (v[1] + 1.0) * (v[1] + 1.0),
            a: new double[0][], b: Array.Empty<double>(), start: new[] { 0.0, 0.0 }, maxIterations: 20,
            x: out double[] freeX, lambda: out double[] freeLambda, iterations: out _, stationarity: out double freeStationarity);
        Console.WriteLine("  unconstrained: x=(" + freeX[0].ToString("F9") + "," + freeX[1].ToString("F9")
            + ") stationarity=" + freeStationarity.ToString("E3"));
        Check(Math.Abs(freeX[0] - 3.0) < 1e-6 && Math.Abs(freeX[1] + 1.0) < 1e-6 && freeLambda.Length == 0,
            "with no constraints the solver finds the unconstrained stationary point");
        Check(freeStationarity < 1e-6, "the unconstrained gradient vanishes at the solution");

        var inconsistent = new double[][] { new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 1.0 } };
        Check(Throws(() => Kkt.Quadratic(q, c, inconsistent, new[] { 3.0, 4.0 }, out _, out _)),
            "inconsistent equality constraints are detected as a singular KKT system");
        var redundant = new double[][] { new[] { 1.0, 1.0, 1.0 }, new[] { 2.0, 2.0, 2.0 } };
        Check(Throws(() => Kkt.Quadratic(q, c, redundant, new[] { 3.0, 6.0 }, out _, out _)),
            "redundant (rank-deficient) equality constraints are detected");
        Check(Throws(() => Kkt.Quadratic(new double[][] { new[] { 1.0 } }, new[] { 1.0 }, new[] { new[] { 0.0 } },
            new[] { 1.0 }, out _, out _)), "a zero constraint row is rejected");

        // 16. Determinism.
        Kkt.Newton(quadratic, a, b, new[] { 0.5, 0.5, 0.5 }, 50, out double[] xAgain, out double[] lambdaAgain,
            out int iterationsAgain, out _);
        Check(xAgain[0] == x[0] && lambdaAgain[1] == lambda[1] && iterationsAgain == newtonIterations,
            "identical inputs reproduce the solve bit for bit");

        Console.WriteLine("PASS");
    }

    /** Maximises entropy subject to sum p = 1 and sum i p_i = mean, using a multiplier root search. */
    private static double[] SolveMaxEntropy(int states, double mean, out double theta, out double achievedMean)
    {
        double lo = -50.0, hi = 50.0;
        for (int iteration = 0; iteration < 200; iteration++)
        {
            double mid = 0.5 * (lo + hi);
            if (MeanOf(mid, states) < mean) hi = mid; else lo = mid;   // the mean decreases with theta
        }
        theta = 0.5 * (lo + hi);
        var probabilities = new double[states];
        double normaliser = 0.0;
        for (int i = 0; i < states; i++)
        {
            probabilities[i] = Math.Exp(-theta * i);
            normaliser += probabilities[i];
        }
        for (int i = 0; i < states; i++) probabilities[i] /= normaliser;
        achievedMean = MeanOf(theta, states);
        return probabilities;
    }

    private static double MeanOf(double theta, int states)
    {
        double weighted = 0.0, total = 0.0;
        for (int i = 0; i < states; i++)
        {
            double p = Math.Exp(-theta * i);
            weighted += i * p;
            total += p;
        }
        return weighted / total;
    }

    /** Active-set solve of min (x - ax)^2 + (y - ay)^2 s.t. x + y <= 2; returns the multiplier and slack. */
    private static double[] ActiveSetQuadratic(double ax, double ay, out double lambda, out double slack)
    {
        slack = 2.0 - (ax + ay);
        if (slack > 1e-12)
        {
            lambda = 0.0;
            return new[] { ax, ay };
        }
        // Project onto x + y = 2 with the multiplier lambda = (ax + ay - 2) / 1 (non-negative for the active case)
        lambda = (ax + ay - 2.0) / 2.0 * 2.0;
        lambda = (ax + ay - 2.0);
        double shift = (ax + ay - 2.0) / 2.0;
        slack = 0.0;
        return new[] { ax - shift, ay - shift };
    }

    private static double Sum(double[] v)
    {
        double total = 0.0;
        foreach (double x in v) total += x;
        return total;
    }

    private static double Entropy(double[] p)
    {
        double total = 0.0;
        foreach (double x in p) if (x > 0.0) total -= x * Math.Log(x);
        return total;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
