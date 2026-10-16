// Task 788 - Implement stiff ODE solver L-stable method concept. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// A stiff problem is one whose Jacobian has widely separated time scales (here lambda = -1e3 next to a slow
// mode near -1). Explicit methods are stability-limited by the fast mode, so the whole point of an *implicit*
// method is that its amplification factor obeys |R(z)| <= 1 for every z in the left half plane (A-stability)
// and additionally R(z) -> 0 as z -> -infinity (L-stability), which kills the stiff transient instead of letting
// it oscillate.
//
// Implemented here: backward (implicit) Euler, the canonical L-stable method,  y_{n+1} = y_n + h f(t+h, y_{n+1}),
// with R(z) = 1/(1-z), and the trapezoidal rule as the A-stable-but-not-L-stable control, with
// R(z) = (1 + z/2)/(1 - z/2) -> -1. Each step is solved by a damped-free Newton iteration on
// G(y) = y - base - h * g(t*, y) with a central-difference Jacobian and a pivoted Gaussian solve, so the code
// is not restricted to the linear test problems: the nonlinear stiff decay y' = -1000 y^2 is integrated too.
//
// Verified, with tolerances taken from closed forms or independent algebra:
//   * R(z) measured by actually stepping y' = lambda y and compared with 1/(1-z) and (1+z/2)/(1-z/2);
//   * A-stability by a dense grid over the left half plane, plus L-stability: |R_IE(-1000)| ~ 1e-3 while
//     |R_TR(-1000)| -> 1, and RK4 (task 786) is shown to fail the same test;
//   * the Newton residual invariant: the accepted y_{n+1} satisfies the implicit equation to ~1e-13, and the
//     iteration count is reported (1 for linear problems, several for the nonlinear one);
//   * the one-step amplification of a 2x2 stiff linear system against (I - hA)^{-1} computed from the closed
//     form 2x2 inverse - an oracle that shares no code with the Newton solver;
//   * the classic stiff problem y' = -1000(y - cos t) - sin t (exact solution cos t) with a step 10x outside
//     explicit RK4's stability region: implicit Euler stays on the slow manifold, RK4 blows up (measured).
//
// Documented simplification (the task asks for the *concept*): the file implements the two textbook one-step
// methods plus a general Newton solver for the implicit stage; there is no variable-order BDF/VODE machinery,
// no step-size control and no error estimator, and the Jacobian is obtained by finite differences rather than
// analytically. The stiffness handling itself - implicit stage, A/L-stability analysis - is the real thing.
using System;

internal delegate double[] VectorField(double t, double[] y);

internal static class Dense
{
    /** Gaussian elimination with partial pivoting; throws when the matrix is numerically singular. */
    internal static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        var m = (double[,])a.Clone();
        var rhs = (double[])b.Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row, col]) > Math.Abs(m[pivot, col])) pivot = row;
            }
            if (Math.Abs(m[pivot, col]) < 1e-300) throw new InvalidOperationException("singular matrix");
            if (pivot != col)
            {
                for (int c = 0; c < n; c++) (m[col, c], m[pivot, c]) = (m[pivot, c], m[col, c]);
                (rhs[col], rhs[pivot]) = (rhs[pivot], rhs[col]);
            }
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row, col] / m[col, col];
                if (factor == 0.0) continue;
                for (int c = col; c < n; c++) m[row, c] -= factor * m[col, c];
                rhs[row] -= factor * rhs[col];
            }
        }
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = rhs[row];
            for (int c = row + 1; c < n; c++) sum -= m[row, c] * x[c];
            x[row] = sum / m[row, row];
        }
        return x;
    }

    /** Closed-form 2x2 inverse, used as an independent oracle for the linear test system. */
    internal static double[,] Inverse2(double[,] a)
    {
        double det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0];
        if (Math.Abs(det) < 1e-300) throw new InvalidOperationException("singular matrix");
        return new double[2, 2] { { a[1, 1] / det, -a[0, 1] / det }, { -a[1, 0] / det, a[0, 0] / det } };
    }

    internal static double[] Multiply(double[,] a, double[] x)
    {
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < x.Length; j++) sum += a[i, j] * x[j];
            r[i] = sum;
        }
        return r;
    }
}

internal enum ImplicitMethod { BackwardEuler, Trapezoid }

internal sealed class StiffSolver
{
    internal int TotalNewtonIterations { get; private set; }
    internal int AcceptedSteps { get; private set; }
    internal int MaxNewtonIterations { get; private set; }

    /** Central-difference Jacobian of the vector field at (t, y). */
    internal static double[,] Jacobian(VectorField f, double t, double[] y)
    {
        int n = y.Length;
        var jac = new double[n, n];
        var probe = (double[])y.Clone();
        for (int j = 0; j < n; j++)
        {
            double eps = 1e-6 * Math.Max(1.0, Math.Abs(y[j]));
            probe[j] = y[j] + eps;
            double[] plus = f(t, probe);
            probe[j] = y[j] - eps;
            double[] minus = f(t, probe);
            probe[j] = y[j];
            for (int i = 0; i < n; i++) jac[i, j] = (plus[i] - minus[i]) / (2.0 * eps);
        }
        return jac;
    }

    /**
     * One implicit step. For backward Euler G(y) = y - y_n - h f(t+h, y); for the trapezoidal rule the explicit
     * part h/2 f(t, y_n) is folded into the constant term. Newton: (I - h g' dg/dy) delta = -G.
     */
    internal double[] Step(VectorField f, ImplicitMethod method, double t, double[] y, double h, out int iterations)
    {
        int n = y.Length;
        double[] f0 = f(t, y);
        var constant = new double[n];
        double weight;
        if (method == ImplicitMethod.BackwardEuler)
        {
            for (int i = 0; i < n; i++) constant[i] = y[i];
            weight = h;
        }
        else
        {
            for (int i = 0; i < n; i++) constant[i] = y[i] + h / 2.0 * f0[i];
            weight = h / 2.0;
        }

        var current = (double[])y.Clone();
        iterations = 0;
        double residual = double.MaxValue;
        for (int it = 0; it < 50; it++)
        {
            iterations++;
            double[] fx = f(t + h, current);
            var g = new double[n];
            for (int i = 0; i < n; i++) g[i] = current[i] - constant[i] - weight * fx[i];
            residual = 0.0;
            foreach (double v in g) residual = Math.Max(residual, Math.Abs(v));
            if (residual < 1e-13 * Math.Max(1.0, Norm(current))) break;

            double[,] jac = Jacobian(f, t + h, current);
            var m = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) m[i, j] = -weight * jac[i, j];
                m[i, i] += 1.0;
            }
            for (int i = 0; i < n; i++) g[i] = -g[i];
            double[] delta = Dense.Solve(m, g);
            for (int i = 0; i < n; i++) current[i] += delta[i];
        }
        if (residual >= 1e-13 * Math.Max(1.0, Norm(current)))
        {
            throw new InvalidOperationException("implicit stage did not converge; I - h*J is singular here");
        }
        if (iterations > MaxNewtonIterations) MaxNewtonIterations = iterations;
        TotalNewtonIterations += iterations;
        return current;
    }

    internal double[] Integrate(VectorField f, ImplicitMethod method, double[] y0, double t0, double t1, double h)
    {
        if (h <= 0.0) throw new ArgumentException("step size must be positive", nameof(h));
        if (t1 < t0) throw new ArgumentException("interval must run forward", nameof(t1));
        double span = t1 - t0;
        if (span > 0.0 && h > span) throw new ArgumentException("step size exceeds the interval", nameof(h));

        TotalNewtonIterations = 0;
        AcceptedSteps = 0;
        MaxNewtonIterations = 0;
        double[] y = (double[])y0.Clone();
        if (span == 0.0) return y;
        int count = (int)Math.Ceiling(span / h - 1e-12);
        double actual = span / count;
        double t = t0;
        for (int s = 0; s < count; s++)
        {
            y = Step(f, method, t, y, actual, out _);
            t += actual;
            AcceptedSteps++;
        }
        return y;
    }

    /** Residual of the implicit equation at the returned state; the invariant the Newton solve must satisfy. */
    internal static double ImplicitResidual(VectorField f, ImplicitMethod method, double[] yOld, double t, double h, double[] yNew)
    {
        double[] f0 = f(t, yOld);
        double[] fx = f(t + h, yNew);
        double worst = 0.0;
        for (int i = 0; i < yNew.Length; i++)
        {
            double g = method == ImplicitMethod.BackwardEuler
                ? yNew[i] - yOld[i] - h * fx[i]
                : yNew[i] - yOld[i] - h / 2.0 * (f0[i] + fx[i]);
            worst = Math.Max(worst, Math.Abs(g));
        }
        return worst;
    }

    internal static double AmplificationBackwardEuler(double z) => 1.0 / (1.0 - z);

    internal static double AmplificationTrapezoid(double z) => (1.0 + z / 2.0) / (1.0 - z / 2.0);

    internal static double AmplificationRk4(double z) =>
        1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0;

    internal static double Norm(double[] v)
    {
        double m = 0.0;
        foreach (double x in v) m = Math.Max(m, Math.Abs(x));
        return m;
    }
}

public static class Solution
{
    private static VectorField LinearScalar(double lambda) => (t, y) => new[] { lambda * y[0] };

    private static VectorField LinearSystem(double[,] a) => (t, y) => new[]
    {
        a[0, 0] * y[0] + a[0, 1] * y[1],
        a[1, 0] * y[0] + a[1, 1] * y[1]
    };

    /** Classic stiff test: y' = -1000 (y - cos t) - sin t, exact solution y = cos t (slow manifold). */
    private static VectorField StiffDrag => (t, y) => new[] { -1000.0 * (y[0] - Math.Cos(t)) - Math.Sin(t) };

    private static VectorField StiffDecayNonlinear => (t, y) => new[] { -1000.0 * y[0] * y[0] };

    /** Reference explicit RK4, kept only to show that it is stability-limited on the stiff problem. */
    private static double[] ExplicitRk4(VectorField f, double[] y0, double t0, double t1, double h, out double peak)
    {
        double[] y = (double[])y0.Clone();
        int count = (int)Math.Ceiling((t1 - t0) / h - 1e-12);
        double actual = (t1 - t0) / count;
        double t = t0;
        peak = Math.Abs(y[0]);
        for (int s = 0; s < count; s++)
        {
            double[] k1 = f(t, y);
            double[] k2 = f(t + actual / 2.0, Add(y, k1, actual / 2.0));
            double[] k3 = f(t + actual / 2.0, Add(y, k2, actual / 2.0));
            double[] k4 = f(t + actual, Add(y, k3, actual));
            for (int i = 0; i < y.Length; i++)
            {
                y[i] += actual / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                if (Math.Abs(y[i]) > peak) peak = Math.Abs(y[i]);
            }
            t += actual;
        }
        return y;
    }

    private static double[] Add(double[] y, double[] k, double h)
    {
        var r = new double[y.Length];
        for (int i = 0; i < y.Length; i++) r[i] = y[i] + h * k[i];
        return r;
    }

    public static void Main()
    {
        var solver = new StiffSolver();

        // 1-2. Amplification factors measured by stepping the linear test equation, against the closed forms.
        double worstAmp = 0.0;
        foreach (double z in new[] { -1.0, -10.0, -100.0, -0.5, 0.0, 0.5 })
        {
            double[] ie = solver.Step(LinearScalar(z), ImplicitMethod.BackwardEuler, 0.0, new[] { 1.0 }, 1.0, out int ieIter);
            double[] tr = solver.Step(LinearScalar(z), ImplicitMethod.Trapezoid, 0.0, new[] { 1.0 }, 1.0, out int trIter);
            worstAmp = Math.Max(worstAmp, Math.Abs(ie[0] - StiffSolver.AmplificationBackwardEuler(z)));
            worstAmp = Math.Max(worstAmp, Math.Abs(tr[0] - StiffSolver.AmplificationTrapezoid(z)));
            Check(ieIter <= 3 && trIter <= 3, "linear step needs at most three residual evaluations at z=" + z
                + " (ie=" + ieIter + ", tr=" + trIter + ")");
        }
        Check(worstAmp < 1e-10, "measured amplification equals 1/(1-z) and (1+z/2)/(1-z/2)");
        Console.WriteLine("  amplification check: max deviation " + worstAmp.ToString("E3"));

        // 3-5. A-stability on a grid over the closed left half plane, then L-stability in the limit z -> -inf.
        double worstIe = 0.0, worstTr = 0.0;
        for (int re = -60; re <= 0; re++)
        {
            for (int im = -60; im <= 60; im += 3)
            {
                double zre = re * 0.5, zim = im * 0.5;
                if (zre > 0.0) continue;
                worstIe = Math.Max(worstIe, ComplexMagnitudeBackwardEuler(zre, zim));
                worstTr = Math.Max(worstTr, ComplexMagnitudeTrapezoid(zre, zim));
            }
        }
        Check(worstIe <= 1.0 + 1e-12, "backward Euler is A-stable (max |R| = " + worstIe.ToString("F6") + ")");
        Check(worstTr <= 1.0 + 1e-12, "trapezoid is A-stable (max |R| = " + worstTr.ToString("F6") + ")");
        Check(Math.Abs(StiffSolver.AmplificationBackwardEuler(-1000.0)) < 2e-3, "backward Euler damps the stiff mode");
        Check(Math.Abs(StiffSolver.AmplificationTrapezoid(-1000.0)) > 0.99,
            "trapezoid keeps the stiff mode at unit amplitude (not L-stable)");
        Check(Math.Abs(StiffSolver.AmplificationRk4(-5.0)) > 5.0, "RK4 is not A-stable at z = -5");

        // 6. Order: backward Euler is first order, the trapezoidal rule second order.
        double[] y0 = { 1.0 };
        var smooth = LinearScalar(-1.0);
        double ieh = Math.Abs(solver.Integrate(smooth, ImplicitMethod.BackwardEuler, y0, 0.0, 1.0, 0.1)[0] - Math.Exp(-1.0));
        double ieh2 = Math.Abs(solver.Integrate(smooth, ImplicitMethod.BackwardEuler, y0, 0.0, 1.0, 0.05)[0] - Math.Exp(-1.0));
        double trh = Math.Abs(solver.Integrate(smooth, ImplicitMethod.Trapezoid, y0, 0.0, 1.0, 0.1)[0] - Math.Exp(-1.0));
        double trh2 = Math.Abs(solver.Integrate(smooth, ImplicitMethod.Trapezoid, y0, 0.0, 1.0, 0.05)[0] - Math.Exp(-1.0));
        Console.WriteLine("  order test y'=-y: backward Euler " + (Math.Log(ieh / ieh2) / Math.Log(2.0)).ToString("F3")
            + " (expect 1), trapezoid " + (Math.Log(trh / trh2) / Math.Log(2.0)).ToString("F3") + " (expect 2)");
        Check(Math.Log(ieh / ieh2) / Math.Log(2.0) > 0.95, "backward Euler has order 1");
        Check(Math.Log(trh / trh2) / Math.Log(2.0) > 1.9, "trapezoid has order 2");

        // 7-9. The stiff problem, h = 0.01 => h * lambda = -10, far outside RK4's stability region.
        var stiff0 = new[] { 1.0 };
        double[] ieStiff = solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 0.0, 1.0, 0.01);
        double[] trStiff = solver.Integrate(StiffDrag, ImplicitMethod.Trapezoid, stiff0, 0.0, 1.0, 0.01);
        double[] explicitStiff = ExplicitRk4(StiffDrag, stiff0, 0.0, 1.0, 0.01, out double peak);
        double exactEnd = Math.Cos(1.0);
        Console.WriteLine("  stiff drag at t=1: exact=" + exactEnd.ToString("F6") + " backwardEuler=" + ieStiff[0].ToString("F6")
            + " trapezoid=" + trStiff[0].ToString("F6") + " explicitRK4=" + explicitStiff[0].ToString("E3")
            + " (peak |y|=" + peak.ToString("E3") + ")");
        Check(Math.Abs(ieStiff[0] - exactEnd) < 0.01, "backward Euler stays on the slow manifold with h*lambda = -10");
        Check(Math.Abs(trStiff[0] - exactEnd) < 0.01, "trapezoid also tracks the slow manifold");
        Check(peak > 1e2, "explicit RK4 is unstable on the same step size (peak " + peak.ToString("E3") + ")");

        // 10. Newton residual invariant, recorded for the stiff run.
        double[] firstStep = solver.Step(StiffDrag, ImplicitMethod.BackwardEuler, 0.0, stiff0, 0.01, out int innerIterations);
        double residual = StiffSolver.ImplicitResidual(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 0.0, 0.01, firstStep);
        Console.WriteLine("  Newton: iterations=" + innerIterations + " residual=" + residual.ToString("E3"));
        Check(residual < 1e-10, "the accepted state satisfies the implicit Euler equation");
        Check(innerIterations <= 6, "Newton converges quickly on the linear stiff problem");

        // 11. Linear-system oracle: one step must equal (I - h A)^{-1} y with a closed-form 2x2 inverse.
        var a = new double[2, 2] { { -1.0, 0.0 }, { 0.0, -1000.0 } };
        var start = new[] { 2.0, 3.0 };
        double hStep = 0.01;
        double[] numeric = solver.Step(LinearSystem(a), ImplicitMethod.BackwardEuler, 0.0, start, hStep, out _);
        var identity = new double[2, 2] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var m = new double[2, 2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) m[i, j] = identity[i, j] - hStep * a[i, j];
        double[] reference = Dense.Multiply(Dense.Inverse2(m), start);
        double oracleError = Math.Max(Math.Abs(numeric[0] - reference[0]), Math.Abs(numeric[1] - reference[1]));
        Console.WriteLine("  linear oracle: max deviation from (I - hA)^-1 y = " + oracleError.ToString("E3"));
        Check(oracleError < 1e-9, "implicit step matches the closed-form (I - hA)^{-1} amplification");
        Check(numeric[1] < 0.1 * Math.Abs(start[1]), "the stiff component is damped by a factor ~ 1/(1 + h lambda)");

        // 12-13. Why L-stability matters: on pure decay with h*lambda = -10 the trapezoidal rule alternates in
        //        sign and stalls, while backward Euler decays monotonically and beats it easily.
        var decay = LinearScalar(-1000.0);
        var ieTrail = new[] { 1.0 };
        var trTrail = new[] { 1.0 };
        var ieValues = new double[5];
        var trValues = new double[5];
        for (int n = 0; n < 5; n++)
        {
            ieTrail = solver.Step(decay, ImplicitMethod.BackwardEuler, n * 0.01, ieTrail, 0.01, out _);
            trTrail = solver.Step(decay, ImplicitMethod.Trapezoid, n * 0.01, trTrail, 0.01, out _);
            ieValues[n] = ieTrail[0];
            trValues[n] = trTrail[0];
        }
        bool alternating = true, monotone = true;
        for (int n = 0; n < 5; n++)
        {
            if (n > 0 && trValues[n] * trValues[n - 1] >= 0.0) alternating = false;
            if (n > 0 && ieValues[n] >= ieValues[n - 1]) monotone = false;
        }
        Console.WriteLine("  y'=-1000y, 5 steps of 0.01: backward Euler=" + ieValues[4].ToString("E3")
            + " trapezoid=" + trValues[4].ToString("E3") + " exact=" + Math.Exp(-50.0).ToString("E3"));
        Check(alternating, "trapezoid rings with alternating sign on the stiff component");
        Check(monotone, "backward Euler decays monotonically");
        Check(Math.Abs(ieValues[4] - Math.Exp(-50.0)) < Math.Abs(trValues[4] - Math.Exp(-50.0)),
            "backward Euler is closer to the exact solution on the stiff transient");

        // 14-15. Nonlinear stiff decay y' = -1000 y^2, exact y = 1/(1 + 1000 t): Newton must genuinely iterate.
        var nonlinearCorpus = new[] { 1.0 };
        double[] nonlinear = solver.Integrate(StiffDecayNonlinear, ImplicitMethod.BackwardEuler, nonlinearCorpus, 0.0, 0.01, 2.5e-4);
        int nonlinearIterations = solver.TotalNewtonIterations, perStep = solver.MaxNewtonIterations;
        double[] nonlinearFiner = solver.Integrate(StiffDecayNonlinear, ImplicitMethod.BackwardEuler, nonlinearCorpus, 0.0, 0.01, 1.25e-4);
        double nonlinearExact = 1.0 / (1.0 + 1000.0 * 0.01);
        double coarseError = Math.Abs(nonlinear[0] - nonlinearExact), fineError = Math.Abs(nonlinearFiner[0] - nonlinearExact);
        double relative = coarseError / nonlinearExact;
        Console.WriteLine("  nonlinear stiff decay: value=" + nonlinear[0].ToString("F6") + " exact=" + nonlinearExact.ToString("F6")
            + " rel error=" + relative.ToString("E3") + " observed order=" + (Math.Log(coarseError / fineError) / Math.Log(2.0)).ToString("F3")
            + " Newton iterations=" + nonlinearIterations + " max per step=" + perStep);
        Check(relative < 0.08, "nonlinear stiff decay integrated with bounded relative error");
        Check(Math.Log(coarseError / fineError) / Math.Log(2.0) > 0.85, "first-order convergence on the nonlinear stiff problem");
        Check(perStep >= 3, "Newton iterates several times on the nonlinear stiff equation");

        // 16-17. The solver is deterministic and never mutates its input.
        var untouched = new[] { 1.0, -2.0 };
        solver.Step(LinearSystem(a), ImplicitMethod.Trapezoid, 0.0, untouched, 0.05, out _);
        Check(untouched[0] == 1.0 && untouched[1] == -2.0, "caller state is not mutated");
        double[] repeatA = solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 0.0, 0.5, 0.01);
        double[] repeatB = solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 0.0, 0.5, 0.01);
        Check(repeatA[0] == repeatB[0], "identical calls give bit-identical results");

        // 18-19. Boundary and rejection behaviour.
        Check(solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 1.0, 1.0, 0.1)[0] == 1.0,
            "zero-length interval returns the initial state");
        Check(Throws(() => solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 0.0, 1.0, 0.0)),
            "h = 0 rejected");
        Check(Throws(() => solver.Integrate(StiffDrag, ImplicitMethod.BackwardEuler, stiff0, 1.0, 0.0, 0.1)),
            "backwards interval rejected");
        Check(Throws(() => Dense.Inverse2(new double[2, 2])), "singular 2x2 matrix rejected");
        Check(Throws(() => solver.Step(LinearScalar(1.0), ImplicitMethod.BackwardEuler, 0.0, new[] { 1.0 }, 1.0, out _)),
            "h * lambda = 1 (the pole of R) is detected instead of spinning");

        Console.WriteLine("PASS");
    }

    private static double ComplexMagnitudeBackwardEuler(double re, double im)
    {
        double dr = 1.0 - re, di = -im;
        return 1.0 / Math.Sqrt(dr * dr + di * di);
    }

    private static double ComplexMagnitudeTrapezoid(double re, double im)
    {
        double nr = 1.0 + re / 2.0, ni = im / 2.0;
        double dr = 1.0 - re / 2.0, di = -im / 2.0;
        return Math.Sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
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
