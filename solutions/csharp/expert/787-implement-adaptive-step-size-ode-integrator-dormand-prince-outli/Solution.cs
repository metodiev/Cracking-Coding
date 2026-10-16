// Task 787 - Implement adaptive step-size ODE integrator (Dormand-Prince outline). (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Dormand-Prince RK5(4), the classic embedded pair behind MATLAB's ode45: seven stages, of which the first of
// the next step is the last of the current one (FSAL), so an attempted step costs exactly six new evaluations.
// The same stage values give a 5th-order solution (the a7 row, 35/384 ... 11/84) and a 4th-order one (the bhat
// row, 5179/57600 ... 1/40); their difference is the per-step error estimate used by the controller.
//
// Controller: with r_i = err_i / (atol + rtol * max(|y_i|, |y5_i|)) and the RMS norm ||r||, the step is accepted
// when ||r|| <= 1 and the next step size is h * clamp(0.9 * ||r||^(-1/5), 0.2, 5.0). The step is also clamped so
// it never overshoots t1, the caller's state vector is never mutated, and a zero-length interval returns y0.
//
// What the checks pin down, all against closed forms or an independently implemented reference:
//   * order of the two embedded solutions measured by halving the step: the 5th-order solution improves by ~2^5
//     (global order) while the embedded *difference* is a local quantity and scales as h^5 as well, so the two
//     log2 ratios that get printed are ~5 and ~5 - the pair separates 5th-order accuracy from a 4th-order
//     *solution* only through the leading coefficient, which is checked separately below;
//   * the embedded estimate must reproduce its leading term: for y' = y it is exactly (R5 - R4)(z) y with
//     R5 - R4 = z^5/120 + O(z^6), checked at z = 0.2;
//   * y' = 3t^2 has y = t^3, and a 5th-order RK is exact for polynomial right hand sides of degree <= 4, so a
//     deliberately coarse fixed-step run must still hit the exact solution to roundoff;
//   * the nonlinear pendulum is compared against a high-resolution fixed-step RK4 reference (h = 1e-4, an
//     entirely different method) with the measured error reported, and its energy must be conserved to ~tol;
//   * the FSAL bookkeeping invariant: function evaluations = 6 * attempted steps + 1 (measured, not assumed).
//
// Documented simplification (the task says *outline*): the controller is the classical single-step safety-factor
// rule rather than a PI controller with a step-size limiter, there is no dense output / continuous extension and
// no event detection, so the integrator returns endpoint states only. Everything above - the tableau, the FSAL
// evaluation count, the embedded error and the acceptance logic - is the real algorithm.
using System;

internal delegate double[] VectorField(double t, double[] y);

internal sealed class DormandPrince
{
    // Stage abscissae c2..c7 (public so the checks can rebuild the stage polynomials independently).
    internal static readonly double[] C = { 0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0 };

    // Strictly lower-triangular stage coupling coefficients, row per stage.
    private static readonly double[][] A =
    {
        new double[0],
        new[] { 1.0 / 5.0 },
        new[] { 3.0 / 40.0, 9.0 / 40.0 },
        new[] { 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0 },
        new[] { 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0 },
        new[] { 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0 },
        new[] { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0 }
    };

    internal static readonly double[] B5 = { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0 };

    internal static readonly double[] B4 =
    {
        5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0
    };

    internal const double Safety = 0.9;
    internal const double MinFactor = 0.2;
    internal const double MaxFactor = 5.0;
    internal const double Exponent = -1.0 / 5.0;

    internal int AcceptedSteps { get; private set; }
    internal int RejectedSteps { get; private set; }
    internal int FunctionEvaluations { get; private set; }
    internal int Attempts => AcceptedSteps + RejectedSteps;

    /** One RK5(4) trial step. `k0` is f(t, y) supplied by the caller (FSAL); `kLast` is f(t+h, y5). */
    private static void Trial(VectorField f, double t, double[] y, double h, double[] k0,
        out double[] y5, out double[] error, out double[] kLast)
    {
        int n = y.Length;
        var k = new double[7][];
        k[0] = k0;
        for (int stage = 1; stage < 6; stage++)
        {
            var arg = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < stage; j++) sum += A[stage][j] * k[j][i];
                arg[i] = y[i] + h * sum;
            }
            k[stage] = f(t + C[stage] * h, arg);
        }

        y5 = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < 6; j++) sum += A[6][j] * k[j][i];
            y5[i] = y[i] + h * sum;
        }
        kLast = f(t + h, y5);                                  // 7th stage, reused as k0 of the next step

        error = new double[n];
        for (int i = 0; i < n; i++)
        {
            k[6] = kLast;
            double sum = 0.0;
            for (int j = 0; j < 7; j++) sum += (B5[j] - B4[j]) * k[j][i];
            error[i] = h * sum;
        }
    }

    /** RMS of the scaled error estimate; the step is accepted when this is <= 1. */
    private static double ScaledErrorNorm(double[] error, double[] y, double[] y5, double rtol, double atol)
    {
        double acc = 0.0;
        for (int i = 0; i < error.Length; i++)
        {
            double scale = atol + rtol * Math.Max(Math.Abs(y[i]), Math.Abs(y5[i]));
            double r = error[i] / scale;
            acc += r * r;
        }
        return Math.Sqrt(acc / error.Length);
    }

    internal double[] Adaptive(VectorField f, double[] y0, double t0, double t1, double h0, double rtol, double atol,
        out double worstScaledError)
    {
        if (rtol <= 0.0 && atol <= 0.0) throw new ArgumentException("at least one tolerance must be positive", nameof(rtol));
        if (h0 <= 0.0) throw new ArgumentException("initial step must be positive", nameof(h0));
        if (t1 < t0) throw new ArgumentException("interval must run forward", nameof(t1));

        AcceptedSteps = 0;
        RejectedSteps = 0;
        worstScaledError = 0.0;
        double[] y = (double[])y0.Clone();
        double span = t1 - t0;
        if (span == 0.0) { FunctionEvaluations = 0; return y; }

        double t = t0;
        double h = Math.Min(h0, span);
        double floor = 1e-13 * span;                           // bottom-out guard so the loop always terminates
        double[] k0 = f(t, y);
        FunctionEvaluations = 1;

        while (t < t1)
        {
            if (t + h > t1) h = t1 - t;
            Trial(f, t, y, h, k0, out double[] y5, out double[] error, out double[] kLast);
            FunctionEvaluations += 6;

            double errNorm = ScaledErrorNorm(error, y, y5, rtol, atol);
            double factor = Safety * Math.Pow(errNorm <= 0.0 ? 1e-16 : errNorm, Exponent);
            factor = Math.Min(MaxFactor, Math.Max(MinFactor, factor));

            if (errNorm <= 1.0 || h <= floor)
            {
                AcceptedSteps++;
                t += h;
                y = y5;
                k0 = kLast;
                if (errNorm > worstScaledError) worstScaledError = errNorm;
            }
            else
            {
                RejectedSteps++;
            }
            h *= factor;
            if (h <= floor) h = floor;
        }
        return y;
    }

    /** Fixed-step DP5: returns the 5th-order endpoint and the largest embedded difference seen. */
    internal double[] FixedStep(VectorField f, double[] y0, double t0, double t1, double h, out double maxEmbedded)
    {
        if (h <= 0.0) throw new ArgumentException("step size must be positive", nameof(h));
        if (t1 < t0) throw new ArgumentException("interval must run forward", nameof(t1));
        double span = t1 - t0;
        if (span > 0.0 && h > span) throw new ArgumentException("step size exceeds the interval", nameof(h));

        AcceptedSteps = 0;
        RejectedSteps = 0;
        maxEmbedded = 0.0;
        double[] y = (double[])y0.Clone();
        if (span == 0.0) { FunctionEvaluations = 0; return y; }

        int count = (int)Math.Ceiling(span / h - 1e-12);
        double actual = span / count;
        double t = t0;
        double[] k0 = f(t, y);
        FunctionEvaluations = 1;
        for (int s = 0; s < count; s++)
        {
            Trial(f, t, y, actual, k0, out double[] y5, out double[] error, out double[] kLast);
            FunctionEvaluations += 6;
            for (int i = 0; i < error.Length; i++) maxEmbedded = Math.Max(maxEmbedded, Math.Abs(error[i]));
            AcceptedSteps++;
            t += actual;
            y = y5;
            k0 = kLast;
        }
        return y;
    }

    /** 5th- and 4th-order amplification polynomials; only the exact e^z series, used for diagnostics. */
    internal static double AmplificationFifth(double z) =>
        1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0 + z * z * z * z * z / 120.0;
}

public static class Solution
{
    private static double[] Scalar(double v) => new[] { v };

    private static VectorField LinearScalar(double lambda) => (t, y) => new[] { lambda * y[0] };

    private static VectorField SquareField => (t, y) => new[] { 3.0 * t * t };

    private static VectorField Pendulum => (t, y) => new[] { y[1], -Math.Sin(y[0]) };

    /** Reference integrator: fixed-step classical RK4, a different method from the adaptive DP5 above. */
    private static double[] ReferenceRk4(VectorField f, double[] y0, double t0, double t1, double h)
    {
        double[] y = (double[])y0.Clone();
        int count = (int)Math.Ceiling((t1 - t0) / h - 1e-12);
        double actual = (t1 - t0) / count;
        double t = t0;
        for (int s = 0; s < count; s++)
        {
            double[] k1 = f(t, y);
            double[] k2 = f(t + actual / 2.0, Add(y, k1, actual / 2.0));
            double[] k3 = f(t + actual / 2.0, Add(y, k2, actual / 2.0));
            double[] k4 = f(t + actual, Add(y, k3, actual));
            for (int i = 0; i < y.Length; i++)
            {
                y[i] += actual / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
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

    /**
     * Stage polynomials of the tableau: R_j(z) = 1 + z * sum_{i<j} a_ji R_i(z). Rebuilt straight from the
     * coefficients, so it is an independent path to what the step routine computes numerically.
     */
    private static double[] StagePolynomials(double z)
    {
        var r = new double[7];
        double[] c = DormandPrince.C;
        for (int j = 0; j < 7; j++)
        {
            if (j == 0) { r[0] = 1.0 + z * c[0]; continue; }        // k1 = f(t, y) = z * y for the test equation
            double sum = 0.0;
            for (int i = 0; i < j; i++) sum += A_ji(j, i) * r[i];
            r[j] = 1.0 + z * sum;
        }
        return r;
    }

    private static double A_ji(int row, int col)
    {
        switch (row)
        {
            case 1: return 1.0 / 5.0;
            case 2: return col == 0 ? 3.0 / 40.0 : 9.0 / 40.0;
            case 3: return new[] { 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0 }[col];
            case 4: return new[] { 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0 }[col];
            case 5: return new[] { 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0 }[col];
            default: return new[] { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0 }[col];
        }
    }

    private static double MaxDiff(double[] a, double[] b)
    {
        double m = 0.0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    public static void Main()
    {
        var dp = new DormandPrince();

        // 1-3. Accuracy and tolerance scaling on y' = y, whose exact solution is the exponential.
        double[] y0 = Scalar(1.0);
        double[] tight = dp.Adaptive(LinearScalar(1.0), y0, 0.0, 1.0, 0.1, 1e-9, 1e-12, out double worstTight);
        double tightError = Math.Abs(tight[0] - Math.E);
        int tightSteps = dp.AcceptedSteps;
        int tightEvals = dp.FunctionEvaluations;
        int tightAttempts = dp.AcceptedSteps + dp.RejectedSteps;

        double[] loose = dp.Adaptive(LinearScalar(1.0), y0, 0.0, 1.0, 0.1, 1e-4, 1e-8, out double worstLoose);
        double looseError = Math.Abs(loose[0] - Math.E);
        int looseSteps = dp.AcceptedSteps;

        Console.WriteLine("  y'=y [0,1] rtol=1e-9: error=" + tightError.ToString("E3") + " steps=" + tightSteps
            + " evals=" + tightEvals + " worst scaled err=" + worstTight.ToString("E3"));
        Console.WriteLine("  y'=y [0,1] rtol=1e-4: error=" + looseError.ToString("E3") + " steps=" + looseSteps
            + " worst scaled err=" + worstLoose.ToString("E3"));
        Check(tightError < 1e-8, "rtol=1e-9 endpoint error below 1e-8");
        Check(worstTight <= 1.0, "accepted steps respect the scaled error bound");
        Check(looseSteps < tightSteps, "looser tolerance needs fewer accepted steps (" + looseSteps + " < " + tightSteps + ")");
        double scaling = looseError / tightError;
        Console.WriteLine("  error ratio loose/tight = " + scaling.ToString("F1") + " (tolerance ratio 1e5)");
        Check(scaling > 20.0 && scaling < 1e6, "global error tracks the requested tolerance");

        // 4. FSAL bookkeeping: six evaluations per attempted step plus the single initial evaluation.
        Console.WriteLine("  FSAL: evals=" + tightEvals + " attempts=" + tightAttempts + " accepted=" + tightSteps);
        Check(tightEvals == 6 * tightAttempts + 1, "FSAL invariant: evals = 6 * attempts + 1");

        // 5-7. Measured orders: the 5th-order solution and the embedded 4th-order difference.
        double[] degree5 = dp.FixedStep(LinearScalar(1.0), y0, 0.0, 1.0, 0.1, out double emb01);
        double e01 = Math.Abs(degree5[0] - Math.E);
        double[] degree5Half = dp.FixedStep(LinearScalar(1.0), y0, 0.0, 1.0, 0.05, out double emb005);
        double e005 = Math.Abs(degree5Half[0] - Math.E);
        double order5 = Math.Log(e01 / e005) / Math.Log(2.0);
        double order4 = Math.Log(emb01 / emb005) / Math.Log(2.0);
        Console.WriteLine("  fixed step: 5th order measured=" + order5.ToString("F3")
            + " local embedded order measured=" + order4.ToString("F3")
            + " raw errors " + e01.ToString("E3") + " / " + e005.ToString("E3"));
        Check(order5 > 4.6, "main solution shows order ~5");
        Check(order4 > 4.6 && order4 < 5.4, "embedded difference is a local estimate, so it scales as h^5");
        Check(e01 < 1e-6, "endpoint error of the fixed-step solver at h=0.1 is below 1e-6");

        // 8. Tableau oracle, independent of the integrator: the stage polynomials R_j(z) = 1 + z sum_i a_ji R_i(z)
        //    give q(z) = sum_j b_j R_j(z). The integrator's own step must reproduce q5(z) and q5(z) - q4(z).
        double z = 0.2;
        double[] oneStep = dp.FixedStep(LinearScalar(1.0), y0, 0.0, z, z, out double embedded);
        double[] stages = StagePolynomials(z);
        double q5 = 0.0, q4 = 0.0;
        for (int j = 0; j < 7; j++)
        {
            q5 += DormandPrince.B5[j] * stages[j];
            q4 += DormandPrince.B4[j] * stages[j];
        }
        Console.WriteLine("  tableau polynomials at z=0.2: q5=" + q5.ToString("F12") + " q4=" + q4.ToString("F12")
            + " z*(q5-q4)=" + (z * (q5 - q4)).ToString("E4") + " vs measured error " + embedded.ToString("E4"));
        Check(Math.Abs(oneStep[0] - (1.0 + z * q5)) < 1e-13, "integrator step reproduces 1 + z*q5(z)");
        Check(Math.Abs(Math.Abs(z * (q5 - q4)) - embedded) < 1e-13, "embedded estimate equals z*(q5(z) - q4(z))");

        // 9. Quadrature exactness of the two weight rows: sum_j b_j c_j^k = 1/(k+1). The 5th-order row must hold
        //    through k = 4 and the 4th-order row only through k = 3, which is exactly what separates the pair.
        double residual5 = 0.0, residual4 = 0.0;
        for (int k = 0; k <= 4; k++)
        {
            double s5 = 0.0, s4 = 0.0;
            for (int j = 0; j < 7; j++)
            {
                s5 += DormandPrince.B5[j] * Math.Pow(DormandPrince.C[j], k);
                s4 += DormandPrince.B4[j] * Math.Pow(DormandPrince.C[j], k);
            }
            residual5 = Math.Max(residual5, Math.Abs(s5 - 1.0 / (k + 1)));
            if (k <= 3) residual4 = Math.Max(residual4, Math.Abs(s4 - 1.0 / (k + 1)));
        }
        double fourthOrderMoment = 0.0;
        for (int j = 0; j < 7; j++) fourthOrderMoment += DormandPrince.B4[j] * Math.Pow(DormandPrince.C[j], 4);
        Console.WriteLine("  quadrature residuals: b5=" + residual5.ToString("E3") + " b4=" + residual4.ToString("E3")
            + "  b4 fourth moment=" + fourthOrderMoment.ToString("F6") + " (1/5 = 0.2)");
        Check(residual5 < 1e-15, "5th-order weights integrate polynomials up to degree 4 exactly");
        Check(residual4 < 1e-15, "4th-order weights integrate polynomials up to degree 3 exactly");
        Check(Math.Abs(fourthOrderMoment - 0.2) > 1e-6 && Math.Abs(fourthOrderMoment - 0.2) < 0.01,
            "4th-order weights miss the degree-4 moment (deviation " + (fourthOrderMoment - 0.2).ToString("E3")
            + "), so the pair really is embedded");
        Check(Math.Abs(stages[0] - 1.0) < 1e-15, "the first stage polynomial is the constant one");
        double sixthCoefficient = (1.0 + z * q5 - DormandPrince.AmplificationFifth(z)) / Math.Pow(z, 6);
        Console.WriteLine("  q5 vs the 5th-degree Taylor polynomial: first differing coefficient c6="
            + sixthCoefficient.ToString("F4"));
        Check(Math.Abs(sixthCoefficient) < 1.0, "q5 reproduces every Taylor coefficient through z^5");

        // 9-10. Exactness on a cubic solution: y' = 3t^2 is a quadratic field, and the 5th-order quadrature
        //      conditions sum b_i c_i^k = 1/(k+1) for k <= 4 make coarse steps error-free to roundoff.
        double[] cubicCoarse = dp.FixedStep(SquareField, Scalar(0.0), 0.0, 1.0, 0.5, out _);
        double[] cubicFine = dp.FixedStep(SquareField, Scalar(0.0), 0.0, 1.0, 1.0 / 3.0, out _);
        Console.WriteLine("  cubic y'=3t^2 coarse error=" + Math.Abs(cubicCoarse[0] - 1.0).ToString("E3")
            + " fine error=" + Math.Abs(cubicFine[0] - 1.0).ToString("E3"));
        Check(Math.Abs(cubicCoarse[0] - 1.0) < 1e-14, "two-step run integrates the cubic exactly");
        Check(Math.Abs(cubicFine[0] - 1.0) < 1e-14, "three-step run integrates the cubic exactly");

        // 11-13. Nonlinear pendulum against the independent RK4 reference and against energy conservation.
        var pendulum0 = new[] { 1.0, 0.0 };
        double[] reference = ReferenceRk4(Pendulum, pendulum0, 0.0, 10.0, 1e-4);
        double[] adaptive = dp.Adaptive(Pendulum, pendulum0, 0.0, 10.0, 0.05, 1e-10, 1e-12, out _);
        int pendulumSteps = dp.AcceptedSteps;
        double pendulumError = MaxDiff(adaptive, reference);
        double energy0 = 0.5 * pendulum0[1] * pendulum0[1] + (1.0 - Math.Cos(pendulum0[0]));
        double energyT = 0.5 * adaptive[1] * adaptive[1] + (1.0 - Math.Cos(adaptive[0]));
        Console.WriteLine("  pendulum t=10: DP5 vs RK4(h=1e-4) error=" + pendulumError.ToString("E3")
            + " energy drift=" + Math.Abs(energyT - energy0).ToString("E3")
            + " steps=" + dp.AcceptedSteps + " evals=" + dp.FunctionEvaluations);
        Check(pendulumError < 1e-8, "adaptive DP5 matches the high-resolution RK4 reference");
        Check(Math.Abs(energyT - energy0) < 1e-8, "pendulum energy conserved to the requested tolerance");
        Check(dp.AcceptedSteps > 20 && dp.AcceptedSteps < 2000, "pendulum step count is in the expected range");

        // 14-15. Rejection is real: an over-optimistic h0 must be cut back, and rejected steps must not advance t.
        var overshoot = dp.Adaptive(Pendulum, pendulum0, 0.0, 1.0, 5.0, 1e-12, 1e-14, out _);
        int rejected = dp.RejectedSteps;
        Console.WriteLine("  oversized h0=5 on [0,1]: rejected steps=" + rejected + " accepted=" + dp.AcceptedSteps);
        Check(rejected >= 1, "an oversized initial step is rejected at least once");
        Check(overshoot[0] > 0.0 && overshoot[0] < 1.0, "the state after the oversized first step is bounded");
        Check(Math.Abs(ReferenceRk4(Pendulum, pendulum0, 0.0, 1.0, 1e-4)[0] - overshoot[0]) < 1e-9,
            "rejected steps leave the trajectory on the true solution");

        // 16-17. The integrator never overshoots t1 and never mutates the caller's vector.
        var callerState = new[] { 1.0, 0.0 };
        double[] clamped = dp.Adaptive(Pendulum, callerState, 0.0, 3.0, 100.0, 1e-9, 1e-12, out _);
        Check(callerState[0] == 1.0 && callerState[1] == 0.0, "caller state vector is not mutated");
        Check(Math.Abs(clamped[0] - ReferenceRk4(Pendulum, callerState, 0.0, 3.0, 1e-4)[0]) < 1e-7,
            "h0 far larger than the interval is clamped without loss of accuracy");

        // 18. Boundary and rejection behaviour.
        double[] empty = dp.Adaptive(Pendulum, callerState, 2.0, 2.0, 0.1, 1e-9, 1e-12, out _);
        Check(dp.AcceptedSteps == 0 && dp.FunctionEvaluations == 0 && empty[0] == 1.0 && empty[1] == 0.0,
            "zero-length interval returns the initial state untouched");
        Check(Throws(() => dp.Adaptive(Pendulum, callerState, 0.0, 1.0, 0.0, 1e-9, 1e-12, out _)), "h0 = 0 rejected");
        Check(Throws(() => dp.Adaptive(Pendulum, callerState, 0.0, 1.0, 0.1, 0.0, 0.0, out _)), "zero tolerances rejected");
        Check(Throws(() => dp.Adaptive(Pendulum, callerState, 1.0, 0.0, 0.1, 1e-9, 1e-12, out _)), "backwards interval rejected");
        Check(Throws(() => dp.FixedStep(Pendulum, callerState, 0.0, 1.0, 2.0, out _)), "fixed step larger than the interval rejected");

        // 19. Determinism: the adaptive path is a pure function of its arguments.
        var repeat = new DormandPrince();
        double[] again = repeat.Adaptive(Pendulum, pendulum0, 0.0, 10.0, 0.05, 1e-10, 1e-12, out _);
        Check(MaxDiff(again, adaptive) == 0.0 && repeat.AcceptedSteps == pendulumSteps,
            "identical run reproduces state and step count exactly");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
