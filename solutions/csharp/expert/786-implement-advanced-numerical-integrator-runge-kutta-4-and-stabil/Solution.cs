// Task 786 - Implement advanced numerical integrator (Runge-Kutta 4) and stability tests. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Classical fourth-order Runge-Kutta for y' = f(t, y), y(t0) = y0:
//
//      k1 = f(t, y)                y_{n+1} = y_n + h/6 * (k1 + 2 k2 + 2 k3 + k4)
//      k2 = f(t + h/2, y + h/2 k1)
//      k3 = f(t + h/2, y + h/2 k2)
//      k4 = f(t + h,   y + h k3)
//
// Four stages per step, local truncation error O(h^5) and *global* error O(h^4), which the order test measures
// directly: refining h by two must cut the endpoint error by (about) 16, and log2(e_h / e_{h/2}) must approach 4.
// The step is written once for a general vector field, so the same code integrates scalar ODEs, linear systems
// and the harmonic oscillator.
//
// Everything the method is judged by is derived, not just asserted:
//   * linear test equation y' = lambda y. One RK4 step is a polynomial in z = lambda h, the stability function
//     R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24, so the numeric amplification computed by actually marching the
//     integrator must equal R(z) to machine precision - an oracle that needs no reference trajectory.
//   * stability region. RK4 is *not* A-stable: |R(z)| <= 1 only inside a bounded region. Two boundaries are
//     located by bisection and compared with their closed forms: the real axis crossing at
//     z = -2.7852935634050738... and the imaginary axis crossing at |Im z| = sqrt(8) = 2.8284271247461903,
//     the latter because |R(iy)|^2 - 1 = y^6 (y^2 - 8) / 576 exactly.
//   * local error constant: R(z) - e^z = z^5/120 + O(z^6), so (R(z) - e^z)/z^5 -> 1/120 as z -> 0.
//   * exponential of a 2x2 matrix computed independently by scaling-and-squaring Taylor series, used as the
//     reference for a coupled linear system.
//
// Invariants asserted: the integrator never mutates the caller's state vector, a zero-length interval returns
// the initial state bit-for-bit, and the harmonic-oscillator energy error after n steps is bounded by the
// accumulated |R(i h)|^6 term (RK4 is dissipative on the imaginary axis, never unstable).
//
// Documented simplification: the fields integrated here are small analytic test problems with known closed
// forms; no step-size adaptation (task 787) and no implicit/stiff handling (task 788) is attempted, only the
// fixed-step classic RK4 plus the stability analysis around it. Deterministic, single-threaded, BCL only.
using System;

internal delegate double[] VectorField(double t, double[] y);

internal static class Rk4
{
    /** One classical Runge-Kutta 4 step; the caller's state is left untouched. */
    internal static double[] Step(VectorField f, double t, double[] y, double h)
    {
        int n = y.Length;
        double[] k1 = f(t, y);
        double[] y2 = AddScaled(y, k1, h / 2.0);
        double[] k2 = f(t + h / 2.0, y2);
        double[] y3 = AddScaled(y, k2, h / 2.0);
        double[] k3 = f(t + h / 2.0, y3);
        double[] y4 = AddScaled(y, k3, h);
        double[] k4 = f(t + h, y4);

        var next = new double[n];
        for (int i = 0; i < n; i++)
        {
            next[i] = y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        return next;
    }

    internal static double[] Integrate(VectorField f, double[] y0, double t0, double t1, double h, out int steps)
    {
        if (h <= 0.0) throw new ArgumentException("step size must be positive", nameof(h));
        if (t1 < t0) throw new ArgumentException("interval must be forward in time", nameof(t1));
        double span = t1 - t0;
        if (h > span && span > 0.0) throw new ArgumentException("step size exceeds the integration interval", nameof(h));

        double[] y = (double[])y0.Clone();
        if (span == 0.0) { steps = 0; return y; }
        int count = (int)Math.Ceiling(span / h - 1e-12);
        double actual = span / count;
        for (int s = 0; s < count; s++)
        {
            y = Step(f, t0 + s * actual, y, actual);
        }
        steps = count;
        return y;
    }

    /** Amplification of the linear test equation after one step with z = lambda * h (real z). */
    internal static double StabilityFunction(double z) => 1.0 + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 24.0;

    internal static double StabilityMagnitude(double re, double im)
    {
        // R(z) evaluated at the complex point z = re + i im.
        double r = 1.0 + re + (re * re - im * im) / 2.0;
        double i = im + re * im;
        // z^3 = (re + i im)^3
        double re2 = re * re - im * im, im2 = 2.0 * re * im;
        double re3 = re2 * re - im2 * im, im3 = re2 * im + im2 * re;
        r += re3 / 6.0;
        i += im3 / 6.0;
        // z^4 = (z^2)^2
        double re4 = re2 * re2 - im2 * im2, im4 = 2.0 * re2 * im2;
        r += re4 / 24.0;
        i += im4 / 24.0;
        return Math.Sqrt(r * r + i * i);
    }

    /** Bisection for the rightmost real point where |R(z)| = 1, z < 0. */
    internal static double RealStabilityBoundary()
    {
        double lo = -2.0, hi = -3.0;                       // |R(lo)| < 1, |R(hi)| > 1
        for (int it = 0; it < 200; it++)
        {
            double mid = 0.5 * (lo + hi);
            if (Math.Abs(StabilityFunction(mid)) > 1.0) hi = mid; else lo = mid;
        }
        return 0.5 * (lo + hi);
    }

    /** Bisection for the imaginary axis crossing where |R(i y)| = 1, y > 0. */
    internal static double ImaginaryStabilityBoundary()
    {
        double lo = 1.0, hi = 4.0;                          // |R(i lo)| < 1, |R(i hi)| > 1
        for (int it = 0; it < 200; it++)
        {
            double mid = 0.5 * (lo + hi);
            if (StabilityMagnitude(0.0, mid) > 1.0) hi = mid; else lo = mid;
        }
        return 0.5 * (lo + hi);
    }

    /** expm(A t) for a 2x2 matrix by scaling and squaring with a 30-term Taylor series (independent oracle). */
    internal static double[,] MatrixExp2(double[,] a, double t, out double scalingError)
    {
        double norm = 0.0;
        for (int i = 0; i < 2; i++)
        {
            double row = 0.0;
            for (int j = 0; j < 2; j++) row += Math.Abs(a[i, j]);
            if (row > norm) norm = row;
        }
        int squarings = 0;
        double scale = Math.Abs(t) * norm;
        while (scale > 0.5) { scale /= 2.0; squarings++; }

        double s = t / Math.Pow(2.0, squarings);
        var result = new double[2, 2] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var term = new double[2, 2] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        double lastTerm = 0.0;
        for (int n = 1; n <= 30; n++)
        {
            term = Scale(Multiply(term, a), s / n);
            result = Add(result, term);
            lastTerm = Math.Abs(term[0, 0]) + Math.Abs(term[0, 1]) + Math.Abs(term[1, 0]) + Math.Abs(term[1, 1]);
        }
        for (int q = 0; q < squarings; q++) result = Multiply(result, result);
        scalingError = lastTerm;                            // residual of the truncated series, for diagnostics
        return result;
    }

    private static double[,] Multiply(double[,] x, double[,] y)
    {
        var z = new double[2, 2];
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                z[i, j] = x[i, 0] * y[0, j] + x[i, 1] * y[1, j];
            }
        }
        return z;
    }

    private static double[,] Scale(double[,] x, double f)
    {
        var z = new double[2, 2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) z[i, j] = x[i, j] * f;
        return z;
    }

    private static double[,] Add(double[,] x, double[,] y)
    {
        var z = new double[2, 2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) z[i, j] = x[i, j] + y[i, j];
        return z;
    }

    internal static double[] AddScaled(double[] y, double[] k, double h)
    {
        var r = new double[y.Length];
        for (int i = 0; i < y.Length; i++) r[i] = y[i] + h * k[i];
        return r;
    }

    internal static double MaxAbs(double[] v)
    {
        double m = 0.0;
        foreach (double x in v) m = Math.Max(m, Math.Abs(x));
        return m;
    }
}

public static class Solution
{
    private static double[] Scalar(double lambda) => new[] { lambda };

    private static VectorField LinearScalar(double lambda) => (t, y) => new[] { lambda * y[0] };

    private static VectorField Logistic => (t, y) => new[] { y[0] * (1.0 - y[0]) };

    private static VectorField Oscillator => (t, y) => new[] { y[1], -y[0] };

    /** Reference trajectory by matrix exponential of the 2x2 companion matrix (independent of the RK4 path). */
    private static VectorField CoupledSystem(double[,] a) => (t, y) => new[]
    {
        a[0, 0] * y[0] + a[0, 1] * y[1],
        a[1, 0] * y[0] + a[1, 1] * y[1]
    };

    private static double EndpointError(VectorField f, double[] y0, double t0, double t1, double h, Func<double, double[]> exact, out int steps)
    {
        double[] numeric = Rk4.Integrate(f, y0, t0, t1, h, out steps);
        double[] truth = exact(t1);
        double worst = 0.0;
        for (int i = 0; i < truth.Length; i++) worst = Math.Max(worst, Math.Abs(numeric[i] - truth[i]));
        return worst;
    }

    public static void Main()
    {
        var y0 = new double[] { 1.0 };

        // 1-2. Measured order of convergence on the exponential and on a nonlinear field with a closed form.
        double e01 = EndpointError(LinearScalar(1.0), y0, 0.0, 1.0, 0.1, t => new[] { Math.Exp(t) }, out int s01);
        double e005 = EndpointError(LinearScalar(1.0), y0, 0.0, 1.0, 0.05, t => new[] { Math.Exp(t) }, out _);
        double observedOrder = Math.Log(e01 / e005) / Math.Log(2.0);
        double relative01 = e01 / Math.E;
        Console.WriteLine("  y'=y  error h=0.1: " + e01.ToString("E3") + "  h=0.05: " + e005.ToString("E3")
            + "  observed order=" + observedOrder.ToString("F3") + "  relative=" + relative01.ToString("E3")
            + "  steps=" + s01);
        Check(observedOrder > 3.9 && observedOrder < 4.3, "RK4 shows 4th order convergence on y'=y");
        Check(relative01 < 2e-6, "relative error on y'=y is below 2e-6");

        double l01 = EndpointError(Logistic, new[] { 0.5 }, 0.0, 5.0, 0.05, t => new[] { 1.0 / (1.0 + Math.Exp(-t)) }, out _);
        double l0025 = EndpointError(Logistic, new[] { 0.5 }, 0.0, 5.0, 0.025, t => new[] { 1.0 / (1.0 + Math.Exp(-t)) }, out _);
        double logisticOrder = Math.Log(l01 / l0025) / Math.Log(2.0);
        Console.WriteLine("  logistic y'=y(1-y) error h=0.05: " + l01.ToString("E3")
            + "  observed order=" + logisticOrder.ToString("F3"));
        Check(logisticOrder > 3.8, "nonlinear logistic problem converges at order ~4");
        Check(l01 < 1e-8, "logistic endpoint error below 1e-8");

        // 3-4. The one-step amplification for y' = lambda y must reproduce the stability polynomial exactly.
        double worstAmpError = 0.0;
        foreach (double z in new[] { -0.5, -1.0, -2.0, -2.5, 0.75, 1.0 })
        {
            int one;
            double[] numeric = Rk4.Integrate(LinearScalar(z), new[] { 1.0 }, 0.0, 1.0, 1.0, out one);
            double expected = Rk4.StabilityFunction(z);
            worstAmpError = Math.Max(worstAmpError, Math.Abs(numeric[0] - expected));
            Check(one == 1, "single step taken for z=" + z);
        }
        Check(worstAmpError < 1e-13, "one-step amplification equals R(z) exactly (max dev "
            + worstAmpError.ToString("E3") + ")");

        // 5. Leading local-error constant, extracted by linear Richardson extrapolation: the truncation error
        //    e^z - R(z) = z^5/120 + z^6/720 + ... is linear in z, so 2*(val at z/2) - (val at z) removes it.
        double v1 = (Math.Exp(0.1) - Rk4.StabilityFunction(0.1)) / Math.Pow(0.1, 5);
        double v2 = (Math.Exp(0.05) - Rk4.StabilityFunction(0.05)) / Math.Pow(0.05, 5);
        double extrapolated = 2.0 * v2 - v1;
        Console.WriteLine("  local error constant: " + extrapolated.ToString("F9") + " vs 1/120 = "
            + (1.0 / 120.0).ToString("F9"));
        Check(Math.Abs(extrapolated - 1.0 / 120.0) < 2e-5, "e^z - R(z) ~ z^5/120 (extrapolated " + extrapolated.ToString("F9") + ")");

        // 6-8. Stability boundaries against their closed forms.
        double realBoundary = Rk4.RealStabilityBoundary();
        double imagBoundary = Rk4.ImaginaryStabilityBoundary();
        Console.WriteLine("  real stability boundary=" + realBoundary.ToString("F10")
            + " (closed form -2.7852935634024)  imaginary boundary=" + imagBoundary.ToString("F10") + " (sqrt 8)");
        Check(Math.Abs(realBoundary + 2.7852935634050738) < 1e-9, "real axis stability boundary = -2.7852935634");
        Check(Math.Abs(imagBoundary - Math.Sqrt(8.0)) < 1e-9, "imaginary axis stability boundary = sqrt(8)");
        Check(Math.Abs(Math.Abs(Rk4.StabilityFunction(realBoundary)) - 1.0) < 1e-12, "|R| = 1 at the real boundary");

        // 9. Inside the region the linear test equation stays bounded; outside it must grow.
        var damped = new[] { 1.0 };
        double[] inside = Rk4.Integrate(LinearScalar(-10.0), damped, 0.0, 100.0 * 0.27, 0.27, out _);
        double[] outside = Rk4.Integrate(LinearScalar(-10.0), damped, 0.0, 100.0 * 0.29, 0.29, out _);
        Check(Math.Abs(inside[0]) < 2.0, "z-h = -2.7 stays bounded over 100 steps");
        Check(Math.Abs(outside[0]) > 10.0, "z-h = -2.9 grows - outside the stability region");
        Check(Math.Abs(Rk4.StabilityMagnitude(0.0, 0.0) - 1.0) < 1e-15, "R(0) = 1");

        // 10-11. Harmonic oscillator: energy drift is the sixth-order damping term and stays tiny.
        double[] state = { 1.0, 0.0 };
        double energy0 = state[0] * state[0] + state[1] * state[1];
        double h = 0.05;
        double t = 0.0;
        for (int i = 0; i < 200; i++)
        {
            state = Rk4.Step(Oscillator, t, state, h);
            t += h;
        }
        double energyDrift = Math.Abs(state[0] * state[0] + state[1] * state[1] - energy0);
        double phaseError = Math.Max(Math.Abs(state[0] - Math.Cos(t)), Math.Abs(state[1] + Math.Sin(t)));
        Console.WriteLine("  oscillator t=" + t.ToString("F2") + " energy drift=" + energyDrift.ToString("E3")
            + " phase error=" + phaseError.ToString("E3"));
        Check(energyDrift < 1e-6, "oscillator energy drift below 1e-6 over 200 steps");
        Check(phaseError < 1e-5, "oscillator phase error below 1e-5");
        Check(state[0] * state[0] + state[1] * state[1] <= energy0, "RK4 damps rather than amplifies oscillator energy");

        // 12-13. Coupled linear system against an independently computed matrix exponential.
        var a = new double[2, 2] { { -15.0, 10.0 }, { 10.0, -15.0 } };   // eigenvalues -5 and -25
        double[,] reference = Rk4.MatrixExp2(a, 1.0, out double seriesResidual);
        double[] numericCoupled = Rk4.Integrate(CoupledSystem(a), new[] { 1.0, 0.0 }, 0.0, 1.0, 0.02, out int coupledSteps);
        double coupledError = Math.Max(Math.Abs(numericCoupled[0] - reference[0, 0]),
            Math.Abs(numericCoupled[1] - reference[1, 0]));
        Console.WriteLine("  coupled system (lambda=-5,-25) h=0.02 steps=" + coupledSteps
            + " error vs expm=" + coupledError.ToString("E3") + " series residual=" + seriesResidual.ToString("E3"));
        Check(coupledError < 1e-5, "coupled linear system matches the matrix exponential reference");
        Check(seriesResidual < 1e-16, "Taylor series for expm converged to machine precision");

        // 14. RK4 at the same step size is dramatically better than the midpoint rule.
        int midSteps;
        double[] midpoint = Midpoint(LinearScalar(1.0), y0, 0.0, 1.0, 0.1, out midSteps);
        double midpointError = Math.Abs(midpoint[0] - Math.E);
        Check(e01 * 100.0 < midpointError, "RK4 beats midpoint rule at equal step size ("
            + e01.ToString("E3") + " vs " + midpointError.ToString("E3") + ")");

        // 15-17. Boundary behaviour and rejection of invalid input.
        double[] untouched = { 3.0, -1.0 };
        Rk4.Integrate(Oscillator, untouched, 0.0, 1.0, 0.1, out _);
        Check(untouched[0] == 3.0 && untouched[1] == -1.0, "caller state vector is not mutated");

        int zeroSteps;
        double[] same = Rk4.Integrate(Oscillator, untouched, 2.0, 2.0, 1.0, out zeroSteps);
        Check(zeroSteps == 0 && same[0] == 3.0 && same[1] == -1.0, "zero-length interval returns the initial state");

        int singleStep;
        double[] oneShot = Rk4.Integrate(LinearScalar(-1.0), new[] { 1.0 }, 0.0, 0.3, 0.3, out singleStep);
        Check(singleStep == 1 && Math.Abs(oneShot[0] - Rk4.StabilityFunction(-0.3)) < 1e-14,
            "h equal to the interval performs exactly one step equal to R(-0.3)");

        Check(Throws(() => Rk4.Integrate(Oscillator, untouched, 0.0, 1.0, 0.0, out _)), "h = 0 rejected");
        Check(Throws(() => Rk4.Integrate(Oscillator, untouched, 0.0, 1.0, -0.1, out _)), "negative h rejected");
        Check(Throws(() => Rk4.Integrate(Oscillator, untouched, 0.0, 1.0, 2.0, out _)), "h larger than the interval rejected");

        // 18. Long-run stability: 5000 steps of y' = -4y must land on the analytic exponential.
        int longSteps;
        double[] longRun = Rk4.Integrate(LinearScalar(-4.0), new[] { 1.0 }, 0.0, 50.0, 0.01, out longSteps);
        Check(longSteps == 5000 && Math.Abs(longRun[0] - Math.Exp(-200.0)) < 1e-12,
            "5000-step damped run matches exp(-200)");

        Console.WriteLine("PASS");
    }

    private static double[] Midpoint(VectorField f, double[] y0, double t0, double t1, double h, out int steps)
    {
        double[] y = (double[])y0.Clone();
        int count = (int)Math.Round((t1 - t0) / h);
        for (int s = 0; s < count; s++)
        {
            double t = t0 + s * h;
            double[] k1 = f(t, y);
            double[] k2 = f(t + h / 2.0, Rk4.AddScaled(y, k1, h / 2.0));
            for (int i = 0; i < y.Length; i++) y[i] += h * k2[i];
        }
        steps = count;
        return y;
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
