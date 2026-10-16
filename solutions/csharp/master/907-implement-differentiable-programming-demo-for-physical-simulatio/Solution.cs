// Task 907 - Implement differentiable programming demo for physical simulation parameter fitting.
// (Tier: Master, Beyond - Machine Learning & Simulation)
//
// A damped spring-mass simulation x'' = -k x - c v is integrated forward with explicit Euler, and the two
// physical parameters (stiffness k, damping c) are recovered from an observed trajectory by gradient
// descent.  The derivatives are not approximated: the simulator is written twice, once over doubles and
// once over a forward-mode dual number (value + derivative with the chain rule implemented for +, -, *, /,
// sin and exp), so the loss derivative with respect to a parameter is obtained *through* the integration
// loop.  Two gradient components cost two dual runs, one seeded with dk = 1 and one with dc = 1.
//
// The optimizer is a backtracking steepest descent: it takes the gradient step and halves the step until
// the loss actually decreases, which makes descent strictly monotone and lets the run survive a badly
// scaled first step.  Convergence is declared on the gradient norm, and the result is compared against the
// parameters that generated the data.
//
// Simplifications: explicit Euler with a fixed step (no adaptive integrator), two parameters, noiseless
// observations, and a hand-rolled dual number instead of a general autodiff graph.  Verified: the dual
// gradient against central finite differences at several points, chain-rule identities against closed
// forms, monotone descent, the near-zero gradient at the recovered optimum, the recovered parameters,
// degenerate-data and oversized-step boundaries, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    /** Forward-mode dual number: value plus the derivative with respect to the seeded parameter. */
    internal struct Dual
    {
        internal double Value;
        internal double Derivative;

        internal Dual(double value, double derivative)
        {
            Value = value;
            Derivative = derivative;
        }

        public static Dual operator +(Dual a, Dual b) => new Dual(a.Value + b.Value, a.Derivative + b.Derivative);

        public static Dual operator -(Dual a, Dual b) => new Dual(a.Value - b.Value, a.Derivative - b.Derivative);

        public static Dual operator -(Dual a) => new Dual(-a.Value, -a.Derivative);

        public static Dual operator -(Dual a, double b) => new Dual(a.Value - b, a.Derivative);

        public static Dual operator *(Dual a, Dual b) =>
            new Dual(a.Value * b.Value, a.Derivative * b.Value + a.Value * b.Derivative);

        public static Dual operator *(Dual a, double scalar) => new Dual(a.Value * scalar, a.Derivative * scalar);

        public static Dual operator /(Dual a, Dual b)
        {
            double denominator = b.Value * b.Value;
            return new Dual(a.Value / b.Value, (a.Derivative * b.Value - a.Value * b.Derivative) / denominator);
        }

        internal static Dual Sin(Dual a) => new Dual(Math.Sin(a.Value), Math.Cos(a.Value) * a.Derivative);

        internal static Dual Exp(Dual a) => new Dual(Math.Exp(a.Value), Math.Exp(a.Value) * a.Derivative);

        internal static Dual Sqrt(Dual a) =>
            new Dual(Math.Sqrt(a.Value), a.Derivative / (2.0 * Math.Sqrt(a.Value)));
    }

    private const int Steps = 120;
    private const double Dt = 0.02;

    /** Explicit Euler integration of the damped oscillator; returns the position trace. */
    private static double[] Simulate(double k, double c)
    {
        double[] trace = new double[Steps + 1];
        double x = 1.0;
        double v = 0.0;
        trace[0] = x;
        for (int i = 1; i <= Steps; i++)
        {
            double acceleration = -k * x - c * v;
            v += acceleration * Dt;
            x += v * Dt;
            trace[i] = x;
        }
        return trace;
    }

    /** The same loop over dual numbers: the derivative flows through the integrator. */
    private static Dual SimulateDual(Dual k, Dual c, out Dual[] trace)
    {
        trace = new Dual[Steps + 1];
        Dual x = new Dual(1.0, 0.0);
        Dual v = new Dual(0.0, 0.0);
        trace[0] = x;
        for (int i = 1; i <= Steps; i++)
        {
            Dual acceleration = -(k * x) - (c * v);
            v = v + acceleration * Dt;
            x = x + v * Dt;
            trace[i] = x;
        }
        return x;
    }

    private static double Loss(double k, double c, double[] observed)
    {
        double[] trace = Simulate(k, c);
        double loss = 0.0;
        for (int i = 1; i <= Steps; i++)
        {
            double residual = trace[i] - observed[i];
            loss += residual * residual;
        }
        return loss;
    }

    /** d(loss)/d(parameter): the dual run seeded on that one parameter. */
    private static Dual LossDual(double k, double c, double[] observed, bool seedK)
    {
        Dual kDual = new Dual(k, seedK ? 1.0 : 0.0);
        Dual cDual = new Dual(c, seedK ? 0.0 : 1.0);
        SimulateDual(kDual, cDual, out Dual[] trace);
        Dual loss = new Dual(0.0, 0.0);
        for (int i = 1; i <= Steps; i++)
        {
            Dual residual = trace[i] - observed[i];
            loss = loss + residual * residual;
        }
        return loss;
    }

    private static (double K, double C) Gradient(double k, double c, double[] observed) =>
        (LossDual(k, c, observed, true).Derivative, LossDual(k, c, observed, false).Derivative);

    /** Central finite differences - the independent oracle for the dual gradients. */
    private static (double K, double C) FiniteDifference(double k, double c, double[] observed, double h) =>
        ((Loss(k + h, c, observed) - Loss(k - h, c, observed)) / (2 * h),
         (Loss(k, c + h, observed) - Loss(k, c - h, observed)) / (2 * h));

    /** Backtracking steepest descent: every accepted step strictly lowers the loss. */
    private static (double K, double C, int Iterations, double Loss, double GradientNorm) Fit(
        double k, double c, double[] observed, int maxIterations)
    {
        double current = Loss(k, c, observed);
        int iteration = 0;
        double norm = 0.0;
        for (; iteration < maxIterations; iteration++)
        {
            (double gk, double gc) = Gradient(k, c, observed);
            norm = Math.Sqrt(gk * gk + gc * gc);
            if (norm < 1e-9) break;
            double step = 0.5;
            bool improved = false;
            for (int shrink = 0; shrink < 40 && !improved; shrink++)
            {
                double kNext = k - step * gk;
                double cNext = c - step * gc;
                double candidate = Loss(kNext, cNext, observed);
                if (candidate < current)
                {
                    k = kNext;
                    c = cNext;
                    current = candidate;
                    improved = true;
                }
                else
                {
                    step *= 0.5;
                }
            }
            if (!improved) break;
        }
        return (k, c, iteration, current, norm);
    }

    public static void Main()
    {
        const double trueK = 3.6;
        const double trueC = 0.55;
        double[] observed = Simulate(trueK, trueC);

        // ---- 1. the dual gradient against central finite differences -------------------------------
        double worst = 0.0;
        int probes = 0;
        foreach ((double k, double c) in new[] { (1.0, 0.1), (3.6, 0.55), (8.0, 1.4), (0.4, 2.2) })
        {
            (double dk, double dc) = Gradient(k, c, observed);
            (double fk, double fc) = FiniteDifference(k, c, observed, 1e-5);
            worst = Math.Max(worst, Math.Abs(dk - fk) / (1 + Math.Abs(fk)));
            worst = Math.Max(worst, Math.Abs(dc - fc) / (1 + Math.Abs(fc)));
            probes++;
        }
        Check(probes == 4 && worst < 1e-4,
            "dual gradients match central differences at 4 points (worst relative gap " +
            worst.ToString("E3", CultureInfo.InvariantCulture) + ")");

        // ---- 2. chain rule identities --------------------------------------------------------------
        Dual z = new Dual(0.7, 1.0);
        Dual f = Dual.Sin(z * z) * Dual.Exp(z);
        double analytic = Math.Cos(0.49) * 2 * 0.7 * Math.Exp(0.7) + Math.Sin(0.49) * Math.Exp(0.7);
        Check(Math.Abs(f.Value - Math.Sin(0.49) * Math.Exp(0.7)) < 1e-12, "dual sin*exp value matches the closed form");
        Check(Math.Abs(f.Derivative - analytic) < 1e-12, "dual sin*exp derivative matches the product rule");
        Check(Math.Abs(Dual.Sqrt(new Dual(9.0, 2.0)).Derivative - 1.0 / 3.0) < 1e-12, "dual sqrt follows the power rule");
        Check(Math.Abs((new Dual(3.0, 1.0) / new Dual(5.0, 2.0)).Derivative - (1 * 5 - 3 * 2) / 25.0) < 1e-12,
            "dual division follows the quotient rule");

        // ---- 3. fitting the physical parameters ----------------------------------------------------
        (double kFit, double cFit, int iterations, double finalLoss, double norm) = Fit(1.0, 2.0, observed, 200);
        double startLoss = Loss(1.0, 2.0, observed);
        Check(finalLoss < startLoss * 1e-6,
            "descent reduced the loss by more than six orders of magnitude (" +
            startLoss.ToString("E3", CultureInfo.InvariantCulture) + " -> " + finalLoss.ToString("E3", CultureInfo.InvariantCulture) + ")");
        Check(Math.Abs(kFit - trueK) < 1e-3 && Math.Abs(cFit - trueC) < 1e-3,
            "the kinematic parameters were recovered (k=" + kFit.ToString("F6", CultureInfo.InvariantCulture) +
            ", c=" + cFit.ToString("F6", CultureInfo.InvariantCulture) + ")");
        Check(norm < 1e-6, "the gradient vanishes at the recovered optimum (" + norm.ToString("E3", CultureInfo.InvariantCulture) + ")");
        Check(iterations > 0 && iterations < 200, "the descent converged in " + iterations + " iterations");

        // ---- 4. monotone descent, and survival of a badly scaled first step -------------------------
        double previous = Loss(1.0, 2.0, observed);
        double kCursor = 1.0;
        double cCursor = 2.0;
        int monotoneSteps = 0;
        for (int i = 0; i < 20; i++)
        {
            (double gk, double gc) = Gradient(kCursor, cCursor, observed);
            double step = 0.5;
            double candidate = double.MaxValue;
            for (int shrink = 0; shrink < 40; shrink++)
            {
                candidate = Loss(kCursor - step * gk, cCursor - step * gc, observed);
                if (candidate < previous) break;
                step *= 0.5;
            }
            Check(candidate < previous, "line search always found a strictly decreasing step");
            kCursor -= step * gk;
            cCursor -= step * gc;
            previous = candidate;
            monotoneSteps++;
        }
        Check(monotoneSteps == 20, "twenty successive accepted steps each lowered the loss");
        (double hugeK, double hugeC, _, double hugeLoss, _) = Fit(12.0, 5.0, observed, 200);
        Check(hugeLoss < startLoss && Math.Abs(hugeK - trueK) < 1e-2 && Math.Abs(hugeC - trueC) < 1e-2,
            "a far-away start still converges once the step is shrunk (k=" + hugeK.ToString("F4", CultureInfo.InvariantCulture) +
            ", c=" + hugeC.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 5. boundaries --------------------------------------------------------------------------
        double[] model = Simulate(1.0, 1.0);
        (double zeroK, double zeroC) = Gradient(1.0, 1.0, model);
        Check(Math.Abs(zeroK) < 1e-12 && Math.Abs(zeroC) < 1e-12,
            "the gradient is exactly zero when the model already matches the data");
        Check(Math.Abs(Loss(1.0, 1.0, model)) < 1e-24, "the loss of an exact match is zero");
        Check(model.Length == Steps + 1 && model[0] == 1.0, "the trace starts at the initial displacement");
        (double stuckK, double stuckC, int stuckIterations, _, _) = Fit(trueK, trueC, observed, 200);
        Check(stuckIterations == 0 && Math.Abs(stuckK - trueK) < 1e-12 && Math.Abs(stuckC - trueC) < 1e-12,
            "starting exactly at the optimum takes no step at all");

        // ---- 6. reproducibility ----------------------------------------------------------------------
        (double kAgain, double cAgain, int iterationsAgain, double lossAgain, _) = Fit(1.0, 2.0, observed, 200);
        Check(Math.Abs(kAgain - kFit) < 1e-15 && Math.Abs(cAgain - cFit) < 1e-15 && iterationsAgain == iterations &&
            lossAgain == finalLoss, "the fit is bit-for-bit reproducible");

        Console.WriteLine("autodiff: " + probes + " dual gradients agree with central differences (worst gap " +
            worst.ToString("E3", CultureInfo.InvariantCulture) + "), chain-rule identities verified");
        Console.WriteLine("fit: k 1.0 -> " + kFit.ToString("F6", CultureInfo.InvariantCulture) + " (true " + trueK +
            "), c 2.0 -> " + cFit.ToString("F6", CultureInfo.InvariantCulture) + " (true " + trueC + ") in " +
            iterations + " backtracking steps");
        Console.WriteLine("loss: " + startLoss.ToString("E3", CultureInfo.InvariantCulture) + " -> " +
            finalLoss.ToString("E3", CultureInfo.InvariantCulture) + ", final gradient norm " + norm.ToString("E3", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
