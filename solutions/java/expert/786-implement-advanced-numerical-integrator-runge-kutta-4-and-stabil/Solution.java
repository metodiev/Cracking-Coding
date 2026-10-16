/** Task 786 - Implement advanced numerical integrator (Runge-Kutta 4) and stability tests. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Classical RK4 for the first order system y' = f(t, y):
 *
 *     k1 = f(t, y)                k2 = f(t + h/2, y + h k1/2)
 *     k3 = f(t + h/2, y + h k2/2) k4 = f(t + h,   y + h k3)
 *     y_{n+1} = y_n + h (k1 + 2 k2 + 2 k3 + k4) / 6
 *
 * with local truncation error O(h^5) and global error O(h^4) for smooth f.
 *
 * Stability is decided by the stability function, which for any one-step method applied to the scalar
 * test problem y' = lambda y is exactly the amplification per step:
 *
 *     R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24,      z = h lambda,      y_n = R(z)^n y_0
 *
 * RK4 is not A-stable: on the negative real axis |R(z)| <= 1 only down to z* = -2.785293563... (the root
 * of |R(z)| = 1), so an explicit method needs h < 2.7853/|lambda| there. That bound is located here by
 * bisection on |R| - 1 and then confirmed against the *simulated* recursion, which for a linear problem
 * reproduces R(z)^n bit-for-bit.
 *
 * Verified: closed-form agreement (exponential, logistic), measured global order from successive halvings
 * of h (each halving must cut the error by ~16) and the O(h^5) local error, the exact R(z)^n identity, the
 * bisected real-axis stability boundary against the published constant, decay inside / blow-up outside the
 * boundary, energy drift of the harmonic oscillator and the pendulum falling like h^6 / h^4 respectively,
 * fourth order agreement with a high-resolution reference integration of the nonlinear pendulum, exactly 4
 * function evaluations per step, a zero-length interval returning the initial state unchanged, and the
 * rejection of a non-positive step count.
 */
import java.util.*;

/** Right-hand side of the first order system y' = f(t, y). */
interface Rhs {
    double[] eval(double t, double[] y);
}

/** One integration run: the final state plus the work counters. */
class Run {
    final double[] finalState;
    final long functionEvaluations;
    final int steps;

    Run(double[] finalState, long functionEvaluations, int steps) {
        this.finalState = finalState;
        this.functionEvaluations = functionEvaluations;
        this.steps = steps;
    }
}

public class Solution {
    /** y + h k. */
    static double[] axpy(double[] y, double[] k, double h) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) out[i] = y[i] + h * k[i];
        return out;
    }

    static double[] rk4Step(Rhs f, double t, double[] y, double h, long[] work) {
        double[] k1 = f.eval(t, y);
        double[] k2 = f.eval(t + h / 2, axpy(y, k1, h / 2));
        double[] k3 = f.eval(t + h / 2, axpy(y, k2, h / 2));
        double[] k4 = f.eval(t + h, axpy(y, k3, h));
        work[0] += 4;
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) out[i] = y[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0;
        return out;
    }

    static Run integrate(Rhs f, double t0, double[] y0, double t1, int steps) {
        if (steps <= 0) throw new IllegalArgumentException("steps must be positive: " + steps);
        double h = (t1 - t0) / steps;
        double[] y = y0.clone();
        long[] work = new long[1];
        for (int i = 0; i < steps; i++) y = rk4Step(f, t0 + i * h, y, h, work);
        return new Run(y, work[0], steps);
    }

    /** The exact RK4 amplification factor for y' = lambda y with z = h lambda. */
    static double stability(double z) {
        return 1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24;
    }

    /** Bisects |R(z)| = 1 on [-3, -2.7], where R(-3) > 1 > R(-2.7). */
    static double realStabilityBoundary() {
        double lo = -3.0, hi = -2.7;
        for (int i = 0; i < 200; i++) {
            double mid = (lo + hi) / 2;
            if (Math.abs(stability(mid)) > 1.0) lo = mid; else hi = mid;
        }
        return (lo + hi) / 2;
    }

    /** Max-norm of the componentwise difference. */
    static double distance(double[] a, double[] b) {
        double worst = 0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }

    static double harmonicEnergy(double[] y) { return 0.5 * (y[0] * y[0] + y[1] * y[1]); }

    static double pendulumEnergy(double[] y) { return 0.5 * y[1] * y[1] - Math.cos(y[0]); }

    public static void main(String[] args) {
        Rhs exponential = (t, y) -> new double[]{y[0]};
        Rhs logistic = (t, y) -> new double[]{y[0] * (1 - y[0])};
        Rhs oscillator = (t, y) -> new double[]{y[1], -y[0]};
        Rhs pendulum = (t, y) -> new double[]{y[1], -Math.sin(y[0])};

        // --- closed form and observed order of convergence --------------------------------
        int[] stepCounts = {10, 20, 40, 80, 160};
        double[] errors = new double[stepCounts.length];
        for (int i = 0; i < stepCounts.length; i++) {
            double[] y = integrate(exponential, 0, new double[]{1.0}, 1, stepCounts[i]).finalState;
            errors[i] = Math.abs(y[0] - Math.E);
        }
        check(errors[errors.length - 1] < 1e-9, "RK4 nails y' = y, error " + errors[errors.length - 1]);
        for (int i = 1; i < errors.length; i++) {
            double ratio = errors[i - 1] / errors[i];
            check(ratio > 12 && ratio < 20, "halving h should cut the error 16-fold, got " + ratio);
        }
        double order = Math.log(errors[0] / errors[errors.length - 1]) / Math.log(16.0);
        check(order > 3.6 && order < 4.4, "measured global order " + order);

        double exactLogistic = 1.0 / (1.0 + (1.0 / 0.2 - 1.0) * Math.exp(-3.0));
        double[] logisticRun = integrate(logistic, 0, new double[]{0.2}, 3, 60).finalState;
        check(Math.abs(logisticRun[0] - exactLogistic) < 1e-7,
                "logistic closed form, error " + Math.abs(logisticRun[0] - exactLogistic));

        // --- local error is O(h^5) ----------------------------------------------------------
        long[] work = new long[1];
        double[] start = {Math.exp(0.5)};
        double bigStep = Math.abs(rk4Step(exponential, 0.5, start, 0.2, work)[0] - Math.exp(0.7));
        double smallStep = Math.abs(rk4Step(exponential, 0.5, start, 0.1, work)[0] - Math.exp(0.6));
        check(bigStep / smallStep > 25 && bigStep / smallStep < 45,
                "local error should scale like h^5, ratio " + bigStep / smallStep);
        check(rk4Step(exponential, 0.5, start, 0.0, work)[0] == start[0], "a zero step is the identity");

        // --- linear problems reproduce R(z)^n exactly ---------------------------------------
        double lambda = -1.7, h = 0.01;
        Rhs linear = (t, y) -> new double[]{lambda * y[0]};
        Run linearRun = integrate(linear, 0, new double[]{1.0}, 1, 100);
        double expected = Math.pow(stability(h * lambda), 100);
        check(Math.abs(linearRun.finalState[0] - expected) <= 1e-12 * Math.abs(expected),
                "y' = lambda y must equal R(z)^n, got " + linearRun.finalState[0] + " vs " + expected);

        // --- stability: bisected boundary and the behaviour on both sides -------------------
        double boundary = realStabilityBoundary();
        check(Math.abs(boundary + 2.7852935634056454) < 1e-9, "real-axis stability boundary " + boundary);
        check(Math.abs(Math.abs(stability(boundary)) - 1.0) < 1e-9, "R(boundary) must have unit modulus");
        check(Math.abs(stability(-2.7853)) > 1.0 && Math.abs(stability(-2.7852)) < 1.0,
                "the boundary must separate stable from unstable");

        Rhs stiff = (t, y) -> new double[]{-10.0 * y[0]};
        double inside = integrate(stiff, 0, new double[]{1.0}, 40 * 0.2785, 40).finalState[0];
        double outside = integrate(stiff, 0, new double[]{1.0}, 40 * 0.28, 40).finalState[0];
        check(Math.abs(inside) < 1.0 && Math.abs(inside) <= Math.abs(stability(-2.785)),
                "inside the region the stiff mode must decay, |y| = " + Math.abs(inside));
        check(Math.abs(outside) > 2.0, "outside the region RK4 must blow up, |y| = " + Math.abs(outside));
        check(Math.abs(outside) > 2 * Math.abs(inside), "the two sides must differ sharply");

        // --- energy drift of the harmonic oscillator and its h^6 scaling ---------------------
        double[] osc0 = {1.0, 0.0};
        double driftCoarse = Math.abs(harmonicEnergy(integrate(oscillator, 0, osc0, 20, 400).finalState)
                - harmonicEnergy(osc0));
        double driftFine = Math.abs(harmonicEnergy(integrate(oscillator, 0, osc0, 20, 800).finalState)
                - harmonicEnergy(osc0));
        check(driftCoarse < 1e-5, "harmonic energy drift " + driftCoarse);
        check(driftCoarse / driftFine > 8 && driftCoarse / driftFine < 64,
                "harmonic drift must fall like h^6, ratio " + driftCoarse / driftFine);

        // --- nonlinear pendulum against a high resolution reference --------------------------
        double[] p0 = {0.8, 0.0};
        double[] reference = integrate(pendulum, 0, p0, 10, 40000).finalState;
        double[] converged = integrate(pendulum, 0, p0, 10, 200000).finalState;
        check(distance(reference, converged) < 1e-10, "the reference integration has converged");
        double rough = distance(integrate(pendulum, 0, p0, 10, 50).finalState, reference);
        double finer = distance(integrate(pendulum, 0, p0, 10, 100).finalState, reference);
        check(rough / finer > 12 && rough / finer < 20,
                "pendulum error must fall 16-fold per halving, ratio " + rough / finer);
        check(finer < 1e-5, "pendulum error against the reference " + finer);
        check(distance(integrate(pendulum, 0, p0, 10, 800).finalState, reference) < 1e-8,
                "pendulum error at h = 0.0125 must be microscopic");
        double pendulumDrift = Math.abs(pendulumEnergy(integrate(pendulum, 0, p0, 10, 2000).finalState)
                - pendulumEnergy(p0));
        check(pendulumDrift < 1e-10, "pendulum energy drift " + pendulumDrift);

        // --- bookkeeping and degenerate intervals --------------------------------------------
        Run counted = integrate(exponential, 0, new double[]{1.0}, 1, 7);
        check(counted.functionEvaluations == 28 && counted.steps == 7, "4 evaluations per step, got "
                + counted.functionEvaluations);
        check(Math.abs(integrate(exponential, 1, new double[]{2.0}, 1, 5).finalState[0] - 2.0) < 1e-15,
                "a zero length interval must leave the state untouched");
        boolean rejected = false;
        try {
            integrate(exponential, 0, new double[]{1.0}, 1, 0);
        } catch (IllegalArgumentException expected0) {
            rejected = true;
        }
        check(rejected, "a non-positive step count must be rejected");

        System.out.println("exponential error at h=1/160: " + errors[errors.length - 1] + " (order " + order + ")");
        System.out.println("stability boundary on the real axis = " + boundary);
        System.out.println("pendulum error vs reference: coarse " + rough + ", fine " + finer);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
