/** Task 907 - Implement differentiable programming demo for physical simulation parameter fitting.
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A damped mass-spring oscillator is integrated with explicit Euler:
 *
 *     v[t+1] = v[t] + dt * (-k * x[t] - c * v[t])        x[t+1] = x[t] + dt * v[t]
 *
 * and the task is to recover the unknown stiffness k and damping c from an observed trajectory by minimising
 * L(k, c) = sum_t (x[t] - observed[t])^2. The gradient comes from the *adjoint* (reverse-mode) recursion:
 * one forward sweep, then xbar[t]/vbar[t] accumulate backwards through time, so the cost is independent of
 * the number of parameters.
 *
 * The gradient is cross-checked three ways - central finite differences, the complex-step derivative
 * (perturb k by i*h, keep the imaginary part of the loss, exact to machine precision) and the fact that a
 * noiseless trajectory recorded at the true parameters must have a vanishing gradient. Descent from a wrong
 * start must then land on the ground truth, not merely reduce the loss.
 *
 * Simplification: a one-degree-of-freedom linear oscillator over 200 Euler steps instead of a PDE solver.
 * Invariants verified: adjoint == complex step == finite differences within their own accuracy, gradient ~ 0
 * at the truth, no descent step ever increases the loss, the fitted parameters match the truth, positive
 * damping never increases the energy of the recorded state sequence, free motion (k = c = 0) is exact, and
 * unphysical parameters are rejected before use.
 */
import java.util.*;

/** A parameter pair and its gradient. */
final class Params {
    final double k;
    final double c;
    final double dk;
    final double dc;

    Params(double k, double c, double dk, double dc) {
        this.k = k;
        this.c = c;
        this.dk = dk;
        this.dc = dc;
    }
}

/** Position and velocity sequences produced by one forward run. */
final class Trajectory {
    final double[] x;
    final double[] v;

    Trajectory(double[] x, double[] v) {
        this.x = x;
        this.v = v;
    }
}

public class Solution {
    static final double DT = 0.01;
    static final int STEPS = 200;
    static final double TRUE_K = 4.0;
    static final double TRUE_C = 0.6;

    public static void main(String[] args) {
        // ---- 1. ground truth trajectory ---------------------------------------------------------------
        double[] observed = simulate(TRUE_K, TRUE_C, 1.0, 0.0).x;
        check(observed.length == STEPS, "the trajectory must hold " + STEPS + " samples");
        check(Math.abs(observed[0] - 1.0) < 1e-15, "the trajectory must start at the initial position");
        check(amplitude(observed) < 0.8, "damping must shrink the amplitude well below the start, got "
                + round(amplitude(observed)));

        // ---- 2. the gradient at the true parameters must vanish ---------------------------------------
        Params atTruth = gradient(TRUE_K, TRUE_C, observed);
        check(Math.hypot(atTruth.dk, atTruth.dc) < 1e-6,
                "the noiseless loss is minimised at the truth, gradient " + atTruth.dk + "," + atTruth.dc);

        // ---- 3. oracle A: central finite differences --------------------------------------------------
        Params adjoint = gradient(3.3, 0.45, observed);
        double fdK = centralDifference(3.3, 0.45, observed, true, 1e-6);
        double fdC = centralDifference(3.3, 0.45, observed, false, 1e-6);
        check(close(adjoint.dk, fdK, 1e-3), "adjoint dk " + adjoint.dk + " must match finite differences "
                + fdK);
        check(close(adjoint.dc, fdC, 1e-3), "adjoint dc " + adjoint.dc + " must match finite differences "
                + fdC);

        // ---- 4. oracle B: complex-step differentiation ------------------------------------------------
        double complexK = complexStep(3.3, 0.45, observed, true);
        double complexC = complexStep(3.3, 0.45, observed, false);
        check(close(adjoint.dk, complexK, 1e-9), "adjoint dk must match the complex-step derivative: "
                + adjoint.dk + " vs " + complexK);
        check(close(adjoint.dc, complexC, 1e-9), "adjoint dc must match the complex-step derivative: "
                + adjoint.dc + " vs " + complexC);
        check(Math.abs(complexK) > 1e-3, "the probe point must produce a non-trivial gradient");

        // ---- 5. the adjoint is exact while finite differences are only approximate --------------------
        double truthK = complexStep(2.5, 0.9, observed, true);
        double errorCoarse = Math.abs(centralDifference(2.5, 0.9, observed, true, 1e-2) - truthK);
        double errorFine = Math.abs(centralDifference(2.5, 0.9, observed, true, 1e-3) - truthK);
        check(errorFine < errorCoarse, "halving the finite-difference step must shrink the O(h^2) error: "
                + errorCoarse + " -> " + errorFine);
        check(Math.abs(gradient(2.5, 0.9, observed).dk - truthK) < errorFine,
                "the adjoint must be more accurate than a finite difference of the same cost");

        // ---- 6. parameter fitting by gradient descent with backtracking --------------------------------
        double k = 2.0;
        double c = 1.4;
        double step = 0.05;
        double bestLoss = loss(k, c, observed);
        int decreases = 0;
        int accepted = 0;
        for (int iteration = 0; iteration < 3000; iteration++) {
            Params g = gradient(k, c, observed);
            double alpha = step;
            double nextK = k;
            double nextC = c;
            double nextLoss = bestLoss;
            for (int attempt = 0; attempt < 80; attempt++) {
                nextK = k - alpha * g.dk;
                nextC = c - alpha * g.dc;
                if (valid(nextK, nextC)) {
                    nextLoss = loss(nextK, nextC, observed);
                    if (nextLoss <= bestLoss) break;
                }
                alpha *= 0.5;
            }
            check(nextLoss <= bestLoss + 1e-15, "the descent must not increase the loss at iteration "
                    + iteration + ": " + bestLoss + " -> " + nextLoss);
            accepted++;
            if (nextLoss < bestLoss) decreases++;
            k = nextK;
            c = nextC;
            bestLoss = nextLoss;
            if (alpha > 0 && bestLoss > 1e-12) step = Math.min(1.0, alpha * 1.5);
        }
        check(accepted == 3000, "every iteration must produce a usable step");
        check(decreases >= 100, "the descent must make real progress, decreased " + decreases);
        check(Math.abs(k - TRUE_K) < 1e-9, "recovered stiffness must match the truth: " + k + " vs " + TRUE_K);
        check(Math.abs(c - TRUE_C) < 1e-9, "recovered damping must match the truth: " + c + " vs " + TRUE_C);
        check(bestLoss < 1e-20, "the fitted loss must be essentially zero: " + bestLoss);
        check(Math.hypot(gradient(k, c, observed).dk, gradient(k, c, observed).dc) < 1e-9,
                "the gradient must vanish at the fitted optimum");

        // ---- 7. physics invariants ----------------------------------------------------------------------
        Trajectory damped = simulate(4.0, 0.6, 1.0, 0.0);
        // Exact consequence of one Euler step: the positive drift term is O(dt^2), the damping term is O(dt),
        // so an explicit step can only add this bounded amount of energy.
        boolean boundedDrift = true;
        for (int t = 0; t + 1 < STEPS; t++) {
            double accel = -4.0 * damped.x[t] - 0.6 * damped.v[t];
            double bound = 0.5 * DT * DT * accel * accel + 0.5 * 4.0 * DT * DT * damped.v[t] * damped.v[t];
            if (energyOf(damped, t + 1, 4.0) - energyOf(damped, t, 4.0) > bound + 1e-15) boundedDrift = false;
        }
        check(boundedDrift, "each Euler step may add at most the O(dt^2) drift term to the energy");
        double startEnergy = energyOf(damped, 0, 4.0);
        double endEnergy = energyOf(damped, STEPS - 1, 4.0);
        check(endEnergy < startEnergy / 2, "damping must remove most of the energy over the run, " + startEnergy
                + " -> " + endEnergy);
        double dissipation = 0;
        double drift = 0;
        for (int t = 0; t + 1 < STEPS; t++) {
            double accel = -4.0 * damped.x[t] - 0.6 * damped.v[t];
            dissipation += DT * 0.6 * damped.v[t] * damped.v[t];
            drift += 0.5 * DT * DT * accel * accel + 0.5 * 4.0 * DT * DT * damped.v[t] * damped.v[t];
        }
        check(Math.abs((endEnergy - startEnergy) - (drift - dissipation)) < 1e-9,
                "the discrete energy identity must hold exactly: " + (endEnergy - startEnergy) + " vs "
                        + (drift - dissipation));
        check(dissipation > 0.9 * (startEnergy - endEnergy),
                "at least 90% of the energy loss must be genuine damping work: " + dissipation + " of "
                        + (startEnergy - endEnergy));
        Trajectory free = simulate(0.0, 0.0, 1.0, 2.0);
        boolean exactFree = true;
        for (int t = 0; t < STEPS; t++) {
            if (Math.abs(free.x[t] - (1.0 + 2.0 * DT * t)) > 1e-13) exactFree = false;
            if (Math.abs(free.v[t] - 2.0) > 1e-15) exactFree = false;
        }
        check(exactFree, "with k = c = 0 the Euler step is exact: x(t) = 1 + 2t");
        Trajectory undamped = simulate(4.0, 0.0, 1.0, 0.0);
        check(energyOf(undamped, STEPS - 1, 4.0) < 1.1 * energyOf(undamped, 0, 4.0),
                "explicit Euler on an undamped spring must not blow up the energy");

        // ---- 8. rejected parameters (negative check) ----------------------------------------------------
        check(valid(TRUE_K, TRUE_C), "the true parameters must be accepted");
        check(!valid(-1.0, 0.6), "negative stiffness is unphysical and must be rejected");
        check(!valid(4.0, -0.1), "negative damping would pump energy in and must be rejected");
        check(valid(0.0, 0.0), "frictionless free motion is physical");
        boolean threw = false;
        try {
            simulate(-1.0, -1.0, observed[0], 0.0);
        } catch (IllegalArgumentException ex) {
            threw = true;
        }
        check(threw, "the simulator must refuse unphysical parameters rather than return numbers");
        check(!close(adjoint.dk, truthK, 1e-9), "gradients at different parameters must differ");

        // ---- 9. determinism ------------------------------------------------------------------------------
        check(gradient(1.5, 0.2, observed).dk == gradient(1.5, 0.2, observed).dk,
                "the adjoint must be deterministic");
        check(Arrays.equals(simulate(TRUE_K, TRUE_C, 1.0, 0.0).x, observed),
                "the forward simulation must be deterministic");

        System.out.println("recovered k = " + round(k) + " (truth " + TRUE_K + "), c = " + round(c) + " (truth "
                + TRUE_C + "), final loss " + bestLoss);
        System.out.println("gradient at truth (" + atTruth.dk + ", " + atTruth.dc + "); adjoint vs complex step "
                + Math.abs(adjoint.dk - complexK));
        System.out.println("finite-difference error " + errorCoarse + " (h=1e-2) -> " + errorFine + " (h=1e-3)");
        System.out.println("descent steps that lowered the loss: " + decreases + "/" + accepted);        System.out.println("PASS");
    }

    /** Forward simulation with a guard against unphysical parameters. */
    static Trajectory simulate(double k, double c, double x0, double v0) {
        if (!valid(k, c)) throw new IllegalArgumentException("unphysical parameters k=" + k + ", c=" + c);
        double[] x = new double[STEPS];
        double[] v = new double[STEPS];
        x[0] = x0;
        v[0] = v0;
        for (int t = 1; t < STEPS; t++) {
            v[t] = v[t - 1] + DT * (-k * x[t - 1] - c * v[t - 1]);
            x[t] = x[t - 1] + DT * v[t - 1];
        }
        return new Trajectory(x, v);
    }

    static boolean valid(double k, double c) { return k >= 0 && c >= 0; }

    static double loss(double k, double c, double[] observed) {
        double[] x = simulate(k, c, observed[0], 0.0).x;
        double total = 0;
        for (int t = 0; t < STEPS; t++) total += (x[t] - observed[t]) * (x[t] - observed[t]);
        return total;
    }

    /** Forward sweep plus the adjoint recursion; returns dL/dk and dL/dc. */
    static Params gradient(double k, double c, double[] observed) {
        Trajectory trajectory = simulate(k, c, observed[0], 0.0);
        double[] x = trajectory.x;
        double[] v = trajectory.v;
        double[] xbar = new double[STEPS];
        double[] vbar = new double[STEPS];
        for (int t = 0; t < STEPS; t++) xbar[t] = 2 * (x[t] - observed[t]);
        double dk = 0;
        double dc = 0;
        for (int t = STEPS - 1; t >= 1; t--) {
            // x[t] = x[t-1] + dt * v[t-1]
            xbar[t - 1] += xbar[t];
            vbar[t - 1] += DT * xbar[t];
            // v[t] = v[t-1] + dt * (-k * x[t-1] - c * v[t-1])
            dk += vbar[t] * (-DT * x[t - 1]);
            dc += vbar[t] * (-DT * v[t - 1]);
            vbar[t - 1] += vbar[t] * (1 - DT * c);
            xbar[t - 1] += vbar[t] * (-DT * k);
        }
        return new Params(k, c, dk, dc);
    }

    /** Central finite difference of the loss with respect to k or c. */
    static double centralDifference(double k, double c, double[] observed, boolean withRespectToK, double h) {
        double plus = withRespectToK ? loss(k + h, c, observed) : loss(k, c + h, observed);
        double minus = withRespectToK ? loss(k - h, c, observed) : loss(k, c - h, observed);
        return (plus - minus) / (2 * h);
    }

    /**
     * Complex-step derivative: only the perturbed parameter gets an imaginary part, so the imaginary part of
     * the loss is exactly h times the derivative - no subtractive cancellation, hence h can be 1e-30.
     */
    static double complexStep(double k, double c, double[] observed, boolean perturbK) {
        double h = 1e-30;
        double ki = perturbK ? h : 0;
        double ci = perturbK ? 0 : h;
        double xr = observed[0];
        double xi = 0;
        double vr = 0;
        double vi = 0;
        double imaginary = 0;
        for (int t = 0; t < STEPS; t++) {
            if (t > 0) {
                double previousVr = vr;
                double previousVi = vi;
                double accelR = -(k * xr - ki * xi) - (c * vr - ci * vi);
                double accelI = -(k * xi + ki * xr) - (c * vi + ci * vr);
                vr += DT * accelR;
                vi += DT * accelI;
                // x[t] = x[t-1] + dt * v[t-1]: the position uses the *previous* velocity, as in simulate()
                xr += DT * previousVr;
                xi += DT * previousVi;
            }
            imaginary += 2 * (xr - observed[t]) * xi;
        }
        return imaginary / h;
    }

    /** 0.5*m*v^2 + 0.5*k*x^2 with unit mass. */
    static double energyOf(Trajectory trajectory, int t, double k) {
        return 0.5 * trajectory.v[t] * trajectory.v[t] + 0.5 * k * trajectory.x[t] * trajectory.x[t];
    }

    static double amplitude(double[] x) {
        double max = 0;
        for (int t = x.length / 2; t < x.length; t++) max = Math.max(max, Math.abs(x[t]));
        return max;
    }

    static boolean close(double a, double b, double tolerance) {
        return Math.abs(a - b) <= tolerance * (1 + Math.max(Math.abs(a), Math.abs(b)));
    }

    static double round(double value) { return Math.round(value * 1e6) / 1e6; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
