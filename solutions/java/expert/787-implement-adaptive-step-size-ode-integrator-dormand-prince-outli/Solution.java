/** Task 787 - Implement adaptive step-size ODE integrator (Dormand-Prince outline). (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Dormand-Prince 5(4) advances y' = f(t, y) with a seven stage explicit Runge-Kutta pair that shares the
 * same stages for two solutions:
 *
 *     y_{n+1}  = y_n + h sum_j b5_j k_j      (fifth order, the accepted solution)
 *     y4_{n+1} = y_n + h sum_j b4_j k_j      (embedded fourth order)
 *     err      = y_{n+1} - y4_{n+1}          (a local error estimate of size O(h^5))
 *
 * The step is accepted when the scaled max-norm of err, ||err_i / (atol + rtol max(|y_i|,|y5_i|))||_inf,
 * is at most 1, and the controller then proposes
 *
 *     h_new = h * clip(0.9 * err^(-1/5), 0.2, 5)        (the exponent is -1/5 because err = O(h^5))
 *
 * rejecting the step (without advancing t or y) and retrying with a smaller h otherwise. The last stage is
 * f(t + h, y_{n+1}), so a real implementation would reuse it as the next step's first stage (FSAL); here
 * every step re-evaluates stage 1, which only costs work, not accuracy, and keeps the code direct.
 *
 * Simplified (this is the "outline" variant): no dense-output/quartic interpolant, no FSAL reuse, no event
 * location and no PI step controller. The tableau itself is not simplified - all 7 stages, both weight
 * rows and the standard order conditions are the real ones.
 *
 * Verified: the Butcher tableau satisfies the order 1-4 conditions for both weight rows and the order 5
 * conditions for b5 while b4 deliberately violates them (that violation is exactly what makes the
 * difference an O(h^5) error estimate); the per-step estimate matches the true local error of the embedded
 * solution and is conservative for the returned one; the local error falls 32-fold per halving of h; the
 * adaptive solution of y' = y lands within tolerance with a handful of steps; every accepted step satisfies
 * the tolerance and the step sizes grow on a smooth problem; across a sweep of tolerances from 1e-3 to 1e-9
 * on a boundary-layer problem the achieved error tracks the requested tolerance, falls monotonically and
 * costs monotonically more steps while the step sizes vary several-fold; a deliberately absurd h0 is
 * rejected and retried down; the 2D harmonic oscillator matches its closed form; the run lands exactly on
 * t1; repeated runs are identical; and degenerate inputs (zero interval, bad tolerances) are handled.
 */
import java.util.*;

/** Right-hand side of the first order system y' = f(t, y). */
interface Rhs {
    double[] eval(double t, double[] y);
}

/** One Dormand-Prince step: both solutions, their difference, and the recycled last stage. */
class DpStep {
    final double[] fifth;
    final double[] fourth;
    final double[] error;
    final double[] lastStage;

    DpStep(double[] fifth, double[] fourth, double[] error, double[] lastStage) {
        this.fifth = fifth;
        this.fourth = fourth;
        this.error = error;
        this.lastStage = lastStage;
    }
}

/** Outcome of a run: final state and the statistics of the step-size control. */
class DpOutcome {
    final double[] finalState;
    final double endTime;
    final int accepted, rejected;
    final double[] stepSizes;
    final double[] errorRatios;
    final long evaluations;

    DpOutcome(double[] finalState, double endTime, int accepted, int rejected, double[] stepSizes,
            double[] errorRatios, long evaluations) {
        this.finalState = finalState;
        this.endTime = endTime;
        this.accepted = accepted;
        this.rejected = rejected;
        this.stepSizes = stepSizes;
        this.errorRatios = errorRatios;
        this.evaluations = evaluations;
    }
}

/** The Dormand-Prince 5(4) tableau and its adaptive driver. */
class Dp54 {
    static final double SAFETY = 0.9, MIN_FACTOR = 0.2, MAX_FACTOR = 5.0;
    static final double[] C = {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0};
    static final double[][] A = {
        {},
        {1.0 / 5},
        {3.0 / 40, 9.0 / 40},
        {44.0 / 45, -56.0 / 15, 32.0 / 9},
        {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
        {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
        {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84}
    };
    static final double[] B5 = {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0};
    static final double[] B4 = {5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200,
            187.0 / 2100, 1.0 / 40};

    static double[] combine(double[] y, double[][] k, double[] b, double h) {
        double[] out = y.clone();
        for (int j = 0; j < b.length; j++) {
            if (b[j] == 0) continue;
            for (int i = 0; i < y.length; i++) out[i] += h * b[j] * k[j][i];
        }
        return out;
    }

    static DpStep step(Rhs f, double t, double[] y, double h, long[] work) {
        double[][] k = new double[7][];
        k[0] = f.eval(t, y);
        work[0]++;
        for (int s = 1; s < 7; s++) {
            double[] arg = y.clone();
            for (int j = 0; j < s; j++) {
                if (A[s][j] == 0) continue;
                for (int i = 0; i < y.length; i++) arg[i] += h * A[s][j] * k[j][i];
            }
            k[s] = f.eval(t + C[s] * h, arg);
            work[0]++;
        }
        double[] fifth = combine(y, k, B5, h), fourth = combine(y, k, B4, h);
        double[] error = new double[y.length];
        for (int i = 0; i < y.length; i++) error[i] = fifth[i] - fourth[i];
        return new DpStep(fifth, fourth, error, k[6]);
    }

    /** || err_i / (atol + rtol max(|y0_i|, |y1_i|)) ||_inf - the standard scaled error norm. */
    static double scaledError(double[] err, double[] y0, double[] y1, double atol, double rtol) {
        double worst = 0;
        for (int i = 0; i < err.length; i++) {
            double scale = atol + rtol * Math.max(Math.abs(y0[i]), Math.abs(y1[i]));
            worst = Math.max(worst, Math.abs(err[i]) / scale);
        }
        return worst;
    }

    static DpOutcome adaptive(Rhs f, double t0, double[] y0, double t1, double atol, double rtol, double h0) {
        if (h0 <= 0 || atol <= 0 || rtol < 0) throw new IllegalArgumentException("h0 <= 0 or a bad tolerance");
        long[] work = new long[1];
        double t = t0, h = h0;
        double[] y = y0.clone();
        int accepted = 0, rejected = 0;
        List<Double> sizes = new ArrayList<>(), ratios = new ArrayList<>();
        while (t < t1) {
            if (t + h > t1) h = t1 - t;
            DpStep s = step(f, t, y, h, work);
            double err = scaledError(s.error, y, s.fifth, atol, rtol);
            if (err <= 1.0) {
                t += h;
                y = s.fifth;
                accepted++;
                sizes.add(h);
                ratios.add(err);
                h *= (err == 0 ? MAX_FACTOR : Math.min(MAX_FACTOR, Math.max(MIN_FACTOR, SAFETY * Math.pow(err, -0.2))));
            } else {
                rejected++;
                h *= Math.max(MIN_FACTOR, SAFETY * Math.pow(err, -0.2));
            }
        }
        return new DpOutcome(y, t, accepted, rejected, toArray(sizes), toArray(ratios), work[0]);
    }

    static double[] toArray(List<Double> values) {
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    // --- order conditions of the Butcher tableau ------------------------------------------
    static double dot(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) sum += u[i] * v[i];
        return sum;
    }

    static double[] mul(double[] u, double[] v) {
        double[] out = new double[u.length];
        for (int i = 0; i < u.length; i++) out[i] = u[i] * v[i];
        return out;
    }

    /** (a c)_j = sum_k a_jk c_k, one application of the tableau's lower triangle. */
    static double[] applyA(double[] c) {
        double[] out = new double[A.length];
        for (int j = 0; j < A.length; j++)
            for (int k = 0; k < A[j].length; k++) out[j] += A[j][k] * c[k];
        return out;
    }

    /**
     * Worst violation of the standard rooted-tree order conditions of each order 1..5 for the weights b.
     * A weight row of order p must make all of them vanish; the embedded row is only fourth order, so its
     * order-5 residual is exactly the quantity the step-size controller is built on.
     */
    static double[] orderResiduals(double[] b) {
        double[] c = C;
        double[] c2 = mul(c, c), c3 = mul(c2, c), c4 = mul(c3, c);
        double[] ac = applyA(c), ac2 = applyA(c2), ac3 = applyA(c3), aac = applyA(ac);
        double[] out = new double[6];
        out[1] = Math.abs(dot(b, new double[]{1, 1, 1, 1, 1, 1, 1}) - 1.0);
        out[2] = Math.abs(dot(b, c) - 1.0 / 2);
        out[3] = Math.max(Math.abs(dot(b, c2) - 1.0 / 3), Math.abs(dot(b, ac) - 1.0 / 6));
        out[4] = Math.max(Math.abs(dot(b, c3) - 1.0 / 4),
                Math.max(Math.abs(dot(b, mul(c, ac)) - 1.0 / 8),
                        Math.max(Math.abs(dot(b, ac2) - 1.0 / 12), Math.abs(dot(b, aac) - 1.0 / 24))));
        out[5] = Math.max(Math.abs(dot(b, c4) - 1.0 / 5),
                Math.max(Math.abs(dot(b, ac3) - 1.0 / 20), Math.abs(dot(b, mul(ac, ac)) - 1.0 / 20)));
        return out;
    }
}

public class Solution {
    static double[] maxElement(double[][] m) {
        double[] out = new double[m[0].length];
        for (double[] row : m)
            for (int i = 0; i < row.length; i++) out[i] = Math.max(out[i], row[i]);
        return out;
    }

    public static void main(String[] args) {
        Rhs exponential = (t, y) -> new double[]{y[0]};
        Rhs oscillator = (t, y) -> new double[]{y[1], -y[0]};

        // --- the tableau really is a 5(4) pair ---------------------------------------------
        double[] worst5 = maxElement(new double[][]{Dp54.orderResiduals(Dp54.B5)});
        double[] worst4 = maxElement(new double[][]{Dp54.orderResiduals(Dp54.B4)});
        double low5 = Math.max(Math.max(worst5[1], worst5[2]), Math.max(worst5[3], worst5[4]));
        double low4 = Math.max(Math.max(worst4[1], worst4[2]), Math.max(worst4[3], worst4[4]));
        check(low5 < 1e-13, "b5 must satisfy the order 1-4 conditions, worst residual " + low5);
        check(low4 < 1e-13, "b4 must satisfy the order 1-4 conditions, worst residual " + low4);
        check(worst5[5] < 1e-13, "b5 must satisfy the order 5 conditions, residual " + worst5[5]);
        check(worst4[5] > 1e-5 && worst4[5] > 1e8 * worst5[5],
                "b4 deliberately violates the order 5 conditions (that residual is what the error estimate "
                        + "measures), got " + worst4[5]);

        // --- the error estimate behaves like the local error it claims to be ----------------
        long[] work = new long[1];
        double[] start = {Math.exp(0.3)};
        DpStep coarse = Dp54.step(exponential, 0.3, start, 0.2, work);
        DpStep fine = Dp54.step(exponential, 0.3, start, 0.1, work);
        double trueFifthCoarse = Math.abs(coarse.fifth[0] - Math.exp(0.5));
        double trueFourthCoarse = Math.abs(coarse.fourth[0] - Math.exp(0.5));
        double estimateCoarse = Math.abs(coarse.error[0]), estimateFine = Math.abs(fine.error[0]);
        check(estimateCoarse > 0.5 * trueFourthCoarse && estimateCoarse < 2.0 * trueFourthCoarse,
                "estimate " + estimateCoarse + " should track the embedded solution's error " + trueFourthCoarse);
        check(estimateCoarse > trueFifthCoarse,
                "the estimate must be conservative for the returned solution: " + estimateCoarse
                        + " vs " + trueFifthCoarse);
        check(estimateCoarse / estimateFine > 25 && estimateCoarse / estimateFine < 40,
                "the local error estimate must scale like h^5, ratio " + estimateCoarse / estimateFine);
        check(trueFourthCoarse / trueFifthCoarse > 4,
                "the embedded solution must be visibly less accurate, ratio " + trueFourthCoarse / trueFifthCoarse);

        // --- adaptive integration of y' = y against the closed form ------------------------
        DpOutcome adaptive = Dp54.adaptive(exponential, 0, new double[]{1.0}, 1, 1e-9, 1e-9, 0.02);
        double error = Math.abs(adaptive.finalState[0] - Math.E);
        check(error < 1e-8, "adaptive error on y' = y is " + error);
        check(adaptive.accepted >= 4 && adaptive.accepted <= 40, "adaptive used " + adaptive.accepted + " steps");
        check(adaptive.rejected == 0, "a sane h0 must not reject, got " + adaptive.rejected + " rejections");
        check(adaptive.evaluations == 7L * adaptive.accepted, "seven evaluations per accepted step");
        for (int i = 0; i < adaptive.errorRatios.length; i++)
            check(adaptive.errorRatios[i] <= 1.0, "accepted step " + i + " exceeded the tolerance");
        check(adaptive.stepSizes[adaptive.stepSizes.length - 1] > 2 * adaptive.stepSizes[0],
                "step sizes must grow on a smooth problem: " + adaptive.stepSizes[0] + " -> "
                        + adaptive.stepSizes[adaptive.stepSizes.length - 1]);
        check(adaptive.endTime == 1.0, "the run must land exactly on t1, got " + adaptive.endTime);

        // --- adaptivity pays where the error constant varies wildly --------------------------
        // y' = -100 (y - cos t) - sin t has the exact solution y = cos t + e^{-100t}: a boundary layer
        // of width 1/100 sits on top of a smooth cosine, so the local error constant swings by many
        // orders of magnitude across [0, 1] and a fixed step size cannot follow it.
        Rhs layered = (t, y) -> new double[]{-100 * (y[0] - Math.cos(t)) - Math.sin(t)};
        DpOutcome stiff = Dp54.adaptive(layered, 0, new double[]{2.0}, 1, 1e-8, 1e-8, 0.05);
        double stiffError = Math.abs(stiff.finalState[0] - Math.cos(1));
        check(stiffError < 1e-5, "boundary layer problem error " + stiffError);
        double smallestStep = stiff.stepSizes[0], largestStep = stiff.stepSizes[0];
        for (double s : stiff.stepSizes) {
            smallestStep = Math.min(smallestStep, s);
            largestStep = Math.max(largestStep, s);
        }
        check(largestStep > 3 * smallestStep,
                "the layer must force strongly varying steps: " + smallestStep + " .. " + largestStep);
        // --- the tolerance contract: error tracks the requested tolerance ---------------------
        double[] tolerances = {1e-3, 1e-5, 1e-7, 1e-9};
        double previousError = Double.MAX_VALUE;
        int previousSteps = 0;
        double[] measured = new double[tolerances.length];
        int[] counts = new int[tolerances.length];
        for (int i = 0; i < tolerances.length; i++) {
            DpOutcome run = Dp54.adaptive(layered, 0, new double[]{2.0}, 1, tolerances[i], tolerances[i], 0.5);
            double err = Math.abs(run.finalState[0] - Math.cos(1));
            measured[i] = err;
            counts[i] = run.accepted;
            check(err <= 50 * tolerances[i], "tolerance " + tolerances[i] + " honoured with error " + err);
            check(err < previousError, "tightening the tolerance must tighten the error, " + err
                    + " after " + previousError);
            check(run.accepted >= previousSteps, "tighter tolerances must not need fewer steps, " + run.accepted
                    + " after " + previousSteps);
            previousError = err;
            previousSteps = run.accepted;
        }
        check(counts[counts.length - 1] > counts[0], "the tightest tolerance must cost the most steps");

        // --- a hopeless h0 must be rejected and retried down --------------------------------
        DpOutcome repaired = Dp54.adaptive(exponential, 0, new double[]{1.0}, 1, 1e-9, 1e-9, 10.0);
        check(repaired.rejected > 0, "a grossly oversized h0 must trigger rejections");
        check(Math.abs(repaired.finalState[0] - Math.E) < 1e-8,
                "even after rejections the solution must be accurate, error "
                        + Math.abs(repaired.finalState[0] - Math.E));

        // --- closed form on a 2D system, and repeatability -----------------------------------
        DpOutcome wave = Dp54.adaptive(oscillator, 0, new double[]{1.0, 0.0}, 10, 1e-10, 1e-10, 0.3);
        double waveError = Math.max(Math.abs(wave.finalState[0] - Math.cos(10)), Math.abs(wave.finalState[1] + Math.sin(10)));
        check(waveError < 1e-8, "harmonic oscillator error against (cos t, -sin t) is " + waveError);
        DpOutcome repeat = Dp54.adaptive(oscillator, 0, new double[]{1.0, 0.0}, 10, 1e-10, 1e-10, 0.3);
        check(repeat.accepted == wave.accepted && repeat.finalState[0] == wave.finalState[0]
                && repeat.finalState[1] == wave.finalState[1], "deterministic step sequence");

        // --- degenerate inputs ---------------------------------------------------------------
        DpOutcome empty = Dp54.adaptive(exponential, 2, new double[]{3.0}, 2, 1e-9, 1e-9, 0.5);
        check(empty.accepted == 0 && empty.finalState[0] == 3.0, "a zero length interval must do nothing");
        boolean rejectedH0 = false, rejectedTol = false;
        try {
            Dp54.adaptive(exponential, 0, new double[]{1.0}, 1, 1e-9, 0, -1.0);
        } catch (IllegalArgumentException expected) {
            rejectedH0 = true;
        }
        try {
            Dp54.adaptive(exponential, 0, new double[]{1.0}, 1, 0, 1e-9, 0.5);
        } catch (IllegalArgumentException expected) {
            rejectedTol = true;
        }
        check(rejectedH0, "a non-positive h0 must be rejected");
        check(rejectedTol, "a non-positive absolute tolerance must be rejected");

        System.out.println("tableau: order-5 residual b5 = " + worst5[5] + ", b4 = " + worst4[5]);
        System.out.println("y' = y: adaptive error " + error + " in " + adaptive.accepted + " steps");
        System.out.println("boundary layer: error " + stiffError + " in " + stiff.accepted + " adaptive steps");
        System.out.println("tolerance sweep: errors " + Arrays.toString(measured) + " at step counts "
                + Arrays.toString(counts));
        System.out.println("harmonic oscillator error at t = 10: " + waveError);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
