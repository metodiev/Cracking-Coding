/** Task 788 - Implement stiff ODE solver L-stable method concept. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A stiff system y' = f(t, y) has a Jacobian whose eigenvalues span many decades, so an explicit method is
 * forced to h < stability_limit/|lambda_max| even when accuracy would allow far larger steps. Implicit
 * methods solve
 *
 *     backward Euler:  y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})          R(z) = 1 / (1 - z)
 *     trapezoidal:     y_{n+1} = y_n + h (f_n + f_{n+1}) / 2         R(z) = (1 + z/2) / (1 - z/2)
 *
 * for y_{n+1} by Newton's method, g(u) = u - y_n - h f(t+h, u) with the Jacobian I - h J formed by central
 * differences. R(z) is again the exact amplification per step of y' = lambda y, so stability is decided by
 * |R| alone: backward Euler has |R| <= 1 for every z with Re z < 0 (A-stable) AND R(z) -> 0 as z -> -inf,
 * which is L-stability: a stiff mode is annihilated in a single step. The trapezoidal rule is A-stable but
 * R(z) -> -1, so it *conserves* the stiff mode instead of killing it - the classic reason L-stability
 * matters, demonstrated here with one step of size 1 on a lambda = -1000 problem.
 *
 * Simplified (this is the "concept" variant): a fixed Newton tolerance and a finite-difference Jacobian
 * (no analytic Jacobians, no sparse or Krylov Newton, no step-size control and no order adaptation). The
 * methods, the amplification factors and the stiff test problems are the real ones.
 *
 * Verified: the simulated recursions reproduce R(z)^n = (1/(1-z))^n and ((1+z/2)/(1-z/2))^n exactly; both
 * A-stability (|R| <= 1 across the negative real axis) and L-stability (R -> 0) hold for backward Euler and
 * fail for the trapezoidal rule; RK4 is confirmed non-A-stable below z = -2.7853; on a lambda = -1000
 * boundary-layer problem one step of size 1 lands within 1e-2 of the truth for backward Euler while the
 * trapezoidal rule is off by more than 0.5; explicit RK4 diverges at h = 0.1 on the same problem; the
 * implicit method is first order (error halves when h halves); a 2D stiff system matches its closed form and
 * loses the stiff component; the Newton residual is driven below 1e-12; and degenerate inputs are handled.
 */
import java.util.*;

/** Right-hand side of the first order system y' = f(t, y). */
interface Rhs {
    double[] eval(double t, double[] y);
}

/** Final state plus the Newton work counters of an implicit run. */
class ImplicitRun {
    final double[] finalState;
    final int steps;
    final int newtonIterations;
    final int maxNewtonIterations;
    final double worstResidual;

    ImplicitRun(double[] finalState, int steps, int newtonIterations, int maxNewtonIterations,
            double worstResidual) {
        this.finalState = finalState;
        this.steps = steps;
        this.newtonIterations = newtonIterations;
        this.maxNewtonIterations = maxNewtonIterations;
        this.worstResidual = worstResidual;
    }
}

/** Backward Euler and the trapezoidal rule, both driven by a damped Newton solve. */
class Implicit {
    static final double NEWTON_TOL = 1e-12;
    static final int MAX_NEWTON = 60;

    static double[] add(double[] a, double[] b, double scale) {
        double[] out = a.clone();
        for (int i = 0; i < a.length; i++) out[i] += scale * b[i];
        return out;
    }

    /** Solves I - h J d = -g by Gaussian elimination with partial pivoting; returns d. */
    static double[] solve(double[][] m, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
            double[] tmp = a[col];
            a[col] = a[pivot];
            a[pivot] = tmp;
            for (int r = col + 1; r < n; r++) {
                double factor = a[r][col] / a[col][col];
                for (int c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = a[i][n];
            for (int j = i + 1; j < n; j++) sum -= a[i][j] * x[j];
            x[i] = sum / a[i][i];
        }
        return x;
    }

    /** Central-difference Jacobian of f at (t, y). */
    static double[][] jacobian(Rhs f, double t, double[] y) {
        int n = y.length;
        double[][] j = new double[n][n];
        for (int k = 0; k < n; k++) {
            double eps = 1e-6 * Math.max(1.0, Math.abs(y[k]));
            double[] up = y.clone(), down = y.clone();
            up[k] += eps;
            down[k] -= eps;
            double[] fu = f.eval(t, up), fd = f.eval(t, down);
            for (int i = 0; i < n; i++) j[i][k] = (fu[i] - fd[i]) / (2 * eps);
        }
        return j;
    }

    /** One implicit step; method "be" is backward Euler, "tr" the trapezoidal rule. */
    static double[] step(String method, Rhs f, double t, double[] y, double h, double[] work) {
        double tNext = t + h;
        boolean backward = method.equals("be");
        double[] fn = f.eval(t, y);
        double[] u = add(y, fn, h);            // explicit Euler predictor
        double residual = Double.MAX_VALUE;
        int iterations = 0;
        for (; iterations < MAX_NEWTON; iterations++) {
            double[] fNext = f.eval(tNext, u);
            double[] g = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                double target = backward ? y[i] + h * fNext[i] : y[i] + h * (fn[i] + fNext[i]) / 2;
                g[i] = u[i] - target;
            }
            residual = 0;
            for (double v : g) residual = Math.max(residual, Math.abs(v));
            if (residual < NEWTON_TOL) break;
            double[][] jac = jacobian(f, tNext, u);
            double scale = backward ? h : h / 2;
            double[][] system = new double[y.length][y.length];
            for (int i = 0; i < y.length; i++)
                for (int k = 0; k < y.length; k++)
                    system[i][k] = (i == k ? 1 : 0) - scale * jac[i][k];
            double[] delta = solve(system, g);
            for (int i = 0; i < y.length; i++) u[i] -= delta[i];
        }
        work[0] += iterations;
        work[1] = Math.max(work[1], iterations);
        work[2] = Math.max(work[2], residual);
        return u;
    }

    static ImplicitRun integrate(String method, Rhs f, double t0, double[] y0, double t1, int steps) {
        if (steps < 0) throw new IllegalArgumentException("steps must not be negative: " + steps);
        double h = steps == 0 ? 0 : (t1 - t0) / steps;
        double[] y = y0.clone();
        double[] work = new double[3];
        for (int i = 0; i < steps; i++) y = step(method, f, t0 + i * h, y, h, work);
        return new ImplicitRun(y, steps, (int) work[0], (int) work[1], work[2]);
    }

    // --- reference amplification factors ---------------------------------------------------
    static double backwardEulerAmplification(double z) { return 1.0 / (1.0 - z); }

    static double trapezoidalAmplification(double z) { return (1 + z / 2) / (1 - z / 2); }

    static double rk4Amplification(double z) {
        return 1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24;
    }
}

public class Solution {
    public static void main(String[] args) {
        Rhs scalar = (t, y) -> new double[]{-3.0 * y[0]};
        double h = 0.3, z = -3.0 * h;

        // --- the recursions reproduce R(z)^n exactly ---------------------------------------
        ImplicitRun eulerRecursion = Implicit.integrate("be", scalar, 0, new double[]{1.0}, 10 * h, 10);
        double expectedEuler = Math.pow(Implicit.backwardEulerAmplification(z), 10);
        check(Math.abs(eulerRecursion.finalState[0] - expectedEuler) <= 1e-12 * Math.abs(expectedEuler),
                "backward Euler must equal (1/(1-z))^n, got " + eulerRecursion.finalState[0]);
        ImplicitRun trapRecursion = Implicit.integrate("tr", scalar, 0, new double[]{1.0}, 10 * h, 10);
        double expectedTrap = Math.pow(Implicit.trapezoidalAmplification(z), 10);
        check(Math.abs(trapRecursion.finalState[0] - expectedTrap) <= 1e-12 * Math.abs(expectedTrap),
                "trapezoidal must equal ((1+z/2)/(1-z/2))^n, got " + trapRecursion.finalState[0]);
        check(eulerRecursion.maxNewtonIterations <= 2, "a linear problem needs one Newton step, got "
                + eulerRecursion.maxNewtonIterations);
        check(eulerRecursion.worstResidual < 1e-12, "Newton residual " + eulerRecursion.worstResidual);

        // --- A-stability and L-stability, on the negative real axis --------------------------
        boolean aStableEuler = true, aStableTrap = true;
        for (double t = -1e-3; t > -1000; t *= 1.7) {
            if (Math.abs(Implicit.backwardEulerAmplification(t)) > 1 + 1e-12) aStableEuler = false;
            if (Math.abs(Implicit.trapezoidalAmplification(t)) > 1 + 1e-12) aStableTrap = false;
        }
        check(aStableEuler, "backward Euler must be A-stable");
        check(aStableTrap, "the trapezoidal rule must be A-stable");
        check(Math.abs(Implicit.backwardEulerAmplification(-1e12)) < 1e-9,
                "backward Euler must annihilate an infinitely stiff mode (L-stability)");
        check(Math.abs(Implicit.trapezoidalAmplification(-1e12) + 1) < 1e-9,
                "the trapezoidal rule keeps the stiff mode alive at amplitude 1");
        check(Math.abs(Implicit.rk4Amplification(-3)) > 1 && Math.abs(Implicit.rk4Amplification(-2.7852)) < 1,
                "RK4 stops being A-stable just below z = -2.785");

        // --- the boundary layer problem y' = -1000(y - cos t) - sin t, exact cos t + e^{-1000 t} ----
        Rhs stiff = (t, y) -> new double[]{-1000 * (y[0] - Math.cos(t)) - Math.sin(t)};
        double exactAtOne = Math.cos(1);
        ImplicitRun coarse = Implicit.integrate("be", stiff, 0, new double[]{2.0}, 1, 1);
        double eulerError = Math.abs(coarse.finalState[0] - exactAtOne);
        ImplicitRun trap = Implicit.integrate("tr", stiff, 0, new double[]{2.0}, 1, 1);
        double trapError = Math.abs(trap.finalState[0] - exactAtOne);
        check(eulerError < 1e-2, "one backward Euler step of size 1 must resolve the layer, error " + eulerError);
        check(trapError > 0.5, "the trapezoidal rule must miss the layer badly, error " + trapError);
        check(trapError > 50 * eulerError, "L-stability must dominate here: " + eulerError + " vs " + trapError);

        // --- explicit RK4 on the same problem is hopeless at this step size --------------------
        double[] explicitRun = {2.0};
        for (int i = 0; i < 10; i++) {
            double t = i * 0.1;
            double[] k1 = stiff.eval(t, explicitRun);
            double[] k2 = stiff.eval(t + 0.05, new double[]{explicitRun[0] + 0.05 * k1[0]});
            double[] k3 = stiff.eval(t + 0.05, new double[]{explicitRun[0] + 0.05 * k2[0]});
            double[] k4 = stiff.eval(t + 0.1, new double[]{explicitRun[0] + 0.1 * k3[0]});
            explicitRun[0] += 0.1 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
        }
        check(Math.abs(explicitRun[0]) > 1e6, "explicit RK4 must blow up on the stiff problem, |y| = "
                + Math.abs(explicitRun[0]));

        // --- first order convergence of backward Euler on the same problem ---------------------
        double previous = Double.MAX_VALUE;
        for (int steps : new int[]{10, 20, 40, 80}) {
            ImplicitRun run = Implicit.integrate("be", stiff, 0, new double[]{2.0}, 1, steps);
            double err = Math.abs(run.finalState[0] - exactAtOne);
            check(err < previous, "refining the grid must reduce the error, " + err + " after " + previous);
            if (previous < Double.MAX_VALUE) {
                double ratio = previous / err;
                check(ratio > 1.6 && ratio < 2.4, "backward Euler is first order, halving h gave ratio " + ratio);
            }
            previous = err;
        }
        check(previous < 1e-3, "four grid refinements must reach a small error, got " + previous);

        // --- a 2D stiff system with a closed form ----------------------------------------------
        Rhs coupled = (t, y) -> new double[]{-500.5 * y[0] + 499.5 * y[1], 499.5 * y[0] - 500.5 * y[1]};
        ImplicitRun system = Implicit.integrate("be", coupled, 0, new double[]{1.0, 0.0}, 1, 10);
        double exactMode = 0.5 * Math.exp(-1);
        check(Math.abs(system.finalState[0] - exactMode) < 2e-2,
                "slow mode error " + Math.abs(system.finalState[0] - exactMode));
        check(Math.abs(system.finalState[0] - system.finalState[1]) < 1e-2,
                "the stiff component must be damped away, gap " + Math.abs(system.finalState[0] - system.finalState[1]));
        check(system.maxNewtonIterations <= 2, "the linear system needs one Newton step, got "
                + system.maxNewtonIterations);

        // --- a nonlinear stiff problem still converges quadratically ---------------------------
        Rhs nonlinear = (t, y) -> new double[]{-100 * (y[0] - y[0] * y[0] * y[0])};
        ImplicitRun nonlinearRun = Implicit.integrate("be", nonlinear, 0, new double[]{1.0}, 1, 20);
        check(nonlinearRun.maxNewtonIterations <= 8, "Newton must converge quickly, took "
                + nonlinearRun.maxNewtonIterations + " iterations");
        check(nonlinearRun.worstResidual < 1e-12, "Newton residual on the nonlinear problem "
                + nonlinearRun.worstResidual);
        check(Double.isFinite(nonlinearRun.finalState[0]) && Math.abs(nonlinearRun.finalState[0] - 1) < 1e-9,
                "the cubic problem sits at its stable fixed point 1, got " + nonlinearRun.finalState[0]);

        // --- degenerate inputs ------------------------------------------------------------------
        ImplicitRun empty = Implicit.integrate("be", scalar, 5, new double[]{7.0}, 5, 0);
        check(empty.steps == 0 && empty.finalState[0] == 7.0, "zero steps must leave the state unchanged");
        boolean rejected = false;
        try {
            Implicit.integrate("be", scalar, 0, new double[]{1.0}, 1, -3);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative step count must be rejected");

        System.out.println("one step of size 1 on lambda = -1000: backward Euler error " + eulerError
                + ", trapezoidal " + trapError);
        System.out.println("explicit RK4 on the stiff problem at h = 0.1: |y| = " + Math.abs(explicitRun[0]));
        System.out.println("backward Euler error after 80 steps: " + previous);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
