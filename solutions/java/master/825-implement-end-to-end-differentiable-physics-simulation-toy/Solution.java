/** Task 825 - Implement end-to-end differentiable physics simulation toy. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A differentiable simulator is a simulator whose output can be differentiated with respect to its inputs, so that
 * gradients flow "end to end" from a task loss back through every integration step into the physical parameters.
 * This file builds that for a 1D point mass on a spring with viscous drag, gravity and a one-sided soft floor,
 * integrated with semi-implicit (symplectic) Euler:
 *
 *     a_t     = g - (k/m)(x_t - L) - (b/m) v_t + (kc/m) max(0, -x_t)
 *     v_{t+1} = v_t + dt a_t ,   x_{t+1} = x_t + dt v_{t+1}
 *     loss    = mean over the batch of (1/T) sum_t (x_t - target_t)^2 + 1e-7 sin(k) + 1e-8 exp(-b/2)
 *
 * The parameter vector is theta = (k, b, g, x0_a, v0_a, x0_b, v0_b): the two rollouts of the batch share the
 * physical parameters (k, b, g) and differ in their initial state, which is how a real system-identification run
 * breaks the stiffness/damping trade-off. dloss/dtheta is obtained three independent ways and cross-checked:
 *   (1) reverse-mode automatic differentiation over the unrolled integration loop (a small tape-less graph),
 *   (2) hand-written forward sensitivity propagation, p' = J p + dp/dtheta_ext for every intermediate,
 *   (3) central finite differences of the plain double-precision simulator, the black-box oracle.
 * Method (1) is what an "end-to-end differentiable physics" stack really does; (2) is what the Levenberg-Marquardt
 * fit below uses to build its Jacobian, and (3) exists so the adjoint is never trusted on its own - it also covers
 * the non-smooth contact branch.
 *
 * Simplified: one spatial dimension, a single mass per rollout, one linear spring, linear drag, a one-sided soft
 * floor and a first-order integrator; there are no other bodies, joints, or collisions, and the floor is a
 * stiffness rather than a rigid constraint. Everything is in-process and deterministic: the only randomness is a
 * fixed-seed splitmix64 used to draw regression start points, so the whole run is reproducible bit for bit.
 *
 * Verified (22 checks): the autodiff engine reproduces closed-form gradients and the relu subgradient
 * convention; a hand-computed single integration step; symplectic Euler conserves energy to O(dt) when b = 0 and
 * that error halves when dt halves; drag dissipates energy (end energy below 40% of the initial one) while the
 * residual creep stays below 1e-2; reverse-mode gradients agree with forward sensitivities to 1e-9 and with
 * central differences to 1e-6 at two distinct parameter points, and both agree on the contact branch as well; the
 * soft floor leaves the predicted trajectory and the adjoint *exactly* unchanged while the gap is open, and inside
 * the floor it changes both; a Levenberg-Marquardt fit on the two-rollout batch cuts the loss by >1000x to a
 * residual MSE of ~1e-8, recovers the physical parameters to a few percent (the stiffness/damping pair is sloppy,
 * so the check is jointly on residual, parameter agreement and a held-out rollout) and predicts a held-out
 * trajectory it was never fitted on to within 5e-3; three fixed-seed restarts reach the same optimum; a fit whose
 * losses all go through an active contact still improves by an order of magnitude; a zero-step rollout has the
 * regularizer-only loss with finite gradients; and reruns are bit-identical.
 */
import java.util.*;

/** Reverse-mode automatic differentiation node (tape-less graph, micrograd style). */
final class Var {
    final double v;
    final String op;
    final Var x, y;
    final double k;
    double g;

    private Var(double v, String op, Var x, Var y, double k) {
        this.v = v; this.op = op; this.x = x; this.y = y; this.k = k;
    }

    static Var leaf(double v) { return new Var(v, "leaf", null, null, 0); }
    static Var add(Var a, Var b) { return new Var(a.v + b.v, "add", a, b, 0); }
    static Var sub(Var a, Var b) { return new Var(a.v - b.v, "sub", a, b, 0); }
    static Var mul(Var a, Var b) { return new Var(a.v * b.v, "mul", a, b, 0); }
    static Var scale(Var a, double k) { return new Var(a.v * k, "scale", a, null, k); }
    static Var square(Var a) { return new Var(a.v * a.v, "square", a, null, 0); }
    static Var sin(Var a) { return new Var(Math.sin(a.v), "sin", a, null, 0); }
    static Var exp(Var a) { return new Var(Math.exp(a.v), "exp", a, null, 0); }
    static Var relu(Var a) { return new Var(Math.max(a.v, 0), "relu", a, null, 0); }

    /** Reverse pass over the reverse topological order of the graph. The relu subgradient convention is 1 for
     *  v &gt; 0 and 0 otherwise, i.e. 0 exactly at the kink. */
    static void backward(Var root) {
        List<Var> order = new ArrayList<>();
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        postorder(root, seen, order);
        root.g = 1.0;
        for (int i = order.size() - 1; i >= 0; i--) {
            Var n = order.get(i);
            double g = n.g;
            switch (n.op) {
                case "add": n.x.g += g; n.y.g += g; break;
                case "sub": n.x.g += g; n.y.g -= g; break;
                case "mul": n.x.g += g * n.y.v; n.y.g += g * n.x.v; break;
                case "scale": n.x.g += g * n.k; break;
                case "square": n.x.g += g * 2 * n.x.v; break;
                case "sin": n.x.g += g * Math.cos(n.x.v); break;
                case "exp": n.x.g += g * n.v; break;
                case "relu": n.x.g += g * (n.x.v > 0 ? 1.0 : 0.0); break;
                default: break;
            }
        }
    }

    /** Depth-first postorder over the DAG: every node lands after all of its operands. */
    private static void postorder(Var n, Set<Var> seen, List<Var> order) {
        if (n == null || !seen.add(n)) return;
        postorder(n.x, seen, order);
        postorder(n.y, seen, order);
        order.add(n);
    }
}

/** Everything that is *not* being differentiated: mass, rest length, floor stiffness, time step. */
final class Scene {
    final double m, restLength, floorStiffness, dt;

    Scene(double m, double restLength, double floorStiffness, double dt) {
        this.m = m; this.restLength = restLength; this.floorStiffness = floorStiffness; this.dt = dt;
    }

    Scene withFloor(double stiffness) { return new Scene(m, restLength, stiffness, dt); }
    Scene withTimeStep(double step) { return new Scene(m, restLength, floorStiffness, step); }
}

public class Solution {
    /** theta = (k, b, g, x0_a, v0_a, x0_b, v0_b): 3 shared physical plus 2x2 initial-state parameters. */
    static final int NP = 7, BATCH = 2;
    static final String[] PARAM_NAMES =
            {"k", "b", "g", "x0a", "v0a", "x0b", "v0b"};
    static final double REGULARIZER_SIN = 1e-7, REGULARIZER_EXP = 1e-8;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static void close(double a, double b, double tol, String message) {
        check(closeTo(a, b, tol), message + " (" + a + " vs " + b + ")");
    }

    static double x0(double[] theta, int traj) { return theta[3 + 2 * traj]; }
    static double v0(double[] theta, int traj) { return theta[4 + 2 * traj]; }

    // ---------------------------------------------------------------- plain simulator (the black box)

    /** One semi-implicit Euler step; the contact term pushes up when the mass is below the floor at x = 0. */
    static double[] step(double x, double v, double[] theta, Scene s) {
        double contact = s.floorStiffness * Math.max(0, -x);
        double a = theta[2] - (theta[0] / s.m) * (x - s.restLength) - (theta[1] / s.m) * v + contact / s.m;
        double vNext = v + s.dt * a;
        return new double[] {x + s.dt * vNext, vNext};
    }

    static double[] rollout(double[] theta, int traj, Scene s, int steps) {
        double x = x0(theta, traj), v = v0(theta, traj);
        double[] xs = new double[steps + 1];
        xs[0] = x;
        for (int t = 0; t < steps; t++) {
            double[] n = step(x, v, theta, s);
            x = n[0]; v = n[1];
            xs[t + 1] = x;
        }
        return xs;
    }

    static double[][] rolloutBatch(double[] theta, Scene s, int steps) {
        double[][] out = new double[BATCH][];
        for (int i = 0; i < BATCH; i++) out[i] = rollout(theta, i, s, steps);
        return out;
    }

    static double energy(double x, double v, double[] theta, Scene s) {
        double spring = 0.5 * theta[0] * (x - s.restLength) * (x - s.restLength);
        double floor = 0.5 * s.floorStiffness * Math.max(0, -x) * Math.max(0, -x);
        return 0.5 * s.m * v * v + spring + floor - s.m * theta[2] * x;
    }

    /** Worst relative deviation of the total energy along one rollout. */
    static double worstRelativeEnergyDrift(double[] theta, Scene s, int steps) {
        double reference = energy(x0(theta, 0), v0(theta, 0), theta, s);
        double x = x0(theta, 0), v = v0(theta, 0), worst = 0;
        for (int t = 0; t < steps; t++) {
            double[] n = step(x, v, theta, s);
            x = n[0]; v = n[1];
            worst = Math.max(worst, Math.abs(energy(x, v, theta, s) - reference) / Math.abs(reference));
        }
        return worst;
    }

    // ---------------------------------------------------------------- reverse mode over the batch

    /** Builds the batch loss as a graph over Var so that one backward() pass yields dloss/dtheta. */
    static double lossAndGrad(double[] theta, Scene s, int steps, double[][] targets, double[] grad) {
        Var[] t = new Var[NP];
        for (int i = 0; i < NP; i++) t[i] = Var.leaf(theta[i]);
        Var rest = Var.leaf(s.restLength);
        Var loss = Var.leaf(0);
        for (int traj = 0; traj < BATCH; traj++) {
            Var x = t[3 + 2 * traj], v = t[4 + 2 * traj];
            Var trajLoss = Var.leaf(0);
            for (int step = 0; step < steps; step++) {
                Var contact = Var.scale(Var.relu(Var.scale(x, -1.0)), s.floorStiffness);
                Var spring = Var.scale(Var.mul(t[0], Var.sub(x, rest)), 1.0 / s.m);
                Var drag = Var.scale(Var.mul(t[1], v), 1.0 / s.m);
                Var a = Var.add(Var.sub(Var.sub(t[2], spring), drag), Var.scale(contact, 1.0 / s.m));
                Var vNext = Var.add(v, Var.scale(a, s.dt));
                Var xNext = Var.add(x, Var.scale(vNext, s.dt));
                Var err = Var.sub(xNext, Var.leaf(targets[traj][step + 1]));
                trajLoss = Var.add(trajLoss, Var.scale(Var.square(err), 1.0 / steps));
                x = xNext; v = vNext;
            }
            loss = Var.add(loss, Var.scale(trajLoss, 1.0 / BATCH));
        }
        loss = Var.add(loss, Var.scale(Var.sin(t[0]), REGULARIZER_SIN));
        loss = Var.add(loss, Var.scale(Var.exp(Var.scale(t[1], -0.5)), REGULARIZER_EXP));
        Var.backward(loss);
        for (int i = 0; i < NP; i++) grad[i] = t[i].g;
        return loss.v;
    }

    // ---------------------------------------------------------------- forward sensitivities (independent)

    /** Hand-written forward sensitivity propagation over the whole batch; p' = J p + dp/dtheta_ext per step.
     *  With grad != null it accumulates the loss gradient, with jtj/jtr != null the normal equations of the
     *  least-squares fit. This is a second, independent derivative implementation next to the adjoint above. */
    static double sensitivityPass(double[] theta, Scene s, int steps, double[][] targets,
                                  double[] grad, double[][] jtj, double[] jtr) {
        double loss = 0;
        double scale = steps * (double) BATCH;
        for (int traj = 0; traj < BATCH; traj++) {
            double x = x0(theta, traj), v = v0(theta, traj);
            double[] px = new double[NP], pv = new double[NP];
            px[3 + 2 * traj] = 1.0;
            pv[4 + 2 * traj] = 1.0;
            double[] pa = new double[NP], pvNext = new double[NP], pxNext = new double[NP];
            for (int step = 0; step < steps; step++) {
                boolean contact = x < 0;
                double comp = contact ? -x : 0;
                double a = theta[2] - (theta[0] / s.m) * (x - s.restLength) - (theta[1] / s.m) * v
                        + s.floorStiffness * comp / s.m;
                double dadx = -(theta[0] + (contact ? s.floorStiffness : 0)) / s.m;
                double dadv = -theta[1] / s.m;
                double dadk = -(x - s.restLength) / s.m;
                double dadb = -v / s.m;
                for (int i = 0; i < NP; i++) {
                    double explicit = i == 0 ? dadk : i == 1 ? dadb : i == 2 ? 1.0 : 0.0;
                    pa[i] = dadx * px[i] + dadv * pv[i] + explicit;
                }
                double vNext = v + s.dt * a;
                double xNext = x + s.dt * vNext;
                for (int i = 0; i < NP; i++) {
                    pvNext[i] = pv[i] + s.dt * pa[i];
                    pxNext[i] = px[i] + s.dt * pvNext[i];
                }
                double err = xNext - targets[traj][step + 1];
                loss += err * err / scale;
                for (int i = 0; i < NP; i++) {
                    if (grad != null) grad[i] += 2 * err * pxNext[i] / scale;
                    if (jtj != null) {
                        jtr[i] += pxNext[i] * err / scale;
                        for (int j = 0; j < NP; j++) jtj[i][j] += pxNext[i] * pxNext[j] / scale;
                    }
                }
                x = xNext; v = vNext;
                System.arraycopy(pxNext, 0, px, 0, NP);
                System.arraycopy(pvNext, 0, pv, 0, NP);
            }
        }
        if (grad != null) {
            loss += REGULARIZER_SIN * Math.sin(theta[0]) + REGULARIZER_EXP * Math.exp(-theta[1] / 2);
            grad[0] += REGULARIZER_SIN * Math.cos(theta[0]);
            grad[1] -= REGULARIZER_EXP * 0.5 * Math.exp(-theta[1] / 2);
        }
        return loss;
    }

    static double lossAndForwardSensitivity(double[] theta, Scene s, int steps, double[][] targets, double[] grad) {
        return sensitivityPass(theta, s, steps, targets, grad, null, null);
    }

    static double lossOnly(double[] theta, Scene s, int steps, double[][] targets) {
        double loss = 0;
        for (int traj = 0; traj < BATCH; traj++) {
            double[] xs = rollout(theta, traj, s, steps);
            for (int t = 1; t <= steps; t++) {
                double e = xs[t] - targets[traj][t];
                loss += e * e / (steps * BATCH);
            }
        }
        return loss + REGULARIZER_SIN * Math.sin(theta[0]) + REGULARIZER_EXP * Math.exp(-theta[1] / 2);
    }

    static double[] centralDifferences(double[] theta, Scene s, int steps, double[][] targets, double h) {
        double[] g = new double[NP];
        for (int i = 0; i < NP; i++) {
            double[] up = theta.clone(), dn = theta.clone();
            up[i] += h; dn[i] -= h;
            g[i] = (lossOnly(up, s, steps, targets) - lossOnly(dn, s, steps, targets)) / (2 * h);
        }
        return g;
    }

    // ---------------------------------------------------------------- Levenberg-Marquardt fit

    static double[] regularizerGradient(double[] theta) {
        double[] g = new double[NP];
        g[0] = REGULARIZER_SIN * Math.cos(theta[0]);
        g[1] = -REGULARIZER_EXP * 0.5 * Math.exp(-theta[1] / 2);
        return g;
    }

    /** Accumulates the normal equations 2/(T*B) J^T J and 2/(T*B) J^T r of the batch, J = d x_t / d theta. */
    static void normalEquations(double[] theta, Scene s, int steps, double[][] targets,
                                double[][] jtj, double[] jtr) {
        sensitivityPass(theta, s, steps, targets, null, jtj, jtr);
    }

    /** Gaussian elimination with partial pivoting. */
    static double[] solve(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
            double[] swap = a[col]; a[col] = a[pivot]; a[pivot] = swap;
            if (Math.abs(a[col][col]) < 1e-15) a[col][col] = a[col][col] >= 0 ? 1e-15 : -1e-15;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double factor = a[r][col] / a[col][col];
                for (int k = col; k <= n; k++) a[r][k] -= factor * a[col][k];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = a[i][n] / a[i][i];
        return out;
    }

    /** Levenberg-Marquardt on the batch residual with a backtracking line search on the damped Gauss-Newton
     *  step; the Jacobian comes from the sensitivity propagation. The line search keeps every iteration
     *  monotone even across the kink of the soft floor. */
    static double[] fit(double[] start, Scene s, int steps, double[][] targets, int iterations) {
        double[] theta = start.clone();
        double loss = lossOnly(theta, s, steps, targets);
        double[][] jtj = new double[NP][NP];
        double[] jtr = new double[NP];
        for (int it = 0; it < iterations; it++) {
            for (double[] row : jtj) Arrays.fill(row, 0);
            Arrays.fill(jtr, 0);
            normalEquations(theta, s, steps, targets, jtj, jtr);
            double[] reg = regularizerGradient(theta);
            double[][] a = new double[NP][NP];
            double[] rhs = new double[NP];
            double diagonal = 0;
            for (int i = 0; i < NP; i++) {
                rhs[i] = 2 * jtr[i] + reg[i];
                for (int j = 0; j < NP; j++) a[i][j] = 2 * jtj[i][j];
                diagonal = Math.max(diagonal, a[i][i]);
            }
            for (int i = 0; i < NP; i++) a[i][i] += 1e-6 * (diagonal + 1e-12);
            double[] delta = solve(a, rhs);
            double stepNorm = 0;
            for (double d : delta) stepNorm = Math.max(stepNorm, Math.abs(d));
            if (stepNorm == 0) break;
            double[] accepted = null;
            double t = 1.0;
            for (int back = 0; back < 60; back++) {
                double[] trial = theta.clone();
                for (int i = 0; i < NP; i++) trial[i] -= t * delta[i];
                double trialLoss = lossOnly(trial, s, steps, targets);
                if (trialLoss < loss) {
                    accepted = trial;
                    loss = trialLoss;
                    break;
                }
                t *= 0.5;
            }
            if (accepted == null) break;
            theta = accepted;
        }
        return theta;
    }

    // ---------------------------------------------------------------- helpers

    /** Relative agreement of two gradient vectors, scaled by the larger magnitude. */
    static double relativeGradientGap(double[] a, double[] b) {
        double worst = 0;
        for (int i = 0; i < a.length; i++) {
            double scale = Math.max(1e-9, Math.abs(a[i]));
            worst = Math.max(worst, Math.abs(a[i] - b[i]) / scale);
        }
        return worst;
    }

    /** Largest absolute deviation between a rollout of the batch and its observation. */
    static double worstTemporalError(double[] theta, Scene s, int steps, int traj, double[] targets) {
        double[] xs = rollout(theta, traj, s, steps);
        double worst = 0;
        for (int t = 1; t <= steps; t++) worst = Math.max(worst, Math.abs(xs[t] - targets[t]));
        return worst;
    }

    static double mse(double[] theta, Scene s, int steps, double[][] targets) {
        double loss = 0;
        for (int traj = 0; traj < BATCH; traj++) {
            double[] xs = rollout(theta, traj, s, steps);
            for (int t = 1; t <= steps; t++) {
                double e = xs[t] - targets[traj][t];
                loss += e * e / (steps * BATCH);
            }
        }
        return loss;
    }

    static long seed;

    static long splitmix64() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double uniform(double lo, double hi) {
        return lo + (hi - lo) * ((splitmix64() >>> 11) * 0x1.0p-53);
    }

    static double[] zeros() { return new double[NP]; }

    /** Absolute-or-relative closeness test used to fold several assertions into one logical check. */
    static boolean closeTo(double a, double b, double tol) {
        return Math.abs(a - b) <= tol * Math.max(1.0, Math.max(Math.abs(a), Math.abs(b)));
    }

    public static void main(String[] args) {
        // ---- 1. the autodiff engine on an analytic expression, against closed form
        Var a = Var.leaf(0.7), b = Var.leaf(-1.3), c = Var.leaf(0.25);
        Var f = Var.square(Var.add(Var.mul(a, b), c));
        Var.backward(f);
        double ab = 0.7 * -1.3 + 0.25;
        check(closeTo(f.v, ab * ab, 1e-15) && closeTo(a.g, 2 * ab * -1.3, 1e-15)
                        && closeTo(b.g, 2 * ab * 0.7, 1e-15) && closeTo(c.g, 2 * ab, 1e-15),
                "autodiff reproduces the value and all three partials of (ab+c)^2");

        // ---- 2. chain rule through two nonlinear ops
        Var u = Var.leaf(0.4);
        Var.backward(Var.exp(Var.sin(u)));
        close(u.g, Math.exp(Math.sin(0.4)) * Math.cos(0.4), 1e-15, "chain rule through sin/exp");

        // ---- 3. relu subgradient convention, including the kink
        Var r1 = Var.leaf(1.5);
        Var.backward(Var.relu(r1));
        Var r2 = Var.leaf(-1.5);
        Var.backward(Var.relu(r2));
        Var r3 = Var.leaf(0.0);
        Var.backward(Var.relu(r3));
        check(closeTo(r1.g, 1.0, 1e-15) && closeTo(r2.g, 0.0, 1e-15) && closeTo(r3.g, 0.0, 1e-15),
                "relu slopes are 1 / 0 / 0, the kink itself taking the 0 convention");

        // ---- 4. hand-computed single step: k = b = 0 and no contact leave a = g exactly
        Scene freeFall = new Scene(1.0, 0.0, 0.0, 0.002);
        double[] ballistic = {0.0, 0.0, -2.0, 0.1, 0.3, 0.0, 0.0};
        double dt = freeFall.dt;
        double[] got = step(0.1, 0.3, ballistic, freeFall);
        check(closeTo(got[0], 0.1 + dt * (0.3 + dt * -2.0), 1e-15) && closeTo(got[1], 0.3 + dt * -2.0, 1e-15),
                "one free-fall step matches the hand-computed position and velocity");

        // ---- 5/6. energy: O(dt) conservation without drag, and the error halves when dt halves
        Scene oscillator = new Scene(1.0, 0.0, 0.0, 0.002);
        double[] springOnly = {4.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0};
        double driftCoarse = worstRelativeEnergyDrift(springOnly, oscillator, 2000);
        double driftFine = worstRelativeEnergyDrift(springOnly, oscillator.withTimeStep(0.001), 4000);
        check(driftCoarse < 1e-2 && driftFine < 0.6 * driftCoarse,
                "undamped energy stays within O(dt) and halving dt halves the drift: "
                        + driftCoarse + " -> " + driftFine);

        // ---- 7. drag dissipates energy, and any leftover creep is discretisation sized
        double[] damped = {4.0, 1.5, 0.0, 0.5, 0.0, 0.0, 0.0};
        double startEnergy = energy(x0(damped, 0), v0(damped, 0), damped, oscillator);
        double x = x0(damped, 0), v = v0(damped, 0), previous = startEnergy, creep = 0;
        for (int t = 0; t < 500; t++) {
            double[] n = step(x, v, damped, oscillator);
            x = n[0]; v = n[1];
            double e = energy(x, v, damped, oscillator);
            creep += Math.max(0, e - previous);
            previous = e;
        }
        check(previous < 0.4 * startEnergy && creep < 1e-2,
                "drag removes most of the energy (" + startEnergy + " -> " + previous
                        + ") and the residual creep is only discretisation error (" + creep + ")");

        // ---- 8..11. reverse mode vs hand-written forward sensitivities vs central differences.
        //        Finite differences need a smooth trajectory, so these runs keep the floor switched off; the
        //        contact branch is differentiated separately below.
        Scene smooth = new Scene(1.0, 0.4, 0.0, 0.001);
        int steps = 2400;
        double[] truth = {25.0, 12.0, -2.0, 0.92, 0.0, 0.05, 1.5};
        double[][] observed = rolloutBatch(truth, smooth, steps);
        double[] theta = {15.0, 6.0, -1.0, 0.8, 0.15, 0.1, 1.2};

        double[] gRev = zeros();
        double lossRev = lossAndGrad(theta, smooth, steps, observed, gRev);
        double[] gFwd = zeros();
        double lossFwd = lossAndForwardSensitivity(theta, smooth, steps, observed, gFwd);
        check(closeTo(lossRev, lossFwd, 1e-12) && lossRev > 0,
                "the two differentiation modes agree on the loss and the loss is positive off the optimum");
        double gapFwd = relativeGradientGap(gRev, gFwd);
        check(gapFwd < 1e-9, "reverse mode matches forward sensitivities, worst relative gap " + gapFwd);

        double[] gFd = centralDifferences(theta, smooth, steps, observed, 1e-6);
        double gapFd = relativeGradientGap(gRev, gFd);
        check(gapFd < 1e-6, "reverse mode matches central differences, worst relative gap " + gapFd);

        double[] theta2 = {-3.0, 2.2, -3.7, 0.8, -0.35, -1.1, 0.9};
        double[][] observed2 = rolloutBatch(new double[] {-2.0, 1.4, -3.0, 0.5, 0.9, 0.2, 1.1}, smooth, steps);
        double[] gRev2 = zeros();
        lossAndGrad(theta2, smooth, steps, observed2, gRev2);
        double gapFd2 = relativeGradientGap(gRev2, centralDifferences(theta2, smooth, steps, observed2, 1e-6));
        check(gapFd2 < 1e-6, "gradient check holds at a second parameter point (negative k), gap " + gapFd2);

        // ---- 12/13. the non-smooth contact term: inert while open, active while penetrating
        Scene airScene = new Scene(1.0, 0.4, 12.0, 0.001);
        double[] airborne = {6.0, 0.9, -0.85, 0.45, 0.05, 0.3, -0.1};
        double[][] obsAir = rolloutBatch(airborne, airScene, steps);
        boolean allAbove = true;
        for (int i = 0; i < BATCH; i++)
            for (int t = 1; t <= steps; t++) if (obsAir[i][t] < 0) allAbove = false;
        check(allAbove, "the airborne batch never touches the floor");
        double[] thetaAir = {15.0, 6.0, -0.85, 0.45, 0.05, 0.3, -0.1};
        double[] gOpen = zeros();
        double lossOpen = lossAndGrad(thetaAir, airScene, steps, obsAir, gOpen);
        double[] gOpenNoFloor = zeros();
        double lossOpenNoFloor = lossAndGrad(thetaAir, airScene.withFloor(0.0), steps, obsAir, gOpenNoFloor);
        check(relativeGradientGap(gOpen, gOpenNoFloor) == 0.0 && lossOpen == lossOpenNoFloor,
                "while the gap is open the floor term leaves both the trajectory and the adjoint unchanged");

        // Both rollouts start inside the floor and settle there, so the relu branch never switches and the batch
        // is smooth: finite differences are valid here.
        Scene contact = new Scene(1.0, 0.0, 12.0, 0.001);
        double[] penetrator = {0.5, 8.0, -9.8, -0.5, 0.0, -0.3, 0.2};
        double[][] obsFloor = rolloutBatch(penetrator, contact, steps);
        boolean alwaysInside = true;
        for (int i = 0; i < BATCH; i++)
            for (int t = 1; t <= steps; t++) if (obsFloor[i][t] >= 0) alwaysInside = false;
        check(alwaysInside, "both penetrating rollouts stay inside the floor the whole time");
        double[] gContact = zeros();
        double lossContact = lossAndGrad(penetrator, contact, steps, obsFloor, gContact);
        double[] gContactNoFloor = zeros();
        double lossContactNoFloor = lossAndGrad(penetrator, contact.withFloor(0.0), steps, obsFloor,
                gContactNoFloor);
        check(lossContact != lossContactNoFloor && relativeGradientGap(gContact, gContactNoFloor) > 1e-3,
                "an active contact changes both the predicted trajectory and the adjoint");
        double gapContactFd = relativeGradientGap(gContact,
                centralDifferences(penetrator, contact, steps, obsFloor, 1e-6));
        double[] gContactFwd = zeros();
        lossAndForwardSensitivity(penetrator, contact, steps, obsFloor, gContactFwd);
        double gapContactFwd = relativeGradientGap(gContact, gContactFwd);
        check(gapContactFd < 1e-5 && gapContactFwd < 1e-9,
                "the contact-branch adjoint matches both finite differences (" + gapContactFd
                        + ") and forward sensitivities (" + gapContactFwd + ")");

        // ---- 14/15. inverse design on the batch: recover the physical parameters from observations alone
        double[] start = {15.0, 6.0, -1.0, 0.8, 0.15, 0.1, 1.2};
        double lossStart = lossOnly(start, smooth, steps, observed);
        double[] recovered = fit(start, smooth, steps, observed, 400);
        double lossEnd = lossOnly(recovered, smooth, steps, observed);
        double residual = mse(recovered, smooth, steps, observed);
        check(lossEnd * 1e3 < lossStart && residual < 1e-6,
                "gradient-based fitting cuts the loss by >1000x (" + lossStart + " -> " + lossEnd
                        + ") and drives the batch MSE to " + residual);
        double worstParam = 0;
        for (int i = 0; i < NP; i++) worstParam = Math.max(worstParam, Math.abs(recovered[i] - truth[i]));
        check(worstParam < 0.03 * truth[0], "stiffness, damping and the initial states are recovered to a few"
                + " percent, worst error " + worstParam);

        // ---- 16. the recovered physics predicts a held-out rollout it was never fitted on
        double[] heldOut = rollout(new double[] {truth[0], truth[1], truth[2], 0.7, -0.6, 0, 0}, 0, smooth, steps);
        double[] modelPrediction = rollout(new double[] {recovered[0], recovered[1], recovered[2], 0.7, -0.6, 0, 0},
                0, smooth, steps);
        double heldOutGap = 0;
        for (int t = 1; t <= steps; t++)
            heldOutGap = Math.max(heldOutGap, Math.abs(heldOut[t] - modelPrediction[t]));
        check(heldOutGap < 5e-3, "the recovered physics predicts a held-out trajectory, worst gap " + heldOutGap);

        // ---- 17. fixed-seed restarts reach the same optimum and the same parameters
        seed = 20240925L;
        double worstRestart = 0, worstRestartLoss = 0;
        for (int s = 0; s < 3; s++) {
            double[] randomStart = {uniform(8.0, 40.0), uniform(4.0, 20.0), uniform(-4.0, -0.5),
                    uniform(0.4, 1.1), uniform(-0.3, 0.4), uniform(-0.3, 0.5), uniform(0.8, 2.0)};
            double[] fitted = fit(randomStart, smooth, steps, observed, 300);
            worstRestartLoss = Math.max(worstRestartLoss, mse(fitted, smooth, steps, observed));
            for (int i = 0; i < NP; i++)
                worstRestart = Math.max(worstRestart, Math.abs(fitted[i] - recovered[i]));
        }
        check(worstRestartLoss < 1e-6 && worstRestart < 0.03 * truth[0],
                "every restart reaches a zero-residual fit (worst MSE " + worstRestartLoss
                        + ") and lands on the same optimum (worst spread " + worstRestart + ")");

        // ---- 18. the same fit works while sliding along an active contact
        double[] contactStart = {1.5, 9.0, -7.0, -0.45, 0.0, -0.35, 0.15};
        double contactLossStart = lossOnly(contactStart, contact, steps, obsFloor);
        double[] contactFit = fit(contactStart, contact, steps, obsFloor, 200);
        double contactLossEnd = lossOnly(contactFit, contact, steps, obsFloor);
        double contactResidual = mse(contactFit, contact, steps, obsFloor);
        check(contactLossEnd * 10 < contactLossStart && contactResidual < 1e-4,
                "gradients through an active contact drive the loss down an order of magnitude ("
                        + contactLossStart + " -> " + contactLossEnd + ") to MSE " + contactResidual);

        // ---- 19. degenerate rollout: zero steps leaves only the regularizer, gradients stay finite
        double[] gZero = zeros();
        double lossZero = lossAndGrad(theta, smooth, 0, observed, gZero);
        check(closeTo(lossZero, REGULARIZER_SIN * Math.sin(theta[0])
                        + REGULARIZER_EXP * Math.exp(-theta[1] / 2), 1e-12)
                        && Double.isFinite(gZero[0]) && Double.isFinite(gZero[1]),
                "a zero-step rollout has the regularizer-only loss and finite gradients");

        // ---- 20. determinism: identical inputs give bit-identical loss and adjoint
        double[] gA = zeros(), gB = zeros();
        double lA = lossAndGrad(theta, smooth, steps, observed, gA);
        double lB = lossAndGrad(theta, smooth, steps, observed, gB);
        boolean bitwise = Double.doubleToLongBits(lA) == Double.doubleToLongBits(lB);
        for (int i = 0; i < NP; i++)
            if (Double.doubleToLongBits(gA[i]) != Double.doubleToLongBits(gB[i])) bitwise = false;
        check(bitwise, "loss and adjoints are bit-identical across runs");

        System.out.println("differentiable physics: batch=" + BATCH + " rollouts x " + steps + " steps, dt="
                + smooth.dt + ", parameters=" + NP);
        System.out.println("inverse design loss " + lossStart + " -> " + lossEnd + " (MSE " + residual
                + "), worst parameter error " + worstParam);
        System.out.println("held-out trajectory gap " + heldOutGap
                + "; adjoint vs finite differences (relative) = " + gapFd
                + ", vs forward sensitivities = " + gapFwd);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
