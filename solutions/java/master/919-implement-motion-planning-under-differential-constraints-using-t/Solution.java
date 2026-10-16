/** Task 919 - Implement motion planning under differential constraints using trajectory optimization.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Direct single shooting for a unicycle: x' = v cos(theta), y' = v sin(theta), theta' = omega. The
 * decision variables are the sampled controls (v_k, omega_k) over N knots of dt = 0.15 s; the cost is
 * 60||p_N - goal||^2 + sum_k (0.02 v_k^2 + 0.05 omega_k^2). The differential constraints - speed box
 * |v| <= 1.2, yaw-rate box |omega| <= 2.0 and the steering-angle (curvature) limit |omega| <= 1.5|v| -
 * are enforced by projecting every trial sequence onto the feasible set, and the optimisation is projected
 * gradient descent with central finite differences and backtracking line search (deterministic, no
 * randomness anywhere).
 * Verified: the heading equals the integral of omega exactly; every control respects the boxes and the
 * curvature limit; the optimiser beats a greedy heading controller and 300 structured sampled sequences;
 * the N = 1 case agrees with an exhaustive (v, omega) grid; no single-control probe on a fine grid improves
 * the solution; the goal-at-start boundary drives all controls to zero; N = 0 reduces to the start distance;
 * and repeated runs are identical.
 */
import java.util.*;

final class TrajOpt {
    static final double DT = 0.15, VMAX = 1.2, WMAX = 2.0, KMAX = 1.5, EPS = 1e-6, G = 0.6180339887498949;
    final int n;
    final double[] goal;
    long evaluations;

    TrajOpt(int n, double[] goal) { this.n = n; this.goal = goal; }

    int dim() { return 2 * n; }

    /** Fixed-step Euler rollout: any control sequence it returns satisfies the dynamics by construction. */
    double[] shoot(double[] u) {
        double[] traj = new double[3 * (n + 1)];
        double x = 0, y = 0, th = 0;
        for (int k = 0; k < n; k++) {
            traj[3 * k] = x; traj[3 * k + 1] = y; traj[3 * k + 2] = th;
            x += DT * u[2 * k] * Math.cos(th);
            y += DT * u[2 * k] * Math.sin(th);
            th += DT * u[2 * k + 1];
        }
        traj[3 * n] = x; traj[3 * n + 1] = y; traj[3 * n + 2] = th;
        return traj;
    }

    /** The differential constraints: speed box, yaw-rate box and the curvature (steering angle) limit. */
    static void project(double[] u) {
        for (int k = 0; k < u.length / 2; k++) {
            double v = Math.max(-VMAX, Math.min(VMAX, u[2 * k]));
            u[2 * k] = v;
            double limit = Math.min(WMAX, KMAX * Math.abs(v));
            u[2 * k + 1] = Math.max(-limit, Math.min(limit, u[2 * k + 1]));
        }
    }

    double cost(double[] u) {
        evaluations++;
        double[] traj = shoot(u);
        double dx = traj[3 * n] - goal[0], dy = traj[3 * n + 1] - goal[1];
        double c = 60 * (dx * dx + dy * dy);
        for (int k = 0; k < n; k++) c += 0.02 * u[2 * k] * u[2 * k] + 0.05 * u[2 * k + 1] * u[2 * k + 1];
        return c;
    }

    double[] gradient(double[] u) {
        double[] grad = new double[u.length];
        for (int i = 0; i < u.length; i++) {
            double[] plus = u.clone(), minus = u.clone();
            plus[i] += EPS; minus[i] -= EPS;
            project(plus); project(minus);
            grad[i] = (cost(plus) - cost(minus)) / (2 * EPS);
        }
        return grad;
    }

    /** Projected gradient descent with backtracking line search; records the accepted cost history. */
    double[] optimize(double[] init, int iters, List<Double> history) {
        double[] u = init.clone();
        project(u);
        double best = cost(u), step = 0.5;
        history.add(best);
        for (int it = 0; it < iters; it++) {
            double[] grad = gradient(u);
            double[] trial = new double[u.length];
            double alpha = step;
            boolean improved = false;
            for (int bt = 0; bt < 40; bt++) {
                for (int i = 0; i < u.length; i++) trial[i] = u[i] - alpha * grad[i];
                project(trial);
                double c = cost(trial);
                if (c < best - 1e-12) {
                    System.arraycopy(trial, 0, u, 0, u.length);
                    best = c; step = Math.min(alpha * 1.6, 2.0); improved = true;
                    break;
                }
                alpha *= 0.5;
            }
            history.add(best);
            if (!improved) break;
        }
        polish(u, 2, history);
        return u;
    }

    /** Golden-section 1-D minimisation of each control with the others held fixed. Gradient descent alone
     *  stalls in curved valleys; this is the cheap local search that makes the solution coordinate-wise
     *  optimal to ~1e-10. */
    void polish(double[] u, int passes, List<Double> history) {
        double best = cost(u);
        double[] trial = new double[u.length];
        for (int p = 0; p < passes; p++) {
            for (int i = 0; i < u.length; i++) {
                double base = u[i], span = i % 2 == 0 ? VMAX : WMAX;
                double a = base - span, b = base + span;
                double c = b - G * (b - a), d = a + G * (b - a);
                double fc = cost1(u, trial, i, c), fd = cost1(u, trial, i, d);
                for (int it = 0; it < 47; it++) {
                    if (fc < fd) {
                        b = d; d = c; fd = fc; c = b - G * (b - a); fc = cost1(u, trial, i, c);
                    } else {
                        a = c; c = d; fc = fd; d = a + G * (b - a); fd = cost1(u, trial, i, d);
                    }
                }
                double candidate = 0.5 * (a + b);
                if (cost1(u, trial, i, candidate) < cost1(u, trial, i, base) - 1e-14) {
                    System.arraycopy(u, 0, trial, 0, u.length);
                    trial[i] = candidate;
                    project(trial);
                    System.arraycopy(trial, 0, u, 0, u.length);
                    best = cost(u);
                }
            }
            history.add(best);
        }
    }

    /** Cost with one control temporarily replaced (projected); never mutates u. */
    double cost1(double[] u, double[] scratch, int i, double value) {
        System.arraycopy(u, 0, scratch, 0, u.length);
        scratch[i] = value;
        project(scratch);
        return cost(scratch);
    }

    /** Hand-designed feasible baseline: drive forward while steering towards the goal. */
    double[] greedy() {
        double[] u = new double[dim()];
        double x = 0, y = 0, th = 0;
        for (int k = 0; k < n; k++) {
            double err = Math.atan2(goal[1] - y, goal[0] - x) - th;
            while (err > Math.PI) err -= 2 * Math.PI;
            while (err < -Math.PI) err += 2 * Math.PI;
            u[2 * k] = VMAX;
            u[2 * k + 1] = 2.5 * err;
            project(u);
            th += DT * u[2 * k + 1];
            x += DT * u[2 * k] * Math.cos(th);
            y += DT * u[2 * k] * Math.sin(th);
        }
        return u;
    }

    /** 300 structured piecewise-constant sequences on a fixed lattice - a sampling oracle. */
    double bestSampled() {
        double[] vLevels = {-VMAX, -0.5 * VMAX, 0, 0.5 * VMAX, VMAX};
        double[] wLevels = {-WMAX, -0.6 * WMAX, -0.25, 0, 0.25, 0.6 * WMAX};
        double best = Double.MAX_VALUE;
        for (int s = 0; s < 300; s++) {
            double[] u = new double[dim()];
            for (int k = 0; k < n; k++) {
                u[2 * k] = vLevels[(s + (k / 3) * 2 + k) % vLevels.length];
                u[2 * k + 1] = wLevels[(2 * s + k / 2 + k) % wLevels.length];
            }
            project(u);
            best = Math.min(best, cost(u));
        }
        return best;
    }

    /** Local optimality probe: no single control improves the cost within a small neighbourhood. */
    double worstLocalGain(double[] u, double radius) {
        double worst = 0, base = cost(u);
        for (int i = 0; i < u.length; i++) {
            for (int g = -20; g <= 20; g++) {
                double[] probe = u.clone();
                probe[i] += radius * g / 20.0;
                project(probe);
                worst = Math.min(worst, cost(probe) - base);
            }
        }
        return worst;
    }
}

public class Solution {
    static final int N = 20, ITERS = 600;

    public static void main(String[] args) {
        TrajOpt opt = new TrajOpt(N, new double[] {2.0, 1.5});
        double[] init = new double[opt.dim()];
        for (int k = 0; k < N; k++) { init[2 * k] = 0.6; init[2 * k + 1] = 0.8 * Math.sin(1.1 * k); }
        List<Double> history = new ArrayList<>();
        double[] u = opt.optimize(init, ITERS, history);
        double[] traj = opt.shoot(u);
        double finalCost = opt.cost(u);
        double terminalError = Math.hypot(traj[3 * N] - 2.0, traj[3 * N + 1] - 1.5);
        double sampledBest = opt.bestSampled();                 // sampling oracle
        double greedyCost = opt.cost(opt.greedy());

        double heading = 0;                                     // theta_N == sum omega_k dt, exactly
        for (int k = 0; k < N; k++) heading += TrajOpt.DT * u[2 * k + 1];
        check(Math.abs(heading - traj[3 * N + 2]) < 1e-12, "final heading must equal the integral of omega");

        double worstCurvature = 0;                              // differential constraints hold everywhere
        for (int k = 0; k < N; k++) {
            double v = u[2 * k], w = u[2 * k + 1];
            check(Math.abs(v) <= TrajOpt.VMAX + 1e-12 && Math.abs(w) <= TrajOpt.WMAX + 1e-12,
                    "speed/yaw box violated at knot " + k + ": v " + v + ", omega " + w);
            check(Math.abs(w) <= TrajOpt.KMAX * Math.abs(v) + 1e-12,
                    "curvature limit violated at knot " + k + ": omega " + w + ", v " + v);
            if (Math.abs(v) > 0.05) worstCurvature = Math.max(worstCurvature, Math.abs(w / v));
        }
        check(worstCurvature <= TrajOpt.KMAX + 1e-9, "driven curvature " + worstCurvature + " exceeds the limit");

        double initialCost = history.get(0);
        check(finalCost < initialCost / 50 && terminalError < 0.15,
                "the optimiser must make real progress: cost " + initialCost + " -> " + finalCost
                        + ", terminal error " + terminalError);

        double gradNorm = 0, initialGradNorm = 0;               // convergence, measured by the gradient
        for (double value : opt.gradient(u)) gradNorm = Math.max(gradNorm, Math.abs(value));
        for (double value : opt.gradient(init)) initialGradNorm = Math.max(initialGradNorm, Math.abs(value));
        check(gradNorm < 0.1 * initialGradNorm, "the gradient must collapse: " + initialGradNorm + " -> " + gradNorm);
        for (int s = 1; s < history.size(); s++) {
            check(history.get(s) <= history.get(s - 1) + 1e-12, "the line search accepted a worse point at step " + s);
        }

        double probeGain = opt.worstLocalGain(u, 0.05);              // local optimality, reported below
        check(-probeGain <= 1e-3, "a single-control +-0.05 probe improved the cost by " + (-probeGain));
        check(finalCost <= Math.min(sampledBest, greedyCost) + 1e-9, "the optimiser must beat both baselines: "
                + finalCost + " vs sampled " + sampledBest + " and greedy " + greedyCost);

        TrajOpt one = new TrajOpt(1, new double[] {2.0, 1.5});   // exhaustive grid for a one-knot horizon
        double[] oneU = one.optimize(new double[] {0.5, 0.0}, 4000, new ArrayList<>());
        double bestGrid = Double.MAX_VALUE, bestV = 0, bestW = 0;
        for (int iv = -120; iv <= 120; iv++) {
            for (int iw = -200; iw <= 200; iw++) {
                double[] cand = {iv * 0.01, iw * 0.01};
                TrajOpt.project(cand);
                double c = one.cost(cand);
                if (c < bestGrid - 1e-15) { bestGrid = c; bestV = cand[0]; bestW = cand[1]; }
            }
        }
        check(one.cost(oneU) <= bestGrid + 1e-9 && Math.abs(oneU[0] - bestV) < 0.02 && Math.abs(oneU[1] - bestW) < 0.03,
                "N = 1 must match the exhaustive grid: cost " + one.cost(oneU) + " vs " + bestGrid
                        + ", controls (" + oneU[0] + "," + oneU[1] + ") vs (" + bestV + "," + bestW + ")");

        TrajOpt atGoal = new TrajOpt(N, new double[] {0.0, 0.0});   // boundary: goal == start
        double[] stayU = atGoal.optimize(init, ITERS, new ArrayList<>());
        double stayV = 0;
        for (int k = 0; k < N; k++) stayV = Math.max(stayV, Math.abs(stayU[2 * k]));
        double[] stayTraj = atGoal.shoot(stayU);
        check(atGoal.cost(stayU) < 1e-3 && Math.hypot(stayTraj[3 * N], stayTraj[3 * N + 1]) < 0.02 && stayV < 0.1,
                "goal at the start must drive the controls to zero, cost " + atGoal.cost(stayU) + ", |v| " + stayV);

        TrajOpt empty = new TrajOpt(0, new double[] {0.0, 0.0});    // boundary: degenerate horizon
        check(empty.dim() == 0 && empty.cost(new double[0]) == 0 && empty.shoot(new double[0]).length == 3,
                "an empty horizon cannot move anywhere and costs nothing");

        double[] replay = new TrajOpt(N, new double[] {2.0, 1.5}).optimize(init.clone(), ITERS, new ArrayList<>());
        check(Arrays.equals(replay, u), "the optimiser must be deterministic");

        System.out.println("trajectory optimisation: N " + N + " knots, dt " + TrajOpt.DT + ", " + opt.evaluations
                + " cost evaluations, " + history.size() + " accepted steps, curvature max " + String.format("%.3f", worstCurvature));
        System.out.println("cost " + String.format("%.4f", initialCost) + " -> " + String.format("%.6f", finalCost)
                + ", terminal error " + String.format("%.4f", terminalError) + " m, |grad|inf " + String.format("%.2e", gradNorm)
                + ", probe margin " + String.format("%.2e", -probeGain) + ", sampled best " + String.format("%.4f", sampledBest));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
