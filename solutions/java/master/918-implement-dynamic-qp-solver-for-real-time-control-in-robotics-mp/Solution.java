/** Task 918 - Implement dynamic QP solver for real-time control in robotics (MPC small demo).
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Condensed MPC for a discrete double integrator: rolling x_{i+1} = A x_i + B u_i out over the horizon
 * turns the quadratic optimal-control problem into one dense box-constrained QP in the control sequence,
 *   minimise 0.5 u'Hu + g'u + base,  H = Bc'QbarBc + r I,  g = Bc'Qbar (Abar x0 - xref),  |u_k| <= u_max.
 * H is symmetric positive definite, so the unique minimiser is exactly where the projected gradient
 * vanishes: grad_k = 0 inside the box, grad_k <= 0 at the upper bound, grad_k >= 0 at the lower bound.
 * Simplified: dense condensation instead of a banded structure, control bounds only (state limits would
 * need linear state constraints), FISTA instead of an active-set solver. Not simplified: the QP really is
 * solved, and optimality is checked against two independent oracles - Gaussian elimination of the free
 * system and a systematic feasible-direction probe - never against the solver's own output.
 * Verified: H symmetry/definiteness; dense cost == rolled-out cost; the free minimum vs the oracle; KKT
 * with an active bound; no feasible perturbation improving the solution; the N = 1 closed form; the
 * degenerate N = 0 horizon; full saturation at a tiny bound; a 60-step closed loop reaching the reference.
 */
import java.util.*;

/** Dense box-constrained QP: minimise 0.5 u'Hu + g'u + base over lo <= u <= hi. */
final class Qp {
    final double[][] h;
    final double[] g;
    final double base, lo, hi, step;
    Qp(double[][] h, double[] g, double base, double lo, double hi) {
        this.h = h; this.g = g; this.base = base; this.lo = lo; this.hi = hi;
        this.step = 1.0 / (rowSumBound(h) * 1.000001 + 1e-12);   // 1/L with L >= lambda_max(H)
    }
    int size() { return g.length; }
    double value(double[] u) { return 0.5 * dot(u, mulVec(h, u)) + dot(g, u) + base; }
    double clamp(double v) { return Math.min(hi, Math.max(lo, v)); }
    double[] grad(double[] u) {
        double[] out = mulVec(h, u);
        for (int i = 0; i < out.length; i++) out[i] += g[i];
        return out;
    }
    /** FISTA: projected accelerated gradient descent at the fixed safe step 1/L. */
    double[] solve(int iters) {
        int n = size();
        double[] u = new double[n], y = new double[n], next = new double[n];
        double t = 1;
        for (int it = 0; it < iters; it++) {
            double[] gr = grad(y);
            for (int i = 0; i < n; i++) next[i] = clamp(y[i] - step * gr[i]);
            double tn = 0.5 * (1 + Math.sqrt(1 + 4 * t * t)), beta = (t - 1) / tn;
            for (int i = 0; i < n; i++) y[i] = next[i] + beta * (next[i] - u[i]);
            System.arraycopy(next, 0, u, 0, n);
            t = tn;
        }
        return u;
    }
    /** Unconstrained minimiser by Gauss-Jordan elimination - the independent oracle. */
    double[] solveUnconstrained() {
        int n = size(), w = n + 1;
        double[] m = new double[n * w];
        for (int i = 0; i < n; i++) { System.arraycopy(h[i], 0, m, i * w, n); m[i * w + n] = -g[i]; }
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) if (Math.abs(m[r * w + c]) > Math.abs(m[p * w + c])) p = r;
            if (p != c) for (int j = c; j <= n; j++) {
                double tmp = m[c * w + j]; m[c * w + j] = m[p * w + j]; m[p * w + j] = tmp;
            }
            for (int r = 0; r < n; r++) {
                if (r == c || m[r * w + c] == 0) continue;
                double f = m[r * w + c] / m[c * w + c];
                for (int j = c; j <= n; j++) m[r * w + j] -= f * m[c * w + j];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = m[i * w + n] / m[i * w + i];
        return x;
    }
    /** Gershgorin row-sum bound: >= lambda_max for a symmetric PSD matrix, reproducible and cheap. */
    static double rowSumBound(double[][] m) {
        double best = 0;
        for (double[] row : m) {
            double s = 0;
            for (double v : row) s += Math.abs(v);
            best = Math.max(best, s);
        }
        return best;
    }
    static double[] mulVec(double[][] m, double[] v) {
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++) for (int j = 0; j < v.length; j++) out[i] += m[i][j] * v[j];
        return out;
    }
    static double[][] mul(double[][] a, double[][] b) {
        double[][] out = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < b[0].length; j++)
            for (int k = 0; k < b.length; k++) out[i][j] += a[i][k] * b[k][j];
        return out;
    }
    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}

/** Condensed MPC: horizon, weights, and the matrices mapping x0 and u to the whole trajectory. */
final class Mpc {
    static final double DT = 0.1;
    final int n;
    final double umax, r = 0.1;
    final double[] wp, wv;          // per-stage weights, index 1..n; the last stage is the terminal one
    final double[][][] ac, bc;      // ac[i]*x0 and bc[i][k]*u_k contribute to x_i
    final double[][] h;
    Mpc(int horizon, double umax) {
        n = horizon; this.umax = umax;
        wp = new double[n + 1]; wv = new double[n + 1];
        for (int i = 1; i <= n; i++) { boolean term = i == n; wp[i] = term ? 40 : 10; wv[i] = term ? 4 : 1; }
        ac = new double[n + 1][2][2]; bc = new double[n + 1][n][2];
        double[][] a = {{1, DT}, {0, 1}};
        double[] b = {0.5 * DT * DT, DT};
        ac[0] = new double[][] {{1, 0}, {0, 1}};
        for (int i = 1; i <= n; i++) ac[i] = Qp.mul(a, ac[i - 1]);
        for (int i = 1; i <= n; i++) for (int k = 0; k < i; k++) bc[i][k] = Qp.mulVec(ac[i - 1 - k], b);
        h = new double[n][n];
        for (int k = 0; k < n; k++) {
            for (int l = 0; l < n; l++) {
                double s = 0;
                for (int i = 1; i <= n; i++) s += wp[i] * bc[i][k][0] * bc[i][l][0] + wv[i] * bc[i][k][1] * bc[i][l][1];
                h[k][l] = s;
            }
            h[k][k] += r;
        }
    }
    Qp qp(double[] x0, double[] ref) {
        double[] g = new double[n];
        double base = 0;
        for (int i = 1; i <= n; i++) {
            double[] free = Qp.mulVec(ac[i], x0);
            double ep = free[0] - ref[0], ev = free[1] - ref[1];
            base += 0.5 * (wp[i] * ep * ep + wv[i] * ev * ev);
            for (int k = 0; k < n; k++) g[k] += wp[i] * ep * bc[i][k][0] + wv[i] * ev * bc[i][k][1];
        }
        return new Qp(h, g, base, -umax, umax);
    }
    /** The same objective recomputed by rolling the dynamics out - the condensation oracle. */
    double simulateCost(double[] x0, double[] ref, double[] u) {
        double[] x = x0.clone();
        double total = 0;
        for (int i = 1; i <= n; i++) {
            double control = u[i - 1];
            x = new double[] {x[0] + DT * x[1] + 0.5 * DT * DT * control, x[1] + DT * control};
            total += 0.5 * (wp[i] * sq(x[0] - ref[0]) + wv[i] * sq(x[1] - ref[1])) + 0.5 * r * control * control;
        }
        return total;
    }
    /** One step of x_{i+1} = A x_i + B u_i. */
    double[] step(double[] x, double u) { return new double[] {x[0] + DT * x[1] + 0.5 * DT * DT * u, x[1] + DT * u}; }
    static double sq(double v) { return v * v; }
}

public class Solution {
    static final int N = 8, ITERS = 4000;

    public static void main(String[] args) {
        Mpc mpc = new Mpc(N, 1.0);
        double[] x0 = {0.0, 0.0}, ref = {1.0, 0.0};
        Qp qp = mpc.qp(x0, ref);

        double asym = 0;                                        // H is symmetric and positive definite
        for (int k = 0; k < N; k++) for (int l = 0; l < N; l++) asym = Math.max(asym, Math.abs(qp.h[k][l] - qp.h[l][k]));
        double minForm = Double.MAX_VALUE;
        for (int p = 1; p < 4; p++) {
            double[] probe = probe(p, N);
            minForm = Math.min(minForm, Qp.dot(probe, Qp.mulVec(qp.h, probe)) / Qp.dot(probe, probe));
        }
        check(asym < 1e-12 && minForm > 0, "H must be SPD: asymmetry " + asym + ", min Rayleigh quotient " + minForm);

        double worstGap = 0;                                    // the condensation == rolling the dynamics out
        for (int p = 0; p < 4; p++) {
            double[] probe = probe(p, N);
            worstGap = Math.max(worstGap, Math.abs(qp.value(probe) - mpc.simulateCost(x0, ref, probe)));
        }
        check(worstGap < 1e-9, "condensed and simulated objective differ by " + worstGap);

        Qp freeQp = new Mpc(N, 1e6).qp(x0, ref);                // free minimum vs Gaussian elimination
        double[] oracle = freeQp.solveUnconstrained(), freeSolved = freeQp.solve(ITERS);
        double gap = 0, stationarity = 0;
        for (int i = 0; i < N; i++) gap = Math.max(gap, Math.abs(oracle[i] - freeSolved[i]));
        for (double value : freeQp.grad(oracle)) stationarity = Math.max(stationarity, Math.abs(value));
        check(gap < 1e-6 && stationarity < 1e-8, "free solve mismatch: |du| " + gap + ", oracle |grad| " + stationarity);

        double[] u = qp.solve(ITERS), grad = qp.grad(u);         // projected KKT conditions
        int atBound = 0;
        double kkt = 0;
        for (int k = 0; k < N; k++) {
            if (u[k] <= qp.lo + 1e-12) { atBound++; kkt = Math.max(kkt, Math.max(0, -grad[k])); }
            else if (u[k] >= qp.hi - 1e-12) { atBound++; kkt = Math.max(kkt, Math.max(0, grad[k])); }
            else kkt = Math.max(kkt, Math.abs(grad[k]));
            check(u[k] >= qp.lo - 1e-12 && u[k] <= qp.hi + 1e-12, "control " + k + " violates the box: " + u[k]);
        }
        check(kkt < 1e-6 && atBound >= 1, "KKT residual " + kkt + " with " + atBound + " active bounds");

        double bestGain = 0;                                    // no feasible perturbation improves it
        double[] alphas = {1e-4, 1e-2, 0.5, 5.0};
        for (int d = -1; d < 2 * N; d++) {
            double[] dir = new double[N];
            if (d < 0) for (int i = 0; i < N; i++) dir[i] = Math.sin(1.3 * i + 0.7);
            else dir[d / 2] = d % 2 == 0 ? 1 : -1;
            for (double alpha : alphas) {
                double[] trial = new double[N];
                for (int i = 0; i < N; i++) trial[i] = qp.clamp(u[i] + alpha * dir[i]);
                bestGain = Math.min(bestGain, qp.value(trial) - qp.value(u));
            }
        }
        check(bestGain >= -1e-9 && qp.value(u) <= qp.value(new double[N]) + 1e-12,
                "not optimal: a feasible perturbation gained " + (-bestGain));

        Qp oneQp = new Mpc(1, 1.0).qp(x0, ref);                 // boundary: the N = 1 closed form
        double closedForm = oneQp.clamp(-oneQp.g[0] / oneQp.h[0][0]);
        check(Math.abs(oneQp.solve(ITERS)[0] - closedForm) < 1e-9, "N = 1 must match the closed form " + closedForm);

        Mpc zero = new Mpc(0, 1.0);                             // boundary: the degenerate horizon
        Qp zeroQp = zero.qp(x0, ref);
        check(zeroQp.size() == 0 && zeroQp.solve(10).length == 0, "a zero-length horizon has no decision variables");
        check(Math.abs(zeroQp.value(new double[0]) - zero.simulateCost(x0, ref, new double[0])) < 1e-12,
                "the empty horizon cost must be the pure initial-state cost");

        Qp tightQp = new Mpc(N, 1e-3).qp(x0, ref);              // boundary: every control saturated
        double[] tightU = tightQp.solve(ITERS), tightGrad = tightQp.grad(tightU);
        int tightActive = 0;
        double tightKkt = 0;
        for (int k = 0; k < N; k++) {
            if (Math.abs(Math.abs(tightU[k]) - 1e-3) < 1e-12) tightActive++;
            tightKkt = Math.max(tightKkt, tightU[k] > 0 ? Math.max(0, tightGrad[k]) : Math.max(0, -tightGrad[k]));
        }
        check(tightActive == N && tightKkt < 1e-6, "u_max = 1e-3 must saturate all controls: " + tightActive + " active");

        double[] x = {0.0, 0.0};                                // receding-horizon closed loop
        double errorAt10 = 0, worstControl = 0;
        for (int s = 0; s < 60; s++) {
            double control = mpc.qp(x, ref).solve(ITERS)[0];
            worstControl = Math.max(worstControl, Math.abs(control));
            x = mpc.step(x, control);
            if (s == 9) errorAt10 = Math.abs(x[0] - ref[0]);
        }
        double finalError = Math.abs(x[0] - ref[0]);
        check(worstControl <= 1.0 + 1e-9, "the applied control left the box: " + worstControl);
        check(finalError < 0.01 && Math.abs(x[1]) < 0.05 && finalError < errorAt10,
                "the closed loop must converge: step 10 error " + errorAt10 + " -> 60 " + finalError + ", v " + x[1]);
        check(Arrays.equals(new Mpc(N, 1.0).qp(new double[] {0.0, 0.0}, ref).solve(ITERS), u), "must be deterministic");

        System.out.println("condensed QP: horizon " + N + ", u_max 1.0, " + ITERS + " FISTA iterations, KKT "
                + String.format("%.2e", kkt) + ", active bounds " + atBound + ", controls " + Arrays.toString(u));
        System.out.println("free optimum vs Gaussian elimination |du| " + String.format("%.2e", gap) + ", probing margin "
                + String.format("%.2e", -bestGain) + ", closed-loop error " + String.format("%.4f", errorAt10) + " -> "
                + String.format("%.5f", finalError));
        System.out.println("PASS");
    }

    /** Deterministic probe vectors: zero, all-ones, alternating, sinusoid. */
    static double[] probe(int p, int n) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = p == 0 ? 0 : p == 1 ? 1 : p == 2 ? (i % 2 == 0 ? 1 : -1) : Math.sin(1.7 * i);
        return v;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
