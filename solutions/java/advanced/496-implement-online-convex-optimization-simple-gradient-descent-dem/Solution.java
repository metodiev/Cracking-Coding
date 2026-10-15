/** Task 496 - Implement online convex optimization simple gradient descent demo. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Two gradient-descent demos. (1) Offline GD on the fixed convex quadratic
 * f(x) = 0.5 x^T A x - b^T x with the SPD matrix A = [[3,1,0],[1,3,1],[0,1,3]] and b = [1,-2,0.5]:
 * with the documented step size eta = 1/L, where L is the largest eigenvalue of A (power
 * iteration), f decreases monotonically every iteration - the proof is one line, f(x - eta g) <=
 * f(x) - (eta - 0.5 eta^2 L)||g||^2 = f(x) - ||g||^2/(2L) - and after 300 iterations x is within
 * 1e-6 (actual: ~5e-16) of the exact minimizer A^-1 b computed by Gaussian elimination. (2) An
 * online convex optimization loop: the loss stream f_t(x) = (x - z_t)^2 for LCG targets z_t in
 * [-1, 1], the 1/sqrt(t) step schedule, and the pseudo-regret R(T) = sum f_t(x_t) - min_x sum
 * f_t(x) against the best fixed comparator (the mean of the targets): R stays under the
 * documented bound of 100 (measured 27.14 on the fixed stream, R/T ~ 0.014), the per-step loss
 * rate of the second half (0.339) is below the first half's (0.369), and the iterate tracks the
 * running mean within the documented 0.1.
 */
import java.util.*;

public class Solution {
    static final double[][] A = {{3.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 3.0}};
    static final double[] B = {1.0, -2.0, 0.5};
    static final double[] X0 = {2.0, 2.0, 2.0};
    static final int OFFLINE_STEPS = 300;
    static final int ONLINE_TRIALS = 2000;
    static final int ONLINE_SEED = 2024;

    static double dot(double[] u, double[] v) {
        double total = 0.0;
        for (int i = 0; i < u.length; i++) total += u[i] * v[i];
        return total;
    }

    static double[] matvec(double[][] matrix, double[] vector) {
        double[] out = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) out[i] = dot(matrix[i], vector);
        return out;
    }

    static double objective(double[][] matrix, double[] rhs, double[] x) {
        return 0.5 * dot(x, matvec(matrix, x)) - dot(rhs, x);
    }

    static double[] gradient(double[][] matrix, double[] rhs, double[] x) {
        double[] ax = matvec(matrix, x);
        for (int i = 0; i < ax.length; i++) ax[i] -= rhs[i];
        return ax;
    }

    /** Gaussian elimination with partial pivoting - the exact-minimizer oracle. */
    static double[] solveLinear(double[][] matrix, double[] rhs) {
        int n = matrix.length;
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, aug[i], 0, n);
            aug[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;
            }
            double[] tmp = aug[col];
            aug[col] = aug[pivot];
            aug[pivot] = tmp;
            for (int r = col + 1; r < n; r++) {
                double factor = aug[r][col] / aug[col][col];
                for (int c = col; c <= n; c++) aug[r][c] -= factor * aug[col][c];
            }
        }
        double[] x = new double[n];
        for (int r = n - 1; r >= 0; r--) {
            double acc = aug[r][n];
            for (int c = r + 1; c < n; c++) acc -= aug[r][c] * x[c];
            x[r] = acc / aug[r][r];
        }
        return x;
    }

    /** Power iteration from the all-ones vector: largest eigenvalue of a symmetric matrix. */
    static double largestEigenvalue(double[][] matrix, int iterations) {
        double[] v = new double[matrix.length];
        Arrays.fill(v, 1.0);
        double norm = Math.sqrt(dot(v, v));
        for (int i = 0; i < v.length; i++) v[i] /= norm;
        for (int i = 0; i < iterations; i++) {
            double[] w = matvec(matrix, v);
            norm = Math.sqrt(dot(w, w));
            for (int j = 0; j < v.length; j++) v[j] = w[j] / norm;
        }
        return dot(v, matvec(matrix, v));
    }

    static class GdResult {
        final double[] x;
        final double[] values;

        GdResult(double[] x, double[] values) {
            this.x = x;
            this.values = values;
        }
    }

    static GdResult gdOffline(double eta, int steps) {
        double[] x = X0.clone();
        double[] values = new double[steps + 1];
        values[0] = objective(A, B, x);
        for (int i = 0; i < steps; i++) {
            double[] g = gradient(A, B, x);
            for (int j = 0; j < x.length; j++) x[j] -= eta * g[j];
            values[i + 1] = objective(A, B, x);
        }
        return new GdResult(x, values);
    }

    static class OnlineResult {
        final double x;
        final double[] zs;
        final double[] losses;
        final double regret;
        final double meanZ;

        OnlineResult(double x, double[] zs, double[] losses, double regret, double meanZ) {
            this.x = x;
            this.zs = zs;
            this.losses = losses;
            this.regret = regret;
            this.meanZ = meanZ;
        }
    }

    static OnlineResult onlineLoop(int trials, int seed) {
        double[] zs = new double[trials];
        for (int i = 0; i < trials; i++) {
            seed = (seed * 75 + 74) % 65537;
            zs[i] = (seed % 2001) / 1000.0 - 1.0;
        }
        double x = 0.0;
        double[] losses = new double[trials];
        double meanZ = 0.0;
        for (double z : zs) meanZ += z;
        meanZ /= trials;
        for (int t = 1; t <= trials; t++) {
            double z = zs[t - 1];
            losses[t - 1] = (x - z) * (x - z);
            double g = 2.0 * (x - z);
            double eta = 1.0 / Math.sqrt(t);        // documented schedule eta_t = 1/sqrt(t)
            x -= eta * g;
        }
        double comparator = 0.0;
        for (double z : zs) comparator += (meanZ - z) * (meanZ - z);
        double regret = 0.0;
        for (double loss : losses) regret += loss;
        regret -= comparator;
        return new OnlineResult(x, zs, losses, regret, meanZ);
    }

    public static void main(String[] args) {
        double lambdaMax = largestEigenvalue(A, 100);
        double eta = 1.0 / lambdaMax;
        check(lambdaMax > 4.0 && lambdaMax < 5.0, "sanity: the documented spectrum of A");

        double[] xStar = solveLinear(A, B);
        double fStar = objective(A, B, xStar);
        check(maxAbs(gradient(A, B, xStar)) <= 1e-12, "Gaussian oracle is exact");

        GdResult offline = gdOffline(eta, OFFLINE_STEPS);
        // eta = 1/L gives f(x - eta g) <= f(x) - ||g||^2/(2L) exactly; in floating point the
        // inequality can be violated by mere rounding (<= 1e-12 relative) once the objective is
        // flat at f*, so monotonicity is asserted with that documented rounding allowance.
        for (int i = 1; i < offline.values.length; i++) {
            double previous = offline.values[i - 1];
            double current = offline.values[i];
            if (Math.abs(previous - fStar) > 1e-9) {
                check(current < previous, "strict decrease until f is within 1e-9 of f*");
            }
            check(current <= previous + 1e-12 * Math.max(1.0, Math.abs(previous)),
                    "monotone decrease");
        }
        check(maxAbs(sub(offline.x, xStar)) <= 1e-6, "converges to the minimizer");
        check(Math.abs(objective(A, B, offline.x) - fStar) <= 1e-9, "converges to f*");
        GdResult big = gdOffline(1.0, 25);
        check(max(big.values) > fStar + 1e-6, "eta = 1 overshoots (control experiment)");

        OnlineResult online = onlineLoop(ONLINE_TRIALS, ONLINE_SEED);
        check(online.regret >= 0.0, "regret non-negative");
        check(online.regret <= 100.0, "documented online regret bound");
        int half = ONLINE_TRIALS / 2;
        double firstRate = 0.0;
        for (int i = 0; i < half; i++) firstRate += online.losses[i];
        firstRate /= half;
        double secondRate = 0.0;
        for (int i = half; i < ONLINE_TRIALS; i++) secondRate += online.losses[i];
        secondRate /= (ONLINE_TRIALS - half);
        check(secondRate < firstRate, "documented: regret rate decreases");
        check(online.regret / ONLINE_TRIALS <= 0.05, "documented: vanishing average regret");
        check(Math.abs(online.x - online.meanZ) <= 0.1, "the iterate tracks the running mean");
        check(min(online.losses) >= 0.0 && max(online.zs) <= 1.0 && min(online.zs) >= -1.0,
                "target and loss ranges");

        System.out.println("PASS");
    }

    static double[] sub(double[] u, double[] v) {
        double[] out = new double[u.length];
        for (int i = 0; i < u.length; i++) out[i] = u[i] - v[i];
        return out;
    }

    static double maxAbs(double[] values) {
        double best = 0.0;
        for (double value : values) best = Math.max(best, Math.abs(value));
        return best;
    }

    static double max(double[] values) {
        double best = Double.NEGATIVE_INFINITY;
        for (double value : values) best = Math.max(best, value);
        return best;
    }

    static double min(double[] values) {
        double best = Double.POSITIVE_INFINITY;
        for (double value : values) best = Math.min(best, value);
        return best;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
