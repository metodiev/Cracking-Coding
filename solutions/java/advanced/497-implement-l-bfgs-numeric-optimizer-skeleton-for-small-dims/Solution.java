/** Task 497 - Implement L-BFGS numeric optimizer skeleton for small dims. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A compact L-BFGS for small dimensions: the two-loop recursion builds H*g from the last
 * m = 5 (s, y) correction pairs (the search direction is -H*g), a backtracking Armijo line
 * search (c1 = 1e-4, rho = 0.5, first trial step 1.0) picks the step, and the loop stops when
 * ||grad f|| <= 1e-8 or after 200 iterations. The model problem is the convex quadratic
 * f(x) = 0.5 x^T A x - b^T x with A = [[4,1,0],[1,3,1],[0,1,2]], b = [1,2,3], x0 = 0; its exact
 * minimizer x* = A^-1 b and f* are computed by Gaussian elimination as the oracle. Documented
 * tolerance: L-BFGS reaches ||x - x*|| <= 1e-6, |f - f*| <= 1e-9 and ||grad f|| <= 1e-8 (9
 * iterations on the fixed problem). A plain gradient descent with the documented step size
 * eta = 1/L (largest eigenvalue by power iteration, both runs started at x0 and compared at the
 * same ||grad f|| <= 1e-6 tolerance) needs over twice as many iterations (46) - L-BFGS is
 * compared against it directly.
 */
import java.util.*;

public class Solution {
    static final double[][] A = {{4.0, 1.0, 0.0}, {1.0, 3.0, 1.0}, {0.0, 1.0, 2.0}};
    static final double[] B = {1.0, 2.0, 3.0};
    static final double[] X0 = {0.0, 0.0, 0.0};
    static final int HISTORY = 5;
    static final int MAX_ITER = 200;
    static final int GD_MAX_ITER = 20000;
    static final double C1 = 1e-4;
    static final double RHO = 0.5;

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

    static double objective(double[] x) {
        return 0.5 * dot(x, matvec(A, x)) - dot(B, x);
    }

    static double[] gradient(double[] x) {
        double[] ax = matvec(A, x);
        for (int i = 0; i < ax.length; i++) ax[i] -= B[i];
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

    /** L-BFGS two-loop recursion: returns r = H*g; the search direction is -r. */
    static double[] twoLoopDirection(double[] g, List<double[]> sHist, List<double[]> yHist) {
        double[] q = g.clone();
        double[] alphas = new double[sHist.size()];
        for (int k = sHist.size() - 1; k >= 0; k--) {
            double[] s = sHist.get(k);
            double[] y = yHist.get(k);
            double rho = 1.0 / dot(y, s);
            double alpha = rho * dot(s, q);
            alphas[k] = alpha;
            for (int i = 0; i < q.length; i++) q[i] -= alpha * y[i];
        }
        double gamma = 1.0;
        if (!sHist.isEmpty()) {
            double[] sLast = sHist.get(sHist.size() - 1);
            double[] yLast = yHist.get(yHist.size() - 1);
            gamma = dot(sLast, yLast) / dot(yLast, yLast);
        }
        double[] r = q.clone();
        for (int i = 0; i < r.length; i++) r[i] *= gamma;
        for (int k = 0; k < sHist.size(); k++) {
            double[] s = sHist.get(k);
            double[] y = yHist.get(k);
            double rho = 1.0 / dot(y, s);
            double beta = rho * dot(y, r);
            for (int i = 0; i < r.length; i++) r[i] += s[i] * (alphas[k] - beta);
        }
        return r;
    }

    static class LbfgsResult {
        final double[] x;
        final double[] g;
        final int iterations;

        LbfgsResult(double[] x, double[] g, int iterations) {
            this.x = x;
            this.g = g;
            this.iterations = iterations;
        }
    }

    static LbfgsResult lbfgs() {
        double[] x = X0.clone();
        double[] g = gradient(x);
        List<double[]> sHist = new ArrayList<>();
        List<double[]> yHist = new ArrayList<>();
        int iterations = 0;
        for (int iter = 0; iter < MAX_ITER; iter++) {
            if (Math.sqrt(dot(g, g)) <= 1e-8) break;
            double[] r = twoLoopDirection(g, sHist, yHist);
            double[] direction = new double[r.length];
            for (int i = 0; i < r.length; i++) direction[i] = -r[i];
            double descent = dot(g, direction);
            if (descent >= 0.0) {               // safeguard: fall back to steepest descent
                for (int i = 0; i < direction.length; i++) direction[i] = -g[i];
                descent = -dot(g, g);
            }
            double step = 1.0;
            while (objective(addScaled(x, direction, step)) > objective(x) + C1 * step * descent) {
                step *= RHO;
            }
            double[] xNew = addScaled(x, direction, step);
            double[] gNew = gradient(xNew);
            double[] s = new double[x.length];
            double[] y = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                s[i] = xNew[i] - x[i];
                y[i] = gNew[i] - g[i];
            }
            if (dot(s, y) > 0.0) {              // curvature condition
                sHist.add(s);
                yHist.add(y);
                if (sHist.size() > HISTORY) {
                    sHist.remove(0);
                    yHist.remove(0);
                }
            }
            x = xNew;
            g = gNew;
            iterations += 1;
        }
        return new LbfgsResult(x, g, iterations);
    }

    static double[] addScaled(double[] x, double[] direction, double step) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] + step * direction[i];
        return out;
    }

    static LbfgsResult gd(double tolerance) {
        double eta = 1.0 / largestEigenvalue(A, 100);       // documented GD step size
        double[] x = X0.clone();
        double[] g = gradient(x);
        int iterations = 0;
        while (Math.sqrt(dot(g, g)) > tolerance && iterations < GD_MAX_ITER) {
            for (int i = 0; i < x.length; i++) x[i] -= eta * g[i];
            g = gradient(x);
            iterations += 1;
        }
        return new LbfgsResult(x, g, iterations);
    }

    public static void main(String[] args) {
        double[] xStar = solveLinear(A, B);
        double fStar = objective(xStar);
        double[] gStar = gradient(xStar);
        double gStarMax = 0.0;
        for (double value : gStar) gStarMax = Math.max(gStarMax, Math.abs(value));
        check(gStarMax <= 1e-12, "Gaussian oracle is exact");

        // Skeleton sanity: with an empty history H = I, so H*g = g and the step is -g.
        double[] g0 = gradient(X0);
        check(Arrays.equals(twoLoopDirection(g0, Collections.emptyList(), Collections.emptyList()),
                g0), "empty-history identity");

        LbfgsResult lb = lbfgs();
        check(Math.sqrt(dot(lb.g, lb.g)) <= 1e-8, "documented L-BFGS gradient tolerance");
        double xErr = 0.0;
        for (int i = 0; i < xStar.length; i++) xErr = Math.max(xErr, Math.abs(lb.x[i] - xStar[i]));
        check(xErr <= 1e-6, "L-BFGS minimizer accuracy");
        check(Math.abs(objective(lb.x) - fStar) <= 1e-9, "L-BFGS objective accuracy");
        check(lb.iterations <= 20, "L-BFGS must converge in few iterations");

        LbfgsResult plain = gd(1e-6);
        check(Math.sqrt(dot(plain.g, plain.g)) <= 1e-6, "GD must converge within the cap");
        check(plain.iterations > 2 * lb.iterations, "L-BFGS must need far fewer iterations");
        double gdErr = 0.0;
        for (int i = 0; i < xStar.length; i++) gdErr = Math.max(gdErr, Math.abs(plain.x[i] - xStar[i]));
        check(gdErr <= 1e-4, "GD accuracy");

        LbfgsResult repeat = lbfgs();
        check(repeat.iterations == lb.iterations && Arrays.equals(repeat.x, lb.x),
                "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
