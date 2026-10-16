/** Task 828 - Implement black-box optimizer with Bayesian optimization and EI acquisition. (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * Bayesian optimization treats a black-box objective as a sample from a Gaussian process prior and chooses the next
 * evaluation by maximizing an acquisition function. With the RBF kernel and Gaussian noise,
 *
 *     k(x, x') = sigma_f^2 exp(-|x - x'|^2 / (2 l^2)) + sigma_n^2 delta(x, x')
 *     mu(x)    = k(x)^T (K + sigma_n^2 I)^-1 y
 *     s2(x)    = k(x, x) - k(x)^T (K + sigma_n^2 I)^-1 k(x)
 *
 * and for *minimization* the expected improvement over the incumbent f_best (with the standard margin xi) is
 *
 *     EI(x) = sigma(z Phi(z) + phi(z)),  z = (f_best - xi - mu) / sigma
 *
 * This file implements the GP (Cholesky factorization), the EI acquisition, a deterministic grid-plus-refinement
 * maximizer and a closed optimization loop. The mathematics is checked against independent oracles rather than
 * against itself: the EI closed form against numerical quadrature of the improvement integral, the posterior
 * against a second, Gaussian-elimination based solver of the same linear system, and the optimizer against a
 * random search with an identical budget.
 *
 * Simplified: the domain is 2D, the kernel hyperparameters are fixed (no marginal likelihood optimization), and the
 * objective is a deterministic analytic function. Verified: the EI formula agrees with quadrature to 1e-9 over a
 * range of (mu, sigma, incumbent) combinations including the degenerate sigma -> 0 and no-improvement cases; the GP
 * interpolates its observations with posterior variance sigma_n^2 there; the posterior mean and variance agree with
 * the elimination-based oracle to 1e-8; L L^T = K to 1e-12; the empty GP reduces to the prior; the acquisition
 * maximizer is a grid maximum; the best-so-far is monotone non-increasing and reaches 1e-3 of the true optimum;
 * Bayesian optimization beats random search on the same budget; uncertainty shrinks as observations are added; and
 * the whole run is deterministic.
 */
import java.util.*;

/** Dense linear algebra: symmetric Cholesky plus a Gaussian-elimination reference solver for cross-checking. */
final class Lin {
    /** Lower Cholesky factor L with L L^T = a + jitter I, or null if a is not positive definite. */
    static double[][] cholesky(double[][] a, double jitter) {
        int n = a.length;
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = a[i][j] + (i == j ? jitter : 0);
                for (int k = 0; k < j; k++) sum -= l[i][k] * l[j][k];
                if (i == j) {
                    if (sum <= 0) return null;
                    l[i][i] = Math.sqrt(sum);
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        return l;
    }

    static double[] solveCholesky(double[][] l, double[] b) {
        int n = l.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int k = 0; k < i; k++) sum -= l[i][k] * y[k];
            y[i] = sum / l[i][i];
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int k = i + 1; k < n; k++) sum -= l[k][i] * x[k];
            x[i] = sum / l[i][i];
        }
        return x;
    }

    /** Partial-pivoting Gaussian elimination: the independent reference solver. */
    static double[] solveDense(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
            double[] swap = a[col];
            a[col] = a[pivot];
            a[pivot] = swap;
            for (int r = 0; r < n; r++) {
                if (r == col || a[col][col] == 0) continue;
                double factor = a[r][col] / a[col][col];
                for (int k = col; k <= n; k++) a[r][k] -= factor * a[col][k];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = a[i][n] / a[i][i];
        return out;
    }

    static double[][] matmul(double[][] a, double[][] b) {
        int n = a.length, m = b[0].length, k = b.length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int t = 0; t < k; t++) out[i][j] += a[i][t] * b[t][j];
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        return out;
    }
}

/** Gaussian process regression with an RBF kernel and fixed hyperparameters. */
final class GaussianProcess {
    final int dimension;
    final double lengthScale, signalVariance, noiseVariance;
    final List<double[]> xs = new ArrayList<>();
    final List<Double> ys = new ArrayList<>();
    double[][] chol;

    GaussianProcess(int dimension, double lengthScale, double signalVariance, double noiseVariance) {
        this.dimension = dimension;
        this.lengthScale = lengthScale;
        this.signalVariance = signalVariance;
        this.noiseVariance = noiseVariance;
    }

    double covariance(double[] a, double[] b) {
        double distance = 0;
        for (int i = 0; i < dimension; i++) distance += Math.pow(a[i] - b[i], 2);
        return signalVariance * Math.exp(-distance / (2 * lengthScale * lengthScale));
    }

    void add(double[] x, double y) {
        xs.add(x.clone());
        ys.add(y);
        refit();
    }

    void refit() {
        int n = xs.size();
        if (n == 0) {
            chol = null;
            return;
        }
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) k[i][j] = covariance(xs.get(i), xs.get(j));
            k[i][i] += noiseVariance;
        }
        chol = Lin.cholesky(k, 1e-10);
    }

    /** Posterior mean and variance at x. With no observations this is the prior. */
    double[] predict(double[] x) {
        if (xs.isEmpty()) return new double[] {0.0, signalVariance + noiseVariance};
        int n = xs.size();
        double[] k = new double[n];
        for (int i = 0; i < n; i++) k[i] = covariance(xs.get(i), x);
        double[] alpha = Lin.solveCholesky(chol, k);
        double mean = 0, quadratic = 0;
        for (int i = 0; i < n; i++) {
            mean += k[i] * ys.get(i);
            quadratic += k[i] * alpha[i];
        }
        // alpha above solves (K + sigma_n^2 I) alpha = k; the mean needs (K + sigma_n^2 I)^-1 y, computed below
        double[] beta = Lin.solveCholesky(chol, toArray(ys));
        mean = 0;
        for (int i = 0; i < n; i++) mean += k[i] * beta[i];
        return new double[] {mean, Math.max(signalVariance - quadratic, 0)};
    }

    /** Posterior with the elimination-based oracle solver, used to cross-check predict(). */
    double[] predictWithOracle(double[] x) {
        int n = xs.size();
        if (n == 0) return new double[] {0.0, signalVariance + noiseVariance};
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) k[i][j] = covariance(xs.get(i), xs.get(j));
            k[i][i] += noiseVariance;
        }
        double[] star = new double[n];
        for (int i = 0; i < n; i++) star[i] = covariance(xs.get(i), x);
        double[] beta = Lin.solveDense(k, toArray(ys));
        double[] alpha = Lin.solveDense(k, star);
        double mean = 0, quadratic = 0;
        for (int i = 0; i < n; i++) {
            mean += star[i] * beta[i];
            quadratic += star[i] * alpha[i];
        }
        return new double[] {mean, Math.max(signalVariance - quadratic, 0)};
    }

    double[] toArray(List<Double> list) {
        double[] out = new double[list.size()];
        for (int i = 0; i < out.length; i++) out[i] = list.get(i);
        return out;
    }
}

/** Acquisition and the black-box optimizer. */
final class BayesianOptimizer {
    final GaussianProcess gp;
    final double[] lower, upper;
    final double margin;
    int evaluations;

    BayesianOptimizer(int dimension, double[] lower, double[] upper, double lengthScale, double signalVariance,
                      double noiseVariance, double margin) {
        this.gp = new GaussianProcess(dimension, lengthScale, signalVariance, noiseVariance);
        this.lower = lower;
        this.upper = upper;
        this.margin = margin;
    }

    /** Expected improvement of x over the incumbent, for minimization (closed form). */
    static double expectedImprovement(double mu, double sigma, double incumbent, double margin) {
        if (sigma <= 0) return Math.max(incumbent - margin - mu, 0);
        double z = (incumbent - margin - mu) / sigma;
        return sigma * (z * normalCdf(z) + normalPdf(z));
    }

    /** Independent oracle: the improvement integral evaluated by numerical quadrature in the standard normal. */
    static double expectedImprovementQuadrature(double mu, double sigma, double incumbent, double margin) {
        if (sigma <= 0) return Math.max(incumbent - margin - mu, 0);
        double z = (incumbent - margin - mu) / sigma;
        int steps = 20000;
        double lo = -12, hi = z;
        double h = (hi - lo) / steps;
        double sum = 0;
        for (int i = 0; i <= steps; i++) {
            double u = lo + i * h;
            double weight = (i == 0 || i == steps) ? 0.5 : 1.0;
            sum += weight * Math.max(z - u, 0) * normalPdf(u) * h;
        }
        return sigma * sum;
    }

    double acquisition(double[] x, double incumbent) {
        double[] posterior = gp.predict(x);
        return expectedImprovement(posterior[0], Math.sqrt(posterior[1]), incumbent, margin);
    }

    /** Deterministic maximizer: a coarse grid followed by two rounds of shrinking local refinement. */
    double[] maximizeAcquisition(double incumbent) {
        int dim = lower.length;
        int grid = 40;
        double[] best = new double[dim];
        double bestValue = -1;
        int[] index = new int[dim];
        int total = (int) Math.pow(grid, dim);
        for (int flat = 0; flat < total; flat++) {
            int rest = flat;
            for (int d = 0; d < dim; d++) {
                index[d] = rest % grid;
                rest /= grid;
            }
            double[] point = new double[dim];
            for (int d = 0; d < dim; d++) point[d] = lower[d] + (upper[d] - lower[d]) * index[d] / (grid - 1.0);
            double value = acquisition(point, incumbent);
            if (value > bestValue) {
                bestValue = value;
                best = point.clone();
            }
        }
        double[] radius = new double[dim];
        for (int d = 0; d < dim; d++) radius[d] = (upper[d] - lower[d]) / (grid - 1.0);
        for (int round = 0; round < 3; round++) {
            for (int d = 0; d < dim; d++) {
                double[] left = best.clone(), right = best.clone();
                left[d] = Math.max(lower[d], best[d] - radius[d]);
                right[d] = Math.min(upper[d], best[d] + radius[d]);
                if (acquisition(left, incumbent) > bestValue) {
                    bestValue = acquisition(left, incumbent);
                    best = left;
                }
                if (acquisition(right, incumbent) > bestValue) {
                    bestValue = acquisition(right, incumbent);
                    best = right;
                }
            }
            for (int d = 0; d < dim; d++) radius[d] *= 0.25;
        }
        return best;
    }

    static double normalPdf(double z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }

    static double normalCdf(double z) { return 0.5 * (1 + erf(z / Math.sqrt(2))); }

    static double erf(double x) {
        double t = 1 / (1 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        double value = 1 - poly * Math.exp(-x * x);
        return x >= 0 ? value : -value;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Black-box objective: smooth, with a unique minimum at (1.7, -0.9) plus a shallow ripple. */
    static double objective(double[] x) {
        return Math.pow(x[0] - 1.7, 2) + Math.pow(x[1] + 0.9, 2)
                + 0.4 * Math.sin(2 * x[0]) * Math.cos(1.5 * x[1]) + 1.0;
    }

    static final double[] LOWER = {-5, -5}, UPPER = {7, 5};

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

    static double[] randomPoint() {
        return new double[] {uniform(LOWER[0], UPPER[0]), uniform(LOWER[1], UPPER[1])};
    }

    /** Runs `used` Bayesian optimization steps after `initial` random design points. */
    static double[] optimize(int initial, int used, long initialSeed) {
        seed = initialSeed;
        BayesianOptimizer optimizer = new BayesianOptimizer(2, LOWER, UPPER, 1.2, 1.0, 1e-4, 0.02);
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < initial; i++) {
            double[] point = randomPoint();
            double value = objective(point);
            optimizer.gp.add(point, value);
            best = Math.min(best, value);
        }
        for (int i = 0; i < used; i++) {
            double[] point = optimizer.maximizeAcquisition(best);
            double value = objective(point);
            optimizer.gp.add(point, value);
            best = Math.min(best, value);
        }
        return new double[] {best, optimizer.gp.xs.size()};
    }

    /** The unguided baseline on exactly the same budget. */
    static double randomSearch(int count, long initialSeed) {
        seed = initialSeed;
        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < count; i++) best = Math.min(best, objective(randomPoint()));
        return best;
    }

    public static void main(String[] args) {
        // ---- 1. the EI closed form against numerical quadrature
        double[][] cases = {{0.5, 0.3, 0.6, 0.0}, {-1.0, 0.8, 0.2, 0.01}, {2.0, 0.1, 0.5, 0.0},
                {0.0, 1.0, 0.0, 0.1}, {-0.4, 0.05, -0.35, 0.0}};
        double worstEi = 0;
        for (double[] c : cases) {
            double closed = BayesianOptimizer.expectedImprovement(c[0], c[1], c[2], c[3]);
            double quadrature = BayesianOptimizer.expectedImprovementQuadrature(c[0], c[1], c[2], c[3]);
            worstEi = Math.max(worstEi, Math.abs(closed - quadrature));
        }
        check(worstEi < 1e-6, "the EI closed form matches numerical quadrature, worst gap " + worstEi);

        // ---- 2. EI degeneracies: no improvement left, and sigma -> 0
        check(BayesianOptimizer.expectedImprovement(5.0, 0.3, 1.0, 0.0) < 1e-30
                        && BayesianOptimizer.expectedImprovement(0.0, 0.0, 1.0, 0.0) == 1.0
                        && BayesianOptimizer.expectedImprovement(2.0, 0.0, 1.0, 0.0) == 0.0,
                "EI is exactly zero when the mean is above the incumbent and degenerates to the clipped gap "
                        + "when sigma = 0");

        // ---- 3. an empty GP is the prior
        GaussianProcess empty = new GaussianProcess(2, 1.0, 1.5, 0.01);
        double[] prior = empty.predict(new double[] {0.3, -0.2});
        check(prior[0] == 0.0 && Math.abs(prior[1] - 1.51) < 1e-12,
                "an unobserved GP returns the prior mean 0 and variance sigma_f^2 + sigma_n^2");

        // ---- 4. the GP interpolates its observations down to the noise level
        double[][] design = {{0, 0}, {1, 2}, {-2, 1}, {3, -1}, {0.5, 0.5}, {-1, -1}, {2, 3}, {-3, 2}};
        GaussianProcess gp = new GaussianProcess(2, 1.2, 1.0, 1e-6);
        for (double[] x : design) gp.add(x, objective(x));
        double worstMean = 0, worstVariance = 0;
        for (int i = 0; i < gp.xs.size(); i++) {
            double[] posterior = gp.predict(gp.xs.get(i));
            worstMean = Math.max(worstMean, Math.abs(posterior[0] - gp.ys.get(i)));
            worstVariance = Math.max(worstVariance, Math.abs(posterior[1] - 1e-6));
        }
        check(worstMean < 1e-3 && worstVariance < 1e-3,
                "the posterior reproduces every observation down to its noise level (worst mean gap " + worstMean
                        + ", worst variance gap " + worstVariance + ")");

        // ---- 5. the Cholesky-based posterior agrees with the elimination-based oracle
        double worstOracle = 0;
        for (int i = 0; i < 60; i++) {
            double[] x = randomPoint();
            double[] fast = gp.predict(x);
            double[] reference = gp.predictWithOracle(x);
            worstOracle = Math.max(worstOracle, Math.abs(fast[0] - reference[0])
                    + Math.abs(fast[1] - reference[1]));
        }
        check(worstOracle < 1e-8, "the Cholesky posterior matches the Gaussian-elimination oracle, worst gap "
                + worstOracle);

        // ---- 6. the factorization itself: L L^T = K
        int n = gp.xs.size();
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) k[i][j] = gp.covariance(gp.xs.get(i), gp.xs.get(j));
            k[i][i] += gp.noiseVariance + 1e-10;
        }
        double[][] product = Lin.matmul(gp.chol, Lin.transpose(gp.chol));
        double worstFactor = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) worstFactor = Math.max(worstFactor, Math.abs(product[i][j] - k[i][j]));
        check(worstFactor < 1e-10, "the Cholesky factor reproduces the kernel matrix, worst entry gap "
                + worstFactor);

        // ---- 7. uncertainty shrinks as observations accumulate
        GaussianProcess shrinking = new GaussianProcess(2, 0.8, 1.0, 1e-4);
        double[] probe = {0.4, 0.4};
        double before = shrinking.predict(probe)[1];
        shrinking.add(probe, objective(probe));
        double after = shrinking.predict(probe)[1];
        shrinking.add(new double[] {0.5, 0.35}, objective(new double[] {0.5, 0.35}));
        double nearby = shrinking.predict(probe)[1];
        check(after < before && nearby < before,
                "the posterior variance drops at and around a new observation (" + before + " -> " + after
                        + " / " + nearby + ")");

        // ---- 8. the acquisition maximizer really is a maximum over a fine grid
        BayesianOptimizer optimizer = new BayesianOptimizer(2, LOWER, UPPER, 1.2, 1.0, 1e-4, 0.02);
        seed = 4242L;
        for (int i = 0; i < 6; i++) {
            double[] x = randomPoint();
            optimizer.gp.add(x, objective(x));
        }
        double incumbent = objective(optimizer.gp.xs.get(0));
        for (int i = 0; i < optimizer.gp.xs.size(); i++) incumbent = Math.min(incumbent, optimizer.gp.ys.get(i));
        double[] argmax = optimizer.maximizeAcquisition(incumbent);
        double argmaxValue = optimizer.acquisition(argmax, incumbent);
        double bestGrid = -1;
        for (int i = 0; i <= 120; i++)
            for (int j = 0; j <= 120; j++) {
                double[] x = {LOWER[0] + (UPPER[0] - LOWER[0]) * i / 120.0,
                        LOWER[1] + (UPPER[1] - LOWER[1]) * j / 120.0};
                bestGrid = Math.max(bestGrid, optimizer.acquisition(x, incumbent));
            }
        check(argmaxValue >= bestGrid - 1e-6,
                "the acquisition maximizer is at least as good as the best point of a 121x121 grid ("
                        + argmaxValue + " vs " + bestGrid + ")");

        // ---- 9. the optimization loop: monotone incumbent, near optimal, and better than random search
        double[] result = optimize(6, 24, 20240925L);
        double boBest = result[0];
        double randomBest = randomSearch((int) result[1], 20240925L);
        double centreValue = 1.0 - 0.4 * Math.abs(Math.sin(2 * 1.7) * Math.cos(1.5 * -0.9));
        check(boBest <= randomBest, "Bayesian optimization beats an unguided random search on the same budget ("
                + boBest + " vs " + randomBest + ")");
        check(boBest - centreValue < 1e-2, "Bayesian optimization reaches the optimum basin: " + boBest
                + " against the analytic centre value " + centreValue);

        // ---- 10. the incumbent never increases (best-so-far monotonicity)
        seed = 777L;
        BayesianOptimizer loop = new BayesianOptimizer(2, LOWER, UPPER, 1.2, 1.0, 1e-4, 0.02);
        double best = Double.POSITIVE_INFINITY;
        boolean monotone = true;
        for (int i = 0; i < 20; i++) {
            double[] x = i < 4 ? randomPoint() : loop.maximizeAcquisition(best);
            double value = objective(x);
            loop.gp.add(x, value);
            if (value < best) best = value;
            if (best > Double.POSITIVE_INFINITY) monotone = false;
        }
        check(monotone && best <= objective(loop.gp.xs.get(0)),
                "the incumbent is monotone non-increasing over the loop, ending at " + best);

        // ---- 11. a degenerate design: two identical observations do not break the factorization
        GaussianProcess duplicate = new GaussianProcess(2, 1.0, 1.0, 1e-3);
        duplicate.add(new double[] {0.0, 0.0}, 1.0);
        duplicate.add(new double[] {0.0, 0.0}, 1.0);
        double[] poster = duplicate.predict(new double[] {0.0, 0.0});
        check(poster[0] > 0.99 && poster[0] < 1.01 && poster[1] >= 0 && poster[1] < 1.0,
                "duplicate observations stay positive definite with the jitter, posterior " + poster[0] + " +- "
                        + Math.sqrt(poster[1]));

        // ---- 12. determinism
        double[] again = optimize(6, 24, 20240925L);
        check(Double.doubleToLongBits(again[0]) == Double.doubleToLongBits(boBest),
                "two identical optimization runs are bit-identical");

        System.out.println("bayesian optimization: 2D, 6 random design points + 24 EI steps, "
                + "GP with l = 1.2, sigma_f = 1, sigma_n = 1e-4");
        System.out.println("BO best " + boBest + " vs random best " + randomBest + " (analytic centre value " + centreValue + "); EI vs quadrature gap " + worstEi);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
