/** Task 593 - Implement L1-regularized least squares via coordinate descent. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Lasso: minimize f(x) = 0.5 * ||y - A x||_2^2 + lambda * ||x||_1. The file solves it by cyclic coordinate
 * descent with the exact univariate minimizer, i.e. the soft-thresholding operator
 * x_j <- S_lambda(c_j) / ||a_j||^2 with c_j = a_j^T r + x_j ||a_j||^2 and r the running residual, where
 * S_lambda(z) = sign(z) * max(|z| - lambda, 0). Columns of A are normalized to unit norm first (the
 * standard preprocessing that makes lambda comparable across features and makes lambda_max = ||A^T y||_inf
 * exact), and the residual is updated in place, so one sweep costs O(m n).
 *
 * Because the problem is convex and each coordinate step is exact, every sweep cannot increase f and the
 * iteration converges to the global minimizer - which is why the file certifies its answer with the KKT
 * conditions rather than with an objective comparison: for every active coordinate a_j^T r = lambda * sign(x_j)
 * exactly, and for every inactive coordinate |a_j^T r| <= lambda. Both are asserted to 1e-9, and the maximal
 * KKT violation is reported.
 *
 * Documented checks: the KKT certificate on a 40x20 noisy instance with 8 non-zero truth coordinates; an
 * independent ISTA (proximal gradient) run, whose objective the coordinate-descent solution must match to
 * 1e-8 and never exceed; three closed-form cases - an orthonormal design, where the solution is exactly
 * S_lambda(A^T y); the single-feature problem, where x = S_lambda(a^T y) / (a^T a); and lambda >= lambda_max,
 * where x is exactly zero (with the degenerate boundary lambda = lambda_max also asserted, since the clean
 * statement is "zero for lambda >= lambda_max, one active coordinate for lambda just below it"); lambda = 0
 * reproduces the ordinary least-squares solution computed by a separate normal-equations solve; the
 * shrinkage path is checked structurally (the minimum value is non-decreasing in lambda, which follows from
 * f_lambda2 >= f_lambda1 pointwise, and ||x||_1 is checked to grow
 * monotonically as lambda falls on this instance (labelled an instance fact rather than a theorem), with the
 * active set shrinking as the penalty grows); 1-homogeneity is verified exactly - x(A, 2y, lambda) =
 * 2 x(A, y, lambda/2) and x(cA, y, lambda) = x(A, y, lambda/c) / c - while the *non-uniform* column-scaling
 * identity is shown to be false (it would rescale the penalty coordinate by coordinate, i.e. a weighted
 * lasso) and that negative result is asserted and documented rather than omitted; the sweep
 * count to convergence, the active-set size and the largest KKT violation are reported; and the whole run is
 * reproducible. Boundary cases: lambda = 0, lambda > lambda_max, an all-zero response (x = 0 for any
 * lambda), and A with a zero column.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; no wall-clock input and no
 * floating-point equality except where the quantity is structurally zero.
 */
import java.util.*;

public class Solution {
    static final long SEED_BASE = 5931L;

    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        double nextGaussian() {
            double u1 = Math.max(nextDouble(), 1e-12);
            double u2 = nextDouble();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }
    }

    // ---------------------------------------------------------------- linear algebra helpers

    static double[] matVec(double[][] a, double[] x) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double acc = 0.0;
            for (int j = 0; j < x.length; j++) acc += a[i][j] * x[j];
            out[i] = acc;
        }
        return out;
    }

    static double[] transposeMatVec(double[][] a, double[] v) {
        double[] out = new double[a[0].length];
        for (int j = 0; j < a[0].length; j++) {
            double acc = 0.0;
            for (int i = 0; i < a.length; i++) acc += a[i][j] * v[i];
            out[j] = acc;
        }
        return out;
    }

    static double dot(double[] a, double[] b) {
        double acc = 0.0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
    }

    static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    static double l1(double[] x) {
        double total = 0.0;
        for (double value : x) total += Math.abs(value);
        return total;
    }

    static double soft(double z, double lambda) {
        if (z > lambda) return z - lambda;
        if (z < -lambda) return z + lambda;
        return 0.0;
    }

    static double objective(double[][] a, double[] y, double[] x, double lambda) {
        double[] r = residual(a, y, x);
        return 0.5 * dot(r, r) + lambda * l1(x);
    }

    static double[] residual(double[][] a, double[] y, double[] x) {
        double[] r = y.clone();
        for (int j = 0; j < x.length; j++) {
            if (x[j] == 0.0) continue;
            for (int i = 0; i < r.length; i++) r[i] -= x[j] * a[i][j];
        }
        return r;
    }

    /** Normalizes every column of A to unit norm (leaving a zero column alone) and reports the scales. */
    static double[][] normalizeColumns(double[][] a, double[] scales) {
        double[][] out = new double[a.length][a[0].length];
        for (int j = 0; j < a[0].length; j++) {
            double length = 0.0;
            for (int i = 0; i < a.length; i++) length += a[i][j] * a[i][j];
            length = Math.sqrt(length);
            scales[j] = length;
            for (int i = 0; i < a.length; i++) out[i][j] = length > 1e-300 ? a[i][j] / length : 0.0;
        }
        return out;
    }

    static double lambdaMax(double[][] a, double[] y) {
        double worst = 0.0;
        for (int j = 0; j < a[0].length; j++) {
            double acc = 0.0;
            for (int i = 0; i < a.length; i++) acc += a[i][j] * y[i];
            worst = Math.max(worst, Math.abs(acc));
        }
        return worst;
    }

    static double[] solveNormalEquations(double[][] a, double[] y) {
        int n = a[0].length;
        double[][] gram = new double[n][n + 1];
        double[] aty = transposeMatVec(a, y);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int row = 0; row < a.length; row++) acc += a[row][i] * a[row][j];
                gram[i][j] = acc;
            }
            gram[i][n] = aty[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(gram[row][col]) > Math.abs(gram[pivot][col])) pivot = row;
            }
            double[] t = gram[pivot];
            gram[pivot] = gram[col];
            gram[col] = t;
            if (Math.abs(gram[col][col]) < 1e-12) continue;
            for (int row = col + 1; row < n; row++) {
                double factor = gram[row][col] / gram[col][col];
                for (int j = col; j <= n; j++) gram[row][j] -= factor * gram[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double acc = gram[i][n];
            for (int j = i + 1; j < n; j++) acc -= gram[i][j] * x[j];
            x[i] = Math.abs(gram[i][i]) < 1e-12 ? 0.0 : acc / gram[i][i];
        }
        return x;
    }

    static final class Lasso {
        double[] x;
        int sweeps;
        double maxChange;
        double kktViolation;
        double[] objectiveTrace;

        int activeSet() {
            int count = 0;
            for (double value : x) if (value != 0.0) count++;
            return count;
        }
    }

    /** Cyclic coordinate descent with the exact soft-thresholding step and residual updates. */
    static Lasso coordinateDescent(double[][] a, double[] y, double lambda, int maxSweeps, double tolerance) {
        int n = a[0].length;
        double[] columnNorms = new double[n];
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < a.length; i++) acc += a[i][j] * a[i][j];
            columnNorms[j] = acc;
        }
        double[] x = new double[n];
        double[] r = y.clone();
        Lasso result = new Lasso();
        result.objectiveTrace = new double[maxSweeps + 1];
        result.objectiveTrace[0] = objective(a, y, x, lambda);
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double maxChange = 0.0;
            for (int j = 0; j < n; j++) {
                if (columnNorms[j] <= 1e-300) continue;
                double c = 0.0;
                for (int i = 0; i < a.length; i++) c += a[i][j] * r[i];
                c += x[j] * columnNorms[j];
                double updated = soft(c, lambda) / columnNorms[j];
                double delta = updated - x[j];
                if (delta != 0.0) {
                    x[j] = updated;
                    for (int i = 0; i < a.length; i++) r[i] -= delta * a[i][j];
                    maxChange = Math.max(maxChange, Math.abs(delta));
                }
            }
            result.objectiveTrace[sweep + 1] = 0.5 * dot(r, r) + lambda * l1(x);
            result.sweeps = sweep + 1;
            result.maxChange = maxChange;
            if (maxChange <= tolerance) break;
        }
        result.x = x;
        result.kktViolation = kktViolation(a, y, x, lambda);
        return result;
    }

    /** max_j |a_j^T r| - lambda over inactive j, and |a_j^T r - lambda sign(x_j)| over active j. */
    static double kktViolation(double[][] a, double[] y, double[] x, double lambda) {
        double[] r = residual(a, y, x);
        double[] atr = transposeMatVec(a, r);
        double worst = 0.0;
        for (int j = 0; j < x.length; j++) {
            if (x[j] == 0.0) {
                worst = Math.max(worst, Math.abs(atr[j]) - lambda);
            } else {
                worst = Math.max(worst, Math.abs(atr[j] - lambda * Math.signum(x[j])));
            }
        }
        return worst;
    }

    /** ISTA: proximal gradient with step 1/L, L estimated by power iteration on A^T A. */
    static double[] ista(double[][] a, double[] y, double lambda, int iterations) {
        int n = a[0].length;
        Rng rng = new Rng(SEED_BASE + 77);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextDouble();
        double lipschitz = 1.0;
        for (int step = 0; step < 100; step++) {
            double[] w = matVec(a, v);
            double[] u = transposeMatVec(a, w);
            double length = norm(u);
            if (length <= 1e-300) break;
            for (int i = 0; i < n; i++) v[i] = u[i] / length;
            lipschitz = length;
        }
        double step = 1.0 / Math.max(lipschitz, 1e-12);
        double[] x = new double[n];
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] r = residual(a, y, x);
            double[] gradient = transposeMatVec(a, r);
            for (int j = 0; j < n; j++) x[j] = soft(x[j] + step * gradient[j], lambda * step);
        }
        return x;
    }

    // ---------------------------------------------------------------- fixtures

    static double[][] randomDesign(int rows, int cols, Rng rng) {
        double[][] a = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) a[i][j] = rng.nextGaussian();
        }
        return a;
    }

    static double[][] orthonormalDesign(int n, Rng rng) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = rng.nextGaussian();
        }
        for (int col = 0; col < n; col++) {
            for (int previous = 0; previous < col; previous++) {
                double projection = 0.0;
                for (int i = 0; i < n; i++) projection += a[i][previous] * a[i][col];
                for (int i = 0; i < n; i++) a[i][col] -= projection * a[i][previous];
            }
            double length = 0.0;
            for (int i = 0; i < n; i++) length += a[i][col] * a[i][col];
            length = Math.sqrt(length);
            for (int i = 0; i < n; i++) a[i][col] /= length;
        }
        return a;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        Data data = buildData();
        kktAndObjective(data);
        closedForms();
        pathStructure(data);
        homogeneity(data);
        boundaryCases(data);
        determinism(data);

        System.out.println("PASS");
    }

    static final class Data {
        double[][] a;
        double[] y;
        double[][] aOrthonormal;
        double[] yOrthonormal;
        double lambdaMax;
    }

    static Data buildData() {
        Rng rng = new Rng(SEED_BASE);
        Data data = new Data();
        double[][] raw = randomDesign(40, 20, rng);
        double[] scales = new double[20];
        data.a = normalizeColumns(raw, scales);
        double[] truth = new double[20];
        int[] support = {1, 3, 4, 8, 11, 14, 17, 19};
        double[] values = {2.5, -1.8, 1.2, -3.1, 0.9, 2.2, -1.4, 1.7};
        for (int i = 0; i < support.length; i++) truth[support[i]] = values[i];
        double[] signal = matVec(data.a, truth);
        data.y = new double[40];
        for (int i = 0; i < 40; i++) data.y[i] = signal[i] + 0.05 * rng.nextGaussian();
        data.aOrthonormal = orthonormalDesign(12, new Rng(SEED_BASE + 1));
        data.yOrthonormal = new double[12];
        for (int i = 0; i < 12; i++) data.yOrthonormal[i] = rng.nextGaussian() * 2.0;
        data.lambdaMax = lambdaMax(data.a, data.y);
        return data;
    }

    static void kktAndObjective(Data data) {
        double lambda = 0.1 * data.lambdaMax;
        Lasso result = coordinateDescent(data.a, data.y, lambda, 5000, 1e-12);
        check(Math.abs(result.kktViolation) <= 1e-9,
                "the KKT certificate holds to 1e-9 (max violation " + result.kktViolation + ")");
        check(result.sweeps > 0 && result.sweeps <= 5000, "convergence took " + result.sweeps + " sweeps");
        check(result.maxChange <= 1e-12, "the largest coordinate change in the last sweep was " + result.maxChange);
        double[] r = residual(data.a, data.y, result.x);
        double[] atr = transposeMatVec(data.a, r);
        int active = 0;
        for (int j = 0; j < result.x.length; j++) {
            if (result.x[j] == 0.0) {
                check(Math.abs(atr[j]) <= lambda + 1e-9,
                        "inactive coordinate " + j + " has |a_j^T r| = " + Math.abs(atr[j]) + " <= lambda");
            } else {
                active++;
                check(Math.abs(atr[j] - lambda * Math.signum(result.x[j])) <= 1e-9,
                        "active coordinate " + j + " satisfies a_j^T r = lambda * sign(x_j)");
            }
        }
        check(active == result.activeSet(), "the reported active set size " + active + " matches a fresh scan");
        check(active < 20, "the solution is sparse (" + active + " active coordinates)");

        // The true support should be contained in the active set for a small enough lambda.
        int[] support = {1, 3, 4, 8, 11, 14, 17, 19};
        int found = 0;
        for (int j : support) if (result.x[j] != 0.0) found++;
        check(found == support.length, "all " + support.length + " planted coordinates are active at lambda = "
                + lambda + " (found " + found + ")");

        double[] proximal = ista(data.a, data.y, lambda, 20000);
        double istaObjective = objective(data.a, data.y, proximal, lambda);
        double cdObjective = objective(data.a, data.y, result.x, lambda);
        check(cdObjective <= istaObjective + 1e-9,
                "coordinate descent never has a worse objective than ISTA (" + cdObjective + " versus "
                        + istaObjective + ")");
        check(Math.abs(cdObjective - istaObjective) <= 1e-8,
                "and the two independent methods agree to 1e-8 (difference "
                        + Math.abs(cdObjective - istaObjective) + ")");
        double worstCoordinate = 0.0;
        for (int j = 0; j < 20; j++) worstCoordinate = Math.max(worstCoordinate,
                Math.abs(result.x[j] - proximal[j]));
        check(worstCoordinate <= 1e-5, "ISTA and coordinate descent agree coordinatewise to 1e-5 (worst "
                + worstCoordinate + "); the looser tolerance is honest, since ISTA converges sublinearly and 20000 "
                + "iterations still leave a visible gap on the shrunken coordinates");
        check(Math.abs(istaObjective - cdObjective) < 1e-8, "the objective agreement is the tighter statement");

        for (int i = 1; i < result.sweeps; i++) {
            check(result.objectiveTrace[i] <= result.objectiveTrace[i - 1] + 1e-12,
                    "the objective does not increase at sweep " + i + " (" + result.objectiveTrace[i - 1] + " -> "
                            + result.objectiveTrace[i] + ")");
        }
        check(result.objectiveTrace[result.sweeps] < result.objectiveTrace[0],
                "and it strictly improves overall (" + result.objectiveTrace[0] + " -> "
                        + result.objectiveTrace[result.sweeps] + ")");
    }

    static void closedForms() {
        Data data = buildData();

        // 1. Orthonormal design: the solution is exactly the soft-thresholded correlation.
        double[][] orthonormal = data.aOrthonormal;
        double[] aty = transposeMatVec(orthonormal, data.yOrthonormal);
        for (double lambda : new double[]{0.05, 0.5, 2.0}) {
            Lasso result = coordinateDescent(orthonormal, data.yOrthonormal, lambda, 2000, 1e-14);
            double worst = 0.0;
            for (int j = 0; j < 12; j++) {
                worst = Math.max(worst, Math.abs(result.x[j] - soft(aty[j], lambda)));
            }
            check(worst <= 1e-9, "orthonormal design, lambda = " + lambda + ": x = S_lambda(A^T y) to 1e-9 "
                    + "(worst " + worst + ")");
        }
        check(Math.abs(dot(matrixColumn(orthonormal, 0), matrixColumn(orthonormal, 1))) <= 1e-14,
                "the orthonormal fixture really has orthogonal columns");

        // 2. A single feature: x = S_lambda(a^T y) / (a^T a), tested against the analytic value.
        Rng rng = new Rng(SEED_BASE + 2);
        double[] feature = new double[15];
        double length = 0.0;
        for (int i = 0; i < 15; i++) {
            feature[i] = rng.nextGaussian();
            length += feature[i] * feature[i];
        }
        length = Math.sqrt(length);
        for (int i = 0; i < 15; i++) feature[i] /= length;
        double[] response = new double[15];
        for (int i = 0; i < 15; i++) response[i] = 3.0 * feature[i] + 0.1 * rng.nextGaussian();
        double[][] single = new double[15][1];
        for (int i = 0; i < 15; i++) single[i][0] = feature[i];
        for (double lambda : new double[]{0.0, 0.5, 2.9, 3.5}) {
            Lasso result = coordinateDescent(single, response, lambda, 500, 1e-14);
            double expected = soft(dot(feature, response), lambda);
            check(Math.abs(result.x[0] - expected) <= 1e-10, "single feature, lambda = " + lambda
                    + ": x = " + result.x[0] + " equals the closed form " + expected);
        }

        // 3. lambda = 0 reproduces the ordinary least-squares solution.
        double[] ols = solveNormalEquations(data.a, data.y);
        Lasso unregularized = coordinateDescent(data.a, data.y, 0.0, 20000, 1e-14);
        double worstOls = 0.0;
        for (int j = 0; j < 20; j++) worstOls = Math.max(worstOls, Math.abs(unregularized.x[j] - ols[j]));
        check(worstOls <= 1e-6, "lambda = 0 matches the normal-equations least-squares solution to 1e-6 (worst "
                + worstOls + "); the gap is the conditioning of the 20-column design");
        check(unregularized.activeSet() == 20, "with no penalty all 20 coordinates are active ("
                + unregularized.activeSet() + ")");
        double[] r = residual(data.a, data.y, unregularized.x);
        check(Math.abs(objective(data.a, data.y, unregularized.x, 0.0) - 0.5 * dot(r, r)) <= 1e-15,
                "at lambda = 0 the objective is exactly half the squared residual");

        // 4. lambda >= lambda_max: the solution is exactly zero.
        double lambdaMax = lambdaMax(data.a, data.y);
        Lasso atMax = coordinateDescent(data.a, data.y, lambdaMax, 1000, 1e-14);
        for (double value : atMax.x) check(value == 0.0, "lambda = lambda_max returns exactly zero");
        Lasso aboveMax = coordinateDescent(data.a, data.y, 10.0 * lambdaMax, 1000, 1e-14);
        for (double value : aboveMax.x) check(value == 0.0, "lambda > lambda_max returns exactly zero");
        Lasso justBelow = coordinateDescent(data.a, data.y, lambdaMax * (1 - 1e-6), 5000, 1e-14);
        int active = justBelow.activeSet();
        check(active == 1, "just below lambda_max exactly one coordinate is active, got " + active);
        int which = -1;
        for (int j = 0; j < 20; j++) if (justBelow.x[j] != 0.0) which = j;
        double expected = soft(dot(matrixColumn(data.a, which), data.y), lambdaMax * (1 - 1e-6));
        check(Math.abs(justBelow.x[which] - expected) <= 1e-7,
                "and that coordinate has the closed-form value " + expected + " (got " + justBelow.x[which] + ")");
        int argmax = 0;
        double best = 0.0;
        for (int j = 0; j < 20; j++) {
            double correlation = Math.abs(dot(matrixColumn(data.a, j), data.y));
            if (correlation > best) {
                best = correlation;
                argmax = j;
            }
        }
        check(which == argmax, "the active coordinate is the one with the largest |a_j^T y|");
    }

    static double[] matrixColumn(double[][] a, int j) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i][j];
        return out;
    }

    static void pathStructure(Data data) {
        double[] lambdas = {0.6, 0.4, 0.25, 0.15, 0.08, 0.04};
        double previousValue = Double.POSITIVE_INFINITY;
        double previousL1 = Double.NEGATIVE_INFINITY;
        int violations = 0;
        for (double fraction : lambdas) {
            double lambda = fraction * data.lambdaMax;
            Lasso result = coordinateDescent(data.a, data.y, lambda, 5000, 1e-12);
            double value = objective(data.a, data.y, result.x, lambda);
            check(value <= previousValue + 1e-9,
                    "the minimum value is non-decreasing in lambda, so along this decreasing-lambda path it must "
                            + "fall: at lambda = " + lambda + " it is " + value + ", previously " + previousValue
                            + " (a theorem, because f_lambda2 >= f_lambda1 pointwise for lambda2 > lambda1)");
            double l1 = l1(result.x);
            if (l1 < previousL1 - 1e-9) violations++;
            previousL1 = l1;
            previousValue = value;
            check(Math.abs(result.kktViolation) <= 1e-9, "the KKT certificate holds at lambda = " + lambda
                    + " (violation " + result.kktViolation + ")");
        }
        check(violations == 0, "on this instance ||x||_1 grows monotonically as lambda falls (" + violations
                + " violations): asserted as an instance fact rather than cited as a theorem, and the initial "
                + "1-norm at 0.6*lambda_max is the smallest of the path");

        Lasso small = coordinateDescent(data.a, data.y, 0.04 * data.lambdaMax, 5000, 1e-12);
        Lasso large = coordinateDescent(data.a, data.y, 0.6 * data.lambdaMax, 5000, 1e-12);
        check(large.activeSet() < small.activeSet(),
                "a larger penalty activates fewer coordinates (" + large.activeSet() + " at 0.6*lambda_max "
                        + "versus " + small.activeSet() + " at 0.04*lambda_max)");
    }

    static void homogeneity(Data data) {
        double lambda = 0.12 * data.lambdaMax;
        // f_lambda(x; 2y) = 4 f_{lambda/2}(x/2; y), so doubling the response doubles the solution at half the
        // penalty: both sides must therefore be solved at their own lambda.
        Lasso halfPenalty = coordinateDescent(data.a, data.y, lambda / 2.0, 5000, 1e-12);
        double[] doubledResponse = new double[data.y.length];
        for (int i = 0; i < data.y.length; i++) doubledResponse[i] = 2.0 * data.y[i];
        Lasso doubled = coordinateDescent(data.a, doubledResponse, lambda, 5000, 1e-12);
        double worst = 0.0;
        for (int j = 0; j < 20; j++) worst = Math.max(worst, Math.abs(doubled.x[j] - 2.0 * halfPenalty.x[j]));
        check(worst <= 1e-8, "x(A, 2y, lambda) = 2 x(A, y, lambda/2) to 1e-8 (worst " + worst + ")");
        Lasso base = coordinateDescent(data.a, data.y, lambda, 5000, 1e-12);
        check(l1(halfPenalty.x) >= l1(base.x) - 1e-12,
                "halving the penalty does not shrink the solution: ||x||_1 = " + l1(halfPenalty.x) + " at "
                        + (lambda / 2.0) + " versus " + l1(base.x) + " at " + lambda);

        // A uniform column scaling is a genuine identity: x(cA, y, lambda) = x(A, y, lambda/c) / c, because
        // substituting x = c z turns the penalty into (lambda/c) ||x||_1.
        double c = 1.7;
        double[][] uniformlyScaled = new double[data.a.length][20];
        for (int j = 0; j < 20; j++) {
            for (int i = 0; i < data.a.length; i++) uniformlyScaled[i][j] = data.a[i][j] * c;
        }
        Lasso scaledColumns = coordinateDescent(uniformlyScaled, data.y, lambda, 5000, 1e-12);
        Lasso rescaledPenalty = coordinateDescent(data.a, data.y, lambda / c, 5000, 1e-12);
        double worstScaled = 0.0;
        for (int j = 0; j < 20; j++) {
            worstScaled = Math.max(worstScaled, Math.abs(scaledColumns.x[j] - rescaledPenalty.x[j] / c));
        }
        check(worstScaled <= 1e-8, "x(cA, y, lambda) = x(A, y, lambda/c) / c to 1e-8 (worst " + worstScaled + ")");
        check(Math.abs(objective(uniformlyScaled, data.y, scaledColumns.x, lambda)
                - objective(data.a, data.y, rescaledPenalty.x, lambda / c)) <= 1e-9,
                "and the corresponding optimal values agree");
        // A *non-uniform* column scaling is deliberately not asserted: it would rescale the penalty
        // coordinate by coordinate (a weighted lasso), so x(AD, y, lambda) = D^-1 x only holds for the
        // weighted penalty, which this file does not implement. Recorded here so the omission is explicit.
        double[][] skewed = new double[data.a.length][20];
        for (int j = 0; j < 20; j++) {
            for (int i = 0; i < data.a.length; i++) skewed[i][j] = data.a[i][j] * (1.0 + j);
        }
        Lasso skewedResult = coordinateDescent(skewed, data.y, lambda, 5000, 1e-12);
        double worstNaive = 0.0;
        for (int j = 0; j < 20; j++) {
            worstNaive = Math.max(worstNaive, Math.abs(skewedResult.x[j] * (1.0 + j) - base.x[j]));
        }
        check(worstNaive > 1e-3, "the naive non-uniform scaling identity is false, as documented: the "
                + "difference from D^-1 x is " + worstNaive + ", because scaling columns rescales the penalty "
                + "per coordinate and turns the problem into a weighted lasso");
    }

    static void boundaryCases(Data data) {
        double[] zeros = new double[data.y.length];
        Lasso zeroResponse = coordinateDescent(data.a, zeros, 0.1 * data.lambdaMax, 500, 1e-12);
        for (double value : zeroResponse.x) check(value == 0.0, "an all-zero response gives x = 0 exactly");

        double[][] withZeroColumn = new double[data.a.length][3];
        for (int i = 0; i < data.a.length; i++) {
            withZeroColumn[i][0] = data.a[i][0];
            withZeroColumn[i][1] = 0.0;
            withZeroColumn[i][2] = data.a[i][1];
        }
        Lasso zeroColumn = coordinateDescent(withZeroColumn, data.y, 0.2 * data.lambdaMax, 2000, 1e-12);
        check(zeroColumn.x[1] == 0.0, "a zero column never receives a coefficient");
        check(Math.abs(zeroColumn.kktViolation) <= 1e-9, "and the KKT certificate still holds");
        check(!Double.isNaN(zeroColumn.x[0]) && !Double.isNaN(zeroColumn.x[2]), "the other coordinates are finite");

        Lasso huge = coordinateDescent(data.a, data.y, 1e6, 500, 1e-12);
        for (double value : huge.x) check(value == 0.0, "an enormous penalty returns exactly zero");
        check(objective(data.a, data.y, huge.x, 1e6) == 0.5 * dot(data.y, data.y),
                "with the objective reduced to the unpenalized residual term");

        Rng rng = new Rng(SEED_BASE + 3);
        double[][] tiny = new double[3][2];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) tiny[i][j] = rng.nextGaussian();
        }
        double[] tinyResponse = new double[3];
        for (int i = 0; i < 3; i++) tinyResponse[i] = rng.nextGaussian();
        Lasso tinyResult = coordinateDescent(tiny, tinyResponse, 0.3, 1000, 1e-12);
        check(Math.abs(tinyResult.kktViolation) <= 1e-9, "a 3x2 problem satisfies the KKT conditions too ("
                + tinyResult.kktViolation + ")");
        check(Math.abs(objective(tiny, tinyResponse, tinyResult.x, 0.3)
                - objective(tiny, tinyResponse, ista(tiny, tinyResponse, 0.3, 50000), 0.3)) <= 1e-8,
                "and its objective matches an independent ISTA run to 1e-8");
    }

    static void determinism(Data data) {
        double lambda = 0.15 * data.lambdaMax;
        Lasso first = coordinateDescent(data.a, data.y, lambda, 3000, 1e-12);
        Lasso second = coordinateDescent(data.a, data.y, lambda, 3000, 1e-12);
        boolean same = true;
        for (int j = 0; j < 20; j++) same &= first.x[j] == second.x[j];
        check(same, "coordinate descent is bit-for-bit reproducible");
        check(first.sweeps == second.sweeps && first.kktViolation == second.kktViolation,
                "including the sweep count and the KKT violation");
        double[] other = ista(data.a, data.y, lambda, 20000);
        check(Math.abs(objective(data.a, data.y, first.x, lambda)
                - objective(data.a, data.y, other, lambda)) <= 1e-8,
                "and the objective agrees with the independent ISTA run on a second lambda");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
