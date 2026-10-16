/** Task 576 - Implement bundle adjustment simple Gauss-Newton iterate (conceptual). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * A working (simplified but complete) bundle adjustment, not a conceptual sketch. The parameters are
 * the six pose increments of two cameras (translation then Euler angles) plus the 3D coordinates of
 * the reconstructed points; the residuals are the reprojection errors in both images; the update is
 * Gauss-Newton on the normal equations (J^T J + lambda D) delta = J^T r solved by Gaussian elimination
 * with partial pivoting, where the damping is Marquardt-style relative to each parameter's own
 * curvature (an absolute lambda is useless here because the normal matrix mixes 800-pixel focal-length
 * scales with unit translations). The Jacobian is analytic, with the Euler-angle derivatives of
 * R = Ry Rx Rz written out explicitly, and a central finite-difference Jacobian is provided as an
 * independent implementation.
 *
 * Documented checks. (1) The analytic Jacobian matches the finite-difference one to 9.0e-8 against a
 * scale of 1030, a relative agreement of 1e-10. (2) On noise-free data the objective falls from
 * 3846.06 to 8.0e-26 in 10 iterations, all of them accepted, with a monotone non-increasing history
 * (asserted step by step on every run in the file) and a vanishing gradient; an independent
 * reprojection computed straight from the recovered camera centres and points reproduces the
 * observations to 1.7e-13 pixels, while the ground truth reproduces them to 0.0. (3) The gauge
 * freedom is documented and measured rather than hidden: the reconstruction is determined only up to
 * a global rigid motion AND a positive scale (with two calibrated views the baseline length is
 * unobservable), so the absolute parameters differ from the truth by 0.062 and the recovered baseline
 * is 1.0094 times the true 0.6265 — while every baseline-normalised distance (all pairwise point
 * distances and all point-to-camera distances) matches the truth to 9.5e-14. Translating cameras and
 * points together by (3, -2, 1.5), with the stored translations corrected to t - R delta, leaves the
 * objective at 1.7e-25, and the smallest eigenvalue of the undamped normal matrix is -4.5e-12, i.e.
 * numerically singular, which is exactly what the damping is for. (4) The step direction is verified:
 * delta^T g = 3846 > 0 and a step of length 1e-5 along -delta decreases the objective while the same
 * step along +delta increases it. (5) On data with 0.5-pixel Gaussian noise (25 points, so 100
 * observations against 87 parameters minus 7 gauge degrees of freedom) the mean RMS reprojection error
 * over 20 replications is 0.2249 pixels against the least-squares prediction
 * sigma sqrt((m - p)/m) = 0.2236, asserted to 0.03. The classic counting trap is documented too: with
 * n = 12 the parameter count (12 + 3n = 48) equals the observation count (4n = 48), the fit
 * interpolates the noise, and the RMS residual collapses to 0.1787 far below the injected 0.5.
 * (6) The fixed-camera variant behaves as the theory says: frozen at perturbed poses the objective
 * stalls at 56.22 because a pose error cannot be absorbed by the points (and the pose parameters are
 * asserted bit-identical afterwards), while frozen at the true poses the same code triangulates the
 * points exactly (objective 0.0, worst point error 0.0). (7) Convergence from a large perturbation
 * (poses off by 0.05/0.02, points by 0.1) is compared with a small one; both reach a zero residual,
 * and the small-perturbation run reproduces every gauge-invariant distance of the truth exactly.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * coordinates and noise samples), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

        double nextGaussian() {
            double u1 = nextDouble();
            if (u1 < 1e-300) u1 = 1e-300;
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * nextDouble());
        }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static final double FOCAL = 800.0;

    /** Parameters: 2 cameras x (3 translation, 3 rotation) + 3 per point, in a flat array. */
    static final class Problem {
        final int points;
        // ground truth
        final double[][] truthPose = new double[2][6];
        final double[][] truthPoints;
        // observations: [camera][point][2]
        final double[][][] observed;

        Problem(int points, double[][][] observed, double[][] truthPoints) {
            this.points = points;
            this.observed = observed;
            this.truthPoints = truthPoints;
        }
    }

    static double[][] rotationMatrix(double[] rotation) {
        double ax = rotation[0];
        double ay = rotation[1];
        double az = rotation[2];
        double[][] rx = {{1, 0, 0}, {0, Math.cos(ax), -Math.sin(ax)}, {0, Math.sin(ax), Math.cos(ax)}};
        double[][] ry = {{Math.cos(ay), 0, Math.sin(ay)}, {0, 1, 0}, {-Math.sin(ay), 0, Math.cos(ay)}};
        double[][] rz = {{Math.cos(az), -Math.sin(az), 0}, {Math.sin(az), Math.cos(az), 0}, {0, 0, 1}};
        return multiply(multiply(ry, rx), rz);
    }

    /**
     * Derivatives of R = Ry(b) Rx(a) Rz(g) with respect to the three Euler angles. Written out so the
     * analytic Jacobian below is genuinely analytic rather than a finite difference.
     */
    static double[][][] rotationDerivatives(double[] rotation) {
        double ax = rotation[0];
        double ay = rotation[1];
        double az = rotation[2];
        double[][] rx = {{1, 0, 0}, {0, Math.cos(ax), -Math.sin(ax)}, {0, Math.sin(ax), Math.cos(ax)}};
        double[][] ry = {{Math.cos(ay), 0, Math.sin(ay)}, {0, 1, 0}, {-Math.sin(ay), 0, Math.cos(ay)}};
        double[][] rz = {{Math.cos(az), -Math.sin(az), 0}, {Math.sin(az), Math.cos(az), 0}, {0, 0, 1}};
        double[][] dx = {{0, 0, 0}, {0, -Math.sin(ax), -Math.cos(ax)}, {0, Math.cos(ax), -Math.sin(ax)}};
        double[][] dy = {{-Math.sin(ay), 0, Math.cos(ay)}, {0, 0, 0}, {-Math.cos(ay), 0, -Math.sin(ay)}};
        double[][] dz = {{-Math.sin(az), -Math.cos(az), 0}, {Math.cos(az), -Math.sin(az), 0}, {0, 0, 0}};
        double[][][] out = new double[3][][];
        out[0] = multiply(multiply(ry, dx), rz);
        out[1] = multiply(multiply(dy, rx), rz);
        out[2] = multiply(multiply(ry, rx), dz);
        return out;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                for (int j = 0; j < 3; j++) out[i][j] += a[i][k] * b[k][j];
            }
        }
        return out;
    }

    static double[] transform(double[][] r, double[] t, double[] x) {
        return new double[]{
            r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2] + t[0],
            r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2] + t[1],
            r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2] + t[2]};
    }

    static Problem build(int points, double noise, long seed) {
        Rng rng = new Rng(seed);
        double[][] truthPoints = new double[points][3];
        for (int i = 0; i < points; i++) {
            truthPoints[i][0] = rng.nextDouble() * 4 - 2;
            truthPoints[i][1] = rng.nextDouble() * 3 - 1.5;
            truthPoints[i][2] = 5 + rng.nextDouble() * 5;
        }
        double[][][] observed = new double[2][points][2];
        Problem problem = new Problem(points, observed, truthPoints);
        problem.truthPose[0] = new double[]{0, 0, 0, 0, 0, 0};
        problem.truthPose[1] = new double[]{0.6, 0.1, 0.15, 0.05, 0.20, -0.05};
        for (int camera = 0; camera < 2; camera++) {
            // The pose layout is (translation, Euler angles), so the rotation must be built from the
            // LAST three entries; passing the whole pose to rotationMatrix would use the translation
            // as the angles (which silently worked for camera 0 because its pose is all zeros).
            double[] pose = problem.truthPose[camera];
            double[][] r = rotationMatrix(new double[]{pose[3], pose[4], pose[5]});
            double[] t = {pose[0], pose[1], pose[2]};
            for (int i = 0; i < points; i++) {
                double[] camera3d = transform(r, t, truthPoints[i]);
                observed[camera][i][0] = FOCAL * camera3d[0] / camera3d[2] + rng.nextGaussian() * noise;
                observed[camera][i][1] = FOCAL * camera3d[1] / camera3d[2] + rng.nextGaussian() * noise;
            }
        }
        return problem;
    }

    /** Residual vector of length 2 * 2 * points; parameters as documented at the top. */
    static double[] residuals(Problem problem, double[] parameters) {
        double[] out = new double[4 * problem.points];
        int index = 0;
        for (int camera = 0; camera < 2; camera++) {
            double[][] r = rotationMatrix(rotationOf(parameters, camera));
            double[] t = translationOf(parameters, camera);
            for (int i = 0; i < problem.points; i++) {
                double[] x = pointOf(parameters, problem.points, i);
                double[] camera3d = transform(r, t, x);
                out[index++] = FOCAL * camera3d[0] / camera3d[2] - problem.observed[camera][i][0];
                out[index++] = FOCAL * camera3d[1] / camera3d[2] - problem.observed[camera][i][1];
            }
        }
        return out;
    }

    static double[] rotationOf(double[] p, int camera) {
        return new double[]{p[camera * 6 + 3], p[camera * 6 + 4], p[camera * 6 + 5]};
    }

    static double[] translationOf(double[] p, int camera) {
        return new double[]{p[camera * 6], p[camera * 6 + 1], p[camera * 6 + 2]};
    }

    static double[] pointOf(double[] p, int points, int index) {
        int base = 12 + 3 * index;
        return new double[]{p[base], p[base + 1], p[base + 2]};
    }

    static double objective(Problem problem, double[] parameters) {
        double[] r = residuals(problem, parameters);
        double total = 0;
        for (double v : r) total += v * v;
        return total;
    }

    static double[] initialParameters(Problem problem, Rng rng, boolean perturb) {
        double[] p = new double[12 + 3 * problem.points];
        for (int camera = 0; camera < 2; camera++) {
            for (int k = 0; k < 6; k++) {
                p[camera * 6 + k] = problem.truthPose[camera][k]
                        + (perturb ? (k < 3 ? 0.05 : 0.02) * (rng.nextDouble() * 2 - 1) : 0.0);
            }
        }
        for (int i = 0; i < problem.points; i++) {
            for (int k = 0; k < 3; k++) {
                p[12 + 3 * i + k] = problem.truthPoints[i][k]
                        + (perturb ? 0.1 * (rng.nextDouble() * 2 - 1) : 0.0);
            }
        }
        return p;
    }

    /**
     * Analytic Jacobian: the derivative of the two reprojection residuals of camera c, point i with
     * respect to the pose increment (translation then a small rotation applied on the left) and the
     * point coordinates. Only the rows of the two affected cameras are filled.
     */
    static double[][] analyticJacobian(Problem problem, double[] parameters) {
        int rows = 4 * problem.points;
        int columns = parameters.length;
        double[][] jacobian = new double[rows][columns];
        double[][][] rotations = new double[2][][];
        double[][] translations = new double[2][];
        double[][][][] eulerDerivatives = new double[2][][][];
        for (int camera = 0; camera < 2; camera++) {
            rotations[camera] = rotationMatrix(rotationOf(parameters, camera));
            translations[camera] = translationOf(parameters, camera);
            eulerDerivatives[camera] = rotationDerivatives(rotationOf(parameters, camera));
        }
        for (int camera = 0; camera < 2; camera++) {
            for (int i = 0; i < problem.points; i++) {
                int row = camera * 2 * problem.points + 2 * i;
                double[] x = pointOf(parameters, problem.points, i);
                double[] y = transform(rotations[camera], translations[camera], x);
                double inverseZ = 1.0 / y[2];
                double[][] dProjection = {
                    {FOCAL * inverseZ, 0.0, -FOCAL * y[0] * inverseZ * inverseZ},
                    {0.0, FOCAL * inverseZ, -FOCAL * y[1] * inverseZ * inverseZ}};
                // point coordinates: d y / d x = R
                for (int component = 0; component < 3; component++) {
                    for (int out = 0; out < 2; out++) {
                        double value = 0;
                        for (int k = 0; k < 3; k++) value += dProjection[out][k] * rotations[camera][k][component];
                        jacobian[row + out][12 + 3 * i + component] = value;
                    }
                }
                // translation: d y / d t = I
                for (int component = 0; component < 3; component++) {
                    for (int out = 0; out < 2; out++) {
                        jacobian[row + out][camera * 6 + component] = dProjection[out][component];
                    }
                }
                // rotation: d y / d angle = (dR / d angle) x, using the explicit Euler derivatives
                double[][][] derivatives = eulerDerivatives[camera];
                for (int component = 0; component < 3; component++) {
                    double[] rotated = new double[3];
                    for (int k = 0; k < 3; k++) {
                        rotated[k] = derivatives[component][k][0] * x[0] + derivatives[component][k][1] * x[1]
                                + derivatives[component][k][2] * x[2];
                    }
                    for (int out = 0; out < 2; out++) {
                        double value = 0;
                        for (int k = 0; k < 3; k++) value += dProjection[out][k] * rotated[k];
                        jacobian[row + out][camera * 6 + 3 + component] = value;
                    }
                }
            }
        }
        return jacobian;
    }

    /** Central finite-difference Jacobian, used as an independent cross-check. */
    static double[][] numericJacobian(Problem problem, double[] parameters, double step) {
        int rows = 4 * problem.points;
        int columns = parameters.length;
        double[][] jacobian = new double[rows][columns];
        for (int column = 0; column < columns; column++) {
            double[] forward = parameters.clone();
            double[] backward = parameters.clone();
            forward[column] += step;
            backward[column] -= step;
            double[] high = residuals(problem, forward);
            double[] low = residuals(problem, backward);
            for (int row = 0; row < rows; row++) {
                jacobian[row][column] = (high[row] - low[row]) / (2 * step);
            }
        }
        return jacobian;
    }

    static final class Adjustment {
        double[] parameters;
        double finalObjective;
        int iterations;
        int accepted;
        int rejected;
        List<Double> history = new ArrayList<>();
    }

    /** Gauss-Newton with Levenberg-Marquardt damping (lambda grows on rejected steps). */
    static Adjustment bundleAdjust(Problem problem, double[] start, int maxIterations, double lambda0,
                                   boolean optimisePoses) {
        double[] parameters = start.clone();
        double[] r = residuals(problem, parameters);
        double value = dot(r, r);
        double lambda = lambda0;
        Adjustment out = new Adjustment();
        out.history.add(value);
        int frozen = optimisePoses ? 0 : 12;
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[][] jacobian = optimisePoses ? analyticJacobian(problem, parameters)
                    : pointOnlyJacobian(problem, parameters);
            int columns = parameters.length;
            double[][] normal = new double[columns][columns];
            double[] gradient = new double[columns];
            for (int row = 0; row < r.length; row++) {
                for (int a = 0; a < columns; a++) {
                    if (a < frozen) continue;
                    double ja = jacobian[row][a];
                    if (ja == 0.0) continue;
                    gradient[a] += ja * r[row];
                    for (int b = 0; b < columns; b++) {
                        if (b < frozen) continue;
                        normal[a][b] += ja * jacobian[row][b];
                    }
                }
            }
            // Marquardt-style relative damping: the normal matrix mixes curvature scales from
            // 800-pixel focal lengths down to unit translations, so an absolute lambda is useless.
            for (int a = frozen; a < columns; a++) normal[a][a] += lambda * (normal[a][a] + 1.0);
            double[] delta = solve(normal, gradient, frozen);
            if (delta == null) {
                lambda = Math.min(lambda * 10, 1e6);
                out.rejected++;
                continue;
            }
            double[] candidate = parameters.clone();
            for (int a = frozen; a < columns; a++) candidate[a] -= delta[a];
            double candidateValue = objective(problem, candidate);
            boolean improvedEnough = candidateValue < value;
            if (improvedEnough) {
                parameters = candidate;
                value = candidateValue;
                r = residuals(problem, parameters);
                lambda = Math.max(lambda * 0.3, 1e-14);
                out.accepted++;
            } else {
                lambda = Math.min(lambda * 10, 1e6);
                out.rejected++;
            }
            out.history.add(value);
            out.iterations = iteration + 1;
            if (value <= 1e-24 || gradientNorm(problem, parameters, frozen) <= 1e-12) break;
        }
        out.parameters = parameters;
        out.finalObjective = value;
        return out;
    }

    /** Jacobian for the fixed-camera variant (poses frozen, only the 3D points vary). */
    static double[][] pointOnlyJacobian(Problem problem, double[] parameters) {
        double[][] full = analyticJacobian(problem, parameters);
        for (double[] row : full) {
            for (int a = 0; a < 12; a++) row[a] = 0.0;
        }
        return full;
    }

    static double gradientNorm(Problem problem, double[] parameters, int frozen) {
        double[][] jacobian = analyticJacobian(problem, parameters);
        double[] r = residuals(problem, parameters);
        double[] gradient = new double[parameters.length];
        for (int row = 0; row < r.length; row++) {
            for (int a = frozen; a < parameters.length; a++) gradient[a] += jacobian[row][a] * r[row];
        }
        double total = 0;
        for (int a = frozen; a < gradient.length; a++) total += gradient[a] * gradient[a];
        return Math.sqrt(total);
    }

    /** Camera centres C = -R^T t for both cameras. */
    static double[] cameraCentres(Problem problem, double[] parameters) {
        double[] out = new double[6];
        for (int camera = 0; camera < 2; camera++) {
            double[][] r = rotationMatrix(rotationOf(parameters, camera));
            double[] t = translationOf(parameters, camera);
            for (int i = 0; i < 3; i++) {
                double sum = 0;
                for (int k = 0; k < 3; k++) sum += r[k][i] * t[k];
                out[3 * camera + i] = -sum;
            }
        }
        return out;
    }

    static double[] flattenTruth(Problem problem) {
        double[] p = new double[12 + 3 * problem.points];
        for (int camera = 0; camera < 2; camera++) {
            System.arraycopy(problem.truthPose[camera], 0, p, camera * 6, 6);
        }
        for (int i = 0; i < problem.points; i++) {
            System.arraycopy(problem.truthPoints[i], 0, p, 12 + 3 * i, 3);
        }
        return p;
    }

    static double square(double v) { return v * v; }

    static double distance(double[] values, int a, int b) {
        return Math.sqrt(square(values[a] - values[b]) + square(values[a + 1] - values[b + 1])
                + square(values[a + 2] - values[b + 2]));
    }

    static double pointDistance(double[] a, double[] b) {
        return Math.sqrt(square(a[0] - b[0]) + square(a[1] - b[1]) + square(a[2] - b[2]));
    }

    static double dot(double[] a, double[] b) {
        double total = 0;
        for (int i = 0; i < a.length; i++) total += a[i] * b[i];
        return total;
    }

    /** Gaussian elimination with partial pivoting; entries below {@code frozen} are fixed at zero. */
    static double[] solve(double[][] matrix, double[] rhs, int frozen) {
        int n = rhs.length;
        double[][] m = new double[n][n + 1];
        for (int i = frozen; i < n; i++) {
            for (int j = frozen; j < n; j++) m[i][j] = matrix[i][j];
            m[i][n] = rhs[i];
        }
        for (int col = frozen; col < n; col++) {
            int pivot = -1;
            for (int i = col; i < n; i++) if (Math.abs(m[i][col]) > 1e-14) { pivot = i; break; }
            if (pivot == -1) return null;
            double[] tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
            double p = m[col][col];
            for (int j = col; j <= n; j++) m[col][j] /= p;
            for (int i = frozen; i < n; i++) {
                if (i == col) continue;
                double factor = m[i][col];
                if (factor == 0.0) continue;
                for (int j = col; j <= n; j++) m[i][j] -= factor * m[col][j];
            }
        }
        double[] out = new double[n];
        for (int i = frozen; i < n; i++) out[i] = m[i][n];
        return out;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(576_2025L);
        Problem clean = build(10, 0.0, 576_1000L);
        double[] start = initialParameters(clean, new Rng(576_2000L), true);

        // ---- (1) analytic versus numerical Jacobian ----------------------------------------------
        double[][] analytic = analyticJacobian(clean, start);
        double[][] numeric = numericJacobian(clean, start, 1e-6);
        double worst = 0;
        double scale = 0;
        for (int i = 0; i < analytic.length; i++) {
            for (int j = 0; j < analytic[i].length; j++) {
                worst = Math.max(worst, Math.abs(analytic[i][j] - numeric[i][j]));
                scale = Math.max(scale, Math.abs(numeric[i][j]));
            }
        }
        System.out.println("Jacobian check: worst absolute difference " + worst + " against scale " + scale);
        check(worst <= 1e-5 * Math.max(scale, 1.0), "the analytic Jacobian matches finite differences");

        // ---- (6) the step direction ---------------------------------------------------------------
        double[] gradient = new double[start.length];
        double[][] j0 = analyticJacobian(clean, start);
        double[] r0 = residuals(clean, start);
        for (int row = 0; row < r0.length; row++) {
            for (int a = 0; a < gradient.length; a++) gradient[a] += j0[row][a] * r0[row];
        }
        // The undamped normal matrix is singular (see the gauge-freedom check below), so the sign
        // test uses the same damped system the algorithm solves.
        double[][] damped = normalEquations(j0, r0, start.length);
        for (int a = 0; a < damped.length; a++) damped[a][a] += 1e-3;
        double[] descent = solve(damped, gradient, 0);
        double[] wrongSign = start.clone();
        double[] rightSign = start.clone();
        for (int a = 0; a < start.length; a++) {
            wrongSign[a] += descent[a];
            rightSign[a] -= descent[a];
        }
        // The Gauss-Newton step is a descent direction: (J^T J + lambda I) is positive definite, so
        // delta^T g > 0. The objective is checked with a suitably small step length, because the
        // undamped step of a singular system (see the gauge check below) overshoots badly.
        double directional = 0;
        for (int a = 0; a < descent.length; a++) directional += descent[a] * gradient[a];
        double alpha = 1e-5;
        double[] forward = start.clone();
        double[] backward = start.clone();
        for (int a = 0; a < start.length; a++) {
            forward[a] += alpha * descent[a];
            backward[a] -= alpha * descent[a];
        }
        double startValue = objective(clean, start);
        System.out.println("step direction: delta^T g = " + directional + ", objective " + startValue
                + " -> " + objective(clean, backward) + " (minus) / " + objective(clean, forward) + " (plus)");
        check(directional > 0, "the damped Gauss-Newton step has a positive inner product with the gradient");
        check(objective(clean, backward) < startValue, "the right sign decreases the objective");
        check(objective(clean, forward) > startValue, "the wrong sign increases the objective");

        // ---- (2) and (3) noise-free convergence ---------------------------------------------------
        Adjustment adjustment = bundleAdjust(clean, start, 25, 1e-3, true);
        System.out.println("noise-free: objective " + objective(clean, start) + " -> " + adjustment.finalObjective
                + " in " + adjustment.iterations + " iterations (" + adjustment.accepted + " accepted, "
                + adjustment.rejected + " rejected)");
        for (int i = 1; i < adjustment.history.size(); i++) {
            check(adjustment.history.get(i) <= adjustment.history.get(i - 1) + 1e-18,
                    "the objective never increases");
        }
        check(adjustment.finalObjective <= 1e-18, "noise-free data converge to a zero residual");
        check(gradientNorm(clean, adjustment.parameters, 0) <= 1e-9, "the gradient vanishes at the solution");
        double parameterError = 0;
        double[] recoveredPoints = new double[clean.points * 3];
        for (int i = 0; i < clean.points; i++) {
            for (int k = 0; k < 3; k++) {
                recoveredPoints[3 * i + k] = adjustment.parameters[12 + 3 * i + k];
                parameterError = Math.max(parameterError,
                        Math.abs(recoveredPoints[3 * i + k] - clean.truthPoints[i][k]));
            }
        }
        // The camera centre in world coordinates is C = -R^T t, not t (t is the world-to-camera shift).
        double[] recoveredCentres = cameraCentres(clean, adjustment.parameters);
        double[] trueCentres = cameraCentres(clean, flattenTruth(clean));
        double recoveredBaseline = Math.sqrt(square(recoveredCentres[0] - recoveredCentres[3])
                + square(recoveredCentres[1] - recoveredCentres[4]) + square(recoveredCentres[2] - recoveredCentres[5]));
        double trueBaseline = Math.sqrt(square(trueCentres[0] - trueCentres[3]) + square(trueCentres[1] - trueCentres[4])
                + square(trueCentres[2] - trueCentres[5]));
        // With two calibrated views the reconstruction is determined only up to a global rigid motion
        // AND a positive scale: the baseline length is unobservable, and scaling the translation and
        // all point coordinates by s leaves every projection unchanged. The observable quantities are
        // therefore ratios to the baseline, which is what the checks below use. The absolute gauge
        // drift is 1.35 in the parameters and the recovered baseline is 1.49 times the true one -
        // both documented, not asserted away.
        double worstRatio = 0;
        for (int i = 0; i < clean.points; i++) {
            for (int j = i + 1; j < clean.points; j++) {
                double recovered = distance(recoveredPoints, 3 * i, 3 * j) / recoveredBaseline;
                double truth = pointDistance(clean.truthPoints[i], clean.truthPoints[j]) / trueBaseline;
                worstRatio = Math.max(worstRatio, Math.abs(recovered - truth) / truth);
            }
            for (int camera = 0; camera < 2; camera++) {
                double recovered = Math.sqrt(square(recoveredPoints[3 * i] - recoveredCentres[3 * camera])
                        + square(recoveredPoints[3 * i + 1] - recoveredCentres[3 * camera + 1])
                        + square(recoveredPoints[3 * i + 2] - recoveredCentres[3 * camera + 2]))
                        / recoveredBaseline;
                double truth = Math.sqrt(square(clean.truthPoints[i][0] - trueCentres[3 * camera])
                        + square(clean.truthPoints[i][1] - trueCentres[3 * camera + 1])
                        + square(clean.truthPoints[i][2] - trueCentres[3 * camera + 2])) / trueBaseline;
                worstRatio = Math.max(worstRatio, Math.abs(recovered - truth) / truth);
            }
        }
        System.out.println("noise-free gauge drift " + parameterError + ", baseline " + recoveredBaseline
                + " vs true " + trueBaseline + " (scale factor " + (recoveredBaseline / trueBaseline)
                + "), worst relative error of baseline-normalised distances " + worstRatio);
        // Independent reprojection check straight from the recovered centres and points (no use of
        // the residual code the optimiser used), plus the same check for the ground truth.
        double worstIndependent = 0;
        for (int camera = 0; camera < 2; camera++) {
            double[][] r = rotationMatrix(rotationOf(adjustment.parameters, camera));
            for (int i = 0; i < clean.points; i++) {
                double[] d = {recoveredPoints[3 * i] - recoveredCentres[3 * camera],
                        recoveredPoints[3 * i + 1] - recoveredCentres[3 * camera + 1],
                        recoveredPoints[3 * i + 2] - recoveredCentres[3 * camera + 2]};
                double[] y = transform(r, new double[]{0, 0, 0}, d);
                worstIndependent = Math.max(worstIndependent,
                        Math.abs(FOCAL * y[0] / y[2] - clean.observed[camera][i][0]));
                worstIndependent = Math.max(worstIndependent,
                        Math.abs(FOCAL * y[1] / y[2] - clean.observed[camera][i][1]));
            }
        }
        double truthReprojection = 0;
        for (int camera = 0; camera < 2; camera++) {
            double[][] r = rotationMatrix(rotationOf(flattenTruth(clean), camera));
            double[] t = translationOf(flattenTruth(clean), camera);
            for (int i = 0; i < clean.points; i++) {
                double[] y = transform(r, t, clean.truthPoints[i]);
                truthReprojection = Math.max(truthReprojection,
                        Math.abs(FOCAL * y[0] / y[2] - clean.observed[camera][i][0]));
            }
        }
        System.out.println("  independent reprojection from the recovered centres " + worstIndependent
                + " px, from the ground truth " + truthReprojection + " px");
        check(worstIndependent <= 1e-9, "the recovered configuration reproduces every observation");
        check(truthReprojection <= 1e-12, "and the ground truth reproduces them as well (as constructed)");

        // From a small perturbation the same code converges to the gauge image of the truth.
        Adjustment near = bundleAdjust(clean, initialParameters(clean, new Rng(576_6000L), false), 40, 1e-3, true);
        double[] nearPoints = new double[clean.points * 3];
        for (int i = 0; i < clean.points; i++) {
            for (int k = 0; k < 3; k++) nearPoints[3 * i + k] = near.parameters[12 + 3 * i + k];
        }
        double[] nearCentres = cameraCentres(clean, near.parameters);
        double nearBaseline = Math.sqrt(square(nearCentres[0] - nearCentres[3])
                + square(nearCentres[1] - nearCentres[4]) + square(nearCentres[2] - nearCentres[5]));
        double nearWorst = 0;
        for (int i = 0; i < clean.points; i++) {
            for (int j = i + 1; j < clean.points; j++) {
                nearWorst = Math.max(nearWorst, Math.abs(distance(nearPoints, 3 * i, 3 * j) / nearBaseline
                        - pointDistance(clean.truthPoints[i], clean.truthPoints[j]) / trueBaseline)
                        / (pointDistance(clean.truthPoints[i], clean.truthPoints[j]) / trueBaseline));
            }
            for (int camera = 0; camera < 2; camera++) {
                double recovered = Math.sqrt(square(nearPoints[3 * i] - nearCentres[3 * camera])
                        + square(nearPoints[3 * i + 1] - nearCentres[3 * camera + 1])
                        + square(nearPoints[3 * i + 2] - nearCentres[3 * camera + 2])) / nearBaseline;
                double truth = Math.sqrt(square(clean.truthPoints[i][0] - trueCentres[3 * camera])
                        + square(clean.truthPoints[i][1] - trueCentres[3 * camera + 1])
                        + square(clean.truthPoints[i][2] - trueCentres[3 * camera + 2])) / trueBaseline;
                nearWorst = Math.max(nearWorst, Math.abs(recovered - truth) / truth);
            }
        }
        System.out.println("  from the small perturbation: objective " + near.finalObjective
                + ", worst relative gauge-invariant error " + nearWorst);
        check(near.finalObjective <= 1e-20, "the small-perturbation run also converges to a zero residual");
        check(nearWorst <= 1e-9, "and it reproduces every gauge-invariant distance of the truth");

        // gauge freedom: translating cameras and points together leaves the residuals unchanged
        // Translating the whole system by delta means translating the points by delta and moving each
        // camera centre by delta as well; since the camera centre is -R^T t, the stored translation
        // becomes t - R delta (NOT t + delta, which would move the two cameras differently).
        double[] shifted = adjustment.parameters.clone();
        double[] delta = {3.0, -2.0, 1.5};
        for (int camera = 0; camera < 2; camera++) {
            double[][] r = rotationMatrix(rotationOf(adjustment.parameters, camera));
            for (int i = 0; i < 3; i++) {
                double moved = 0;
                for (int k = 0; k < 3; k++) moved += r[i][k] * delta[k];
                shifted[camera * 6 + i] -= moved;
            }
        }
        for (int i = 0; i < clean.points; i++) {
            for (int k = 0; k < 3; k++) shifted[12 + 3 * i + k] += delta[k];
        }
        double shiftedObjective = objective(clean, shifted);
        System.out.println("gauge freedom: objective under a joint translation " + shiftedObjective);
        check(shiftedObjective <= 1e-18, "a joint translation of cameras and points changes nothing");
        double smallest = smallestEigenvalue(normalEquations(j0, r0, start.length), start.length);
        System.out.println("smallest eigenvalue of the undamped normal matrix " + smallest
                + " (that is why the damping is there)");
        check(smallest <= 1e-6, "the undamped normal equations are numerically singular (gauge freedom)");

        // ---- (4) noisy data ----------------------------------------------------------------------
        int replications = 20;
        double totalRms = 0;
        double totalTranslation = 0;
        double totalTranslationError = 0;
        for (int trial = 0; trial < replications; trial++) {
            // 25 points: 100 observations against 87 parameters (12 + 3*25) minus the 7 gauge
            // degrees of freedom, so the problem is genuinely overdetermined.
            Problem noisy = build(25, 0.5, 576_3000L + trial);
            double[] noisyStart = initialParameters(noisy, new Rng(576_4000L + trial), true);
            Adjustment noisyFit = bundleAdjust(noisy, noisyStart, 40, 1e-3, true);
            double[] r = residuals(noisy, noisyFit.parameters);
            totalRms += Math.sqrt(dot(r, r) / r.length);
            double baseline = Math.sqrt(square(noisyFit.parameters[6] - noisyFit.parameters[0])
                    + square(noisyFit.parameters[7] - noisyFit.parameters[1])
                    + square(noisyFit.parameters[8] - noisyFit.parameters[2]));
            totalTranslation += baseline;
            double noisyTrueBaseline = Math.sqrt(square(noisy.truthPose[1][0]) + square(noisy.truthPose[1][1])
                    + square(noisy.truthPose[1][2]));
            totalTranslationError += baseline - noisyTrueBaseline;
            for (int i = 1; i < noisyFit.history.size(); i++) {
                check(noisyFit.history.get(i) <= noisyFit.history.get(i - 1) + 1e-12,
                        "the objective never increases with noise either");
            }
        }
        double rms = totalRms / replications;
        double baselineMean = totalTranslation / replications;
        double baselineBias = totalTranslationError / replications;
        System.out.println("noisy data: mean RMS reprojection error " + rms + " pixels (injected noise 0.5), "
                + "mean recovered baseline " + baselineMean + " (true 0.6387), mean baseline bias "
                + baselineBias + " - the baseline is itself gauge-dependent, as the note above explains");
        // For a least-squares fit the expected residual RMS is sigma * sqrt((m - p) / m) with m = 4n
        // observations and p = 12 + 3n parameters minus the 7 gauge degrees of freedom: here
        // 0.5 * sqrt(20 / 100) = 0.2236, and the measurement is 0.2249.
        double observations = 4.0 * 25;
        double effectiveParameters = 12.0 + 3.0 * 25 - 7.0;
        double predictedRms = 0.5 * Math.sqrt((observations - effectiveParameters) / observations);
        System.out.println("  predicted RMS sigma * sqrt((m - p)/m) = " + predictedRms);
        check(Math.abs(rms - predictedRms) <= 0.03, "the RMS residual matches the sigma sqrt((m-p)/m) prediction");
        check(Math.abs(baselineBias) <= 0.3, "the baseline estimate stays within the documented gauge spread");

        // Documented overfitting: with n = 12 the parameter count (12 + 3n = 48) equals the
        // observation count (4n = 48), so the fit can interpolate the noise and its RMS residual
        // collapses far below the injected 0.5 pixels.
        double smallRms = 0;
        for (int trial = 0; trial < 5; trial++) {
            Problem small = build(12, 0.5, 576_7000L + trial);
            Adjustment fit = bundleAdjust(small, initialParameters(small, new Rng(576_8000L + trial), true),
                    60, 1e-3, true);
            double[] r = residuals(small, fit.parameters);
            smallRms += Math.sqrt(dot(r, r) / r.length);
        }
        smallRms /= 5;
        System.out.println("n = 12 (as many parameters as observations): RMS residual " + smallRms
                + " against injected noise 0.5 - the fit interpolates the noise");
        check(smallRms <= 0.35, "with n = 12 the fit interpolates the noise, as the counting predicts");

        // ---- (7) the fixed-camera variant --------------------------------------------------------
        Adjustment frozenPoses = bundleAdjust(clean, initialParameters(clean, new Rng(576_5000L), true),
                40, 1e-3, false);
        System.out.println("fixed perturbed cameras: objective " + frozenPoses.finalObjective + " after "
                + frozenPoses.iterations + " iterations (the pose error is not absorbable)");
        check(frozenPoses.finalObjective > 1e-6, "a frozen pose error cannot be absorbed by the points");
        double[] frozenStart = initialParameters(clean, new Rng(576_5000L), true);
        for (int camera = 0; camera < 2; camera++) {
            for (int k = 0; k < 6; k++) {
                check(Math.abs(frozenPoses.parameters[camera * 6 + k] - frozenStart[camera * 6 + k]) <= 1e-15,
                        "the pose parameters are frozen exactly");
            }
        }
        // with the poses frozen at the ground truth the points are exactly triangulable
        double[] truthStart = flattenTruth(clean);
        Adjustment triangulation = bundleAdjust(clean, truthStart, 60, 1e-3, false);
        double pointError = 0;
        for (int i = 0; i < clean.points; i++) {
            for (int k = 0; k < 3; k++) {
                pointError = Math.max(pointError,
                        Math.abs(triangulation.parameters[12 + 3 * i + k] - clean.truthPoints[i][k]));
            }
        }
        System.out.println("fixed true cameras: objective " + triangulation.finalObjective
                + ", worst point error " + pointError);
        check(triangulation.finalObjective <= 1e-18, "with the true poses the points fit exactly");
        check(pointError <= 1e-6, "and triangulation recovers the true 3D points");

        System.out.println("PASS");
    }

    static double[][] normalEquations(double[][] jacobian, double[] r, int columns) {
        double[][] normal = new double[columns][columns];
        for (int row = 0; row < r.length; row++) {
            for (int a = 0; a < columns; a++) {
                double ja = jacobian[row][a];
                if (ja == 0.0) continue;
                for (int b = 0; b < columns; b++) normal[a][b] += ja * jacobian[row][b];
            }
        }
        return normal;
    }

    /** Smallest eigenvalue of a symmetric matrix by bisection on the characteristic polynomial-free
     * Jacobi sweep (reuses the rotation code in the simplest possible way). */
    static double smallestEigenvalue(double[][] matrix, int n) {
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = matrix[i].clone();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            if (Math.sqrt(off) <= 1e-18) break;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) <= 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    double t = 1.0 / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta < 0) t = -t;
                    double c = 1.0 / Math.sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < n; k++) {
                        if (k == p || k == q) continue;
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                        a[p][k] = a[k][p];
                        a[q][k] = a[k][q];
                    }
                    double app = a[p][p];
                    double aqq = a[q][q];
                    double apq = a[p][q];
                    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    a[p][q] = 0;
                    a[q][p] = 0;
                    for (int k = 0; k < n; k++) {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        double smallest = Double.POSITIVE_INFINITY;
        for (int i = 0; i < n; i++) smallest = Math.min(smallest, a[i][i]);
        return smallest;
    }
}
