/** Task 790 - Implement bundle adjustment sparse solver outline with Levenberg-Marquardt. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Bundle adjustment refines camera poses and world points so the reprojection error is minimal. Cameras
 * live in SE(2) (theta, t) and point j seen by camera i gives the two residuals
 *
 *     r_ij = R(theta_i)^T (P_j - t_i) - z_ij
 *
 * with the analytic blocks dr/dtheta = dR^T/dtheta (P - t), dr/dt = -R^T and dr/dP = R^T. Packing the
 * parameters as (free cameras, points) makes the normal equations block sparse,
 *
 *     (J^T J + lambda I) = [ U + lambda I   W    ]        U: 3x3 per camera (block diagonal)
 *                          [ W^T           V + lambda I ]  V: 2x2 per point  (block diagonal)
 *
 * and the point block is eliminated with the Schur complement:
 *
 *     S = U + lambda I - sum_j W_j (V_j + lambda I)^-1 W_j^T,   b = g_c - sum_j W_j (V_j + lambda I)^-1 g_j
 *     S dc = -b,   then   dp_j = -(V_j + lambda I)^-1 (g_j + W_j^T dc)
 *
 * so the linear solve is 3*|cameras| square instead of 3*|cameras| + 2*|points|, which is the entire point
 * of sparse bundle adjustment. Levenberg-Marquardt accepts a step only when the cost falls (then lambda is
 * divided by 3) and otherwise restores the state and multiplies lambda by 10. Camera 0 is the gauge and is
 * held fixed, removing the 3 remaining gauge freedoms.
 *
 * Simplified (this is the "outline" variant): 2D poses, three cameras and six points, dense Gaussian
 * elimination on the small reduced system (no sparse Cholesky, no fill-reducing ordering, no inner Krylov
 * solver, no robust loss and no covariance estimation). The analytic Jacobian, the normal equations, the
 * Schur elimination and the damping loop are the real ones.
 *
 * Verified: noise-free data has exactly zero residual at the ground truth; the analytic Jacobian matches a
 * central-difference Jacobian; the Schur step equals a dense solve of the same damped normal equations to
 * 1e-10 for several lambdas - that is the independent oracle for the sparse algebra - while the reduced
 * system is 6x6 instead of 18x18; the Jacobian is only about a third dense; LM recovers the ground truth
 * from noise-free data and from a badly perturbed start, with a non-increasing cost trace, a vanishing
 * gradient and a gauge camera that never moves; an undamped Gauss-Newton run from the same start has to
 * bail out on its first rejection while LM raises lambda and continues; a noisy problem converges to the
 * noise floor with a non-increasing cost; a problem with no free cameras still solves through the same code
 * path; bad lambdas are rejected; and repeated runs are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A small 2D bundle adjustment problem: analytic Jacobian, Schur-complement step, LM damping. */
class Bundle {
    final int cameras;              // camera 0 is the gauge and never moves
    final int points;
    final double[] theta, tx, ty;
    final double[] wx, wy;
    final int[] obsCamera, obsPoint;
    final double[] zx, zy;

    Bundle(int cameras, int points, int[] obsCamera, int[] obsPoint) {
        this.cameras = cameras; this.points = points; this.obsCamera = obsCamera; this.obsPoint = obsPoint;
        theta = new double[cameras]; tx = new double[cameras]; ty = new double[cameras]; wx = new double[points];
        wy = new double[points]; zx = new double[obsCamera.length]; zy = new double[obsCamera.length];
    }

    int freeCameras() { return cameras - 1; }

    int parameterCount() { return 3 * freeCameras() + 2 * points; }

    int cameraBase(int c) { return 3 * (c - 1); }

    int pointBase(int j) { return 3 * freeCameras() + 2 * j; }

    double[] project(int c, double worldX, double worldY) {
        double ct = Math.cos(theta[c]), st = Math.sin(theta[c]); double ux = worldX - tx[c], uy = worldY - ty[c];
        return new double[]{ct * ux + st * uy, -st * ux + ct * uy};
    }

    /** Fills z from the current state, i.e. noise-free measurements of the ground truth. */
    void synthesise() {
        for (int k = 0; k < obsCamera.length; k++) {
            double[] p = project(obsCamera[k], wx[obsPoint[k]], wy[obsPoint[k]]); zx[k] = p[0]; zy[k] = p[1];
        }
    }

    double[] residuals() {
        double[] r = new double[2 * obsCamera.length];
        for (int k = 0; k < obsCamera.length; k++) {
            double[] p = project(obsCamera[k], wx[obsPoint[k]], wy[obsPoint[k]]);
            r[2 * k] = p[0] - zx[k]; r[2 * k + 1] = p[1] - zy[k];
        }
        return r;
    }

    double cost() {
        double sum = 0; for (double v : residuals()) sum += v * v; return sum;
    }

    double[][] jacobian() {
        int rows = 2 * obsCamera.length; double[][] j = new double[rows][parameterCount()];
        for (int k = 0; k < obsCamera.length; k++) {
            int c = obsCamera[k], p = obsPoint[k]; double ct = Math.cos(theta[c]), st = Math.sin(theta[c]);
            double ux = wx[p] - tx[c], uy = wy[p] - ty[c];
            if (c > 0) {
                int base = cameraBase(c); j[2 * k][base] = -st * ux + ct * uy;
                j[2 * k + 1][base] = -ct * ux - st * uy; j[2 * k][base + 1] = -ct;
                j[2 * k + 1][base + 1] = st; j[2 * k][base + 2] = -st; j[2 * k + 1][base + 2] = -ct;
            }
            int base = pointBase(p); j[2 * k][base] = ct;
            j[2 * k + 1][base] = -st; j[2 * k][base + 1] = st; j[2 * k + 1][base + 1] = ct;
        }
        return j;
    }

    double[] gradient() {
        double[][] j = jacobian(); double[] r = residuals(); double[] g = new double[parameterCount()];
        for (int row = 0; row < r.length; row++)
            for (int col = 0; col < g.length; col++) g[col] += j[row][col] * r[row];
        return g;
    }

    /** Dense reference solve of (J^T J + lambda I) d = -g, used as the oracle for the Schur step. */
    double[] denseStep(double lambda) {
        int n = parameterCount(); double[][] j = jacobian();
        double[] r = residuals(); double[][] h = new double[n][n]; double[] g = new double[n];
        for (int row = 0; row < j.length; row++) {
            gVectorsAdd(g, j[row], r[row]);
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++) h[a][b] += j[row][a] * j[row][b];
        }
        for (int a = 0; a < n; a++) h[a][a] += lambda; for (int a = 0; a < n; a++) g[a] = -g[a]; return solve(h, g);
    }

    private static void gVectorsAdd(double[] g, double[] row, double r) {
        for (int i = 0; i < g.length; i++) g[i] += row[i] * r;
    }

    /**
     * Sparse step: build U, V, W and g block by block, eliminate the points with the Schur complement, solve
     * the reduced camera system and back-substitute the point updates.
     */
    double[] schurStep(double lambda) {
        if (lambda < 0) throw new IllegalArgumentException("lambda must not be negative: " + lambda);
        int fc = freeCameras(), n = parameterCount(); double[][][] u = new double[fc][3][3];
        double[][][] v = new double[points][2][2]; double[][][][] w = new double[points][fc][3][2];
        double[][] gc = new double[fc][3]; double[] gp = new double[2 * points]; double[] r = residuals();
        for (int k = 0; k < obsCamera.length; k++) {
            int c = obsCamera[k], p = obsPoint[k]; double ct = Math.cos(theta[c]), st = Math.sin(theta[c]);
            double ux = wx[p] - tx[c], uy = wy[p] - ty[c];
            double r0 = r[2 * k], r1 = r[2 * k + 1]; double[][] jp = {{ct, st}, {-st, ct}};
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) v[p][a][b] += jp[0][a] * jp[0][b] + jp[1][a] * jp[1][b];
                gp[2 * p + a] += jp[0][a] * r0 + jp[1][a] * r1;
            }
            if (c == 0) continue; int index = c - 1;
            double[][] jc = {{-st * ux + ct * uy, -ct, -st}, {-ct * ux - st * uy, st, -ct}};
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) u[index][a][b] += jc[0][a] * jc[0][b] + jc[1][a] * jc[1][b];
                gc[index][a] += jc[0][a] * r0 + jc[1][a] * r1;
                for (int b = 0; b < 2; b++) w[p][index][a][b] += jc[0][a] * jp[0][b] + jc[1][a] * jp[1][b];
            }
        }
        double[][] vInv = new double[points][4];        // [00, 01, 10, 11] of (V_j + lambda I)^-1
        for (int j = 0; j < points; j++) {
            double a = v[j][0][0] + lambda, b = v[j][0][1], c = v[j][1][0], d = v[j][1][1] + lambda;
            double det = a * d - b * c; vInv[j][0] = d / det;
            vInv[j][1] = -b / det; vInv[j][2] = -c / det; vInv[j][3] = a / det;
        }
        double[][] s = new double[3 * fc][3 * fc]; double[] rhs = new double[3 * fc];
        for (int i = 0; i < fc; i++) {
            for (int a = 0; a < 3; a++) {
                rhs[3 * i + a] = -gc[i][a];
                for (int b = 0; b < 3; b++) s[3 * i + a][3 * i + b] = u[i][a][b] + (a == b ? lambda : 0);
            }
        }
        for (int j = 0; j < points; j++) {
            double g0 = gp[2 * j], g1 = gp[2 * j + 1];
            for (int c1 = 0; c1 < fc; c1++) {
                if (isZero(w[j][c1])) continue;
                double[][] z = new double[2][3];        // (V_j + lambda I)^-1 W_jc1^T
                for (int a = 0; a < 3; a++) {
                    double t0 = w[j][c1][a][0], t1 = w[j][c1][a][1];
                    z[0][a] = vInv[j][0] * t0 + vInv[j][1] * t1;
                    z[1][a] = vInv[j][2] * t0 + vInv[j][3] * t1;
                }
                for (int c2 = 0; c2 < fc; c2++) {
                    if (isZero(w[j][c2])) continue;
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            s[3 * c1 + a][3 * c2 + b] -= z[0][a] * w[j][c2][b][0] + z[1][a] * w[j][c2][b][1];
                }
                double y0 = vInv[j][0] * g0 + vInv[j][1] * g1, y1 = vInv[j][2] * g0 + vInv[j][3] * g1;
                for (int a = 0; a < 3; a++) rhs[3 * c1 + a] += w[j][c1][a][0] * y0 + w[j][c1][a][1] * y1;
            }
        }
        double[] dc = solve(s, rhs); double[] delta = new double[n];
        for (int i = 0; i < fc; i++) for (int a = 0; a < 3; a++) delta[3 * i + a] = dc[3 * i + a];
        for (int j = 0; j < points; j++) {
            double y0 = gp[2 * j], y1 = gp[2 * j + 1];
            for (int c = 0; c < fc; c++)
                for (int a = 0; a < 3; a++) {
                    y0 += w[j][c][a][0] * delta[3 * c + a]; y1 += w[j][c][a][1] * delta[3 * c + a];
                }
            delta[pointBase(j)] = -(vInv[j][0] * y0 + vInv[j][1] * y1);
            delta[pointBase(j) + 1] = -(vInv[j][2] * y0 + vInv[j][3] * y1);
        }
        return delta;
    }

    private static boolean isZero(double[][] block) {
        for (double[] row : block) for (double v : row) if (v != 0) return false; return true;
    }

    void apply(double[] delta) {
        for (int i = 0; i < freeCameras(); i++) {
            theta[i + 1] += delta[3 * i]; tx[i + 1] += delta[3 * i + 1]; ty[i + 1] += delta[3 * i + 2];
        }
        for (int j = 0; j < points; j++) {
            wx[j] += delta[pointBase(j)]; wy[j] += delta[pointBase(j) + 1];
        }
    }

    double[] pack() {
        double[] p = new double[parameterCount()];
        for (int i = 0; i < freeCameras(); i++) {
            p[3 * i] = theta[i + 1]; p[3 * i + 1] = tx[i + 1]; p[3 * i + 2] = ty[i + 1];
        }
        for (int j = 0; j < points; j++) {
            p[pointBase(j)] = wx[j]; p[pointBase(j) + 1] = wy[j];
        }
        return p;
    }

    void unpack(double[] p) {
        for (int i = 0; i < freeCameras(); i++) {
            theta[i + 1] = p[3 * i]; tx[i + 1] = p[3 * i + 1]; ty[i + 1] = p[3 * i + 2];
        }
        for (int j = 0; j < points; j++) {
            wx[j] = p[pointBase(j)]; wy[j] = p[pointBase(j) + 1];
        }
    }

    static double[] solve(double[][] a, double[] b) {
        int n = b.length; double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n); m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
            double[] t = m[col]; m[col] = m[pivot]; m[pivot] = t;
            for (int r = col + 1; r < n; r++) {
                double f = m[r][col] / m[col][col]; for (int c = col; c <= n; c++) m[r][c] -= f * m[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = m[i][n]; for (int j = i + 1; j < n; j++) sum -= m[i][j] * x[j]; x[i] = sum / m[i][i];
        }
        return x;
    }
}

/** Result of a Levenberg-Marquardt run. */
class LmRun {
    final int accepted, rejected;
    final double[] costTrace;
    final double finalCost;

    LmRun(int accepted, int rejected, double[] costTrace, double finalCost) {
        this.accepted = accepted; this.rejected = rejected; this.costTrace = costTrace; this.finalCost = finalCost;
    }
}

public class Solution {
    static LmRun bundleLm(Bundle problem, int maxIterations, double lambda0, boolean damped, double floor) {
        double[] saved = problem.pack(); double lambda = lambda0;
        double cost = problem.cost(); List<Double> trace = new ArrayList<>();
        trace.add(cost); int accepted = 0, rejected = 0;
        for (int it = 0; it < maxIterations; it++) {
            double[] step = problem.schurStep(lambda); problem.unpack(saved);
            problem.apply(step); double candidate = problem.cost();
            if (candidate < cost) {
                saved = problem.pack(); cost = candidate; trace.add(cost); accepted++;
                lambda = Math.max(floor, lambda / 3); if (maxAbs(step) < 1e-12) break;
            } else {
                problem.unpack(saved); rejected++;
                if (!damped) break;             // undamped Gauss-Newton has nowhere to go
                lambda *= 10;
            }
        }
        double[] array = new double[trace.size()]; for (int i = 0; i < array.length; i++) array[i] = trace.get(i);
        return new LmRun(accepted, rejected, array, cost);
    }

    static double maxAbs(double[] v) {
        double best = 0; for (double x : v) best = Math.max(best, Math.abs(x)); return best;
    }

    static boolean nonIncreasing(double[] trace) {
        for (int i = 1; i < trace.length; i++) if (trace[i] > trace[i - 1] + 1e-15) return false; return true;
    }

    static double parameterError(Bundle b, double[] a, double[] c) {
        double worst = 0; for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - c[i]));
        return worst;
    }

    static Bundle makeProblem(SplitMix64 rng, double noise) {
        int[] obsCamera = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2};
        int[] obsPoint = {0, 1, 2, 3, 1, 2, 3, 4, 5, 0, 2, 4, 5}; Bundle b = new Bundle(3, 6, obsCamera, obsPoint);
        double[][] pose = {{0, 0, 0}, {0.7, -1.5, 0.3}, {-0.5, -1.2, -0.9}};
        for (int c = 0; c < 3; c++) {
            b.theta[c] = pose[c][0]; b.tx[c] = pose[c][1]; b.ty[c] = pose[c][2];
        }
        double[] wx = {0.5, 1.4, 2.2, 0.9, 1.7, 2.6}; double[] wy = {0.7, 0.9, 1.3, 1.8, 2.1, 1.9};
        System.arraycopy(wx, 0, b.wx, 0, 6); System.arraycopy(wy, 0, b.wy, 0, 6); b.synthesise();
        if (noise > 0) {
            for (int k = 0; k < b.zx.length; k++) {
                b.zx[k] += noise * (rng.nextDouble() - 0.5); b.zy[k] += noise * (rng.nextDouble() - 0.5);
            }
        }
        return b;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240790L); Bundle problem = makeProblem(rng, 0.0);
        double[] truth = problem.pack();
        // --- the noise-free data is exactly consistent with the ground truth ----------------
        check(maxAbs(problem.residuals()) == 0.0, "synthesised measurements must be exact at the truth");
        check(problem.cost() == 0.0, "the ground truth must have zero cost on noise-free data");
        check(problem.parameterCount() == 18, "parameter count " + problem.parameterCount());
        check(problem.freeCameras() == 2, "camera 0 is the gauge");
        // --- the analytic Jacobian against central differences ------------------------------
        double[][] analytic = problem.jacobian().clone(); double eps = 1e-6; double worstDeviation = 0;
        for (int col = 0; col < problem.parameterCount(); col++) {
            double[] saved = problem.pack(); double[] plus = saved.clone(), minus = saved.clone();
            plus[col] += eps; minus[col] -= eps; problem.unpack(plus); double[] up = problem.residuals();
            problem.unpack(minus); double[] down = problem.residuals(); problem.unpack(saved);
            for (int row = 0; row < up.length; row++)
                worstDeviation = Math.max(worstDeviation, Math.abs((up[row] - down[row]) / (2 * eps)
                        - analytic[row][col]));
        }
        check(worstDeviation < 1e-7, "analytic Jacobian deviation from finite differences "
                + worstDeviation);
        // --- Schur step against the dense solve (the oracle for the sparse algebra) ----------
        double[][] probes = {{0.7, -1.5, 0.3}, {-0.2, -1.0, -0.5}}; double[] tweaked = truth.clone();
        System.arraycopy(probes[1], 0, tweaked, 0, 3); problem.unpack(tweaked);
        for (double lambda : new double[]{1e-3, 1.0, 100.0}) {
            double[] schur = problem.schurStep(lambda);
            double[] dense = problem.denseStep(lambda); double gap = 0;
            for (int i = 0; i < schur.length; i++)
                gap = Math.max(gap, Math.abs(schur[i] - dense[i]) / (1 + Math.abs(dense[i])));
            check(gap < 1e-10, "Schur step must match the dense solve at lambda = " + lambda + ", gap " + gap);
        }
        // and the normal-equation blocks really are sparse
        double[][] j = problem.jacobian(); int nonzeros = 0;
        for (double[] row : j) for (double x : row) if (x != 0) nonzeros++;
        double density = (double) nonzeros / (j.length * j[0].length);
        check(density < 0.5, "the Jacobian is sparse, density " + density);
        check(3 * problem.freeCameras() == 6 && problem.parameterCount() == 18,
                "the reduced system is 6x6 instead of 18x18");
        boolean rejectedNegative = false;
        try {
            problem.schurStep(-1.0);
        } catch (IllegalArgumentException expected) {
            rejectedNegative = true;
        }
        check(rejectedNegative, "a negative lambda must be rejected");
        // --- recovery from a perturbed start: LM versus undamped Gauss-Newton ---------------
        double[] badStart = truth.clone(); badStart[0] = 0.7 + 0.45;
        badStart[2] = 0.3 - 0.4; badStart[10] = truth[10] + 0.5;
        badStart[13] = truth[13] - 0.4; problem.unpack(badStart);
        double startCost = problem.cost(); LmRun lm = bundleLm(problem, 200, 1e-3, true, 1e-12);
        problem.unpack(badStart); LmRun gaussNewton = bundleLm(problem, 200, 1e-3, false, 1e-12);
        problem.unpack(badStart); LmRun lmAgain = bundleLm(problem, 200, 1e-3, true, 1e-12);
        check(lm.finalCost < startCost * 1e-12, "LM must reach the exact solution, cost " + lm.finalCost);
        check(lm.finalCost <= gaussNewton.finalCost, "LM " + lm.finalCost + " must beat Gauss-Newton "
                + gaussNewton.finalCost);
        check(gaussNewton.rejected > 0, "undamped Gauss-Newton must hit a rejection on this start");
        check(nonIncreasing(lm.costTrace), "the LM cost trace must be non-increasing");
        check(lm.rejected > 0, "the damped run must record its rejections, got " + lm.rejected);
        check(parameterError(problem, problem.pack(), truth) < 1e-8, "parameters recovered to "
                + parameterError(problem, problem.pack(), truth));
        check(problem.theta[0] == 0 && problem.tx[0] == 0 && problem.ty[0] == 0,
                "the gauge camera must never move");
        check(maxAbs(problem.gradient()) < 1e-8, "the gradient must vanish at the solution, "
                + maxAbs(problem.gradient()));
        check(lmAgain.finalCost == lm.finalCost
                && parameterError(problem, problem.pack(), truth) < 1e-8, "runs must be repeatable");
        // --- a noisy problem converges to the noise floor ------------------------------------
        SplitMix64 noiseRng = new SplitMix64(31337L); Bundle noisy = makeProblem(noiseRng, 0.01);
        double[] noisyTruth = noisy.pack(); double[] noisyStart = noisyTruth.clone();
        noisyStart[0] = 0.55; noisyStart[6] = noisyTruth[6] + 0.15;
        noisy.unpack(noisyStart); LmRun noisyRun = bundleLm(noisy, 300, 1.0, true, 1e-12);
        double rms = Math.sqrt(noisyRun.finalCost / noisy.residuals().length);
        check(nonIncreasing(noisyRun.costTrace), "the noisy cost trace must be non-increasing");
        check(rms < 0.02, "noisy residual RMS " + rms);
        check(rms > 0.0005, "the noisy problem must not fit the noise away, RMS " + rms);
        check(parameterError(noisy, noisy.pack(), noisyTruth) < 0.3, "noisy pose error "
                + parameterError(noisy, noisy.pack(), noisyTruth));
        // --- degenerate case: no free cameras, only points ------------------------------------
        Bundle fixed = new Bundle(1, 1, new int[]{0}, new int[]{0}); fixed.theta[0] = 0.4;
        fixed.tx[0] = -0.3; fixed.ty[0] = 0.2; fixed.wx[0] = 1.1; fixed.wy[0] = -0.6;
        fixed.synthesise(); fixed.wx[0] += 0.25; fixed.wy[0] += 0.1;
        check(fixed.parameterCount() == 2, "with no free cameras only the point is unknown");
        LmRun pointOnly = bundleLm(fixed, 60, 1e-3, true, 1e-14);
        check(pointOnly.finalCost < 1e-20, "a single fixed camera still solves for its point, cost "
                + pointOnly.finalCost);
        check(parameterError(fixed, fixed.pack(), new double[]{1.1, -0.6}) < 1e-9,
                "the point must return to its true position");
        System.out.println("analytic Jacobian deviation " + worstDeviation + ", Jacobian density " + density);
        System.out.println("LM cost " + lm.finalCost + " in " + lm.accepted + " accepted / " + lm.rejected
                + " rejected steps, Gauss-Newton " + gaussNewton.finalCost);
        System.out.println("noisy RMS " + rms + " after " + noisyRun.accepted + " accepted steps");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
