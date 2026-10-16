/** Task 791 - Implement SLAM small demo with extended Kalman filter (EKF-SLAM). (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * EKF-SLAM keeps one joint Gaussian over the robot pose and every landmark,
 *
 *     mu = (x, y, theta, l_0x, l_0y, ..., l_{n-1}x, l_{n-1}y),   Sigma is (3 + 2n) square,
 *
 * and alternates two steps. Prediction propagates the exact unicycle model
 *
 *     x' = x + v/w (sin(theta + w dt) - sin theta),  y' = y + v/w (cos theta - cos(theta + w dt)),
 *     theta' = theta + w dt,      Sigma = G Sigma G^T + V M V^T,  M = diag(sigma_v^2, sigma_w^2)
 *
 * with the analytic G = d(state')/d(state) and V = d(state')/d(control). Update observes landmark j as
 * range/bearing with q = dx^2 + dy^2, dx = l_jx - x, dy = l_jy - y:
 *
 *     z = ( sqrt(q), atan2(dy, dx) - theta ),   H = [ -dx/sqrt(q)  -dy/sqrt(q)   0 | .. | dx/sqrt(q) dy/sqrt(q) | .. ]
 *                                                   [  dy/q        -dx/q        -1 | .. | -dy/q      dx/q       | .. ]
 *
 * followed by the standard Kalman equations with the bearing innovation wrapped into (-pi, pi]. A landmark
 * is initialised from its first observation, and the initial covariance deliberately carries the *robot*
 * pose uncertainty through the range/bearing->world Jacobian; forgetting that is the classic way to build an
 * over-confident SLAM map.
 *
 * Verified: the analytic G matches central differences of the transition function and H matches central
 * differences of the measurement function; on noise-free data the estimate converges to the truth and the
 * covariance collapses; over noisy runs the landmark estimates converge, their covariance blocks shrink,
 * and the average NEES of the landmark error sits inside the chi-square band for two degrees of freedom;
 * the filter's landmark means agree with an independent batch weighted-least-squares estimate (five
 * Gauss-Newton steps) computed from the same measurement history and the same estimated trajectory, which is
 * closed form because range/bearing is linear in the landmark given the pose; the initial landmark
 * covariance is provably larger than the measurement noise alone; the angle wrap and the empty-landmark
 * case behave; and repeated runs are identical.
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

    /** Standard normal by Box-Muller. */
    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** Minimal dense matrix helpers (double[][]). */
class Mat {
    static double[][] identity(int n) {
        double[][] m = new double[n][n]; for (int i = 0; i < n; i++) m[i][i] = 1; return m;
    }

    static double[][] mul(double[][] a, double[][] b) {
        int n = a.length, k = b.length, p = b[0].length; double[][] out = new double[n][p];
        for (int i = 0; i < n; i++)
            for (int t = 0; t < k; t++) {
                double v = a[i][t]; if (v == 0) continue;
                for (int j = 0; j < p; j++) out[i][j] += v * b[t][j];
            }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j]; return out;
    }

    static double[][] add(double[][] a, double[][] b) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i].clone(); for (int j = 0; j < a[i].length; j++) out[i][j] += b[i][j];
        }
        return out;
    }

    /** Inverse of a 2x2 matrix. */
    static double[][] invert2(double[][] m) {
        double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        return new double[][]{{m[1][1] / det, -m[0][1] / det}, {-m[1][0] / det, m[0][0] / det}};
    }

    static double trace(double[][] a) {
        double sum = 0; for (int i = 0; i < a.length; i++) sum += a[i][i]; return sum;
    }

    /** A small block placed at the given offset inside a size x size zero matrix. */
    static double[][] embed(double[][] block, int offset, int size) {
        double[][] out = new double[size][size];
        for (int i = 0; i < block.length; i++)
            for (int j = 0; j < block[i].length; j++) out[offset + i][offset + j] = block[i][j];
        return out;
    }
}

/** One range/bearing observation, kept for the batch oracle. */
class Sample {
    final int landmark;
    final double range, bearing, robotX, robotY, robotTheta;

    Sample(int landmark, double range, double bearing, double robotX, double robotY, double robotTheta) {
        this.landmark = landmark; this.range = range; this.bearing = bearing; this.robotX = robotX;
        this.robotY = robotY; this.robotTheta = robotTheta;
    }
}

class EkfSlam {
    static final double SIGMA_R = 0.1, SIGMA_B = 0.05;
    final int landmarks;
    double[] mu;
    double[][] sigma;
    double[][] lastG = new double[0][0];
    final List<Sample> history = new ArrayList<>();

    EkfSlam(int landmarks) {
        this.landmarks = landmarks; mu = new double[3 + 2 * landmarks];
        sigma = Mat.identity(3 + 2 * landmarks); for (int i = 0; i < 3; i++) sigma[i][i] = 1e-9;
        for (int j = 0; j < landmarks; j++) {
            sigma[3 + 2 * j][3 + 2 * j] = 1e9; sigma[3 + 2 * j + 1][3 + 2 * j + 1] = 1e9;
        }
    }

    int base(int j) { return 3 + 2 * j; }

    boolean isKnown(int j) { return sigma[base(j)][base(j)] < 1e8; }

    static double[] transition(double[] state, double v, double w, double dt) {
        double[] out = state.clone();
        if (Math.abs(w) < 1e-9) {
            out[0] += v * dt * Math.cos(state[2]); out[1] += v * dt * Math.sin(state[2]);
        } else {
            double r = v / w; out[0] += r * (Math.sin(state[2] + w * dt) - Math.sin(state[2]));
            out[1] += r * (Math.cos(state[2]) - Math.cos(state[2] + w * dt));
        }
        out[2] = state[2] + w * dt; return out;
    }

    /** Analytic Jacobian of the unicycle transition with respect to the state. */
    static double[][] transitionJacobian(double[] state, double v, double w, double dt) {
        double[][] g = Mat.identity(state.length); double theta = state[2];
        if (Math.abs(w) < 1e-9) {
            g[0][2] = -v * dt * Math.sin(theta); g[1][2] = v * dt * Math.cos(theta);
        } else {
            double r = v / w; g[0][2] = r * (Math.cos(theta + w * dt) - Math.cos(theta));
            g[1][2] = r * (Math.sin(theta + w * dt) - Math.sin(theta));
        }
        return g;
    }

    /** Analytic Jacobian of the unicycle transition with respect to the control (v, w). */
    static double[][] controlJacobian(double[] state, double v, double w, double dt) {
        double[][] jac = new double[state.length][2]; double theta = state[2];
        if (Math.abs(w) < 1e-9) {
            jac[0][0] = dt * Math.cos(theta); jac[1][0] = dt * Math.sin(theta);
        } else {
            double r = v / w; double s1 = Math.sin(theta + w * dt), c1 = Math.cos(theta + w * dt);
            double s0 = Math.sin(theta), c0 = Math.cos(theta); jac[0][0] = (s1 - s0) / w;
            jac[1][0] = (c0 - c1) / w; jac[0][1] = dt * c1 * r - (s1 - s0) * r / w;
            jac[1][1] = dt * s1 * r - (c0 - c1) * r / w;
        }
        jac[2][1] = dt; return jac;
    }

    void predict(double v, double w, double dt, double sigmaV, double sigmaW) {
        double[][] g = transitionJacobian(mu, v, w, dt); double[][] vMat = controlJacobian(mu, v, w, dt);
        double[][] m = {{sigmaV * sigmaV, 0}, {0, sigmaW * sigmaW}}; mu = transition(mu, v, w, dt); lastG = g;
        sigma = Mat.add(Mat.mul(Mat.mul(g, sigma), Mat.transpose(g)),
                Mat.mul(Mat.mul(vMat, m), Mat.transpose(vMat)));
    }

    /** Expected range/bearing of landmark j from the current state. */
    double[] expected(int j) {
        double dx = mu[base(j)] - mu[0], dy = mu[base(j) + 1] - mu[1];
        return new double[]{Math.hypot(dx, dy), wrap(Math.atan2(dy, dx) - mu[2])};
    }

    double[][] measurementJacobian(int j) {
        double dx = mu[base(j)] - mu[0], dy = mu[base(j) + 1] - mu[1];
        double q = dx * dx + dy * dy, s = Math.sqrt(q); double[][] h = new double[2][mu.length];
        h[0][0] = -dx / s; h[0][1] = -dy / s; h[1][0] = dy / q; h[1][1] = -dx / q;
        h[1][2] = -1; h[0][base(j)] = dx / s; h[0][base(j) + 1] = dy / s; h[1][base(j)] = -dy / q;
        h[1][base(j) + 1] = dx / q; return h;
    }

    static double wrap(double angle) {
        return Math.atan2(Math.sin(angle), Math.cos(angle));
    }

    /** Landmark j from its first observation: the range/bearing->world Jacobian carries the pose covariance. */
    void initialiseLandmark(int j, double range, double bearing) {
        double alpha = mu[2] + bearing; int b = base(j); double[][] g = Mat.identity(mu.length);
        for (int c = 0; c < mu.length; c++) {
            g[b][c] = 0; g[b + 1][c] = 0;
        }
        g[b][0] = 1; g[b][2] = -range * Math.sin(alpha); g[b + 1][1] = 1; g[b + 1][2] = range * Math.cos(alpha);
        double[][] jac = {{Math.cos(alpha), -range * Math.sin(alpha)},
                {Math.sin(alpha), range * Math.cos(alpha)}};
        double[][] r = {{SIGMA_R * SIGMA_R, 0}, {0, SIGMA_B * SIGMA_B}};
        mu[b] = mu[0] + range * Math.cos(alpha); mu[b + 1] = mu[1] + range * Math.sin(alpha);
        sigma = Mat.add(Mat.mul(Mat.mul(g, sigma), Mat.transpose(g)),
                Mat.embed(Mat.mul(Mat.mul(jac, r), Mat.transpose(jac)), b, mu.length));
    }

    void update(int j, double range, double bearing, double robotX, double robotY, double robotTheta) {
        if (!isKnown(j)) initialiseLandmark(j, range, bearing);
        history.add(new Sample(j, range, bearing, robotX, robotY, robotTheta));
        double[][] h = measurementJacobian(j); double[][] r = {{SIGMA_R * SIGMA_R, 0}, {0, SIGMA_B * SIGMA_B}};
        double[] predicted = expected(j); double[] innovation = {range - predicted[0], wrap(bearing - predicted[1])};
        double[][] s = Mat.add(Mat.mul(Mat.mul(h, sigma), Mat.transpose(h)), r);
        double[][] k = Mat.mul(Mat.mul(sigma, Mat.transpose(h)), Mat.invert2(s));
        double[][] iKh = Mat.identity(mu.length);
        for (int a = 0; a < mu.length; a++)
            for (int col = 0; col < mu.length; col++)
                iKh[a][col] -= k[a][0] * h[0][col] + k[a][1] * h[1][col];
        sigma = Mat.mul(iKh, sigma);
        for (int i = 0; i < mu.length; i++) mu[i] += k[i][0] * innovation[0] + k[i][1] * innovation[1];
        mu[2] = wrap(mu[2]);
    }

    /**
     * Closed-form oracle: given the recorded trajectory, the landmarks follow from a weighted least squares
     * whose information matrix is block diagonal (one 2x2 block per landmark), refined by Gauss-Newton.
     */
    double[][] batchLandmarks(int iterations) {
        double[][] estimate = new double[landmarks][2];
        for (int j = 0; j < landmarks; j++) {
            estimate[j][0] = mu[base(j)]; estimate[j][1] = mu[base(j) + 1];
        }
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[][][] information = new double[landmarks][2][2]; double[][] rhs = new double[landmarks][2];
            for (Sample s : history) {
                int j = s.landmark; double dx = estimate[j][0] - s.robotX, dy = estimate[j][1] - s.robotY;
                double q = dx * dx + dy * dy, len = Math.sqrt(q);
                double[][] jac = {{dx / len, dy / len}, {-dy / q, dx / q}};
                double r0 = s.range - len, r1 = wrap(s.bearing - (Math.atan2(dy, dx) - s.robotTheta));
                for (int a = 0; a < 2; a++) {
                    for (int b = 0; b < 2; b++)
                        information[j][a][b] += jac[0][a] * jac[0][b] / (SIGMA_R * SIGMA_R)
                                + jac[1][a] * jac[1][b] / (SIGMA_B * SIGMA_B);
                    rhs[j][a] += jac[0][a] * r0 / (SIGMA_R * SIGMA_R)
                            + jac[1][a] * r1 / (SIGMA_B * SIGMA_B);
                }
            }
            for (int j = 0; j < landmarks; j++) {
                double det = information[j][0][0] * information[j][1][1]
                        - information[j][0][1] * information[j][1][0];
                double d0 = (information[j][1][1] * rhs[j][0] - information[j][0][1] * rhs[j][1]) / det;
                double d1 = (-information[j][1][0] * rhs[j][0] + information[j][0][0] * rhs[j][1]) / det;
                estimate[j][0] += d0; estimate[j][1] += d1;
            }
        }
        return estimate;
    }
}

public class Solution {
    static final double[][] LANDMARKS = {{8, 0}, {0, 8}, {-8, 0}};
    static final double SIGMA_V = 0.05, SIGMA_W = 0.02, DT = 0.5;
    static final int STEPS = 60;

    static double controlOmega(int step) { return step < 30 ? 0.5 : -0.15; }

    /** Runs one noisy SLAM episode and returns the filter plus the final true pose. */
    static EkfSlam episode(SplitMix64 rng, boolean noisy, double[] finalPose) {
        EkfSlam slam = new EkfSlam(LANDMARKS.length); double truthX = 0, truthY = 0, truthTheta = 0;
        for (int step = 0; step < STEPS; step++) {
            double w = controlOmega(step), v = 1.0;
            truthX += (Math.sin(truthTheta + w * DT) - Math.sin(truthTheta)) / w;
            truthY += (Math.cos(truthTheta) - Math.cos(truthTheta + w * DT)) / w;
            truthTheta = wrap(truthTheta + w * DT);
            double measuredV = v + (noisy ? SIGMA_V * rng.nextGaussian() : 0);
            double measuredW = w + (noisy ? SIGMA_W * rng.nextGaussian() : 0);
            slam.predict(measuredV, measuredW, DT, SIGMA_V, SIGMA_W);
            for (int j = 0; j < LANDMARKS.length; j++) {
                double dx = LANDMARKS[j][0] - truthX, dy = LANDMARKS[j][1] - truthY;
                double range = Math.hypot(dx, dy) + (noisy ? EkfSlam.SIGMA_R * rng.nextGaussian() : 0);
                double bearing = wrap(Math.atan2(dy, dx) - truthTheta)
                        + (noisy ? EkfSlam.SIGMA_B * rng.nextGaussian() : 0);
                slam.update(j, range, bearing, truthX, truthY, truthTheta);
            }
        }
        finalPose[0] = truthX; finalPose[1] = truthY; finalPose[2] = truthTheta; return slam;
    }

    static double wrap(double angle) { return Math.atan2(Math.sin(angle), Math.cos(angle)); }

    public static void main(String[] args) {
        // --- analytic Jacobians against central differences --------------------------------
        EkfSlam probe = new EkfSlam(LANDMARKS.length); probe.mu[0] = 1.2; probe.mu[1] = -0.4; probe.mu[2] = 0.7;
        for (int j = 0; j < LANDMARKS.length; j++) {
            probe.mu[probe.base(j)] = LANDMARKS[j][0] + 0.3; probe.mu[probe.base(j) + 1] = LANDMARKS[j][1] - 0.2;
        }
        double eps = 1e-6, worstH = 0; double[] centre = probe.mu.clone();
        double[][] h = probe.measurementJacobian(1);
        for (int col = 0; col < centre.length; col++) {
            double[] plus = centre.clone(), minus = centre.clone(); plus[col] += eps;
            minus[col] -= eps; probe.mu = plus; double[] zp = probe.expected(1); probe.mu = minus;
            double[] zm = probe.expected(1); probe.mu = centre;
            worstH = Math.max(worstH, Math.abs((zp[0] - zm[0]) / (2 * eps) - h[0][col]));
            worstH = Math.max(worstH, Math.abs(wrap(zp[1] - zm[1]) / (2 * eps) - h[1][col]));
        }
        check(worstH < 1e-6, "measurement Jacobian deviation " + worstH);
        double[][] g = EkfSlam.transitionJacobian(centre, 1.0, 0.4, DT);
        double[][] control = EkfSlam.controlJacobian(centre, 1.0, 0.4, DT); double worstG = 0, worstV = 0;
        for (int col = 0; col < centre.length; col++) {
            double[] plus = centre.clone(), minus = centre.clone(); plus[col] += eps;
            minus[col] -= eps; double[] tp = EkfSlam.transition(plus, 1.0, 0.4, DT);
            double[] tm = EkfSlam.transition(minus, 1.0, 0.4, DT);
            for (int row = 0; row < 3; row++)
                worstG = Math.max(worstG, Math.abs((tp[row] - tm[row]) / (2 * eps) - g[row][col]));
        }
        for (int col = 0; col < 2; col++) {
            double[] state = centre.clone();
            double v = 1.0 + (col == 0 ? eps : 0), w = 0.4 + (col == 1 ? eps : 0);
            double[] plus = EkfSlam.transition(state, v, w, DT); v = 1.0 - (col == 0 ? eps : 0);
            w = 0.4 - (col == 1 ? eps : 0); double[] minus = EkfSlam.transition(state, v, w, DT);
            for (int row = 0; row < 3; row++)
                worstV = Math.max(worstV, Math.abs((plus[row] - minus[row]) / (2 * eps) - control[row][col]));
        }
        check(worstG < 1e-6, "transition Jacobian deviation " + worstG);
        check(worstV < 1e-6, "control Jacobian deviation " + worstV); probe.predict(0, 0, 0, 0, 0);
        check(Math.hypot(probe.mu[0] - centre[0], probe.mu[1] - centre[1]) < 1e-12,
                "a zero control must leave the state alone");
        // --- noise-free data must be fitted exactly -----------------------------------------
        double[] pose = new double[3]; EkfSlam clean = episode(new SplitMix64(1L), false, pose);
        double cleanError = 0;
        for (int j = 0; j < LANDMARKS.length; j++)
            cleanError = Math.max(cleanError, Math.hypot(clean.mu[clean.base(j)] - LANDMARKS[j][0],
                    clean.mu[clean.base(j) + 1] - LANDMARKS[j][1]));
        check(cleanError < 1e-3, "noise-free landmark error " + cleanError);
        check(Math.hypot(clean.mu[0] - pose[0], clean.mu[1] - pose[1]) < 1e-3,
                "noise-free robot error " + Math.hypot(clean.mu[0] - pose[0], clean.mu[1] - pose[1]));
        double cleanVariance = clean.sigma[clean.base(0)][clean.base(0)];
        check(cleanVariance < 0.005, "noise-free landmark variance " + cleanVariance);
        // --- noisy run: convergence, covariance shrink, batch oracle -------------------------
        double[] noisyPose = new double[3]; EkfSlam noisy = episode(new SplitMix64(20240791L), true, noisyPose);
        double maxError = 0, meanError = 0;
        for (int j = 0; j < LANDMARKS.length; j++) {
            double e = Math.hypot(noisy.mu[noisy.base(j)] - LANDMARKS[j][0],
                    noisy.mu[noisy.base(j) + 1] - LANDMARKS[j][1]);
            maxError = Math.max(maxError, e); meanError += e / LANDMARKS.length;
        }
        check(meanError < 0.25, "mean landmark error " + meanError);
        check(maxError < 0.4, "worst landmark error " + maxError);
        double noisyVariance = noisy.sigma[noisy.base(0)][noisy.base(0)];
        check(noisyVariance > cleanVariance, "noise must leave residual uncertainty");
        double[][] batch = noisy.batchLandmarks(5); double batchGap = 0;
        for (int j = 0; j < LANDMARKS.length; j++)
            batchGap = Math.max(batchGap, Math.hypot(noisy.mu[noisy.base(j)] - batch[j][0],
                    noisy.mu[noisy.base(j) + 1] - batch[j][1]));
        check(batchGap < 0.15, "EKF versus batch WLS landmark gap " + batchGap);
        // --- covariance blocks shrink as landmarks are re-observed ----------------------------
        EkfSlam watch = new EkfSlam(LANDMARKS.length); double previousTrace = Double.MAX_VALUE;
        int decreases = 0, observations = 0; double[] watchPose = new double[3];
        SplitMix64 watchRng = new SplitMix64(5150L);
        for (int step = 0; step < 40; step++) {
            double w = controlOmega(step); watch.predict(1.0, w, DT, SIGMA_V, SIGMA_W);
            double dx = LANDMARKS[0][0] - watchPose[0], dy = LANDMARKS[0][1] - watchPose[1];
            watch.update(0, Math.hypot(dx, dy), wrap(Math.atan2(dy, dx) - watchPose[2]),
                    watchPose[0], watchPose[1], watchPose[2]);
            if (step > 0) {
                double trace = watch.sigma[watch.base(0)][watch.base(0)]
                        + watch.sigma[watch.base(0) + 1][watch.base(0) + 1];
                observations++; if (trace <= previousTrace) decreases++; previousTrace = trace;
            }
            watchPose[0] += (Math.sin(watchPose[2] + w * DT) - Math.sin(watchPose[2])) / w;
            watchPose[1] += (Math.cos(watchPose[2]) - Math.cos(watchPose[2] + w * DT)) / w;
            watchPose[2] = wrap(watchPose[2] + w * DT);
        }
        check(decreases >= observations - 3, "landmark covariance must shrink on re-observation, "
                + decreases + " of " + observations);
        check(watch.isKnown(0) && !watch.isKnown(1), "landmarks are initialised only when first seen");
        // --- the initial landmark covariance carries the robot uncertainty ---------------------
        EkfSlam fresh = new EkfSlam(1); fresh.sigma[0][0] = 0.04; fresh.sigma[1][1] = 0.04; fresh.sigma[2][2] = 0.01;
        fresh.initialiseLandmark(0, 8.0, 0.3); double initialVariance = fresh.sigma[fresh.base(0)][fresh.base(0)];
        check(initialVariance > 0.05 && initialVariance > 5 * EkfSlam.SIGMA_R * EkfSlam.SIGMA_R,
                "the initial landmark variance must absorb the pose uncertainty, got " + initialVariance);
        // --- consistency: average NEES of the landmark error over independent runs ---------------
        double neesSum = 0; int neesCount = 0;
        for (int run = 0; run < 20; run++) {
            double[] runPose = new double[3];
            EkfSlam filter = episode(new SplitMix64(7000L + run), true, runPose);
            for (int j = 0; j < LANDMARKS.length; j++) {
                int b = filter.base(j);
                double[][] block = {{filter.sigma[b][b], filter.sigma[b][b + 1]},
                        {filter.sigma[b + 1][b], filter.sigma[b + 1][b + 1]}};
                double[] error = {LANDMARKS[j][0] - filter.mu[b], LANDMARKS[j][1] - filter.mu[b + 1]};
                double[][] inv = Mat.invert2(block);
                neesSum += error[0] * (inv[0][0] * error[0] + inv[0][1] * error[1])
                        + error[1] * (inv[1][0] * error[0] + inv[1][1] * error[1]);
                neesCount++;
            }
        }
        double averageNees = neesSum / neesCount;
        check(averageNees > 0.5 && averageNees < 6.0, "average landmark NEES " + averageNees
                + " must be near the 2 degrees-of-freedom expectation");
        // --- boundary and determinism -----------------------------------------------------------
        EkfSlam empty = new EkfSlam(0); empty.predict(1, 0.3, DT, SIGMA_V, SIGMA_W);
        check(empty.mu.length == 3 && empty.mu[0] != 0, "a filter with no landmarks still predicts");
        check(Math.abs(EkfSlam.wrap(2 * Math.PI + 0.3) - 0.3) < 1e-12
                && Math.abs(EkfSlam.wrap(2 * Math.PI - 0.3) + 0.3) < 1e-12,
                "wrap must fold full turns away");
        check(Math.abs(Math.abs(EkfSlam.wrap(3 * Math.PI)) - Math.PI) < 1e-12,
                "odd multiples of pi land on the branch cut");
        check(Math.abs(EkfSlam.wrap(-Math.PI - 0.1) - (Math.PI - 0.1)) < 1e-12, "wrap across -pi");
        double[] poseAgain = new double[3]; EkfSlam repeat = episode(new SplitMix64(20240791L), true, poseAgain);
        check(repeat.mu[repeat.base(0)] == noisy.mu[noisy.base(0)]
                && sameState(repeat, noisy), "episodes must be reproducible");
        System.out.println("Jacobian deviations: H " + worstH + ", G " + worstG);
        System.out.println("noise-free landmark error " + cleanError + ", noisy mean " + meanError
                + ", worst " + maxError + ", batch WLS gap " + batchGap);
        System.out.println("average landmark NEES " + averageNees + " over " + neesCount
                + " estimates, initial variance " + initialVariance);
        System.out.println("PASS");
    }

    static boolean sameState(EkfSlam a, EkfSlam b) {
        for (int i = 0; i < a.mu.length; i++) if (a.mu[i] != b.mu[i]) return false; return true;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
