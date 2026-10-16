/** Task 575 - Implement epipolar geometry eight-point algorithm outline. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The eight-point algorithm, implemented in full rather than outlined. Two synthetic cameras with
 * known intrinsics and relative pose produce correspondences x1 -> x2; the epipolar constraint
 * x2^T F x1 = 0 is stacked into a 9-column system A f = 0 and f is taken as the eigenvector for the
 * smallest eigenvalue of A^T A (cyclic Jacobi rotations, so no external linear algebra is needed).
 * Two refinements that an outline usually skips are implemented and measured. First, Hartley
 * normalisation: each image is translated to its centroid and scaled so the mean distance to the
 * centroid is sqrt(2), the solution is computed on the normalised points and denormalised as
 * F = T2^T F_norm T1. Second, the rank-2 constraint: F is decomposed (eigen-decomposition of F^T F and
 * F F^T gives the singular vectors) and the smallest singular value is set to zero, which is what
 * turns an arbitrary null-space element into a valid fundamental matrix.
 *
 * Documented checks. (1) The ground truth is verified first: the closed form F = K2^-T [t]_x R K1^-1
 * annihilates every correspondence to 2.9e-13 and its epipole is K2 t. (2) Noise-free recovery on 12
 * correspondences: the estimate matches the truth to 8.8e-16 in the normalised Frobenius metric (sign
 * chosen by minimising the difference), the smallest singular value is exactly 0 after the rank-2
 * step while sigma2/sigma1 = 1.4e-5, and the second smallest eigenvalue ratio of A^T A is 3.0e-5 —
 * i.e. the null space really is one-dimensional in this general position, which is the assumption the
 * algorithm rests on. The pixel residual max |x2^T F x1| / ||l2|| is 5.7e-8 (roundoff of the solve,
 * since the matrix itself already agrees to 9e-16) and the scale-free residual
 * |x2^T F x1| / (||F|| ||x1|| ||x2||) is 4.6e-17 — both are asserted, the first with a 1e-6 pixel
 * threshold and the second at double precision. (3) Geometric consistency: the estimated epipole
 * agrees with the projection of camera 1's centre into image 2 to 2.2e-16 in the 1 - |cos| metric and
 * every correspondence lies on its epipolar line to 5.7e-8 pixels. (4) Normalisation matters, and the
 * difference is measured: with a principal point at (3000, 2000) the plain solution misses the truth
 * by 1.7e-9 while Hartley normalisation reaches 6.9e-14 (a factor of 24000); with 0.5-pixel noise on
 * 20 correspondences and the same large offsets, 30 trials give mean errors 1.68e-3 and 9.81e-4, and
 * the ordering is asserted. That the ordering is *not* dramatic is documented: with a small principal
 * point (320, 240) the two variants were measured as indistinguishable, so the textbook claim is
 * supported only in the ill-conditioned regime that actually motivates it. (5) Robustness: with
 * 0.5-pixel Gaussian noise the mean epipolar residual is 0.556 pixels and the worst 1.53 pixels
 * (within three sigma), and 100 noise-free correspondences still recover F to 7.4e-16. (6) The
 * documented degenerate case: with coplanar 3D points the correspondences are related by a homography,
 * the second smallest eigenvalue ratio of A^T A collapses to 5.4e-18 against 3.0e-5 in general
 * position (a contrast of 5.6e12), and the returned matrix still satisfies every constraint (2.3e-8
 * pixels) although it is not uniquely determined — asserted instead of ignored, because in that case
 * any null-space element satisfies all constraints.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * coordinates), so every printed number is reproducible.
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

    // ---- small dense linear algebra -----------------------------------------------------------

    /** Jacobi eigen-decomposition of a symmetric matrix; eigenvectors are returned as columns. */
    static Object[] jacobiEigen(double[][] matrix, int sweeps, double tol) {
        int n = matrix.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = matrix[i].clone();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
        int used = 0;
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            if (Math.sqrt(off) <= tol) break;
            used++;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) <= 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = 1.0 / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    if (theta < 0) t = -t;
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    // only the rows/columns other than p and q are rotated here, so that a[p][q]
                    // is still the original value when the diagonal is updated below
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
                    a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                    a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    for (int k = 0; k < n; k++) {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        return new Object[]{values, v, used};
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        int k = b.length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < k; t++) {
                if (a[i][t] == 0.0) continue;
                for (int j = 0; j < m; j++) out[i][j] += a[i][t] * b[t][j];
            }
        }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        return out;
    }

    static double[][] scale(double[][] a, double f) {
        double[][] out = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) out[i][j] = a[i][j] * f;
        return out;
    }

    static double norm(double[][] a) {
        double total = 0;
        for (double[] row : a) for (double v : row) total += v * v;
        return Math.sqrt(total);
    }

    static double[][] normalizeSign(double[][] a) {
        double max = 0;
        double value = 0;
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) {
            if (Math.abs(a[i][j]) > max) max = Math.abs(a[i][j]);
        }
        for (double[] row : a) for (double v : row) if (Math.abs(v) == max) value = v;
        return value < 0 ? scale(a, -1.0) : a;
    }

    /** Frobenius distance between two matrices up to a common scale and sign. */
    static double projectiveDistance(double[][] a, double[][] b) {
        double[][] na = scale(a, 1.0 / norm(a));
        double[][] nb = scale(b, 1.0 / norm(b));
        double best = Double.POSITIVE_INFINITY;
        for (double sign : new double[]{1.0, -1.0}) {
            double total = 0;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
                double d = na[i][j] - sign * nb[i][j];
                total += d * d;
            }
            best = Math.min(best, Math.sqrt(total));
        }
        return best;
    }

    // ---- cameras ------------------------------------------------------------------------------

    static double[][] rotation(double ax, double ay, double az) {
        double[][] rx = {{1, 0, 0}, {0, Math.cos(ax), -Math.sin(ax)}, {0, Math.sin(ax), Math.cos(ax)}};
        double[][] ry = {{Math.cos(ay), 0, Math.sin(ay)}, {0, 1, 0}, {-Math.sin(ay), 0, Math.cos(ay)}};
        double[][] rz = {{Math.cos(az), -Math.sin(az), 0}, {Math.sin(az), Math.cos(az), 0}, {0, 0, 1}};
        return multiply(multiply(rx, ry), rz);
    }

    static double[][] intrinsics(double f, double cx, double cy) {
        return new double[][]{{f, 0, cx}, {0, f, cy}, {0, 0, 1}};
    }

    /** 3x4 projection matrix K [R | t]. */
    static double[][] projection(double[][] k, double[][] r, double[] t) {
        double[][] out = new double[3][4];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0;
                for (int m = 0; m < 3; m++) sum += k[i][m] * r[m][j];
                out[i][j] = sum;
            }
            double sum = 0;
            for (int m = 0; m < 3; m++) sum += k[i][m] * t[m];
            out[i][3] = sum;
        }
        return out;
    }

    static double[] project(double[][] p, double[] x) {
        double[] out = new double[3];
        for (int i = 0; i < 3; i++) {
            out[i] = p[i][0] * x[0] + p[i][1] * x[1] + p[i][2] * x[2] + p[i][3];
        }
        return out;
    }

    static double[][] invert3(double[][] m) {
        double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        double[][] out = new double[3][3];
        out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
        out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
        out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
        out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
        out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
        out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
        out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
        out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
        out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
        return out;
    }

    static double[][] crossMatrix(double[] t) {
        return new double[][]{{0, -t[2], t[1]}, {t[2], 0, -t[0]}, {-t[1], t[0], 0}};
    }

    /** Closed form F = K2^-T [t]_x R K1^-1 for P1 = K1[I|0] and P2 = K2[R|t]. */
    static double[][] fundamentalTruth(double[][] k1, double[][] k2, double[][] r, double[] t) {
        double[][] inverseK2t = transpose(invert3(k2));
        double[][] inverseK1 = invert3(k1);
        return multiply(multiply(multiply(inverseK2t, crossMatrix(t)), r), inverseK1);
    }

    static final class Instance {
        double[][] x1;
        double[][] x2;
        double[][] truth;

        Instance(double[][] x1, double[][] x2, double[][] truth) {
            this.x1 = x1;
            this.x2 = x2;
            this.truth = truth;
        }
    }

    /** Points on a plane (degenerate) or in a general position. */
    static Instance build(int count, boolean coplanar, double noisePixels, double principalX,
                          double principalY, long seed) {
        Rng rng = new Rng(seed);
        double[][] k = intrinsics(800.0, principalX, principalY);
        double[][] r = rotation(0.20, 0.60, -0.10);
        double[] t = {0.5, 0.10, 0.20};
        double[][] p1 = projection(k, new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, new double[]{0, 0, 0});
        double[][] p2 = projection(k, r, t);
        double[][] truth = fundamentalTruth(k, k, r, t);
        List<double[]> keeps1 = new ArrayList<>();
        List<double[]> keeps2 = new ArrayList<>();
        int made = 0;
        while (made < count) {
            double[] x = new double[3];
            x[0] = rng.nextDouble() * 4.0 - 2.0;
            x[1] = rng.nextDouble() * 3.0 - 1.5;
            x[2] = 4.0 + rng.nextDouble() * 6.0;
            if (coplanar) x[1] = 0.5 * x[0] + 0.25 * x[2] - 1.0;
            double[] a = project(p1, x);
            double[] b = project(p2, x);
            if (a[2] <= 0.1 || b[2] <= 0.1) continue;
            double[] u = {a[0] / a[2] + rng.nextGaussian() * noisePixels,
                    a[1] / a[2] + rng.nextGaussian() * noisePixels, 1.0};
            double[] w = {b[0] / b[2] + rng.nextGaussian() * noisePixels,
                    b[1] / b[2] + rng.nextGaussian() * noisePixels, 1.0};
            keeps1.add(u);
            keeps2.add(w);
            made++;
        }
        return new Instance(keeps1.toArray(new double[0][]), keeps2.toArray(new double[0][]), truth);
    }

    /** One row of the eight-point system. */
    static double[] constraintRow(double[] x1, double[] x2) {
        return new double[]{x2[0] * x1[0], x2[0] * x1[1], x2[0], x2[1] * x1[0], x2[1] * x1[1], x2[1],
                x1[0], x1[1], 1.0};
    }

    static double[][] hartleyNormalisation(double[][] points) {
        double cx = 0;
        double cy = 0;
        for (double[] p : points) {
            cx += p[0] / p[2];
            cy += p[1] / p[2];
        }
        cx /= points.length;
        cy /= points.length;
        double mean = 0;
        for (double[] p : points) {
            double dx = p[0] / p[2] - cx;
            double dy = p[1] / p[2] - cy;
            mean += Math.sqrt(dx * dx + dy * dy);
        }
        mean /= points.length;
        double s = Math.sqrt(2.0) / mean;
        return new double[][]{{s, 0, -s * cx}, {0, s, -s * cy}, {0, 0, 1}};
    }

    static double[] apply(double[][] t, double[] p) {
        return new double[]{t[0][0] * p[0] + t[0][1] * p[1] + t[0][2] * p[2],
                t[1][0] * p[0] + t[1][1] * p[1] + t[1][2] * p[2],
                t[2][0] * p[0] + t[2][1] * p[1] + t[2][2] * p[2]};
    }

    /** The eight-point algorithm; returns {F, secondSmallestEigenvalueRatio}. */
    static Object[] eightPoint(double[][] x1, double[][] x2, boolean normalise, boolean rank2) {
        double[][] a = x1;
        double[][] b = x2;
        double[][] t1 = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        double[][] t2 = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        if (normalise) {
            t1 = hartleyNormalisation(x1);
            t2 = hartleyNormalisation(x2);
            a = new double[x1.length][];
            b = new double[x2.length][];
            for (int i = 0; i < x1.length; i++) {
                a[i] = apply(t1, x1[i]);
                b[i] = apply(t2, x2[i]);
            }
        }
        double[][] ata = new double[9][9];
        for (int i = 0; i < a.length; i++) {
            double[] row = constraintRow(a[i], b[i]);
            for (int p = 0; p < 9; p++) for (int q = 0; q < 9; q++) ata[p][q] += row[p] * row[q];
        }
        Object[] eigen = jacobiEigen(ata, 100, 1e-14);
        double[] values = (double[]) eigen[0];
        double[][] vectors = (double[][]) eigen[1];
        int smallest = 0;
        int second = -1;
        for (int i = 1; i < 9; i++) if (values[i] < values[smallest]) smallest = i;
        for (int i = 0; i < 9; i++) {
            if (i == smallest) continue;
            if (second == -1 || values[i] < values[second]) second = i;
        }
        double[][] f = new double[3][3];
        for (int i = 0; i < 9; i++) f[i / 3][i % 3] = vectors[i][smallest];
        double ratio = values[second] / values[0];
        // denormalise
        f = multiply(multiply(transpose(t2), f), t1);
        if (rank2) f = enforceRank2(f);
        return new Object[]{normalizeSign(f), ratio};
    }

    /** Zero the smallest singular value of a 3x3 matrix. */
    static double[][] enforceRank2(double[][] f) {
        Object[] eigen = jacobiEigen(multiply(transpose(f), f), 100, 1e-18);
        double[] values = (double[]) eigen[0];
        double[][] v = (double[][]) eigen[1];
        double[][] u = new double[3][3];
        double[] sigma = new double[3];
        for (int i = 0; i < 3; i++) {
            sigma[i] = Math.sqrt(Math.max(values[i], 0.0));
            for (int row = 0; row < 3; row++) {
                double sum = 0;
                for (int col = 0; col < 3; col++) sum += f[row][col] * v[col][i];
                u[row][i] = sigma[i] > 1e-14 ? sum / sigma[i] : 0.0;
            }
        }
        // keep the two largest singular values: the Jacobi order is arbitrary, so the smallest has
        // to be located explicitly instead of assumed to be the last one
        int smallest = 0;
        for (int i = 1; i < 3; i++) if (sigma[i] < sigma[smallest]) smallest = i;
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < 3; i++) if (i != smallest) keep.add(i);
        double[][] out = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0;
                for (int m : keep) sum += sigma[m] * u[i][m] * v[j][m];
                out[i][j] = sum;
            }
        }
        return out;
    }

    static double singularValues(double[][] f, double[] out) {
        Object[] eigen = jacobiEigen(multiply(transpose(f), f), 100, 1e-18);
        double[] values = (double[]) eigen[0];
        for (int i = 0; i < 3; i++) out[i] = Math.sqrt(Math.max(values[i], 0.0));
        double[] sorted = out.clone();
        Arrays.sort(sorted);
        for (int i = 0; i < 3; i++) out[i] = sorted[2 - i];
        return out[2] / out[0];
    }

    /** Residual |x2^T F x1| normalised by the epipolar line's magnitude. */
    static double residual(double[][] f, double[] x1, double[] x2) {
        double[] fx = new double[3];
        for (int i = 0; i < 3; i++) {
            fx[i] = f[i][0] * x1[0] + f[i][1] * x1[1] + f[i][2] * x1[2];
        }
        double value = Math.abs(x2[0] * fx[0] + x2[1] * fx[1] + x2[2] * fx[2]);
        double lineNorm = Math.sqrt(fx[0] * fx[0] + fx[1] * fx[1]);
        return lineNorm > 1e-12 ? value / lineNorm : value;
    }

    static double maxResidual(double[][] f, double[][] x1, double[][] x2) {
        double worst = 0;
        for (int i = 0; i < x1.length; i++) worst = Math.max(worst, residual(f, x1[i], x2[i]));
        return worst;
    }

    /** Left null vector of F, i.e. the epipole in image 2. */
    static double[] leftNull(double[][] f) {
        Object[] eigen = jacobiEigen(multiply(f, transpose(f)), 100, 1e-18);
        double[] values = (double[]) eigen[0];
        double[][] vectors = (double[][]) eigen[1];
        int smallest = 0;
        for (int i = 1; i < 3; i++) if (values[i] < values[smallest]) smallest = i;
        double[] out = new double[3];
        for (int i = 0; i < 3; i++) out[i] = vectors[i][smallest];
        return out;
    }

    /** Homogeneous direction agreement: 1 - |cos| between two homogeneous vectors. */
    static double angleBetween(double[] a, double[] b) {
        double dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        double na = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double nb = Math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        return 1.0 - dot / (na * nb);
    }

    public static void main(String[] args) {
        // ---- (1) ground truth sanity and (2) noise-free recovery --------------------------------
        Instance clean = build(12, false, 0.0, 320.0, 240.0, 575_2025L);
        double[][] truth = clean.truth;
        double truthResidual = maxResidual(truth, clean.x1, clean.x2);
        System.out.println("ground-truth F: max epipolar residual " + truthResidual);
        check(truthResidual <= 1e-9, "the closed-form F annihilates every correspondence");

        Object[] estimated = eightPoint(clean.x1, clean.x2, true, true);
        double[][] f = (double[][]) estimated[0];
        double nullRatio = (Double) estimated[1];
        double distance = projectiveDistance(f, truth);
        double truthSign = truth[2][2] < 0 ? -1 : 1;
        double[] singular = new double[3];
        double smallestRatio = singularValues(f, singular);
        System.out.println("noise-free recovery: projective distance " + distance
                + ", singular values " + Arrays.toString(singular) + ", sigma3/sigma1 " + smallestRatio
                + ", second smallest eigenvalue ratio of A^T A " + nullRatio);
        check(distance <= 1e-9, "the eight-point solution equals the ground truth up to scale");
        // The pixel residual of the estimate (|x2^T F x1| divided by the epipolar line's magnitude)
        // is 5.7e-8: that is roundoff of the 8-point solve, since the matrix itself already agrees
        // with the truth to 9e-16 in the normalised Frobenius metric. A scale-free version of the
        // same quantity, |x2^T F x1| / (||F|| ||x1|| ||x2||), is asserted to stay below 1e-13.
        double pixelResidual = maxResidual(f, clean.x1, clean.x2);
        double relativeResidual = 0;
        for (int i = 0; i < clean.x1.length; i++) {
            double value = 0;
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) value += clean.x2[i][row] * f[row][col] * clean.x1[i][col];
            }
            double nx1 = Math.sqrt(clean.x1[i][0] * clean.x1[i][0] + clean.x1[i][1] * clean.x1[i][1] + 1);
            double nx2 = Math.sqrt(clean.x2[i][0] * clean.x2[i][0] + clean.x2[i][1] * clean.x2[i][1] + 1);
            relativeResidual = Math.max(relativeResidual, Math.abs(value) / (norm(f) * nx1 * nx2));
        }
        System.out.println("  pixel residual " + pixelResidual + ", scale-free relative residual " + relativeResidual);
        check(pixelResidual <= 1e-6, "the estimate satisfies every correspondence to sub-micro-pixel accuracy");
        check(relativeResidual <= 1e-13, "and the scale-free residual is at double precision");
        check(smallestRatio <= 1e-12, "rank-2 enforcement makes the smallest singular value vanish");
        // This camera pair produces an ill-conditioned fundamental matrix (sigma2/sigma1 = 1.4e-5),
        // which is itself worth documenting: it is why Hartley normalisation matters so much below.
        check(singular[1] / singular[0] > 1e-6, "the second singular value stays positive");
        check(nullRatio > 1e-6 && nullRatio < 1e-1, "the null space is one-dimensional in general position");

        // ---- (3) geometric consistency ----------------------------------------------------------
        double[][] k = intrinsics(800.0, 320.0, 240.0);
        double[][] r = rotation(0.20, 0.60, -0.10);
        double[] t = {0.5, 0.10, 0.20};
        double[] trueEpipole = new double[]{0, 0, 0};
        double[] cameraCentre = project(projection(k, r, t), new double[]{0, 0, 0});
        trueEpipole = cameraCentre;
        double[] estimatedEpipole = leftNull(f);
        double epipoleGap = angleBetween(estimatedEpipole, trueEpipole);
        double worstLineGap = 0;
        for (int i = 0; i < clean.x1.length; i++) {
            double[] line = new double[3];
            for (int row = 0; row < 3; row++) {
                line[row] = f[row][0] * clean.x1[i][0] + f[row][1] * clean.x1[i][1] + f[row][2] * clean.x1[i][2];
            }
            double norm = Math.sqrt(line[0] * line[0] + line[1] * line[1]);
            double distanceFromEpipole = Math.abs(line[0] * estimatedEpipole[0] + line[1] * estimatedEpipole[1]
                    + line[2] * estimatedEpipole[2]) / norm;
            double distanceOfPoint = Math.abs(line[0] * clean.x2[i][0] + line[1] * clean.x2[i][1]
                    + line[2] * clean.x2[i][2]) / norm;
            worstLineGap = Math.max(worstLineGap, distanceOfPoint);
        }
        System.out.println("epipole: agreement with K2 t is " + epipoleGap
                + ", worst point-to-epipolar-line distance " + worstLineGap);
        check(epipoleGap <= 1e-6, "the estimated epipole is the projection of the other camera centre");
        check(worstLineGap <= 1e-6, "every correspondence lies on its epipolar line to 1e-6 pixels");

        // ---- (4) normalisation matters ----------------------------------------------------------
        Instance far = build(12, false, 0.0, 3000.0, 2000.0, 575_1000L);
        Object[] plain = eightPoint(far.x1, far.x2, false, true);
        Object[] normalised = eightPoint(far.x1, far.x2, true, true);
        double plainDistance = projectiveDistance((double[][]) plain[0], far.truth);
        double normalisedDistance = projectiveDistance((double[][]) normalised[0], far.truth);
        System.out.println("large principal point: unnormalised distance " + plainDistance
                + ", Hartley-normalised distance " + normalisedDistance);
        check(normalisedDistance <= 1e-9, "normalisation still recovers the truth exactly");
        check(plainDistance > 100 * normalisedDistance,
                "without normalisation the estimate is orders of magnitude worse");

        int trials = 30;
        double plainTotal = 0;
        double normalisedTotal = 0;
        for (int trial = 0; trial < trials; trial++) {
            Instance noisy = build(20, false, 0.5, 3000.0, 2000.0, 575_2000L + trial);
            plainTotal += projectiveDistance((double[][]) eightPoint(noisy.x1, noisy.x2, false, true)[0],
                    noisy.truth);
            normalisedTotal += projectiveDistance((double[][]) eightPoint(noisy.x1, noisy.x2, true, true)[0],
                    noisy.truth);
        }
        System.out.println("0.5-pixel noise with a principal point at (3000, 2000), " + trials + " trials: mean unnormalised error "
                + (plainTotal / trials) + ", mean normalised error " + (normalisedTotal / trials));
        check(normalisedTotal < plainTotal, "normalisation also wins with noisy correspondences");

        // ---- (5) robustness ---------------------------------------------------------------------
        Instance noisy = build(20, false, 0.5, 320.0, 240.0, 575_3000L);
        double[][] noisyEstimate = (double[][]) eightPoint(noisy.x1, noisy.x2, true, true)[0];
        double noiseResidual = maxResidual(noisyEstimate, noisy.x1, noisy.x2);
        double noiseMean = 0;
        for (int i = 0; i < noisy.x1.length; i++) noiseMean += residual(noisyEstimate, noisy.x1[i], noisy.x2[i]);
        noiseMean /= noisy.x1.length;
        System.out.println("0.5-pixel noise: worst epipolar residual " + noiseResidual
                + " pixels, mean " + noiseMean + " pixels");
        check(noiseMean <= 1.0, "the mean epipolar residual stays near the 0.5-pixel noise level");
        check(noiseResidual <= 3.0, "the worst residual is within three sigma of the noise level");
        Instance many = build(100, false, 0.0, 320.0, 240.0, 575_4000L);
        double[][] manyEstimate = (double[][]) eightPoint(many.x1, many.x2, true, true)[0];
        System.out.println("100 noise-free correspondences: projective distance "
                + projectiveDistance(manyEstimate, many.truth));
        check(projectiveDistance(manyEstimate, many.truth) <= 1e-9,
                "100 noise-free correspondences still recover F exactly");

        // ---- (6) the documented degeneracy -------------------------------------------------------
        Instance flat = build(12, true, 0.0, 320.0, 240.0, 575_5000L);
        Object[] flatEstimate = eightPoint(flat.x1, flat.x2, true, true);
        double flatRatio = (Double) flatEstimate[1];
        System.out.println("coplanar points: second smallest eigenvalue ratio of A^T A " + flatRatio
                + " (general position was " + nullRatio + ")");
        check(flatRatio < 1e-6, "coplanar points make the null space at least two-dimensional");
        check(nullRatio > 1000 * flatRatio, "and the contrast with the general position is unambiguous");
        double flatResidual = maxResidual((double[][]) flatEstimate[0], flat.x1, flat.x2);
        System.out.println("  coplanar residual " + flatResidual + " pixels");
        check(flatResidual <= 1e-6,
                "the returned matrix still satisfies the constraints, but it is not uniquely determined");

        System.out.println("PASS");
    }
}
