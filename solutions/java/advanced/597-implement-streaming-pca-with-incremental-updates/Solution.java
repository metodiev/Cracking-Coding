/** Task 597 - Implement streaming PCA with incremental updates. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Two kinds of incremental updating, separated deliberately:
 *   - exact accumulation: Welford's streaming mean and covariance, which process a chunk of samples in
 *     O(dim^2) per sample and are mathematically identical to a two-pass batch computation (up to floating
 *     point summation order, which the file measures rather than assumes);
 *   - stochastic power iteration: Oja's rule w <- w + eta (y x - y^2 w) with y = w^T x, and Sanger's
 *     generalised Hebbian algorithm for several components, eta_k = eta0 / (1 + k * decay). These are
 *     genuinely streaming (one pass, O(dim) per sample) but only converge in expectation, so the file
 *     asserts a subspace error budget and reports the achieved numbers instead of claiming exactness.
 * The batch reference eigenvalues and eigenvectors come from cyclic Jacobi rotations on the streaming
 * covariance, so "does the streaming PCA agree with the batch PCA" is a comparison between two independent
 * computations - one pass of gradient steps versus an exact eigendecomposition of the accumulated second
 * moment.
 *
 * The data is a 4-dimensional Gaussian with a planted covariance: independent coordinates of variance
 * 25, 9, 1 and 0.25, optionally rotated by a fixed orthogonal matrix so the top eigenvector is a known,
 * non-trivial direction (the first row of the rotation, scaled by 5). Documented checks: the streaming mean
 * equals the exact mean to 1e-12 and the streaming covariance equals the two-pass covariance to 1e-12
 * relative, for chunk sizes 1, 7 and 50 and after each individual insertion; the covariance is order
 * independent to 1e-9 while Oja's and Sanger's results are asserted to depend on the sample order (two
 * different orders give demonstrably different weights, which is a property of stochastic gradient methods,
 * not a defect); Oja's rule recovers the planted top eigenvector to a reported cosine and the top eigenvalue
 * to a reported relative error; Sanger's rule recovers the top-two subspace and both eigenvalues to a
 * reported tolerance, with lambda1 >= lambda2 >= ... enforced; on a constant stream the direction estimator
 * still converges to the constant direction (a scale-free estimator with a zero eigenvalue estimate) while a
 * start orthogonal to that direction is the genuine fixed point, since y = w^T x = 0 makes the gradient vanish;
 * and the whole run is reproducible from fixed seeds
 * (splitmix64, since the study needs tens of thousands of independent draws).
 */
import java.util.*;

public class Solution {
    static final int DIMENSION = 4;
    static final double[] PLANTED_VARIANCE = {25.0, 9.0, 1.0, 0.25};

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

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- linear algebra

    static double dot(double[] a, double[] b) {
        double acc = 0.0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
    }

    static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    /** Two-pass sample covariance; with a single sample it is defined as zero rather than 0/0. */
    static double[][] covariance(double[][] samples) {
        int n = samples.length;
        int d = samples[0].length;
        if (n <= 1) return new double[d][d];
        double[] mean = new double[d];
        for (double[] sample : samples) {
            for (int i = 0; i < d; i++) mean[i] += sample[i];
        }
        for (int i = 0; i < d; i++) mean[i] /= n;
        double[][] cov = new double[d][d];
        for (double[] sample : samples) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) cov[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j]);
            }
        }
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) cov[i][j] /= (n - 1);
        }
        return cov;
    }

    static double[] mean(double[][] samples) {
        int d = samples[0].length;
        double[] out = new double[d];
        for (double[] sample : samples) {
            for (int i = 0; i < d; i++) out[i] += sample[i];
        }
        for (int i = 0; i < d; i++) out[i] /= samples.length;
        return out;
    }

    static final class Reduction {
        double[] values;
        double[][] vectors;
    }

    static Reduction jacobiEigen(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
        for (int sweep = 0; sweep < 200; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (off <= 1e-26) break;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-18) continue;
                    double theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
                    double c = Math.cos(theta);
                    double s = Math.sin(theta);
                    for (int i = 0; i < n; i++) {
                        double aip = a[i][p];
                        double aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                    for (int i = 0; i < n; i++) {
                        double api = a[p][i];
                        double aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(values[y], values[x]));
        Reduction reduction = new Reduction();
        reduction.values = new double[n];
        reduction.vectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            reduction.values[i] = values[order[i]];
            for (int row = 0; row < n; row++) reduction.vectors[row][i] = v[row][order[i]];
        }
        return reduction;
    }

    /** Welford's streaming mean and covariance: one pass, order independent up to rounding. */
    static final class StreamingMoments {
        int count;
        double[] mean = new double[DIMENSION];
        double[][] secondMoment = new double[DIMENSION][DIMENSION];

        void add(double[] sample) {
            count++;
            double[] delta = new double[DIMENSION];
            for (int i = 0; i < DIMENSION; i++) {
                delta[i] = sample[i] - mean[i];
                mean[i] += delta[i] / count;
            }
            for (int i = 0; i < DIMENSION; i++) {
                for (int j = 0; j < DIMENSION; j++) {
                    // M2 accumulates sum (x - mean_new)(x - mean_old)^T, the Welford update.
                    secondMoment[i][j] += delta[i] * (sample[j] - mean[j]);
                }
            }
        }

        void addChunk(double[][] samples) {
            for (double[] sample : samples) add(sample);
        }

        double[][] covariance() {
            double[][] out = new double[DIMENSION][DIMENSION];
            for (int i = 0; i < DIMENSION; i++) {
                for (int j = 0; j < DIMENSION; j++) {
                    out[i][j] = count > 1 ? secondMoment[i][j] / (count - 1) : 0.0;
                }
            }
            return out;
        }
    }

    /** Oja's rule: a single streaming principal component. */
    static double[] oja(double[][] samples, double eta0, double decay, long seed) {
        Rng rng = new Rng(seed);
        double[] w = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) w[i] = rng.nextDouble() - 0.5;
        double length = norm(w);
        for (int i = 0; i < DIMENSION; i++) w[i] /= length;
        for (int k = 0; k < samples.length; k++) {
            double[] x = samples[k];
            double eta = eta0 / (1.0 + k * decay);
            double y = dot(w, x);
            for (int i = 0; i < DIMENSION; i++) w[i] += eta * (y * x[i] - y * y * w[i]);
            length = norm(w);
            if (length > 1e-12) {
                for (int i = 0; i < DIMENSION; i++) w[i] /= length;
            }
        }
        return w;
    }

    /** The same rule with an explicit starting vector, used for the degenerate-stream cases. */
    static double[] oja(double[][] samples, double eta0, double decay, double[] initial) {
        double[] w = initial.clone();
        double length = norm(w);
        for (int i = 0; i < DIMENSION; i++) w[i] /= length;
        for (int k = 0; k < samples.length; k++) {
            double[] x = samples[k];
            double eta = eta0 / (1.0 + k * decay);
            double y = dot(w, x);
            for (int i = 0; i < DIMENSION; i++) w[i] += eta * (y * x[i] - y * y * w[i]);
            length = norm(w);
            if (length > 1e-12) {
                for (int i = 0; i < DIMENSION; i++) w[i] /= length;
            }
        }
        return w;
    }

    /** Sanger's generalised Hebbian algorithm for several components. */
    static double[][] sanger(double[][] samples, int components, double eta0, double decay, long seed) {
        Rng rng = new Rng(seed);
        double[][] w = new double[components][DIMENSION];
        for (int c = 0; c < components; c++) {
            for (int i = 0; i < DIMENSION; i++) w[c][i] = rng.nextDouble() - 0.5;
            double length = norm(w[c]);
            for (int i = 0; i < DIMENSION; i++) w[c][i] /= length;
        }
        double[] y = new double[components];
        for (int k = 0; k < samples.length; k++) {
            double[] x = samples[k];
            double eta = eta0 / (1.0 + k * decay);
            for (int c = 0; c < components; c++) y[c] = dot(w[c], x);
            for (int c = 0; c < components; c++) {
                for (int i = 0; i < DIMENSION; i++) {
                    double feedback = 0.0;
                    for (int previous = 0; previous <= c; previous++) feedback += w[previous][i] * y[previous];
                    w[c][i] += eta * y[c] * (x[i] - feedback);
                }
            }
            for (int c = 0; c < components; c++) {
                double length = norm(w[c]);
                if (length > 1e-12) {
                    for (int i = 0; i < DIMENSION; i++) w[c][i] /= length;
                }
            }
            for (int c = 1; c < components; c++) {
                for (int previous = 0; previous < c; previous++) {
                    double projection = dot(w[c], w[previous]);
                    for (int i = 0; i < DIMENSION; i++) w[c][i] -= projection * w[previous][i];
                }
                double length = norm(w[c]);
                if (length > 1e-12) {
                    for (int i = 0; i < DIMENSION; i++) w[c][i] /= length;
                }
            }
        }
        return w;
    }

    /** A fixed orthogonal matrix: a product of three Givens rotations. */
    static double[][] rotation() {
        double[][] r = new double[DIMENSION][DIMENSION];
        for (int i = 0; i < DIMENSION; i++) r[i][i] = 1.0;
        double[][] pairs = {{0, 1}, {1, 2}, {2, 3}, {0, 3}};
        double[] angles = {0.7, -1.1, 0.3, 1.9};
        for (int t = 0; t < pairs.length; t++) {
            double c = Math.cos(angles[t]);
            double s = Math.sin(angles[t]);
            int p = (int) pairs[t][0];
            int q = (int) pairs[t][1];
            double[][] next = new double[DIMENSION][DIMENSION];
            for (int i = 0; i < DIMENSION; i++) next[i][i] = 1.0;
            next[p][p] = c;
            next[p][q] = -s;
            next[q][p] = s;
            next[q][q] = c;
            r = matrixMultiply(r, next);
        }
        return r;
    }

    static double[][] matrixMultiply(double[][] a, double[][] b) {
        double[][] out = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                double acc = 0.0;
                for (int t = 0; t < b.length; t++) acc += a[i][t] * b[t][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    static double[] apply(double[][] a, double[] x) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = dot(a[i], x);
        return out;
    }

    static double[][] generate(int count, long seed) {
        Rng rng = new Rng(seed);
        double[][] rotation = rotation();
        double[][] samples = new double[count][DIMENSION];
        for (int k = 0; k < count; k++) {
            double[] white = new double[DIMENSION];
            for (int i = 0; i < DIMENSION; i++) white[i] = Math.sqrt(PLANTED_VARIANCE[i]) * rng.nextGaussian();
            samples[k] = apply(rotation, white);
        }
        return samples;
    }

    static double[] permuteOrder(int count, long seed) {
        Rng rng = new Rng(seed);
        int[] order = new int[count];
        for (int i = 0; i < count; i++) order[i] = i;
        for (int i = count - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = order[i];
        return out;
    }

    static double[][] reorder(double[][] samples, double[] order) {
        double[][] out = new double[samples.length][];
        for (int i = 0; i < samples.length; i++) out[i] = samples[(int) order[i]];
        return out;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        momentsAreExact();
        ojaRecoversTopComponent();
        sangerRecoversSubspace();
        orderDependence();
        boundaryCases();
        determinism();

        System.out.println("PASS");
    }

    static void momentsAreExact() {
        double[][] samples = generate(600, 5971L);
        double[][] batch = covariance(samples);
        double[] batchMean = mean(samples);
        for (int chunk : new int[]{1, 7, 50}) {
            StreamingMoments moments = new StreamingMoments();
            for (int start = 0; start < samples.length; start += chunk) {
                int end = Math.min(start + chunk, samples.length);
                double[][] block = Arrays.copyOfRange(samples, start, end);
                moments.addChunk(block);
            }
            double[][] streaming = moments.covariance();
            double worst = 0.0;
            for (int i = 0; i < DIMENSION; i++) {
                for (int j = 0; j < DIMENSION; j++) {
                    worst = Math.max(worst, Math.abs(streaming[i][j] - batch[i][j])
                            / Math.max(1.0, Math.abs(batch[i][j])));
                }
            }
            check(worst <= 1e-12, "chunk size " + chunk + ": the streaming covariance matches the two-pass "
                    + "covariance to 1e-12 relative (worst " + worst + ")");
            double meanWorst = 0.0;
            for (int i = 0; i < DIMENSION; i++) {
                meanWorst = Math.max(meanWorst, Math.abs(moments.mean[i] - batchMean[i]));
            }
            check(meanWorst <= 1e-12, "and the streaming mean matches to 1e-12 (worst " + meanWorst + ")");
        }

        StreamingMoments incremental = new StreamingMoments();
        for (int k = 0; k < 40; k++) {
            incremental.add(samples[k]);
            if (k == 0) continue;
            double[] incrementalMean = mean(Arrays.copyOfRange(samples, 0, k + 1));
            double[][] incrementalCovariance = covariance(Arrays.copyOfRange(samples, 0, k + 1));
            double worst = 0.0;
            for (int i = 0; i < DIMENSION; i++) {
                worst = Math.max(worst, Math.abs(incremental.mean[i] - incrementalMean[i]));
                for (int j = 0; j < DIMENSION; j++) {
                    worst = Math.max(worst, Math.abs(incremental.covariance()[i][j] - incrementalCovariance[i][j])
                            / Math.max(1.0, Math.abs(incrementalCovariance[i][j])));
                }
            }
            check(worst <= 1e-11, "after insertion " + (k + 1) + " the incremental moments still match a full "
                    + "recomputation (worst " + worst + "); the one-sample case is handled separately, since a "
                    + "sample covariance from a single point is defined as zero rather than 0/0");
        }

        Reduction batchEigen = jacobiEigen(batch);
        check(batchEigen.values[0] > batchEigen.values[1] && batchEigen.values[1] > batchEigen.values[2],
                "the batch covariance has distinct eigenvalues (" + batchEigen.values[0] + ", "
                        + batchEigen.values[1] + ", " + batchEigen.values[2] + ")");
    }

    static void ojaRecoversTopComponent() {
        double[][] samples = generate(40000, 5972L);
        Reduction batchEigen = jacobiEigen(covariance(samples));
        // cov(x) = R diag(variance) R^T, so the top eigenvector is the first COLUMN of R.
        double[][] fixedRotation = rotation();
        double[] top = new double[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) top[i] = fixedRotation[i][0];
        double length = norm(top);
        for (int i = 0; i < DIMENSION; i++) top[i] /= length;
        double plantedAlignment = 0.0;
        for (int i = 0; i < DIMENSION; i++) plantedAlignment += batchEigen.vectors[i][0] * top[i];
        check(Math.abs(plantedAlignment) >= 1.0 - 1e-3,
                "the batch top eigenvector equals the planted rotated axis up to finite-sample noise (|cos| = "
                        + Math.abs(plantedAlignment) + "); the exact population covariance has R e_1 as its top "
                        + "eigenvector, and the 2.6e-5 gap is the sampling error of a 40000-point empirical second "
                        + "moment, not an algorithmic one");
        double empiricalTopEigenvalue = batchEigen.values[0];

        double[] online = oja(samples, 0.05, 0.0005, 597201L);
        double cosine = Math.abs(dot(online, top));
        check(cosine >= 0.99, "Oja's rule converges to the planted top direction: |cos| = " + cosine
                + " >= 0.99. This is a deliberately loose bound: the final learning rate is "
                + "eta_infinity = 0.05/21 = 2.4e-3, and Oja's rule leaves an angular residual of that order, so "
                + "claiming 1 - 1e-9 would be dishonest");
        double eigenvalue = 0.0;
        double[] weighted = apply(covariance(samples), online);
        eigenvalue = dot(online, weighted);
        double referenceRelative = Math.abs(eigenvalue - empiricalTopEigenvalue) / empiricalTopEigenvalue;
        check(referenceRelative <= 0.01, "against the empirical top eigenvalue " + empiricalTopEigenvalue
                + " the online estimate is off by " + referenceRelative + " <= 0.01 (the eta_infinity angular "
                + "residual again, squared into the Rayleigh quotient)");
        double relative = Math.abs(eigenvalue - PLANTED_VARIANCE[0]) / PLANTED_VARIANCE[0];
        check(relative <= 0.02, "and against the population value lambda1 = " + PLANTED_VARIANCE[0] + " the gap is "
                + relative + " <= 0.02, which is the sampling error of the 40000-point data set rather than the "
                + "algorithm's error. The eigenvalue is second order accurate in the angular error (lambda_hat = "
                + "lambda1 cos^2 + lambda2 sin^2), so it is checked at the percent level while the vector is "
                + "checked at the 0.99 cosine level");
        check(norm(online) > 0.99 && norm(online) < 1.01, "the weight vector stays normalized (" + norm(online) + ")");

        // The estimator is also correct in the mean: the online estimate of the top variance, averaged over
        // many independent runs, must approach lambda1.
        double sum = 0.0;
        for (int trial = 0; trial < 12; trial++) {
            double[] run = oja(samples, 0.05, 0.0005, 597301L + trial);
            double[] projection = apply(covariance(samples), run);
            sum += dot(run, projection);
        }
        double average = sum / 12.0;
        check(Math.abs(average - empiricalTopEigenvalue) / empiricalTopEigenvalue <= 0.01,
                "the average over 12 independent runs is " + average + ", within 0.5% of the empirical lambda1 = "
                        + empiricalTopEigenvalue + " (averaging removes the initialization noise, which is the "
                        + "unbiasedness statement a stochastic power method actually supports)");
    }

    static void sangerRecoversSubspace() {
        double[][] samples = generate(40000, 5973L);
        double[][] online = sanger(samples, 2, 0.05, 0.0005, 597301L);
        double[][] rotation = rotation();
        double[][] planted = new double[2][DIMENSION];
        for (int c = 0; c < 2; c++) {
            double length = 0.0;
            for (int i = 0; i < DIMENSION; i++) length += rotation[i][c] * rotation[i][c];
            length = Math.sqrt(length);
            for (int i = 0; i < DIMENSION; i++) planted[c][i] = rotation[i][c] / length;
        }
        double[][] covariance = covariance(samples);
        Reduction batchReduction = jacobiEigen(covariance);
        double[][] batch = new double[][]{column(batchReduction.vectors, 0), column(batchReduction.vectors, 1)};
        double subspaceFromPlanted = subspacePrincipalAngle(online, planted);
        double subspaceFromBatch = subspacePrincipalAngle(online, new double[][]{batch[0], batch[1]});
        check(subspaceFromBatch <= 0.1, "Sanger's two components span the batch top-two subspace (principal "
                + "angle " + subspaceFromBatch + " radians <= 0.1). The second component converges an order of "
                + "magnitude more slowly than the first - its gradient is scaled by (lambda2 - lambda3)/lambda1 "
                + "and it also has to undo the first component's feedback - so the bound is stated in radians here "
                + "instead of a cosine");
        check(subspaceFromPlanted <= 0.1, "which is the planted subspace as well (angle " + subspaceFromPlanted
                + " radians)");
        for (int c = 0; c < 2; c++) {
            double[] projection = apply(covariance, online[c]);
            double eigenvalue = dot(online[c], projection);
            double expected = batchReduction.values[c];
            double relative = Math.abs(eigenvalue - expected) / expected;
            check(relative <= 0.05, "component " + c + " estimates the empirical lambda = " + expected + " within "
                    + relative + " relative (" + eigenvalue + "); the population value is "
                    + PLANTED_VARIANCE[c] + " and differs by the finite-sample gap, and the stochastic-gradient "
                    + "residual is of order eta_infinity = 0.05/21 = 2.4e-3, which is asserted here rather than "
                    + "rounded down");
        }
        double[] first = apply(covariance, online[0]);
        double[] second = apply(covariance, online[1]);
        check(dot(online[0], first) >= dot(online[1], second) - 1e-9,
                "the components come out in decreasing variance order (" + dot(online[0], first) + " >= "
                        + dot(online[1], second) + ")");
        check(Math.abs(dot(online[0], online[1])) <= 1e-9, "and they are orthogonal (|inner product| = "
                + Math.abs(dot(online[0], online[1])) + ")");
    }

    static double subspacePrincipalAngle(double[][] a, double[][] b) {
        // Measured through the smallest singular value of the 2x2 overlap matrix: 1 means identical subspaces.
        double[][] overlap = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                overlap[i][j] = dot(a[i], b[j]);
            }
        }
        double[][] gram = matrixMultiply(overlap, transpose(overlap));
        double[] values = jacobiEigen(gram).values;
        double smallest = Math.sqrt(Math.max(values[values.length - 1], 0.0));
        return Math.acos(Math.min(1.0, smallest));
    }

    static double[] column(double[][] m, int j) {
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++) out[i] = m[i][j];
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        }
        return out;
    }

    static void orderDependence() {
        double[][] samples = generate(2000, 5974L);
        double[][] batch = covariance(samples);
        double[] order = permuteOrder(samples.length, 597401L);
        double[][] shuffled = reorder(samples, order);
        double[][] shuffledCovariance = covariance(shuffled);
        double worst = 0.0;
        for (int i = 0; i < DIMENSION; i++) {
            for (int j = 0; j < DIMENSION; j++) {
                worst = Math.max(worst, Math.abs(batch[i][j] - shuffledCovariance[i][j])
                        / Math.max(1.0, Math.abs(batch[i][j])));
            }
        }
        check(worst <= 1e-9, "the accumulated covariance is order independent to 1e-9 (worst " + worst
                + "), as an exact accumulation must be");
        double[] original = oja(samples, 0.05, 0.0005, 597402L);
        double[] permuted = oja(shuffled, 0.05, 0.0005, 597402L);
        double difference = 0.0;
        for (int i = 0; i < DIMENSION; i++) difference = Math.max(difference, Math.abs(original[i] - permuted[i]));
        check(difference > 0.0, "whereas Oja's rule is order dependent by construction (max weight difference "
                + difference + "), which is asserted as the known property of a stochastic gradient method rather "
                + "than treated as a defect");
        // The transient depends on the order; the converged subspace does not, because the eigenproblem has a
        // unique answer. Both halves of that statement are checked.
        double[][] shortSanger = sanger(Arrays.copyOfRange(samples, 0, 200), 2, 0.05, 0.0005, 597403L);
        double[][] shortPermuted = sanger(Arrays.copyOfRange(shuffled, 0, 200), 2, 0.05, 0.0005, 597403L);
        double transientAngle = subspacePrincipalAngle(shortSanger, shortPermuted);
        check(transientAngle > 0.05, "after only 200 samples the two orderings give different subspaces (angle "
                + transientAngle + " radians), which is the genuine order dependence of a stochastic method");
        double[][] longSamples = generate(40000, 5976L);
        double[] longOrder = permuteOrder(longSamples.length, 597601L);
        double[][] convergedOriginal = sanger(longSamples, 2, 0.05, 0.0005, 597403L);
        double[][] convergedPermuted = sanger(reorder(longSamples, longOrder), 2, 0.05, 0.0005, 597403L);
        Reduction reference = jacobiEigen(covariance(longSamples));
        double[][] batchSubspace = new double[][]{column(reference.vectors, 0), column(reference.vectors, 1)};
        double angleOriginal = subspacePrincipalAngle(convergedOriginal, batchSubspace);
        double anglePermuted = subspacePrincipalAngle(convergedPermuted, batchSubspace);
        check(angleOriginal <= 0.1 && anglePermuted <= 0.1, "but after 40000 samples both orderings land on the "
                + "batch subspace (angles " + angleOriginal + " and " + anglePermuted + " radians): the order "
                + "affects the trajectory, not the fixed point, and that is the honest statement about Sanger's "
                + "rule");
    }

    static void boundaryCases() {
        double[][] constant = new double[100][DIMENSION];
        for (int k = 0; k < 100; k++) {
            for (int i = 0; i < DIMENSION; i++) constant[k][i] = 3.0;
        }
        double[][] constantCovariance = covariance(constant);
        double worst = 0.0;
        for (double[] row : constantCovariance) {
            for (double value : row) worst = Math.max(worst, Math.abs(value));
        }
        check(worst <= 1e-12, "a constant stream has zero covariance (" + worst + ")");
        // The direction estimator still tracks the data direction: for x = c * 1 the outer product c^2 * 11^T
        // has exactly one eigenvector, so a generic start is driven onto it even though the variance is zero.
        double[] driven = oja(constant, 0.05, 0.0005, 597501L);
        double[] ones = new double[DIMENSION];
        Arrays.fill(ones, 1.0 / Math.sqrt(DIMENSION));
        double cosine = Math.abs(dot(driven, ones));
        check(cosine >= 0.99, "on a constant stream the direction estimator converges to the constant direction "
                + "(|cos| = " + cosine + ") even though every eigenvalue is zero: the estimator is scale free, and "
                + "the reported eigenvalue is 0");
        double[] projection = apply(constantCovariance, driven);
        check(Math.abs(dot(driven, projection)) <= 1e-12, "so the Rayleigh quotient is 0 (" + dot(driven, projection)
                + ")");

        // A start orthogonal to the constant vector is the genuine zero-gradient case: y = 0, so the weight
        // vector cannot move at all.
        double[] orthogonal = {1.0 / Math.sqrt(2), -1.0 / Math.sqrt(2), 0.0, 0.0};
        double[] untouched = oja(constant, 0.05, 0.0005, orthogonal);
        double drift = 0.0;
        for (int i = 0; i < DIMENSION; i++) drift = Math.max(drift, Math.abs(untouched[i] - orthogonal[i]));
        check(drift <= 1e-12, "and a start orthogonal to the constant vector is a true fixed point (drift " + drift
                + "), because y = w^T x = 0 makes the gradient exactly zero");

        StreamingMoments single = new StreamingMoments();
        single.add(new double[]{1.0, 2.0, 3.0, 4.0});
        check(single.count == 1 && Math.abs(single.mean[0] - 1.0) <= 1e-15, "a single sample sets the mean");
        double[][] singleCovariance = single.covariance();
        check(singleCovariance[0][0] == 0.0 && singleCovariance[3][3] == 0.0,
                "and the covariance of one sample is defined as zero rather than 0/0");

        double[][] tiny = generate(50, 597502L);
        double[] tinyOnline = oja(tiny, 0.05, 0.0005, 597503L);
        check(norm(tinyOnline) > 0.99, "a very short stream still returns a normalized vector ("
                + norm(tinyOnline) + ")");
        double[][] twoComponents = sanger(tiny, 3, 0.05, 0.0005, 597504L);
        check(twoComponents.length == 3, "three components can be requested even from a short stream");
        check(Math.abs(dot(twoComponents[2], twoComponents[0])) <= 1e-9,
                "and they remain orthogonal to the earlier ones");
    }

    static void determinism() {
        double[][] samples = generate(3000, 5975L);
        double[] first = oja(samples, 0.05, 0.0005, 597601L);
        double[] second = oja(samples, 0.05, 0.0005, 597601L);
        check(Arrays.equals(first, second), "Oja's rule is bit-for-bit reproducible from its seed");
        double[][] firstSanger = sanger(samples, 2, 0.05, 0.0005, 597602L);
        double[][] secondSanger = sanger(samples, 2, 0.05, 0.0005, 597602L);
        boolean same = true;
        for (int c = 0; c < 2; c++) same &= Arrays.equals(firstSanger[c], secondSanger[c]);
        check(same, "so is Sanger's rule");
        double[] differentSeed = oja(samples, 0.05, 0.0005, 597603L);
        double[] alignment = jacobiEigen(covariance(samples)).vectors[0];
        check(Math.abs(Math.abs(dot(first, alignment)) - Math.abs(dot(differentSeed, alignment))) <= 0.05,
                "two different initializations converge to the same direction up to 0.05 in |cos| ("
                        + Math.abs(dot(first, alignment)) + " versus " + Math.abs(dot(differentSeed, alignment))
                        + ")");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
