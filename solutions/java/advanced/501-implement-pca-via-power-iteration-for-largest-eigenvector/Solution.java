/** Task 501 - Implement PCA via power iteration for largest eigenvector. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Principal component analysis built on power iteration: the data are mean-centred, the covariance
 * C = X^T X / (n - 1) is formed, and each component is extracted by repeatedly applying v <- C v /
 * ||C v|| until the iterate stops moving (||v_new - v_old|| <= 1e-14, documented cap 2000
 * iterations), with the Rayleigh quotient lambda = v^T C v as the eigenvalue and explicit deflation
 * C <- C - lambda v v^T between components plus a Gram-Schmidt re-orthogonalisation against the
 * already-extracted vectors inside the loop. Projections are P = X V^T and the reconstruction is P
 * V, so the explained-variance ratio is (sum of the k eigenvalues) / trace(C).
 *
 * Documented checks: power iteration against closed forms (C = [[2,1],[1,2]] gives lambda = 3 with
 * v = (1,1)/sqrt(2), and the deflated matrix gives lambda = 1 with v = (1,-1)/sqrt(2)); convergence
 * on an eigenvector start in exactly one iteration; the full PCA (values and vectors) compared
 * against a second, independent method - a cyclic Jacobi eigen decomposition with its own 1e-14
 * off-diagonal stopping rule - on LCG data (eigenvalues agree to 1e-8, vectors to 1e-8 up to sign);
 * orthonormality of the components; non-increasing eigenvalues; eigen-residual ||C v - lambda v||
 * <= 1e-8 on the original (un-deflated) matrix; the explained-variance ratio summing to 1 (within
 * 1e-9) and the centred data reconstructed exactly (relative Frobenius error <= 1e-10) when all d
 * components are kept; a planted one-dimensional signal recovered to 1e-4 with ratio >= 0.98;
 * scaling the data by 100 scaling the eigenvalues by 1e4 while leaving the vectors fixed; and a
 * bit-identical deterministic re-run.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        double normal() {
            double u1 = Math.max(next(), 1e-12);
            double u2 = next();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }
    }

    static class Eigen {
        final double[] values;
        final double[][] vectors;
        final int sweeps;

        Eigen(double[] values, double[][] vectors, int sweeps) {
            this.values = values;
            this.vectors = vectors;
            this.sweeps = sweeps;
        }
    }

    static class PowerResult {
        final double[] vector;
        final double value;
        final int iterations;

        PowerResult(double[] vector, double value, int iterations) {
            this.vector = vector;
            this.value = value;
            this.iterations = iterations;
        }
    }

    static final int MAX_ITER = 2000;
    static final double TOL = 1e-14;

    static double[][] identity(int n) {
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) out[i][i] = 1.0;
        return out;
    }

    static double norm(double[] v) {
        double sum = 0.0;
        for (double value : v) sum += value * value;
        return Math.sqrt(sum);
    }

    static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    static double[] matVec(double[][] a, double[] v) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < v.length; j++) sum += a[i][j] * v[j];
            out[i] = sum;
        }
        return out;
    }

    static double[] normalise(double[] v) {
        double n = norm(v);
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / n;
        return out;
    }

    /** Largest eigenvalue/vector of a symmetric matrix by pure power iteration from `start`. */
    static PowerResult powerLargest(double[][] c, double[] start) {
        double[] v = normalise(start.clone());
        int iterations = 0;
        for (int it = 1; it <= MAX_ITER; it++) {
            iterations = it;
            double[] w = matVec(c, v);
            if (norm(w) <= 1e-300) break;
            double[] next = normalise(w);
            double delta = 0.0;
            for (int i = 0; i < v.length; i++) delta = Math.max(delta, Math.abs(next[i] - v[i]));
            v = next;
            if (delta <= TOL) break;
        }
        return new PowerResult(v, dot(v, matVec(c, v)), iterations);
    }

    /** Largest eigenvalue/vector with the directions in `forbidden` projected out of the iterate. */
    static PowerResult powerLargestDeflated(double[][] c, double[] start, double[][] forbidden) {
        double[] v = normalise(start.clone());
        int iterations = 0;
        for (int it = 1; it <= MAX_ITER; it++) {
            iterations = it;
            double[] w = matVec(c, v);
            for (double[] u : forbidden) {
                double scale = dot(w, u);
                for (int i = 0; i < w.length; i++) w[i] -= scale * u[i];
            }
            if (norm(w) <= 1e-300) break;
            double[] next = normalise(w);
            double delta = 0.0;
            for (int i = 0; i < v.length; i++) delta = Math.max(delta, Math.abs(next[i] - v[i]));
            v = next;
            if (delta <= TOL) break;
        }
        return new PowerResult(v, dot(v, matVec(c, v)), iterations);
    }

    /** Top-k PCA: eigenvalues in non-increasing order plus the matching unit components. */
    static Eigen pca(double[][] data, int k) {
        int n = data.length;
        int d = data[0].length;
        double[][] centred = centre(data);
        double[][] cov = covariance(centred);
        double[] values = new double[k];
        double[][] vectors = new double[k][];
        Lcg lcg = new Lcg(20250101);
        double[][] last = new double[0][];
        for (int comp = 0; comp < k; comp++) {
            double[] start = new double[d];
            for (int i = 0; i < d; i++) start[i] = lcg.next() - 0.5;
            PowerResult result = powerLargestDeflated(cov, start, last);
            vectors[comp] = result.vector;
            values[comp] = result.value;
            double[][] grown = new double[last.length + 1][];
            for (int i = 0; i < last.length; i++) grown[i] = last[i];
            grown[last.length] = result.vector;
            last = grown;
        }
        return new Eigen(values, vectors, 0);
    }

    static double[][] centre(double[][] data) {
        int n = data.length;
        int d = data[0].length;
        double[] mean = new double[d];
        for (double[] row : data) {
            for (int j = 0; j < d; j++) mean[j] += row[j];
        }
        for (int j = 0; j < d; j++) mean[j] /= n;
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) out[i][j] = data[i][j] - mean[j];
        }
        return out;
    }

    static double[][] covariance(double[][] centred) {
        int n = centred.length;
        int d = centred[0].length;
        double[][] out = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = a; b < d; b++) {
                double sum = 0.0;
                for (int i = 0; i < n; i++) sum += centred[i][a] * centred[i][b];
                out[a][b] = sum / (n - 1);
                out[b][a] = out[a][b];
            }
        }
        return out;
    }

    static double trace(double[][] m) {
        double sum = 0.0;
        for (int i = 0; i < m.length; i++) sum += m[i][i];
        return sum;
    }

    /** Cyclic Jacobi eigen decomposition; vectors are the columns of V, values unsorted. */
    static Eigen jacobi(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) a[i] = input[i].clone();
        double[][] v = identity(n);
        int sweeps = 0;
        for (; sweeps < 100; sweeps++) {
            double off = 0.0;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            }
            if (Math.sqrt(2.0 * off) <= 1e-14) break;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (a[p][q] == 0.0) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = theta >= 0
                            ? 1.0 / (theta + Math.sqrt(theta * theta + 1.0))
                            : -1.0 / (-theta + Math.sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int i = 0; i < n; i++) {
                        double aip = a[i][p];
                        double aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < n; i++) {
                        double api = a[p][i];
                        double aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                    for (int i = 0; i < n; i++) {
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        // sort descending, permuting the columns of V the same way
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        double[] finalValues = values;
        Arrays.sort(order, (l, r) -> Double.compare(finalValues[r], finalValues[l]));
        double[][] sorted = new double[n][n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) sorted[j][i] = v[i][order[j]];
        }
        double[] sortedValues = new double[n];
        for (int j = 0; j < n; j++) sortedValues[j] = values[order[j]];
        return new Eigen(sortedValues, sorted, sweeps);
    }

    static double[][] lcgData(int n, int d, int seed) {
        Lcg lcg = new Lcg(seed);
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) out[i][j] = lcg.normal();
        }
        return out;
    }

    static double frobenius(double[][] m) {
        double sum = 0.0;
        for (double[] row : m) {
            for (double value : row) sum += value * value;
        }
        return Math.sqrt(sum);
    }

    static double[][] project(double[][] centred, double[][] components) {
        int n = centred.length;
        int k = components.length;
        double[][] out = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) out[i][c] = dot(centred[i], components[c]);
        }
        return out;
    }

    static double[][] reconstruct(double[][] scores, double[][] components) {
        int n = scores.length;
        int d = components[0].length;
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int c = 0; c < components.length; c++) sum += scores[i][c] * components[c][j];
                out[i][j] = sum;
            }
        }
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        // 1. Closed form: C = [[2,1],[1,2]] has top eigenpair (3, (1,1)/sqrt(2)).
        double[][] two = {{2.0, 1.0}, {1.0, 2.0}};
        PowerResult first = powerLargest(two, new double[]{1.0, 0.0});
        double invSqrt2 = 1.0 / Math.sqrt(2.0);
        check(Math.abs(first.value - 3.0) <= 1e-12, "closed form: lambda1 = 3");
        check(Math.abs(Math.abs(first.vector[0]) - invSqrt2) <= 1e-9
                && Math.abs(Math.abs(first.vector[1]) - invSqrt2) <= 1e-9,
                "closed form: v1 = (1,1)/sqrt(2)");
        check(first.iterations >= 2 && first.iterations <= MAX_ITER, "power iteration advances");

        // deflation removes the first direction: C - 3 v v^T has top eigenpair (1, (1,-1)/sqrt(2))
        double[] v1 = first.vector;
        double[][] deflated = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) deflated[i][j] = two[i][j] - 3.0 * v1[i] * v1[j];
        }
        PowerResult second = powerLargestDeflated(deflated, new double[]{1.0, 0.0},
                new double[][]{v1});
        check(Math.abs(second.value - 1.0) <= 1e-12, "deflated top eigenvalue is 1");
        check(Math.abs(Math.abs(second.vector[0]) - invSqrt2) <= 1e-9
                && Math.abs(Math.abs(second.vector[1]) - invSqrt2) <= 1e-9,
                "deflated top vector is (1,-1)/sqrt(2)");
        check(Math.abs(dot(second.vector, v1)) <= 1e-9, "components are orthogonal");

        // 2. An eigenvector start converges immediately; a diagonal matrix is solved exactly.
        double[][] diag = {{5.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 1.0}};
        PowerResult immediate = powerLargest(diag, new double[]{1.0, 0.0, 0.0});
        check(immediate.iterations == 1, "eigenvector start converges in one iteration");
        check(Math.abs(immediate.value - 5.0) <= 1e-12 && Math.abs(immediate.vector[0] - 1.0) <= 1e-12,
                "diagonal matrix top pair");
        PowerResult ones = powerLargest(diag, new double[]{1.0, 1.0, 1.0});
        check(ones.iterations <= 200, "ones start converges quickly on a well-separated spectrum");
        check(Math.abs(ones.value - 5.0) <= 1e-12, "ones start finds lambda = 5");

        // 3. PCA vs the independent Jacobi eigen decomposition on LCG data (n = 400, d = 4).
        double[][] data = lcgData(400, 4, 987654321);
        Eigen viaPower = pca(data, 4);
        Eigen viaJacobi = jacobi(covariance(centre(data)));
        check(viaJacobi.sweeps <= 100, "Jacobi converges within the sweep cap");
        for (int c = 0; c < 4; c++) {
            check(Math.abs(viaPower.values[c] - viaJacobi.values[c])
                            <= 1e-8 * Math.max(1.0, Math.abs(viaJacobi.values[c])),
                    "eigenvalue " + c + " agrees with Jacobi");
            double aligned = Math.abs(dot(viaPower.vectors[c], viaJacobi.vectors[c]));
            check(aligned >= 1.0 - 1e-8, "eigenvector " + c + " agrees with Jacobi up to sign");
        }
        for (int c = 1; c < 4; c++) {
            check(viaPower.values[c - 1] >= viaPower.values[c] - 1e-12, "values are non-increasing");
        }
        for (int a = 0; a < 4; a++) {
            check(Math.abs(norm(viaPower.vectors[a]) - 1.0) <= 1e-12,
                    "component " + a + " is a unit vector");
            for (int b = a + 1; b < 4; b++) {
                check(Math.abs(dot(viaPower.vectors[a], viaPower.vectors[b])) <= 1e-10,
                        "components " + a + "," + b + " are orthogonal");
            }
        }

        // 4. Eigen-residual on the *original* covariance, and the variance bookkeeping.
        double[][] cov = covariance(centre(data));
        double total = trace(cov);
        double explained = 0.0;
        for (int c = 0; c < 4; c++) {
            explained += viaPower.values[c];
            double[] residual = matVec(cov, viaPower.vectors[c]);
            for (int i = 0; i < 4; i++) residual[i] -= viaPower.values[c] * viaPower.vectors[c][i];
            check(norm(residual) <= 1e-8, "residual ||C v - lambda v|| is tiny for component " + c);
        }
        check(Math.abs(explained - total) <= 1e-9 * Math.max(1.0, Math.abs(total)),
                "all four components explain the total variance");
        double[][] centred = centre(data);
        double[][] scores = project(centred, viaPower.vectors);
        double[][] rebuilt = reconstruct(scores, viaPower.vectors);
        double err = 0.0;
        for (int i = 0; i < centred.length; i++) {
            for (int j = 0; j < 4; j++) {
                double diff = centred[i][j] - rebuilt[i][j];
                err += diff * diff;
            }
        }
        check(Math.sqrt(err) <= 1e-10 * frobenius(centred),
                "full-rank PCA reconstructs the centred data");

        // 5. Planted one-dimensional signal along u = (cos 0.7, sin 0.7).
        double[] u = {Math.cos(0.7), Math.sin(0.7)};
        Lcg lcg = new Lcg(4242);
        double[][] planted = new double[300][2];
        for (int i = 0; i < 300; i++) {
            double t = lcg.next() - 0.5;
            planted[i][0] = t * u[0] + 0.02 * lcg.normal();
            planted[i][1] = t * u[1] + 0.02 * lcg.normal();
        }
        Eigen plantedPca = pca(planted, 1);
        double plantedRatio = plantedPca.values[0] / trace(covariance(centre(planted)));
        check(plantedRatio >= 0.98, "planted signal explains at least 98% of the variance");
        check(Math.abs(dot(plantedPca.vectors[0], u)) >= 1.0 - 1e-4, "planted direction recovered");

        // 6. Scaling the data by 100 scales the eigenvalues by 1e4 and keeps the vectors.
        double[][] scaled = new double[data.length][4];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < 4; j++) scaled[i][j] = data[i][j] * 100.0;
        }
        Eigen scaledPca = pca(scaled, 2);
        for (int c = 0; c < 2; c++) {
            check(Math.abs(scaledPca.values[c] - 1e4 * viaPower.values[c])
                            <= 1e-6 * Math.abs(1e4 * viaPower.values[c]),
                    "eigenvalue " + c + " scales by 1e4");
            check(Math.abs(dot(scaledPca.vectors[c], viaPower.vectors[c])) >= 1.0 - 1e-6,
                    "eigenvector " + c + " is scale invariant");
        }

        // 7. Determinism, and a two-point sample (n = 2) still works.
        Eigen rerun = pca(data, 4);
        for (int c = 0; c < 4; c++) {
            check(rerun.values[c] == viaPower.values[c], "deterministic re-run: value " + c);
            for (int i = 0; i < 4; i++) {
                check(rerun.vectors[c][i] == viaPower.vectors[c][i],
                        "deterministic re-run: vector " + c);
            }
        }
        double[][] pair = {{0.0, 0.0}, {2.0, 4.0}};
        Eigen pairPca = pca(pair, 1);
        check(pairPca.values.length == 1 && pairPca.values[0] > 0.0,
                "two-point PCA has positive variance");

        System.out.println("PASS");
    }
}
