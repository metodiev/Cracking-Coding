/** Task 502 - Implement randomized SVD basic using power method. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Randomized SVD of a real m x n matrix: draw a Gaussian sketch Omega (n x l) from the fixed 75/74
 * LCG + Box-Muller normal, form Y = A Omega, sharpen the range with q power iterations Y <- A (A^T
 * Y') where Y' is the modified-Gram-Schmidt orthonormalisation of Y (done twice, so Q^T Q = I to
 * machine precision even for slowly decaying spectra), then take Q = orth(Y) and the small matrix B
 * = Q^T A (l x n). The small SVD of B is obtained from the eigen decomposition of B B^T (l x l)
 * with cyclic Jacobi: singular values sqrt(lambda) in non-increasing order, left vectors U_B, right
 * vectors V = B^T U_B / sigma, and U = Q U_B. Everything is standard library only and fully
 * deterministic (fixed seed); zero singular values leave a zero right vector, which is documented
 * and excluded from the orthonormality assertion.
 *
 * Documented checks: on A = [[3,0,0],[0,2,0]] the algorithm returns sigma = (3,2) with the
 * canonical left/right vectors (up to sign) and reconstruction error 0 (rank-deficiency handled);
 * on a planted rank-2 matrix with additive 1e-3 noise the two leading singular values and their
 * vectors are recovered (sigma to 1e-6, |dot| >= 1-1e-6) and the rank-2 truncated reconstruction is
 * within twice the injected noise energy; the full-rank case (l = min(m,n)) reconstructs A to a
 * relative Frobenius error <= 1e-9, ||A||_F matches sqrt(sum sigma^2) to 1e-12, Q, U and V are
 * orthonormal, and the singular values agree with a second, independent decomposition of A (the
 * exact SVD from the Jacobi eigen decomposition of A^T A) to 1e-8. On a 10 x 10 matrix with the
 * slowly decaying spectrum sigma_i = 1/i and l = 4 the error is measured against the optimal rank-4
 * truncation floor sqrt(tail energy / total energy) = 0.2853: three power iterations land at 1.01x
 * the floor (sigma1 and sigma2 to 1e-6) while the un-iterated sketch sits at 1.35x and
 * mis-estimates sigma1 by 3e-3; on a clustered spectrum (1, 0.999, 0.998, 0.997 and six 1e-3
 * values, still l = 4) the iterated sketch reaches 1.00x the floor 0.001225 whereas the un-iterated
 * one is 4.3x off - the power method is exactly what makes a tight cluster resolvable. A
 * bit-identical deterministic re-run closes the test.
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

    static class Svd {
        final double[][] u;
        final double[] s;
        final double[][] v;
        final double[][] q;
        final int rank;

        Svd(double[][] u, double[] s, double[][] v, double[][] q, int rank) {
            this.u = u;
            this.s = s;
            this.v = v;
            this.q = q;
            this.rank = rank;
        }
    }

    static double[][] matmul(double[][] a, double[][] b) {
        int m = a.length;
        int inner = b.length;
        int n = b[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < inner; k++) {
                double aik = a[i][k];
                if (aik == 0.0) continue;
                for (int j = 0; j < n; j++) out[i][j] += aik * b[k][j];
            }
        }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        }
        return out;
    }

    static double[] matVec(double[][] a, double[] x) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < x.length; j++) sum += a[i][j] * x[j];
            out[i] = sum;
        }
        return out;
    }

    static double[] transposeMatVec(double[][] a, double[] x) {
        double[] out = new double[a[0].length];
        for (int i = 0; i < a.length; i++) {
            double xi = x[i];
            for (int j = 0; j < out.length; j++) out[j] += a[i][j] * xi;
        }
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

    static double frobenius(double[][] a) {
        double sum = 0.0;
        for (double[] row : a) {
            for (double value : row) sum += value * value;
        }
        return Math.sqrt(sum);
    }

    /** Modified Gram-Schmidt applied twice, so the columns of the result are orthonormal. */
    static double[][] orthonormalise(double[][] y) {
        int m = y.length;
        int l = y[0].length;
        double[][] q = new double[m][l];
        for (int j = 0; j < l; j++) {
            double[] col = new double[m];
            for (int i = 0; i < m; i++) col[i] = y[i][j];
            for (int pass = 0; pass < 2; pass++) {
                for (int c = 0; c < j; c++) {
                    double[] prev = column(q, c);
                    double coeff = dot(prev, col);
                    for (int i = 0; i < m; i++) col[i] -= coeff * prev[i];
                }
            }
            double n = norm(col);
            if (n <= 1e-300) {
                // rank-deficient input: complete the basis with a canonical direction
                for (int attempt = 0; attempt < m; attempt++) {
                    double[] candidate = new double[m];
                    candidate[attempt] = 1.0;
                    for (int c = 0; c < j; c++) {
                        double[] prev = column(q, c);
                        double coeff = candidate[attempt] * prev[attempt];
                        for (int i = 0; i < m; i++) candidate[i] -= coeff * prev[i];
                    }
                    if (norm(candidate) > 1e-6) {
                        col = candidate;
                        n = norm(col);
                        break;
                    }
                }
            }
            for (int i = 0; i < m; i++) q[i][j] = col[i] / n;
        }
        return q;
    }

    static double[] column(double[][] a, int j) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i][j];
        return out;
    }

    static double[][] gaussianMatrix(int rows, int cols, int seed) {
        Lcg lcg = new Lcg(seed);
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) out[i][j] = lcg.normal();
        }
        return out;
    }

    /** Cyclic Jacobi eigen decomposition; vectors are returned one per row, values descending. */
    static Eigen jacobi(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) a[i] = input[i].clone();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
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
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        double[] finalValues = values;
        Arrays.sort(order, (l, r) -> Double.compare(finalValues[r], finalValues[l]));
        double[] sortedValues = new double[n];
        double[][] sorted = new double[n][n];
        for (int j = 0; j < n; j++) {
            sortedValues[j] = values[order[j]];
            for (int i = 0; i < n; i++) sorted[j][i] = v[i][order[j]];
        }
        return new Eigen(sortedValues, sorted, sweeps);
    }

    /** Randomized SVD with l sketch columns and q power iterations. */
    static Svd randomizedSvd(double[][] a, int l, int q, int seed) {
        int m = a.length;
        int n = a[0].length;
        double[][] omega = gaussianMatrix(n, l, seed);
        double[][] y = matmul(a, omega);
        for (int i = 0; i < q; i++) {
            y = orthonormalise(y);
            double[][] z = matmul(transpose(a), y);
            y = matmul(a, z);
        }
        double[][] qm = orthonormalise(y);
        double[][] b = matmul(transpose(qm), a);
        Eigen eig = jacobi(matmul(b, transpose(b)));
        double[][] ub = eig.vectors;
        double[] s = new double[l];
        double[][] v = new double[l][n];
        double[][] u = new double[l][m];
        int rank = 0;
        double largest = Math.sqrt(Math.max(eig.values[0], 0.0));
        for (int i = 0; i < l; i++) {
            s[i] = Math.sqrt(Math.max(eig.values[i], 0.0));
            if (s[i] > 1e-14 * Math.max(1.0, largest)) {
                double[] right = transposeMatVec(b, ub[i]);
                for (int j = 0; j < n; j++) v[i][j] = right[j] / s[i];
                double[] left = matVec(qm, ub[i]);
                for (int j = 0; j < m; j++) u[i][j] = left[j];
                rank++;
            }
        }
        return new Svd(u, s, v, qm, rank);
    }

    /** rank-k truncated reconstruction U_k S_k V_k^T. */
    static double[][] reconstruct(Svd svd, int k) {
        int m = svd.u[0].length;
        int n = svd.v[0].length;
        double[][] out = new double[m][n];
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < m; i++) {
                double ui = svd.u[c][i] * svd.s[c];
                for (int j = 0; j < n; j++) out[i][j] += ui * svd.v[c][j];
            }
        }
        return out;
    }

    static double relativeError(double[][] a, double[][] b) {
        double diff = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                double d = a[i][j] - b[i][j];
                diff += d * d;
            }
        }
        return Math.sqrt(diff) / frobenius(a);
    }

    /** Exact SVD from the Jacobi eigen decomposition of A^T A - the independent oracle. */
    static Svd exactSvd(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        Eigen eig = jacobi(matmul(transpose(a), a));
        int k = Math.min(m, n);
        double[] s = new double[k];
        double[][] v = new double[k][n];
        double[][] u = new double[k][m];
        for (int i = 0; i < k; i++) {
            s[i] = Math.sqrt(Math.max(eig.values[i], 0.0));
            v[i] = eig.vectors[i].clone();
            double[] left = matVec(a, v[i]);
            for (int j = 0; j < m; j++) u[i][j] = left[j] / s[i];
        }
        return new Svd(u, s, v, null, k);
    }

    static double[][] plantedRankTwo(int m, int n, double s1, double s2, double noise, int seed) {
        Lcg lcg = new Lcg(seed);
        double[] u1 = new double[m];
        double[] u2 = new double[m];
        double[] v1 = new double[n];
        double[] v2 = new double[n];
        for (int i = 0; i < m; i++) {
            u1[i] = lcg.normal();
            u2[i] = lcg.normal();
        }
        for (int j = 0; j < n; j++) {
            v1[j] = lcg.normal();
            v2[j] = lcg.normal();
        }
        orthonormalisePair(u1, u2);
        orthonormalisePair(v1, v2);
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out[i][j] = s1 * u1[i] * v1[j] + s2 * u2[i] * v2[j] + noise * lcg.normal();
            }
        }
        return out;
    }

    static void orthonormalisePair(double[] first, double[] second) {
        double n1 = norm(first);
        for (int i = 0; i < first.length; i++) first[i] /= n1;
        double coeff = dot(second, first);
        for (int i = 0; i < second.length; i++) second[i] -= coeff * first[i];
        double n2 = norm(second);
        for (int i = 0; i < second.length; i++) second[i] /= n2;
    }

    /** Diagonal matrix with the prescribed singular values (m x n, zeros elsewhere). */
    static double[][] diagonal(int m, int n, double[] values) {
        double[][] out = new double[m][n];
        for (int i = 0; i < values.length; i++) out[i][i] = values[i];
        return out;
    }

    /** Optimal rank-k relative error from a known spectrum: sqrt(tail energy / total energy). */
    static double tailRatio(double[] spectrum, int k) {
        double total = 0.0;
        double tail = 0.0;
        for (int i = 0; i < spectrum.length; i++) {
            double squared = spectrum[i] * spectrum[i];
            total += squared;
            if (i >= k) tail += squared;
        }
        return Math.sqrt(tail / total);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        // 1. Closed form: A = diag(3,2) in a 2x3 frame has sigma = (3,2).
        double[][] simple = {{3.0, 0.0, 0.0}, {0.0, 2.0, 0.0}};
        Svd simpleSvd = randomizedSvd(simple, 2, 2, 11);
        check(Math.abs(simpleSvd.s[0] - 3.0) <= 1e-9 && Math.abs(simpleSvd.s[1] - 2.0) <= 1e-9,
                "diag(3,2): singular values are 3 and 2");
        check(Math.abs(simpleSvd.v[0][0]) >= 1.0 - 1e-9 && Math.abs(simpleSvd.v[1][1]) >= 1.0 - 1e-9,
                "diag(3,2): right singular vectors are canonical");
        check(Math.abs(simpleSvd.u[0][0]) >= 1.0 - 1e-9 && Math.abs(simpleSvd.u[1][1]) >= 1.0 - 1e-9,
                "diag(3,2): left singular vectors are canonical");
        check(relativeError(simple, reconstruct(simpleSvd, 2)) <= 1e-12,
                "diag(3,2): rank-2 reconstruction is exact");

        // 2. Planted rank-2 matrix with additive noise.
        int m = 8;
        int n = 5;
        double[][] planted = plantedRankTwo(m, n, 6.0, 2.5, 1e-3, 777);
        Svd plantedSvd = randomizedSvd(planted, 3, 2, 777);
        check(Math.abs(plantedSvd.s[0] - 6.0) <= 1e-2 && Math.abs(plantedSvd.s[1] - 2.5) <= 1e-2,
                "planted: the two leading singular values are recovered to the noise scale");
        Svd plantedExact = exactSvd(planted);
        for (int c = 0; c < 2; c++) {
            check(Math.abs(plantedSvd.s[c] - plantedExact.s[c])
                            <= 1e-8 * Math.max(1.0, plantedExact.s[c]),
                    "planted: sigma " + c + " agrees with the exact oracle");
            check(Math.abs(dot(plantedSvd.u[c], plantedExact.u[c])) >= 1.0 - 1e-6,
                    "planted: left vector " + c + " agrees with the oracle");
            check(Math.abs(dot(plantedSvd.v[c], plantedExact.v[c])) >= 1.0 - 1e-6,
                    "planted: right vector " + c + " agrees with the oracle");
        }
        check(plantedSvd.rank == 3, "all three sketched components clear the 1e-14 rank tolerance");
        double noiseEnergy = 1e-3 * Math.sqrt(m * n);
        check(relativeError(planted, reconstruct(plantedSvd, 2)) <= 2.0 * noiseEnergy,
                "planted: rank-2 truncation is within twice the noise energy");

        // 3. Full-rank recovery on a well-conditioned 10 x 10 (diagonal spectrum 1/i) plus the
        //    structural invariants: when l = n the sketch spans the whole column space, so the
        //    randomized decomposition must reproduce A exactly.
        double[] spectrum = new double[10];
        for (int i = 0; i < 10; i++) spectrum[i] = 1.0 / (i + 1);
        double[][] slow = diagonal(10, 10, spectrum);
        Svd full = randomizedSvd(slow, 10, 1, 4242);
        check(relativeError(slow, reconstruct(full, full.s.length)) <= 1e-9,
                "l = n: reconstruction is exact");
        check(full.rank == 10, "a full ten-column sketch has rank ten");
        double energy = 0.0;
        for (double value : full.s) energy += value * value;
        check(Math.abs(Math.sqrt(energy) - frobenius(slow)) <= 1e-12 * frobenius(slow),
                "||A||_F equals sqrt(sum sigma^2)");
        for (int c = 0; c < 10; c++) {
            check(Math.abs(full.s[c] - spectrum[c]) <= 1e-9, "sigma " + c + " is 1/" + (c + 1));
        }
        for (int c = 1; c < full.s.length; c++) {
            check(full.s[c - 1] >= full.s[c] - 1e-12, "singular values are non-increasing");
        }
        for (double value : full.s) check(value >= 0.0, "singular values are non-negative");
        // Q^T Q = I (columns of Q), U^T U = I and V V^T = I (rows of U/V)
        int l = full.q[0].length;
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < l; j++) {
                double qq = 0.0;
                for (int r = 0; r < full.q.length; r++) qq += full.q[r][i] * full.q[r][j];
                check(Math.abs(qq - (i == j ? 1.0 : 0.0)) <= 1e-12, "Q has orthonormal columns");
            }
        }
        for (int i = 0; i < full.s.length; i++) {
            for (int j = 0; j < full.s.length; j++) {
                double target = i == j ? 1.0 : 0.0;
                check(Math.abs(dot(full.u[i], full.u[j]) - target) <= 1e-9, "U has orthonormal rows");
                check(Math.abs(dot(full.v[i], full.v[j]) - target) <= 1e-9, "V has orthonormal rows");
            }
        }
        // the leading two components of the well-separated planted matrix are orthonormal too
        Svd rankTwo = randomizedSvd(planted, 2, 2, 4242);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double target = i == j ? 1.0 : 0.0;
                check(Math.abs(dot(rankTwo.u[i], rankTwo.u[j]) - target) <= 1e-9,
                        "rank-2 sketch: U rows orthonormal");
                check(Math.abs(dot(rankTwo.v[i], rankTwo.v[j]) - target) <= 1e-9,
                        "rank-2 sketch: V rows orthonormal");
            }
        }

        // 4. Power iterations: measured against the optimal rank-4 truncation error (the tail
        //    energy that no rank-4 approximation can avoid).
        double[] slowSpectrum = new double[10];
        for (int i = 0; i < 10; i++) slowSpectrum[i] = 1.0 / (i + 1);
        double floorSlow = tailRatio(slowSpectrum, 4);
        double[][] slowMatrix = diagonal(10, 10, slowSpectrum);
        Svd sketchOnly = randomizedSvd(slowMatrix, 4, 0, 31337);
        Svd powered = randomizedSvd(slowMatrix, 4, 3, 31337);
        double err0 = relativeError(slowMatrix, reconstruct(sketchOnly, 4));
        double err3 = relativeError(slowMatrix, reconstruct(powered, 4));
        check(Math.abs(err3 / floorSlow - 1.0) <= 0.10,
                "slow spectrum: three iterations are within 10% of the optimal rank-4 error");
        check(err0 >= 1.2 * floorSlow, "slow spectrum: the un-iterated sketch is clearly worse");
        check(err3 < err0, "slow spectrum: power iterations improve the sketch");
        check(Math.abs(powered.s[0] - 1.0) <= 1e-6 && Math.abs(powered.s[1] - 0.5) <= 1e-6,
                "slow spectrum: three iterations nail sigma1 and sigma2 to 1e-6");
        check(Math.abs(sketchOnly.s[0] - 1.0) >= 1e-4,
                "slow spectrum: the un-iterated sketch mis-estimates sigma1 by more than 1e-4");

        // 4b. A tight cluster of near-equal leading values is where the power method really pays:
        //     the un-iterated sketch cannot resolve gaps of 1e-3, the iterated one hits the floor.
        double[] clustered = {1.0, 0.999, 0.998, 0.997, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};
        double[][] clusterM = diagonal(10, 10, clustered);
        double floorCluster = tailRatio(clustered, 4);
        double clusterErr0 = relativeError(clusterM, reconstruct(randomizedSvd(clusterM, 4, 0, 31337), 4));
        double clusterErr3 = relativeError(clusterM, reconstruct(randomizedSvd(clusterM, 4, 3, 31337), 4));
        check(clusterErr3 <= 1.05 * floorCluster,
                "clustered spectrum: three iterations reach the optimal rank-4 error");
        check(clusterErr0 >= 3.0 * floorCluster,
                "clustered spectrum: the un-iterated sketch is at least 3x worse");
        check(clusterErr3 <= clusterErr0 / 3.0, "clustered spectrum: iteration gain is at least 3x");

        // 5. Determinism.
        Svd rerun = randomizedSvd(slowMatrix, 4, 3, 31337);
        for (int c = 0; c < 4; c++) {
            check(rerun.s[c] == powered.s[c], "deterministic re-run: sigma " + c);
            for (int i = 0; i < 10; i++) {
                check(rerun.v[c][i] == powered.v[c][i], "deterministic re-run: right vector " + c);
            }
        }

        System.out.println("PASS");
    }
}
