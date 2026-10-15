/** Task 500 - Implement spectral clustering small example (eigen decomposition). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral 2-clustering of a handful of points: build the Gaussian similarity matrix
 * W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), form the combinatorial Laplacian L = D - W (D the
 * row sums), and take the Fiedler vector - the eigenvector of the second-smallest eigenvalue -
 * via cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below 1e-12,
 * documented cap 100 sweeps). The split threshold is the mean of the Fiedler vector, with the
 * sign fixed by making point 0 lie at or above the mean. Documented checks: the zero eigenvalue
 * and the residual ||L v - lambda2 v|| (<= 1e-9), lambda2 < lambda3 with the vector orthogonal
 * to the constant, and the resulting cut equals the minimum over all balanced 6-6 cuts of the
 * canonical two-blob example (the planted split). The canonical example is two six-point
 * clusters around (0,0) and (10,10) with sigma = 3.0 (cross-cluster similarities ~1e-4, so
 * lambda2 ~ 4.85e-4 - comfortably above the Jacobi tolerance); the tiny second example is two
 * LCG-jittered 4-point blobs around (0,0) and (3,0) with sigma = 0.5, where the Fiedler sign
 * pattern must again reproduce the planted partition.
 */
import java.util.*;

public class Solution {
    static final double[][] CANONICAL = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},
            {0.5, 1.5}, {1.5, 0.5}, {10.0, 10.0}, {11.0, 10.0}, {10.0, 11.0}, {11.0, 11.0},
            {10.5, 10.5}, {9.5, 10.5}};
    static final int[] CANONICAL_TRUTH = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
    static final double SIGMA_CANONICAL = 3.0;
    static final double SIGMA_TINY = 0.5;

    static double sqdist(double[] a, double[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    /** Two 4-point blobs around (0,0) and (3,0), jitter +/-0.4 in 0.01 steps via the LCG. */
    static Object[] tinyBlobs(int seed) {
        double[][] points = new double[8][2];
        int index = 0;
        for (double cx : new double[]{0.0, 3.0}) {
            for (int i = 0; i < 4; i++) {
                seed = (seed * 75 + 74) % 65537;
                points[index][0] = cx + (seed % 81 - 40) / 100.0;
                seed = (seed * 75 + 74) % 65537;
                points[index][1] = (seed % 81 - 40) / 100.0;
                index++;
            }
        }
        return new Object[]{points, new int[]{0, 0, 0, 0, 1, 1, 1, 1}};
    }

    static double[][] similarity(double[][] points, double sigma) {
        int n = points.length;
        double[][] w = new double[n][n];
        double denom = 2.0 * sigma * sigma;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double value = Math.exp(-sqdist(points[i], points[j]) / denom);
                w[i][j] = value;
                w[j][i] = value;
            }
        }
        return w;
    }

    static double[][] laplacian(double[][] w) {
        int n = w.length;
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            double degree = 0.0;
            for (int j = 0; j < n; j++) degree += w[i][j];
            for (int j = 0; j < n; j++) out[i][j] = (i == j ? degree : 0.0) - w[i][j];
        }
        return out;
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

    /** Cyclic Jacobi eigen-decomposition of a symmetric matrix; eigenvectors are columns. */
    static Eigen jacobiEigh(double[][] matrix, int maxSweeps, double tol) {
        int n = matrix.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = matrix[i].clone();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
        int sweeps = 0;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (Math.sqrt(off) <= tol) break;
            sweeps += 1;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) <= 1e-300) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = 1.0 / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    if (theta < 0.0) t = -t;
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
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
                    a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
                    a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
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
        return new Eigen(values, v, sweeps);
    }

    static class Split {
        final int[] labels;
        final double lam2;
        final double residual;
        final double orthogonality;
        final int sweeps;

        Split(int[] labels, double lam2, double residual, double orthogonality, int sweeps) {
            this.labels = labels;
            this.lam2 = lam2;
            this.residual = residual;
            this.orthogonality = orthogonality;
            this.sweeps = sweeps;
        }
    }

    static Split spectralSplit(double[][] points, double sigma) {
        int n = points.length;
        double[][] w = similarity(points, sigma);
        double[][] lmat = laplacian(w);
        Eigen eigen = jacobiEigh(lmat, 100, 1e-12);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(eigen.values[x], eigen.values[y]));
        int fiedler = order[1];
        double lam2 = eigen.values[fiedler];
        double[] vec = new double[n];
        double mean = 0.0;
        for (int i = 0; i < n; i++) {
            vec[i] = eigen.vectors[i][fiedler];
            mean += vec[i];
        }
        mean /= n;
        if (vec[0] < mean) {
            for (int i = 0; i < n; i++) vec[i] = -vec[i];
        }
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = vec[i] >= mean ? 0 : 1;
        double residual = 0.0;
        double orthogonality = 0.0;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) acc += lmat[i][j] * vec[j];
            residual += (acc - lam2 * vec[i]) * (acc - lam2 * vec[i]);
            orthogonality += vec[i];
        }
        return new Split(labels, lam2, Math.sqrt(residual), Math.abs(orthogonality),
                eigen.sweeps);
    }

    static double cutWeight(double[][] w, int[] side) {
        int n = w.length;
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (side[i] == 0 && side[j] == 1) total += w[i][j];
            }
        }
        return total;
    }

    /** Brute force over all balanced half-half splits (point 0 in side A, mirrors skipped). */
    static double minBalancedCut(double[][] w, int half) {
        int n = w.length;
        int choose = half - 1;
        int[] idx = new int[choose];
        for (int i = 0; i < choose; i++) idx[i] = i + 1;
        double best = Double.POSITIVE_INFINITY;
        while (true) {
            int[] side = new int[n];
            Arrays.fill(side, 1);
            side[0] = 0;
            for (int i : idx) side[i] = 0;
            if (side[0] != side[n - 1]) best = Math.min(best, cutWeight(w, side));
            int i = choose - 1;
            while (i >= 0 && idx[i] == n - choose + i) i--;
            if (i < 0) break;
            idx[i] += 1;
            for (int j = i + 1; j < choose; j++) idx[j] = idx[j - 1] + 1;
        }
        return best;
    }

    public static void main(String[] args) {
        Eigen small = jacobiEigh(new double[][]{{2.0, 1.0}, {1.0, 2.0}}, 100, 1e-12);
        double[] smallSorted = small.values.clone();
        Arrays.sort(smallSorted);
        check(Math.abs(smallSorted[0] - 1.0) <= 1e-12 && Math.abs(smallSorted[1] - 3.0) <= 1e-12,
                "2x2 eigenvalues");

        Split canonical = spectralSplit(CANONICAL, SIGMA_CANONICAL);
        check(canonical.residual <= 1e-9, "eigen-pair residual");
        check(canonical.orthogonality <= 1e-8, "Fiedler vector orthogonal to the constant");
        check(canonical.lam2 >= 1e-6 && canonical.lam2 <= 1e-2,
                "two separated blobs: tiny positive Fiedler value");
        check(canonical.sweeps <= 100, "sweep cap");

        double[][] w = similarity(CANONICAL, SIGMA_CANONICAL);
        Eigen spectrum = jacobiEigh(laplacian(w), 100, 1e-12);
        double[] asc = spectrum.values.clone();
        Arrays.sort(asc);
        check(Math.abs(asc[0]) <= 1e-9, "connected graph: one zero eigenvalue");
        check(asc[1] < asc[2], "Fiedler eigenvalue is strictly second smallest");
        check(asc[1] <= 1e-2 * asc[2], "documented: the Fiedler value separates the blobs");

        check(Arrays.equals(canonical.labels, CANONICAL_TRUTH),
                "canonical split must match the planted structure");
        double spectralCut = cutWeight(w, canonical.labels);
        double bestBalanced = minBalancedCut(w, 6);
        check(Math.abs(spectralCut - bestBalanced) <= 1e-9,
                "spectral cut equals the min 6-6 cut");
        double totalMass = 0.0;
        for (double[] row : w) {
            for (double value : row) totalMass += value;
        }
        check(spectralCut <= 1e-3 * totalMass, "cut fraction");

        Object[] tinyData = tinyBlobs(2024);
        double[][] tiny = (double[][]) tinyData[0];
        int[] truth = (int[]) tinyData[1];
        Split tinySplit = spectralSplit(tiny, SIGMA_TINY);
        check(Arrays.equals(tinySplit.labels, truth), "tiny LCG blobs split correctly");
        check(tinySplit.residual <= 1e-9 && tinySplit.orthogonality <= 1e-8, "tiny residuals");
        check(tinySplit.lam2 >= 1e-8 && tinySplit.lam2 <= 1e-2, "tiny positive Fiedler value");
        check(Arrays.equals(spectralSplit(tiny, SIGMA_TINY).labels, tinySplit.labels),
                "deterministic re-run");

        double[][] spaced = new double[tiny.length][2];
        for (int i = 0; i < tiny.length; i++) {
            spaced[i][0] = tiny[i][0] * 10.0;
            spaced[i][1] = tiny[i][1] * 10.0;
        }
        check(Arrays.equals(spectralSplit(spaced, SIGMA_TINY * 10.0).labels, truth),
                "scaled blobs split correctly");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
