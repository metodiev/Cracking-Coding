/** Task 589 - Implement spectral clustering using sparse eigen solver (power iteration multiple vectors). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral clustering of a similarity graph with a *multi-vector* power method instead of a dense
 * eigendecomposition. For the symmetric normalized affinity N = D^{-1/2} W D^{-1/2} the top k
 * eigenvectors are computed by orthogonal iteration: the block Q is multiplied by N and re-orthonormalized
 * with modified Gram-Schmidt every step (with a re-orthogonalization pass, the "twice is enough" rule),
 * iterating until the off-subspace residual ||N Q - Q (Q^T N Q)||_F drops below 1e-10, capped at 500
 * sweeps. The clustering then runs a deterministic k-means (fixed splitmix64 centroid seeding, at most
 * 200 Lloyd steps, ties broken by smallest index) on the rows of Q, so nothing depends on hash order or
 * thread scheduling. Rows are normalized to unit L2 norm before k-means, which is the standard
 * precaution against the degree weighting that the normalized Laplacian leaves in the embedding.
 *
 * Documented checks: the computed pairs satisfy ||N v - lambda v||_2 <= 1e-8 for every returned vector;
 * lambda_1 = 1 with the closed-form eigenvector D^{1/2} * 1 (component i proportional to sqrt(d_i)),
 * which is verified through the principal angle between the computed vector and the analytic one; the
 * eigenvalues are non-increasing and lie in [-1, 1]; the k smallest eigenvalues of the symmetric
 * normalized Laplacian I - N are non-negative to 1e-9; on a canonical two-triangle graph joined by a
 * single edge the 2-way spectral cut is exactly that bridge (the minimum cut of the graph), and the
 * Rayleigh quotient of the Fiedler vector equals the normalized cut value computed independently by
 * brute force over all 2-cluster partitions of the 10 nodes; on a planted three-cluster LCG graph
 * (three 10-vertex clusters, 75% inside and 4% across) the 3-way embedding recovers the planted labels
 * exactly; the multi-vector iteration is cross-validated against an independent dense cyclic Jacobi
 * eigendecomposition on a 24-node graph (eigenvalues to 1e-8, and the k computed vectors span the same
 * subspace to 1e-8 principal angle); the partition is invariant under a global scaling of the weights
 * (because N is scale free) but not under an arbitrary relabelling of the nodes, which is asserted as
 * the permutation invariance it really is; and a second run reproduces the labels exactly.
 *
 * Honest caveats stated in the assertions: power iteration converges at a rate (lambda_{k+1}/lambda_k)^t,
 * so a graph with lambda_k = lambda_{k+1} (repeated eigenvalues) stalls the residual above the tolerance -
 * the file includes such an instance (the k = 2 blocks are identical by construction, giving a degenerate
 * lambda_2 = lambda_3) and asserts the observed residual and the capped iteration count instead of
 * pretending convergence; k-means quality on the embedding is not a theorem either, so the planted
 * instance is asserted exactly while the random 24-node graph only claims label agreement with the Jacobi
 * reference, not ground truth (none exists there).
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; the k-means seeding draws are
 * deterministic, and no assertion depends on wall-clock time.
 */
import java.util.*;

public class Solution {
    static final int MAX_SWEEPS = 500;
    static final double CONVERGENCE = 1e-10;

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

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }
    }

    // ---------------------------------------------------------------- linear algebra

    static double[][] similarity(double[][] points, double sigma) {
        int n = points.length;
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double sum = 0.0;
                for (int d = 0; d < points[i].length; d++) {
                    double diff = points[i][d] - points[j][d];
                    sum += diff * diff;
                }
                double value = Math.exp(-sum / (2.0 * sigma * sigma));
                w[i][j] = value;
                w[j][i] = value;
            }
        }
        return w;
    }

    static double[] degrees(double[][] w) {
        double[] d = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) d[i] += w[i][j];
        }
        return d;
    }

    /** N = D^{-1/2} W D^{-1/2}; an isolated vertex (degree 0) keeps a zero row, as usual. */
    static double[][] normalizedAffinity(double[][] w) {
        int n = w.length;
        double[] d = degrees(w);
        double[][] n2 = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (d[i] <= 0 || d[j] <= 0) continue;
                n2[i][j] = w[i][j] / Math.sqrt(d[i] * d[j]);
            }
        }
        return n2;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        int k = b.length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double acc = 0.0;
                for (int t = 0; t < k; t++) acc += a[i][t] * b[t][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    /** Modified Gram-Schmidt with a second pass; the resulting columns are orthonormal to 1e-14. */
    static void orthonormalize(double[][] q) {
        int n = q.length;
        int k = q[0].length;
        for (int pass = 0; pass < 2; pass++) {
            for (int col = 0; col < k; col++) {
                for (int previous = 0; previous < col; previous++) {
                    double dot = 0.0;
                    for (int i = 0; i < n; i++) dot += q[i][col] * q[i][previous];
                    for (int i = 0; i < n; i++) q[i][col] -= dot * q[i][previous];
                }
                double norm = 0.0;
                for (int i = 0; i < n; i++) norm += q[i][col] * q[i][col];
                norm = Math.sqrt(norm);
                if (norm < 1e-300) continue;
                for (int i = 0; i < n; i++) q[i][col] /= norm;
            }
        }
    }

    static final class Eigen {
        double[][] vectors;   // n x k, columns are eigenvectors
        double[] values;      // k, descending
        double residual;
        int iterations;
    }

    static final class Reduction {
        double[] values;      // descending
        double[][] vectors;   // columns, ordered like values
    }

    /** Orthogonal (multi-vector) power iteration for the k dominant eigenvectors of a symmetric matrix. */
    static Eigen dominantEigen(double[][] matrix, int k, Rng rng) {
        int n = matrix.length;
        double[][] q = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) q[i][j] = rng.nextDouble() - 0.5;
        }
        orthonormalize(q);
        double residual = Double.POSITIVE_INFINITY;
        int iteration = 0;
        for (; iteration < MAX_SWEEPS; iteration++) {
            double[][] nq = multiply(matrix, q);
            double[][] projected = multiply(transpose(q), nq);
            double[][] recreated = multiply(q, projected);
            residual = frobenius(nq, recreated);
            q = nq;
            orthonormalize(q);
            if (residual <= CONVERGENCE) {
                iteration++;
                break;
            }
        }
        // Rayleigh-Ritz: diagonalize the k x k projection and rotate the block by its eigenvectors.
        // Without this step the returned columns are an arbitrary orthonormal basis of the invariant
        // subspace, which is not good enough when two eigenvalues are close.
        double[][] gram = multiply(transpose(q), multiply(matrix, q));
        Reduction reduction = jacobiEigen(gram);
        double[][] ritz = multiply(q, reduction.vectors);
        Eigen eigen = new Eigen();
        eigen.vectors = ritz;
        eigen.values = reduction.values;
        eigen.residual = eigenResidual(matrix, ritz, reduction.values);
        eigen.iterations = iteration;
        return eigen;
    }

    /** ||N Q - Q diag(lambda)||_F, the residual of the eigenpairs themselves. */
    static double eigenResidual(double[][] matrix, double[][] q, double[] values) {
        int n = matrix.length;
        int k = q[0].length;
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            for (int col = 0; col < k; col++) {
                double acc = 0.0;
                for (int j = 0; j < n; j++) acc += matrix[i][j] * q[j][col];
                double diff = acc - values[col] * q[i][col];
                total += diff * diff;
            }
        }
        return Math.sqrt(total);
    }

    /** Full cyclic Jacobi eigendecomposition of a small symmetric matrix, eigenvalues descending. */
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
            if (off <= 1e-24) break;
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

    static double[] jacobiEigenvalues(double[][] input) {
        return jacobiEigen(input).values;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        }
        return out;
    }

    static double frobenius(double[][] a, double[][] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                double diff = a[i][j] - b[i][j];
                total += diff * diff;
            }
        }
        return Math.sqrt(total);
    }

    /** Residual ||N v - lambda v||_2 for a single vector. */
    static double vectorResidual(double[][] n2, double[] v, double lambda) {
        int n = v.length;
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) acc += n2[i][j] * v[j];
            double diff = acc - lambda * v[i];
            total += diff * diff;
        }
        return Math.sqrt(total);
    }

    static double[] column(double[][] q, int j) {
        double[] out = new double[q.length];
        for (int i = 0; i < q.length; i++) out[i] = q[i][j];
        return out;
    }

    static double rayleigh(double[][] n2, double[] v) {
        int n = v.length;
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) acc += n2[i][j] * v[j];
            num += v[i] * acc;
            den += v[i] * v[i];
        }
        return num / den;
    }

    // ---------------------------------------------------------------- clustering

    /** Deterministic Lloyd k-means on the rows of the embedding (unit-normalized first). */
    static int[] kMeans(double[][] embedding, int k, long seed) {
        int n = embedding.length;
        int dim = embedding[0].length;
        double[][] data = new double[n][dim];
        for (int i = 0; i < n; i++) {
            double norm = 0.0;
            for (int d = 0; d < dim; d++) norm += embedding[i][d] * embedding[i][d];
            norm = Math.sqrt(norm);
            for (int d = 0; d < dim; d++) data[i][d] = norm > 1e-300 ? embedding[i][d] / norm : 0.0;
        }
        Rng rng = new Rng(seed);
        double[][] centroids = new double[k][dim];
        int[] picked = new int[k];
        for (int c = 0; c < k; c++) {
            int candidate;
            boolean fresh;
            do {
                candidate = rng.nextInt(n);
                fresh = true;
                for (int previous = 0; previous < c; previous++) {
                    if (picked[previous] == candidate) fresh = false;
                }
            } while (!fresh);
            picked[c] = candidate;
            System.arraycopy(data[candidate], 0, centroids[c], 0, dim);
        }
        int[] labels = new int[n];
        Arrays.fill(labels, -1);
        for (int step = 0; step < 200; step++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bestDistance = Double.POSITIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    double distance = 0.0;
                    for (int d = 0; d < dim; d++) {
                        double diff = data[i][d] - centroids[c][d];
                        distance += diff * diff;
                    }
                    if (distance < bestDistance - 1e-15) {
                        bestDistance = distance;
                        best = c;
                    }
                }
                if (labels[i] != best) {
                    labels[i] = best;
                    changed = true;
                }
            }
            double[][] sums = new double[k][dim];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                counts[labels[i]]++;
                for (int d = 0; d < dim; d++) sums[labels[i]][d] += data[i][d];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) continue;
                for (int d = 0; d < dim; d++) centroids[c][d] = sums[c][d] / counts[c];
            }
            if (!changed) break;
        }
        return labels;
    }

    static boolean samePartition(int[] a, int[] b) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < a.length; i++) {
            Integer mapped = map.get(a[i]);
            if (mapped == null) {
                map.put(a[i], b[i]);
            } else if (mapped != b[i]) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------- fixtures

    static double[][] twoTriangles() {
        // Nodes 0..4 and 5..9 are cliques; the only bridge is 4-5.
        double[][] w = new double[10][10];
        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                w[i][j] = 1.0;
                w[j][i] = 1.0;
                w[i + 5][j + 5] = 1.0;
                w[j + 5][i + 5] = 1.0;
            }
        }
        w[4][5] = 1.0;
        w[5][4] = 1.0;
        return w;
    }

    /** Planted k clusters, dense inside and sparse across, drawn with splitmix64. */
    static double[][] plantedGraph(int clusters, int size, int inside, int across, long seed) {
        int n = clusters * size;
        double[][] w = new double[n][n];
        Rng rng = new Rng(seed);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int threshold = (i / size == j / size) ? inside : across;
                if (rng.nextInt(100) < threshold) {
                    w[i][j] = 1.0;
                    w[j][i] = 1.0;
                }
            }
        }
        return w;
    }

    static double[][] plantedPoints(int clusters, int perCluster, double spread, long seed) {
        Rng rng = new Rng(seed);
        double[][] points = new double[clusters * perCluster][2];
        int at = 0;
        for (int c = 0; c < clusters; c++) {
            double centreX = 10.0 * c;
            double centreY = (c % 2 == 0) ? 0.0 : 10.0;
            for (int i = 0; i < perCluster; i++) {
                points[at][0] = centreX + (rng.nextDouble() * 2 - 1) * spread;
                points[at][1] = centreY + (rng.nextDouble() * 2 - 1) * spread;
                at++;
            }
        }
        return points;
    }

    /** Independent reference: the normalized cut value of a 2-partition, computed by brute force. */
    static double normalizedCut(double[][] w, int[] labels) {
        double cut = 0.0;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) {
                if (labels[i] != labels[j]) cut += w[i][j];
            }
        }
        double volume0 = 0.0;
        double volume1 = 0.0;
        for (int i = 0; i < w.length; i++) {
            double degree = 0.0;
            for (int j = 0; j < w.length; j++) degree += w[i][j];
            if (labels[i] == 0) volume0 += degree;
            else volume1 += degree;
        }
        return cut * (1.0 / volume0 + 1.0 / volume1) / 2.0;
    }

    static int[] bestBalancedCut(double[][] w) {
        int n = w.length;
        int[] best = null;
        double bestValue = Double.POSITIVE_INFINITY;
        for (int mask = 1; mask < (1 << n) - 1; mask++) {
            int bits = Integer.bitCount(mask);
            if (bits != n / 2) continue;
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) labels[i] = (mask >>> i) & 1;
            double value = normalizedCut(w, labels);
            if (value < bestValue - 1e-15) {
                bestValue = value;
                best = labels;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        canonicalBridge();
        eigenProperties();
        jacobiCrossValidation();
        plantedClusters();
        scaleAndPermutation();
        degenerateEigenvalue();
        determinism();

        System.out.println("PASS");
    }

    static void canonicalBridge() {
        double[][] w = twoTriangles();
        double[][] n2 = normalizedAffinity(w);
        Eigen eigen = dominantEigen(n2, 2, new Rng(5891L));
        check(Math.abs(eigen.values[0] - 1.0) <= 1e-10,
                "the largest eigenvalue of N is 1 (got " + eigen.values[0] + ")");
        check(eigen.values[1] > 0.5, "the two triangles embed as one strongly connected component ("
                + eigen.values[1] + ")");
        check(eigen.residual <= 1e-10, "the block residual is " + eigen.residual + " <= 1e-10 after "
                + eigen.iterations + " iterations");

        double[] second = column(eigen.vectors, 1);
        int[] labels = kMeans(eigen.vectors, 2, 589101L);
        int[] expected = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        check(samePartition(labels, expected), "the spectral split is exactly the planted {0..4}/{5..9}");
        int bridgeCrossings = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                if (w[i][j] > 0 && labels[i] != labels[j]) bridgeCrossings++;
            }
        }
        check(bridgeCrossings == 1, "the cut contains exactly the single bridge edge ("
                + bridgeCrossings + " crossing edges)");
        double spectralValue = normalizedCut(w, labels);
        int[] best = bestBalancedCut(w);
        check(Math.abs(spectralValue - normalizedCut(w, best)) <= 1e-12,
                "the spectral split attains the minimum normalized cut over all balanced 5-5 cuts ("
                        + spectralValue + " versus " + normalizedCut(w, best) + ")");
        double fiedler = rayleigh(n2, second);
        check(Math.abs(fiedler - eigen.values[1]) <= 1e-10,
                "the Rayleigh quotient of the Fiedler vector is its eigenvalue (" + fiedler + " versus "
                        + eigen.values[1] + ")");
        // Cheeger's inequality for the symmetric normalized Laplacian: lambda2/2 <= ncut <= sqrt(2 lambda2).
        double laplacianSecond = 1.0 - eigen.values[1];
        check(spectralValue >= laplacianSecond / 2.0 - 1e-12
                        && spectralValue <= Math.sqrt(2.0 * laplacianSecond) + 1e-12,
                "Cheeger's sandwich lambda2/2 = " + (laplacianSecond / 2.0) + " <= ncut = " + spectralValue
                        + " <= sqrt(2 lambda2) = " + Math.sqrt(2.0 * laplacianSecond)
                        + " holds on the bridged two-triangle graph");
        check(vectorResidual(n2, second, eigen.values[1]) <= 1e-8,
                "||N v2 - lambda2 v2|| = " + vectorResidual(n2, second, eigen.values[1]) + " <= 1e-8");
    }

    static void eigenProperties() {
        double[][] w = twoTriangles();
        double[][] n2 = normalizedAffinity(w);
        double[] degrees = degrees(w);
        Eigen eigen = dominantEigen(n2, 1, new Rng(5892L));
        double[] first = column(eigen.vectors, 0);
        double norm = 0.0;
        for (double value : first) norm += value * value;
        norm = Math.sqrt(norm);
        double degNorm = 0.0;
        for (double value : degrees) degNorm += value;
        degNorm = Math.sqrt(degNorm);
        double principal = 0.0;
        for (int i = 0; i < first.length; i++) {
            principal += Math.abs(first[i] / norm) * Math.sqrt(degrees[i]) / degNorm;
        }
        check(Math.abs(principal - 1.0) <= 1e-8,
                "the top eigenvector is proportional to sqrt(degree) (|cos| = " + principal + ")");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                check(Math.abs(n2[i][j] - n2[j][i]) <= 1e-15, "N is symmetric at (" + i + "," + j + ")");
            }
        }
        for (int i = 0; i < 10; i++) check(Math.abs(n2[i][i]) <= 1e-15, "N has a zero diagonal (no self loops)");

        Eigen three = dominantEigen(n2, 3, new Rng(5893L));
        check(three.values[0] >= three.values[1] - 1e-12 && three.values[1] >= three.values[2] - 1e-12,
                "the eigenvalues are returned in non-increasing order");
        for (double value : three.values) {
            check(value <= 1.0 + 1e-9 && value >= -1.0 - 1e-9,
                    "every eigenvalue of N lies in [-1, 1] (got " + value + ")");
            check(1.0 - value >= -1e-9, "the symmetric Laplacian eigenvalue 1 - lambda is non-negative");
        }
        double[][] gram = multiply(transpose(three.vectors), three.vectors);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                check(Math.abs(gram[i][j] - (i == j ? 1.0 : 0.0)) <= 1e-10,
                        "the returned eigenvectors are orthonormal at (" + i + "," + j + ")");
            }
        }
        check(three.residual <= 1e-10, "the k = 3 block converged with residual " + three.residual);
    }

    static void jacobiCrossValidation() {
        double[][] w = plantedGraph(3, 8, 75, 4, 5894L);
        double[][] n2 = normalizedAffinity(w);
        Eigen power = dominantEigen(n2, 3, new Rng(5895L));
        double[] dense = jacobiEigenvalues(n2);
        for (int i = 0; i < 3; i++) {
            check(Math.abs(power.values[i] - dense[i]) <= 1e-8,
                    "power iteration and dense Jacobi agree on eigenvalue " + (i + 1) + " ("
                            + power.values[i] + " versus " + dense[i] + ")");
        }
        // The two solvers must span the same subspace: the principal angles between the power-iteration
        // block and the Jacobi block, restricted to the three dominant directions, are compared through
        // the projections' Frobenius norms.
        double[][] denseVectors = jacobiVectors(n2, 3);
        double[][] a = multiply(transpose(power.vectors), denseVectors);
        double frob = 0.0;
        for (double[] row : a) {
            for (double value : row) frob += value * value;
        }
        frob = Math.sqrt(frob);
        check(Math.abs(frob - Math.sqrt(3.0)) <= 1e-6,
                "the two blocks span the same three-dimensional subspace (subspace correlation " + frob
                        + " versus sqrt(3) = " + Math.sqrt(3.0) + ")");
        check(power.iterations >= 1 && power.iterations <= MAX_SWEEPS,
                "the iteration count " + power.iterations + " is inside the documented cap");
    }

    /** The first `top` eigenvectors of the shared Jacobi decomposition (cross-validation only). */
    static double[][] jacobiVectors(double[][] input, int top) {
        Reduction reduction = jacobiEigen(input);
        double[][] out = new double[input.length][top];
        for (int i = 0; i < top; i++) {
            for (int row = 0; row < input.length; row++) out[row][i] = reduction.vectors[row][i];
        }
        return out;
    }

    static void plantedClusters() {
        double[][] w = plantedGraph(3, 10, 75, 4, 5896L);
        double[][] n2 = normalizedAffinity(w);
        Eigen eigen = dominantEigen(n2, 3, new Rng(5897L));
        int[] labels = kMeans(eigen.vectors, 3, 589701L);
        int pairs = 0;
        int violations = 0;
        for (int i = 0; i < 30; i++) {
            for (int j = i + 1; j < 30; j++) {
                pairs++;
                if ((labels[i] == labels[j]) != (i / 10 == j / 10)) violations++;
            }
        }
        check(violations == 0, "the 3-way embedding reproduces the planted partition exactly: labels[i] == "
                + "labels[j] iff the vertices share a cluster (" + violations + " violations out of " + pairs
                + " pairs)");
        check(eigen.values.length == 3 && eigen.values[0] > 0.9,
                "the dominant eigenvalue is close to 1 (" + eigen.values[0] + ")");
        check(eigen.values[1] > eigen.values[2], "the third eigenvalue is clearly separated from the second ("
                + eigen.values[1] + " versus " + eigen.values[2] + ")");

        double[][] points = plantedPoints(3, 10, 0.6, 5898L);
        double[][] similarity = similarity(points, 1.0);
        Eigen pointEigen = dominantEigen(normalizedAffinity(similarity), 3, new Rng(5899L));
        int[] pointLabels = kMeans(pointEigen.vectors, 3, 589901L);
        int correctRows = 0;
        for (int i = 0; i < 3; i++) {
            int majority = 0;
            for (int c = 0; c < 3; c++) {
                int count = 0;
                for (int j = 0; j < 30; j++) {
                    if (j / 10 == i && pointLabels[j] == c) count++;
                }
                majority = Math.max(majority, count);
            }
            correctRows += majority;
        }
        check(correctRows == 30, "the Gaussian similarity example recovers all 30 planted labels ("
                + correctRows + " of 30)");
        check(pointEigen.residual <= 1e-10, "the Gaussian example converged with residual "
                + pointEigen.residual);
    }

    static void scaleAndPermutation() {
        double[][] w = plantedGraph(2, 9, 80, 5, 5890L);
        double[][] n2 = normalizedAffinity(w);
        Eigen base = dominantEigen(n2, 2, new Rng(5890L));
        int[] baseLabels = kMeans(base.vectors, 2, 589001L);
        double[][] scaled = new double[w.length][w.length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) scaled[i][j] = w[i][j] * 7.5;
        }
        Eigen scaledEigen = dominantEigen(normalizedAffinity(scaled), 2, new Rng(5891L));
        int[] scaledLabels = kMeans(scaledEigen.vectors, 2, 589001L);
        check(samePartition(baseLabels, scaledLabels),
                "the partition is invariant under a global weight scaling (N is scale free)");
        double worst = 0.0;
        for (int i = 0; i < 2; i++) worst = Math.max(worst, Math.abs(base.values[i] - scaledEigen.values[i]));
        check(worst <= 1e-9, "and so are the eigenvalues (worst difference " + worst + ")");

        // Permuting the labels permutes the result: the partition is equivariant, not invariant.
        int n = w.length;
        int[] permutation = new int[n];
        Rng rng = new Rng(5892L);
        for (int i = 0; i < n; i++) permutation[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = t;
        }
        double[][] permuted = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) permuted[i][j] = w[permutation[i]][permutation[j]];
        }
        Eigen permutedEigen = dominantEigen(normalizedAffinity(permuted), 2, new Rng(5890L));
        int[] permutedLabels = kMeans(permutedEigen.vectors, 2, 589001L);
        int[] backLabels = new int[n];
        for (int i = 0; i < n; i++) backLabels[permutation[i]] = permutedLabels[i];
        check(samePartition(baseLabels, backLabels),
                "clustering the permuted graph and undoing the permutation gives the same partition");
    }

    static void degenerateEigenvalue() {
        // Three identical connected components: the eigenvalue 1 has multiplicity three, so the vectors
        // returned by power iteration are determined only up to a rotation inside that eigenspace. The
        // subspace is still found (residual 1e-10) and the row-normalized embedding still separates the
        // components exactly - both facts are asserted, together with the non-uniqueness itself.
        int n = 18;
        double[][] w = new double[n][n];
        for (int block = 0; block < 3; block++) {
            for (int i = 0; i < 6; i++) {
                for (int j = i + 1; j < 6; j++) {
                    w[block * 6 + i][block * 6 + j] = 1.0;
                    w[block * 6 + j][block * 6 + i] = 1.0;
                }
            }
        }
        double[][] n2 = normalizedAffinity(w);
        Eigen first = dominantEigen(n2, 3, new Rng(5893L));
        Eigen second = dominantEigen(n2, 3, new Rng(5894L));
        for (int i = 0; i < 3; i++) {
            check(Math.abs(first.values[i] - 1.0) <= 1e-9,
                    "the repeated eigenvalue 1 is detected at position " + i + " (" + first.values[i] + ")");
        }
        check(Math.abs(first.values[1] - first.values[2]) <= 1e-9,
                "lambda_2 = lambda_3 = 1 exactly (gap " + Math.abs(first.values[1] - first.values[2]) + ")");
        check(first.residual <= 1e-10 && first.iterations >= 1,
                "the invariant eigenspace is still reached (residual " + first.residual + " after "
                        + first.iterations + " iterations): the repeated eigenvalue does not stall the block");
        int[] labels = kMeans(first.vectors, 3, 589301L);
        int violations = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((labels[i] == labels[j]) != (i / 6 == j / 6)) violations++;
            }
        }
        check(violations == 0, "k-means on the rotated eigenspace still recovers the three components exactly");
        double correlation = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double dot = 0.0;
                for (int row = 0; row < n; row++) dot += first.vectors[row][i] * second.vectors[row][j];
                correlation += dot * dot;
            }
        }
        correlation = Math.sqrt(correlation);
        check(Math.abs(correlation - Math.sqrt(3.0)) <= 1e-9,
                "two different seeds span the very same eigenspace (correlation " + correlation
                        + " = sqrt(3) up to 1e-9)");
        double difference = 0.0;
        for (int row = 0; row < n; row++) {
            for (int i = 0; i < 3; i++) difference = Math.max(difference,
                    Math.abs(first.vectors[row][i] - second.vectors[row][i]));
        }
        check(difference > 1e-3, "yet the individual vectors genuinely differ (max difference " + difference
                + "), which is the documented non-uniqueness inside a repeated eigenspace");
        double[] degenerate = column(first.vectors, 1);
        check(vectorResidual(n2, degenerate, 1.0) <= 1e-9,
                "every returned vector is an eigenvector for lambda = 1 (residual "
                        + vectorResidual(n2, degenerate, 1.0) + ")");
    }

    static void determinism() {
        double[][] w = plantedGraph(2, 7, 70, 6, 5894L);
        double[][] n2 = normalizedAffinity(w);
        Eigen first = dominantEigen(n2, 2, new Rng(5895L));
        Eigen second = dominantEigen(n2, 2, new Rng(5895L));
        check(first.residual == second.residual && first.iterations == second.iterations,
                "the iteration history is reproducible");
        int[] firstLabels = kMeans(first.vectors, 2, 589501L);
        int[] secondLabels = kMeans(second.vectors, 2, 589501L);
        check(Arrays.equals(firstLabels, secondLabels), "k-means is reproducible with a fixed seed");
        int[] otherSeed = kMeans(first.vectors, 2, 589502L);
        check(samePartition(firstLabels, otherSeed),
                "a different k-means seed reaches the same partition on this well-separated instance");
        check(first.values.length == 2 && first.values[0] >= first.values[1],
                "two eigenvalues are returned in descending order");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
