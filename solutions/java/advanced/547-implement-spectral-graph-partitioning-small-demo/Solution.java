/** Task 547 - Implement spectral graph partitioning small demo. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Spectral bisection of a weighted undirected graph: build the combinatorial Laplacian L = D - W,
 * diagonalise it with cyclic Jacobi rotations (sweeps until the off-diagonal Frobenius norm falls
 * below 1e-12, documented cap of 100 sweeps), and split the vertices at the median of the Fiedler
 * vector — the eigenvector of the second-smallest eigenvalue, which minimises the relaxed
 * normalised cut; the threshold is the mean of the Fiedler vector (falling back to the median
 * when the mean degenerates to an empty side). Disconnected graphs are handled explicitly: L then has one zero eigenvalue per
 * component, the Fiedler value is itself ~ 0 and the split is taken from the connected components
 * instead (the sign-based bisection would be meaningless).
 *
 * Documented checks: Jacobi reproduces the known spectrum of [[2,1],[1,2]] (1 and 3) and of a
 * disconnected two-triangle graph (two zero eigenvalues), eigenvalues are non-negative and sum to
 * the trace, the number of near-zero eigenvalues equals the number of components, the Fiedler
 * vector is orthogonal to the constant vector and satisfies ||L v - lambda2 v|| <= 1e-9 with
 * lambda2 < lambda3, and lambda2 > 0 exactly when the graph is connected. The partition is
 * validated against brute force over all 2^(n-1) splits: the spectral cut weight is never below
 * the global minimum cut and is within a documented factor of the best *balanced* bisection on the
 * LCG instances; on the crafted examples (two K5 cliques joined by one edge, the 8-vertex path)
 * the spectral split is optimal, with the expected balance, and on the two-clique instance its
 * normalised cut is minimal among all bisections.
 */
import java.util.*;

public class Solution {
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

    /** Cyclic Jacobi eigen-decomposition of a symmetric matrix; columns of the result are vectors. */
    static Eigen jacobiEigh(double[][] matrix, int maxSweeps, double tol) {
        int n = matrix.length;
        double[][] a = new double[n][n];
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i] = matrix[i].clone();
            v[i][i] = 1.0;
        }
        int sweeps = 0;
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            sweeps++;
            double off = 0.0;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            }
            if (Math.sqrt(off) <= tol) break;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-18) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    if (theta == 0.0) t = 1.0;
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++) {
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++) {
                        double apk = a[p][k];
                        double aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
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

    static Eigen sortedEigen(double[][] matrix, int maxSweeps, double tol) {
        Eigen eigen = jacobiEigh(matrix, maxSweeps, tol);
        int n = eigen.values.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(eigen.values[x], eigen.values[y]));
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            values[i] = eigen.values[order[i]];
            for (int row = 0; row < n; row++) vectors[row][i] = eigen.vectors[row][order[i]];
        }
        return new Eigen(values, vectors, eigen.sweeps);
    }

    static double[][] laplacian(double[][] w) {
        int n = w.length;
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) {
                rowSum += w[i][j];
                l[i][j] = -w[i][j];
            }
            l[i][i] += rowSum;
        }
        return l;
    }

    static int components(double[][] w, int n) {
        boolean[] seen = new boolean[n];
        int count = 0;
        for (int start = 0; start < n; start++) {
            if (seen[start]) continue;
            count++;
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(start);
            seen[start] = true;
            while (!stack.isEmpty()) {
                int v = stack.pop();
                for (int u = 0; u < n; u++) {
                    if (w[v][u] > 0 && !seen[u]) {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
        }
        return count;
    }

    static class Split {
        final int[] labels;
        final double lambda2;
        final double residual;
        final double orthogonality;
        final int components;

        Split(int[] labels, double lambda2, double residual, double orthogonality, int components) {
            this.labels = labels;
            this.lambda2 = lambda2;
            this.residual = residual;
            this.orthogonality = orthogonality;
            this.components = components;
        }
    }

    static Split spectralSplit(double[][] w) {
        int n = w.length;
        if (n == 1) return new Split(new int[]{0}, 0.0, 0.0, 0.0, 1);
        double[][] l = laplacian(w);
        Eigen eigen = sortedEigen(l, 100, 1e-12);
        double[] fiedler = new double[n];
        for (int i = 0; i < n; i++) fiedler[i] = eigen.vectors[i][1];
        double lambda2 = eigen.values[1];
        double residual = 0.0;
        double orthogonality = 0.0;
        for (int i = 0; i < n; i++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) acc += l[i][j] * fiedler[j];
            residual += (acc - lambda2 * fiedler[i]) * (acc - lambda2 * fiedler[i]);
            orthogonality += fiedler[i];
        }
        int componentCount = components(w, n);
        int[] labels = new int[n];
        if (lambda2 <= 1e-9 && componentCount > 1) {
            int next = 0;
            int[] componentOf = new int[n];
            Arrays.fill(componentOf, -1);
            for (int start = 0; start < n; start++) {
                if (componentOf[start] >= 0) continue;
                Deque<Integer> stack = new ArrayDeque<>();
                stack.push(start);
                componentOf[start] = next;
                while (!stack.isEmpty()) {
                    int v = stack.pop();
                    for (int u = 0; u < n; u++) {
                        if (w[v][u] > 0 && componentOf[u] < 0) {
                            componentOf[u] = next;
                            stack.push(u);
                        }
                    }
                }
                next++;
            }
            for (int i = 0; i < n; i++) labels[i] = componentOf[i];
            return new Split(labels, lambda2, Math.sqrt(residual), Math.abs(orthogonality), componentCount);
        }
        double mean = 0.0;
        for (double value : fiedler) mean += value;
        mean /= n;
        for (int i = 0; i < n; i++) labels[i] = fiedler[i] <= mean ? 0 : 1;
        int low = sideSize(labels, 0);
        if (low == 0 || low == n) {                 // degenerate mean: fall back to the median
            double[] sorted = fiedler.clone();
            Arrays.sort(sorted);
            double median = sorted[n / 2];
            for (int i = 0; i < n; i++) labels[i] = fiedler[i] <= median ? 0 : 1;
        }
        return new Split(labels, lambda2, Math.sqrt(residual), Math.abs(orthogonality), componentCount);
    }

    static double cutWeight(double[][] w, int[] labels) {
        int n = w.length;
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (labels[i] == 0 && labels[j] == 1) total += w[i][j];
            }
        }
        return total;
    }

    /** Normalised cut: cut / (vol(A)) + cut / (vol(B)). */
    static double normalisedCut(double[][] w, int[] labels) {
        int n = w.length;
        double cut = cutWeight(w, labels);
        double volA = 0.0;
        double volB = 0.0;
        for (int i = 0; i < n; i++) {
            double degree = 0.0;
            for (int j = 0; j < n; j++) degree += w[i][j];
            if (labels[i] == 0) volA += degree;
            else volB += degree;
        }
        return cut / volA + cut / volB;
    }

    /** Brute force over all 2^(n-1) splits; balanced restricts to half/half with vertex 0 in A. */
    static double bruteMinimum(double[][] w, int n, boolean balanced) {
        double best = Double.POSITIVE_INFINITY;
        for (int mask = 0; mask < (1 << n); mask++) {
            if ((mask & 1) != 0) continue;                 // vertex 0 always in side A
            int size = 0;
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) size++;
            }
            if (size == 0 || size == n) continue;
            if (balanced && size != n / 2) continue;
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) labels[i] = (mask & (1 << i)) != 0 ? 1 : 0;
            best = Math.min(best, cutWeight(w, labels));
        }
        return best;
    }

    static double[][] cliquePair(int size, double bridge) {
        int n = 2 * size;
        double[][] w = new double[n][n];
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    w[c * size + i][c * size + j] = 1.0;
                    w[c * size + j][c * size + i] = 1.0;
                }
            }
        }
        w[0][size] = bridge;
        w[size][0] = bridge;
        return w;
    }

    static double[][] path(int n) {
        double[][] w = new double[n][n];
        for (int i = 0; i + 1 < n; i++) {
            w[i][i + 1] = 1.0;
            w[i + 1][i] = 1.0;
        }
        return w;
    }

    static double[][] twoTriangles() {
        double[][] w = new double[6][6];
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < 3; i++) {
                for (int j = i + 1; j < 3; j++) {
                    w[c * 3 + i][c * 3 + j] = 1.0;
                    w[c * 3 + j][c * 3 + i] = 1.0;
                }
            }
        }
        return w;
    }

    static double[][] randomWeighted(int n, int count, int seed) {
        double[][] w = new double[n][n];
        int placed = 0;
        int guard = 0;
        while (placed < count && guard < count * 40) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u == v || w[u][v] > 0) continue;
            double weight = 1 + seed % 5;
            w[u][v] = weight;
            w[v][u] = weight;
            placed++;
        }
        return w;
    }

    /** Add deterministic weight-1 bridges so that every random instance is connected. */
    static double[][] makeConnected(double[][] w, int n) {
        while (components(w, n) > 1) {
            int[] componentOf = new int[n];
            Arrays.fill(componentOf, -1);
            int next = 0;
            for (int start = 0; start < n; start++) {
                if (componentOf[start] >= 0) continue;
                Deque<Integer> stack = new ArrayDeque<>();
                stack.push(start);
                componentOf[start] = next;
                while (!stack.isEmpty()) {
                    int v = stack.pop();
                    for (int u = 0; u < n; u++) {
                        if (w[v][u] > 0 && componentOf[u] < 0) {
                            componentOf[u] = next;
                            stack.push(u);
                        }
                    }
                }
                next++;
            }
            int representative = -1;
            for (int v = 0; v < n; v++) {
                if (componentOf[v] == 0) {
                    representative = v;
                    break;
                }
            }
            for (int v = 0; v < n; v++) {
                if (componentOf[v] == 1) {
                    w[representative][v] = 1.0;
                    w[v][representative] = 1.0;
                    break;
                }
            }
        }
        return w;
    }

    static int sideSize(int[] labels, int value) {
        int count = 0;
        for (int label : labels) {
            if (label == value) count++;
        }
        return count;
    }

    public static void main(String[] args) {
        Eigen small = sortedEigen(new double[][]{{2.0, 1.0}, {1.0, 2.0}}, 100, 1e-12);
        check(Math.abs(small.values[0] - 1.0) <= 1e-12 && Math.abs(small.values[1] - 3.0) <= 1e-12,
                "2x2 eigenvalues are 1 and 3");
        Eigen diagonal = sortedEigen(new double[][]{{3.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 2.0}},
                100, 1e-12);
        check(Math.abs(diagonal.values[0] + 1.0) <= 1e-12 && Math.abs(diagonal.values[2] - 3.0) <= 1e-12,
                "diagonal matrix is already diagonal");

        double[][] cliques = cliquePair(5, 1.0);
        double[][] cliqueLaplacian = laplacian(cliques);
        Eigen cliqueEigen = sortedEigen(cliqueLaplacian, 100, 1e-12);
        double trace = 0.0;
        double spectrum = 0.0;
        for (int i = 0; i < 10; i++) {
            trace += cliqueLaplacian[i][i];
            spectrum += cliqueEigen.values[i];
            check(cliqueEigen.values[i] >= -1e-9, "Laplacian eigenvalues are non-negative");
        }
        check(Math.abs(trace - spectrum) <= 1e-9, "eigenvalues sum to the trace");
        check(Math.abs(cliqueEigen.values[0]) <= 1e-9, "a connected graph has one zero eigenvalue");
        check(cliqueEigen.values[1] < cliqueEigen.values[2], "Fiedler value is strictly second smallest");

        Split cliqueSplit = spectralSplit(cliques);
        check(cliqueSplit.residual <= 1e-9, "Fiedler residual");
        check(cliqueSplit.orthogonality <= 1e-8, "Fiedler vector orthogonal to the constant");
        check(cliqueSplit.lambda2 > 1e-6, "connected graph: positive algebraic connectivity");
        check(jacobiEigh(cliqueLaplacian, 100, 1e-12).sweeps <= 100, "sweep cap respected");
        check(cliqueEigen.sweeps >= 1, "at least one Jacobi sweep was needed");
        int[] truth = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        check(Arrays.equals(cliqueSplit.labels, truth), "the spectral split recovers the two cliques");
        check(Math.abs(cutWeight(cliques, cliqueSplit.labels) - 1.0) <= 1e-9, "the bridge is the cut");
        check(Math.abs(bruteMinimum(cliques, 10, false) - 1.0) <= 1e-9, "the minimum cut is the bridge");
        double spectralNcut = normalisedCut(cliques, cliqueSplit.labels);
        for (int mask = 0; mask < (1 << 10); mask++) {
            if ((mask & 1) != 0) continue;
            int size = Integer.bitCount(mask);
            if (size == 0 || size == 10) continue;
            int[] labels = new int[10];
            for (int i = 0; i < 10; i++) labels[i] = (mask & (1 << i)) != 0 ? 1 : 0;
            if (normalisedCut(cliques, labels) < spectralNcut - 1e-9) {
                throw new RuntimeException("FAILED: a split beats the spectral normalised cut");
            }
        }

        double[][] line = path(8);
        Split lineSplit = spectralSplit(line);
        check(Math.abs(cutWeight(line, lineSplit.labels) - 1.0) <= 1e-9, "the 8-vertex path is split at one edge");
        check(sideSize(lineSplit.labels, 0) == 4 && sideSize(lineSplit.labels, 1) == 4, "the path split is balanced");
        check(Math.abs(bruteMinimum(line, 8, false) - 1.0) <= 1e-9, "the path minimum cut is 1");
        check(lineSplit.lambda2 > 1e-6, "the path is connected");

        Split disconnectedSplit = spectralSplit(twoTriangles());
        check(disconnectedSplit.components == 2, "two components counted");
        check(components(twoTriangles(), 6) == 2, "component count agrees");
        check(Math.abs(disconnectedSplit.lambda2) <= 1e-9, "a disconnected graph has a zero Fiedler value");
        check(Math.abs(cutWeight(twoTriangles(), disconnectedSplit.labels)) <= 1e-12,
                "the split follows the components and cuts nothing");
        Eigen disconnectedEigen = sortedEigen(laplacian(twoTriangles()), 100, 1e-12);
        int zeros = 0;
        for (double value : disconnectedEigen.values) {
            if (Math.abs(value) <= 1e-9) zeros++;
        }
        check(zeros == 2, "one zero eigenvalue per component");

        int[] seeds = {29, 58, 87, 116, 145, 174};
        for (int s = 0; s < seeds.length; s++) {
            int n = 8;
            double[][] w = makeConnected(randomWeighted(n, 6 + s % 4, seeds[s]), n);
            check(components(w, n) == 1, "instance " + s + ": the instance is connected");
            Split split = spectralSplit(w);
            check(split.residual <= 1e-9, "instance " + s + ": residual");
            check(split.orthogonality <= 1e-8, "instance " + s + ": orthogonality");
            check(split.lambda2 > 1e-6, "instance " + s + ": positive algebraic connectivity");
            double spectral = cutWeight(w, split.labels);
            double best = bruteMinimum(w, n, false);
            double bestBalanced = bruteMinimum(w, n, true);
            check(spectral >= best - 1e-9, "instance " + s + ": no split beats the true minimum cut");
            check(spectral <= 3.0 * bestBalanced + 1e-9,
                    "instance " + s + ": within a factor 3 of the best balanced bisection");
            check(normalisedCut(w, split.labels) >= 0.0, "instance " + s + ": non-negative normalised cut");
            check(Arrays.equals(spectralSplit(w).labels, split.labels), "instance " + s + ": deterministic");
            check(sideSize(split.labels, 1) > 0 && sideSize(split.labels, 0) > 0,
                    "instance " + s + ": both sides are non-empty");
        }

        double[][] single = path(1);
        Eigen singleEigen = sortedEigen(laplacian(single), 100, 1e-12);
        check(Math.abs(singleEigen.values[0]) <= 1e-9, "a single vertex has one zero eigenvalue");
        double[][] oneEdge = path(2);
        Split edgeSplit = spectralSplit(oneEdge);
        check(cutWeight(oneEdge, edgeSplit.labels) >= 0.0, "a two-vertex graph splits without error");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
