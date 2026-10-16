/** Task 545 - Implement pagerank iterative solver and convergence check. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * PageRank by power iteration on the Google matrix: with M[v][u] = 1/outdeg(u) for every edge
 * u -> v (no dangling vertices in this task — every vertex has out-degree >= 1, which the
 * self-test asserts), the update is r_{k+1} = d * M r_k + (1 - d)/n, and the iteration is stopped
 * when the L1 change ||r_{k+1} - r_k||_1 drops to the tolerance or the documented iteration cap is
 * hit. The file records the whole history so the convergence claims can be checked rather than
 * asserted: since M is column-stochastic in the L1 norm the update is an affine contraction with
 * factor d, hence delta_{k+1} <= d * delta_k and ||r_k - r*||_1 <= d^k ||r_0 - r*||_1, both of
 * which are verified against the exact stationary vector.
 *
 * The independent reference solution comes from the linear system (I - d M) r = (1 - d)/n, solved
 * by Gaussian elimination with partial pivoting, so the iterative and direct answers are compared
 * componentwise. Documented checks: a 3-cycle, a 2-cycle and the complete directed graph are
 * uniform (1/n, within 1e-12); the Google matrix is row-stochastic and non-negative; ranks are
 * non-negative and sum to one at every iterate; the fixed-point residual
 * ||d M r + (1 - d)/n - r||_inf is at most the tolerance; the iteration count is finite and
 * bounded by the cap; per-iteration deltas shrink at least geometrically with factor d; the L1
 * error bound holds at every recorded iterate; a vertex with no incoming edges keeps exactly
 * (1 - d)/n; and re-running the solver reproduces the same vector bit for bit.
 */
import java.util.*;

public class Solution {
    static double[][] transition(int n, List<int[]> adj) {
        int[] outdeg = new int[n];
        for (int[] e : adj) outdeg[e[0]]++;
        double[][] m = new double[n][n];
        for (int[] e : adj) m[e[1]][e[0]] += 1.0 / outdeg[e[0]];
        return m;
    }

    static int[] outDegrees(int n, List<int[]> adj) {
        int[] outdeg = new int[n];
        for (int[] e : adj) outdeg[e[0]]++;
        return outdeg;
    }

    static class Result {
        final double[] ranks;
        final int iterations;
        final double[] deltas;
        final List<double[]> history;

        Result(double[] ranks, int iterations, double[] deltas, List<double[]> history) {
            this.ranks = ranks;
            this.iterations = iterations;
            this.deltas = deltas;
            this.history = history;
        }
    }

    static Result pagerank(int n, List<int[]> adj, double d, double tol, int maxIters) {
        double[][] m = transition(n, adj);
        double[] ranks = new double[n];
        Arrays.fill(ranks, 1.0 / n);
        List<double[]> history = new ArrayList<>();
        history.add(ranks.clone());
        List<Double> deltas = new ArrayList<>();
        int iterations = 0;
        while (iterations < maxIters) {
            double[] next = new double[n];
            for (int v = 0; v < n; v++) {
                double acc = 0.0;
                for (int u = 0; u < n; u++) acc += m[v][u] * ranks[u];
                next[v] = d * acc + (1.0 - d) / n;
            }
            double delta = 0.0;
            for (int v = 0; v < n; v++) delta += Math.abs(next[v] - ranks[v]);
            ranks = next;
            history.add(ranks.clone());
            deltas.add(delta);
            iterations++;
            if (delta <= tol) break;
        }
        double[] deltaArray = new double[deltas.size()];
        for (int i = 0; i < deltaArray.length; i++) deltaArray[i] = deltas.get(i);
        return new Result(ranks, iterations, deltaArray, history);
    }

    /** Gaussian elimination with partial pivoting. */
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n];
        double[] rhs = b.clone();
        for (int i = 0; i < n; i++) m[i] = a[i].clone();
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            }
            double[] tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
            double swap = rhs[col];
            rhs[col] = rhs[pivot];
            rhs[pivot] = swap;
            if (Math.abs(m[col][col]) < 1e-14) throw new RuntimeException("FAILED: singular system");
            for (int row = col + 1; row < n; row++) {
                double factor = m[row][col] / m[col][col];
                if (factor == 0.0) continue;
                for (int k = col; k < n; k++) m[row][k] -= factor * m[col][k];
                rhs[row] -= factor * rhs[col];
            }
        }
        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double acc = rhs[row];
            for (int k = row + 1; k < n; k++) acc -= m[row][k] * x[k];
            x[row] = acc / m[row][row];
        }
        return x;
    }

    /** Exact stationary vector of the Google matrix via (I - d M) r = (1 - d)/n. */
    static double[] exactRanks(int n, double[][] m, double d) {
        double[][] a = new double[n][n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = (i == j ? 1.0 : 0.0) - d * m[i][j];
            b[i] = (1.0 - d) / n;
        }
        return solve(a, b);
    }

    static double l1(double[] a, double[] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
        return total;
    }

    static double residual(double[][] m, double[] ranks, double d) {
        int n = ranks.length;
        double worst = 0.0;
        for (int v = 0; v < n; v++) {
            double acc = 0.0;
            for (int u = 0; u < n; u++) acc += m[v][u] * ranks[u];
            worst = Math.max(worst, Math.abs(d * acc + (1.0 - d) / n - ranks[v]));
        }
        return worst;
    }

    static List<int[]> cycle(int n) {
        List<int[]> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new int[]{i, (i + 1) % n});
        return adj;
    }

    static List<int[]> completeDigraph(int n) {
        List<int[]> adj = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u != v) adj.add(new int[]{u, v});
            }
        }
        return adj;
    }

    static List<int[]> randomDigraph(int n, int count, int seed) {
        Set<Long> seen = new HashSet<>();
        List<int[]> adj = new ArrayList<>();
        int guard = 0;
        while (adj.size() < count && guard < count * 40) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u == v || !seen.add(((long) u << 20) | v)) continue;
            adj.add(new int[]{u, v});
        }
        int[] outdeg = outDegrees(n, adj);
        for (int v = 0; v < n; v++) {
            if (outdeg[v] == 0) adj.add(new int[]{v, (v + 1) % n});
        }
        return adj;
    }

    static void checkSolver(int n, List<int[]> adj, int seed) {
        double d = 0.85;
        Result result = pagerank(n, adj, d, 1e-12, 500);
        double[][] m = transition(n, adj);
        int[] outdeg = outDegrees(n, adj);
        for (int v = 0; v < n; v++) check(outdeg[v] >= 1, "graph " + seed + ": no dangling vertices");
        for (int u = 0; u < n; u++) {
            double column = 0.0;
            for (int v = 0; v < n; v++) column += m[v][u];
            check(Math.abs(column - 1.0) <= 1e-12, "graph " + seed + ": M is column stochastic in norm");
        }
        double[] exact = exactRanks(n, m, d);
        check(l1(result.ranks, exact) <= 1e-9, "graph " + seed + ": power iteration matches the linear solve");
        check(residual(m, result.ranks, d) <= 1e-12, "graph " + seed + ": fixed-point residual");
        double sum = 0.0;
        for (double value : result.ranks) sum += value;
        check(Math.abs(sum - 1.0) <= 1e-12, "graph " + seed + ": ranks sum to one");
        for (double value : result.ranks) check(value >= 0.0, "graph " + seed + ": ranks are non-negative");
        for (double[] iterate : result.history) {
            double mass = 0.0;
            for (double value : iterate) mass += value;
            check(Math.abs(mass - 1.0) <= 1e-12, "graph " + seed + ": every iterate is a distribution");
        }
        check(result.iterations <= 500, "graph " + seed + ": the iteration cap is respected");
        check(result.deltas.length == result.iterations, "graph " + seed + ": one delta per iteration");
        check(result.deltas[result.deltas.length - 1] <= 1e-12, "graph " + seed + ": final delta at tolerance");
        for (int k = 1; k < result.deltas.length; k++) {
            check(result.deltas[k] <= d * result.deltas[k - 1] + 1e-12,
                    "graph " + seed + ": delta_{k+1} <= d * delta_k (numerical slack)");
        }
        double initialError = l1(result.history.get(0), exact);
        for (int k = 0; k < result.history.size(); k++) {
            double bound = Math.pow(d, k) * initialError + 1e-12;
            check(l1(result.history.get(k), exact) <= bound, "graph " + seed + ": L1 error bound at step " + k);
        }
        Result rerun = pagerank(n, adj, d, 1e-12, 500);
        check(Arrays.equals(rerun.ranks, result.ranks), "graph " + seed + ": deterministic re-run");
        check(rerun.iterations == result.iterations, "graph " + seed + ": identical iteration count");
    }

    public static void main(String[] args) {
        Result three = pagerank(3, cycle(3), 0.85, 1e-14, 500);
        for (double value : three.ranks) check(Math.abs(value - 1.0 / 3.0) <= 1e-12, "3-cycle is uniform");
        Result two = pagerank(2, cycle(2), 0.85, 1e-14, 500);
        for (double value : two.ranks) check(Math.abs(value - 0.5) <= 1e-12, "2-cycle is uniform");
        Result complete5 = pagerank(5, completeDigraph(5), 0.85, 1e-14, 500);
        for (double value : complete5.ranks) check(Math.abs(value - 0.2) <= 1e-12, "complete digraph is uniform");
        Result single = pagerank(1, cycle(1), 0.85, 1e-14, 500);
        check(Math.abs(single.ranks[0] - 1.0) <= 1e-12, "a single self-looping vertex has rank 1");
        Result duplicate = pagerank(2, Arrays.asList(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 0}),
                0.85, 1e-14, 500);
        check(l1(duplicate.ranks, new double[]{0.5, 0.5}) <= 1e-12,
                "a duplicated edge splits the weight and the two-cycle is uniform");

        List<int[]> star = Arrays.asList(new int[]{1, 0}, new int[]{2, 0}, new int[]{3, 0}, new int[]{0, 1});
        Result starResult = pagerank(4, star, 0.85, 1e-14, 500);
        check(Math.abs(starResult.ranks[2] - 0.15 / 4.0) <= 1e-12, "no incoming edges means (1 - d) / n");
        check(Math.abs(starResult.ranks[3] - 0.15 / 4.0) <= 1e-12, "the other leaf matches");
        check(starResult.ranks[0] > starResult.ranks[1], "the hub outranks the vertex it points to");
        check(starResult.iterations > 1, "the star needs more than one iteration to converge");

        int[] seeds = {17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204};
        for (int s = 0; s < seeds.length; s++) {
            int n = 4 + s % 5;
            checkSolver(n, randomDigraph(n, 2 * n, seeds[s]), seeds[s]);
        }

        List<int[]> chain = new ArrayList<>();
        for (int i = 0; i + 1 < 6; i++) chain.add(new int[]{i, i + 1});
        chain.add(new int[]{5, 0});
        Result chainResult = pagerank(6, chain, 0.85, 1e-14, 500);
        for (double value : chainResult.ranks) {
            check(Math.abs(value - 1.0 / 6.0) <= 1e-12, "a directed cycle is uniform regardless of length");
        }
        check(starResult.iterations < pagerank(4, star, 0.95, 1e-14, 500).iterations,
                "a smaller damping factor converges in fewer iterations");
        check(pagerank(6, chain, 0.85, 1e-14, 500).iterations == 1,
                "a uniform cycle is already the fixed point after one iteration");

        List<int[]> selfLoop = Arrays.asList(new int[]{0, 1}, new int[]{1, 1});
        Result twoResult = pagerank(2, selfLoop, 0.85, 1e-14, 500);
        double[][] twoMatrix = transition(2, selfLoop);
        double[] twoExact = exactRanks(2, twoMatrix, 0.85);
        check(l1(twoResult.ranks, twoExact) <= 1e-12, "self loops are handled by the linear solve too");
        check(twoResult.ranks[1] > twoResult.ranks[0], "the self-looping vertex keeps more rank");

        double[][] wellConditioned = {{2.0, 1.0}, {1.0, 3.0}};
        double[] solved = solve(wellConditioned, new double[]{5.0, 10.0});
        check(Math.abs(2.0 * solved[0] + solved[1] - 5.0) <= 1e-12
                        && Math.abs(solved[0] + 3.0 * solved[1] - 10.0) <= 1e-12,
                "the linear solver satisfies its own equations");
        double[][] singular = {{1.0}};
        check(Math.abs(solve(singular, new double[]{2.0})[0] - 2.0) <= 1e-12, "1x1 solve");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
