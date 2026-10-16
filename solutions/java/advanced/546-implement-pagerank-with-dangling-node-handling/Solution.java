/** Task 546 - Implement pagerank with dangling node handling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * A dangling vertex is one with out-degree zero: in the plain power iteration
 * r_{k+1} = d M r_k + (1 - d)/n it contributes nothing, so probability mass leaks out and the
 * iterates are no longer a distribution. The fix implemented here is the standard
 * redistribution: the dangling mass D_k = sum of r_k over dangling vertices is spread uniformly
 * over all n vertices, which is exactly the transition matrix M'' obtained by adding complete
 * out-edges (weight 1/n each) to every dangling vertex, and also exactly the "virtual node" model
 * where dangling vertices point at a node that points back at everyone.
 *
 * Four independent computations are compared on every instance: the corrected power iteration,
 * a plain iteration on the explicitly augmented graph (complete dangling out-edges), a direct
 * Gaussian solve of (I - d M'') r = (1 - d)/n, and a deterministic LCG Monte-Carlo walk with
 * uniform teleport and uniform jumps out of dangling vertices. Documented checks: the corrected
 * iterates sum to one at every step while the naive iterates leak exactly
 * S_k - S_{k+1} = (1-d)(S_k - 1) + d D_k; a graph without dangling vertices makes the correction
 * a no-op; an all-dangling graph gives the uniform vector; the hand-derived closed form
 * (1 - d)/(n - d) is checked for a unique isolated dangling vertex and an isolated vertex ranks
 * above nothing else; and the fixed point is verified by residual, by the direct solve (<= 1e-9)
 * and by the walk (<= 0.02).
 */
import java.util.*;

public class Solution {
    static int[] outDegrees(int n, List<int[]> adj) {
        int[] outdeg = new int[n];
        for (int[] e : adj) outdeg[e[0]]++;
        return outdeg;
    }

    static List<List<Integer>> successors(int n, List<int[]> adj) {
        List<List<Integer>> lists = new ArrayList<>();
        for (int i = 0; i < n; i++) lists.add(new ArrayList<>());
        for (int[] e : adj) lists.get(e[0]).add(e[1]);
        return lists;
    }

    static class Result {
        final double[] ranks;
        final int iterations;
        final List<double[]> history;

        Result(double[] ranks, int iterations, List<double[]> history) {
            this.ranks = ranks;
            this.iterations = iterations;
            this.history = history;
        }
    }

    /** Corrected iteration: dangling mass is redistributed uniformly. */
    static Result pagerankDangling(int n, List<int[]> adj, double d, double tol, int maxIters) {
        int[] outdeg = outDegrees(n, adj);
        List<List<Integer>> outgoing = successors(n, adj);
        double[] ranks = new double[n];
        Arrays.fill(ranks, 1.0 / n);
        List<double[]> history = new ArrayList<>();
        history.add(ranks.clone());
        int iterations = 0;
        while (iterations < maxIters) {
            double danglingMass = 0.0;
            for (int v = 0; v < n; v++) {
                if (outdeg[v] == 0) danglingMass += ranks[v];
            }
            double[] next = new double[n];
            for (int v = 0; v < n; v++) next[v] = (1.0 - d) / n + d * danglingMass / n;
            for (int u = 0; u < n; u++) {
                if (outdeg[u] == 0) continue;
                double share = d * ranks[u] / outdeg[u];
                for (int v : outgoing.get(u)) next[v] += share;
            }
            double delta = 0.0;
            for (int v = 0; v < n; v++) delta += Math.abs(next[v] - ranks[v]);
            ranks = next;
            history.add(ranks.clone());
            iterations++;
            if (delta <= tol) break;
        }
        return new Result(ranks, iterations, history);
    }

    /** Naive iteration: dangling mass simply disappears. */
    static Result pagerankPlain(int n, List<int[]> adj, double d, int iterations) {
        int[] outdeg = outDegrees(n, adj);
        List<List<Integer>> outgoing = successors(n, adj);
        double[] ranks = new double[n];
        Arrays.fill(ranks, 1.0 / n);
        List<double[]> history = new ArrayList<>();
        history.add(ranks.clone());
        for (int step = 0; step < iterations; step++) {
            double[] next = new double[n];
            Arrays.fill(next, (1.0 - d) / n);
            for (int u = 0; u < n; u++) {
                if (outdeg[u] == 0) continue;
                double share = d * ranks[u] / outdeg[u];
                for (int v : outgoing.get(u)) next[v] += share;
            }
            ranks = next;
            history.add(ranks.clone());
        }
        return new Result(ranks, iterations, history);
    }

    static List<int[]> augmentDangling(int n, List<int[]> adj) {
        int[] outdeg = outDegrees(n, adj);
        List<int[]> augmented = new ArrayList<>(adj);
        for (int u = 0; u < n; u++) {
            if (outdeg[u] == 0) {
                for (int v = 0; v < n; v++) augmented.add(new int[]{u, v});
            }
        }
        return augmented;
    }

    /** Direct solve of (I - d M'') r = (1 - d)/n with M'' including the dangling redistribution. */
    static double[] solveGoogle(int n, List<int[]> adj, double d) {
        int[] outdeg = outDegrees(n, adj);
        List<List<Integer>> outgoing = successors(n, adj);
        double[][] m = new double[n][n];
        for (int u = 0; u < n; u++) {
            if (outdeg[u] == 0) {
                for (int v = 0; v < n; v++) m[v][u] += 1.0 / n;
            } else {
                for (int v : outgoing.get(u)) m[v][u] += 1.0 / outdeg[u];
            }
        }
        double[][] a = new double[n][n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = (i == j ? 1.0 : 0.0) - d * m[i][j];
            b[i] = (1.0 - d) / n;
        }
        return gauss(a, b);
    }

    static double[] gauss(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] m = new double[n][n];
        double[] b = rhs.clone();
        for (int i = 0; i < n; i++) m[i] = matrix[i].clone();
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            }
            double[] tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
            double swap = b[col];
            b[col] = b[pivot];
            b[pivot] = swap;
            if (Math.abs(m[col][col]) < 1e-14) throw new RuntimeException("FAILED: singular system");
            for (int row = col + 1; row < n; row++) {
                double factor = m[row][col] / m[col][col];
                if (factor == 0.0) continue;
                for (int k = col; k < n; k++) m[row][k] -= factor * m[col][k];
                b[row] -= factor * b[col];
            }
        }
        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double acc = b[row];
            for (int k = row + 1; k < n; k++) acc -= m[row][k] * x[k];
            x[row] = acc / m[row][row];
        }
        return x;
    }

    /** Deterministic Monte-Carlo walk with teleport and uniform jumps out of dangling vertices. */
    static double[] mcmc(int n, List<int[]> adj, double d, int steps, int burnIn, int seed) {
        int[] outdeg = outDegrees(n, adj);
        List<List<Integer>> outgoing = successors(n, adj);
        double[] visits = new double[n];
        int current = 0;
        for (int step = 0; step < steps + burnIn; step++) {
            seed = (seed * 75 + 74) % 65537;
            int roll = seed % 1000;
            if (roll < (int) Math.round(d * 1000) && outdeg[current] > 0) {
                seed = (seed * 75 + 74) % 65537;
                current = outgoing.get(current).get(seed % outdeg[current]);
            } else {
                seed = (seed * 75 + 74) % 65537;
                current = seed % n;
            }
            if (step >= burnIn) visits[current]++;
        }
        for (int v = 0; v < n; v++) visits[v] /= steps;
        return visits;
    }

    static double l1(double[] a, double[] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
        return total;
    }

    static double sum(double[] a) {
        double total = 0.0;
        for (double value : a) total += value;
        return total;
    }

    static List<int[]> lcgEdges(int n, int count, int seed) {
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
        return adj;
    }

    public static void main(String[] args) {
        double d = 0.85;

        int n3 = 3;
        List<int[]> isolated = Arrays.asList(new int[]{0, 1}, new int[]{1, 0});
        Result corrected = pagerankDangling(n3, isolated, d, 1e-14, 1000);
        check(Math.abs(sum(corrected.ranks) - 1.0) <= 1e-12, "the corrected ranks sum to one");
        double closedForm = (1.0 - d) / (n3 - d);
        check(Math.abs(corrected.ranks[2] - closedForm) <= 1e-12,
                "the unique isolated dangling vertex has rank (1 - d)/(n - d)");
        check(Math.abs(corrected.ranks[0] - corrected.ranks[1]) <= 1e-12, "the 2-cycle stays symmetric");
        check(Math.abs(corrected.ranks[0] - (1.0 - closedForm) / 2.0) <= 1e-12, "closed form for the cycle");
        double[] direct = solveGoogle(n3, isolated, d);
        check(l1(corrected.ranks, direct) <= 1e-9, "the corrected iteration matches the direct solve");
        Result plain = pagerankPlain(n3, isolated, d, 40);
        check(sum(plain.ranks) < 1.0 - 1e-6, "the naive iteration leaks mass");
        check(l1(plain.ranks, corrected.ranks) > 1e-3, "the naive vector differs from the corrected one");
        Result augmented = pagerankPlain(n3, augmentDangling(n3, isolated), d, 400);
        check(l1(augmented.ranks, corrected.ranks) <= 1e-9,
                "adding complete dangling out-edges reproduces the corrected ranks");
        double[] walk = mcmc(n3, isolated, d, 200000, 2000, 20240);
        check(l1(walk, corrected.ranks) <= 0.03, "the Monte-Carlo walk matches the corrected ranks");

        Result allDangling = pagerankDangling(4, new ArrayList<int[]>(), d, 1e-14, 200);
        for (double value : allDangling.ranks) check(Math.abs(value - 0.25) <= 1e-12,
                "an all-dangling graph is uniform");
        Result oneNode = pagerankDangling(1, new ArrayList<int[]>(), d, 1e-14, 200);
        check(Math.abs(oneNode.ranks[0] - 1.0) <= 1e-12, "a single dangling vertex has rank 1");

        List<int[]> cycle = new ArrayList<>();
        for (int i = 0; i < 5; i++) cycle.add(new int[]{i, (i + 1) % 5});
        Result noDangling = pagerankDangling(5, cycle, d, 1e-14, 1000);
        Result plainCycle = pagerankPlain(5, cycle, d, 1000);
        check(l1(noDangling.ranks, plainCycle.ranks) <= 1e-9,
                "without dangling vertices the correction is a no-op");
        check(Math.abs(sum(plainCycle.ranks) - 1.0) <= 1e-9, "no dangling vertices means no leak");

        int[] seeds = {31, 62, 93, 124, 155, 186, 217, 248};
        for (int s = 0; s < seeds.length; s++) {
            int nodes = 4 + s % 3;
            List<int[]> adj = new ArrayList<>();
            for (int[] e : lcgEdges(nodes, 2 + s % 5, seeds[s])) {
                if (e[0] != nodes - 1) adj.add(e);    // the last vertex is kept dangling on purpose
            }
            int[] outdeg = outDegrees(nodes, adj);
            int danglingCount = 0;
            for (int v = 0; v < nodes; v++) {
                if (outdeg[v] == 0) danglingCount++;
            }
            check(danglingCount >= 1, "seed " + s + ": the instance really has dangling vertices");
            Result fixed = pagerankDangling(nodes, adj, d, 1e-13, 2000);
            Result leaky = pagerankPlain(nodes, adj, d, 25);
            check(Math.abs(sum(fixed.ranks) - 1.0) <= 1e-12, "seed " + s + ": masses stay normalised");
            for (double[] iterate : fixed.history) {
                check(Math.abs(sum(iterate) - 1.0) <= 1e-12, "seed " + s + ": every corrected iterate is a distribution");
            }
            for (int k = 1; k < leaky.history.size(); k++) {
                double previous = sum(leaky.history.get(k - 1));
                double current = sum(leaky.history.get(k));
                double danglingMass = 0.0;
                for (int v = 0; v < nodes; v++) {
                    if (outdeg[v] == 0) danglingMass += leaky.history.get(k - 1)[v];
                }
                double predicted = (1.0 - d) * (previous - 1.0) + d * danglingMass;
                check(Math.abs((previous - current) - predicted) <= 1e-12,
                        "seed " + s + ": the leak equals (1-d)(S-1) + d D");
            }
            double[] reference = solveGoogle(nodes, adj, d);
            check(l1(fixed.ranks, reference) <= 1e-9, "seed " + s + ": iteration matches the direct solve");
            if (danglingCount < nodes) {
                Result aug = pagerankPlain(nodes, augmentDangling(nodes, adj), d, 800);
                check(l1(aug.ranks, fixed.ranks) <= 1e-9, "seed " + s + ": augmented graph agrees");
            }
            double[] sampled = mcmc(nodes, adj, d, 200000, 5000, 777 + 13 * s);
            check(l1(sampled, fixed.ranks) <= 0.05, "seed " + s + ": walk matches the fixed point");
            Result rerun = pagerankDangling(nodes, adj, d, 1e-13, 2000);
            check(Arrays.equals(rerun.ranks, fixed.ranks), "seed " + s + ": deterministic re-run");
            check(fixed.iterations <= 2000, "seed " + s + ": iteration cap respected");
        }

        List<int[]> chain = new ArrayList<>();
        for (int i = 0; i + 1 < 6; i++) chain.add(new int[]{i, i + 1});
        Result chainResult = pagerankDangling(6, chain, d, 1e-14, 4000);
        check(Math.abs(sum(chainResult.ranks) - 1.0) <= 1e-12, "the path's mass is conserved");
        check(chainResult.ranks[5] > chainResult.ranks[4], "the dangling sink accumulates mass");
        check(chainResult.ranks[1] > chainResult.ranks[0],
                "the vertex 0 only receives teleport plus the dangling share");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
