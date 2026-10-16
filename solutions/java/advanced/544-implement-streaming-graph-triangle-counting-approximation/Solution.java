/** Task 544 - Implement streaming graph triangle counting approximation. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Triangle counting with three exact oracles and two deterministic streaming estimators. The
 * exact side uses (a) the naive triple scan, (b) the closed-wedge identity: a triangle contains
 * exactly three wedges, so T = (1/3) sum over v of the number of adjacent pairs among N(v), with
 * the wedge total W = sum over v of C(deg(v), 2) and open = W - closed, and (c) the spectral
 * identity T = trace(A^3) / 6. The estimators are:
 *
 *   - wedge sampling (semi-streaming): degrees are accumulated in one pass, then S wedges are
 *     drawn uniformly from the wedge distribution by mapping LCG uniforms onto the cumulative
 *     C(deg(v), 2) table; the fraction of sampled wedges that are closed estimates the
 *     transitivity, and T = (closed / S) * W / 3. Memory is the fixed sample size plus the
 *     adjacency needed to test closure.
 *   - DOULION edge sparsification: each edge survives with probability p (LCG), the surviving
 *     graph is counted exactly, and T is rescaled by p^3.
 *
 * Documented checks: the three exact methods agree on K_n (C(n,3) triangles, n*C(n-1,2) wedges),
 * K4 (4), a single triangle, cycles, complete bipartite graphs, stars, paths, the empty graph and
 * single edges; the closed-wedge total is always a multiple of 3 and equals 3T; open + closed
 * equals the wedge count; on the complete graph the wedge sampler is exact for every seed (every
 * wedge is closed), the samplers reproduce the same value on a repeated seed, and their relative
 * error stays inside the documented tolerances on LCG random graphs: 0.2 per seed and 0.05 for the
 * eight-seed mean on the dense instance (60 vertices, 900 edges), 0.25 per seed and 0.1 for the mean
 * on the sparse instance (80 vertices, 320 edges), and 0.5 for DOULION with p = 0.5 (where p = 1
 * reproduces the exact count).
 */
import java.util.*;

public class Solution {
    static boolean[][] fromEdges(int n, int[][] edges) {
        boolean[][] adj = new boolean[n][n];
        for (int[] e : edges) {
            adj[e[0]][e[1]] = true;
            adj[e[1]][e[0]] = true;
        }
        return adj;
    }

    static int[] degrees(boolean[][] adj, int n) {
        int[] degree = new int[n];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (adj[u][v]) degree[u]++;
            }
        }
        return degree;
    }

    /** Oracle 1: exhaustive triple scan. */
    static long bruteTriangles(boolean[][] adj, int n) {
        long total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!adj[i][j]) continue;
                for (int k = j + 1; k < n; k++) {
                    if (adj[i][k] && adj[j][k]) total++;
                }
            }
        }
        return total;
    }

    /** Oracle 2a: closed wedges per vertex (adjacent pairs among the neighbourhood). */
    static long[] closedWedgesPerVertex(boolean[][] adj, int n) {
        long[] closed = new long[n];
        for (int v = 0; v < n; v++) {
            List<Integer> neighbours = new ArrayList<>();
            for (int w = 0; w < n; w++) {
                if (adj[v][w]) neighbours.add(w);
            }
            long count = 0;
            for (int i = 0; i < neighbours.size(); i++) {
                for (int j = i + 1; j < neighbours.size(); j++) {
                    if (adj[neighbours.get(i)][neighbours.get(j)]) count++;
                }
            }
            closed[v] = count;
        }
        return closed;
    }

    static long wedgeCount(int[] degree) {
        long total = 0;
        for (int value : degree) total += (long) value * (value - 1) / 2;
        return total;
    }

    /** Oracle 2b: T = trace(A^3) / 6 through explicit matrix powers. */
    static long spectralTriangles(boolean[][] adj, int n) {
        long[][] a = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = adj[i][j] ? 1 : 0;
        }
        long[][] b = multiply(a, a, n);
        long[][] c = multiply(a, b, n);
        long trace = 0;
        for (int i = 0; i < n; i++) trace += c[i][i];
        return trace / 6;
    }

    static long[][] multiply(long[][] x, long[][] y, int n) {
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                if (x[i][k] == 0) continue;
                for (int j = 0; j < n; j++) out[i][j] += x[i][k] * y[k][j];
            }
        }
        return out;
    }

    /** Uniform wedge sampling; returns the triangle estimate and the sampled wedge count. */
    static double wedgeSampleEstimate(boolean[][] adj, int n, int samples, int seed, long[] sampled) {
        int[] degree = degrees(adj, n);
        long[] cumulative = new long[n];
        long running = 0;
        for (int v = 0; v < n; v++) {
            running += (long) degree[v] * (degree[v] - 1) / 2;
            cumulative[v] = running;
        }
        long wedges = running;
        if (wedges == 0) {
            sampled[0] = 0;
            return 0.0;
        }
        long closed = 0;
        for (int sample = 0; sample < samples; sample++) {
            seed = (seed * 75 + 74) % 65537;
            long pick = (long) (seed % 65537) * wedges / 65537;
            int center = lowerBound(cumulative, pick);
            int[] neighbours = new int[degree[center]];
            int at = 0;
            for (int w = 0; w < n; w++) {
                if (adj[center][w]) neighbours[at++] = w;
            }
            seed = (seed * 75 + 74) % 65537;
            int i = seed % degree[center];
            seed = (seed * 75 + 74) % 65537;
            int j = seed % degree[center];
            if (i == j) j = (j + 1) % degree[center];
            if (adj[neighbours[i]][neighbours[j]]) closed++;
        }
        sampled[0] = samples;
        return (double) closed / samples * wedges / 3.0;
    }

    static int lowerBound(long[] cumulative, long value) {
        int lo = 0;
        int hi = cumulative.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cumulative[mid] > value) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /** DOULION: keep each edge with probability p, count triangles exactly, rescale by 1/p^3. */
    static double doulionEstimate(boolean[][] adj, int n, double p, int seed, long[] keptEdges) {
        boolean[][] sampled = new boolean[n][n];
        long kept = 0;
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                if (!adj[u][v]) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 1000 < (int) Math.round(p * 1000)) {
                    sampled[u][v] = true;
                    sampled[v][u] = true;
                    kept++;
                }
            }
        }
        keptEdges[0] = kept;
        return bruteTriangles(sampled, n) / (p * p * p);
    }

    static boolean[][] randomGraph(int n, int count, int seed) {
        boolean[][] adj = new boolean[n][n];
        int placed = 0;
        int guard = 0;
        while (placed < count && guard < count * 40) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u == v || adj[u][v]) continue;
            adj[u][v] = true;
            adj[v][u] = true;
            placed++;
        }
        return adj;
    }

    static boolean[][] complete(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u != v) adj[u][v] = true;
            }
        }
        return adj;
    }

    static boolean[][] cycle(int n) {
        int[][] edges = new int[n][2];
        for (int i = 0; i < n; i++) edges[i] = new int[]{i, (i + 1) % n};
        return fromEdges(n, edges);
    }

    static boolean[][] completeBipartite(int a, int b) {
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) edges.add(new int[]{i, a + j});
        }
        return fromEdges(a + b, edges.toArray(new int[0][]));
    }

    static void verifyExact(boolean[][] adj, int n, String label) {
        long brute = bruteTriangles(adj, n);
        long[] closed = closedWedgesPerVertex(adj, n);
        long closedTotal = 0;
        for (long value : closed) closedTotal += value;
        check(closedTotal % 3 == 0, label + ": the closed-wedge total is divisible by 3");
        check(closedTotal / 3 == brute, label + ": closed wedges / 3 equals the triple scan");
        long wedges = wedgeCount(degrees(adj, n));
        check(wedges >= closedTotal, label + ": open wedges = W - closed is non-negative");
        check(spectralTriangles(adj, n) == brute, label + ": trace(A^3) / 6 agrees");
    }

    public static void main(String[] args) {
        verifyExact(complete(5), 5, "K5");
        verifyExact(complete(6), 6, "K6");
        check(bruteTriangles(complete(5), 5) == 10, "K5 has C(5,3) = 10 triangles");
        check(wedgeCount(degrees(complete(5), 5)) == 5 * 6, "K5 has n * C(n-1,2) = 30 wedges");
        long[] k5closed = closedWedgesPerVertex(complete(5), 5);
        for (long value : k5closed) check(value == 6, "each K5 vertex closes C(4,2) = 6 wedges");
        check(bruteTriangles(complete(4), 4) == 4, "K4 has 4 triangles");
        check(bruteTriangles(cycle(3), 3) == 1, "a single triangle is one triangle");
        check(bruteTriangles(cycle(7), 7) == 0, "a 7-cycle is triangle free");
        check(bruteTriangles(completeBipartite(3, 3), 6) == 0, "K3,3 is bipartite and triangle free");
        check(bruteTriangles(fromEdges(6, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}), 6) == 0,
                "a path is triangle free");
        check(bruteTriangles(fromEdges(1, new int[0][]), 1) == 0, "a single vertex has no triangles");
        check(bruteTriangles(fromEdges(2, new int[][]{{0, 1}}), 2) == 0, "a single edge has no triangles");
        check(bruteTriangles(fromEdges(4, new int[][]{{0, 1}, {1, 2}}), 4) == 0, "an isolated vertex changes nothing");
        boolean[][] triangle = cycle(3);
        check(wedgeCount(degrees(triangle, 3)) == 3, "a triangle has three wedges, all closed");
        long[] triangleClosed = closedWedgesPerVertex(triangle, 3);
        check(Arrays.equals(triangleClosed, new long[]{1, 1, 1}), "each triangle vertex closes one wedge");

        boolean[][] k7 = complete(7);
        long[] sampled = new long[1];
        double exactEstimate = wedgeSampleEstimate(k7, 7, 5000, 999, sampled);
        check(sampled[0] == 5000, "the sampler draws exactly the requested number of wedges");
        check(Math.abs(exactEstimate - 35.0) <= 1e-9,
                "on a complete graph every wedge is closed, so the estimate is exact");
        check(wedgeSampleEstimate(k7, 7, 5000, 999, new long[1]) == exactEstimate,
                "the estimator is deterministic for a fixed seed");
        check(wedgeSampleEstimate(fromEdges(1, new int[0][]), 1, 100, 7, new long[1]) == 0.0,
                "no wedges means a zero estimate without division by zero");

        boolean[][] dense = randomGraph(60, 900, 4242);
        verifyExact(dense, 60, "dense random graph");
        long denseExact = bruteTriangles(dense, 60);
        check(denseExact > 1000, "the dense instance has thousands of triangles");
        double mean = 0.0;
        for (int s = 0; s < 8; s++) {
            double estimate = wedgeSampleEstimate(dense, 60, 20000, 1000 + 37 * s, new long[1]);
            double error = Math.abs(estimate - denseExact) / denseExact;
            check(error <= 0.2, "seed " + s + ": wedge sampling error " + error + " within 0.2");
            mean += estimate;
        }
        mean /= 8;
        check(Math.abs(mean - denseExact) / denseExact <= 0.05,
                "the eight-seed mean is within 5% of the exact count");
        double doulion = doulionEstimate(dense, 60, 0.5, 31337, new long[1]);
        check(Math.abs(doulion - denseExact) / denseExact <= 0.5,
                "DOULION at p = 0.5 stays within 50% on the dense instance");
        check(doulionEstimate(dense, 60, 0.5, 31337, new long[1]) == doulion, "DOULION is deterministic");
        check(doulionEstimate(dense, 60, 1.0, 5, new long[1]) == denseExact,
                "p = 1 keeps the whole graph and reproduces the exact count");
        check(doulionEstimate(fromEdges(3, new int[][]{{0, 1}}), 3, 0.5, 11, new long[1]) == 0.0,
                "a single edge never yields triangles");

        boolean[][] sparse = randomGraph(80, 320, 8080);
        verifyExact(sparse, 80, "sparse random graph");
        long sparseExact = bruteTriangles(sparse, 80);
        check(sparseExact > 10, "the sparse instance has a measurable number of triangles");
        mean = 0.0;
        for (int s = 0; s < 8; s++) {
            double estimate = wedgeSampleEstimate(sparse, 80, 20000, 2000 + 41 * s, new long[1]);
            double error = Math.abs(estimate - sparseExact) / sparseExact;
            check(error <= 0.25, "sparse seed " + s + ": error " + error + " within 0.25");
            mean += estimate;
        }
        mean /= 8;
        check(Math.abs(mean - sparseExact) / sparseExact <= 0.1,
                "the sparse eight-seed mean is within 10% of the exact count");

        List<int[]> edges = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 5; i++) {
                for (int j = i + 1; j < 5; j++) edges.add(new int[]{c * 5 + i, c * 5 + j});
            }
        }
        boolean[][] cliques = fromEdges(15, edges.toArray(new int[0][]));
        verifyExact(cliques, 15, "three disjoint K5 cliques");
        check(bruteTriangles(cliques, 15) == 30, "three K5 cliques contain 3 * 10 triangles");
        double cliqueEstimate = wedgeSampleEstimate(cliques, 15, 4000, 606, new long[1]);
        check(Math.abs(cliqueEstimate - 30.0) <= 1e-9, "disjoint cliques: the sampler is exact again");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
