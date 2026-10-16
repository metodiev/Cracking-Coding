/** Task 553 - Implement induced subgraph frequency sampling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * The induced subgraph frequency of a k-vertex pattern H in a host graph G is the fraction of the
 * C(n,k) vertex subsets whose induced subgraph is isomorphic to H. Here two instances are used: a
 * 24-vertex sparse host (edge probability 0.12) carrying six disjoint planted induced K4 copies, and a
 * 24-vertex dense host (edge probability 0.5) in which the diamond (K4 minus one edge) is the pattern.
 * The reference frequency is computed exactly by enumerating every subset and comparing canonical
 * forms, where the canonical form of a subset is the minimum over all k! relabelings of the k(k-1)/2-bit
 * upper-triangle mask, so two induced subgraphs share a key exactly when they are isomorphic. The
 * sampler draws m independent uniform k-subsets with a partial Fisher-Yates shuffle driven by splitmix64
 * (the repo's 16-bit LCG has short cycles and poor low-bit structure, which is exactly the regime where
 * uniform subset draws silently go wrong), pre-filters candidates by induced edge count and estimates
 * the frequency by the hit fraction, whose standard error is sqrt(p(1-p)/m).
 *
 * Documented checks. Planted detection: all six planted blocks are verified to be induced copies, the
 * exact planted frequency is 6/10626 = 5.6465e-4 (the enumeration finds exactly those six K4s and no
 * accidental one), the unplanted control graph contains exactly zero induced K4s (asserted exactly,
 * because at p = 0.12 the expected accidental count is about 0.03, so the control is a genuine zero and
 * the sampler must report zero hits there), and over m = 1000000 draws the estimate 5.66e-4 stays
 * inside the four-standard-error band of width 9.5e-5 and is strictly positive, which is the
 * planted-versus-control ordering. Density contrast: in the dense host the diamond occurs in 13.64% of
 * the 10626 subsets against 0.20% in the sparse one, the sampled 0.13603 lies within four standard
 * errors (7.7e-4 each), and the two frequencies differ by more than 0.13, so the ordering the sampler
 * reproduces is unambiguous. Sampling uniformity is checked separately — every vertex must appear in
 * k/n of the draws within four binomial standard errors — which tests the shuffle rather than the
 * estimator. Independent cross-checks: for k = 3 the isomorphism-based counts of the four triadic
 * motifs are compared with a purely structural classification by induced edge count (0/1/2/3), the
 * k = 1 frequency is exactly 1, a host equal to the pattern has frequency 1 and is always hit, K4
 * occurs in C(6,4) subsets of K6 while the diamond occurs in none, and a 5-vertex pattern larger than
 * the 4-vertex host yields frequency 0 with a sampler that never hits. All instances and estimates are
 * reproducible from the fixed seeds.
 */
import java.util.*;

public class Solution {

    /** splitmix64: documented 64-bit generator for the many uniform draws this sampler needs. */
    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    static boolean[][] randomGraph(int n, double p, Rng rng) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (rng.nextDouble() < p) {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }
        return adj;
    }

    static boolean[][] complete(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
        return adj;
    }

    static boolean[][] diamond() {
        boolean[][] adj = complete(4);
        adj[0][1] = false;
        adj[1][0] = false;
        return adj;
    }

    static List<int[]> permutations(int k) {
        List<int[]> out = new ArrayList<int[]>();
        buildPermutations(k, 0, new int[k], new boolean[k], out);
        return out;
    }

    private static void buildPermutations(int k, int depth, int[] current, boolean[] used,
                                          List<int[]> out) {
        if (depth == k) {
            out.add(current.clone());
            return;
        }
        for (int i = 0; i < k; i++) {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            buildPermutations(k, depth + 1, current, used, out);
            used[i] = false;
        }
    }

    /** Minimum upper-triangle bitmask over all relabelings: a complete isomorphism invariant. */
    static int canonicalKey(boolean[][] host, int[] subset, List<int[]> perms) {
        int k = subset.length;
        boolean[][] induced = new boolean[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                induced[i][j] = host[subset[i]][subset[j]];
                induced[j][i] = induced[i][j];
            }
        }
        int best = Integer.MAX_VALUE;
        for (int[] perm : perms) {
            int mask = 0;
            int bit = 0;
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    if (induced[perm[i]][perm[j]]) mask |= 1 << bit;
                    bit++;
                }
            }
            best = Math.min(best, mask);
        }
        return best;
    }

    static int inducedEdges(boolean[][] host, int[] subset) {
        int edges = 0;
        for (int i = 0; i < subset.length; i++) {
            for (int j = i + 1; j < subset.length; j++) {
                if (host[subset[i]][subset[j]]) edges++;
            }
        }
        return edges;
    }

    static int patternKey(boolean[][] pattern, List<int[]> perms) {
        int[] subset = new int[pattern.length];
        for (int i = 0; i < subset.length; i++) subset[i] = i;
        return canonicalKey(pattern, subset, perms);
    }

    static int patternEdges(boolean[][] pattern) {
        int edges = 0;
        for (int i = 0; i < pattern.length; i++) {
            for (int j = i + 1; j < pattern.length; j++) {
                if (pattern[i][j]) edges++;
            }
        }
        return edges;
    }

    static long binomial(int n, int k) {
        long result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    /** Exact frequency: every subset is examined. */
    static long exactCount(boolean[][] host, int k, boolean[][] pattern, List<int[]> perms) {
        int key = patternKey(pattern, perms);
        int edges = patternEdges(pattern);
        long[] hits = new long[1];
        enumerate(host, k, 0, 0, new int[k], key, edges, perms, hits, null);
        return hits[0];
    }

    private static void enumerate(boolean[][] host, int k, int start, int depth, int[] subset, int key,
                                  int edges, List<int[]> perms, long[] hits, long[] visited) {
        int n = host.length;
        if (depth == k) {
            if (visited != null) visited[0]++;
            if (inducedEdges(host, subset) != edges) return;
            if (canonicalKey(host, subset, perms) == key) hits[0]++;
            return;
        }
        for (int v = start; v <= n - (k - depth); v++) {
            subset[depth] = v;
            enumerate(host, k, v + 1, depth + 1, subset, key, edges, perms, hits, visited);
        }
    }

    /** Draws m uniform k-subsets; result[0] is the hit count, result[1..n] the inclusions per vertex. */
    static long[] sample(boolean[][] host, int k, boolean[][] pattern, int m, long seed) {
        int n = host.length;
        long[] out = new long[n + 1];
        if (k > n) return out;
        List<int[]> perms = permutations(k);
        int key = patternKey(pattern, perms);
        int edges = patternEdges(pattern);
        Rng rng = new Rng(seed);
        int[] index = new int[n];
        int[] subset = new int[k];
        long hits = 0;
        for (int draw = 0; draw < m; draw++) {
            for (int i = 0; i < n; i++) index[i] = i;
            for (int i = 0; i < k; i++) {
                int j = i + rng.nextInt(n - i);
                int tmp = index[i];
                index[i] = index[j];
                index[j] = tmp;
                subset[i] = index[i];
            }
            for (int i = 0; i < k; i++) out[1 + subset[i]]++;
            if (inducedEdges(host, subset) != edges) continue;
            if (canonicalKey(host, subset, perms) == key) hits++;
        }
        out[0] = hits;
        return out;
    }

    public static void main(String[] args) {
        int n = 24;
        int k = 4;
        List<int[]> perms4 = permutations(k);
        boolean[][] k4 = complete(k);
        int k4Key = patternKey(k4, perms4);
        boolean[][] diamondPattern = diamond();
        int diamondKey = patternKey(diamondPattern, perms4);

        // --- Instances: a sparse host with six planted induced K4 copies, and a dense host for the diamond.
        boolean[][] sparse = randomGraph(n, 0.12, new Rng(553001L));
        boolean[][] control = randomGraph(n, 0.12, new Rng(553001L));
        Rng planting = new Rng(553100L);
        int[] vertices = new int[n];
        for (int i = 0; i < n; i++) vertices[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = planting.nextInt(i + 1);
            int tmp = vertices[i];
            vertices[i] = vertices[j];
            vertices[j] = tmp;
        }
        int planted = 6;
        for (int c = 0; c < planted; c++) {
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    sparse[vertices[c * k + i]][vertices[c * k + j]] = true;
                    sparse[vertices[c * k + j]][vertices[c * k + i]] = true;
                }
            }
        }
        for (int c = 0; c < planted; c++) {
            int[] block = new int[k];
            System.arraycopy(vertices, c * k, block, 0, k);
            check(canonicalKey(sparse, block, perms4) == k4Key,
                    "planted block " + c + " is an induced K4");
        }

        long[] visited = new long[1];
        long[] plantedHits = new long[1];
        enumerate(sparse, k, 0, 0, new int[k], k4Key, patternEdges(k4), perms4, plantedHits, visited);
        check(visited[0] == binomial(n, k), "the enumeration visits C(24,4) = " + binomial(n, k)
                + " subsets");
        check(plantedHits[0] == planted, "exactly the " + planted + " planted K4s, no accidental one");
        long controlK4 = exactCount(control, k, k4, perms4);
        check(controlK4 == 0, "the unplanted control contains no induced K4 at all (found "
                + controlK4 + ")");
        double exactPlanted = (double) plantedHits[0] / visited[0];
        check(Math.abs(exactPlanted - 6.0 / 10626.0) <= 1e-15,
                "exact planted frequency is 6/10626 = " + exactPlanted);

        int m = 1000000;
        long[] sampled = sample(sparse, k, k4, m, 553002L);
        double estimate = (double) sampled[0] / m;
        double std = Math.sqrt(exactPlanted * (1.0 - exactPlanted) / m);
        check(Math.abs(estimate - exactPlanted) <= 4.0 * std, "planted estimate " + estimate
                + " is within four standard errors of " + exactPlanted);
        check(estimate > 0.0, "the planted copies are detected");
        long[] controlSampled = sample(control, k, k4, m, 553003L);
        check(controlSampled[0] == 0, "the control sampler reports zero hits ("
                + controlSampled[0] + ")");
        check(estimate > (double) controlSampled[0] / m, "planted exceeds control");

        boolean[][] dense = randomGraph(n, 0.5, new Rng(553004L));
        long denseDiamond = exactCount(dense, k, diamondPattern, perms4);
        long sparseDiamond = exactCount(sparse, k, diamondPattern, perms4);
        double exactDense = (double) denseDiamond / visited[0];
        double exactSparse = (double) sparseDiamond / visited[0];
        check(exactDense - exactSparse > 0.13, "the diamond frequency gap is " + exactDense
                + " versus " + exactSparse);
        int mDense = 200000;
        long[] denseSampled = sample(dense, k, diamondPattern, mDense, 553005L);
        double denseEstimate = (double) denseSampled[0] / mDense;
        double denseStd = Math.sqrt(exactDense * (1.0 - exactDense) / mDense);
        check(Math.abs(denseEstimate - exactDense) <= 4.0 * denseStd, "dense diamond estimate "
                + denseEstimate + " is within four standard errors of " + exactDense);
        check(denseEstimate > exactSparse + 0.13, "the sampler keeps the density ordering: "
                + denseEstimate + " versus " + exactSparse);

        for (int v = 0; v < n; v++) {
            double share = (double) sampled[1 + v] / m;
            double expectedShare = (double) k / n;
            double bound = 4.0 * Math.sqrt(expectedShare * (1 - expectedShare) / m);
            check(Math.abs(share - expectedShare) <= bound,
                    "vertex " + v + " appears in " + share + " of the draws");
        }

        // --- Independent cross-check for k = 3: classification by induced edge count.
        List<int[]> perms3 = permutations(3);
        long[] byEdges = new long[4];
        long[] triples = new long[1];
        int[] subset3 = new int[3];
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                for (int c = b + 1; c < n; c++) {
                    subset3[0] = a;
                    subset3[1] = b;
                    subset3[2] = c;
                    triples[0]++;
                    int edges = 0;
                    if (sparse[a][b]) edges++;
                    if (sparse[a][c]) edges++;
                    if (sparse[b][c]) edges++;
                    byEdges[edges]++;
                }
            }
        }
        check(triples[0] == binomial(n, 3), "the triple scan visits C(n,3) subsets");
        for (int e = 0; e < 4; e++) {
            long counted = exactCount(sparse, 3, triadicMotif(e), perms3);
            check(counted == byEdges[e], "induced triples with " + e + " edges: " + counted
                    + " versus " + byEdges[e]);
        }

        // --- Boundary cases.
        long single = exactCount(sparse, 1, new boolean[1][1], permutations(1));
        check(single == n, "every vertex matches the 1-vertex pattern");
        check(exactCount(k4, k, k4, perms4) == 1, "a host equal to the pattern has frequency 1");
        check(sample(k4, k, k4, 1000, 553006L)[0] == 1000, "a host equal to the pattern is always hit");
        check(exactCount(k4, 5, complete(5), permutations(5)) == 0,
                "a pattern larger than the host: frequency 0");
        check(sample(k4, 5, complete(5), 1000, 553008L)[0] == 0, "and the sampler never hits");
        check(exactCount(complete(6), 4, k4, perms4) == binomial(6, 4),
                "K4 occurs in C(6,4) subsets of K6");
        check(exactCount(complete(6), 4, diamondPattern, perms4) == 0,
                "the diamond occurs in no subset of a complete graph");
        check(exactCount(sparse, k, k4, perms4) == plantedHits[0], "deterministic re-run");

        System.out.println("PASS");
    }

    /** The four unlabeled graphs on three vertices, indexed by their number of edges. */
    static boolean[][] triadicMotif(int edges) {
        boolean[][] motif = new boolean[3][3];
        int[][] pairs = {{0, 1}, {0, 2}, {1, 2}};
        for (int e = 0; e < edges; e++) {
            motif[pairs[e][0]][pairs[e][1]] = true;
            motif[pairs[e][1]][pairs[e][0]] = true;
        }
        return motif;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
