/** Task 555 - Implement dynamic programming for counting paths of length k in DAG (matrix exponentiation). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Counting length-k paths in a DAG three ways, which is the point of the task: (1) the textbook dynamic
 * program over a topological order, dp[i][l] = number of paths of exactly l edges starting at vertex i,
 * with dp[i][0] = 1 and dp[i][l] = sum over successors j of dp[j][l-1]; (2) the same recursion evaluated
 * as memoised recursion, i.e. in a completely different traversal order; and (3) matrix exponentiation,
 * since in an acyclic graph every walk is automatically a simple path, so the length-k walk count
 * computed from the k-th power of the adjacency matrix must equal the DP value. Instances are seeded
 * DAGs whose vertices 0..n-1 are already in topological order (edge i -> j drawn with probability p for
 * i < j) plus the canonical chain.
 *
 * Documented checks. All three methods agree on every (instance, k) pair tested: the seeded 14-vertex
 * DAGs for k = 0..20 and the chain graph, where the counts are also known in closed form (n - k paths of
 * length k for k < n and 0 thereafter) and which the code verifies separately. Because a DAG cannot
 * repeat a vertex, every walk of length k is a simple path, so all counts are bounded by C(n, k+1) <=
 * 2^n; the self-test asserts that bound (and therefore that no long arithmetic overflow can hide in the
 * result), reconstructs every path of a small instance explicitly and verifies each one has exactly k
 * edges and strictly increasing vertex indices, confirms that the brute-force walk enumeration visits
 * all-distinct vertices, checks that the k = 0 count is n and that the count is 0 once k >= n, checks
 * that a graph with no edges has only the k = 0 paths, and re-runs the DP to confirm determinism.
 */
import java.util.*;

public class Solution {

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
    }

    /** Vertices 0..n-1 in topological order; edge i -> j exists with probability p for i < j. */
    static boolean[][] randomDag(int n, double p, Rng rng) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) adj[i][j] = rng.nextDouble() < p;
        }
        return adj;
    }

    static boolean[][] chain(int n) {
        boolean[][] adj = new boolean[n][n];
        for (int i = 0; i + 1 < n; i++) adj[i][i + 1] = true;
        return adj;
    }

    /** Textbook DP over the topological order; dpFrom[i][l] counts paths of exactly l edges at i. */
    static long[][] dpTable(boolean[][] adj, int k) {
        int n = adj.length;
        long[][] dp = new long[n][k + 1];
        for (int i = 0; i < n; i++) dp[i][0] = 1;
        for (int i = n - 1; i >= 0; i--) {
            for (int l = 1; l <= k; l++) {
                long sum = 0;
                for (int j = i + 1; j < n; j++) {
                    if (adj[i][j]) sum += dp[j][l - 1];
                }
                dp[i][l] = sum;
            }
        }
        return dp;
    }

    static long countPathsDp(boolean[][] adj, int k) {
        if (k < 0) return 0;
        long[][] dp = dpTable(adj, k);
        long total = 0;
        for (int i = 0; i < adj.length; i++) total += dp[i][k];
        return total;
    }

    /** The same recursion as memoised recursion: no topological order is used. */
    static long countPathsMemo(boolean[][] adj, int k) {
        int n = adj.length;
        Long[][] memo = new Long[n][k + 1];
        long total = 0;
        for (int i = 0; i < n; i++) total += memoDfs(adj, i, k, memo);
        return total;
    }

    private static long memoDfs(boolean[][] adj, int v, int remaining, Long[][] memo) {
        if (remaining == 0) return 1;
        if (memo[v][remaining] != null) return memo[v][remaining];
        long total = 0;
        for (int j = 0; j < adj.length; j++) {
            if (adj[v][j]) total += memoDfs(adj, j, remaining - 1, memo);
        }
        memo[v][remaining] = total;
        return total;
    }

    static long[][] matMul(long[][] a, long[][] b) {
        int n = a.length;
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < n; t++) {
                if (a[i][t] == 0) continue;
                for (int j = 0; j < n; j++) out[i][j] += a[i][t] * b[t][j];
            }
        }
        return out;
    }

    static long[][] matPow(long[][] base, int exponent) {
        int n = base.length;
        long[][] result = new long[n][n];
        for (int i = 0; i < n; i++) result[i][i] = 1;
        long[][] power = new long[n][n];
        for (int i = 0; i < n; i++) power[i] = base[i].clone();
        int e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = matMul(result, power);
            power = matMul(power, power);
            e >>= 1;
        }
        return result;
    }

    /** Length-k walks via the k-th power of the adjacency matrix; in a DAG these are paths. */
    static long countPathsMatrix(boolean[][] adj, int k) {
        int n = adj.length;
        long[][] a = new long[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) if (adj[i][j]) a[i][j] = 1;
        }
        long[][] power = matPow(a, k);
        long total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) total += power[i][j];
        }
        return total;
    }

    /** Brute force: enumerate every walk of exactly k edges and record it. */
    static List<int[]> enumerateWalks(boolean[][] adj, int k) {
        List<int[]> out = new ArrayList<int[]>();
        int n = adj.length;
        for (int start = 0; start < n; start++) {
            int[] path = new int[k + 1];
            path[0] = start;
            walkDfs(adj, start, k, path, out);
        }
        return out;
    }

    private static void walkDfs(boolean[][] adj, int v, int remaining, int[] path, List<int[]> out) {
        if (remaining == 0) {
            out.add(path.clone());
            return;
        }
        int depth = path.length - remaining - 1;
        for (int j = 0; j < adj.length; j++) {
            if (!adj[v][j]) continue;
            path[depth + 1] = j;
            walkDfs(adj, j, remaining - 1, path, out);
        }
    }

    static long binomial(int n, int k) {
        if (k < 0 || k > n) return 0;
        long result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    public static void main(String[] args) {
        int n = 14;
        Rng rng = new Rng(20240555L);
        int dags = 0;
        for (int t = 0; t < 3; t++) {
            boolean[][] adj = randomDag(n, 0.2 + 0.15 * t, rng);
            dags++;
            for (int k = 0; k <= 20; k++) {
                long dp = countPathsDp(adj, k);
                long memo = countPathsMemo(adj, k);
                long matrix = countPathsMatrix(adj, k);
                check(dp == memo, "DP and memoised recursion agree (graph " + t + ", k = " + k + ")");
                check(dp == matrix, "DP and matrix exponentiation agree (graph " + t + ", k = " + k
                        + "): " + dp + " versus " + matrix);
                check(dp <= binomial(n, Math.min(k + 1, n)),
                        "the count respects the C(n,k+1) bound (graph " + t + ", k = " + k + ")");
                if (k == 0) check(dp == n, "k = 0 counts the n single-vertex paths");
                if (k >= n) check(dp == 0, "a path cannot be longer than n-1 edges");
            }
        }
        check(dags == 3, "three seeded DAGs were tested");

        // --- Chain graph: known closed form.
        int m = 9;
        boolean[][] chainGraph = chain(m);
        for (int k = 0; k <= 12; k++) {
            long expected = k < m ? m - k : 0;
            check(countPathsDp(chainGraph, k) == expected, "chain: " + m + "-vertex chain has " + expected
                    + " paths of length " + k);
            check(countPathsMatrix(chainGraph, k) == expected, "chain via matrix power, k = " + k);
            check(countPathsMemo(chainGraph, k) == expected, "chain via memoised recursion, k = " + k);
        }

        // --- Paths reconstructed explicitly: k edges, strictly increasing indices, all vertices distinct.
        boolean[][] small = randomDag(6, 0.55, new Rng(555002L));
        for (int k = 0; k <= 5; k++) {
            List<int[]> walks = enumerateWalks(small, k);
            check(walks.size() == countPathsDp(small, k), "brute force walk count, k = " + k);
            for (int[] walk : walks) {
                check(walk.length == k + 1, "the walk has k + 1 vertices");
                boolean[] seen = new boolean[small.length];
                for (int i = 0; i <= k; i++) {
                    check(!seen[walk[i]], "a walk in a DAG cannot repeat a vertex");
                    seen[walk[i]] = true;
                    if (i > 0) {
                        check(walk[i - 1] < walk[i], "indices increase along the topological order");
                        check(small[walk[i - 1]][walk[i]], "consecutive vertices are adjacent");
                    }
                }
            }
        }

        // --- Edge cases.
        boolean[][] noEdges = new boolean[5][5];
        check(countPathsDp(noEdges, 0) == 5, "no edges: five paths of length 0");
        check(countPathsDp(noEdges, 1) == 0 && countPathsMatrix(noEdges, 1) == 0,
                "no edges: no paths of positive length");
        check(countPathsDp(chainGraph, -1) == 0, "a negative length yields 0");
        check(countPathsDp(new boolean[0][0], 0) == 0, "the empty graph has no paths");
        check(countPathsDp(chainGraph, 12) == 0 && countPathsMatrix(chainGraph, 12) == 0,
                "beyond n-1 edges both methods return 0");

        long dpFirst = countPathsDp(small, 4);
        check(dpFirst == countPathsDp(small, 4) && dpFirst == countPathsMatrix(small, 4),
                "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
