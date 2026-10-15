/** Task 397 - Implement Knuth optimization (DP speedup demonstration). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    /** weights: n positive weights; returns the minimum weighted search cost. */
    static long obstPlain(int[] weights) {
        int n = weights.length;
        if (n == 0) return 0;
        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
        long[][] e = new long[n + 2][n + 2];
        for (int i = 1; i <= n; i++) e[i][i] = weights[i - 1];
        for (int length = 2; length <= n; length++) {
            for (int i = 1; i + length - 1 <= n; i++) {
                int j = i + length - 1;
                long total = prefix[j] - prefix[i - 1];
                long best = INF;
                for (int k = i; k <= j; k++) {
                    long left = k - 1 >= i ? e[i][k - 1] : 0;
                    long right = k + 1 <= j ? e[k + 1][j] : 0;
                    best = Math.min(best, left + right + total);
                }
                e[i][j] = best;
            }
        }
        return e[1][n];
    }

    /** Knuth-optimised DP; returns {cost, opt[i][j] in a packed long[][]}. */
    static Object[] obstKnuth(int[] weights) {
        int n = weights.length;
        long[][] e = new long[n + 2][n + 2];
        int[][] opt = new int[n + 2][n + 2];
        if (n == 0) return new Object[]{0L, opt};
        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + weights[i - 1];
        for (int i = 1; i <= n; i++) {
            e[i][i] = weights[i - 1];
            opt[i][i] = i;
        }
        for (int length = 2; length <= n; length++) {
            for (int i = 1; i + length - 1 <= n; i++) {
                int j = i + length - 1;
                long total = prefix[j] - prefix[i - 1];
                int lo = Math.max(opt[i][j - 1], i);
                int hi = Math.min(opt[i + 1][j], j);
                long best = INF;
                int bestK = lo;
                for (int k = lo; k <= hi; k++) {
                    long left = k - 1 >= i ? e[i][k - 1] : 0;
                    long right = k + 1 <= j ? e[k + 1][j] : 0;
                    long candidate = left + right + total;
                    if (candidate < best) {
                        best = candidate;
                        bestK = k;
                    }
                }
                e[i][j] = best;
                opt[i][j] = bestK;
            }
        }
        return new Object[]{e[1][n], opt};
    }

    static boolean optIsMonotone(int[][] opt, int n) {
        for (int length = 2; length <= n; length++) {
            for (int i = 1; i + length - 1 <= n; i++) {
                int j = i + length - 1;
                if (opt[i][j] < opt[i][j - 1] || opt[i][j] > opt[i + 1][j]) return false;
            }
        }
        return true;
    }

    /** Rebuilds the tree from the opt table and re-scores it by walking the structure. */
    static long rebuildCost(int[] weights, int[][] opt, int i, int j, int depth) {
        if (i > j) return 0;
        int root = opt[i][j];
        return (long) weights[root - 1] * (depth + 1)
                + rebuildCost(weights, opt, i, root - 1, depth + 1)
                + rebuildCost(weights, opt, root + 1, j, depth + 1);
    }

    static List<Integer> rebuildInorder(int[][] opt, int i, int j) {
        List<Integer> out = new ArrayList<>();
        if (i > j) return out;
        int root = opt[i][j];
        out.addAll(rebuildInorder(opt, i, root - 1));
        out.add(root);
        out.addAll(rebuildInorder(opt, root + 1, j));
        return out;
    }

    private static int seed = 2026;

    private static int[][] lcgWeightSets(int count, int maxN) {
        int[][] sets = new int[count][];
        for (int s = 0; s < count; s++) {
            int n = 2 + seed % (maxN - 1);
            int[] weights = new int[n];
            for (int i = 0; i < n; i++) {
                seed = (seed * 75 + 74) % 65537;
                weights[i] = 1 + seed % 50;
            }
            sets[s] = weights;
        }
        return sets;
    }

    public static void main(String[] args) {
        check(obstPlain(new int[]{5}) == 5, "single key");
        check((Long) obstKnuth(new int[]{5})[0] == 5, "single key knuth");
        check(obstPlain(new int[]{1, 1}) == 3, "two equal keys");
        check(obstPlain(new int[]{3, 2, 1}) == 10, "three keys");
        check((Long) obstKnuth(new int[]{3, 2, 1})[0] == 10, "three keys knuth");
        check(obstPlain(new int[]{1, 1, 1, 1}) == 8, "four equal keys");
        check((Long) obstKnuth(new int[]{1, 1, 1, 1})[0] == 8, "four equal keys knuth");

        for (int[] weights : lcgWeightSets(120, 40)) {
            long plain = obstPlain(weights);
            Object[] result = obstKnuth(weights);
            long knuth = (Long) result[0];
            int[][] opt = (int[][]) result[1];
            int n = weights.length;
            check(knuth == plain, "random agreement");
            check(optIsMonotone(opt, n), "monotone opt");
            check(rebuildCost(weights, opt, 1, n, 0) == knuth, "rebuild scoring");
            check(rebuildInorder(opt, 1, n).toString().equals(rangeString(n)), "inorder keys");
        }

        int[] sample = lcgWeightSets(1, 40)[0];
        int n = sample.length;
        int[][] opt = (int[][]) obstKnuth(sample)[1];
        long scanned = 0;
        for (int length = 2; length <= n; length++) {
            for (int i = 1; i + length - 1 <= n; i++) {
                int j = i + length - 1;
                scanned += opt[i][j] - opt[i][j - 1] + 1;
            }
        }
        check(scanned < (long) n * n, "windows stay quadratic");
        System.out.println("PASS");
    }

    private static String rangeString(int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 1; i <= n; i++) {
            if (i > 1) sb.append(", ");
            sb.append(i);
        }
        return sb.append("]").toString();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
