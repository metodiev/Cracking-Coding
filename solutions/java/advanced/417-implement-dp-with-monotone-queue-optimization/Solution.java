/** Task 417 - Implement DP with monotone queue optimization. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Sliding-window minimum DP: jump from j to i at cost a[i], landing windows of size k:
 *     dp[0] = a[0],   dp[i] = a[i] + min(dp[j] for j in [max(0, i - k), i - 1]).
 * The naive transition scans k predecessors (O(n*k) overall). A monotone deque keeps the
 * candidate indices with STRICTLY increasing dp values: expired indices (j < i - k) leave
 * from the front, and while adding i every index with dp >= dp[i] is popped from the back,
 * because it is both older and worse. Every step is amortized O(1).
 * Tie-break: newer index wins (the pop condition keeps equal-valued indices newest-first),
 * which the tests only use through value equality.
 */
public class Solution {
    /** Returns {dp, pred}; pred[i] is the j achieving the window minimum. */
    static int[][] slidingMinDp(int[] a, int k) {
        int n = a.length;
        int[] dp = new int[n];
        int[] pred = new int[n];
        if (n == 0) return new int[][]{dp, pred};
        dp[0] = a[0];
        pred[0] = -1;
        int[] dq = new int[n + 1];
        int head = 0, tail = 0;
        dq[tail++] = 0;
        for (int i = 1; i < n; i++) {
            while (dq[head] < i - k) head++;
            int j = dq[head];
            dp[i] = a[i] + dp[j];
            pred[i] = j;
            while (tail > head && dp[dq[tail - 1]] >= dp[i]) tail--;
            dq[tail++] = i;
        }
        return new int[][]{dp, pred};
    }

    /** Oracle: O(n*k) direct evaluation of the recurrence. */
    static int[] slidingMinDpNaive(int[] a, int k) {
        int n = a.length;
        if (n == 0) return new int[0];
        int[] dp = new int[n];
        dp[0] = a[0];
        for (int i = 1; i < n; i++) {
            int best = Integer.MAX_VALUE;
            for (int j = Math.max(0, i - k); j < i; j++) best = Math.min(best, dp[j]);
            dp[i] = a[i] + best;
        }
        return dp;
    }

    private static int seed = 417;

    private static int[] lcgArray(int length, int lo, int hi) {
        int[] out = new int[length];
        for (int k = 0; k < length; k++) {
            seed = (seed * 75 + 74) % 65537;
            out[k] = lo + seed % (hi - lo + 1);
        }
        return out;
    }

    private static boolean same(int[] a, int[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void main(String[] args) {
        // hand-derived values, a = [10, 1, 5, 7, 3, 2]
        int[] a = {10, 1, 5, 7, 3, 2};
        int[][] r2 = slidingMinDp(a, 2);
        check(same(r2[0], new int[]{10, 11, 15, 18, 18, 20}), "dp k2");
        // at i = 4 the equal-valued index 3 is dropped (newer ties win), so pred[5] = 4
        check(same(r2[1], new int[]{-1, 0, 0, 1, 2, 4}), "pred k2");
        check(r2[0][4] == a[4] + Math.min(r2[0][2], r2[0][3]), "dp4 window min");
        check(r2[0][5] == a[5] + Math.min(r2[0][3], r2[0][4]), "dp5 window min");

        int[][] r3 = slidingMinDp(a, 3);
        check(same(r3[0], new int[]{10, 11, 15, 17, 14, 16}), "dp k3");
        int[][] r1 = slidingMinDp(a, 1);
        check(same(r1[0], new int[]{10, 11, 16, 23, 26, 28}), "dp k1");
        check(same(r1[1], new int[]{-1, 0, 1, 2, 3, 4}), "pred k1");

        // k = 1 makes dp a running prefix sum of a
        int running = 0;
        for (int i = 0; i < a.length; i++) {
            running += a[i];
            check(r1[0][i] == running, "prefix sum " + i);
        }
        // k >= n opens the whole history
        int[][] big = slidingMinDp(a, 100);
        check(big[0][0] == 10 && big[0][1] == 11 && big[0][2] == 15, "big window front");
        check(big[0][3] == 17 && big[0][4] == 13 && big[0][5] == 12, "big window back");

        // every hand value also matches the naive oracle and the predecessor optimality
        for (int k = 1; k <= 7; k++) {
            int[][] res = slidingMinDp(a, k);
            check(same(res[0], slidingMinDpNaive(a, k)), "oracle " + k);
            for (int i = 1; i < a.length; i++) {
                check(res[1][i] >= Math.max(0, i - k) && res[1][i] < i, "pred range " + k + " " + i);
                int windowMin = Integer.MAX_VALUE;
                for (int j = Math.max(0, i - k); j < i; j++) windowMin = Math.min(windowMin, res[0][j]);
                check(res[0][res[1][i]] == windowMin, "pred optimal " + k + " " + i);
            }
        }

        // degenerate sizes
        check(slidingMinDp(new int[0], 5)[0].length == 0, "empty");
        int[][] single = slidingMinDp(new int[]{7}, 3);
        check(single[0].length == 1 && single[0][0] == 7 && single[1][0] == -1, "single");
        check(slidingMinDpNaive(new int[0], 5).length == 0, "empty naive");
        check(slidingMinDp(new int[]{-4, -8}, 2)[0][1] == -12, "negative");

        for (int round = 0; round < 8; round++) {
            int n = 120 + seed % 180;
            int[] values = lcgArray(n, -50, 50);
            int k = 1 + seed % 15;
            int[][] res = slidingMinDp(values, k);
            int[] dp = res[0], pred = res[1];
            check(same(dp, slidingMinDpNaive(values, k)), "lcg oracle " + round);
            for (int i = 1; i < n; i++) {
                int j = pred[i];
                check(j >= Math.max(0, i - k) && j < i, "lcg pred range " + round);
                check(dp[i] == values[i] + dp[j], "lcg recurrence " + round);
                int windowMin = Integer.MAX_VALUE;
                for (int t = Math.max(0, i - k); t < i; t++) windowMin = Math.min(windowMin, dp[t]);
                check(dp[j] == windowMin, "lcg pred optimal " + round);
            }
            // monotone deque invariant: candidate values strictly increase front to back
            int[] dq = new int[n + 1];
            int head = 0, tail = 0;
            dq[tail++] = 0;
            for (int i = 1; i < n; i++) {
                while (dq[head] < i - k) head++;
                int windowMin = Integer.MAX_VALUE;
                for (int t = Math.max(0, i - k); t < i; t++) windowMin = Math.min(windowMin, dp[t]);
                check(dp[dq[head]] == windowMin, "deque front " + round + " " + i);
                while (tail > head && dp[dq[tail - 1]] >= dp[i]) tail--;
                dq[tail++] = i;
            }
        }

        // window bound: widening k never increases any dp value (only relaxes the minimum)
        int[] values = lcgArray(150, 0, 30);
        int[] previous = slidingMinDpNaive(values, 1);
        for (int k = 2; k <= 10; k++) {
            int[] current = slidingMinDpNaive(values, k);
            for (int i = 0; i < values.length; i++) {
                check(current[i] <= previous[i], "monotone k " + k + " i " + i);
            }
            previous = current;
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
