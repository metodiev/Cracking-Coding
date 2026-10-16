/** Task 541 - Implement dynamic programming with divide-and-conquer monotone optimization. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Optimal m-partition of a sequence: split a[0..n-1] into m contiguous groups minimising the sum
 * of squared group sums, with cost(j, i) = (prefix[i] - prefix[j])^2 for the segment a[j..i-1].
 * The naive DP is dp[k][i] = min over j in [k-1, i-1] of dp[k-1][j] + cost(j, i), costing
 * O(m n^2); because this cost obeys the quadrangle inequality, the optimal split point opt[k][i]
 * is non-decreasing in i, so each layer is filled by divide and conquer: solve the middle column
 * by scanning its whole candidate window, then recurse into both halves with the window clamped
 * to [opt(mid - 1), opt(mid + 1)], making each layer O(n log n).
 *
 * The file counts cost evaluations so that claim is concrete. Documented checks: the quadrangle
 * inequality C(a,c) + C(b,d) <= C(a,d) + C(b,c) is verified over sampled quadruples a <= b <= c
 * <= d of every instance (it reduces to 2 * (prefix[c]-prefix[b]) * (prefix[d]-prefix[c]) >= 0);
 * the naive and divide-and-conquer tables agree entry by entry for every layer on LCG arrays up
 * to n = 45, m = 6 and with an exhaustive enumeration of all C(n-1, m-1) cut placements on small
 * instances; opt[k][i] is asserted non-decreasing within every layer; the evaluation counts
 * satisfy dcCalls < naiveCalls and the documented O(m n log n) bound; and the closed forms m = 1
 * (whole-array square), m = n (sum of squares) and an all-zero array (cost 0) are checked,
 * together with the hand-computed values 52 for 1 2 3 4 split in two and 18 for 1 2 3 split in
 * two.
 */
import java.util.*;

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    static long[] prefix(int[] a) {
        long[] pre = new long[a.length + 1];
        for (int i = 0; i < a.length; i++) pre[i + 1] = pre[i] + a[i];
        return pre;
    }

    /** cost(j, i) for the segment a[j..i-1]. */
    static long cost(long[] prefix, int j, int i) {
        long sum = prefix[i] - prefix[j];
        return sum * sum;
    }

    /** Naive O(m n^2) DP over all layers; calls[0] counts inner-loop evaluations. */
    static long[][] naiveDp(int[] a, int m, long[] calls) {
        int n = a.length;
        long[] pre = prefix(a);
        long[][] dp = new long[m + 1][n + 1];
        for (long[] row : dp) Arrays.fill(row, INF);
        dp[0][0] = 0;
        for (int k = 1; k <= m; k++) {
            for (int i = k; i <= n; i++) {
                long best = INF;
                for (int j = k - 1; j < i; j++) {
                    calls[0]++;
                    if (dp[k - 1][j] == INF) continue;
                    best = Math.min(best, dp[k - 1][j] + cost(pre, j, i));
                }
                dp[k][i] = best;
            }
        }
        return dp;
    }

    static class DcResult {
        final long[][] dp;
        final int[][] opt;
        final long calls;

        DcResult(long[][] dp, int[][] opt, long calls) {
            this.dp = dp;
            this.opt = opt;
            this.calls = calls;
        }
    }

    /** Divide-and-conquer DP over the m layers, retaining the optimal split points. */
    static DcResult dcDp(int[] a, int m, long[] calls) {
        int n = a.length;
        long[] pre = prefix(a);
        long[][] dp = new long[m + 1][n + 1];
        int[][] opt = new int[m + 1][n + 1];
        for (long[] row : dp) Arrays.fill(row, INF);
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            dp[1][i] = cost(pre, 0, i);
            opt[1][i] = 0;
        }
        long[] previous = dp[1].clone();
        for (int k = 2; k <= m; k++) {
            long[] current = new long[n + 1];
            int[] currentOpt = new int[n + 1];
            Arrays.fill(current, INF);
            solve(k, k, n, k - 1, n - 1, previous, current, currentOpt, pre, calls);
            for (int i = 0; i <= n; i++) {
                dp[k][i] = current[i];
                opt[k][i] = currentOpt[i];
            }
            previous = current;
        }
        return new DcResult(dp, opt, calls[0]);
    }

    static void solve(int k, int iLo, int iHi, int jLo, int jHi, long[] previous, long[] current,
                      int[] opt, long[] pre, long[] calls) {
        if (iLo > iHi) return;
        int mid = (iLo + iHi) / 2;
        int upper = Math.min(jHi, mid - 1);
        long best = INF;
        int bestJ = -1;
        for (int j = jLo; j <= upper; j++) {
            calls[0]++;
            if (previous[j] == INF) continue;
            long candidate = previous[j] + cost(pre, j, mid);
            if (candidate < best) {
                best = candidate;
                bestJ = j;
            }
        }
        current[mid] = best;
        opt[mid] = bestJ;
        solve(k, iLo, mid - 1, jLo, bestJ, previous, current, opt, pre, calls);
        solve(k, mid + 1, iHi, bestJ, jHi, previous, current, opt, pre, calls);
    }

    /** Exhaustive enumeration of every placement of m-1 cuts among the n-1 gaps. */
    static long brutePartition(int[] a, int m) {
        int n = a.length;
        long[] pre = prefix(a);
        long best = INF;
        int gaps = n - 1;
        for (int mask = 0; mask < (1 << gaps); mask++) {
            if (Integer.bitCount(mask) != m - 1) continue;
            long total = 0;
            int start = 0;
            for (int g = 0; g < gaps; g++) {
                if ((mask & (1 << g)) != 0) {
                    total += cost(pre, start, g + 1);
                    start = g + 1;
                }
            }
            total += cost(pre, start, n);
            best = Math.min(best, total);
        }
        return best;
    }

    static int[] lcgArray(int n, int seed, int limit) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            a[i] = seed % limit;
        }
        return a;
    }

    static int log2(int n) {
        int result = 0;
        while ((1 << result) < n) result++;
        return result;
    }

    public static void main(String[] args) {
        int[] hand = {1, 2, 3, 4};
        long[][] handTable = naiveDp(hand, 2, new long[1]);
        check(handTable[2][4] == 52, "hand-checked: 1 2 3 4 in two groups costs (1+2+3)^2 + 4^2 = 52");
        check(handTable[2][3] == 18, "hand-checked: 1 2 3 in two groups costs 3^2 + 3^2 = 18");
        long[] pre = prefix(hand);
        check(cost(pre, 0, 3) == 36 && cost(pre, 1, 3) == 25 && cost(pre, 3, 4) == 16,
                "hand-checked segment costs");
        DcResult handDc = dcDp(hand, 2, new long[1]);
        check(handDc.dp[2][4] == 52 && handDc.opt[2][4] == 3, "the D&C DP finds the same split");
        check(naiveDp(new int[]{5}, 1, new long[1])[1][1] == 25, "single element, single group");
        check(naiveDp(new int[6], 3, new long[1])[3][6] == 0, "an all-zero array costs nothing");
        check(dcDp(hand, 4, new long[1]).dp[4][4] == 30, "m = n forces the sum of squares (1+4+9+16)");

        int[] seeds = {11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 143};
        for (int s = 0; s < seeds.length; s++) {
            int n = 12 + s * 3;
            int m = 2 + s % 5;
            int[] a = lcgArray(n, seeds[s], 7);
            long[] naiveCalls = new long[1];
            long[] dcCalls = new long[1];
            long[][] viaNaive = naiveDp(a, m, naiveCalls);
            DcResult viaDc = dcDp(a, m, dcCalls);
            for (int k = 1; k <= m; k++) {
                for (int i = 0; i <= n; i++) {
                    check(viaNaive[k][i] == viaDc.dp[k][i],
                            "instance " + s + ": layer " + k + " column " + i + " agrees with the naive DP");
                }
                for (int i = k; i < n; i++) {
                    check(viaDc.opt[k][i] <= viaDc.opt[k][i + 1],
                            "instance " + s + ": opt non-decreasing at layer " + k + ", i = " + i);
                }
            }
            check(viaDc.calls < naiveCalls[0],
                    "instance " + s + ": D&C evaluates fewer costs (" + viaDc.calls + " < " + naiveCalls[0] + ")");
            check(viaDc.calls <= 2L * m * n * (log2(n) + 1),
                    "instance " + s + ": D&C stays within the documented O(m n log n) bound");

            long[] preA = prefix(a);
            int stride = Math.max(1, n / 8);
            for (int p = 0; p < n - 3; p += stride) {
                for (int q = p + 1; q < n - 2; q += stride) {
                    for (int r = q + 1; r < n - 1; r += stride) {
                        int d = Math.min(n, r + 1 + stride);
                        check(cost(preA, p, r) + cost(preA, q, d) <= cost(preA, p, d) + cost(preA, q, r),
                                "quadrangle inequality holds for " + p + " " + q + " " + r + " " + d);
                    }
                }
            }
        }

        for (int n = 3; n <= 8; n++) {
            for (int m = 1; m <= n; m++) {
                int[] a = lcgArray(n, 1000 + n * 10 + m, 6);
                long dc = dcDp(a, m, new long[1]).dp[m][n];
                long naiveValue = naiveDp(a, m, new long[1])[m][n];
                long brute = brutePartition(a, m);
                check(dc == brute, "n=" + n + " m=" + m + ": D&C equals exhaustive cut enumeration");
                check(naiveValue == brute, "n=" + n + " m=" + m + ": naive DP equals exhaustive cuts");
            }
        }

        int[] big = lcgArray(40, 777, 7);
        long[] dcCalls = new long[1];
        long[] naiveCallsBig = new long[1];
        long viaNaive = naiveDp(big, 6, naiveCallsBig)[6][40];
        long viaDc = dcDp(big, 6, dcCalls).dp[6][40];
        check(viaNaive == viaDc, "n = 40, m = 6: D&C and naive agree");
        check(dcCalls[0] < naiveCallsBig[0] / 2,
                "n = 40, m = 6: D&C needs less than half the naive evaluations");

        int[] ones = {2, 3, 5, 7, 11};
        long total = 0;
        for (int value : ones) total += value;
        check(naiveDp(ones, 1, new long[1])[1][5] == total * total,
                "m = 1 collapses to the whole-array square");
        long expected = 0;
        for (int value : ones) expected += (long) value * value;
        check(dcDp(ones, 5, new long[1]).dp[5][5] == expected, "m = n gives the sum of squares");
        check(dcDp(new int[]{0, 0, 0, 0}, 2, new long[1]).dp[2][4] == 0, "zeros stay zero");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
