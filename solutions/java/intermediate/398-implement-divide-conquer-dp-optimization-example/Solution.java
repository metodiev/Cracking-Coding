/** Task 398 - Implement a divide-and-conquer DP optimization example. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    static long partitionNaive(int[] values, int k) {
        int n = values.length;
        if (k > n) return INF;
        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + values[i - 1];
        long[] previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        for (int t = 1; t <= k; t++) {
            long[] current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            for (int i = t; i <= n; i++) {
                for (int j = t - 1; j < i; j++) {
                    if (previous[j] == INF) continue;
                    long group = prefix[i] - prefix[j];
                    long cost = previous[j] + group * group;
                    if (cost < current[i]) current[i] = cost;
                }
            }
            previous = current;
        }
        return previous[n];
    }

    private static long[] dcPrevious, dcCurrent, dcPrefix;
    private static int[] dcOpt;

    /** Returns {answer, opt tables per layer flattened as int[][];} - see main for usage. */
    static Object[] partitionDc(int[] values, int k) {
        int n = values.length;
        if (k > n) return new Object[]{INF, new ArrayList<int[]>()};
        dcPrefix = new long[n + 1];
        for (int i = 1; i <= n; i++) dcPrefix[i] = dcPrefix[i - 1] + values[i - 1];
        long[] previous = new long[n + 1];
        for (int i = 1; i <= n; i++) previous[i] = INF;
        List<int[]> allOpts = new ArrayList<>();
        for (int t = 1; t <= k; t++) {
            long[] current = new long[n + 1];
            for (int i = 1; i <= n; i++) current[i] = INF;
            dcOpt = new int[n + 1];
            // i == t: the only split is j == t-1; the last singleton group costs values[t-1]^2
            current[t] = previous[t - 1] + (long) values[t - 1] * values[t - 1];
            dcOpt[t] = t - 1;
            dcPrevious = previous;
            dcCurrent = current;
            if (t + 1 <= n) solve(t + 1, n, t - 1, n - 1);
            allOpts.add(dcOpt.clone());
            previous = current;
        }
        return new Object[]{previous[n], allOpts};
    }

    private static void solve(int l, int r, int optl, int optr) {
        if (l > r) return;
        int mid = (l + r) >> 1;
        long best = INF;
        int bestJ = optl;
        int upper = Math.min(optr, mid - 1);
        for (int j = optl; j <= upper; j++) {
            if (dcPrevious[j] == INF) continue;
            long group = dcPrefix[mid] - dcPrefix[j];
            long cost = dcPrevious[j] + group * group;
            if (cost < best) {
                best = cost;
                bestJ = j;
            }
        }
        dcCurrent[mid] = best;
        dcOpt[mid] = bestJ;
        solve(l, mid - 1, optl, bestJ);
        solve(mid + 1, r, bestJ, optr);
    }

    static boolean optIsMonotone(int[] opt, int t, int n) {
        int last = -1;
        for (int i = t; i <= n; i++) {
            if (opt[i] < last) return false;
            last = opt[i];
        }
        return true;
    }

    private static int seed = 804;

    private static int[][] lcgArrays(int count, int maxLen) {
        int[][] arrays = new int[count][];
        for (int a = 0; a < count; a++) {
            int n = 2 + seed % (maxLen - 1);
            int[] values = new int[n];
            for (int i = 0; i < n; i++) {
                seed = (seed * 75 + 74) % 65537;
                values[i] = 1 + seed % 9;
            }
            arrays[a] = values;
        }
        return arrays;
    }

    public static void main(String[] args) {
        check(partitionNaive(new int[]{1, 2, 3, 4}, 2) == 52, "naive 52");
        check((Long) partitionDc(new int[]{1, 2, 3, 4}, 2)[0] == 52, "dc 52");
        check(partitionNaive(new int[]{1, 2, 3, 4}, 1) == 100, "single group");
        check(partitionNaive(new int[]{1, 2, 3, 4}, 4) == 30, "all singletons");
        check(partitionNaive(new int[]{5}, 1) == 25, "one element");
        check(partitionNaive(new int[]{1, 2}, 3) == INF, "infeasible");
        check((Long) partitionDc(new int[]{1, 2, 3, 4}, 4)[0] == 30, "dc all singletons");

        for (int[] values : lcgArrays(40, 60)) {
            int n = values.length;
            for (int k = 1; k <= Math.min(n, 8); k++) {
                long naive = partitionNaive(values, k);
                Object[] result = partitionDc(values, k);
                check((Long) result[0] == naive, "agreement");
                @SuppressWarnings("unchecked")
                List<int[]> allOpts = (List<int[]>) result[1];
                for (int t = 1; t <= allOpts.size(); t++) {
                    check(optIsMonotone(allOpts.get(t - 1), t, n), "monotone layer " + t);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
