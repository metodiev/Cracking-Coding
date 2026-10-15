/** Task 396 - Implement the offline convex hull trick with divide-and-conquer DP optimization. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class MinHull {
    final List<Long> ms = new ArrayList<>();
    final List<Long> bs = new ArrayList<>();

    /** Insert lines with NON-INCREASING slopes; query with non-decreasing x. */
    void addLine(long m, long b) {
        if (!ms.isEmpty() && m == ms.get(ms.size() - 1)) {
            if (b >= bs.get(bs.size() - 1)) return;      // same slope, worse intercept
            ms.remove(ms.size() - 1);
            bs.remove(bs.size() - 1);
        }
        while (ms.size() >= 2) {
            long m1 = ms.get(ms.size() - 2), b1 = bs.get(bs.size() - 2);
            long m2 = ms.get(ms.size() - 1), b2 = bs.get(bs.size() - 1);
            if ((b - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m)) {
                ms.remove(ms.size() - 1);
                bs.remove(bs.size() - 1);
            } else {
                break;
            }
        }
        ms.add(m);
        bs.add(b);
    }

    long query(long x) {
        if (ms.isEmpty()) throw new IllegalStateException("query on an empty hull");
        int lo = 0, hi = ms.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ms.get(mid) * x + bs.get(mid) <= ms.get(mid + 1) * x + bs.get(mid + 1)) hi = mid;
            else lo = mid + 1;
        }
        return ms.get(lo) * x + bs.get(lo);
    }
}

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    static long partitionBruteforce(long[] xs, int k) {
        int n = xs.length;
        if (k > n) return INF;
        long[] previous = new long[n + 1];
        Arrays.fill(previous, INF);
        previous[0] = 0;
        for (int t = 1; t <= k; t++) {
            long[] current = new long[n + 1];
            Arrays.fill(current, INF);
            for (int i = t; i <= n; i++) {
                for (int j = t - 1; j < i; j++) {
                    if (previous[j] == INF) continue;
                    long span = xs[i - 1] - xs[j];
                    long cost = previous[j] + span * span;
                    if (cost < current[i]) current[i] = cost;
                }
            }
            previous = current;
        }
        return previous[n];
    }

    static long partitionOfflineCht(long[] xs, int k) {
        int n = xs.length;
        if (k > n) return INF;
        long[] previous = new long[n + 1];
        Arrays.fill(previous, INF);
        previous[0] = 0;
        for (int t = 1; t <= k; t++) {
            long[] current = new long[n + 1];
            Arrays.fill(current, INF);
            MinHull hull = new MinHull();
            for (int i = t; i <= n; i++) {
                int j = i - 1;
                if (previous[j] != INF) hull.addLine(-2 * xs[j], previous[j] + xs[j] * xs[j]);
                long x = xs[i - 1];
                current[i] = x * x + hull.query(x);
            }
            previous = current;
        }
        return previous[n];
    }

    private static long[] dcPrevious, dcCurrent, dcXs;

    static long partitionDivideConquer(long[] xs, int k) {
        int n = xs.length;
        if (k > n) return INF;
        long[] previous = new long[n + 1];
        Arrays.fill(previous, INF);
        previous[0] = 0;
        for (int t = 1; t <= k; t++) {
            long[] current = new long[n + 1];
            Arrays.fill(current, INF);
            dcPrevious = previous;
            dcCurrent = current;
            dcXs = xs;
            if (previous[t - 1] != INF) current[t] = previous[t - 1];
            solveDc(t, t + 1, n, t - 1);
            previous = current;
        }
        return previous[n];
    }

    private static void solveDc(int t, int l, int r, int optl) {
        // optR is the current layer's opt bound; tracked implicitly as the previous call's best
        solveDc(t, l, r, optl, dcXs.length);
    }

    private static void solveDc(int t, int l, int r, int optl, int optr) {
        if (l > r) return;
        int mid = (l + r) / 2;
        long best = INF;
        int bestJ = -1;
        int upper = Math.min(optr, mid - 1);
        for (int j = optl; j <= upper; j++) {
            if (dcPrevious[j] == INF) continue;
            long span = dcXs[mid - 1] - dcXs[j];
            long cost = dcPrevious[j] + span * span;
            if (cost < best) {
                best = cost;
                bestJ = j;
            }
        }
        dcCurrent[mid] = best;
        if (bestJ == -1) bestJ = optl;
        solveDc(t, l, mid - 1, optl, bestJ);
        solveDc(t, mid + 1, r, bestJ, optr);
    }

    private static int seed = 99;

    private static long[][] lcgArrays(int count, int length) {
        long[][] arrays = new long[count][];
        for (int k = 0; k < count; k++) {
            long[] xs = new long[length];
            for (int i = 1; i < length; i++) {
                seed = (seed * 75 + 74) % 65537;
                xs[i] = xs[i - 1] + 1 + seed % 20;
            }
            arrays[k] = xs;
        }
        return arrays;
    }

    public static void main(String[] args) {
        check(partitionBruteforce(new long[]{1, 3, 5}, 2) == 4, "brute 4");
        check(partitionBruteforce(new long[]{1, 3, 5}, 1) == 16, "brute 16");
        check(partitionBruteforce(new long[]{1, 3, 5}, 3) == 0, "brute 0");
        check(partitionBruteforce(new long[]{1, 3}, 2) == 0, "two points two segments");
        check(partitionBruteforce(new long[]{1, 3}, 3) == INF, "infeasible");
        check(partitionBruteforce(new long[]{1}, 1) == 0, "single point");

        check(partitionOfflineCht(new long[]{1, 3, 5}, 2) == 4, "cht hand");
        check(partitionDivideConquer(new long[]{1, 3, 5}, 2) == 4, "dc hand");

        MinHull hull = new MinHull();
        for (long[] line : new long[][]{{1, 0}, {0, 0}, {-1, 0}}) hull.addLine(line[0], line[1]);   // slopes non-increasing
        for (int x = -6; x <= 6; x++) {
            long expected = Math.min(x, Math.min(-x, 0));
            check(hull.query(x) == expected, "hull hand " + x);
        }

        for (long[] xs : lcgArrays(8, 40)) {
            for (int k = 1; k <= 6; k++) {
                long brute = partitionBruteforce(xs, k);
                check(partitionOfflineCht(xs, k) == brute, "cht random k=" + k);
                check(partitionDivideConquer(xs, k) == brute, "dc random k=" + k);
            }
        }

        boolean threw = false;
        try { new MinHull().query(3); }
        catch (IllegalStateException e) { threw = true; }
        check(threw, "empty hull rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
