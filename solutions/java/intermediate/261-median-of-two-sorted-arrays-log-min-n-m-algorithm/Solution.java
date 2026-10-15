/** Task 261 - Median of two sorted arrays (log min(n,m) algorithm). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    private static final double NEG_INF = Double.NEGATIVE_INFINITY;
    private static final double POS_INF = Double.POSITIVE_INFINITY;

    /**
     * Partition-based binary search on the SHORTER array: i = number of a-elements on
     * the left side, j = (m+n+1)/2 - i follows. Valid when a[i-1] <= b[j] and
     * b[j-1] <= a[i]; +-inf sentinels cover empty arrays and fully-left/right cases.
     * O(log min(m, n)); two empty arrays raise IllegalArgumentException.
     */
    static double medianOfTwoSorted(int[] a, int[] b) {
        if (a.length == 0 && b.length == 0) throw new IllegalArgumentException("at least one array must be non-empty");
        if (a.length > b.length) { int[] t = a; a = b; b = t; }   // binary search the shorter array
        int m = a.length, n = b.length;
        int half = (m + n + 1) / 2;
        int lo = 0, hi = m;
        while (lo <= hi) {
            int i = (lo + hi) / 2;
            int j = half - i;
            double aLeft = i > 0 ? a[i - 1] : NEG_INF;
            double aRight = i < m ? a[i] : POS_INF;
            double bLeft = j > 0 ? b[j - 1] : NEG_INF;
            double bRight = j < n ? b[j] : POS_INF;
            if (aLeft <= bRight && bLeft <= aRight) {             // valid partition found
                if ((m + n) % 2 == 1) return Math.max(aLeft, bLeft);
                return (Math.max(aLeft, bLeft) + Math.min(aRight, bRight)) / 2.0;
            }
            if (aLeft > bRight) hi = i - 1;                       // too many a on the left
            else lo = i + 1;                                      // too few a on the left
        }
        throw new IllegalArgumentException("inputs are not sorted");
    }

    static int[] mergeSorted(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        int i = 0, j = 0, k = 0;
        while (i < a.length && j < b.length) out[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.length) out[k++] = a[i++];
        while (j < b.length) out[k++] = b[j++];
        return out;
    }

    /** O(m+n) merged-middle reference used only for cross-checks. */
    static double medianOracle(int[] a, int[] b) {
        int[] merged = mergeSorted(a, b);
        int n = merged.length;
        return n % 2 == 1 ? merged[n / 2] : (merged[n / 2 - 1] + merged[n / 2]) / 2.0;
    }

    static int[] sortedRandom(Random rng, int len, int lo, int hi) {
        int[] out = new int[len];
        for (int i = 0; i < len; i++) out[i] = lo + rng.nextInt(hi - lo + 1);
        Arrays.sort(out);
        return out;
    }

    public static void main(String[] args) {
        check(medianOfTwoSorted(new int[]{1, 3}, new int[]{2}) == 2.0, "2+1");
        check(medianOfTwoSorted(new int[]{1, 2}, new int[]{3, 4}) == 2.5, "even total");
        check(medianOfTwoSorted(new int[]{}, new int[]{1}) == 1.0, "empty first");
        check(medianOfTwoSorted(new int[]{1, 2, 3, 4, 5}, new int[]{}) == 3.0, "empty second");
        check(medianOfTwoSorted(new int[]{0, 0}, new int[]{0, 0}) == 0.0, "zeros");
        check(medianOfTwoSorted(new int[]{2}, new int[]{1, 3}) == 2.0, "wraps both sides");
        check(medianOfTwoSorted(new int[]{1, 3, 8}, new int[]{7, 9, 10, 11}) == 8.0, "classic mixed");
        check(medianOfTwoSorted(new int[]{1, 2}, new int[]{3, 4, 5, 6}) == 3.5, "fully left");
        check(medianOfTwoSorted(new int[]{5, 6, 7, 8}, new int[]{1, 2, 3, 4}) == 4.5, "fully right");
        check(medianOfTwoSorted(new int[]{1, 1}, new int[]{1, 1}) == 1.0, "duplicates");
        check(medianOfTwoSorted(new int[]{-5, -3, -1}, new int[]{-2}) == -2.5, "negatives");
        check(medianOfTwoSorted(new int[]{1, 2, 3}, new int[]{4, 5, 6}) == 3.5, "interleaved halves");

        boolean threw = false;
        try { medianOfTwoSorted(new int[]{}, new int[]{}); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "both empty raises");

        Random rng = new Random(261);
        for (int test = 0; test < 300; test++) {      // cross-check vs the merge oracle
            int[] a = sortedRandom(rng, rng.nextInt(9), -20, 20);
            int[] b = sortedRandom(rng, rng.nextInt(9), -20, 20);
            if (a.length == 0 && b.length == 0) continue;
            check(Math.abs(medianOfTwoSorted(a, b) - medianOracle(a, b)) < 1e-9, "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
