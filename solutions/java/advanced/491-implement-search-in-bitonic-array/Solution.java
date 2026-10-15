/** Task 491 - Implement search in bitonic array. (Tier: Advanced, CTCI Ch 10)
 *
 * A bitonic (mountain) array strictly rises to a single peak and then strictly falls; the peak
 * may sit at either end, so purely increasing and purely decreasing arrays are the degenerate
 * cases. findPeak is the standard O(log n) binary search that compares values[mid] with its
 * neighbour; search then binary searches the ascending side [0, peak] and the descending side
 * (peak, n-1] separately, for O(log n) total. An absent target returns the documented sentinel
 * -1 (indices are non-negative, so -1 is unambiguous). Verified against a linear-scan oracle
 * on every element and on absent probes over LCG-built bitonic arrays of many sizes, including
 * single-element arrays and arrays whose peak is at a boundary.
 */
import java.util.*;

public class Solution {
    static final int SENTINEL = -1;

    /** Index of the unique maximum (the peak) of a strictly bitonic array. */
    static int findPeak(int[] values) {
        int lo = 0;
        int hi = values.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (values[mid] < values[mid + 1]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** Index of target in a strictly bitonic array, or SENTINEL (-1) when absent. */
    static int search(int[] values, int target) {
        int n = values.length;
        if (n == 0) return SENTINEL;
        int peak = findPeak(values);

        int lo = 0;
        int hi = peak;                              // ascending side
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (values[mid] == target) return mid;
            if (values[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        lo = peak + 1;
        hi = n - 1;                                 // descending side
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (values[mid] == target) return mid;
            if (values[mid] > target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        return SENTINEL;
    }

    /** Simple oracle: first index holding target, or SENTINEL. */
    static int linearSearch(int[] values, int target) {
        for (int i = 0; i < values.length; i++) {
            if (values[i] == target) return i;
        }
        return SENTINEL;
    }

    /** Strictly bitonic array of n distinct LCG values with the peak at `peak`. */
    static int[] lcgBitonic(int n, int seed, int peak) {
        int[] distinct = new int[n];
        boolean[] seen = new boolean[1000];
        int count = 0;
        while (count < n) {
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 1000;
            if (!seen[value]) {
                seen[value] = true;
                distinct[count++] = value;
            }
        }
        Arrays.sort(distinct);
        int tailLen = n - 1 - peak;                 // values below the peak, descending
        int[] out = new int[n];
        for (int i = 0; i < n - tailLen; i++) out[i] = distinct[tailLen + i];
        for (int i = 0; i < tailLen; i++) out[n - tailLen + i] = distinct[tailLen - 1 - i];
        return out;
    }

    public static void main(String[] args) {
        int[] sample = {1, 3, 5, 9, 7, 4, 2};
        check(findPeak(sample) == 3, "hand peak");
        for (int i = 0; i < sample.length; i++) check(search(sample, sample[i]) == i, "hand " + i);
        for (int absent : new int[]{0, 6, 8, 10, -5}) {
            check(search(sample, absent) == SENTINEL, "hand absent " + absent);
        }

        check(findPeak(new int[]{5}) == 0, "single peak");
        check(search(new int[]{5}, 5) == 0, "single hit");
        check(search(new int[]{5}, 4) == SENTINEL, "single miss");
        check(findPeak(new int[]{1, 2, 3, 4}) == 3, "increasing peak");
        check(search(new int[]{1, 2, 3, 4}, 4) == 3, "increasing hit");
        check(search(new int[]{1, 2, 3, 4}, 0) == SENTINEL, "increasing miss");
        check(findPeak(new int[]{4, 3, 2, 1}) == 0, "decreasing peak");
        check(search(new int[]{4, 3, 2, 1}, 1) == 3, "decreasing hit");
        check(search(new int[]{4, 3, 2, 1}, 5) == SENTINEL, "decreasing miss");

        int checkedArrays = 0;
        for (int n = 1; n <= 17; n++) {
            for (int peak = 0; peak < n; peak++) {
                int[] values = lcgBitonic(n, 1000 + 13 * n + peak, peak);
                checkedArrays++;
                check(findPeak(values) == peak, "peak " + n + "/" + peak);
                for (int i = 0; i < n; i++) {
                    check(search(values, values[i]) == i, "element " + n + "/" + peak + "/" + i);
                    check(search(values, values[i]) == linearSearch(values, values[i]), "oracle");
                }
                for (int absent : new int[]{values[0] - 1, values[peak] + 1, 1000, -1, 65537}) {
                    check(search(values, absent) == SENTINEL, "absent " + n + "/" + peak);
                    check(linearSearch(values, absent) == SENTINEL, "oracle absent");
                }
            }
        }
        check(checkedArrays == 17 * 18 / 2, "array count " + checkedArrays);

        int[][] larger = {{256, 4242, 100}, {257, 777, 0}, {255, 31337, 254}, {300, 5, 150}};
        for (int[] spec : larger) {
            int n = spec[0];
            int[] values = lcgBitonic(n, spec[1], spec[2]);
            check(findPeak(values) == spec[2], "large peak");
            int[] probes = {0, 1, n / 2, spec[2] - 1, spec[2], spec[2] + 1, n - 2, n - 1};
            for (int i : probes) {
                if (i >= 0 && i < n) check(search(values, values[i]) == i, "large element " + i);
            }
            check(search(values, values[spec[2]] + 7) == SENTINEL, "large absent");
        }

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
