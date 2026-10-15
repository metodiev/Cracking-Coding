/** Task 072 - Count sort for small integer range. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static int[] countingSort(int[] a) {
        if (a.length == 0) return new int[0];
        int lo = a[0], hi = a[0];
        for (int x : a) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
        int[] counts = new int[hi - lo + 1];
        for (int x : a) counts[x - lo]++;
        int[] out = new int[a.length];
        int k = 0;
        for (int offset = 0; offset < counts.length; offset++)
            for (int c = 0; c < counts[offset]; c++) out[k++] = lo + offset;
        return out;
    }

    public static void main(String[] args) {
        check(Arrays.equals(countingSort(new int[]{4, 2, 2, 8, 3, 3, 1}),
                new int[]{1, 2, 2, 3, 3, 4, 8}), "basic");
        check(countingSort(new int[]{}).length == 0, "empty");
        check(Arrays.equals(countingSort(new int[]{-3, -1, -3}), new int[]{-3, -3, -1}), "negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
