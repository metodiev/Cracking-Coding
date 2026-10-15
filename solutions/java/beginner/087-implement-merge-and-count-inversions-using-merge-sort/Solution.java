/** Task 087 - Implement merge and count inversions using merge sort. (Tier: Beginner, CTCI Ch 10)
 *
 * Augmented merge sort: when an element from the right half wins the merge, every
 * remaining element of the left half is larger than it, so the running count
 * grows by the number of those remaining elements. O(n log n) time.
 */
import java.util.*;

public class Solution {
    static long countInversions(int[] nums) {
        int[] a = Arrays.copyOf(nums, nums.length);
        int[] buffer = new int[a.length];
        return mergeSortCount(a, buffer, 0, a.length);
    }

    static long mergeSortCount(int[] a, int[] buffer, int lo, int hi) {
        if (hi - lo <= 1) return 0;
        int mid = lo + (hi - lo) / 2;
        long count = mergeSortCount(a, buffer, lo, mid) + mergeSortCount(a, buffer, mid, hi);
        System.arraycopy(a, lo, buffer, lo, hi - lo);
        int i = lo, j = mid, k = lo;
        while (i < mid && j < hi) {
            if (buffer[i] <= buffer[j]) {
                a[k++] = buffer[i++];
            } else {
                a[k++] = buffer[j++];
                count += mid - i;
            }
        }
        while (i < mid) a[k++] = buffer[i++];
        while (j < hi) a[k++] = buffer[j++];
        return count;
    }

    static long countInversionsNaive(int[] nums) {
        long count = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] > nums[j]) count++;
            }
        }
        return count;
    }

    static int[] pseudoRandomList(int size) {
        int[] values = new int[size];
        long state = 7;
        for (int i = 0; i < size; i++) {
            state = state * 48271 % 2147483647L;
            values[i] = (int) (state % 1000);
        }
        return values;
    }

    public static void main(String[] args) {
        check(countInversions(new int[]{}) == 0, "empty");
        check(countInversions(new int[]{1}) == 0, "single");
        check(countInversions(new int[]{1, 2, 3, 4}) == 0, "sorted");
        check(countInversions(new int[]{3, 2, 1}) == 3, "reverse 3");
        check(countInversions(new int[]{5, 4, 3, 2, 1}) == 10, "reverse 5");
        check(countInversions(new int[]{2, 1, 2, 1}) == 3, "duplicates");
        check(countInversions(new int[]{2, 4, 1, 3, 5}) == 3, "mixed");
        check(countInversions(new int[]{1, 1, 1}) == 0, "all equal");

        int[] reversed = new int[300];
        for (int i = 0; i < 300; i++) reversed[i] = 300 - i;
        check(countInversions(reversed) == 44850L, "reverse 300");

        for (int size : new int[]{2, 7, 50, 200}) {
            int[] nums = pseudoRandomList(size);
            check(countInversions(nums) == countInversionsNaive(nums), "agrees with naive, n=" + size);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
