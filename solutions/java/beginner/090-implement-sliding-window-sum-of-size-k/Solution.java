/** Task 090 - Implement sliding window sum of size k. (Tier: Beginner, CTCI Ch 1)
 *
 * O(n) running-sum window: subtract the element leaving the window and add the
 * one entering it. Returns the sums of every contiguous window of size k in
 * left-to-right order; k == 0 or k > n returns an empty array (no windows);
 * k < 0 throws IllegalArgumentException.
 */
import java.util.*;

public class Solution {
    static int[] windowSums(int[] nums, int k) {
        if (k < 0) throw new IllegalArgumentException("k must be >= 0");
        int n = nums.length;
        if (k == 0 || k > n) return new int[0];
        int[] result = new int[n - k + 1];
        int window = 0;
        for (int i = 0; i < k; i++) window += nums[i];
        result[0] = window;
        for (int i = k; i < n; i++) {
            window += nums[i] - nums[i - k];
            result[i - k + 1] = window;
        }
        return result;
    }

    public static void main(String[] args) {
        check(Arrays.equals(windowSums(new int[]{1, 2, 3, 4, 5}, 3), new int[]{6, 9, 12}), "k=3");
        check(Arrays.equals(windowSums(new int[]{1, 2, 3, 4, 5}, 1), new int[]{1, 2, 3, 4, 5}), "k=1");
        check(Arrays.equals(windowSums(new int[]{1, 2, 3, 4, 5}, 5), new int[]{15}), "k=n");
        check(windowSums(new int[]{1, 2, 3, 4, 5}, 6).length == 0, "k > n");
        check(windowSums(new int[]{1, 2, 3, 4, 5}, 0).length == 0, "k == 0");
        check(windowSums(new int[]{}, 1).length == 0, "empty, k=1");
        check(windowSums(new int[]{}, 0).length == 0, "empty, k=0");
        check(Arrays.equals(windowSums(new int[]{4}, 1), new int[]{4}), "single");
        check(Arrays.equals(windowSums(new int[]{-1, -2, -3, -4}, 2), new int[]{-3, -5, -7}), "negatives");
        check(Arrays.equals(windowSums(new int[]{2, 2, 2, 2}, 2), new int[]{4, 4, 4}), "duplicates");
        checkThrows(() -> windowSums(new int[]{1, 2}, -1), "k < 0");

        int n = 20000, k = 100;
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) nums[i] = i;
        int[] sums = windowSums(nums, k);
        check(sums.length == n - k + 1, "large: window count");
        check(sums[0] == k * (k - 1) / 2, "large: first window");
        check(sums[sums.length - 1] == (n - k) * k + k * (k - 1) / 2, "large: last window");
        System.out.println("PASS");
    }

    static void checkThrows(Runnable r, String label) {
        try {
            r.run();
        } catch (IllegalArgumentException e) {
            return;
        }
        throw new AssertionError("FAILED: expected IllegalArgumentException: " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
