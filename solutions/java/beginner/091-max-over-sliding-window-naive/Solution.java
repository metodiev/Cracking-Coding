/** Task 091 - Max over sliding window naive. (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static int[] maxSlidingWindowNaive(int[] nums, int k) {
        int n = nums.length;
        if (k <= 0 || k > n) return new int[0];
        int[] result = new int[n - k + 1];
        for (int start = 0; start + k <= n; start++) {
            int best = nums[start];
            for (int i = start + 1; i < start + k; i++) {
                if (nums[i] > best) best = nums[i];
            }
            result[start] = best;
        }
        return result;
    }

    public static void main(String[] args) {
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        check(Arrays.equals(maxSlidingWindowNaive(nums, 3), new int[]{3, 3, 5, 5, 6, 7}), "k=3");
        check(Arrays.equals(maxSlidingWindowNaive(nums, 1), nums), "k=1");
        check(Arrays.equals(maxSlidingWindowNaive(nums, 8), new int[]{7}), "k=n");
        check(Arrays.equals(maxSlidingWindowNaive(new int[]{9, 9, 9}, 2), new int[]{9, 9}), "duplicates");
        check(Arrays.equals(maxSlidingWindowNaive(new int[]{-5, -2, -9}, 2), new int[]{-2, -2}), "negatives");
        check(Arrays.equals(maxSlidingWindowNaive(new int[]{5}, 1), new int[]{5}), "single");
        check(Arrays.equals(maxSlidingWindowNaive(new int[]{1, 2, 3}, 0), new int[0]), "k=0");
        check(Arrays.equals(maxSlidingWindowNaive(new int[]{1, 2, 3}, 4), new int[0]), "k>n");
        check(Arrays.equals(maxSlidingWindowNaive(new int[0], 1), new int[0]), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
