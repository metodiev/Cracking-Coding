/** Task 092 - Max over sliding window using deque. (Tier: Beginner, CTCI Ch 3) */
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

    static int[] maxSlidingWindowDeque(int[] nums, int k) {
        int n = nums.length;
        if (k <= 0 || k > n) return new int[0];
        Deque<Integer> dq = new ArrayDeque<>();  // indices, values decreasing
        int[] result = new int[n - k + 1];
        int out = 0;
        for (int i = 0; i < n; i++) {
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) dq.pollLast();
            dq.addLast(i);
            if (dq.peekFirst() <= i - k) dq.pollFirst();
            if (i >= k - 1) result[out++] = nums[dq.peekFirst()];
        }
        return result;
    }

    public static void main(String[] args) {
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        check(Arrays.equals(maxSlidingWindowDeque(nums, 3), new int[]{3, 3, 5, 5, 6, 7}), "k=3");
        check(Arrays.equals(maxSlidingWindowDeque(nums, 1), nums), "k=1");
        check(Arrays.equals(maxSlidingWindowDeque(nums, 8), new int[]{7}), "k=n");
        check(Arrays.equals(maxSlidingWindowDeque(new int[]{9, 9, 9}, 2), new int[]{9, 9}), "duplicates");
        check(Arrays.equals(maxSlidingWindowDeque(new int[]{-5, -2, -9}, 2), new int[]{-2, -2}), "negatives");
        check(Arrays.equals(maxSlidingWindowDeque(new int[0], 1), new int[0]), "empty");
        check(Arrays.equals(maxSlidingWindowDeque(new int[]{1, 2, 3}, 4), new int[0]), "k>n");
        check(Arrays.equals(maxSlidingWindowDeque(new int[]{1, 2, 3}, 0), new int[0]), "k=0");

        int[] arr = {9, -4, 7, 7, 2, -1, 8, 0, 5, 3};
        for (int k = 0; k <= arr.length + 1; k++) {
            check(Arrays.equals(maxSlidingWindowDeque(arr, k), maxSlidingWindowNaive(arr, k)), "cross-check k=" + k);
        }

        int[][] samples = {
            {4, 2, 12, 3, 8, 7, 5, 9, 1, 6},
            {-1, -1, -2, -3},
            {1, 1, 1, 1},
            {10, -10, 5, -5, 0},
            {7, 2},
        };
        int[] ks = {4, 2, 3, 5, 2};
        for (int s = 0; s < samples.length; s++) {
            check(Arrays.equals(maxSlidingWindowDeque(samples[s], ks[s]), maxSlidingWindowNaive(samples[s], ks[s])), "cross-check sample " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
