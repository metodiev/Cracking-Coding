/** Task 208 - Implement sliding window minimum using deque. (Tier: Intermediate, CTCI Ch 3) */
import java.util.*;

public class Solution {
    static List<Integer> slidingWindowMinimum(int[] nums, int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        List<Integer> result = new ArrayList<>();
        Deque<Integer> dq = new ArrayDeque<>();      // indices, values increasing
        for (int i = 0; i < nums.length; i++) {
            while (!dq.isEmpty() && nums[dq.peekLast()] >= nums[i]) dq.pollLast();
            dq.addLast(i);
            if (dq.peekFirst() <= i - k) dq.pollFirst();
            if (i >= k - 1) result.add(nums[dq.peekFirst()]);
        }
        return result;
    }

    public static void main(String[] args) {
        check(slidingWindowMinimum(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3).toString()
                .equals("[-1, -3, -3, -3, 3, 3]"), "classic");
        check(slidingWindowMinimum(new int[]{1}, 1).toString().equals("[1]"), "single");
        check(slidingWindowMinimum(new int[]{1, 2}, 5).isEmpty(), "k > n");
        check(slidingWindowMinimum(new int[]{5, 5, 5}, 2).toString().equals("[5, 5]"), "duplicates");
        check(slidingWindowMinimum(new int[]{9, 8, 7}, 1).toString().equals("[9, 8, 7]"), "k=1");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
