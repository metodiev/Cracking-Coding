/** Task 210 - Longest subarray with sum equal to K (with negatives) using hashmap. (Tier: Intermediate, CTCI Ch 16) */
import java.util.*;

public class Solution {
    static int longestSubarraySumK(int[] nums, int target) {
        Map<Integer, Integer> firstIndex = new HashMap<>();
        firstIndex.put(0, -1);                 // prefix 0 occurs before the array
        int prefix = 0, best = 0;
        for (int j = 0; j < nums.length; j++) {
            prefix += nums[j];
            Integer idx = firstIndex.get(prefix - target);
            if (idx != null) best = Math.max(best, j - idx);
            firstIndex.putIfAbsent(prefix, j); // keep earliest occurrence
        }
        return best;
    }

    public static void main(String[] args) {
        check(longestSubarraySumK(new int[]{1, 2, 3, 1, 1, 1, 1}, 6) == 4, "positives");
        check(longestSubarraySumK(new int[]{-1, 1, 1}, 1) == 3, "whole array sums to 1");
        check(longestSubarraySumK(new int[]{1, -1, 5, -2, 3}, 3) == 4, "mixed");
        check(longestSubarraySumK(new int[]{-2, -1, 2, 1}, 1) == 2, "negatives");
        check(longestSubarraySumK(new int[]{1, 2, 3}, 100) == 0, "none");
        check(longestSubarraySumK(new int[]{0, 0, 0}, 0) == 3, "zeros");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
