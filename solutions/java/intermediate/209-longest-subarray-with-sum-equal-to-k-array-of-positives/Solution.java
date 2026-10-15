/** Task 209 - Longest subarray with sum equal to K (array of positives). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static int longestSubarraySumK(int[] nums, int target) {
        for (int x : nums)
            if (x <= 0) throw new IllegalArgumentException("requires positive values");
        int lo = 0, total = 0, best = 0;
        for (int hi = 0; hi < nums.length; hi++) {
            total += nums[hi];
            while (total > target && lo <= hi) {
                total -= nums[lo];
                lo++;
            }
            if (total == target) best = Math.max(best, hi - lo + 1);
        }
        return best;
    }

    public static void main(String[] args) {
        check(longestSubarraySumK(new int[]{1, 2, 3, 1, 1, 1, 1}, 6) == 4, "classic");
        check(longestSubarraySumK(new int[]{1, 1, 1, 1}, 2) == 2, "all ones");
        check(longestSubarraySumK(new int[]{5}, 5) == 1, "single");
        check(longestSubarraySumK(new int[]{1, 2, 3}, 100) == 0, "too small");
        check(longestSubarraySumK(new int[]{2, 2, 2}, 6) == 3, "whole array");

        boolean threw = false;
        try { longestSubarraySumK(new int[]{1, -1, 1}, 0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "rejects non-positive input");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
