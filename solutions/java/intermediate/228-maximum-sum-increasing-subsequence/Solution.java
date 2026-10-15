/** Task 228 - Maximum sum increasing subsequence. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Largest sum of a strictly increasing subsequence. */
    static int maxSumIncreasingSubsequence(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = nums.clone();
        int best = dp[0];
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[j] + nums[i] > dp[i]) dp[i] = dp[j] + nums[i];
            }
            best = Math.max(best, dp[i]);
        }
        return best;
    }

    public static void main(String[] args) {
        check(maxSumIncreasingSubsequence(new int[]{1, 101, 2, 3, 100, 4}) == 106, "1+2+3+100");
        check(maxSumIncreasingSubsequence(new int[]{10, 1}) == 10, "falling pair");
        check(maxSumIncreasingSubsequence(new int[]{5, 4, 3, 2, 1}) == 5, "all falling");
        check(maxSumIncreasingSubsequence(new int[]{1, 2, 3, 4}) == 10, "whole array");
        check(maxSumIncreasingSubsequence(new int[]{7}) == 7, "single");
        check(maxSumIncreasingSubsequence(new int[]{}) == 0, "empty");
        check(maxSumIncreasingSubsequence(new int[]{3, 3}) == 3, "not strict");
        check(maxSumIncreasingSubsequence(new int[]{-2, -1, -3}) == -1, "all negative");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
