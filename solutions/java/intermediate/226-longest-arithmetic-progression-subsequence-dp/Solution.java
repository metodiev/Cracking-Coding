/** Task 226 - Longest arithmetic progression subsequence (DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /** Length of the longest arithmetic subsequence (not necessarily contiguous). */
    static int longestArithmeticSubsequence(int[] nums) {
        if (nums.length == 0) return 0;
        List<Map<Integer, Integer>> dp = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) dp.add(new HashMap<>());
        int best = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                int d = nums[i] - nums[j];
                int length = dp.get(j).getOrDefault(d, 1) + 1;
                if (length > dp.get(i).getOrDefault(d, 0)) dp.get(i).put(d, length);
                best = Math.max(best, length);
            }
        }
        return best;
    }

    public static void main(String[] args) {
        check(longestArithmeticSubsequence(new int[]{3, 6, 9, 12}) == 4, "whole array");
        check(longestArithmeticSubsequence(new int[]{9, 4, 7, 2, 10}) == 3, "4, 7, 10");
        check(longestArithmeticSubsequence(new int[]{1, 2}) == 2, "pair");
        check(longestArithmeticSubsequence(new int[]{4}) == 1, "single");
        check(longestArithmeticSubsequence(new int[]{}) == 0, "empty");
        check(longestArithmeticSubsequence(new int[]{1, 1, 1}) == 3, "difference 0");
        check(longestArithmeticSubsequence(new int[]{2, 4, 6, 8, 10}) == 5, "five terms");
        check(longestArithmeticSubsequence(new int[]{1, 5, 9, 13, 2, 6, 10, 14}) == 4, "two interleaved APs");
        check(longestArithmeticSubsequence(new int[]{5, 5, 3, 1}) == 3, "5, 3, 1 (step -2)");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
