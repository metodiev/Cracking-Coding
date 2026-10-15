/** Task 229 - Longest alternating subsequence. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Longest up-down (wiggle) subsequence length. */
    static int longestAlternatingSubsequence(int[] nums) {
        if (nums.length == 0) return 0;
        int up = 1, down = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) up = down + 1;
            else if (nums[i] < nums[i - 1]) down = up + 1;
        }
        return Math.max(up, down);
    }

    public static void main(String[] args) {
        check(longestAlternatingSubsequence(new int[]{1, 7, 4, 9, 2, 5}) == 6, "full zigzag");
        check(longestAlternatingSubsequence(new int[]{1, 17, 5, 10, 13, 15, 10, 5, 16, 8}) == 7, "long sample");
        check(longestAlternatingSubsequence(new int[]{8}) == 1, "single");
        check(longestAlternatingSubsequence(new int[]{}) == 0, "empty");
        check(longestAlternatingSubsequence(new int[]{1, 2, 3, 4}) == 2, "monotone");
        check(longestAlternatingSubsequence(new int[]{2, 1}) == 2, "one down step");
        check(longestAlternatingSubsequence(new int[]{3, 3}) == 1, "equal is not a step");
        check(longestAlternatingSubsequence(new int[]{3, 3, 3, 2}) == 2, "equal then down");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
