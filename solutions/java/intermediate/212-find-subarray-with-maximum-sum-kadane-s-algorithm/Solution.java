/** Task 212 - Find subarray with maximum sum (Kadane's algorithm). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Returns {bestSum, start, endExclusive}. */
    static int[] maxSubarray(int[] nums) {
        if (nums.length == 0) throw new IllegalArgumentException("empty array has no subarray");
        int best = nums[0], cur = nums[0];
        int bestStart = 0, bestEnd = 0, start = 0;
        for (int i = 1; i < nums.length; i++) {
            if (cur + nums[i] < nums[i]) {
                start = i;
                cur = nums[i];
            } else {
                cur += nums[i];
            }
            if (cur > best) {
                best = cur;
                bestStart = start;
                bestEnd = i;
            }
        }
        return new int[]{best, bestStart, bestEnd + 1};
    }

    public static void main(String[] args) {
        int[] r = maxSubarray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        check(r[0] == 6 && r[1] == 3 && r[2] == 7, "classic");
        check(maxSubarray(new int[]{-3, -1, -2})[0] == -1, "all negative");
        check(maxSubarray(new int[]{1})[0] == 1, "single");
        check(maxSubarray(new int[]{5, -1, 5})[0] == 9, "dip in middle");

        boolean threw = false;
        try { maxSubarray(new int[]{}); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "empty rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
